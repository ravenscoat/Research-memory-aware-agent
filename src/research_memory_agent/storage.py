from __future__ import annotations

import json
import re
import uuid
from array import array
from contextlib import contextmanager
from typing import Any, Iterator

from .models import MemoryItem, MemoryType

TOOL_ID_NAMESPACE = uuid.UUID("a85d4069-95b4-48d9-b964-58bb6ab955bb")


VECTOR_TABLES = {
    MemoryType.SEMANTIC: "SEMANTIC_MEMORY",
    MemoryType.WORKFLOW: "WORKFLOW_MEMORY",
    MemoryType.TOOLBOX: "TOOLBOX_MEMORY",
    MemoryType.ENTITY: "ENTITY_MEMORY",
    MemoryType.SUMMARY: "SUMMARY_MEMORY",
}

CONVERSATION_TABLE = "CONVERSATIONAL_MEMORY"
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY"


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,29}", value):
        raise ValueError(f"Unsafe Oracle identifier: {value!r}")
    return value


class OracleMemoryStore:
    """Oracle is the durable source of truth for all seven memory types."""

    def __init__(self, connection: Any, embedder: Any, *, dimensions: int = 768) -> None:
        self.connection = connection
        self.embedder = embedder
        self.dimensions = dimensions

    @classmethod
    def connect(
        cls,
        *,
        user: str,
        password: str,
        dsn: str,
        embedder: Any,
        dimensions: int = 768,
    ) -> "OracleMemoryStore":
        import oracledb

        connection = oracledb.connect(user=user, password=password, dsn=dsn)
        return cls(connection, embedder, dimensions=dimensions)

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        cursor = self.connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def initialize(self, *, drop_existing: bool = False) -> None:
        """Create all memory tables. Destructive reset requires an explicit flag."""
        tables = [CONVERSATION_TABLE, TOOL_LOG_TABLE, *VECTOR_TABLES.values()]
        if drop_existing:
            with self.cursor() as cursor:
                for table in tables:
                    table = _safe_identifier(table)
                    try:
                        cursor.execute(f"DROP TABLE {table} PURGE")
                    except Exception as exc:
                        if "ORA-00942" not in str(exc):
                            raise

        self._create_table(
            CONVERSATION_TABLE,
            f"""
            CREATE TABLE {CONVERSATION_TABLE} (
                id VARCHAR2(36) PRIMARY KEY,
                thread_id VARCHAR2(128) NOT NULL,
                role VARCHAR2(32) NOT NULL,
                content CLOB NOT NULL,
                summarized NUMBER(1) DEFAULT 0 NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
            )
            """,
        )
        self._create_table(
            TOOL_LOG_TABLE,
            f"""
            CREATE TABLE {TOOL_LOG_TABLE} (
                id VARCHAR2(36) PRIMARY KEY,
                thread_id VARCHAR2(128) NOT NULL,
                tool_call_id VARCHAR2(128),
                tool_name VARCHAR2(128) NOT NULL,
                tool_args CLOB,
                result CLOB,
                status VARCHAR2(32) NOT NULL,
                error_message CLOB,
                metadata CLOB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT tool_log_args_json CHECK (tool_args IS JSON),
                CONSTRAINT tool_log_metadata_json CHECK (metadata IS JSON)
            )
            """,
        )
        for table in VECTOR_TABLES.values():
            self._create_table(
                table,
                f"""
                CREATE TABLE {table} (
                    id VARCHAR2(36) PRIMARY KEY,
                    thread_id VARCHAR2(128),
                    text CLOB NOT NULL,
                    metadata CLOB,
                    source_ids CLOB,
                    embedding VECTOR({int(self.dimensions)}, FLOAT32),
                    active NUMBER(1) DEFAULT 1 NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                    CONSTRAINT {_safe_identifier(table[:18] + '_META_JSON')} CHECK (metadata IS JSON),
                    CONSTRAINT {_safe_identifier(table[:18] + '_SRC_JSON')} CHECK (source_ids IS JSON)
                )
                """,
            )
        self.connection.commit()

    def _create_table(self, table: str, ddl: str) -> None:
        table = _safe_identifier(table)
        with self.cursor() as cursor:
            try:
                cursor.execute(ddl)
            except Exception as exc:
                if "ORA-00955" not in str(exc):
                    raise

    def append_conversation(self, thread_id: str, role: str, content: str) -> str:
        memory_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {CONVERSATION_TABLE} (id, thread_id, role, content) "
                "VALUES (:id, :thread_id, :role, :content)",
                id=memory_id,
                thread_id=str(thread_id),
                role=role,
                content=content,
            )
        self.connection.commit()
        return memory_id

    def recent_conversation(
        self, thread_id: str, *, limit: int = 12, include_summarized: bool = False
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        summarized_clause = "" if include_summarized else "AND summarized = 0"
        sql = f"""
            SELECT id, role, content, summarized, created_at FROM (
                SELECT id, role, content, summarized, created_at
                FROM {CONVERSATION_TABLE}
                WHERE thread_id = :thread_id {summarized_clause}
                ORDER BY created_at DESC
            ) WHERE ROWNUM <= {limit}
        """
        with self.cursor() as cursor:
            cursor.execute(sql, thread_id=str(thread_id))
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "role": row[1],
                "content": _lob_text(row[2]),
                "summarized": bool(row[3]),
                "created_at": row[4],
            }
            for row in reversed(rows)
        ]

    def mark_conversation_summarized(self, thread_id: str, ids: list[str]) -> None:
        if not ids:
            return
        with self.cursor() as cursor:
            cursor.executemany(
                f"UPDATE {CONVERSATION_TABLE} SET summarized = 1 "
                "WHERE thread_id = :thread_id AND id = :id",
                [{"thread_id": str(thread_id), "id": item_id} for item_id in ids],
            )
        self.connection.commit()

    def conversation_by_ids(self, thread_id: str, ids: list[str]) -> list[dict[str, str]]:
        if not ids:
            return []
        rows: list[dict[str, str]] = []
        # Bind each identifier rather than interpolating values into SQL.
        binds: dict[str, Any] = {"thread_id": str(thread_id)}
        placeholders = []
        for index, item_id in enumerate(ids):
            key = f"id_{index}"
            binds[key] = item_id
            placeholders.append(f":{key}")
        with self.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role, content FROM {CONVERSATION_TABLE} "
                f"WHERE thread_id = :thread_id AND id IN ({', '.join(placeholders)}) "
                "ORDER BY created_at",
                binds,
            )
            for row in cursor.fetchall():
                rows.append({"id": row[0], "role": row[1], "content": _lob_text(row[2])})
        return rows

    def append_vector(
        self,
        memory_type: MemoryType,
        text: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_ids: list[str] | None = None,
    ) -> str:
        if memory_type not in VECTOR_TABLES:
            raise ValueError(f"{memory_type} is not vector-backed")
        vector = self.embedder.embed(text)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match schema {self.dimensions}"
            )
        memory_id = str(uuid.uuid4())
        table = _safe_identifier(VECTOR_TABLES[memory_type])
        with self.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(id, thread_id, text, metadata, source_ids, embedding) "
                "VALUES (:id, :thread_id, :text, :metadata, :source_ids, :embedding)",
                id=memory_id,
                thread_id=str(thread_id) if thread_id is not None else None,
                text=text,
                metadata=json.dumps(metadata or {}, ensure_ascii=False),
                source_ids=json.dumps(source_ids or []),
                embedding=array("f", vector),
            )
        self.connection.commit()
        return memory_id

    def upsert_tool(
        self, *, name: str, description: str, parameters: dict[str, Any]
    ) -> str:
        """Idempotently index one tool description using a stable UUID."""
        memory_id = str(uuid.uuid5(TOOL_ID_NAMESPACE, name))
        text = f"{name}: {description}"
        vector = self.embedder.embed(text)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match schema {self.dimensions}"
            )
        metadata = json.dumps(
            {"name": name, "description": description, "parameters": parameters},
            ensure_ascii=False,
        )
        table = VECTOR_TABLES[MemoryType.TOOLBOX]
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                MERGE INTO {table} target
                USING (SELECT :id AS id FROM dual) source
                ON (target.id = source.id)
                WHEN MATCHED THEN UPDATE SET
                    target.text = :text,
                    target.metadata = :metadata,
                    target.embedding = :embedding,
                    target.active = 1
                WHEN NOT MATCHED THEN INSERT
                    (id, thread_id, text, metadata, source_ids, embedding, active)
                VALUES
                    (:id, NULL, :text, :metadata, '[]', :embedding, 1)
                """,
                id=memory_id,
                text=text,
                metadata=metadata,
                embedding=array("f", vector),
            )
        self.connection.commit()
        return memory_id

    def search_vector(
        self,
        memory_type: MemoryType,
        query: str,
        *,
        limit: int = 4,
        thread_id: str | None = None,
    ) -> list[MemoryItem]:
        if memory_type not in VECTOR_TABLES:
            raise ValueError(f"{memory_type} is not vector-backed")
        limit = max(1, min(int(limit), 25))
        table = _safe_identifier(VECTOR_TABLES[memory_type])
        vector = array("f", self.embedder.embed(query))
        scope = "AND (thread_id = :thread_id OR thread_id IS NULL)" if thread_id else ""
        sql = f"""
            SELECT id, text, thread_id, metadata, source_ids
            FROM {table}
            WHERE active = 1 {scope}
            ORDER BY VECTOR_DISTANCE(embedding, :query_vector, COSINE)
            FETCH FIRST {limit} ROWS ONLY
        """
        binds: dict[str, Any] = {"query_vector": vector}
        if thread_id:
            binds["thread_id"] = str(thread_id)
        with self.cursor() as cursor:
            cursor.execute(sql, binds)
            rows = cursor.fetchall()
        return [
            MemoryItem(
                id=row[0],
                memory_type=memory_type,
                text=_lob_text(row[1]),
                thread_id=row[2],
                metadata=_json_value(row[3], {}),
                source_ids=_json_value(row[4], []),
            )
            for row in rows
        ]

    def get_vector(self, memory_type: MemoryType, memory_id: str) -> MemoryItem | None:
        table = _safe_identifier(VECTOR_TABLES[memory_type])
        with self.cursor() as cursor:
            cursor.execute(
                f"SELECT id, text, thread_id, metadata, source_ids FROM {table} "
                "WHERE id = :id AND active = 1",
                id=memory_id,
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return MemoryItem(
            id=row[0],
            memory_type=memory_type,
            text=_lob_text(row[1]),
            thread_id=row[2],
            metadata=_json_value(row[3], {}),
            source_ids=_json_value(row[4], []),
        )

    def write_tool_log(
        self,
        *,
        thread_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        status: str,
        error_message: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        log_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {TOOL_LOG_TABLE}
                    (id, thread_id, tool_call_id, tool_name, tool_args, result,
                     status, error_message, metadata)
                VALUES
                    (:id, :thread_id, :tool_call_id, :tool_name, :tool_args, :result,
                     :status, :error_message, :metadata)
                """,
                id=log_id,
                thread_id=str(thread_id),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=json.dumps(tool_args, ensure_ascii=False),
                result=result,
                status=status,
                error_message=error_message,
                metadata=json.dumps(metadata or {}),
            )
        self.connection.commit()
        return log_id

    def close(self) -> None:
        self.connection.close()


def _lob_text(value: Any) -> str:
    return value.read() if hasattr(value, "read") else str(value or "")


def _json_value(value: Any, fallback: Any) -> Any:
    raw = _lob_text(value)
    try:
        return json.loads(raw) if raw else fallback
    except json.JSONDecodeError:
        return fallback
