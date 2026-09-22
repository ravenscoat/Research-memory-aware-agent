from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from psycopg.types.json import Jsonb

from .models import MemoryItem, MemoryType

TOOL_ID_NAMESPACE = uuid.UUID("a85d4069-95b4-48d9-b964-58bb6ab955bb")
VECTOR_TABLES = {
    MemoryType.SEMANTIC: "research_semantic_memory",
    MemoryType.WORKFLOW: "research_workflow_memory",
    MemoryType.TOOLBOX: "research_toolbox_memory",
    MemoryType.ENTITY: "research_entity_memory",
    MemoryType.SUMMARY: "research_summary_memory",
}
CONVERSATION_TABLE = "research_conversational_memory"
TOOL_LOG_TABLE = "research_tool_log_memory"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


class PostgresMemoryStore:
    """PostgreSQL is durable truth; pgvector provides semantic retrieval."""

    def __init__(self, connection: Any, embedder: Any, *, dimensions: int = 768) -> None:
        self.connection = connection
        self.embedder = embedder
        self.dimensions = dimensions

    @classmethod
    def connect(cls, *, dsn: str, embedder: Any, dimensions: int = 768) -> PostgresMemoryStore:
        import psycopg

        return cls(psycopg.connect(dsn), embedder, dimensions=dimensions)

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        with self.connection.cursor() as cursor:
            yield cursor

    def initialize(self, *, drop_existing: bool = False) -> None:
        tables = [CONVERSATION_TABLE, TOOL_LOG_TABLE, *VECTOR_TABLES.values()]
        with self.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if drop_existing:
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {CONVERSATION_TABLE} (
                    id uuid PRIMARY KEY, thread_id text NOT NULL, role text NOT NULL,
                    content text NOT NULL, summarized boolean NOT NULL DEFAULT false,
                    created_at timestamptz NOT NULL DEFAULT now())"""
            )
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {TOOL_LOG_TABLE} (
                    id uuid PRIMARY KEY, thread_id text NOT NULL, tool_call_id text,
                    tool_name text NOT NULL, tool_args jsonb NOT NULL DEFAULT '{{}}',
                    result text NOT NULL, status text NOT NULL, error_message text,
                    metadata jsonb NOT NULL DEFAULT '{{}}', created_at timestamptz NOT NULL DEFAULT now())"""
            )
            for table in VECTOR_TABLES.values():
                cursor.execute(
                    f"""CREATE TABLE IF NOT EXISTS {table} (
                        id uuid PRIMARY KEY, thread_id text, text text NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{{}}', source_ids jsonb NOT NULL DEFAULT '[]',
                        embedding vector({self.dimensions}) NOT NULL,
                        active boolean NOT NULL DEFAULT true, created_at timestamptz NOT NULL DEFAULT now())"""
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx "
                    f"ON {table} USING hnsw (embedding vector_cosine_ops)"
                )
        self.connection.commit()

    def append_conversation(self, thread_id: str, role: str, content: str) -> str:
        memory_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {CONVERSATION_TABLE} (id,thread_id,role,content) VALUES (%s,%s,%s,%s)",
                (memory_id, str(thread_id), role, content),
            )
        self.connection.commit()
        return memory_id

    def recent_conversation(
        self, thread_id: str, *, limit: int = 12, include_summarized: bool = False
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        clause = "" if include_summarized else "AND summarized = false"
        with self.cursor() as cursor:
            cursor.execute(
                f"""SELECT id,role,content,summarized,created_at FROM (
                    SELECT id,role,content,summarized,created_at FROM {CONVERSATION_TABLE}
                    WHERE thread_id=%s {clause} ORDER BY created_at DESC LIMIT %s
                ) recent ORDER BY created_at""",
                (str(thread_id), limit),
            )
            rows = cursor.fetchall()
        return [{"id": str(r[0]), "role": r[1], "content": r[2], "summarized": r[3],
                 "created_at": r[4]} for r in rows]

    def mark_conversation_summarized(self, thread_id: str, ids: list[str]) -> None:
        if ids:
            with self.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {CONVERSATION_TABLE} SET summarized=true "
                    "WHERE thread_id=%s AND id=ANY(%s::uuid[])", (str(thread_id), ids)
                )
            self.connection.commit()

    def conversation_by_ids(self, thread_id: str, ids: list[str]) -> list[dict[str, str]]:
        if not ids:
            return []
        with self.cursor() as cursor:
            cursor.execute(
                f"SELECT id,role,content FROM {CONVERSATION_TABLE} "
                "WHERE thread_id=%s AND id=ANY(%s::uuid[]) ORDER BY created_at",
                (str(thread_id), ids),
            )
            return [{"id": str(r[0]), "role": r[1], "content": r[2]} for r in cursor.fetchall()]

    def _embedding(self, text: str) -> str:
        vector = self.embedder.embed(text)
        if len(vector) != self.dimensions:
            raise ValueError(f"Embedding dimension {len(vector)} does not match schema {self.dimensions}")
        return _vector_literal(vector)

    def append_vector(
        self, memory_type: MemoryType, text: str, *, thread_id: str | None = None,
        metadata: dict[str, Any] | None = None, source_ids: list[str] | None = None,
    ) -> str:
        if memory_type not in VECTOR_TABLES:
            raise ValueError(f"{memory_type} is not vector-backed")
        memory_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {VECTOR_TABLES[memory_type]} "
                "(id,thread_id,text,metadata,source_ids,embedding) VALUES (%s,%s,%s,%s,%s,%s::vector)",
                (memory_id, str(thread_id) if thread_id is not None else None, text,
                 Jsonb(metadata or {}), Jsonb(source_ids or []), self._embedding(text)),
            )
        self.connection.commit()
        return memory_id

    def upsert_tool(self, *, name: str, description: str, parameters: dict[str, Any]) -> str:
        memory_id = str(uuid.uuid5(TOOL_ID_NAMESPACE, name))
        text = f"{name}: {description}"
        metadata = {"name": name, "description": description, "parameters": parameters}
        table = VECTOR_TABLES[MemoryType.TOOLBOX]
        with self.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {table} (id,text,metadata,source_ids,embedding,active)
                VALUES (%s,%s,%s,'[]',%s::vector,true)
                ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata,
                embedding=EXCLUDED.embedding, active=true""",
                (memory_id, text, Jsonb(metadata), self._embedding(text)),
            )
        self.connection.commit()
        return memory_id

    def search_vector(
        self, memory_type: MemoryType, query: str, *, limit: int = 4,
        thread_id: str | None = None,
    ) -> list[MemoryItem]:
        if memory_type not in VECTOR_TABLES:
            raise ValueError(f"{memory_type} is not vector-backed")
        table = VECTOR_TABLES[memory_type]
        scope = "AND (thread_id=%s OR thread_id IS NULL)" if thread_id else ""
        params: list[Any] = []
        if thread_id:
            params.append(str(thread_id))
        params.extend([self._embedding(query), max(1, min(int(limit), 25))])
        with self.cursor() as cursor:
            cursor.execute(
                f"SELECT id,text,thread_id,metadata,source_ids FROM {table} "
                f"WHERE active=true {scope} ORDER BY embedding <=> %s::vector LIMIT %s", params
            )
            rows = cursor.fetchall()
        return [MemoryItem(id=str(r[0]), memory_type=memory_type, text=r[1], thread_id=r[2],
                           metadata=r[3] or {}, source_ids=r[4] or []) for r in rows]

    def get_vector(self, memory_type: MemoryType, memory_id: str) -> MemoryItem | None:
        with self.cursor() as cursor:
            cursor.execute(
                f"SELECT id,text,thread_id,metadata,source_ids FROM {VECTOR_TABLES[memory_type]} "
                "WHERE id=%s AND active=true", (memory_id,)
            )
            row = cursor.fetchone()
        return None if row is None else MemoryItem(
            id=str(row[0]), memory_type=memory_type, text=row[1], thread_id=row[2],
            metadata=row[3] or {}, source_ids=row[4] or []
        )

    def write_tool_log(
        self, *, thread_id: str, tool_call_id: str, tool_name: str,
        tool_args: dict[str, Any], result: str, status: str,
        error_message: str | None, metadata: dict[str, Any] | None = None,
    ) -> str:
        log_id = str(uuid.uuid4())
        with self.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {TOOL_LOG_TABLE}
                (id,thread_id,tool_call_id,tool_name,tool_args,result,status,error_message,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (log_id, str(thread_id), tool_call_id, tool_name, Jsonb(tool_args), result,
                 status, error_message, Jsonb(metadata or {})),
            )
        self.connection.commit()
        return log_id

    def close(self) -> None:
        self.connection.close()
