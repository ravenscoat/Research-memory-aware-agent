"""Print row counts for all seven durable memory stores."""

import os

import psycopg

tables = [
    "research_conversational_memory",
    "research_semantic_memory",
    "research_workflow_memory",
    "research_toolbox_memory",
    "research_entity_memory",
    "research_summary_memory",
    "research_tool_log_memory",
]

with psycopg.connect(os.environ["POSTGRES_DSN"]) as connection, connection.cursor() as cursor:
    for table in tables:
        cursor.execute(f"SELECT count(*) FROM {table}")
        print(f"{table}={cursor.fetchone()[0]}")
    cursor.execute("SELECT count(*) FROM research_tool_log_memory WHERE status = 'failed'")
    print(f"failed_tool_logs={cursor.fetchone()[0]}")
