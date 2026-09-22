"""Enable pgvector and verify that cosine distance works."""

import os

import psycopg

dsn = os.environ["POSTGRES_DSN"]
with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    print("PGVECTOR_VERSION=" + cursor.fetchone()[0])
    cursor.execute("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector")
    print("COSINE_DISTANCE=" + str(cursor.fetchone()[0]))
