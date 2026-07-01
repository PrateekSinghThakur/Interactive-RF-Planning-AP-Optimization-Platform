"""Optional PostgreSQL JSONB project persistence for Building Model artifacts.

The Building Model JSON is stored as-is; there is no alternate wire/storage shape.
"""
from __future__ import annotations

import os
import uuid
from typing import Any


def _connect():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set; PostgreSQL persistence is disabled")
    import psycopg
    return psycopg.connect(database_url)


def ensure_schema() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                  id UUID PRIMARY KEY,
                  name TEXT NOT NULL,
                  building_model JSONB NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )


def save_project(model: dict[str, Any], name: str = "Untitled project", project_id: str | None = None) -> str:
    ensure_schema()
    pid = project_id or str(uuid.uuid4())
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (id, name, building_model)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                  name = EXCLUDED.name,
                  building_model = EXCLUDED.building_model,
                  updated_at = now()
                """,
                (pid, name, __import__("json").dumps(model)),
            )
    return pid


def load_project(project_id: str) -> dict[str, Any]:
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT building_model FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
    if not row:
        raise KeyError(project_id)
    return row[0]
