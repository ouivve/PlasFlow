from __future__ import annotations

from pathlib import Path
from typing import Any
import sqlite3


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              progress REAL NOT NULL,
              mode TEXT NOT NULL,
              task TEXT,
              threshold REAL NOT NULL,
              read_type TEXT,
              min_length INTEGER,
              coverage_source TEXT,
              circularity_check INTEGER,
              polish TEXT,
              input_path TEXT NOT NULL,
              output_prefix TEXT NOT NULL,
              error TEXT,
              requested_mode TEXT NOT NULL,
              used_mode TEXT,
              fallback_reason TEXT,
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT
            )
            """
        )
        for stmt in (
            "ALTER TABLE jobs ADD COLUMN task TEXT",
            "ALTER TABLE jobs ADD COLUMN read_type TEXT",
            "ALTER TABLE jobs ADD COLUMN min_length INTEGER",
            "ALTER TABLE jobs ADD COLUMN coverage_source TEXT",
            "ALTER TABLE jobs ADD COLUMN circularity_check INTEGER",
            "ALTER TABLE jobs ADD COLUMN polish TEXT",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                # Column already exists.
                pass
        conn.commit()


def create_job(db_path: Path, payload: dict[str, Any]) -> None:
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(["?"] * len(payload))
    values = list(payload.values())
    with _connect(db_path) as conn:
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", values)
        conn.commit()


def update_job(db_path: Path, job_id: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values()) + [job_id]
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", values)
        conn.commit()


def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None
