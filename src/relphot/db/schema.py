"""Versioned schema migrations for the relphot results database.

Migration files live in :mod:`relphot.db.sql` as ``NNN_description.sql``,
applied in ascending numeric order. Each file is applied once, inside its
own transaction, and its number is recorded as a row in
``relphot.schema_version``; :func:`init_schema` is therefore idempotent --
a second call applies nothing.
"""

from __future__ import annotations

import re
from importlib import resources

import psycopg

__all__ = ["current_version", "init_schema", "migration_files"]

_MIGRATION_RE = re.compile(r"^(\d+)_.*\.sql$")


def migration_files() -> list[tuple[int, str]]:
    """(version, filename) pairs for every packaged migration, sorted by version."""
    sql_dir = resources.files("relphot.db").joinpath("sql")
    found = []
    for entry in sql_dir.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry.name))
    found.sort(key=lambda pair: pair[0])
    return found


def current_version(conn: psycopg.Connection) -> int:
    """Highest applied migration version, or 0 if ``relphot.schema_version`` does not exist yet."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(version), 0) FROM relphot.schema_version")
            (version,) = cur.fetchone()
            return int(version)
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return 0


def init_schema(conn: psycopg.Connection) -> list[int]:
    """Apply every packaged migration newer than the database's current version.

    Each migration runs in its own transaction (its SQL, then the
    ``schema_version`` insert), committed only on success. Returns the
    versions actually applied, in order (empty when already current).
    """
    applied: list[int] = []
    sql_dir = resources.files("relphot.db").joinpath("sql")
    for version, filename in migration_files():
        if version <= current_version(conn):
            continue
        sql_text = sql_dir.joinpath(filename).read_text()
        try:
            with conn.cursor() as cur:
                cur.execute(sql_text)
                cur.execute(
                    "INSERT INTO relphot.schema_version (version) VALUES (%s)", (version,)
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        applied.append(version)
    return applied
