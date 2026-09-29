"""Database access for the relphot web API: DSN resolution, per-request
connections, and a small cache of which optional columns exist.

Two roles are used: ``relphot_ro`` (SELECT only, every read endpoint) and
``relphot_web`` (SELECT plus UPDATE on manual-edit columns of ``relphot.object``
and ``relphot.user_night_review``, and INSERT/UPDATE/DELETE on
``relphot.user_night_review``, used by PATCH and PUT endpoints). Each DSN is
resolved fresh on every call -- never cached at import time -- so a test process
can set the environment before each request rather than before the app module
is first imported.

``default_env_path``/``read_env_value`` are intentionally duplicated here
from :mod:`relphot.db.connect` rather than imported: importing anything
under ``relphot.db`` runs ``relphot/db/__init__.py``, which imports
``relphot.db.load_night`` (needs pandas) and ``relphot.db.analyze`` -- heavy
dependencies the ``web`` extra does not install and the web container does
not need for these two small, dependency-free helpers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg

from relphot.exceptions import ConfigError

__all__ = [
    "column_exists",
    "get_ro_conn",
    "get_rw_conn",
    "resolve_ro_dsn",
    "resolve_rw_dsn",
]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = "5433"
_DEFAULT_DBNAME = "relphot"


def default_env_path() -> Path:
    """Path to the installer-written environment file (see
    :func:`relphot.db.connect.default_env_path`, duplicated here -- see the
    module docstring for why)."""
    return Path("~/.config/relphot/relphotdb.env").expanduser()


def read_env_value(path: Path, key: str) -> str | None:
    """``key``'s value from a simple ``KEY=value`` env file, or ``None`` (see
    :func:`relphot.db.connect.read_env_value`, duplicated here -- see the
    module docstring for why)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return None


def _resolve_dsn(*, dsn_env_var: str, password_env_var: str, role: str) -> str:
    """``dsn_env_var`` if set; else built from ``password_env_var`` + host/port/db.

    The password is read from ``password_env_var`` (environment, then the
    installer's env file) via :func:`relphot.db.connect.read_env_value`.
    Raises :class:`~relphot.exceptions.ConfigError` if neither yields a DSN.
    """
    explicit = os.environ.get(dsn_env_var)
    if explicit:
        return explicit
    password = os.environ.get(password_env_var) or read_env_value(
        default_env_path(), password_env_var
    )
    if not password:
        msg = (
            f"no database DSN found: set {dsn_env_var}, or set {password_env_var} "
            f"(env var or ~/.config/relphot/relphotdb.env)"
        )
        raise ConfigError(msg)
    host = os.environ.get("RELPHOT_DB_HOST", _DEFAULT_HOST)
    port = os.environ.get("RELPHOT_DB_PORT", _DEFAULT_PORT)
    return f"postgresql://{role}:{password}@{host}:{port}/{_DEFAULT_DBNAME}"


def resolve_ro_dsn() -> str:
    """Resolve the ``relphot_ro`` DSN: ``RELPHOT_WEB_RO_DSN``, else built from
    ``RELPHOT_RO_PASSWORD``."""
    return _resolve_dsn(
        dsn_env_var="RELPHOT_WEB_RO_DSN", password_env_var="RELPHOT_RO_PASSWORD", role="relphot_ro"
    )


def resolve_rw_dsn() -> str:
    """Resolve the ``relphot_web`` DSN: ``RELPHOT_WEB_RW_DSN``, else built from
    ``RELPHOT_WEB_PASSWORD``."""
    return _resolve_dsn(
        dsn_env_var="RELPHOT_WEB_RW_DSN",
        password_env_var="RELPHOT_WEB_PASSWORD",
        role="relphot_web",
    )


@contextmanager
def get_ro_conn() -> Iterator[psycopg.Connection]:
    """A fresh ``relphot_ro`` connection for the duration of one request."""
    conn = psycopg.connect(resolve_ro_dsn())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_rw_conn() -> Iterator[psycopg.Connection]:
    """A fresh ``relphot_web`` connection for the duration of one request."""
    conn = psycopg.connect(resolve_rw_dsn())
    try:
        yield conn
    finally:
        conn.close()


_column_cache: dict[str, frozenset[str]] = {}


def column_exists(conn: psycopg.Connection, table: str, column: str) -> bool:
    """Whether ``relphot.<table>.<column>`` exists, cached per ``table`` for
    the life of the process.

    Lets a query select an optional column (added by a later migration,
    e.g. ``object.data_updated_at`` or ``periodogram.extra``) only when it is
    actually present, instead of failing on a database that has not been
    migrated yet.
    """
    if table not in _column_cache:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'relphot' AND table_name = %s",
                (table,),
            )
            _column_cache[table] = frozenset(row[0] for row in cur.fetchall())
    return column in _column_cache[table]
