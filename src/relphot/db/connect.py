"""Resolve a PostgreSQL DSN and connect to the relphot results database."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

from relphot.exceptions import ConfigError

__all__ = ["connect", "default_env_path", "read_env_value", "resolve_dsn"]


def default_env_path() -> Path:
    """Path to the installer-written environment file, written by ``deploy/install.sh``."""
    return Path("~/.config/relphot/relphotdb.env").expanduser()


def read_env_value(path: Path, key: str) -> str | None:
    """Return ``key``'s value from a simple ``KEY=value`` env file, or ``None``.

    ``None`` covers both a missing file and a missing key. Lines are
    ``KEY=value``; a value may be wrapped in matching single or double
    quotes, which are stripped. Blank lines and ``#``-prefixed comments are
    ignored. This is deliberately not a shell parser -- it matches the
    plain ``KEY=value`` lines written by ``deploy/install.sh``.
    """
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


def resolve_dsn(
    dsn: str | None = None,
    *,
    env_var: str = "RELPHOT_DB_DSN",
    env_path: Path | None = None,
) -> str:
    """Resolve a PostgreSQL DSN: explicit ``dsn`` > ``env_var`` > the env file.

    ``env_path`` defaults to :func:`default_env_path`
    (``~/.config/relphot/relphotdb.env``), from which the ``env_var`` line
    is read via :func:`read_env_value`. Raises
    :class:`~relphot.exceptions.ConfigError` if none of the three sources
    yields a DSN.
    """
    if dsn:
        return dsn
    env_dsn = os.environ.get(env_var)
    if env_dsn:
        return env_dsn
    path = env_path if env_path is not None else default_env_path()
    file_dsn = read_env_value(path, env_var)
    if file_dsn:
        return file_dsn
    msg = f"no PostgreSQL DSN found: pass dsn=..., set {env_var}, or write {env_var}= in {path}"
    raise ConfigError(msg)


def connect(dsn: str | None = None, *, env_path: Path | None = None) -> psycopg.Connection:
    """Open a :class:`psycopg.Connection` using :func:`resolve_dsn`."""
    return psycopg.connect(resolve_dsn(dsn, env_path=env_path))
