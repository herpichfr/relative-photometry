"""Tests for relphot.db.schema against a live PostgreSQL test database.

Needs RELPHOT_TEST_DSN (env var, or the ``RELPHOT_TEST_DSN`` line in
``~/.config/relphot/relphotdb.env`` via :mod:`relphot.db.connect`) -- an
owner-privileged DSN to the ``relphot_test`` database created by
``deploy/install.sh``. Skipped with an explicit reason when no such DSN is
available. ``relphot_ro``'s own DSN is derived from the owner DSN plus
``RELPHOT_RO_PASSWORD`` (env var or the same env file); that one test is
skipped separately if the password is not available.
"""

from __future__ import annotations

import os
import re

import psycopg
import pytest

from relphot.db.connect import default_env_path, read_env_value, resolve_dsn
from relphot.db.schema import current_version, init_schema, migration_files
from relphot.exceptions import ConfigError


def _test_dsn() -> str:
    try:
        return resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")


def _ro_dsn(owner_dsn: str) -> str:
    ro_password = os.environ.get("RELPHOT_RO_PASSWORD") or read_env_value(
        default_env_path(), "RELPHOT_RO_PASSWORD"
    )
    if not ro_password:
        pytest.skip("RELPHOT_RO_PASSWORD not available to build a relphot_ro DSN")
    return re.sub(r"//[^:]+:[^@]+@", f"//relphot_ro:{ro_password}@", owner_dsn)


@pytest.fixture
def test_conn():
    dsn = _test_dsn()
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    yield conn
    conn.close()


def test_init_schema_creates_tables(test_conn) -> None:
    init_schema(test_conn)
    expected = {
        "night", "frame", "object", "star_night", "lightcurve", "detection",
        "catalog_match", "mn_run", "tie", "periodogram", "schema_version",
    }
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'relphot'"
        )
        found = {row[0] for row in cur.fetchall()}
    assert expected <= found


def test_init_schema_idempotent(test_conn) -> None:
    all_versions = [v for v, _ in migration_files()]
    applied_first = init_schema(test_conn)
    assert applied_first == all_versions
    applied_second = init_schema(test_conn)
    assert applied_second == []
    assert current_version(test_conn) == max(all_versions)


def test_q3c_extension_works(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute("SELECT q3c_dist(0, 0, 0, 1)")
        (dist,) = cur.fetchone()
    assert dist == pytest.approx(1.0, abs=1e-6)


def test_object_class_check_rejects_bad_value(test_conn) -> None:
    init_schema(test_conn)
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, class) VALUES ('t', 0, 0, 'BOGUS')"
        )
    test_conn.rollback()


def test_relphot_ro_cannot_insert(test_conn) -> None:
    init_schema(test_conn)
    owner_dsn = resolve_dsn(env_var="RELPHOT_TEST_DSN")
    ro_dsn = _ro_dsn(owner_dsn)
    with psycopg.connect(ro_dsn, autocommit=True) as ro_conn, ro_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.object")
        cur.fetchone()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("INSERT INTO relphot.object (name, ra, dec) VALUES ('t', 0, 0)")
