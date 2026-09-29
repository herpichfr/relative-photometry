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
from importlib import resources

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
        "transit_shape", "transit_match", "period_estimate",
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


def _apply_up_to(conn: psycopg.Connection, max_version: int) -> None:
    """Apply the packaged migrations 1..max_version as ``init_schema`` would."""
    sql_dir = resources.files("relphot.db").joinpath("sql")
    for version, filename in migration_files():
        if version > max_version:
            break
        with conn.cursor() as cur:
            cur.execute(sql_dir.joinpath(filename).read_text())
            cur.execute("INSERT INTO relphot.schema_version (version) VALUES (%s)", (version,))
        conn.commit()


def test_migration_004_from_v3_maps_class_to_flags(test_conn) -> None:
    _apply_up_to(test_conn, 3)
    assert current_version(test_conn) == 3
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, class, class_source, period) VALUES "
            "('auto_exop', 1, 1, 'EXOP', 'auto', NULL), "
            "('auto_var', 2, 2, 'VAR', 'auto', NULL), "
            "('auto_unc', 3, 3, 'UNC', 'auto', NULL), "
            "('manual_exop', 4, 4, 'EXOP', 'manual', NULL), "
            "('manual_var', 5, 5, 'VAR', 'manual', NULL), "
            "('manual_unc', 6, 6, 'UNC', 'manual', NULL)"
        )
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n', '/tmp/n') RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
        cur.execute("SELECT obj_id FROM relphot.object WHERE name = 'auto_exop'")
        (obj_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, flags) VALUES "
            "(%s, %s, 'transit', 'SHARED_EPOCH|EDGE'), (%s, %s, 'transit', 'OK'), "
            "(%s, %s, 'transit', 'PARTIAL'), (%s, %s, 'transit', 'NEIGHBOUR_BLEND')",
            (obj_id, night_id) * 4,
        )
        cur.execute(
            "INSERT INTO relphot.catalog_match (obj_id, catalog, name, period) "
            "VALUES (%s, 'TOI', 'TOI-1', 2.0)",
            (obj_id,),
        )
    test_conn.commit()

    assert init_schema(test_conn) == [4, 5]
    assert current_version(test_conn) == 5

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT name, is_exop, is_var, exop_source, var_source, class, class_source "
            "FROM relphot.object ORDER BY obj_id"
        )
        rows = {r[0]: r[1:] for r in cur.fetchall()}
    assert rows["auto_exop"] == (True, False, "auto", "auto", "EXOP", "auto")
    assert rows["auto_var"] == (False, True, "auto", "auto", "VAR", "auto")
    assert rows["auto_unc"] == (False, False, "auto", "auto", "UNC", "auto")
    # a manual EXOP keeps its flag manual; the other flag goes back to automatic
    assert rows["manual_exop"] == (True, False, "manual", "auto", "EXOP", "manual")
    assert rows["manual_var"] == (False, True, "auto", "manual", "VAR", "manual")
    # a manual UNC pins both flags to false
    assert rows["manual_unc"] == (False, False, "manual", "manual", "UNC", "manual")

    with test_conn.cursor() as cur:
        cur.execute("SELECT status, notes FROM relphot.detection")
        assert cur.fetchone() == ("UNCONFIRMED", None)
        # R6: an existing EDGE / PARTIAL transit becomes a duration lower limit
        cur.execute("SELECT flags, duration_lower_limit FROM relphot.detection ORDER BY det_id")
        assert cur.fetchall() == [
            ("SHARED_EPOCH|EDGE", True), ("OK", False), ("PARTIAL", True),
            ("NEIGHBOUR_BLEND", False),
        ]
        cur.execute("SELECT duration_lower_limit FROM relphot.object LIMIT 1")
        assert cur.fetchone() == (None,)
        cur.execute("SELECT period_err FROM relphot.catalog_match")
        assert cur.fetchone() == (None,)
        cur.execute("UPDATE relphot.object SET class = 'EXOP+VAR' WHERE name = 'auto_exop'")


def test_migration_004_new_tables_and_constraints(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 0, 0) RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n', '/tmp/n') RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
        det_ids = []
        for _ in range(2):
            cur.execute(
                "INSERT INTO relphot.detection (obj_id, night_id, kind) "
                "VALUES (%s, %s, 'transit') RETURNING det_id",
                (obj_id, night_id),
            )
            det_ids.append(cur.fetchone()[0])
        det_a, det_b = det_ids
        cur.execute(
            "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, input, converged) "
            "VALUES (%s, %s, 2460000.5, 'tied', true)",
            (det_a, obj_id),
        )
        cur.execute(
            "SELECT t14_lower_limit, incomplete_reason FROM relphot.transit_shape "
            "WHERE det_id = %s",
            (det_a,),
        )
        assert cur.fetchone() == (False, None)
        cur.execute(
            "INSERT INTO relphot.detection_review_orphan (obj_id, kind, status) "
            "VALUES (%s, 'transit', 'CONFIRMED')",
            (obj_id,),
        )
        cur.execute(
            "INSERT INTO relphot.transit_match (det_a, det_b, obj_id, p_match) "
            "VALUES (%s, %s, %s, 0.5)",
            (det_a, det_b, obj_id),
        )
        cur.execute(
            "INSERT INTO relphot.period_estimate (obj_id, method, night_ids) "
            "VALUES (%s, 'LS', %s)",
            (obj_id, [1, 2]),
        )
    test_conn.commit()

    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.transit_match (det_a, det_b, obj_id) VALUES (%s, %s, %s)",
            (det_b, det_a, obj_id),
        )
    test_conn.rollback()
    with pytest.raises(psycopg.errors.UniqueViolation), test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.period_estimate (obj_id, method, night_ids) "
            "VALUES (%s, 'LS', %s)",
            (obj_id, [1, 2]),
        )
    test_conn.rollback()
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'MERGED' WHERE det_id = %s", (det_a,))
    test_conn.rollback()

    # deleting a detection cascades to its shape and to every match it is in
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.detection WHERE det_id = %s", (det_a,))
        cur.execute("SELECT count(*) FROM relphot.transit_shape")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM relphot.transit_match")
        assert cur.fetchone() == (0,)
    test_conn.commit()


def test_object_class_accepts_exop_var(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, class) VALUES ('t', 0, 0, 'EXOP+VAR')"
        )
    test_conn.commit()


def test_migration_005_verify_status_backfill_and_check(test_conn) -> None:
    _apply_up_to(test_conn, 4)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 0, 0) RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        rows = (([1], None, None), ([1, 2], 1.2, 1.0), ([1, 2, 3], 1.2, None))
        for ids, lit, harmonic in rows:
            cur.execute(
                "INSERT INTO relphot.period_estimate (obj_id, method, night_ids, lit_period, "
                "harmonic) VALUES (%s, 'LS', %s, %s, %s)",
                (obj_id, ids, lit, harmonic),
            )
    test_conn.commit()

    assert init_schema(test_conn) == [5]

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT verify_status, verify_note FROM relphot.period_estimate ORDER BY est_id"
        )
        assert cur.fetchall() == [("no_literature", None), ("verified", None), (None, None)]
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.period_estimate SET verify_status = 'bogus'")
    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.period_estimate SET verify_status = 'insufficient_data'")
