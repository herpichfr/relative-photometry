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
        "night_tile", "tile_lc", "reference_member", "comparison_member",
        "repeat_link", "repeat_family", "repeat_family_member", "repeat_ephemeris",
        "repeat_decision",
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

    assert init_schema(test_conn) == [4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert current_version(test_conn) == 12

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

    assert init_schema(test_conn) == [5, 6, 7, 8, 9, 10, 11, 12]

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


def test_migration_007_night_zero_point_defaults_existing_nights_to_assumed(test_conn) -> None:
    _apply_up_to(test_conn, 6)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n', '/tmp/n')"
        )
    test_conn.commit()

    assert init_schema(test_conn) == [7, 8, 9, 10, 11, 12]

    with test_conn.cursor() as cur:
        cur.execute("SELECT zp, zp_source FROM relphot.night")
        assert cur.fetchall() == [(20.0, "assumed")]
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, zp, zp_source) "
            "VALUES ('T80S', '2025-01-02', 'm', '/tmp/m', 24.5, 'gaia')"
        )
    test_conn.commit()
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.night SET zp_source = 'bogus'")
    test_conn.rollback()


def test_migration_008_night_zero_point_source_accepts_measured(test_conn) -> None:
    _apply_up_to(test_conn, 7)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n', '/tmp/n')"
        )
    test_conn.commit()
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.night SET zp_source = 'measured'")
    test_conn.rollback()

    assert init_schema(test_conn) == [8, 9, 10, 11, 12]
    assert init_schema(test_conn) == []

    with test_conn.cursor() as cur:
        # the existing night is untouched; the backfill of deploy/README.md rewrites it
        cur.execute("SELECT zp, zp_source FROM relphot.night")
        assert cur.fetchall() == [(20.0, "assumed")]
        cur.execute(
            "UPDATE relphot.night SET zp = 27.85, zp_source = 'measured' "
            "WHERE telescope = 'T80S' AND zp_source = 'assumed'"
        )
        cur.execute("SELECT zp, zp_source FROM relphot.night")
        assert cur.fetchall() == [(pytest.approx(27.85), "measured")]
        for source in ("gaia", "assumed"):
            cur.execute("UPDATE relphot.night SET zp_source = %s", (source,))
    test_conn.commit()
    with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.night SET zp_source = 'bogus'")
    test_conn.rollback()


def test_migration_009_coincidence_columns_and_table_from_v8(test_conn) -> None:
    _apply_up_to(test_conn, 8)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n', '/tmp/n') RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 1, 1) RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind) VALUES (%s, %s, 'transit') "
            "RETURNING det_id",
            (obj_id, night_id),
        )
        (det_id,) = cur.fetchone()
    test_conn.commit()

    assert init_schema(test_conn) == [9, 10, 11, 12]
    assert init_schema(test_conn) == []

    with test_conn.cursor() as cur:
        # an existing detection has no automatic verdict; the person's status is its own column
        cur.execute(
            "SELECT status, auto_status, auto_reason FROM relphot.detection WHERE det_id = %s",
            (det_id,),
        )
        assert cur.fetchone() == ("UNCONFIRMED", None, None)
        cur.execute(
            "UPDATE relphot.detection SET auto_status = 'REJECTED', auto_reason = 'why' "
            "WHERE det_id = %s",
            (det_id,),
        )
        cur.execute(
            "INSERT INTO relphot.transit_coincidence (det_id, night_id, n_similar, n_expected, "
            "p_chance, similar_det_ids, rejected) VALUES (%s, %s, 3, 0.5, 1e-9, %s, true)",
            (det_id, night_id, [det_id + 1, det_id + 2]),
        )
        cur.execute(
            "SELECT n_similar, n_expected, p_chance, similar_det_ids, rejected, "
            "computed_at IS NOT NULL FROM relphot.transit_coincidence WHERE det_id = %s",
            (det_id,),
        )
        assert cur.fetchone() == (3, 0.5, 1e-9, [det_id + 1, det_id + 2], True, True)
        cur.execute(
            "SELECT relname FROM pg_class WHERE relname = 'transit_coincidence_night_idx'"
        )
        assert cur.fetchone() == ("transit_coincidence_night_idx",)
        for role in ("relphot_ro", "relphot_web"):
            cur.execute(
                "SELECT has_table_privilege(%s, 'relphot.transit_coincidence', 'SELECT'), "
                "has_table_privilege(%s, 'relphot.transit_coincidence', 'INSERT')",
                (role, role),
            )
            assert cur.fetchone() == (True, False)
    test_conn.commit()

    bad = (
        (
            "UPDATE relphot.detection SET auto_status = 'CONFIRMED'", (),
            psycopg.errors.CheckViolation,
        ),
        (
            "INSERT INTO relphot.transit_coincidence (det_id, night_id, rejected) "
            "VALUES (999999, %s, false)",
            (night_id,), psycopg.errors.ForeignKeyViolation,
        ),
        (
            "INSERT INTO relphot.transit_coincidence (det_id, night_id) VALUES (%s, %s)",
            (det_id + 100, night_id), psycopg.errors.NotNullViolation,
        ),
        (
            "INSERT INTO relphot.transit_coincidence (det_id, night_id, rejected) "
            "VALUES (%s, %s, false)",
            (det_id, night_id), psycopg.errors.UniqueViolation,
        ),
    )
    for sql, params, error in bad:
        with pytest.raises(error), test_conn.cursor() as cur:
            cur.execute(sql, params)
        test_conn.rollback()

    # the rows go with their detection
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.detection WHERE det_id = %s", (det_id,))
        cur.execute("SELECT count(*) FROM relphot.transit_coincidence")
        assert cur.fetchone() == (0,)
    test_conn.commit()


def test_migration_006_guided_reprocessing_schema_from_v5(test_conn) -> None:
    _apply_up_to(test_conn, 5)
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
        cur.execute(
            "INSERT INTO relphot.star_night (obj_id, night_id, star_id) VALUES (%s, %s, 1)",
            (obj_id, night_id),
        )
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind) VALUES (%s, %s, 'transit')",
            (obj_id, night_id),
        )
        cur.execute(
            "INSERT INTO relphot.period_estimate (obj_id, method, night_ids) "
            "VALUES (%s, 'LS', %s)",
            (obj_id, [1]),
        )
    test_conn.commit()

    assert init_schema(test_conn) == [6, 7, 8, 9, 10, 11, 12]
    assert current_version(test_conn) == 12

    with test_conn.cursor() as cur:
        # existing rows: searches' detections, no inflation information, no new estimate info
        cur.execute("SELECT origin FROM relphot.detection")
        assert cur.fetchall() == [("search",)]
        cur.execute("SELECT err_scale, blended FROM relphot.star_night")
        assert cur.fetchall() == [(None, None)]
        cur.execute(
            "SELECT guess, phase_coverage, n_cycles, alias_periods, alias_powers "
            "FROM relphot.period_estimate"
        )
        assert cur.fetchall() == [(None, None, None, None, None)]

    def rejected(sql: str, params: tuple = ()) -> None:
        with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
            cur.execute(sql, params)
        test_conn.rollback()

    rejected("UPDATE relphot.detection SET origin = 'bogus'")
    rejected("UPDATE relphot.period_estimate SET method = 'bogus'")
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.period_estimate SET method = 'LS-guided', guess = 1.5")
        cur.execute(
            "UPDATE relphot.period_estimate SET verify_status = 'long_period_needs_tie'"
        )
        # the queue: kinds and the guesses each kind needs are enforced by the table
        for sql in (
            "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess) "
            "VALUES (%s, 'variable', 1.0)",
            "INSERT INTO relphot.reprocess_request (obj_id, kind, tc_guess, width_guess_h) "
            "VALUES (%s, 'transit', 2460000.5, 2.0)",
        ):
            cur.execute(sql, (obj_id,))
        cur.execute("SELECT status, requested_at IS NOT NULL FROM relphot.reprocess_request")
        assert cur.fetchall() == [("queued", True)] * 2
    test_conn.commit()
    rejected(
        "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess) "
        "VALUES (%s, 'variable', 0)", (obj_id,)
    )
    rejected(
        "INSERT INTO relphot.reprocess_request (obj_id, kind, tc_guess) "
        "VALUES (%s, 'transit', 2460000.5)", (obj_id,)
    )
    rejected(
        "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess) "
        "VALUES (%s, 'bls', 1.0)", (obj_id,)
    )
    rejected("UPDATE relphot.reprocess_request SET status = 'bogus'")
    # deleting the object takes its requests with it
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.object WHERE obj_id = %s", (obj_id,))
        cur.execute("SELECT count(*) FROM relphot.reprocess_request")
        assert cur.fetchone() == (0,)
    test_conn.commit()


def test_migration_006_user_night_review_from_v5_backfills_object_level_flags(test_conn) -> None:
    _apply_up_to(test_conn, 5)
    with test_conn.cursor() as cur:
        night_ids = []
        for label in ("n1", "n2"):
            cur.execute(
                "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
                "VALUES ('T80S', '2025-01-01', %s, %s) RETURNING night_id",
                (label, f"/tmp/{label}"),
            )
            night_ids.append(cur.fetchone()[0])
        # (name, is_exop, exop_source, is_var, var_source, nights it was observed on)
        specs = [
            ("auto", False, "auto", False, "auto", night_ids),
            ("confirmed_exop", True, "manual", False, "auto", night_ids),
            ("rejected_exop_confirmed_var", False, "manual", True, "manual", night_ids[:1]),
            ("no_nights", True, "manual", False, "auto", []),
        ]
        obj_ids = {}
        for star_id, (name, is_exop, exop_source, is_var, var_source, nights) in enumerate(
            specs
        ):
            cur.execute(
                "INSERT INTO relphot.object (name, ra, dec, is_exop, exop_source, is_var, "
                "var_source) VALUES (%s, 0, 0, %s, %s, %s, %s) RETURNING obj_id",
                (name, is_exop, exop_source, is_var, var_source),
            )
            obj_ids[name] = cur.fetchone()[0]
            for night_id in nights:
                cur.execute(
                    "INSERT INTO relphot.star_night (obj_id, night_id, star_id) "
                    "VALUES (%s, %s, %s)",
                    (obj_ids[name], night_id, star_id),
                )
    test_conn.commit()

    assert init_schema(test_conn) == [6, 7, 8, 9, 10, 11, 12]

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, night_id, exop_verdict, var_verdict, note FROM "
            "relphot.user_night_review ORDER BY obj_id, night_id"
        )
        rows = cur.fetchall()
        cur.execute("SELECT n_review_pending, n_nights_reviewed FROM relphot.object")
        counters = set(cur.fetchall())
    n1, n2 = night_ids
    assert rows == [
        (obj_ids["confirmed_exop"], n1, "CONFIRMED", None, None),
        (obj_ids["confirmed_exop"], n2, "CONFIRMED", None, None),
        (obj_ids["rejected_exop_confirmed_var"], n1, "REJECTED", "CONFIRMED", None),
    ]  # 'auto' gets no row; an object without nights gets none either
    assert counters == {(0, 0)}  # filled by `relphot db analyze --all`


def test_migration_006_grants_on_user_night_review(test_conn) -> None:
    init_schema(test_conn)
    table = "relphot.user_night_review"

    def can(role: str, privilege: str, column: str | None = None) -> bool:
        with test_conn.cursor() as cur:
            if column is None:
                cur.execute("SELECT has_table_privilege(%s, %s, %s)", (role, table, privilege))
            else:
                cur.execute(
                    "SELECT has_column_privilege(%s, %s, %s, %s)", (role, table, column, privilege)
                )
            return cur.fetchone()[0]

    assert can("relphot_ro", "SELECT")
    assert not any(can("relphot_ro", p) for p in ("INSERT", "UPDATE", "DELETE"))
    assert can("relphot_web", "SELECT")
    assert can("relphot_web", "DELETE")
    for column in ("obj_id", "night_id", "exop_verdict", "var_verdict", "note"):
        assert can("relphot_web", "INSERT", column)
    assert not can("relphot_web", "INSERT", "updated_at")
    for column in ("exop_verdict", "var_verdict", "note", "updated_at"):
        assert can("relphot_web", "UPDATE", column)
    for column in ("obj_id", "night_id"):
        assert not can("relphot_web", "UPDATE", column)
    with test_conn.cursor() as cur:
        for column in ("n_review_pending", "n_nights_reviewed"):
            cur.execute(
                "SELECT has_column_privilege('relphot_web', 'relphot.object', %s, 'UPDATE')",
                (column,),
            )
            assert cur.fetchone()[0]


def test_user_night_review_check_cascades_and_independence_from_star_night(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 0, 0) RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        night_ids = []
        for label in ("n1", "n2"):
            cur.execute(
                "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
                "VALUES ('T80S', '2025-01-01', %s, %s) RETURNING night_id",
                (label, f"/tmp/{label}"),
            )
            night_ids.append(cur.fetchone()[0])
        cur.execute(
            "INSERT INTO relphot.star_night (obj_id, night_id, star_id) VALUES (%s, %s, 1)",
            (obj_id, night_ids[0]),
        )
        cur.execute(
            "SELECT n_review_pending, n_nights_reviewed FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (0, 0)
    test_conn.commit()

    def rejected(sql: str, params: tuple) -> None:
        with pytest.raises(psycopg.errors.CheckViolation), test_conn.cursor() as cur:
            cur.execute(sql, params)
        test_conn.rollback()

    rejected(
        "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
        "VALUES (%s, %s, 'MAYBE')", (obj_id, night_ids[0]),
    )
    rejected(
        "INSERT INTO relphot.user_night_review (obj_id, night_id, var_verdict) "
        "VALUES (%s, %s, 'confirmed')", (obj_id, night_ids[0]),
    )

    def insert_reviews() -> None:
        with test_conn.cursor() as cur:
            for night_id in night_ids:
                cur.execute(
                    "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict, "
                    "var_verdict, note) VALUES (%s, %s, 'CONFIRMED', NULL, 'n') "
                    "ON CONFLICT DO NOTHING",
                    (obj_id, night_id),
                )
        test_conn.commit()

    def n_reviews() -> int:
        with test_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM relphot.user_night_review")
            return cur.fetchone()[0]

    insert_reviews()
    assert n_reviews() == 2
    # a reload deletes and re-creates star_night rows: the review must not follow them
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.star_night WHERE obj_id = %s", (obj_id,))
    test_conn.commit()
    assert n_reviews() == 2
    # a review row is unique per (object, night)
    with pytest.raises(psycopg.errors.UniqueViolation), test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id) VALUES (%s, %s)",
            (obj_id, night_ids[0]),
        )
    test_conn.rollback()

    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.night WHERE night_id = %s", (night_ids[0],))
    test_conn.commit()
    assert n_reviews() == 1  # deleting a night takes its reviews with it

    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.object WHERE obj_id = %s", (obj_id,))
    test_conn.commit()
    assert n_reviews() == 0  # so does deleting the object


def test_reprocess_insert_notifies_the_worker_channel(test_conn) -> None:
    init_schema(test_conn)
    listener = psycopg.connect(_test_dsn(), autocommit=True)
    try:
        listener.execute("LISTEN relphot_reprocess")
        with test_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 0, 0) RETURNING obj_id"
            )
            (obj_id,) = cur.fetchone()
            cur.execute(
                "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess) "
                "VALUES (%s, 'variable', 1.0) RETURNING req_id",
                (obj_id,),
            )
            (req_id,) = cur.fetchone()
        test_conn.commit()
        notes = list(listener.notifies(timeout=5.0, stop_after=1))
    finally:
        listener.close()
    assert [n.payload for n in notes] == [str(req_id)]


def test_deleting_a_night_cascade_deletes_its_reprocess_requests(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('obj', 0, 0) "
            "RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'n1', '/tmp') RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.reprocess_request "
            "(obj_id, kind, period_guess, night_id) "
            "VALUES (%s, 'variable', 1.0, %s)",
            (obj_id, night_id),
        )
    test_conn.commit()

    with test_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM relphot.reprocess_request")
        assert cur.fetchone()[0] == 1

    # Delete the night
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.night WHERE night_id = %s", (night_id,))
    test_conn.commit()

    # The reprocess_request should be cascade-deleted
    with test_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM relphot.reprocess_request")
        assert cur.fetchone()[0] == 0


def test_reprocess_request_without_night_id_is_valid_all_nights(test_conn) -> None:
    init_schema(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('obj', 0, 0) "
            "RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.reprocess_request "
            "(obj_id, kind, period_guess) "
            "VALUES (%s, 'variable', 1.0) RETURNING req_id",
            (obj_id,),
        )
        (req_id,) = cur.fetchone()
    test_conn.commit()

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, kind, period_guess, night_id FROM relphot.reprocess_request "
            "WHERE req_id = %s",
            (req_id,),
        )
        row = cur.fetchone()
    assert row == (obj_id, "variable", 1.0, None)


def test_migration_010_loose_night_ids_from_v9(test_conn) -> None:
    _apply_up_to(test_conn, 9)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, labels, anchor) VALUES ('old', %s, 'a')",
            (["a", "b"],),
        )
    test_conn.commit()
    assert init_schema(test_conn) == [10, 11, 12]
    assert init_schema(test_conn) == []

    with test_conn.cursor() as cur:
        # a run loaded before the version has no loose night; the column can never be NULL
        cur.execute("SELECT loose_night_ids FROM relphot.mn_run WHERE stem = 'old'")
        assert cur.fetchone() == ([],)
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, loose_night_ids) VALUES ('new', %s) "
            "RETURNING loose_night_ids",
            ([7, 9],),
        )
        assert cur.fetchone() == ([7, 9],)
        for role in ("relphot_ro", "relphot_web"):
            cur.execute(
                "SELECT has_column_privilege(%s, 'relphot.mn_run', 'loose_night_ids', 'SELECT')",
                (role,),
            )
            assert cur.fetchone() == (True,)
    test_conn.commit()
    with pytest.raises(psycopg.errors.NotNullViolation), test_conn.cursor() as cur:
        cur.execute("INSERT INTO relphot.mn_run (stem, loose_night_ids) VALUES ('bad', NULL)")
    test_conn.rollback()


def test_migration_011_members_from_v10(test_conn) -> None:
    _apply_up_to(test_conn, 10)
    with test_conn.cursor() as cur:
        # Insert a night
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-01', 'test', '/tmp/test') RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
        # Insert an object
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('obj1', 1.0, 1.0) "
            "RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
    test_conn.commit()

    assert init_schema(test_conn) == [11, 12]
    assert init_schema(test_conn) == []

    with test_conn.cursor() as cur:
        # Test night_tile creation and PK uniqueness
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile) VALUES (%s, 0)",
            (night_id,),
        )
        cur.execute(
            "INSERT INTO relphot.tile_lc (night_id, tile, aperture, ref_flux, ref_flux_err) "
            "VALUES (%s, 0, 0, %s, %s)",
            (night_id, [1.0, 2.0, float('nan')], [0.1, 0.2, float('nan')]),
        )
        cur.execute(
            "INSERT INTO relphot.reference_member"
            " (night_id, tile, star_id, obj_id, ra, dec, mag, weight, in_core) "
            "VALUES (%s, 0, 100, %s, 1.0, 1.0, 10.0, 0.5, true)",
            (night_id, obj_id),
        )
        cur.execute(
            "INSERT INTO relphot.comparison_member"
            " (night_id, tile, aperture, star_id, obj_id, ra, dec, mag, weight, norm_flux) "
            "VALUES (%s, 0, 0, 101, NULL, 2.0, 2.0, 11.0, 0.3, %s)",
            (night_id, [1.0, 1.1, float('nan')]),
        )
    test_conn.commit()

    # Test PK duplicate rejection
    with pytest.raises(psycopg.errors.IntegrityError), test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile) VALUES (%s, 0)",
            (night_id,),
        )
    test_conn.rollback()

    # Test cascade delete from night
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.night WHERE night_id = %s", (night_id,))
    test_conn.commit()

    with test_conn.cursor() as cur:
        # All related rows should be deleted
        cur.execute("SELECT COUNT(*) FROM relphot.night_tile WHERE night_id = %s", (night_id,))
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT COUNT(*) FROM relphot.reference_member WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT COUNT(*) FROM relphot.comparison_member WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone() == (0,)

    # Insert again to test cascade from night_tile
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
            "VALUES ('T80S', '2025-01-02', 'test2', '/tmp/test2') RETURNING night_id"
        )
        (night_id2,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile) VALUES (%s, 1)",
            (night_id2,),
        )
        cur.execute(
            "INSERT INTO relphot.tile_lc (night_id, tile, aperture, ref_flux, ref_flux_err) "
            "VALUES (%s, 1, 1, %s, %s)",
            (night_id2, [1.0], [0.1]),
        )
        cur.execute(
            "INSERT INTO relphot.reference_member (night_id, tile, star_id, ra, dec) "
            "VALUES (%s, 1, 200, 3.0, 3.0)",
            (night_id2,),
        )
        cur.execute(
            "INSERT INTO relphot.comparison_member"
            " (night_id, tile, aperture, star_id, ra, dec, norm_flux) "
            "VALUES (%s, 1, 1, 201, 4.0, 4.0, %s)",
            (night_id2, [1.0]),
        )
    test_conn.commit()

    # Test cascade delete from night_tile
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.night_tile WHERE night_id = %s AND tile = 1", (night_id2,))
    test_conn.commit()

    with test_conn.cursor() as cur:
        # tile_lc, reference_member, comparison_member should be deleted
        cur.execute(
            "SELECT COUNT(*) FROM relphot.tile_lc WHERE night_id = %s AND tile = 1",
            (night_id2,),
        )
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT COUNT(*) FROM relphot.reference_member WHERE night_id = %s AND tile = 1",
            (night_id2,),
        )
        assert cur.fetchone() == (0,)
        cur.execute(
            "SELECT COUNT(*) FROM relphot.comparison_member WHERE night_id = %s AND tile = 1",
            (night_id2,),
        )
        assert cur.fetchone() == (0,)

    # Test obj_id SET NULL on object delete
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile) VALUES (%s, 2)",
            (night_id2,),
        )
        cur.execute(
            "INSERT INTO relphot.reference_member (night_id, tile, star_id, obj_id, ra, dec) "
            "VALUES (%s, 2, 300, %s, 5.0, 5.0)",
            (night_id2, obj_id),
        )
    test_conn.commit()

    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.object WHERE obj_id = %s", (obj_id,))
    test_conn.commit()

    with test_conn.cursor() as cur:
        # obj_id should be NULL but row still exists
        cur.execute(
            "SELECT obj_id FROM relphot.reference_member WHERE night_id = %s AND star_id = 300",
            (night_id2,),
        )
        assert cur.fetchone() == (None,)

    # Test role permissions
    with test_conn.cursor() as cur:
        for table in ("night_tile", "tile_lc", "reference_member", "comparison_member"):
            for role in ("relphot_ro", "relphot_web"):
                cur.execute(
                    "SELECT has_table_privilege(%s, 'relphot.' || %s, 'SELECT')",
                    (role, table),
                )
                assert cur.fetchone() == (True,), f"{role} should have SELECT on {table}"
            cur.execute(
                "SELECT has_table_privilege('relphot_ro', 'relphot.' || %s, 'INSERT')",
                (table,),
            )
            assert cur.fetchone() == (False,), f"relphot_ro should not have INSERT on {table}"
    test_conn.commit()


def test_migration_012_repeat_events_from_v11(test_conn) -> None:
    _apply_up_to(test_conn, 11)
    test_conn.commit()
    assert init_schema(test_conn) == [12]
    assert current_version(test_conn) == 12

    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec) VALUES ('o', 0, 0) RETURNING obj_id"
        )
        (obj_id,) = cur.fetchone()
        night_ids = []
        for label in ("a", "b"):
            cur.execute(
                "INSERT INTO relphot.night (telescope, night_date, label, source_dir) "
                "VALUES ('T80S', '2025-01-01', %s, %s) RETURNING night_id",
                (label, f"/tmp/{label}"),
            )
            night_ids.append(cur.fetchone()[0])
        det_ids = []
        for night_id in night_ids:
            cur.execute(
                "INSERT INTO relphot.detection (obj_id, night_id, kind) "
                "VALUES (%s, %s, 'transit') RETURNING det_id",
                (obj_id, night_id),
            )
            det_ids.append(cur.fetchone()[0])
        det_a, det_b = det_ids
        cur.execute(
            "INSERT INTO relphot.repeat_link (det_a, det_b, obj_id, night_a, night_b, phys_ok, "
            "linked, decision) VALUES (%s, %s, %s, %s, %s, true, true, 'SAME')",
            (det_a, det_b, obj_id, *night_ids),
        )
        cur.execute(
            "INSERT INTO relphot.repeat_family (obj_id, family_key, n_members, member_night_ids) "
            "VALUES (%s, 'k', 2, %s) RETURNING fam_id",
            (obj_id, night_ids),
        )
        (fam_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.repeat_family_member (fam_id, det_id) VALUES (%s, %s), (%s, %s)",
            (fam_id, det_a, fam_id, det_b),
        )
        cur.execute(
            "INSERT INTO relphot.repeat_ephemeris (fam_id, obj_id, family_key, alias_k, period, "
            "tc0, status) VALUES (%s, %s, 'k', 1, 2.0, 2460000.0, 'allowed')",
            (fam_id, obj_id),
        )
        cur.execute(
            "INSERT INTO relphot.repeat_decision (obj_id, night_a, night_b, tc_a, tc_b, decision) "
            "VALUES (%s, %s, %s, 1.0, 2.0, 'SAME')",
            (obj_id, *night_ids),
        )
        cur.execute("SELECT decision, updated_at IS NOT NULL FROM relphot.repeat_decision")
        assert cur.fetchone() == ("SAME", True)
        for role in ("relphot_ro", "relphot_web"):
            for table in ("repeat_link", "repeat_family", "repeat_family_member",
                          "repeat_ephemeris", "repeat_decision"):
                cur.execute(
                    "SELECT has_table_privilege(%s, 'relphot.' || %s, 'SELECT'), "
                    "has_table_privilege(%s, 'relphot.' || %s, 'INSERT')",
                    (role, table, role, table),
                )
                assert cur.fetchone() == (True, False)  # no role has a table-wide INSERT
        cur.execute(
            "SELECT has_table_privilege('relphot_web', 'relphot.repeat_decision', 'DELETE'), "
            "has_column_privilege('relphot_web', 'relphot.repeat_decision', 'decision', 'UPDATE'), "
            "has_column_privilege('relphot_web', 'relphot.repeat_decision', 'tc_a', 'UPDATE'), "
            "has_column_privilege('relphot_web', 'relphot.repeat_decision', 'decision', 'INSERT'), "
            "has_column_privilege('relphot_ro', 'relphot.repeat_decision', 'decision', 'INSERT'), "
            "has_column_privilege('relphot_web', 'relphot.repeat_link', 'linked', 'INSERT')"
        )
        # the web may only write the person's decisions
        assert cur.fetchone() == (True, True, False, True, False, False)
    test_conn.commit()

    bad = (
        (
            "INSERT INTO relphot.repeat_decision (obj_id, night_a, night_b, tc_a, tc_b, decision) "
            "VALUES (%s, %s, %s, 1.0, 3.0, 'MAYBE')", (obj_id, *night_ids),
            psycopg.errors.CheckViolation,
        ),
        (
            "INSERT INTO relphot.repeat_decision (obj_id, night_a, night_b, tc_a, tc_b, decision) "
            "VALUES (%s, %s, %s, 1.0, 3.0, 'SAME')", (obj_id, night_ids[1], night_ids[0]),
            psycopg.errors.CheckViolation,
        ),
        (
            "INSERT INTO relphot.repeat_ephemeris (obj_id, family_key, alias_k, period, tc0, "
            "status) VALUES (%s, 'x', 1, 1.0, 1.0, 'maybe')", (obj_id,),
            psycopg.errors.CheckViolation,
        ),
        (
            "INSERT INTO relphot.repeat_link (det_a, det_b, obj_id, night_a, night_b, phys_ok, "
            "linked) VALUES (%s, %s, %s, 1, 2, true, true)", (det_b, det_a, obj_id),
            psycopg.errors.CheckViolation,
        ),
        (
            "INSERT INTO relphot.repeat_ephemeris (obj_id, family_key, alias_k, period, tc0, "
            "status) VALUES (%s, 'k', 1, 1.0, 1.0, 'allowed')", (obj_id,),
            psycopg.errors.UniqueViolation,
        ),
    )
    for sql, params, error in bad:
        with pytest.raises(error), test_conn.cursor() as cur:
            cur.execute(sql, params)
        test_conn.rollback()

    with test_conn.cursor() as cur:
        # the rows of a reloaded detection go with it; the ephemeris is kept as history
        cur.execute("DELETE FROM relphot.detection WHERE det_id = %s", (det_a,))
        cur.execute("SELECT count(*) FROM relphot.repeat_link")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM relphot.repeat_family_member")
        assert cur.fetchone() == (1,)
        cur.execute("DELETE FROM relphot.repeat_family WHERE fam_id = %s", (fam_id,))
        cur.execute("SELECT count(*), count(fam_id) FROM relphot.repeat_ephemeris")
        assert cur.fetchone() == (1, 0)
        cur.execute("SELECT count(*) FROM relphot.repeat_decision")
        assert cur.fetchone() == (1,)
    test_conn.commit()
