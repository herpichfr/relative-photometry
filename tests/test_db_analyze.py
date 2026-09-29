"""Tests for relphot.db.analyze against a live PostgreSQL test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for
how it is resolved). Skipped with an explicit reason when no such DSN is
available. Fixtures insert night/object/star_night/lightcurve/detection rows
directly with plain SQL (not through :func:`relphot.db.load_night.load_night`)
so each test controls its light curve's timestamps and signal precisely.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import psycopg
import pytest
from psycopg.types.json import Jsonb

from relphot.config import DbSettings, Settings
from relphot.db.analyze import (
    _NightData,
    _ObjTask,
    _phase_stats,
    _refine_fourier,
    _tied_series,
    _verification_windows,
    analyze,
)
from relphot.db.connect import resolve_dsn
from relphot.db.refresh import refresh_objects
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError
from relphot.objflags import night_state, refresh_flags

_SETTINGS = replace(Settings(), db=replace(DbSettings(), max_expected_noise=0.05))


@pytest.fixture
def test_conn():
    try:
        dsn = resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    init_schema(conn)
    yield conn
    conn.close()


def _insert_night(conn: psycopg.Connection, label: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-01', %s, %s, now()) RETURNING night_id",
            (label, f"/tmp/{label}"),
        )
        (night_id,) = cur.fetchone()
    return night_id


def _insert_object(conn: psycopg.Connection, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, data_updated_at) "
            "VALUES (%s, 10.0, -20.0, now()) RETURNING obj_id",
            (name,),
        )
        (obj_id,) = cur.fetchone()
    return obj_id


def _insert_star_night_and_lc(
    conn: psycopg.Connection, obj_id: int, night_id: int, star_id: int,
    bjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night "
            "(obj_id, night_id, star_id, tile, mag, best_aperture, rms, "
            "expected_noise, chi2_reduced, n_epochs, is_comparison) "
            "VALUES (%s, %s, %s, 0, 15.0, 1, 0.01, 0.01, 1.0, %s, false)",
            (obj_id, night_id, star_id, bjd.size),
        )
        cur.execute(
            "INSERT INTO relphot.lightcurve "
            "(obj_id, night_id, frame_index, bjd_tdb, flux, flux_err, flux_raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                obj_id, night_id, list(range(bjd.size)),
                [float(v) for v in bjd], [float(v) for v in flux],
                [float(v) for v in flux_err], [float(v) for v in flux],
            ),
        )


def _insert_detection(conn: psycopg.Connection, obj_id: int, night_id: int, kind: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr) VALUES (%s, %s, %s, 10.0)",
            (obj_id, night_id, kind),
        )


def _refetch_object(conn: psycopg.Connection, obj_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT class, period, period_source FROM relphot.object WHERE obj_id = %s", (obj_id,)
        )
        return cur.fetchone()


def _insert_mn_run(conn: psycopg.Connection, stem: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, labels, anchor, loaded_at) "
            "VALUES (%s, %s, %s, now()) RETURNING mn_run_id",
            (stem, ["20250101", "20250102"], "20250101"),
        )
        (mn_run_id,) = cur.fetchone()
    return mn_run_id


def _insert_multinight_detection(
    conn: psycopg.Connection, obj_id: int, mn_run_id: int, kind: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, mn_run_id, kind, snr) "
            "VALUES (%s, %s, %s, 10.0)",
            (obj_id, mn_run_id, kind),
        )


def test_combined_ls_recovers_sinusoid_period(test_conn) -> None:
    obj_id = _insert_object(test_conn, "sinusoid")
    period_true = 0.3
    rng = np.random.default_rng(1)
    for i, label in enumerate(["20250101", "20250102", "20250103"]):
        night_id = _insert_night(test_conn, label)
        _insert_detection(test_conn, obj_id, night_id, "variable")
        t = np.sort(rng.uniform(i * 1.0, i * 1.0 + 0.4, 150))
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / period_true)
        flux_err = np.full_like(t, 0.001)
        flux = flux + rng.normal(0, 0.001, t.size)
        _insert_star_night_and_lc(test_conn, obj_id, night_id, i, t, flux, flux_err)
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    assert report.n_ls_combined == 1
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT fmin, df, n, power, peak_period, fap FROM relphot.periodogram "
            "WHERE obj_id = %s AND scope = 'combined' AND method = 'LS'",
            (obj_id,),
        )
        fmin, df, n, power, peak_period, fap = cur.fetchone()

    assert peak_period == pytest.approx(period_true, rel=0.01)
    assert fap <= _SETTINGS.db.ls_fap_threshold

    best_k = int(np.argmax(power))
    assert 1.0 / (fmin + best_k * df) == pytest.approx(peak_period, rel=1e-9)
    assert len(power) == n

    klass, period, period_source = _refetch_object(test_conn, obj_id)
    assert klass == "VAR"
    assert period == pytest.approx(period_true, rel=0.01)
    assert period_source == "LS"


def test_catalog_period_beats_combined_ls(test_conn) -> None:
    obj_id = _insert_object(test_conn, "catalog_beats_ls")
    period_true = 0.3
    rng = np.random.default_rng(1)
    for i, label in enumerate(["20250101", "20250102", "20250103"]):
        night_id = _insert_night(test_conn, label)
        _insert_detection(test_conn, obj_id, night_id, "variable")
        t = np.sort(rng.uniform(i * 1.0, i * 1.0 + 0.4, 150))
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / period_true)
        flux_err = np.full_like(t, 0.001)
        flux = flux + rng.normal(0, 0.001, t.size)
        _insert_star_night_and_lc(test_conn, obj_id, night_id, i, t, flux, flux_err)

    # Insert catalog match with a different period
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.catalog_match (obj_id, catalog, name, type, period) "
            "VALUES (%s, 'VSX', 'TEST V1', 'EW', 0.6)",
            (obj_id,),
        )
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    assert report.n_ls_combined == 1
    # Check that combined LS periodogram still has peak near true period
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT peak_period FROM relphot.periodogram "
            "WHERE obj_id = %s AND scope = 'combined' AND method = 'LS'",
            (obj_id,),
        )
        ls_peak_period, = cur.fetchone()

    assert ls_peak_period == pytest.approx(period_true, rel=0.01)

    # But the object's period should be the catalog period
    klass, period, period_source = _refetch_object(test_conn, obj_id)
    assert klass == "VAR"
    assert period == pytest.approx(0.6)
    assert period_source == "catalog"


def test_combined_bls_recovers_transit_period(test_conn) -> None:
    obj_id = _insert_object(test_conn, "transit")
    period_true = 1.3
    t0 = 0.05
    duration = 2.0 / 24.0
    depth = 0.02
    rng = np.random.default_rng(2)
    windows = [(0.0, 0.4), (1.0, 1.4), (2.6, 3.0)]
    for i, (lo, hi) in enumerate(windows):
        label = f"2025010{i + 1}"
        night_id = _insert_night(test_conn, label)
        _insert_detection(test_conn, obj_id, night_id, "transit")
        t = np.sort(rng.uniform(lo, hi, 200))
        phase = ((t - t0 + period_true / 2.0) % period_true) - period_true / 2.0
        in_transit = np.abs(phase) < duration / 2.0
        flux = np.where(in_transit, 1.0 - depth, 1.0)
        flux_err = np.full_like(t, 0.0005)
        flux = flux + rng.normal(0, 0.0005, t.size)
        _insert_star_night_and_lc(test_conn, obj_id, night_id, i, t, flux, flux_err)
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    assert report.n_bls == 1
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT fmin, df, n, power, peak_period, extra FROM relphot.periodogram "
            "WHERE obj_id = %s AND scope = 'combined' AND method = 'BLS'",
            (obj_id,),
        )
        fmin, df, n, power, peak_period, extra = cur.fetchone()

    assert peak_period == pytest.approx(period_true, rel=0.01)
    assert extra["depth_snr"] >= _SETTINGS.db.bls_min_snr
    best_k = int(np.argmax(power))
    assert 1.0 / (fmin + best_k * df) == pytest.approx(peak_period, rel=1e-9)
    assert len(power) == n

    klass, period, period_source = _refetch_object(test_conn, obj_id)
    assert klass == "EXOP"
    assert period == pytest.approx(period_true, rel=0.01)
    assert period_source == "BLS"


def test_dirty_logic_skips_clean_reanalyses_dirty(test_conn) -> None:
    obj_id = _insert_object(test_conn, "dirty")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "variable")
    t = np.linspace(0.0, 0.4, 50)
    flux = np.ones_like(t)
    flux_err = np.full_like(t, 0.01)
    _insert_star_night_and_lc(test_conn, obj_id, night_id, 0, t, flux, flux_err)
    test_conn.commit()

    report1 = analyze(test_conn, settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report1.n_objects == 1

    report2 = analyze(test_conn, settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report2.n_objects == 0

    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.object SET data_updated_at = now() WHERE obj_id = %s", (obj_id,)
        )
    test_conn.commit()

    report3 = analyze(test_conn, settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report3.n_objects == 1


def test_manual_period_is_not_overwritten(test_conn) -> None:
    obj_id = _insert_object(test_conn, "manual")
    period_true = 0.3
    rng = np.random.default_rng(3)
    for i, label in enumerate(["20250101", "20250102", "20250103"]):
        night_id = _insert_night(test_conn, label)
        _insert_detection(test_conn, obj_id, night_id, "variable")
        t = np.sort(rng.uniform(i * 1.0, i * 1.0 + 0.4, 150))
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / period_true) + rng.normal(0, 0.001, t.size)
        flux_err = np.full_like(t, 0.001)
        _insert_star_night_and_lc(test_conn, obj_id, night_id, i, t, flux, flux_err)
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.object SET period = 42.0, period_source = 'manual' WHERE obj_id = %s",
            (obj_id,),
        )
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    _klass, period, period_source = _refetch_object(test_conn, obj_id)
    assert period == pytest.approx(42.0)
    assert period_source == "manual"


def test_multinight_bls_detection_excluded_from_class_by_default(test_conn) -> None:
    obj_id = _insert_object(test_conn, "mn_bls_default")
    mn_run_id = _insert_mn_run(test_conn, "mn_bls_default_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "bls")
    test_conn.commit()

    refresh_objects(test_conn, [obj_id])
    test_conn.commit()

    klass, _period, _period_source = _refetch_object(test_conn, obj_id)
    assert klass == "UNC"


def test_multinight_bls_detection_included_when_configured(test_conn) -> None:
    obj_id = _insert_object(test_conn, "mn_bls_configured")
    mn_run_id = _insert_mn_run(test_conn, "mn_bls_configured_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "bls")
    test_conn.commit()

    refresh_objects(test_conn, [obj_id], class_multinight_kinds=("bls",))
    test_conn.commit()

    klass, _period, _period_source = _refetch_object(test_conn, obj_id)
    assert klass == "EXOP"


def test_multinight_recurrent_detection_sets_var_by_default(test_conn) -> None:
    obj_id = _insert_object(test_conn, "mn_recurrent_default")
    mn_run_id = _insert_mn_run(test_conn, "mn_recurrent_default_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "recurrent")
    test_conn.commit()

    refresh_objects(test_conn, [obj_id])
    test_conn.commit()

    klass, _period, _period_source = _refetch_object(test_conn, obj_id)
    assert klass == "VAR"


def test_coarsening_triggers_with_tiny_max_points(test_conn) -> None:
    obj_id = _insert_object(test_conn, "coarse")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "variable")
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 0.4, 200))
    flux = 1.0 + 0.02 * np.sin(2 * np.pi * t / 0.05) + rng.normal(0, 0.001, t.size)
    flux_err = np.full_like(t, 0.001)
    _insert_star_night_and_lc(test_conn, obj_id, night_id, 0, t, flux, flux_err)
    test_conn.commit()

    tiny_settings = replace(_SETTINGS, db=replace(_SETTINGS.db, max_periodogram_points=5))
    report = analyze(test_conn, obj_ids=[obj_id], settings=tiny_settings, workers=1)
    test_conn.commit()

    assert report.n_coarsened >= 1
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT coarsened, n FROM relphot.periodogram "
            "WHERE obj_id = %s AND scope LIKE 'night:%%'",
            (obj_id,),
        )
        coarsened, n = cur.fetchone()
    assert coarsened is True
    assert n <= 5


# --------------------------------------------------------------------------
# CLASS flags: is_exop / is_var are independent, derived from the literature, the detections
# and the person's per-night verdicts (relphot.user_night_review)
# --------------------------------------------------------------------------


def _insert_catalog_match(
    conn: psycopg.Connection, obj_id: int, catalog: str, name: str, *,
    var_type: str | None = None, period: float | None = None, period_err: float | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.catalog_match (obj_id, catalog, name, type, period, period_err) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (obj_id, catalog, name, var_type, period, period_err),
        )


def _flags(conn: psycopg.Connection, obj_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, is_var, exop_source, var_source, class, class_source, status "
            "FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        return cur.fetchone()


def test_known_planet_and_variable_detection_are_both_flags(test_conn) -> None:
    obj_id = _insert_object(test_conn, "planet_and_var")
    night_id = _insert_night(test_conn, "20250101")
    _insert_catalog_match(test_conn, obj_id, "NASA Exoplanet Archive", "Test b", period=3.5)
    _insert_detection(test_conn, obj_id, night_id, "variable")
    test_conn.commit()

    refresh_objects(test_conn, [obj_id])
    test_conn.commit()

    is_exop, is_var, exop_source, var_source, klass, class_source, status = _flags(
        test_conn, obj_id
    )
    assert (is_exop, is_var) == (True, True)
    assert (exop_source, var_source) == ("auto", "auto")
    assert (klass, class_source) == ("EXOP+VAR", "auto")
    assert status == "UNCONFIRMED"  # object status is never set automatically


def test_transit_detection_on_known_variable_is_both_flags(test_conn) -> None:
    obj_id = _insert_object(test_conn, "transit_on_var")
    night_id = _insert_night(test_conn, "20250101")
    _insert_catalog_match(test_conn, obj_id, "VSX", "V* Test", var_type="EA", period=1.2)
    _insert_detection(test_conn, obj_id, night_id, "transit")
    test_conn.commit()

    refresh_objects(test_conn, [obj_id])
    test_conn.commit()

    is_exop, is_var, *_rest, klass, _class_source, _status = _flags(test_conn, obj_id)
    assert (is_exop, is_var, klass) == (True, True, "EXOP+VAR")


def _insert_review(
    conn: psycopg.Connection, obj_id: int, night_id: int, *,
    exop: str | None = None, var: str | None = None, note: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.user_night_review "
            "(obj_id, night_id, exop_verdict, var_verdict, note) VALUES (%s, %s, %s, %s, %s)",
            (obj_id, night_id, exop, var, note),
        )


def _insert_status_detection(
    conn: psycopg.Connection, obj_id: int, night_id: int, kind: str, status: str | None,
    origin: str = "search",
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, status, origin) "
            "VALUES (%s, %s, %s, 10.0, %s, %s) RETURNING det_id",
            (obj_id, night_id, kind, status, origin),
        )
        (det_id,) = cur.fetchone()
    return det_id


def _refresh(conn: psycopg.Connection, obj_id: int, **kwargs) -> None:
    refresh_objects(conn, [obj_id], **kwargs)
    conn.commit()


def _counts(conn: psycopg.Connection, obj_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n_review_pending, n_nights_reviewed FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        return cur.fetchone()


def test_night_reject_verdict_cannot_remove_a_known_planet_and_var_stays_automatic(
    test_conn,
) -> None:
    obj_id = _insert_object(test_conn, "planet_rejected_night")
    night_id = _insert_night(test_conn, "20250101")
    _insert_catalog_match(test_conn, obj_id, "TOI", "TOI-1.01", period=2.0)
    _insert_detection(test_conn, obj_id, night_id, "variable")
    _insert_review(test_conn, obj_id, night_id, exop="REJECTED")
    test_conn.commit()

    _refresh(test_conn, obj_id)

    is_exop, is_var, exop_source, var_source, klass, class_source, _status = _flags(
        test_conn, obj_id
    )
    assert (is_exop, exop_source) == (True, "manual")  # literature always counts
    assert (is_var, var_source) == (True, "auto")  # the other flag still follows the data
    assert (klass, class_source) == ("EXOP+VAR", "manual")


@pytest.mark.parametrize(
    ("catalog", "verdict_key", "flag_index"),
    [("NASA Exoplanet Archive", "exop", 0), ("TOI", "exop", 0), ("VSX", "var", 1)],
)
def test_a_rejected_night_never_removes_literature_evidence(
    test_conn, catalog, verdict_key, flag_index
) -> None:
    obj_id = _insert_object(test_conn, "literature")
    night_id = _insert_night(test_conn, "20250101")
    _insert_catalog_match(test_conn, obj_id, catalog, "Known 1", period=2.0)
    _insert_review(test_conn, obj_id, night_id, **{verdict_key: "REJECTED"})
    test_conn.commit()

    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[flag_index] is True


def test_var_confirmed_verdict_without_evidence_counts_and_is_manual(test_conn) -> None:
    obj_id = _insert_object(test_conn, "confirmed_var")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "transit")
    _insert_review(test_conn, obj_id, night_id, var="CONFIRMED")
    test_conn.commit()

    for _ in range(2):  # a second refresh changes nothing
        _refresh(test_conn, obj_id)
        is_exop, is_var, exop_source, var_source, klass, class_source, _status = _flags(
            test_conn, obj_id
        )
        assert (is_exop, exop_source) == (True, "auto")
        assert (is_var, var_source) == (True, "manual")
        assert (klass, class_source) == ("EXOP+VAR", "manual")


def test_rejected_night_is_not_exop_and_a_new_night_is_evaluated_automatically(
    test_conn,
) -> None:
    obj_id = _insert_object(test_conn, "per_night")
    night1 = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night1, "transit")
    _insert_review(test_conn, obj_id, night1, exop="REJECTED")
    test_conn.commit()

    _refresh(test_conn, obj_id)
    is_exop, _is_var, exop_source, _vs, klass, class_source, _status = _flags(test_conn, obj_id)
    assert (is_exop, exop_source, klass, class_source) == (False, "manual", "UNC", "manual")
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT exop_verdict, var_verdict, note, updated_at FROM relphot.user_night_review"
        )
        review_before = cur.fetchall()

    # a new night arrives with its own transit: evaluated automatically, the old verdict stays
    night2 = _insert_night(test_conn, "20250102")
    _insert_detection(test_conn, obj_id, night2, "transit")
    test_conn.commit()
    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[:5:4] == (True, "EXOP")
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT night_id, exop_verdict, var_verdict, note, updated_at "
            "FROM relphot.user_night_review"
        )
        rows = cur.fetchall()
    assert rows == [(night1, *review_before[0])]


def test_confirmed_night_verdict_without_a_detection_sets_the_flag(test_conn) -> None:
    obj_id = _insert_object(test_conn, "confirmed_exop")
    night_id = _insert_night(test_conn, "20250101")
    _insert_review(test_conn, obj_id, night_id, exop="CONFIRMED")
    test_conn.commit()

    _refresh(test_conn, obj_id)

    is_exop, is_var, exop_source, var_source, klass, class_source, _status = _flags(
        test_conn, obj_id
    )
    assert (is_exop, is_var) == (True, False)
    assert (exop_source, var_source) == ("manual", "auto")
    assert (klass, class_source) == ("EXOP", "manual")
    assert _counts(test_conn, obj_id) == (0, 1)


def test_exop_and_var_verdicts_are_independent_and_both_can_hold(test_conn) -> None:
    obj_id = _insert_object(test_conn, "both_flags")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "transit")
    _insert_detection(test_conn, obj_id, night_id, "variable")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[:2] == (True, True)

    _insert_review(test_conn, obj_id, night_id, exop="REJECTED", var="CONFIRMED")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[:6] == (False, True, "manual", "manual", "VAR", "manual")

    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.user_night_review SET exop_verdict = 'CONFIRMED', "
            "var_verdict = 'REJECTED'"
        )
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[:2] == (True, False)

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.user_night_review SET var_verdict = 'CONFIRMED'")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[:2] == (True, True)
    assert _flags(test_conn, obj_id)[4] == "EXOP+VAR"


def test_sources_are_derived_from_the_presence_of_verdicts(test_conn) -> None:
    obj_id = _insert_object(test_conn, "sources")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "transit")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[2:6:1] == ("auto", "auto", "EXOP", "auto")

    _insert_review(test_conn, obj_id, night_id, exop="CONFIRMED")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[2:6:1] == ("manual", "auto", "EXOP", "manual")

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.user_night_review SET var_verdict = 'REJECTED'")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[2:6:1] == ("manual", "manual", "EXOP", "manual")

    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.user_night_review")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[2:6:1] == ("auto", "auto", "EXOP", "auto")


def test_a_rejected_detection_removes_only_that_events_evidence(test_conn) -> None:
    obj_id = _insert_object(test_conn, "two_transits")
    night_id = _insert_night(test_conn, "20250101")
    det_a = _insert_status_detection(test_conn, obj_id, night_id, "transit", "UNCONFIRMED")
    det_b = _insert_status_detection(test_conn, obj_id, night_id, "transit", "UNCONFIRMED")
    test_conn.commit()

    def set_status(det_id: int, status: str) -> None:
        with test_conn.cursor() as cur:
            cur.execute(
                "UPDATE relphot.detection SET status = %s WHERE det_id = %s", (status, det_id)
            )
        test_conn.commit()

    set_status(det_a, "REJECTED")
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[0] is True  # the other event still stands

    set_status(det_b, "REJECTED")
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[0] is False

    set_status(det_b, "CONFIRMED")  # a CONFIRMED event is evidence, and no longer awaits review
    _refresh(test_conn, obj_id)
    assert _flags(test_conn, obj_id)[0] is True
    assert _counts(test_conn, obj_id) == (0, 1)


def test_a_null_detection_status_reads_as_unconfirmed(test_conn) -> None:
    obj_id = _insert_object(test_conn, "null_status")
    night_id = _insert_night(test_conn, "20250101")
    _insert_status_detection(test_conn, obj_id, night_id, "transit", None)
    mn_run_id = _insert_mn_run(test_conn, "null_status_stem")
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, mn_run_id, kind, snr, status) "
            "VALUES (%s, %s, 'recurrent', 10.0, NULL)",
            (obj_id, mn_run_id),
        )
    test_conn.commit()

    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[:2] == (True, True)
    assert _counts(test_conn, obj_id) == (2, 0)


def test_user_origin_detections_never_count(test_conn) -> None:
    obj_id = _insert_object(test_conn, "user_origin")
    night_id = _insert_night(test_conn, "20250101")
    _insert_status_detection(test_conn, obj_id, night_id, "transit", None, origin="user")
    test_conn.commit()

    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[:2] == (False, False)
    assert _counts(test_conn, obj_id) == (0, 0)


def test_gated_multinight_detections_count_and_a_rejected_one_does_not(test_conn) -> None:
    obj_id = _insert_object(test_conn, "mn_gated")
    mn_run_id = _insert_mn_run(test_conn, "mn_gated_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "recurrent")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "bls")
    test_conn.commit()

    _refresh(test_conn, obj_id)  # default gate: only 'recurrent' -> a variable, not a planet host
    assert _flags(test_conn, obj_id)[:2] == (False, True)
    assert _flags(test_conn, obj_id)[4] == "VAR"

    _refresh(test_conn, obj_id, class_multinight_kinds=("recurrent", "bls"))
    assert _flags(test_conn, obj_id)[:2] == (True, True)
    assert _flags(test_conn, obj_id)[4] == "EXOP+VAR"
    assert _counts(test_conn, obj_id) == (2, 0)  # both gated events still await a verdict

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED' WHERE kind = 'recurrent'")
    test_conn.commit()
    _refresh(test_conn, obj_id, class_multinight_kinds=("recurrent", "bls"))
    assert _flags(test_conn, obj_id)[:2] == (True, False)
    assert _counts(test_conn, obj_id) == (1, 0)

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED' WHERE kind = 'bls'")
    test_conn.commit()
    _refresh(test_conn, obj_id, class_multinight_kinds=("recurrent", "bls"))
    assert _flags(test_conn, obj_id)[:2] == (False, False)
    assert _counts(test_conn, obj_id) == (0, 0)


def test_a_night_verdict_does_not_touch_multinight_evidence(test_conn) -> None:
    obj_id = _insert_object(test_conn, "mn_and_night_verdict")
    night_id = _insert_night(test_conn, "20250101")
    mn_run_id = _insert_mn_run(test_conn, "mn_and_night_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "recurrent")
    _insert_review(test_conn, obj_id, night_id, var="REJECTED")
    test_conn.commit()

    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[:4] == (False, True, "auto", "manual")


def test_review_counters_lifecycle(test_conn) -> None:
    obj_id = _insert_object(test_conn, "counters")
    night1 = _insert_night(test_conn, "20250101")
    night2 = _insert_night(test_conn, "20250102")
    night3 = _insert_night(test_conn, "20250103")
    _insert_detection(test_conn, obj_id, night1, "transit")
    det2 = _insert_status_detection(test_conn, obj_id, night2, "transit", "UNCONFIRMED")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (2, 0)

    _insert_review(test_conn, obj_id, night1, exop="REJECTED")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (1, 1)

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'CONFIRMED' WHERE det_id = %s", (det2,))
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (0, 2)  # a decided event is a reviewed night

    # a night with both kinds of event stays pending until BOTH flags have a verdict
    _insert_detection(test_conn, obj_id, night3, "transit")
    _insert_detection(test_conn, obj_id, night3, "variable")
    _insert_review(test_conn, obj_id, night3, exop="CONFIRMED")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (1, 3)
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.user_night_review SET var_verdict = 'REJECTED' WHERE night_id = %s",
            (night3,),
        )
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (0, 3)

    # an unconfirmed gated multi-night event awaits review too
    mn_run_id = _insert_mn_run(test_conn, "counters_stem")
    _insert_multinight_detection(test_conn, obj_id, mn_run_id, "recurrent")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (1, 3)

    # deleting every review and resetting the event leaves only the automatic state
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.user_night_review")
        cur.execute("UPDATE relphot.detection SET status = 'UNCONFIRMED'")
    test_conn.commit()
    _refresh(test_conn, obj_id)
    assert _counts(test_conn, obj_id) == (4, 0)


def test_a_note_only_review_row_counts_as_a_reviewed_night_but_sets_no_source(test_conn) -> None:
    obj_id = _insert_object(test_conn, "note_only")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "transit")
    _insert_review(test_conn, obj_id, night_id, note="looks like a systematic")
    test_conn.commit()

    _refresh(test_conn, obj_id)

    assert _flags(test_conn, obj_id)[:4] == (True, False, "auto", "auto")
    assert _counts(test_conn, obj_id) == (1, 1)


def test_refresh_flags_rederives_every_object_and_an_empty_list_is_a_noop(test_conn) -> None:
    obj_id = _insert_object(test_conn, "stale")
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.object SET is_exop = true, is_var = true, class = 'EXOP+VAR' "
            "WHERE obj_id = %s",
            (obj_id,),
        )
    test_conn.commit()

    refresh_flags(test_conn, [], class_multinight_kinds=("recurrent",))
    assert _flags(test_conn, obj_id)[:2] == (True, True)
    refresh_flags(test_conn, class_multinight_kinds=("recurrent",))  # every object
    test_conn.commit()
    is_exop, is_var, exop_source, var_source, klass, class_source, _status = _flags(
        test_conn, obj_id
    )
    assert (is_exop, is_var, klass) == (False, False, "UNC")
    assert (exop_source, var_source, class_source) == ("auto", "auto", "auto")


def test_period_logic_runs_on_the_flags_of_this_refresh(test_conn) -> None:
    def add_combined_bls(obj_id: int) -> None:
        with test_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relphot.periodogram (obj_id, scope, method, fmin, df, n, power, "
                "peak_period, peak_power, fap, computed_at, extra) "
                "VALUES (%s, 'combined', 'BLS', 0.1, 0.01, 3, %s, 1.3, 0.9, NULL, now(), %s)",
                (obj_id, [0.1, 0.9, 0.1], Jsonb({"depth_snr": 50.0})),
            )

    # a planet host only through the person's CONFIRMED verdict: the BLS period is used
    confirmed = _insert_object(test_conn, "bls_confirmed")
    night_id = _insert_night(test_conn, "20250101")
    _insert_review(test_conn, confirmed, night_id, exop="CONFIRMED")
    add_combined_bls(confirmed)
    # the only transit night is REJECTED: not a planet host, so the BLS branch is not taken
    rejected = _insert_object(test_conn, "bls_rejected")
    _insert_detection(test_conn, rejected, night_id, "transit")
    _insert_review(test_conn, rejected, night_id, exop="REJECTED")
    add_combined_bls(rejected)
    test_conn.commit()

    refresh_objects(test_conn, [confirmed, rejected])
    test_conn.commit()

    assert _refetch_object(test_conn, confirmed) == ("EXOP", pytest.approx(1.3), "BLS")
    assert _refetch_object(test_conn, rejected) == ("UNC", None, None)


_NIGHT_MATRIX_DETECTIONS = [
    [],
    [("transit", "UNCONFIRMED", "search")],
    [("transit", None, "search")],
    [("transit", "REJECTED", "search")],
    [("transit", "CONFIRMED", "search")],
    [("transit", "UNCONFIRMED", "search"), ("transit", "REJECTED", "search")],
    [("transit", "CONFIRMED", "search"), ("variable", "UNCONFIRMED", "search")],
    [("bls", "UNCONFIRMED", "search")],
    [("variable", "UNCONFIRMED", "search")],
    [("variable", "REJECTED", "search")],
    [("internight", "UNCONFIRMED", "search"), ("ls_periodic", "CONFIRMED", "search")],
    [("recurrent", "UNCONFIRMED", "search")],
    [("transit", "UNCONFIRMED", "user")],
    [("variable", "UNCONFIRMED", "user"), ("transit", "UNCONFIRMED", "search")],
]


def test_night_state_agrees_with_the_sql_on_a_matrix(test_conn) -> None:
    night_id = _insert_night(test_conn, "20250101")
    cases = []
    for dets in _NIGHT_MATRIX_DETECTIONS:
        for exop in (None, "CONFIRMED", "REJECTED"):
            for var in (None, "CONFIRMED", "REJECTED"):
                obj_id = _insert_object(test_conn, f"matrix{len(cases)}")
                for kind, status, origin in dets:
                    _insert_status_detection(test_conn, obj_id, night_id, kind, status, origin)
                if exop or var:
                    _insert_review(test_conn, obj_id, night_id, exop=exop, var=var)
                cases.append((obj_id, dets, exop, var))
    test_conn.commit()

    refresh_flags(test_conn, class_multinight_kinds=("recurrent",))
    test_conn.commit()

    for obj_id, dets, exop, var in cases:
        state = night_state(
            [{"kind": k, "status": s, "origin": o} for k, s, o in dets], exop, var
        )
        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT is_exop, is_var, n_review_pending FROM relphot.object WHERE obj_id = %s",
                (obj_id,),
            )
            got = cur.fetchone()
        want = (state["exop_effective"], state["var_effective"], int(state["pending"]))
        assert got == want, (dets, exop, var, got, want)


def test_night_state_reports_the_automatic_evidence_and_open_events() -> None:
    dets = [
        {"kind": "transit", "status": "UNCONFIRMED", "origin": "search"},
        {"kind": "variable", "status": "CONFIRMED", "origin": "search"},
        {"kind": "transit", "status": "REJECTED", "origin": "search"},
        {"kind": "bls", "status": "UNCONFIRMED", "origin": "user"},
    ]
    assert night_state(dets, None, None) == {
        "auto_exop": True, "auto_var": True, "exop_open": True, "var_open": False,
        "exop_effective": True, "var_effective": True, "pending": True,
    }
    decided = night_state(dets, "REJECTED", None)
    assert (decided["auto_exop"], decided["exop_effective"], decided["pending"]) == (
        True, False, False,
    )


def test_period_uses_best_period_estimate_and_catalog_period_beats_it(test_conn) -> None:
    obj_id = _insert_object(test_conn, "period_priority")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "variable")

    def insert_estimate(n_nights: int, period: float, fap: float, days_ago: int) -> None:
        with test_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relphot.period_estimate (obj_id, computed_at, method, input, "
                "night_ids, n_nights, period, period_err, fap) "
                "VALUES (%s, now() - make_interval(days => %s), 'LS', 'night', %s, %s, %s, "
                "%s, %s)",
                (obj_id, days_ago, list(range(n_nights)), n_nights, period, period * 1e-3, fap),
            )

    insert_estimate(2, 0.31, 1e-6, 0)
    insert_estimate(3, 0.30, 1e-6, 5)  # more nights wins over a later computation
    test_conn.commit()
    refresh_objects(test_conn, [obj_id])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source, period_err, period_n_nights FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        period, source, err, n_nights = cur.fetchone()
    assert (period, source, n_nights) == (pytest.approx(0.30), "LS", 3)
    assert err == pytest.approx(0.30e-3)

    # a significance gate, not a precision gate: a FAP above the threshold is not used
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.period_estimate SET fap = 0.5 WHERE obj_id = %s", (obj_id,))
    test_conn.commit()
    refresh_objects(test_conn, [obj_id])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period_source, period_err, period_n_nights FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (None, None, None)

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.period_estimate SET fap = 1e-6 WHERE obj_id = %s", (obj_id,))
    _insert_catalog_match(
        test_conn, obj_id, "VSX", "V* P", var_type="EW", period=0.6, period_err=0.002
    )
    test_conn.commit()
    refresh_objects(test_conn, [obj_id])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source, period_err, period_n_nights FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (pytest.approx(0.6), "catalog", pytest.approx(0.002), None)


# --------------------------------------------------------------------------
# D1/D2: trapezoid transit shapes and matching-transit probabilities
# --------------------------------------------------------------------------


def _trapezoid(t, tc, t14, q, depth):
    half = 0.5 * t14
    tau = max(q * t14, 1e-4)
    return 1.0 - depth * np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _insert_transit(
    conn: psycopg.Connection, obj_id: int, night_id: int, star_id: int, *,
    t_lo: float, tc: float, t14_h: float, q: float, depth: float, noise: float, seed: int,
    detection_offsets: tuple[float, float, float] = (0.002, 1.1, 0.9),
    flags: str | None = None, gap: float = 0.0, gap_at: float = 0.0,
) -> int:
    """A night's light curve with an injected trapezoid transit, plus its detection row.

    ``gap`` (days) removes the epochs within ``(tc + gap_at) +/- gap / 2``.
    """
    rng = np.random.default_rng(seed)
    t = 2460000.0 + t_lo + np.linspace(0.0, 0.4, 320)
    if gap > 0:
        t = t[np.abs(t - (2460000.0 + tc + gap_at)) > gap / 2.0]
    flux = _trapezoid(t, 2460000.0 + tc, t14_h / 24.0, q, depth)
    flux = flux + rng.normal(0.0, noise, t.size)
    _insert_star_night_and_lc(
        conn, obj_id, night_id, star_id, t, flux, np.full_like(t, noise)
    )
    d_tc, d_dur, d_depth = detection_offsets
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, flags) VALUES (%s, %s, 'transit', 12.0, %s, %s, %s, 1, %s) "
            "RETURNING det_id",
            (
                obj_id, night_id, depth * d_depth, 2460000.0 + tc + d_tc,
                t14_h * d_dur, flags,
            ),
        )
        (det_id,) = cur.fetchone()
    return det_id


def test_transit_shape_recovers_injected_trapezoid(test_conn) -> None:
    obj_id = _insert_object(test_conn, "shape")
    night_id = _insert_night(test_conn, "20250101")
    det_id = _insert_transit(
        test_conn, obj_id, night_id, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0006, seed=11,
    )
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report.n_transit_shapes == 1

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT tc, tc_err, depth, depth_err, t14_h, t14_err, ingress_frac, ingress_err, "
            "chi2_red, n_points, input, converged FROM relphot.transit_shape WHERE det_id = %s",
            (det_id,),
        )
        (tc, tc_err, depth, depth_err, t14_h, t14_err, ingress, ingress_err, chi2_red,
         n_points, input_label, converged) = cur.fetchone()
    assert converged is True
    assert input_label == "night"
    assert abs(tc - (2460000.2)) < max(4 * tc_err, 1e-3)
    assert depth == pytest.approx(0.02, abs=max(4 * depth_err, 1e-3))
    assert t14_h == pytest.approx(2.4, abs=max(4 * t14_err, 0.1))
    assert ingress == pytest.approx(0.2, abs=max(4 * ingress_err, 0.02))
    assert 0.5 < chi2_red < 2.0
    assert n_points > 30

    # a complete, unflagged event: its duration is a measurement, not a lower limit
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT t14_lower_limit, incomplete_reason FROM relphot.transit_shape "
            "WHERE det_id = %s",
            (det_id,),
        )
        assert cur.fetchone() == (False, None)
        cur.execute(
            "SELECT duration_lower_limit FROM relphot.detection WHERE det_id = %s", (det_id,)
        )
        assert cur.fetchone() == (False,)
        cur.execute("SELECT duration_lower_limit FROM relphot.object WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (False,)


def _shape_row(test_conn, det_id: int) -> tuple:
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT t14_h, t14_err, t14_lower_limit, incomplete_reason, ingress_frac, "
            "ingress_err, converged FROM relphot.transit_shape WHERE det_id = %s",
            (det_id,),
        )
        return cur.fetchone()


def test_transit_truncated_by_night_start_is_a_duration_lower_limit(test_conn) -> None:
    obj_id = _insert_object(test_conn, "edge")
    night_id = _insert_night(test_conn, "20250101")
    # the night starts mid-ingress: predicted ingress (tc - t14/2 = -0.03 d) is before the
    # first epoch, so only the part of the transit after the first epoch (~1.7 h) is seen
    det_id = _insert_transit(
        test_conn, obj_id, night_id, 0, t_lo=0.0, tc=0.02, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0006, seed=61,
    )
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    t14_h, t14_err, lower, reason, ingress, ingress_err, _converged = _shape_row(
        test_conn, det_id
    )
    assert lower is True
    assert "truncated: predicted ingress before the first epoch" in reason
    assert 1.3 < t14_h < 2.35  # the observed in-transit span, below the full ~2.4 h
    assert t14_err is None
    assert (ingress, ingress_err) == (None, None)  # the ingress was not observed
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT duration_lower_limit FROM relphot.detection WHERE det_id = %s", (det_id,)
        )
        assert cur.fetchone() == (True,)  # ORed into the detection by analyze
        cur.execute(
            "SELECT duration_h, duration_lower_limit FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        duration_h, object_lower = cur.fetchone()
    assert object_lower is True
    assert duration_h == pytest.approx(2.4 * 1.1, rel=1e-3)  # the detection's own duration


def test_transit_flagged_edge_is_a_lower_limit_with_reason(test_conn) -> None:
    obj_id = _insert_object(test_conn, "flag_edge")
    night_id = _insert_night(test_conn, "20250101")
    det_id = _insert_transit(
        test_conn, obj_id, night_id, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0006, seed=62, flags="SHARED_EPOCH|EDGE",
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    t14_h, t14_err, lower, reason, ingress, _ingress_err, converged = _shape_row(
        test_conn, det_id
    )
    assert converged is True
    assert (lower, reason) == (True, "flag EDGE")
    assert t14_h == pytest.approx(2.4, abs=0.1)  # both edges are in the data: span == fitted T14
    assert t14_err is None
    assert ingress == pytest.approx(0.2, abs=0.03)  # both ingress and egress observed: kept


def _run_gap_case(test_conn, name: str, seed: int, gap_at: float) -> tuple:
    obj_id = _insert_object(test_conn, name)
    night_id = _insert_night(test_conn, f"{name}_night")
    det_id = _insert_transit(
        test_conn, obj_id, night_id, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0006, seed=seed, gap=0.03, gap_at=gap_at,
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    return _shape_row(test_conn, det_id), det_id


def test_gap_in_the_middle_of_the_transit_is_not_a_lower_limit(test_conn) -> None:
    (t14_h, _err, lower, reason, ingress, _ingress_err, converged), det_id = _run_gap_case(
        test_conn, "gap_mid", 63, 0.0
    )
    assert converged is True
    assert (lower, reason) == (False, None)  # dropped frames do not shorten the transit
    assert t14_h == pytest.approx(2.4, abs=0.15)
    assert ingress is not None
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT duration_lower_limit FROM relphot.detection WHERE det_id = %s", (det_id,)
        )
        assert cur.fetchone() == (False,)


def test_gap_covering_egress_or_ingress_is_a_lower_limit(test_conn) -> None:
    # egress is at tc + t14/2 = tc + 0.05 d; a 0.03 d gap centred there hides it
    (t14_h, t14_err, lower, reason, ingress, ingress_err, _c), _ = _run_gap_case(
        test_conn, "gap_egress", 64, 0.05
    )
    assert lower is True
    assert "gap covers egress" in reason
    assert t14_h < 2.35  # only the part up to the last epoch before the gap is seen
    assert t14_err is None
    assert (ingress, ingress_err) == (None, None)  # one edge not observed

    (_t14, _e, lower_i, reason_i, ingress_i, _ie, _c2), _ = _run_gap_case(
        test_conn, "gap_ingress", 65, -0.05
    )
    assert lower_i is True
    assert "gap covers ingress" in reason_i
    assert ingress_i is None


def test_analyze_clears_a_stale_duration_lower_limit(test_conn) -> None:
    obj_id = _insert_object(test_conn, "stale")
    night_id = _insert_night(test_conn, "20250101")
    det_id = _insert_transit(
        test_conn, obj_id, night_id, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0006, seed=66,
    )
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET duration_lower_limit = true WHERE det_id = %s",
            (det_id,),
        )
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT duration_lower_limit FROM relphot.detection WHERE det_id = %s", (det_id,)
        )
        assert cur.fetchone() == (False,)  # complete and unflagged: the old true is not kept
        cur.execute("SELECT duration_lower_limit FROM relphot.object WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (False,)

    with test_conn.cursor() as cur:  # a search flag keeps it true whatever the fit says
        cur.execute("UPDATE relphot.detection SET flags = 'PARTIAL' WHERE det_id = %s", (det_id,))
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT duration_lower_limit FROM relphot.detection WHERE det_id = %s", (det_id,)
        )
        assert cur.fetchone() == (True,)


def _shape(det_id: int, t14: float, *, lower: bool = False, ingress: float | None = 0.2) -> dict:
    return {
        "det_id": det_id, "converged": True, "tc": 2460000.0 + det_id, "depth": 0.02,
        "depth_err": 0.001, "t14_h": t14, "t14_err": None if lower else 0.1,
        "t14_lower_limit": lower, "ingress_frac": ingress,
        "ingress_err": None if ingress is None else 0.05,
    }


def test_match_duration_term_is_one_sided_for_a_lower_limit() -> None:
    from relphot.db.analyze import _match_pairs

    settings = DbSettings()
    tele = {1: "T80S", 2: "T80S"}

    def match(a: dict, b: dict) -> dict:
        (row,) = _match_pairs([a, b], tele, settings)
        return row

    # measured T >= limit: no penalty, the (zero) term still counts
    ok = match(_shape(1, 2.0, lower=True, ingress=None), _shape(2, 2.5))
    assert ok["t14_z"] == 0.0
    assert ok["ingress_z"] is None  # ingress term dropped: the lower limit has none
    assert ok["dof"] == 2

    # measured T below the limit: penalised, (L - T) / err_T
    bad = match(_shape(1, 3.0, lower=True, ingress=None), _shape(2, 2.5))
    assert bad["t14_z"] == pytest.approx(0.5 / 0.1)
    assert bad["p_match"] < ok["p_match"]

    # the same, with the lower limit on the other side of the pair
    assert match(_shape(1, 2.5), _shape(2, 3.0, lower=True, ingress=None))["t14_z"] == (
        pytest.approx(5.0)
    )
    assert match(_shape(1, 3.5), _shape(2, 3.0, lower=True, ingress=None))["t14_z"] == 0.0

    # two lower limits: no duration term at all (and no ingress term): depth only
    both = match(
        _shape(1, 2.0, lower=True, ingress=None), _shape(2, 9.0, lower=True, ingress=None)
    )
    assert both["t14_z"] is None
    assert both["dof"] == 1

    # two measured durations keep the signed two-sided z
    two_sided = match(_shape(1, 2.0), _shape(2, 2.5))
    assert two_sided["t14_z"] == pytest.approx(-0.5 / np.hypot(0.1, 0.1))
    assert two_sided["dof"] == 3


def test_transit_shape_unusable_night_is_stored_unconverged(test_conn) -> None:
    obj_id = _insert_object(test_conn, "shape_fail")
    night_id = _insert_night(test_conn, "20250101")
    t = 2460000.0 + np.linspace(0.0, 0.4, 40)
    _insert_star_night_and_lc(
        test_conn, obj_id, night_id, 0, t, np.ones_like(t), np.full_like(t, 0.001)
    )
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier) VALUES (%s, %s, 'transit', 9.0, 0.01, %s, 2.0, 1) "
            "RETURNING det_id",
            (obj_id, night_id, 2460000.0 + 5.0),  # transit time far outside the light curve
        )
        (det_id,) = cur.fetchone()
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT converged, depth_err, tc_err, n_points FROM relphot.transit_shape "
            "WHERE det_id = %s",
            (det_id,),
        )
        assert cur.fetchone() == (False, None, None, 0)


def test_transit_match_high_for_identical_events_low_for_depth_ratio_two(test_conn) -> None:
    same = _insert_object(test_conn, "match_same")
    diff = _insert_object(test_conn, "match_diff")
    dets = {}
    for obj_id, name, depth_b, seed in ((same, "same", 0.02, 21), (diff, "diff", 0.04, 31)):
        n1 = _insert_night(test_conn, f"{name}_n1")
        n2 = _insert_night(test_conn, f"{name}_n2")
        a = _insert_transit(
            test_conn, obj_id, n1, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
            noise=0.0004, seed=seed,
        )
        b = _insert_transit(
            test_conn, obj_id, n2, 1, t_lo=3.0, tc=3.2, t14_h=2.4, q=0.2, depth=depth_b,
            noise=0.0004, seed=seed + 1,
        )
        dets[obj_id] = (a, b)
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[same, diff], settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report.n_transit_matches == 2

    def match(obj_id: int) -> tuple:
        with test_conn.cursor() as cur:
            cur.execute(
                "SELECT det_a, det_b, dt_days, depth_z, p_match, dof, same_telescope, "
                "commensurate_periods FROM relphot.transit_match WHERE obj_id = %s",
                (obj_id,),
            )
            return cur.fetchone()

    det_a, det_b, dt_days, _depth_z, p_match, dof, same_telescope, periods = match(same)
    assert (det_a, det_b) == dets[same]
    assert det_a < det_b
    assert dt_days == pytest.approx(3.0, abs=0.01)
    assert p_match > 0.05
    assert dof == 3
    assert same_telescope is True
    assert periods[0] == pytest.approx(dt_days)
    assert periods[1] == pytest.approx(dt_days / 2)
    assert min(periods) >= 0.2
    assert len(periods) <= 50

    *_ids, _dt, depth_z_diff, p_diff, _dof, _same, _periods = match(diff)
    assert p_diff < 1e-6
    assert abs(depth_z_diff) > 5

    # events are never merged and no status is touched
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*), count(*) FILTER (WHERE status = 'UNCONFIRMED') "
                    "FROM relphot.detection WHERE kind = 'transit'")
        assert cur.fetchone() == (4, 4)


def test_transit_match_cross_telescope_adds_depth_floor(test_conn) -> None:
    obj_id = _insert_object(test_conn, "match_cross")
    n1 = _insert_night(test_conn, "cross_n1")  # T80S
    n2 = _insert_night(test_conn, "cross_n2")
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.night SET telescope = 'ROBO43' WHERE night_id = %s", (n2,))
    _insert_transit(
        test_conn, obj_id, n1, 0, t_lo=0.0, tc=0.2, t14_h=2.4, q=0.2, depth=0.02,
        noise=0.0002, seed=41,
    )
    _insert_transit(
        test_conn, obj_id, n2, 1, t_lo=3.0, tc=3.2, t14_h=2.4, q=0.2, depth=0.0215,
        noise=0.0002, seed=42,
    )
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT same_telescope, p_match FROM relphot.transit_match WHERE obj_id = %s",
            (obj_id,),
        )
        same_telescope, p_cross = cur.fetchone()
    assert same_telescope is False

    # the same pair with both nights on one telescope has a tighter floor -> lower p
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.night SET telescope = 'T80S'")
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT same_telescope, p_match FROM relphot.transit_match WHERE obj_id = %s",
            (obj_id,),
        )
        same_telescope, p_same = cur.fetchone()
    assert same_telescope is True
    assert p_cross > p_same


def test_reanalyze_replaces_shapes_and_matches(test_conn) -> None:
    obj_id = _insert_object(test_conn, "rerun")
    for i, seed in enumerate((51, 52)):
        night_id = _insert_night(test_conn, f"rerun_n{i}")
        _insert_transit(
            test_conn, obj_id, night_id, i, t_lo=3.0 * i, tc=0.2 + 3.0 * i, t14_h=2.4, q=0.2,
            depth=0.02, noise=0.0005, seed=seed,
        )
    test_conn.commit()
    for _ in range(2):
        analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
        test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.transit_shape WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (2,)
        cur.execute("SELECT count(*) FROM relphot.transit_match WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (1,)


# --------------------------------------------------------------------------
# D3: period estimate / verification
# --------------------------------------------------------------------------


def _insert_sinusoid_nights(
    conn: psycopg.Connection, obj_id: int, period: float, n_nights: int, *, seed: int = 7,
    amplitude: float = 0.05, first_label: int = 1,
) -> list[int]:
    rng = np.random.default_rng(seed)
    night_ids = []
    for i in range(n_nights):
        label = f"2025010{first_label + i}"
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
                "VALUES ('T80S', %s, %s, %s, now()) RETURNING night_id",
                (f"2025-01-0{first_label + i}", label, f"/tmp/{label}"),
            )
            (night_id,) = cur.fetchone()
        night_ids.append(night_id)
        _insert_detection(conn, obj_id, night_id, "variable")
        t = np.sort(rng.uniform(i * 1.0, i * 1.0 + 0.4, 150))
        flux = 1.0 + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.001, t.size)
        _insert_star_night_and_lc(
            conn, obj_id, night_id, i, 2460000.0 + t, flux, np.full_like(t, 0.001)
        )
    return night_ids


def _estimates(conn: psycopg.Connection, obj_id: int) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT night_ids, n_nights, last_night, baseline_days, period, period_err, fap, "
            "input, lit_period, harmonic, delta, delta_err, delta_z "
            "FROM relphot.period_estimate WHERE obj_id = %s ORDER BY n_nights",
            (obj_id,),
        )
        return cur.fetchall()


def _verify_status(conn: psycopg.Connection, obj_id: int) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT verify_status, verify_note FROM relphot.period_estimate "
            "WHERE obj_id = %s ORDER BY n_nights",
            (obj_id,),
        )
        return cur.fetchall()


def test_period_estimate_recovers_injected_period_with_error(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_sinusoid")
    night_ids = _insert_sinusoid_nights(test_conn, obj_id, 0.3, 3)
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report.n_period_estimates == 1

    ((ids, n_nights, last_night, baseline, period, period_err, fap, input_label, lit, harmonic,
      delta, _delta_err, _delta_z),) = _estimates(test_conn, obj_id)
    assert ids == sorted(night_ids)
    assert n_nights == 3
    assert str(last_night) == "2025-01-03"
    assert baseline == pytest.approx(2.4, abs=0.1)
    assert period == pytest.approx(0.3, rel=2e-3)
    assert 0 < period_err < 1e-3
    assert abs(period - 0.3) < 5 * period_err
    assert fap <= _SETTINGS.db.ls_fap_threshold
    assert input_label == "night"
    assert (lit, harmonic, delta) == (None, None, None)
    assert _verify_status(test_conn, obj_id) == [("no_literature", None)]

    # the object's PERIOD comes from the estimate, with its error and night count
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source, period_err, period_n_nights FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        obj_period, source, obj_err, obj_n = cur.fetchone()
    assert (obj_period, source, obj_err, obj_n) == (period, "LS", period_err, 3)


def test_period_estimate_is_stored_for_a_single_night(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_single")
    _insert_sinusoid_nights(test_conn, obj_id, 0.12, 1)
    test_conn.commit()

    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    ((_ids, n_nights, *_rest),) = _estimates(test_conn, obj_id)
    assert n_nights == 1


def test_period_estimate_history_one_row_per_set_of_nights(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_history")
    _insert_sinusoid_nights(test_conn, obj_id, 0.3, 2)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)  # same night set
    test_conn.commit()
    assert len(_estimates(test_conn, obj_id)) == 1

    # a new night: a new row; the old one is kept
    rng = np.random.default_rng(99)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-09', '20250109', '/tmp/20250109', now()) "
            "RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
    t = np.sort(rng.uniform(5.0, 5.4, 150))
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / 0.3) + rng.normal(0, 0.001, t.size)
    _insert_star_night_and_lc(
        test_conn, obj_id, night_id, 9, 2460000.0 + t, flux, np.full_like(t, 0.001)
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    rows = _estimates(test_conn, obj_id)
    assert [r[1] for r in rows] == [2, 3]
    assert str(rows[1][2]) == "2025-01-09"
    assert rows[1][5] < rows[0][5]  # a longer baseline gives a smaller period error


def test_period_verification_against_literature_period(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_lit")
    _insert_sinusoid_nights(test_conn, obj_id, 0.300, 3)
    _insert_catalog_match(
        test_conn, obj_id, "VSX", "V* Lit", var_type="ROT", period=0.3003, period_err=0.0005
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    ((_ids, _n, _last, _base, period, period_err, _fap, _inp, lit, harmonic, delta, delta_err,
      delta_z),) = _estimates(test_conn, obj_id)
    assert lit == pytest.approx(0.3003)
    assert harmonic == 1.0
    assert delta == pytest.approx(period - 0.3003)
    assert delta_err == pytest.approx(np.hypot(period_err, 0.0005))
    assert delta_z == pytest.approx(delta / delta_err, rel=1e-4)
    assert abs(delta_z) < 4
    assert _verify_status(test_conn, obj_id) == [("verified", None)]

    # the literature period wins as the object's PERIOD, with the catalogue's error
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source, period_err FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (pytest.approx(0.3003), "catalog", pytest.approx(0.0005))


def test_period_verification_picks_half_harmonic_for_eb_like_signal(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_eb")
    # two equal minima per orbit: the light curve varies at P_lit / 2
    _insert_sinusoid_nights(test_conn, obj_id, 0.300, 3)
    _insert_catalog_match(
        test_conn, obj_id, "VSX", "V* EB", var_type="EA", period=0.600, period_err=0.001
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    ((_ids, _n, _last, _base, period, _err, _fap, _inp, lit, harmonic, delta, _delta_err,
      _delta_z),) = _estimates(test_conn, obj_id)
    assert lit == pytest.approx(0.600)
    assert harmonic == 0.5
    assert period == pytest.approx(0.300, rel=2e-3)
    assert abs(delta) < 0.005  # period / 0.5 - P_lit ~ 0


def test_period_estimate_skipped_without_variable_or_literature_period(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_none")
    night_id = _insert_night(test_conn, "20250101")
    _insert_detection(test_conn, obj_id, night_id, "transit")
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0.0, 0.4, 100))
    _insert_star_night_and_lc(
        test_conn, obj_id, night_id, 0, 2460000.0 + t, 1.0 + rng.normal(0, 0.001, t.size),
        np.full_like(t, 0.001),
    )
    test_conn.commit()
    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    assert report.n_period_estimates == 0
    assert _estimates(test_conn, obj_id) == []


def test_literature_period_outside_the_ls_grid_still_gets_a_row_per_observation(
    test_conn,
) -> None:
    obj_id = _insert_object(test_conn, "pe_outside")
    _insert_sinusoid_nights(test_conn, obj_id, 0.3, 2)
    _insert_catalog_match(
        test_conn, obj_id, "VSX", "V* Long", var_type="M", period=217.0, period_err=1.0
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    ((_ids, n_nights, _last, _base, period, _err, _fap, _inp, lit, harmonic, delta, _de, _dz),
     ) = _estimates(test_conn, obj_id)
    assert n_nights == 2
    assert lit == pytest.approx(217.0)
    assert (harmonic, delta) == (None, None)  # not verifiable ...
    assert period == pytest.approx(0.3, rel=5e-3)  # ... but the data's own period is kept
    ((status, note),) = _verify_status(test_conn, obj_id)
    assert status == "lit_period_outside_grid"
    assert note.startswith("P_lit 217 d > LS max period ")

    # a new night is a new re-observation: a second row, again with its reason
    rng = np.random.default_rng(3)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-09', '20250109', '/tmp/20250109', now()) RETURNING night_id"
        )
        (night_id,) = cur.fetchone()
    t = np.sort(rng.uniform(5.0, 5.4, 150))
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / 0.3) + rng.normal(0, 0.001, t.size)
    _insert_star_night_and_lc(
        test_conn, obj_id, night_id, 9, 2460000.0 + t, flux, np.full_like(t, 0.001)
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    statuses = _verify_status(test_conn, obj_id)
    assert [s for s, _ in statuses] == ["lit_period_outside_grid"] * 2


def test_no_peak_in_the_literature_window(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_nopeak")
    _insert_sinusoid_nights(test_conn, obj_id, 0.3, 3)
    _insert_catalog_match(test_conn, obj_id, "VSX", "V* Off", var_type="ROT", period=0.5)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    ((_ids, _n, _last, _base, period, _err, _fap, _inp, lit, harmonic, delta, _de, _dz),
     ) = _estimates(test_conn, obj_id)
    assert lit == pytest.approx(0.5)
    assert (harmonic, delta) == (None, None)
    assert period == pytest.approx(0.3, rel=5e-3)
    ((status, note),) = _verify_status(test_conn, obj_id)
    assert status == "no_peak_in_window"
    assert "P_lit 0.5 d" in note


def test_literature_period_with_too_little_data_is_still_recorded(test_conn) -> None:
    obj_id = _insert_object(test_conn, "pe_short")
    night_id = _insert_night(test_conn, "20250101")
    t = 2460000.0 + np.linspace(0.0, 0.05, 5)  # 5 epochs: below the 10-point minimum
    _insert_star_night_and_lc(
        test_conn, obj_id, night_id, 0, t, np.ones_like(t), np.full_like(t, 0.01)
    )
    _insert_catalog_match(test_conn, obj_id, "VSX", "V* Short", var_type="EA", period=1.2)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    ((_ids, n_nights, _last, baseline, period, _err, fap, _inp, lit, harmonic, *_rest),
     ) = _estimates(test_conn, obj_id)
    assert (n_nights, period, fap, harmonic) == (1, None, None, None)
    assert lit == pytest.approx(1.2)
    assert baseline == pytest.approx(0.05)
    assert _verify_status(test_conn, obj_id)[0][0] == "insufficient_data"


# --------------------------------------------------------------------------
# long periods (> 1 d, >= 2 nights): tie-calibrated magnitudes, no per-night offsets
# --------------------------------------------------------------------------

_LONG_P = 2.5
#: three nights about a day apart, half a night long: baseline 2.8 d = 1.1 cycles of 2.5 d
_LONG_NIGHT_STARTS = (0.0, 1.1, 2.3)


def _long_period_series(seed: int = 3, amp: float = 0.15, noise: float = 0.003):
    """Per night ``(t, mag)`` of a 2.5 d sinusoid, and the night's median (the tie's mean mag)."""
    rng = np.random.default_rng(seed)
    out = []
    for t0 in _LONG_NIGHT_STARTS:
        t = 2460000.0 + np.sort(t0 + rng.uniform(0.0, 0.5, 120))
        mag = 15.0 + amp * np.sin(2 * np.pi * (t - 2460000.0) / _LONG_P) + rng.normal(0, noise, 120)
        out.append((t, mag, float(np.median(mag))))
    return out


def _insert_long_period_object(
    conn: psycopg.Connection, name: str, *, tied: bool, seed: int = 3
) -> int:
    obj_id = _insert_object(conn, name)
    mn_run = _insert_mn_run(conn, f"mn_{name}") if tied else None
    for i, (t, mag, mag_median) in enumerate(_long_period_series(seed)):
        night_id = _insert_night(conn, f"{name}_{i}")
        _insert_detection(conn, obj_id, night_id, "variable")
        flux = 10.0 ** (-0.4 * (mag - mag_median))
        _insert_star_night_and_lc(conn, obj_id, night_id, i, t, flux, np.full_like(t, 0.003) * flux)
        if tied:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO relphot.tie (mn_run_id, obj_id, night_id, mag, mag_err) "
                    "VALUES (%s, %s, %s, %s, 0.005)",
                    (mn_run, obj_id, night_id, mag_median),
                )
    return obj_id


def _estimate_row(conn: psycopg.Connection, obj_id: int, method: str = "LS") -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT input, n_nights, period, period_err, verify_status, verify_note, harmonic, "
            "delta, phase_coverage, n_cycles, alias_periods, alias_powers, baseline_days "
            "FROM relphot.period_estimate WHERE obj_id = %s AND method = %s",
            (obj_id, method),
        )
        cols = [d.name for d in cur.description]
        return dict(zip(cols, cur.fetchone(), strict=True))


def test_long_period_is_recovered_on_tied_nights_with_aliases_and_coverage(test_conn) -> None:
    obj_id = _insert_long_period_object(test_conn, "long_tied", tied=True)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    est = _estimate_row(test_conn, obj_id)
    assert est["input"] == "tied" and est["n_nights"] == 3
    assert est["period"] == pytest.approx(_LONG_P, abs=0.03)
    assert 0 < est["period_err"] < 0.02
    assert abs(est["period"] - _LONG_P) < 5 * est["period_err"]
    assert est["verify_status"] == "no_literature"
    # the baseline holds 1.1 cycles: the note says the period is only bounded from below
    assert est["n_cycles"] == pytest.approx(est["baseline_days"] / est["period"], rel=1e-3)
    assert 1.0 < est["n_cycles"] < 2.0
    assert "baseline covers" in est["verify_note"] and "< 2" in est["verify_note"]
    assert 0.3 < est["phase_coverage"] <= 1.0
    # the one-day aliases 1/(f + 1 d^-1) and 1/|f - 1 d^-1| of f = 1/2.5 are listed
    f0 = 1.0 / est["period"]
    aliases = est["alias_periods"]
    assert 1 <= len(aliases) <= _SETTINGS.db.max_alias_candidates
    assert any(a == pytest.approx(1.0 / (f0 + 1.0), rel=0.03) for a in aliases)
    assert any(a == pytest.approx(1.0 / abs(f0 - 1.0), rel=0.03) for a in aliases)
    assert len(est["alias_powers"]) == len(aliases)
    assert est["alias_powers"] == sorted(est["alias_powers"], reverse=True)


def test_long_period_is_verified_against_the_literature_on_tied_nights(test_conn) -> None:
    obj_id = _insert_long_period_object(test_conn, "long_lit", tied=True)
    _insert_catalog_match(
        test_conn, obj_id, "VSX", "V* Slow", var_type="ROT", period=_LONG_P, period_err=0.01
    )
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    est = _estimate_row(test_conn, obj_id)
    assert est["verify_status"] == "verified" and est["harmonic"] == 1.0
    assert abs(est["delta"]) < 0.03


def test_long_period_on_untied_nights_is_flagged_not_measured(test_conn) -> None:
    """Per-night normalisation removes the night-to-night changes a 2.5 d period lives in."""
    obj_id = _insert_long_period_object(test_conn, "long_untied", tied=False)
    _insert_catalog_match(test_conn, obj_id, "VSX", "V* Slow", var_type="ROT", period=_LONG_P)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()

    est = _estimate_row(test_conn, obj_id)
    assert est["input"] == "night"
    assert est["verify_status"] == "long_period_needs_tie"
    assert "long period needs a multi-night tie" in est["verify_note"]
    assert est["period_err"] is None
    assert (est["harmonic"], est["delta"]) == (None, None)
    # such a row never becomes the object's PERIOD_err / n_nights source
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source, period_n_nights FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (_LONG_P, "catalog", None)


def test_untied_long_period_without_literature_does_not_set_period_from_the_estimate(
    test_conn,
) -> None:
    obj_id = _insert_long_period_object(test_conn, "long_untied_nolit", tied=False)
    test_conn.commit()
    analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    est = _estimate_row(test_conn, obj_id)
    assert est["verify_status"] == "long_period_needs_tie" and est["period_err"] is None
    with test_conn.cursor() as cur:
        cur.execute("SELECT period_n_nights FROM relphot.object WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (None,)


def _long_period_task(seed: int = 3) -> _ObjTask:
    nights, tie = [], {}
    for i, (t, mag, mag_median) in enumerate(_long_period_series(seed)):
        flux = 10.0 ** (-0.4 * (mag - mag_median))
        nights.append(
            _NightData(i + 1, date(2025, 1, 1 + i), t, flux, np.full_like(t, 0.003) * flux)
        )
        tie[i + 1] = (mag_median, 0.005)
    return _ObjTask(
        obj_id=1, is_exop=False, bls_eligible=False, nights=nights, tie=tie, settings=DbSettings(),
        is_var=True, night_ties=tie,
    )


def test_a_free_offset_per_night_absorbs_the_long_period_signal() -> None:
    """The offsets-free fit measures the period ~9x better than one offset per night would."""
    task = _long_period_task()
    t, y, dy, label, span, night_idx = _tied_series(task, task.night_ties)
    assert label == "tied-mag"
    f0 = 1.0 / 2.53
    free = _refine_fourier(t, y, dy, night_idx, f0, span, per_night_offsets=False)
    offsets = _refine_fourier(t, y, dy, night_idx, f0, span, per_night_offsets=True)
    assert free is not None and offsets is not None
    assert 1.0 / free[0] == pytest.approx(_LONG_P, abs=0.02)
    err_free, err_off = free[1] / free[0] ** 2, offsets[1] / offsets[0] ** 2
    assert err_off > 4.0 * err_free


def test_phase_stats_coverage_and_cycles() -> None:
    t = 2460000.0 + np.linspace(0.0, 0.5, 200)
    coverage, cycles = _phase_stats(t, 2.5, 20)
    assert cycles == pytest.approx(0.2)
    assert coverage == pytest.approx(0.2, abs=0.05)  # 0.2 of the cycle: 4 of 20 bins (+/- 1)
    coverage_short, cycles_short = _phase_stats(t, 0.1, 20)
    assert coverage_short == 1.0 and cycles_short == pytest.approx(5.0)


def test_a_long_period_peak_on_the_grid_edge_is_not_verified() -> None:
    """A maximum on the first grid point is the grid's edge, not a resolved peak."""
    freq = np.linspace(0.4, 2.0, 100)
    power = np.linspace(1.0, 0.0, 100)  # falls from the very first point
    edge = _verification_windows(freq, power, 2.4, 0.05, long_period_days=1.0)
    assert edge[0] is None and edge[1] == "no_peak_in_window"
    # a short-period window keeps the old rule: the same first-point maximum counts
    short = _verification_windows(freq, power, 2.4, 0.05)
    assert short[0] is not None and short[1] == "verified"
    interior = power.copy()
    interior[5] = 2.0
    ok = _verification_windows(freq, interior, 2.4, 0.15, long_period_days=1.0)
    assert ok[0] is not None and ok[0][1] == 5
