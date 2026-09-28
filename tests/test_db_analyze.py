"""Tests for relphot.db.analyze against a live PostgreSQL test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for
how it is resolved). Skipped with an explicit reason when no such DSN is
available. Fixtures insert night/object/star_night/lightcurve/detection rows
directly with plain SQL (not through :func:`relphot.db.load_night.load_night`)
so each test controls its light curve's timestamps and signal precisely.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import psycopg
import pytest

from relphot.config import DbSettings, Settings
from relphot.db.analyze import analyze
from relphot.db.connect import resolve_dsn
from relphot.db.refresh import refresh_objects
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError

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
