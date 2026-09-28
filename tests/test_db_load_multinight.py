"""Tests for relphot.db.load_multinight against a live PostgreSQL test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for
how it is resolved). Skipped with an explicit reason when no such DSN is
available. Builds two synthetic nights with the same fixtures
tests/test_db_load_night.py uses (``_write_night1``/``_write_night2``,
imported rather than duplicated) and loads them with
:func:`relphot.db.load_night.load_night`; the multi-night run itself is a
hand-written ``STEM.npz`` (+ search-metrics parquet) using the exact key
names :func:`relphot.multinight.save_multinight`/``multinight_search``
write, with every key :func:`relphot.multinight.load_multinight` reads
present -- most filled with a placeholder shape, since
:func:`relphot.db.load_multinight.load_multinight` only actually reads
``labels``, ``xmatch.index``, ``tie.anchor_index``,
``mlc.night_mean_mag``/``mlc.night_mean_err``, ``night_info``, and
``settings``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest
from test_db_load_night import _SETTINGS, _write_night1, _write_night2

from relphot.config import Settings, settings_to_dict
from relphot.db.analyze import analyze
from relphot.db.connect import resolve_dsn
from relphot.db.load_multinight import load_multinight
from relphot.db.load_night import load_night
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError, NightLoadError


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


def _write_synthetic_multinight_npz(
    path: Path,
    *,
    labels: list[str],
    anchor_index: int,
    night_info: list[dict],
    xmatch_index: np.ndarray,
    night_mean_mag: np.ndarray,
    night_mean_err: np.ndarray,
) -> None:
    """A ``STEM.npz`` with every key :func:`relphot.multinight.load_multinight` reads.

    Only ``labels_json``, ``xmatch_index``, ``tie_anchor_index``,
    ``mlc_night_mean_mag``/``mlc_night_mean_err``, ``night_info_json``, and
    ``settings_json`` carry values this test cares about (what
    :func:`relphot.db.load_multinight.load_multinight` actually reads);
    every other key is a placeholder of a shape consistent with its own
    field, never read by the code under test.
    """
    n_nights, n_global = xmatch_index.shape
    n_aper = 1
    n_bins = 1

    np.savez(
        path,
        labels_json=json.dumps(labels),
        xmatch_ra=np.zeros(n_global),
        xmatch_dec=np.zeros(n_global),
        xmatch_index=xmatch_index,
        xmatch_n_matched=np.zeros(n_nights, dtype=np.int64),
        tie_anchor_index=np.int64(anchor_index),
        tie_coef=np.zeros((n_nights, n_aper, 1)),
        tie_basis_terms_json=json.dumps(["const"]),
        tie_xi=np.zeros(n_global),
        tie_eta=np.zeros(n_global),
        tie_centre_ra=np.float64(0.0),
        tie_centre_dec=np.float64(0.0),
        tie_scale_deg=np.float64(1.0),
        tie_mag0=np.zeros(n_aper),
        tie_zp=np.zeros((n_nights, n_global, n_aper)),
        tie_mean_mag=np.zeros(n_global),
        tie_night_mag=np.zeros((n_nights, n_global, n_aper)),
        tie_night_mag_err=np.zeros((n_nights, n_global, n_aper)),
        tie_tie_star=np.zeros((n_nights, n_global, n_aper), dtype=bool),
        tie_rejected=np.zeros((n_global, n_aper), dtype=bool),
        tie_floor_mag_centres=np.zeros((n_aper, n_bins)),
        tie_floor=np.zeros((n_nights, n_aper, n_bins)),
        tie_n_tie=np.zeros((n_nights, n_aper), dtype=np.int64),
        tie_resid_mad=np.zeros((n_nights, n_aper)),
        tie_resid_mad_bright=np.zeros((n_nights, n_aper)),
        tie_chi2_after=np.zeros((n_nights, n_aper)),
        tie_chi2_holdout=np.zeros((n_nights, n_aper)),
        tie_chi2_holdout_bins=np.zeros((n_nights, n_aper, n_bins)),
        tie_n_iter=np.zeros(n_aper, dtype=np.int64),
        tie_seeing_basis_terms_json=json.dumps([]),
        tie_seeing_coef=np.zeros((n_aper, 0)),
        tie_seeing_mag0=np.zeros(n_aper),
        tie_seeing_crowd0=np.zeros(n_aper),
        tie_night_fwhm=np.zeros(n_nights),
        tie_crowding=np.zeros(n_global),
        mlc_night_of_frame=np.zeros(0, dtype=np.int64),
        mlc_frame_in_night=np.zeros(0, dtype=np.int64),
        mlc_bjd_tdb=np.zeros(0),
        mlc_airmass=np.zeros(0),
        mlc_fwhm=np.zeros(0),
        mlc_aperture=np.zeros(n_global, dtype=np.int64),
        mlc_mag=np.zeros((n_global, 0)),
        mlc_mag_err=np.zeros((n_global, 0)),
        mlc_flux_norm=np.zeros((n_global, 0)),
        mlc_flux_norm_err=np.zeros((n_global, 0)),
        mlc_night_mean_mag=night_mean_mag,
        mlc_night_mean_err=night_mean_err,
        mlc_mean_mag=np.zeros(n_global),
        mlc_n_nights=np.zeros(n_global, dtype=np.int64),
        night_info_json=json.dumps(night_info),
        settings_json=json.dumps(settings_to_dict(Settings())),
    )


def _write_synthetic_search_metrics(path: Path, n_global: int) -> None:
    """``multinight_search_metrics.parquet`` with the real column names.

    global 0: internight candidate (maps to obj_clean).
    global 1: periodic *and* recurrent candidate (maps to obj_var).
    global 2: BLS candidate (maps to the conflict winner, objC).
    global 3: internight candidate but unmapped (no star_night anywhere)
        -- must be silently skipped, not raise or get counted.
    global 4: no candidate flags (single-night object, obj_exop).
    """
    nan = np.nan
    sm = pd.DataFrame({
        "global_id": np.arange(n_global, dtype=np.int64),
        "internight_candidate": np.array([True, False, False, True, False]),
        "internight_chi2": np.array([12.5, nan, nan, 9.0, nan]),
        "internight_p": np.array([0.001, nan, nan, 0.02, nan]),
        "internight_amplitude": np.array([0.05, nan, nan, 0.03, nan]),
        "periodic_candidate": np.array([False, True, False, False, False]),
        "ls_period_days": np.array([nan, 1.234, nan, nan, nan]),
        "ls_fap": np.array([nan, 0.0005, nan, nan, nan]),
        "ls_power": np.array([nan, 0.42, nan, nan, nan]),
        "ls_second_period_days": np.array([nan, 2.468, nan, nan, nan]),
        "bls_candidate": np.array([False, False, True, False, False]),
        "bls_period_days": np.array([nan, nan, 3.3, nan, nan]),
        "bls_t0": np.array([nan, nan, 2460000.6, nan, nan]),
        "bls_depth": np.array([nan, nan, 0.015, nan, nan]),
        "bls_depth_snr": np.array([nan, nan, 9.2, nan, nan]),
        "bls_duration_days": np.array([nan, nan, 0.08, nan, nan]),
        "bls_nights_in_transit": np.array([0, 0, 2, 0, 0], dtype=np.int64),
        "bls_flags": np.array(["", "", "", "", ""], dtype=object),
        "period_compat_frac_excluded": np.array([nan, nan, 0.1, nan, nan]),
        "period_compat_frac_supported": np.array([nan, nan, 0.8, nan, nan]),
        "period_compat_frac_unconstrained": np.array([nan, nan, 0.1, nan, nan]),
        "period_compat_allowed_intervals": np.array(["", "", "0.2-0.4", "", ""], dtype=object),
        "period_compat_best_period_days": np.array([nan, nan, 3.31, nan, nan]),
        "period_compat_best_snr": np.array([nan, nan, 9.0, nan, nan]),
        "recurrent_variable": np.array([False, True, False, False, False]),
        "n_nights_var_candidate": np.array([0, 2, 0, 0, 0], dtype=np.int64),
    })
    sm.to_parquet(path)


def _write_synthetic_stars_parquet(
    path: Path, labels: list[str], xmatch_index: np.ndarray,
    night_mean_mag: np.ndarray, night_mean_err: np.ndarray,
) -> None:
    """``STEM_stars.parquet`` with the real column names (not read by the loader)."""
    _n_nights, n_global = xmatch_index.shape
    columns: dict = {
        "global_id": np.arange(n_global, dtype=np.int64),
        "ra": np.zeros(n_global),
        "dec": np.zeros(n_global),
        "aperture": np.zeros(n_global, dtype=np.int64),
        "mean_mag": np.zeros(n_global),
        "n_nights": np.zeros(n_global, dtype=np.int64),
    }
    for n, label in enumerate(labels):
        columns[f"mag_{label}"] = night_mean_mag[n]
        columns[f"err_{label}"] = night_mean_err[n]
        columns[f"star_id_{label}"] = xmatch_index[n]
    pd.DataFrame(columns).to_parquet(path)


def _build_scenario(test_conn, tmp_path):
    """Two loaded nights, a matching hand-written multi-night run, and its object ids.

    Global stars (see :func:`_write_synthetic_search_metrics` for the
    matching detections):

    - g0: night1 star0 <-> night2 star1 -- a clean 2-night tie (obj_clean).
    - g1: night1 star1 <-> night2 star2 -- a clean 2-night tie, the known
      variable (obj_var).
    - g2: night1 star2 <-> night2 star3 -- these resolve to *different*
      objects (a manufactured cross-match disagreement); the anchor night
      (night1) wins, so the winner is objC.
    - g3: night1 star5 (dropped by the noise cut, never stored) <-> nothing
      -- unmapped.
    - g4: night1 star4 only (the forced transit candidate, obj_exop) --
      single-night.
    """
    root1 = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root1)
    report1 = load_night(test_conn, root1, settings=_SETTINGS)

    root2 = tmp_path / "T80S_reduced" / "20250102" / "relphot"
    _write_night2(root2)
    report2 = load_night(test_conn, root2, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT star_id, obj_id FROM relphot.star_night WHERE night_id = %s",
            (report1.night_id,),
        )
        night1_map = dict(cur.fetchall())
        cur.execute(
            "SELECT star_id, obj_id FROM relphot.star_night WHERE night_id = %s",
            (report2.night_id,),
        )
        night2_map = dict(cur.fetchall())

    assert night1_map[0] == night2_map[1]  # obj_clean: confirmed by test_db_load_night
    assert night1_map[1] == night2_map[2]  # obj_var
    assert night1_map[2] != night2_map[3]  # objC vs objB: a genuine disagreement
    assert 5 not in night1_map  # star5 failed the noise cut -- never stored

    obj_clean = night1_map[0]
    obj_var = night1_map[1]
    obj_c = night1_map[2]
    obj_exop = night1_map[4]

    labels = [report1.label, report2.label]
    night_info = [
        {
            "label": report1.label, "directory": str(root1.resolve()), "filter": "R",
            "object": "TESTFIELD", "n_frames": 5, "n_kept": 4,
        },
        {
            "label": report2.label, "directory": str(root2.resolve()), "filter": "R",
            "object": "TESTFIELD", "n_frames": 4, "n_kept": 4,
        },
    ]

    n_global = 5
    xmatch_index = np.full((2, n_global), -1, dtype=np.int64)
    xmatch_index[0, 0], xmatch_index[1, 0] = 0, 1  # g0: clean tie
    xmatch_index[0, 1], xmatch_index[1, 1] = 1, 2  # g1: clean tie (VAR)
    xmatch_index[0, 2], xmatch_index[1, 2] = 2, 3  # g2: conflict
    xmatch_index[0, 3] = 5  # g3: dropped star -> unmapped
    xmatch_index[0, 4] = 4  # g4: single-night (EXOP)

    night_mean_mag = np.full((2, n_global), np.nan)
    night_mean_err = np.full((2, n_global), np.nan)
    night_mean_mag[0, 0], night_mean_err[0, 0] = 15.10, 0.010
    night_mean_mag[1, 0], night_mean_err[1, 0] = 15.12, 0.011
    night_mean_mag[0, 1], night_mean_err[0, 1] = 15.20, 0.010
    night_mean_mag[1, 1], night_mean_err[1, 1] = 15.19, 0.012
    night_mean_mag[0, 2], night_mean_err[0, 2] = 15.30, 0.010
    night_mean_mag[1, 2], night_mean_err[1, 2] = 15.35, 0.020
    night_mean_mag[0, 4], night_mean_err[0, 4] = 15.40, 0.015

    mn_dir = tmp_path / "multinight"
    mn_dir.mkdir()
    stem = mn_dir / "mn_test"
    _write_synthetic_multinight_npz(
        stem.with_suffix(".npz"), labels=labels, anchor_index=0, night_info=night_info,
        xmatch_index=xmatch_index, night_mean_mag=night_mean_mag, night_mean_err=night_mean_err,
    )
    _write_synthetic_stars_parquet(
        mn_dir / "mn_test_stars.parquet", labels, xmatch_index, night_mean_mag, night_mean_err,
    )

    search_dir = tmp_path / "search"
    search_dir.mkdir()
    _write_synthetic_search_metrics(search_dir / "multinight_search_metrics.parquet", n_global)

    return {
        "stem": stem, "search_dir": search_dir,
        "obj_clean": obj_clean, "obj_var": obj_var, "obj_c": obj_c, "obj_exop": obj_exop,
        "report1": report1, "report2": report2, "root1": root1, "root2": root2,
    }


def test_load_multinight_tie_rows_and_conflicts(test_conn, tmp_path) -> None:
    scenario = _build_scenario(test_conn, tmp_path)
    report = load_multinight(test_conn, scenario["stem"], search_dir=scenario["search_dir"])

    assert report.labels == ["20250101", "20250102"]
    assert report.anchor == "20250101"
    assert report.n_tie_rows == 6  # g0(2) + g1(2) + g2(1, anchor only) + g4(1)
    assert report.n_objects_mapped == 4  # obj_clean, obj_var, obj_c, obj_exop
    assert report.n_globals_unmapped == 1  # g3
    assert report.n_conflicts == 1  # g2

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, night_id, mag, mag_err FROM relphot.tie "
            "WHERE mn_run_id = %s ORDER BY obj_id, night_id",
            (report.mn_run_id,),
        )
        tie_rows = cur.fetchall()
    assert len(tie_rows) == 6

    tie_by_obj: dict[int, list[tuple]] = {}
    for obj_id, night_id, mag, mag_err in tie_rows:
        tie_by_obj.setdefault(obj_id, []).append((night_id, mag, mag_err))

    assert len(tie_by_obj[scenario["obj_clean"]]) == 2
    assert len(tie_by_obj[scenario["obj_var"]]) == 2
    assert len(tie_by_obj[scenario["obj_exop"]]) == 1
    # g2's conflict: only the anchor night (report1) made it in, with objC's mag
    assert tie_by_obj[scenario["obj_c"]] == [
        (scenario["report1"].night_id, pytest.approx(15.30), pytest.approx(0.010))
    ]


def test_load_multinight_detections(test_conn, tmp_path) -> None:
    scenario = _build_scenario(test_conn, tmp_path)
    report = load_multinight(test_conn, scenario["stem"], search_dir=scenario["search_dir"])

    assert report.n_detections_internight == 1  # g0 only -- g3 is unmapped, silently skipped
    assert report.n_detections_ls_periodic == 1
    assert report.n_detections_bls == 1
    assert report.n_detections_recurrent == 1

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, kind, snr, depth, tc_bjd_tdb, duration_h, flags, period, fap, "
            "amplitude, extra FROM relphot.detection WHERE mn_run_id = %s ORDER BY kind",
            (report.mn_run_id,),
        )
        rows = cur.fetchall()
    by_kind = {r[1]: r for r in rows}
    assert set(by_kind) == {"internight", "ls_periodic", "bls", "recurrent"}

    internight = by_kind["internight"]
    assert internight[0] == scenario["obj_clean"]
    assert internight[9] == pytest.approx(0.05)  # amplitude
    assert internight[10] == {"chi2": pytest.approx(12.5), "p": pytest.approx(0.001)}

    ls_periodic = by_kind["ls_periodic"]
    assert ls_periodic[0] == scenario["obj_var"]
    assert ls_periodic[7] == pytest.approx(1.234)  # period
    assert ls_periodic[8] == pytest.approx(0.0005)  # fap
    assert ls_periodic[10] == {
        "power": pytest.approx(0.42), "second_period_days": pytest.approx(2.468),
    }

    bls = by_kind["bls"]
    assert bls[0] == scenario["obj_c"]
    assert bls[2] == pytest.approx(9.2)  # snr = bls_depth_snr
    assert bls[3] == pytest.approx(0.015)  # depth
    assert bls[4] == pytest.approx(2460000.6)  # tc_bjd_tdb = bls_t0
    assert bls[5] == pytest.approx(0.08 * 24.0)  # duration_h = 24 * bls_duration_days
    assert bls[10]["nights_in_transit"] == 2
    assert bls[10]["period_compat_best_period_days"] == pytest.approx(3.31)

    recurrent = by_kind["recurrent"]
    assert recurrent[0] == scenario["obj_var"]
    assert recurrent[10] == {"n_nights_var_candidate": 2}


def test_load_multinight_missing_night_raises(test_conn, tmp_path) -> None:
    scenario = _build_scenario(test_conn, tmp_path)

    mn_dir = tmp_path / "multinight_missing"
    mn_dir.mkdir()
    stem = mn_dir / "mn_missing"
    labels = [scenario["report1"].label, "20259999"]
    night_info = [
        {
            "label": labels[0], "directory": str(scenario["root1"].resolve()),
            "filter": "R", "object": "TESTFIELD", "n_frames": 5, "n_kept": 4,
        },
        {
            "label": labels[1], "directory": str(tmp_path / "nowhere" / "relphot"),
            "filter": "R", "object": "TESTFIELD", "n_frames": 1, "n_kept": 1,
        },
    ]
    xmatch_index = np.full((2, 1), -1, dtype=np.int64)
    night_mean_mag = np.full((2, 1), np.nan)
    night_mean_err = np.full((2, 1), np.nan)
    _write_synthetic_multinight_npz(
        stem.with_suffix(".npz"), labels=labels, anchor_index=0, night_info=night_info,
        xmatch_index=xmatch_index, night_mean_mag=night_mean_mag, night_mean_err=night_mean_err,
    )

    with pytest.raises(NightLoadError, match="20259999"):
        load_multinight(test_conn, stem)


def test_load_multinight_reload_is_idempotent(test_conn, tmp_path) -> None:
    scenario = _build_scenario(test_conn, tmp_path)
    report1 = load_multinight(test_conn, scenario["stem"], search_dir=scenario["search_dir"])
    report2 = load_multinight(test_conn, scenario["stem"], search_dir=scenario["search_dir"])

    assert report2.mn_run_id == report1.mn_run_id
    assert report2.n_tie_rows == report1.n_tie_rows
    assert report2.n_conflicts == report1.n_conflicts

    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.tie WHERE mn_run_id = %s", (report1.mn_run_id,))
        (n_tie,) = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM relphot.detection WHERE mn_run_id = %s", (report1.mn_run_id,)
        )
        (n_det,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM relphot.mn_run")
        (n_runs,) = cur.fetchone()
    assert n_tie == report1.n_tie_rows
    assert n_det == 4
    assert n_runs == 1


def test_combined_analyze_uses_tied_mag(test_conn, tmp_path) -> None:
    scenario = _build_scenario(test_conn, tmp_path)
    load_multinight(test_conn, scenario["stem"], search_dir=scenario["search_dir"])
    obj_clean = scenario["obj_clean"]
    report1, report2 = scenario["report1"], scenario["report2"]

    # _write_night1/_write_night2 give each night only 4 kept-frame points
    # (8 combined) -- below analyze._MIN_NIGHT_POINTS (10), so no periodogram
    # would be computed at all without more points. Pad obj_clean's own two
    # relphot.lightcurve rows (already written by load_night) to 6 points
    # each, purely to clear that floor; the values themselves need only be
    # finite and positive.
    with test_conn.cursor() as cur:
        for report, t0 in ((report1, 2460000.5), (report2, 2460001.5)):
            bjd = [t0 + 0.001 * i for i in range(6)]
            flux = [10000.0 + 5.0 * (i % 2) for i in range(6)]
            err = [50.0] * 6
            cur.execute(
                "UPDATE relphot.lightcurve SET frame_index = %s, bjd_tdb = %s, flux = %s, "
                "flux_err = %s, flux_raw = %s WHERE obj_id = %s AND night_id = %s",
                (list(range(6)), bjd, flux, err, flux, obj_clean, report.night_id),
            )
        test_conn.commit()

    analyze(test_conn, obj_ids=[obj_clean], settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT input FROM relphot.periodogram "
            "WHERE obj_id = %s AND scope = 'combined' AND method = 'LS'",
            (obj_clean,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "tied-mag"
