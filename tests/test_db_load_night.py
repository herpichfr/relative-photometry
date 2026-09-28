"""Tests for relphot.db.load_night against a live PostgreSQL test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for
how it is resolved). Skipped with an explicit reason when no such DSN is
available; otherwise these tests run for real against a throwaway
``relphot`` schema, dropped and rebuilt by the ``test_conn`` fixture.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

from relphot.config import DbSettings, SearchSettings, Settings
from relphot.db.connect import resolve_dsn
from relphot.db.load_night import _format_dec_dms, _format_ra_hms, load_night
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError, NightLoadError

_SETTINGS = replace(
    Settings(),
    search=replace(SearchSettings(), min_epochs=3, min_epoch_fraction=0.0),
    db=replace(
        DbSettings(), max_expected_noise=0.05, keep_candidates=True, match_radius_arcsec=1.0
    ),
)


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


def _write_night1(root: Path) -> None:
    """6 stars, 5 frames (1 dropped). star4: transit candidate failing the noise
    cut (must be forced in). star5: noisy, not a candidate (must not be stored).
    star1: a known variable (Gaia DR3-format name).
    """
    lc_dir = root / "lc"
    lc_dir.mkdir(parents=True)

    n_frames = 5
    frame_meta = [
        {
            "file": f"/data/T80S_reduced/20250101/frame{i:03d}_proc.fits",
            "date_obs": f"2025-01-02T0{i}:00:00.000000",
            "exptime": 90.0,
            "jd_utc": 2460000.5 + i * 0.01,
            "bjd_tdb": 2460000.5006 + i * 0.01,
            "airmass": 1.2 + 0.01 * i,
            "filter": "R",
            "object": "TESTFIELD",
            "median_fwhm": 2.5,
            "n_sources": 100,
            "aperture_radii_px": [2.0, 3.0, 4.0],
        }
        for i in range(n_frames)
    ]
    config = {"site": {"latitude_deg": -30.2, "longitude_deg": -70.8, "elevation_m": 2200.0}}
    np.savez(
        root / "night.npz",
        frame_meta_json=json.dumps(frame_meta),
        config_json=json.dumps(config),
    )
    np.savez(root / "ref.npz", frame_kept=np.array([True, True, True, True, False]))

    n_stars = 6
    star_id = np.arange(n_stars)
    df_ss = pd.DataFrame(
        {
            "star_id": star_id,
            "tile": np.zeros(n_stars, dtype=int),
            "ra": 10.0 + 0.001 * star_id,
            "dec": -20.0 + 0.001 * star_id,
            "mag": 15.0 + 0.1 * star_id,
            "best_aperture": np.ones(n_stars, dtype=int),
            "rms": np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.05]),
            "chi2_reduced": np.ones(n_stars),
            "expected_noise": np.array([0.01, 0.01, 0.01, 0.01, 0.2, 0.2]),
            "n_epochs": np.full(n_stars, 4),
            "is_comparison": np.array([True, True, False, False, False, False]),
        }
    )
    df_ss.to_parquet(lc_dir / "night_lc_starstats.parquet")

    rows = [
        {
            "star_id": s, "tile": 0, "frame": f, "bjd_tdb": frame_meta[f]["bjd_tdb"],
            "airmass": frame_meta[f]["airmass"], "aperture": 1,
            "lc": 10000.0 + s * 10 + f, "lc_err": 50.0, "lc_raw": 10010.0 + s * 10 + f,
        }
        for s in star_id
        for f in range(4)
    ]
    pd.DataFrame(rows).to_parquet(lc_dir / "night_lc_lightcurves.parquet")

    n_aper = 3
    sm = {
        "star_id": star_id, "tile": np.zeros(n_stars, dtype=int),
        "ra": df_ss["ra"].to_numpy(), "dec": df_ss["dec"].to_numpy(),
        "mag": df_ss["mag"].to_numpy(), "best_aperture": np.ones(n_stars, dtype=int),
        "rms": df_ss["rms"].to_numpy(), "n_epochs": df_ss["n_epochs"].to_numpy(),
        "transit_searched": np.ones(n_stars, dtype=bool),
        "transit_snr": np.array([1.0, 1.0, 1.0, 1.0, 8.5, 1.0]),
        "transit_depth": np.array([np.nan] * 4 + [0.02, np.nan]),
        "transit_tc_bjd_tdb": np.array([np.nan] * 4 + [2460000.55, np.nan]),
        "transit_duration_hours": np.array([np.nan] * 4 + [1.5, np.nan]),
        "transit_n_in": np.zeros(n_stars, dtype=int),
        "transit_beta": np.ones(n_stars),
        "transit_coverage": np.ones(n_stars),
        "transit_partial": np.zeros(n_stars, dtype=bool),
        "transit_tier": np.array([0, 0, 0, 0, 1, 0], dtype=int),
        "transit_frame_error_scale_at_tc": np.ones(n_stars),
        "transit_dchi2_box_vs_flat": np.zeros(n_stars),
        "transit_dchi2_box_vs_step": np.zeros(n_stars),
        "transit_coincidence_count": np.zeros(n_stars, dtype=int),
        "transit_flags": np.zeros(n_stars, dtype=int),
        "transit_flags_str": np.array([""] * 4 + ["OK"] + [""], dtype=object),
        "transit_candidate": np.array([False, False, False, False, True, False]),
        "variability_searched": np.ones(n_stars, dtype=bool),
        "variability_rms_robust": np.full(n_stars, 0.01),
        "variability_rms_std": np.full(n_stars, 0.01),
        "variability_excess": np.zeros(n_stars),
        "variability_von_neumann": np.ones(n_stars),
        "variability_von_neumann_significance": np.zeros(n_stars),
        "variability_ls_period_days": np.full(n_stars, np.nan),
        "variability_ls_power": np.zeros(n_stars),
        "variability_ls_fap": np.full(n_stars, np.nan),
        "variability_amplitude": np.full(n_stars, np.nan),
        "variability_trend_slope": np.zeros(n_stars),
        "variability_trend_significance": np.zeros(n_stars),
        "variability_systematic_excluded": np.zeros(n_stars, dtype=bool),
        "variability_candidate": np.zeros(n_stars, dtype=bool),
        "variability_class": np.array([""] * n_stars, dtype=object),
        "known_variable": np.array([False, True, False, False, False, False]),
        "known_variable_name": np.array(["", "Gaia DR3 12345", "", "", "", ""], dtype=object),
        "known_variable_type": np.array(["", "EA", "", "", "", ""], dtype=object),
        "known_variable_period_days": np.array([np.nan, 3.3, np.nan, np.nan, np.nan, np.nan]),
        "known_planet": np.zeros(n_stars, dtype=bool),
        "known_planet_name": np.array([""] * n_stars, dtype=object),
        "known_planet_period_days": np.full(n_stars, np.nan),
        "known_planet_depth": np.full(n_stars, np.nan),
        "known_planet_is_toi": np.zeros(n_stars, dtype=bool),
        "gaia_id": np.array([""] * n_stars, dtype=object),
        "neighbour_sep_arcsec": np.full(n_stars, np.nan),
        "max_dilutable_depth": np.full(n_stars, np.nan),
    }
    for a in range(n_aper):
        sm[f"transit_depth_aper{a}"] = np.full(n_stars, np.nan)
        sm[f"transit_sigma_depth_aper{a}"] = np.full(n_stars, np.nan)
    pd.DataFrame(sm).to_parquet(lc_dir / "night_lc_search_metrics.parquet")


def _empty_flags(n: int, *cols: str) -> dict:
    return {c: np.zeros(n, dtype=bool) for c in cols}


def _write_night2(root: Path) -> None:
    """4 stars matched against night 1's objects: star0/star1 both claim obj1
    (star1 nearer -> keeps it, star0 becomes new); star2 cleanly matches obj2;
    star3 is 2" away from obj3 -> a new object.
    """
    lc_dir = root / "lc"
    lc_dir.mkdir(parents=True)

    n_frames = 4
    frame_meta = [
        {
            "file": f"/data/T80S_reduced/20250102/frame{i:03d}_proc.fits",
            "date_obs": f"2025-01-03T0{i}:00:00.000000",
            "exptime": 90.0, "jd_utc": 2460001.5 + i * 0.01, "bjd_tdb": 2460001.5006 + i * 0.01,
            "airmass": 1.2, "filter": "R", "object": "TESTFIELD", "median_fwhm": 2.5,
            "n_sources": 100, "aperture_radii_px": [2.0, 3.0, 4.0],
        }
        for i in range(n_frames)
    ]
    np.savez(root / "night.npz", frame_meta_json=json.dumps(frame_meta), config_json=json.dumps({}))
    np.savez(root / "ref.npz", frame_kept=np.array([True] * n_frames))

    shift_03 = 0.3 / 3600.0
    shift_2 = 2.0 / 3600.0
    star_id = np.arange(4)
    ra = np.array([10.0 + shift_03, 10.0 + shift_03 * 0.5, 10.001 + shift_03, 10.002 + shift_2])
    dec = np.array([-20.0, -20.0, -19.999, -19.998])
    df_ss = pd.DataFrame(
        {
            "star_id": star_id, "tile": np.zeros(4, dtype=int), "ra": ra, "dec": dec,
            "mag": np.array([15.0, 15.05, 15.1, 15.2]), "best_aperture": np.ones(4, dtype=int),
            "rms": np.full(4, 0.01), "chi2_reduced": np.ones(4), "expected_noise": np.full(4, 0.01),
            "n_epochs": np.full(4, 4), "is_comparison": np.array([True, False, False, False]),
        }
    )
    df_ss.to_parquet(lc_dir / "night_lc_starstats.parquet")

    rows = [
        {
            "star_id": s, "tile": 0, "frame": f, "bjd_tdb": frame_meta[f]["bjd_tdb"],
            "airmass": 1.2, "aperture": 1, "lc": 20000.0 + s * 10 + f,
            "lc_err": 50.0, "lc_raw": 20010.0 + s * 10 + f,
        }
        for s in star_id
        for f in range(n_frames)
    ]
    pd.DataFrame(rows).to_parquet(lc_dir / "night_lc_lightcurves.parquet")

    n = 4
    sm = _empty_flags(
        n, "transit_searched", "transit_partial", "transit_candidate", "variability_searched",
        "variability_systematic_excluded", "variability_candidate", "known_variable",
        "known_planet", "known_planet_is_toi",
    )
    sm.update(
        {
            "star_id": star_id, "tile": np.zeros(4, dtype=int), "ra": ra, "dec": dec,
            "mag": df_ss["mag"].to_numpy(), "best_aperture": np.ones(n, dtype=int),
            "rms": np.full(n, 0.01), "n_epochs": np.full(n, 4),
            "transit_snr": np.ones(n), "transit_depth": np.full(n, np.nan),
            "transit_tc_bjd_tdb": np.full(n, np.nan), "transit_duration_hours": np.full(n, np.nan),
            "transit_n_in": np.zeros(n, dtype=int), "transit_beta": np.ones(n),
            "transit_coverage": np.ones(n), "transit_tier": np.zeros(n, dtype=int),
            "transit_frame_error_scale_at_tc": np.ones(n),
            "transit_dchi2_box_vs_flat": np.zeros(n), "transit_dchi2_box_vs_step": np.zeros(n),
            "transit_coincidence_count": np.zeros(n, dtype=int),
            "transit_flags": np.zeros(n, dtype=int),
            "transit_flags_str": np.array([""] * n, dtype=object),
            "variability_rms_robust": np.full(n, 0.01), "variability_rms_std": np.full(n, 0.01),
            "variability_excess": np.zeros(n), "variability_von_neumann": np.ones(n),
            "variability_von_neumann_significance": np.zeros(n),
            "variability_ls_period_days": np.full(n, np.nan), "variability_ls_power": np.zeros(n),
            "variability_ls_fap": np.full(n, np.nan), "variability_amplitude": np.full(n, np.nan),
            "variability_trend_slope": np.zeros(n), "variability_trend_significance": np.zeros(n),
            "variability_class": np.array([""] * n, dtype=object),
            "known_variable_name": np.array([""] * n, dtype=object),
            "known_variable_type": np.array([""] * n, dtype=object),
            "known_variable_period_days": np.full(n, np.nan),
            "known_planet_name": np.array([""] * n, dtype=object),
            "known_planet_period_days": np.full(n, np.nan),
            "known_planet_depth": np.full(n, np.nan),
            "gaia_id": np.array([""] * n, dtype=object),
            "neighbour_sep_arcsec": np.full(n, np.nan),
            "max_dilutable_depth": np.full(n, np.nan),
        }
    )
    for a in range(3):
        sm[f"transit_depth_aper{a}"] = np.full(n, np.nan)
        sm[f"transit_sigma_depth_aper{a}"] = np.full(n, np.nan)
    pd.DataFrame(sm).to_parquet(lc_dir / "night_lc_search_metrics.parquet")


def test_load_night_counts_and_noise_cut(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)

    report = load_night(test_conn, root, settings=_SETTINGS)

    assert report.telescope == "T80S"
    assert report.label == "20250101"
    assert report.n_frames == 5
    assert report.n_kept == 4
    assert report.n_stars == 6
    assert report.n_passed_cut == 4
    assert report.n_candidates_forced == 1
    assert report.n_stored == 5
    assert report.n_new_objects == 5
    assert report.n_transit_detections == 1
    assert report.n_variable_detections == 0
    assert report.n_catalog_matches == 1

    with test_conn.cursor() as cur:
        cur.execute("SELECT noise_cut FROM relphot.night WHERE night_id = %s", (report.night_id,))
        (noise_cut,) = cur.fetchone()
    assert noise_cut == {
        "max_expected_noise": 0.05, "min_epochs": 3, "keep_candidates": True,
        "n_stars": 6, "n_passed_cut": 4, "n_candidates_forced": 1, "n_stored": 5,
    }


def test_load_night_frames(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    report = load_night(test_conn, root, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT frame_index, file_name, kept FROM relphot.frame "
            "WHERE night_id = %s ORDER BY frame_index",
            (report.night_id,),
        )
        frames = cur.fetchall()
    assert len(frames) == 5
    assert [f[0] for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0][1] == "frame000_proc.fits"
    assert [f[2] for f in frames] == [True, True, True, True, False]


def test_load_night_lightcurve_matches_input(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    report = load_night(test_conn, root, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT sn.star_id, lc.frame_index, lc.flux "
            "FROM relphot.lightcurve lc JOIN relphot.star_night sn "
            "ON sn.obj_id = lc.obj_id AND sn.night_id = lc.night_id "
            "WHERE lc.night_id = %s ORDER BY sn.star_id",
            (report.night_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 5  # star5 was not stored -> no lightcurve row
    for star_id, frame_index, flux in rows:
        assert frame_index == [0, 1, 2, 3]
        expected_flux = [10000.0 + star_id * 10 + f for f in range(4)]
        assert flux == pytest.approx(expected_flux, abs=1e-2)


def test_load_night_classes_and_detections(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    load_night(test_conn, root, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT o.name, o.class, o.known FROM relphot.object o "
            "JOIN relphot.star_night sn ON sn.obj_id = o.obj_id "
            "ORDER BY sn.star_id"
        )
        rows = cur.fetchall()
    classes = [r[1] for r in rows]
    known = [r[2] for r in rows]
    assert classes == ["UNC", "VAR", "UNC", "UNC", "EXOP"]
    assert known == [False, True, False, False, False]

    with test_conn.cursor() as cur:
        cur.execute("SELECT kind, snr, depth, tier, flags FROM relphot.detection")
        detections = cur.fetchall()
    assert detections == [("transit", 8.5, 0.02, 1, "OK")]

    with test_conn.cursor() as cur:
        cur.execute("SELECT catalog, name, type, period FROM relphot.catalog_match")
        matches = cur.fetchall()
    assert matches == [("Gaia DR3", "Gaia DR3 12345", "EA", 3.3)]


def test_load_night_reload_is_idempotent(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    report1 = load_night(test_conn, root, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute("SELECT obj_id FROM relphot.object ORDER BY obj_id")
        obj_ids_before = [r[0] for r in cur.fetchall()]

    report2 = load_night(test_conn, root, settings=_SETTINGS)

    assert report2.night_id == report1.night_id
    assert report2.n_new_objects == 0
    assert report2.n_matched_objects == report1.n_new_objects

    with test_conn.cursor() as cur:
        cur.execute("SELECT obj_id FROM relphot.object ORDER BY obj_id")
        obj_ids_after = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT count(*) FROM relphot.star_night WHERE night_id = %s", (report1.night_id,)
        )
        (n_star_night,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM relphot.frame WHERE night_id = %s", (report1.night_id,))
        (n_frame,) = cur.fetchone()

    assert obj_ids_after == obj_ids_before
    assert n_star_night == report1.n_stored
    assert n_frame == report1.n_frames


def test_load_night_crossmatch_and_new_objects(test_conn, tmp_path) -> None:
    root1 = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root1)
    load_night(test_conn, root1, settings=_SETTINGS)

    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.object")
        (n_before,) = cur.fetchone()

    root2 = tmp_path / "T80S_reduced" / "20250102" / "relphot"
    _write_night2(root2)
    report2 = load_night(test_conn, root2, settings=_SETTINGS)

    assert report2.n_new_objects == 2  # star0 (lost the duplicate claim) and star3 (2" away)
    assert report2.n_matched_objects == 2  # star1 (won the duplicate claim) and star2

    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.object")
        (n_after,) = cur.fetchone()
        cur.execute(
            "SELECT sn.star_id, sn.obj_id FROM relphot.star_night sn "
            "WHERE sn.night_id = %s ORDER BY sn.star_id",
            (report2.night_id,),
        )
        mapping = dict(cur.fetchall())
        # obj1 is night 1's star0 (ra=10.0, dec=-20.0)
        cur.execute("SELECT obj_id FROM relphot.object WHERE ra = 10.0 AND dec = -20.0")
        (obj1,) = cur.fetchone()
        # obj2 is night 1's star1 (ra=10.001, dec=-19.999)
        cur.execute("SELECT obj_id FROM relphot.object WHERE ra = 10.001 AND dec = -19.999")
        (obj2,) = cur.fetchone()

    assert n_after == n_before + 2
    assert mapping[1] == obj1  # night2 star1 (nearer) won obj1's claim
    assert mapping[0] != obj1  # night2 star0 (farther) became a new object
    assert mapping[2] == obj2


def test_missing_search_metrics_warns_and_loads(test_conn, tmp_path, caplog) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    (root / "lc" / "night_lc_search_metrics.parquet").unlink()

    with caplog.at_level("WARNING"):
        report = load_night(test_conn, root, settings=_SETTINGS)

    assert "no *_search_metrics.parquet" in caplog.text
    assert report.n_transit_detections == 0
    assert report.n_catalog_matches == 0
    # without search metrics no star is a forced candidate, so the noisy
    # candidate (star4) is no longer stored -- only the 4 that pass the cut.
    assert report.n_stored == 4


def test_lc_stem_discovery_requires_exactly_one(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    # a second starstats file makes discovery ambiguous
    (root / "lc" / "other_starstats.parquet").write_bytes(
        (root / "lc" / "night_lc_starstats.parquet").read_bytes()
    )
    with pytest.raises(NightLoadError, match="multiple"):
        load_night(test_conn, root, settings=_SETTINGS)

    # explicit --lc-stem resolves the ambiguity
    report = load_night(test_conn, root, lc_stem="night_lc", settings=_SETTINGS)
    assert report.n_stored == 5


def test_telescope_required_when_not_inferrable(test_conn, tmp_path) -> None:
    root = tmp_path / "somewhere" / "20250101" / "relphot"
    _write_night1(root)
    with pytest.raises(NightLoadError, match="telescope"):
        load_night(test_conn, root, settings=_SETTINGS)
    report = load_night(test_conn, root, telescope="ROBO43", settings=_SETTINGS)
    assert report.telescope == "ROBO43"


def test_name_formatting_carries_59_999_seconds() -> None:
    # 01h02m59.999s -> rounds to 60.00s -> carries to 01h03m00.00s
    hours = 1 + 2 / 60 + 59.999 / 3600
    assert _format_ra_hms(hours * 15.0) == "010300.00"
    # 23h59m59.999s -> carries all the way through midnight
    hours = 23 + 59 / 60 + 59.999 / 3600
    assert _format_ra_hms(hours * 15.0) == "000000.00"
    # dec carry: 10d30m59.99s -> rounds to 60.0s -> carries to 10d31m00.0s
    deg = 10 + 30 / 60 + 59.99 / 3600
    assert _format_dec_dms(deg) == "+103100.0"
    assert _format_dec_dms(-deg) == "-103100.0"
