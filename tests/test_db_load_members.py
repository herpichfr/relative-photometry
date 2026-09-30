"""Tests for relphot.db.load_members against a live PostgreSQL test database.

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
from relphot.db.load_members import load_members
from relphot.db.load_night import load_night
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError, NightLoadError
from relphot.members import MembersProduct, save_members_npz

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
    """5 frames, 3 apertures, 1 tile. Used for members tests."""
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
        c: np.zeros(n_stars, dtype=bool)
        for c in [
            "transit_searched",
            "transit_partial",
            "transit_candidate",
            "variability_searched",
            "variability_systematic_excluded",
            "variability_candidate",
            "known_variable",
            "known_planet",
            "known_planet_is_toi",
        ]
    }
    sm.update(
        {
            "star_id": star_id, "tile": np.zeros(n_stars, dtype=int),
            "ra": df_ss["ra"].to_numpy(), "dec": df_ss["dec"].to_numpy(),
            "mag": df_ss["mag"].to_numpy(), "best_aperture": np.ones(n_stars, dtype=int),
            "rms": df_ss["rms"].to_numpy(), "n_epochs": df_ss["n_epochs"].to_numpy(),
            "transit_snr": np.ones(n_stars),
            "transit_depth": np.array([np.nan] * 4 + [0.02, np.nan]),
            "transit_tc_bjd_tdb": np.array([np.nan] * 4 + [2460000.55, np.nan]),
            "transit_duration_hours": np.array([np.nan] * 4 + [1.5, np.nan]),
            "transit_n_in": np.zeros(n_stars, dtype=int),
            "transit_beta": np.ones(n_stars),
            "transit_coverage": np.ones(n_stars),
            "transit_tier": np.array([0, 0, 0, 0, 1, 0], dtype=int),
            "transit_frame_error_scale_at_tc": np.ones(n_stars),
            "transit_dchi2_box_vs_flat": np.zeros(n_stars),
            "transit_dchi2_box_vs_step": np.zeros(n_stars),
            "transit_coincidence_count": np.zeros(n_stars, dtype=int),
            "transit_flags": np.zeros(n_stars, dtype=int),
            "transit_flags_str": np.array([""] * 4 + ["OK"] + [""], dtype=object),
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
            "variability_class": np.array([""] * n_stars, dtype=object),
            "known_variable_name": np.array([""] * n_stars, dtype=object),
            "known_variable_type": np.array([""] * n_stars, dtype=object),
            "known_variable_period_days": np.full(n_stars, np.nan),
            "known_planet_name": np.array([""] * n_stars, dtype=object),
            "known_planet_period_days": np.full(n_stars, np.nan),
            "known_planet_depth": np.full(n_stars, np.nan),
            "gaia_id": np.array([""] * n_stars, dtype=object),
            "neighbour_sep_arcsec": np.full(n_stars, np.nan),
            "max_dilutable_depth": np.full(n_stars, np.nan),
        }
    )
    for a in range(n_aper):
        sm[f"transit_depth_aper{a}"] = np.full(n_stars, np.nan)
        sm[f"transit_sigma_depth_aper{a}"] = np.full(n_stars, np.nan)
    pd.DataFrame(sm).to_parquet(lc_dir / "night_lc_search_metrics.parquet")


def _write_members_npz(root: Path) -> None:
    """Write a members.npz file for night1.

    Create a MembersProduct with:
    - 1 tile (tile 0)
    - 3 apertures
    - 5 frames (one dropped)
    - 3 reference members (one stored, one failed noise cut, one in core)
    - Comparison members for the used aperture
    """
    lc_dir = root / "lc"
    n_frames = 5
    n_aper = 3
    n_tiles = 1

    # Reference members: 3 members for tile 0
    # Member 0: stored (obj_id available)
    # Member 1: not stored, no nearby object (obj_id NULL via position)
    # Member 2: not stored but within 1" of an object (obj_id linked by position)
    ref_offsets = np.array([0, 3], dtype=np.int64)  # 3 ref members for tile 0
    ref_star = np.array([10, 11, 12], dtype=np.int64)
    ref_ra = np.array([10.0, 10.001, 10.002], dtype=np.float64)
    ref_dec = np.array([-20.0, -20.001, -20.002], dtype=np.float64)
    ref_mag = np.array([15.0, 15.1, 15.2], dtype=np.float32)
    ref_weight = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    ref_in_core = np.array([True, True, False])

    # Comparison members: 4 members for the used pair (tile=0, aperture=1)
    # Member 0: stored
    # Member 1: failed noise cut
    # Member 2: in core, stored
    # Member 3: not stored, no nearby object
    used_pairs = np.array([[0, 1]], dtype=np.int32)  # one used pair
    comp_offsets = np.array([0, 4], dtype=np.int64)  # 4 comp members
    comp_star = np.array([1, 2, 3, 4], dtype=np.int64)
    comp_ra = np.array([10.0005, 10.0015, 10.0025, 10.0035], dtype=np.float64)
    comp_dec = np.array([-20.0005, -20.0015, -20.0025, -20.0035], dtype=np.float64)
    comp_mag = np.array([16.0, 16.1, 16.2, 16.3], dtype=np.float32)
    comp_weight = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    comp_n_clipped = np.array([0, 1, 0, 0], dtype=np.int16)
    comp_clip_offsets = np.array([0, 0, 1, 1, 1], dtype=np.int64)  # member 1 has 1 clipped frame
    comp_clip_frames = np.array([2], dtype=np.int16)  # frame 2
    comp_norm_flux = np.random.randn(4, n_frames).astype(np.float32)

    # Tile curves
    tile_R = np.random.randn(n_tiles, n_frames, n_aper).astype(np.float64) + 10.0
    tile_sigma_R = np.abs(np.random.randn(n_tiles, n_frames, n_aper).astype(np.float64)) + 0.01
    tile_ens = np.random.randn(n_tiles, n_frames, n_aper).astype(np.float64) + 10.0
    tile_sigma_ens = np.abs(np.random.randn(n_tiles, n_frames, n_aper).astype(np.float64)) + 0.01

    tile_n_ensemble = np.array([[4, 0, 0]], dtype=np.int64)  # only aperture 1 used
    tile_n_rounds = np.array([[1, 1, 1]], dtype=np.int16)

    # Mark frame 4 as dropped by setting to NaN in arrays
    tile_R[0, 4, :] = np.nan
    tile_sigma_R[0, 4, :] = np.nan
    tile_ens[0, 4, :] = np.nan
    tile_sigma_ens[0, 4, :] = np.nan
    comp_norm_flux[:, 4] = np.nan

    product = MembersProduct(
        n_frames=n_frames,
        n_aper=n_aper,
        ref_aper=1,
        meta={"version": 1, "ref_method": "weighted_fixed", "comp_method": "median"},
        tile_xmin=np.array([0.0], dtype=np.float64),
        tile_xmax=np.array([512.0], dtype=np.float64),
        tile_ymin=np.array([0.0], dtype=np.float64),
        tile_ymax=np.array([512.0], dtype=np.float64),
        tile_n_core=np.array([3], dtype=np.int64),
        tile_n_extended=np.array([1], dtype=np.int64),
        ref_offsets=ref_offsets,
        ref_star=ref_star,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
        ref_mag=ref_mag,
        ref_weight=ref_weight,
        ref_in_core=ref_in_core,
        tile_R=tile_R,
        tile_sigma_R=tile_sigma_R,
        tile_ens=tile_ens,
        tile_sigma_ens=tile_sigma_ens,
        tile_n_ensemble=tile_n_ensemble,
        tile_n_rounds=tile_n_rounds,
        used_pairs=used_pairs,
        comp_offsets=comp_offsets,
        comp_star=comp_star,
        comp_ra=comp_ra,
        comp_dec=comp_dec,
        comp_mag=comp_mag,
        comp_weight=comp_weight,
        comp_n_clipped=comp_n_clipped,
        comp_clip_offsets=comp_clip_offsets,
        comp_clip_frames=comp_clip_frames,
        comp_norm_flux=comp_norm_flux,
    )

    save_members_npz(product, lc_dir / "night_lc_members.npz")


def test_load_members_row_counts(test_conn, tmp_path) -> None:
    """Test that insert_members creates the correct number of rows."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    # Load night first
    night_report = load_night(test_conn, root, settings=_SETTINGS)
    night_id = night_report.night_id

    # Load members
    members_report = load_members(test_conn, root)

    # Check basic counts
    assert members_report.night_id == night_id
    assert members_report.n_tiles == 1
    assert members_report.n_tile_lc == 3  # 1 tile * 3 apertures
    assert members_report.n_reference_members == 3
    assert members_report.n_comparison_members == 4

    # Check database rows
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM relphot.tile_lc WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone()[0] == 3  # 1 tile * 3 apertures

        cur.execute(
            "SELECT COUNT(*) FROM relphot.reference_member WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone()[0] == 3

        cur.execute(
            "SELECT COUNT(*) FROM relphot.comparison_member WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone()[0] == 4


def test_load_members_tile_lc_all_apertures(test_conn, tmp_path) -> None:
    """Test that tile_lc has entries for all (tile, aperture) pairs, including unused ones."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    night_report = load_night(test_conn, root, settings=_SETTINGS)
    night_id = night_report.night_id
    load_members(test_conn, root)

    # Check that all (tile, aperture) pairs exist
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT tile, aperture, n_comp FROM relphot.tile_lc "
            "WHERE night_id = %s ORDER BY tile, aperture",
            (night_id,),
        )
        rows = cur.fetchall()

    # Should have 3 rows (1 tile * 3 apertures)
    assert len(rows) == 3
    assert rows[0] == (0, 0, 0)  # unused aperture
    assert rows[1] == (0, 1, 4)  # used aperture with 4 members
    assert rows[2] == (0, 2, 0)  # unused aperture


def test_load_members_nan_preserved(test_conn, tmp_path) -> None:
    """Test that NaN values in dropped frames are preserved in arrays."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    night_report = load_night(test_conn, root, settings=_SETTINGS)
    night_id = night_report.night_id
    load_members(test_conn, root)

    # Check that NaN is preserved in ref_flux for the dropped frame
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT ref_flux FROM relphot.tile_lc "
            "WHERE night_id = %s AND tile = 0 AND aperture = 1",
            (night_id,),
        )
        ref_flux = cur.fetchone()[0]

    # Frame 4 (index 4) should be NaN
    assert len(ref_flux) == 5
    assert np.isnan(ref_flux[4])
    assert not np.isnan(ref_flux[0])


def test_load_members_comparison_members(test_conn, tmp_path) -> None:
    """Test that comparison members have correct weights and clipped frames."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    night_report = load_night(test_conn, root, settings=_SETTINGS)
    night_id = night_report.night_id
    load_members(test_conn, root)

    # Check comparison members
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT star_id, weight, n_clipped, clipped_frames FROM relphot.comparison_member "
            "WHERE night_id = %s ORDER BY star_id",
            (night_id,),
        )
        rows = cur.fetchall()

    assert len(rows) == 4
    # Member 0 (star_id 1): no clipped frames
    assert rows[0][2] == 0  # n_clipped
    assert rows[0][3] == []  # clipped_frames
    # Member 1 (star_id 2): 1 clipped frame
    assert rows[1][2] == 1
    assert rows[1][3] == [2]  # frame 2


def test_load_members_object_linking(test_conn, tmp_path) -> None:
    """Test that members are linked to objects correctly."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    load_night(test_conn, root, settings=_SETTINGS)

    # Load members
    members_report = load_members(test_conn, root)

    # Check n_linked counts member rows with obj_id not null
    # We should have some linked (stored members)
    assert members_report.n_linked > 0
    assert members_report.n_unlinked > 0


def test_load_members_reload_replaces(test_conn, tmp_path) -> None:
    """Test that reloading members replaces existing data without duplicates."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    night_report = load_night(test_conn, root, settings=_SETTINGS)
    night_id = night_report.night_id

    # First load
    report1 = load_members(test_conn, root)

    # Second load (reload)
    report2 = load_members(test_conn, root)

    # Counts should be identical
    assert report1.n_tile_lc == report2.n_tile_lc
    assert report1.n_reference_members == report2.n_reference_members
    assert report1.n_comparison_members == report2.n_comparison_members

    # Check no duplicates in database
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM relphot.tile_lc WHERE night_id = %s",
            (night_id,),
        )
        assert cur.fetchone()[0] == 3


def test_load_members_no_file(test_conn, tmp_path) -> None:
    """Test that loading a night without a members file raises NightLoadError."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    # Don't write members file

    load_night(test_conn, root, settings=_SETTINGS)

    # Try to load members without the file
    with pytest.raises(NightLoadError, match="no such file"):
        load_members(test_conn, root)


def test_load_members_unknown_night(test_conn, tmp_path) -> None:
    """Test that loading members for an unknown night raises NightLoadError."""
    # Create night structure but don't load it into DB
    root = tmp_path / "T80S_reduced" / "20250102" / "relphot"
    _write_night1(root)
    _write_members_npz(root)
    # Don't call load_night

    with pytest.raises(NightLoadError, match="night not loaded"):
        load_members(test_conn, root)


def test_load_members_n_frames_mismatch(test_conn, tmp_path) -> None:
    """Test that a mismatch in n_frames raises NightLoadError."""
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _write_members_npz(root)

    load_night(test_conn, root, settings=_SETTINGS)

    # Modify members file to have wrong n_frames
    lc_dir = root / "lc"
    with np.load(lc_dir / "night_lc_members.npz", allow_pickle=False) as data:
        arrays = dict(data.items())

    # Save with wrong n_frames
    arrays["n_frames"] = np.array(10)  # Wrong!
    np.savez(lc_dir / "night_lc_members.npz", **arrays)

    with pytest.raises(NightLoadError, match="n_frames"):
        load_members(test_conn, root)
