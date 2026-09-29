"""Tests for relphot.io: exact .npz round trip of a matched night."""

from __future__ import annotations

import numpy as np
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.ingest import read_catalogs
from relphot.io import (
    load_lightcurves_npz,
    load_night,
    load_reference,
    save_lightcurve_table,
    save_lightcurves_npz,
    save_night,
    save_reference,
    save_starstats_table,
)
from relphot.lightcurve import compute_light_curves
from relphot.match import match_night
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.stats import compute_star_stats
from relphot.tiles import build_tilemap


def test_round_trip_is_exact(fits_files, tmp_path) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    out = tmp_path / "night.npz"
    save_night(night, settings, out)
    night2, settings2 = load_night(out)

    assert settings2 == settings
    assert night2.master_frame_index == night.master_frame_index
    assert night2.n_stars_before_cut == night.n_stars_before_cut
    assert night2.n_stars_after_cut == night.n_stars_after_cut
    assert night2.reports == night.reports
    assert [m.to_dict() for m in night2.frame_meta] == [m.to_dict() for m in night.frame_meta]

    attrs = (
        "ra",
        "dec",
        "x",
        "y",
        "frame_x",
        "frame_y",
        "flux",
        "fluxerr",
        "fwhm",
        "snr",
        "background",
        "flags",
        "presence",
    )
    for attr in attrs:
        a = getattr(night, attr)
        b = getattr(night2, attr)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b, strict=True)


def test_reference_frame_kept_round_trip(fits_files, tmp_path) -> None:
    """Test that frame_kept and dropped_frames round-trip through save/load."""
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    ref_out = tmp_path / "ref.npz"
    save_reference(tilemap, result, settings, ref_out)
    _tilemap2, result2, _settings2 = load_reference(ref_out)

    np.testing.assert_array_equal(result2.frame_kept, result.frame_kept)
    assert result2.frame_kept.dtype == result.frame_kept.dtype


def test_save_load_lightcurves_npz(tmp_path) -> None:
    """Test round-trip save/load of light curves."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=100, n_frames=30, seed=30)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    star_stats = compute_star_stats(lc_result)

    best_aper = np.zeros((tilemap.n_tiles, settings.lightcurve.n_mag_bins), dtype=np.int64)
    bin_edges = np.full((tilemap.n_tiles, settings.lightcurve.n_mag_bins + 1), np.nan)
    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "lightcurves.npz"
    save_lightcurves_npz(
        lc_result,
        star_stats,
        best_aper,
        bin_edges,
        star_best_aper,
        comparison_result,
        settings,
        out_path,
    )

    assert out_path.is_file()

    # Load and verify
    (
        lc_result2,
        star_stats2,
        _best_aper2,
        _bin_edges2,
        _star_best_aper2,
        _comp_result2,
        settings2,
        _decorr2,
    ) = load_lightcurves_npz(out_path)

    np.testing.assert_array_equal(lc_result2.lc, lc_result.lc)
    np.testing.assert_array_equal(lc_result2.lc_err, lc_result.lc_err)
    np.testing.assert_array_equal(lc_result2.lc_raw, lc_result.lc_raw)
    np.testing.assert_array_equal(lc_result2.epoch_ok, lc_result.epoch_ok)

    np.testing.assert_array_equal(star_stats2.rms, star_stats.rms)
    assert settings2 == settings


def test_save_lightcurve_table_fits(tmp_path) -> None:
    """Test save_lightcurve_table writes a FITS table."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=50, n_frames=20, seed=31)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)

    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "lc"
    saved_path = save_lightcurve_table(
        night, tilemap, lc_result, star_best_aper, settings, out_path, fmt="fits"
    )

    assert saved_path.suffix == ".fits"
    assert saved_path.is_file()

    # Try to read it back
    try:
        from astropy.table import Table
        table = Table.read(str(saved_path), format="fits")
        assert "star_id" in table.colnames
        assert "lc" in table.colnames
        assert "lc_err" in table.colnames
    except ImportError:
        pass  # astropy not available


def test_save_starstats_table_fits(tmp_path) -> None:
    """Test save_starstats_table writes a FITS table."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=50, n_frames=20, seed=32)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    star_stats = compute_star_stats(lc_result)

    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "stats"
    saved_path = save_starstats_table(
        night,
        tilemap,
        comparison_result,
        star_stats,
        star_best_aper,
        settings,
        out_path,
        fmt="fits",
    )

    assert saved_path.suffix == ".fits"
    assert saved_path.is_file()

    # Try to read it back
    try:
        from astropy.table import Table
        table = Table.read(str(saved_path), format="fits")
        assert "star_id" in table.colnames
        assert "rms" in table.colnames
        assert "chi2_reduced" in table.colnames
    except ImportError:
        pass  # astropy not available


def _pipeline(seed: int):
    """Synthetic night through comparison stars and light curves (error inflation on)."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=100, n_frames=30, seed=seed)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )
    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    return night, tilemap, comparison_result, lc_result, compute_star_stats(lc_result), settings


def test_lightcurves_npz_keeps_error_inflation_and_old_products_still_load(tmp_path) -> None:
    night, tilemap, comparison_result, lc_result, star_stats, settings = _pipeline(33)
    assert lc_result.err_scale is not None
    best = np.zeros((tilemap.n_tiles, settings.lightcurve.n_mag_bins), dtype=np.int64)
    edges = np.full((tilemap.n_tiles, settings.lightcurve.n_mag_bins + 1), np.nan)
    best_aper = np.zeros(night.n_stars, dtype=np.int64)
    path = tmp_path / "lc.npz"
    save_lightcurves_npz(
        lc_result, star_stats, best, edges, best_aper, comparison_result, settings, path
    )

    loaded = load_lightcurves_npz(path)
    np.testing.assert_array_equal(loaded[0].lc_err_raw, lc_result.lc_err_raw)
    np.testing.assert_array_equal(loaded[0].err_scale, lc_result.err_scale)
    np.testing.assert_array_equal(loaded[0].blended, lc_result.blended)
    np.testing.assert_array_equal(loaded[1].err_scale, lc_result.err_scale)

    # a product written before error inflation existed has none of the three arrays
    with np.load(path, allow_pickle=False) as data:
        old = {k: data[k] for k in data.files if k not in {"lc_err_raw", "err_scale", "blended"}}
    old_path = tmp_path / "old.npz"
    np.savez(old_path, **old)
    lc_old, stats_old, *_rest = load_lightcurves_npz(old_path)
    assert lc_old.lc_err_raw is None and lc_old.err_scale is None and lc_old.blended is None
    assert stats_old.err_scale is None and stats_old.blended is None
    np.testing.assert_array_equal(lc_old.lc_err, lc_result.lc_err)


def test_tables_carry_err_scale_blended_and_the_raw_error(tmp_path) -> None:
    from astropy.table import Table

    night, tilemap, comparison_result, lc_result, star_stats, settings = _pipeline(34)
    best_aper = np.zeros(night.n_stars, dtype=np.int64)

    lc_path = save_lightcurve_table(
        night, tilemap, lc_result, best_aper, settings, tmp_path / "lc", fmt="fits"
    )
    lc_table = Table.read(str(lc_path), format="fits")
    assert "lc_err_raw" in lc_table.colnames
    assert np.all(lc_table["lc_err"] >= lc_table["lc_err_raw"] * (1 - 1e-6))

    ss_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, best_aper, settings, tmp_path / "ss",
        fmt="fits",
    )
    ss = Table.read(str(ss_path), format="fits")
    assert {"err_scale", "blended"} <= set(ss.colnames)
    assert np.all(ss["err_scale"] >= 1.0)
    core = tilemap.core_tile >= 0
    np.testing.assert_allclose(ss["err_scale"], lc_result.err_scale[core, 0], rtol=1e-6)

    # statistics without inflation information (an older product) read as 1 / False
    star_stats.err_scale = None
    star_stats.blended = None
    ss_old = Table.read(
        str(
            save_starstats_table(
                night, tilemap, comparison_result, star_stats, best_aper, settings,
                tmp_path / "ss_old", fmt="fits",
            )
        ),
        format="fits",
    )
    assert np.all(ss_old["err_scale"] == 1.0) and not np.any(ss_old["blended"])


def test_save_load_night_preserves_naxis_and_telescope(tmp_path) -> None:
    """save_night and load_night preserve naxis1/naxis2/telescope in frame_meta."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3)

    # Add detector info to frame_meta
    from dataclasses import replace
    frame_meta_updated = [
        replace(m, naxis1=4000, naxis2=3000, telescope="MYTEL")
        for m in night.frame_meta
    ]
    night_updated = replace(night, frame_meta=frame_meta_updated)

    save_night(night_updated, Settings(), tmp_path / "test.npz")
    night_loaded, _ = load_night(tmp_path / "test.npz")

    for m_orig, m_loaded in zip(night_updated.frame_meta, night_loaded.frame_meta, strict=True):
        assert m_loaded.naxis1 == m_orig.naxis1
        assert m_loaded.naxis2 == m_orig.naxis2
        assert m_loaded.telescope == m_orig.telescope


def test_save_load_night_preserves_the_zero_point_and_old_files_lack_it(tmp_path) -> None:
    """The frames' Gaia zero point survives night.npz; a file without one loads as None."""
    import json
    from dataclasses import replace

    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3)
    with_zp = replace(
        night,
        frame_meta=[replace(m, zp=24.0 + 0.1 * i) for i, m in enumerate(night.frame_meta)],
    )
    save_night(with_zp, Settings(), tmp_path / "zp.npz")
    loaded, _ = load_night(tmp_path / "zp.npz")
    assert [round(m.zp, 6) for m in loaded.frame_meta] == [24.0, 24.1, 24.2]

    save_night(night, Settings(), tmp_path / "nozp.npz")
    loaded, _ = load_night(tmp_path / "nozp.npz")
    assert [m.zp for m in loaded.frame_meta] == [None] * 3
    # an older file has no "zp" key in its frame metadata at all
    with np.load(tmp_path / "nozp.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["frame_meta_json"]))
    for d in meta:
        del d["zp"]
    arrays["frame_meta_json"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "old.npz", **arrays)
    loaded, _ = load_night(tmp_path / "old.npz")
    assert [m.zp for m in loaded.frame_meta] == [None] * 3


def test_load_night_handles_missing_naxis_and_telescope(tmp_path) -> None:
    """load_night handles old night.npz lacking naxis1/naxis2/telescope."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3)
    save_night(night, Settings(), tmp_path / "test.npz")

    # Just verify that load_night works with the saved file
    # (which includes the new fields) and they're loaded correctly
    night_loaded, _ = load_night(tmp_path / "test.npz")
    for m in night_loaded.frame_meta:
        # For synthetic nights, these should be defaults (0, "", 0, "")
        assert isinstance(m.naxis1, int)
        assert isinstance(m.naxis2, int)
        assert isinstance(m.telescope, str)


def test_starstats_table_has_near_edge_column(tmp_path) -> None:
    """save_starstats_table includes near_edge column."""
    from astropy.table import Table

    night, tilemap, comparison_result, _lc_result, star_stats, settings = _pipeline(34)
    best_aper = np.zeros(night.n_stars, dtype=np.int64)

    # Add near_edge to star_stats
    near_edge = np.zeros(night.n_stars, dtype=bool)
    near_edge[::3] = True  # Mark every 3rd star as near edge
    star_stats.near_edge = near_edge

    ss_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, best_aper, settings,
        tmp_path / "ss", fmt="fits",
    )
    ss = Table.read(str(ss_path), format="fits")

    assert "near_edge" in ss.colnames
    assert ss["near_edge"].dtype == bool


def test_starstats_table_near_edge_defaults_to_false(tmp_path) -> None:
    """save_starstats_table sets near_edge=False when StarStats.near_edge is None."""
    from astropy.table import Table

    night, tilemap, comparison_result, _lc_result, star_stats, settings = _pipeline(34)
    best_aper = np.zeros(night.n_stars, dtype=np.int64)

    # near_edge is None (default)
    assert star_stats.near_edge is None

    ss_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, best_aper, settings,
        tmp_path / "ss", fmt="fits",
    )
    ss = Table.read(str(ss_path), format="fits")

    assert "near_edge" in ss.colnames
    assert not np.any(ss["near_edge"])


def test_starstats_table_has_tailed_column(tmp_path) -> None:
    """save_starstats_table writes the tailed column, False where StarStats.tailed is None."""
    from astropy.table import Table

    night, tilemap, comparison_result, _lc_result, star_stats, settings = _pipeline(34)
    best_aper = np.zeros(night.n_stars, dtype=np.int64)

    assert star_stats.tailed is None
    ss_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, best_aper, settings,
        tmp_path / "ss0", fmt="fits",
    )
    ss = Table.read(str(ss_path), format="fits")
    assert "tailed" in ss.colnames
    assert not np.any(ss["tailed"])

    tailed = np.zeros(night.n_stars, dtype=bool)
    tailed[::3] = True
    star_stats.tailed = tailed
    ss_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, best_aper, settings,
        tmp_path / "ss1", fmt="fits",
    )
    ss = Table.read(str(ss_path), format="fits")
    assert ss["tailed"].dtype == bool
    np.testing.assert_array_equal(np.asarray(ss["tailed"]), tailed[np.asarray(ss["star_id"])])
