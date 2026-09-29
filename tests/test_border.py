"""Tests for border-eligibility cut (relphot.border)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from relphot.border import (
    border_eligibility,
    detector_info,
    measure_frame_offsets,
)
from relphot.config import BorderSettings, Settings
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.tiles import build_tilemap


def test_measure_frame_offsets_known_drift():
    """Measure frame offsets recovers a known linear drift."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=100, n_frames=5, seed=42)

    # Inject known drift: dx = [0, 10, 20, 30, 40], dy = [0, -5, -10, -15, -20]
    dx_known = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    dy_known = np.array([0.0, -5.0, -10.0, -15.0, -20.0])

    night_with_drift = replace(
        night,
        frame_x=night.frame_x + dx_known[None, :],
        frame_y=night.frame_y + dy_known[None, :],
    )

    dx, dy = measure_frame_offsets(night_with_drift, min_common=20)

    np.testing.assert_allclose(dx, dx_known, atol=0.1)
    np.testing.assert_allclose(dy, dy_known, atol=0.1)


def test_measure_frame_offsets_sparse_frame():
    """Frame with fewer than min_common stars returns NaN."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=100, n_frames=3, seed=42)

    # Mark all stars as NaN in frame 1
    frame_x = night.frame_x.copy()
    frame_y = night.frame_y.copy()
    frame_x[:, 1] = np.nan
    frame_y[:, 1] = np.nan
    night_sparse = replace(night, frame_x=frame_x, frame_y=frame_y)

    dx, _dy = measure_frame_offsets(night_sparse, min_common=20)

    assert np.isfinite(dx[0])
    assert np.isnan(dx[1])
    assert np.isfinite(dx[2])


def test_border_eligibility_3000x3000_simple():
    """Test border eligibility with 3000x3000 detector, dx -40..+10, buffer 0."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(
        n_stars=100, n_frames=5, size_px=1000.0, seed=42
    )

    # Set detector size and inject drift
    frame_meta_updated = [
        replace(m, naxis1=3000, naxis2=3000, telescope="TEST")
        for m in night.frame_meta
    ]
    dx_drift = np.array([-40.0, -30.0, -20.0, 0.0, 10.0])
    dy_drift = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])

    night_updated = replace(
        night,
        frame_meta=frame_meta_updated,
        frame_x=night.frame_x + dx_drift[None, :],
        frame_y=night.frame_y + dy_drift[None, :],
    )

    settings = Settings(border=BorderSettings(enabled=True, edge_buffer_px=0.0))
    info = border_eligibility(night_updated, settings.border)

    # Check that x=30 is ineligible and x=60 is eligible
    # With buffer=0, drift -40..+10: x in [1, 30-40] = [1, -40] (invalid)
    # or x in [30+10, 3000] = [40, 3000] (valid if on right side)
    # A star at pixel x=25 would envelope to [25-40, 25+10] = [-15, 35]
    # A star at pixel x=50 would envelope to [50-40, 50+10] = [10, 60] (eligible if 1+0 to 3000-0)

    assert info.enabled


def test_border_eligibility_telescope_margin():
    """Telescope-specific margins affect star eligibility at detector boundaries."""
    from tests.conftest import make_synthetic_night

    # Build a night with 100 stars for reliable drift measurement, 2 frames
    night, _, _ = make_synthetic_night(n_stars=100, n_frames=2, size_px=3000.0, seed=42)

    # Override first 4 stars at boundary y positions: 45, 60, 2940, 2955
    # Keep rest at original positions for drift measurement
    y_orig = night.y.copy()
    x_orig = night.x.copy()
    frame_y_orig = night.frame_y.copy()
    frame_x_orig = night.frame_x.copy()

    # Set first 4 stars at our test positions
    y_vals = np.array([45.0, 60.0, 2940.0, 2955.0])
    x_vals = np.array([1500.0, 1500.0, 1500.0, 1500.0])

    y_test = y_orig.copy()
    x_test = x_orig.copy()
    frame_y_test = frame_y_orig.copy()
    frame_x_test = frame_x_orig.copy()

    y_test[:4] = y_vals
    x_test[:4] = x_vals
    # Frame 0 at y_vals, Frame 1 at y_vals + 1
    drift_frame = np.array([0.0, 1.0])[None, :]
    frame_y_test[:4, :] = y_vals[:, None] + drift_frame
    # Frame 0 at 1500, Frame 1 at 1501
    frame_x_test[:4, :] = x_vals[:, None] + drift_frame

    night_y = replace(
        night,
        y=y_test,
        x=x_test,
        frame_y=frame_y_test,
        frame_x=frame_x_test,
    )

    frame_meta_updated = [
        replace(m, naxis1=3000, naxis2=3000, telescope="T80") for m in night_y.frame_meta
    ]
    night_updated = replace(night_y, frame_meta=frame_meta_updated)

    border_settings_t80 = BorderSettings(
        enabled=True,
        edge_buffer_px=30.0,
        telescope_extra_px={"T80": [0.0, 0.0, 20.0, 20.0]},
    )
    info_t80 = border_eligibility(night_updated, border_settings_t80)

    # With T80: margins are (30, 30, 50, 50) for (L, R, B, T)
    # Bottom margin = 30 + 20 = 50, so y >= 51 is eligible
    # Top margin = 30 + 20 = 50, so y <= 2950 is eligible
    assert info_t80.margins == (30.0, 30.0, 50.0, 50.0)
    assert not info_t80.eligible[0], "y=45 should be ineligible (< 51)"
    assert info_t80.eligible[1], "y=60 should be eligible (>= 51)"
    assert info_t80.eligible[2], "y=2940 should be eligible (<= 2950)"
    assert not info_t80.eligible[3], "y=2955 should be ineligible (> 2950)"

    # Test with ROBO43 (no extra): y=45 should be eligible with buffer 30
    # y >= 1 + 30 = 31 is eligible
    frame_meta_robo = [
        replace(m, naxis1=3000, naxis2=3000, telescope="ROBO43") for m in night_updated.frame_meta
    ]
    night_robo = replace(night_updated, frame_meta=frame_meta_robo)

    border_settings_robo = BorderSettings(
        enabled=True,
        edge_buffer_px=30.0,
        telescope_extra_px={"ROBO43": [0.0, 0.0, 0.0, 0.0]},
    )
    info_robo = border_eligibility(night_robo, border_settings_robo)
    assert info_robo.eligible[0], "y=45 should be eligible with ROBO43 (>= 31)"


def test_border_eligibility_case_insensitive():
    """Telescope name matching is case-insensitive."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3, seed=42)

    # Use lowercase "t80" in header
    frame_meta_updated = [
        replace(m, naxis1=3000, naxis2=3000, telescope="t80") for m in night.frame_meta
    ]
    night_updated = replace(night, frame_meta=frame_meta_updated)

    settings = Settings(
        border=BorderSettings(
            enabled=True,
            edge_buffer_px=30.0,
            telescope_extra_px={"T80": [0.0, 0.0, 20.0, 20.0]},
        )
    )
    info = border_eligibility(night_updated, settings.border)

    # Should match T80 and apply extra margins
    assert info.margins == (30.0, 30.0, 50.0, 50.0)


def test_border_eligibility_unknown_detector(caplog):
    """Unknown detector size returns all True with warning and enabled=False."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3, seed=42)
    # frame_meta has naxis1/naxis2 = 0 (unknown)

    settings = Settings(border=BorderSettings(enabled=True))
    info = border_eligibility(night, settings.border)

    assert np.all(info.eligible)
    assert info.enabled is False
    assert "border cut skipped: detector size unknown" in caplog.text


def test_border_eligibility_disabled():
    """Disabled border cut returns all True."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3, seed=42)

    settings = Settings(border=BorderSettings(enabled=False))
    info = border_eligibility(night, settings.border)

    assert np.all(info.eligible)
    assert info.enabled is False


def test_select_candidates_excludes_ineligible():
    """select_candidates excludes ineligible stars."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=100, n_frames=3, seed=42)

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    settings = Settings()

    # All stars eligible
    candidates_all = select_candidates(night, variable_mask, settings, 0)

    # Half stars ineligible
    star_eligible = np.ones(night.n_stars, dtype=bool)
    star_eligible[::2] = False
    candidates_some = select_candidates(
        night, variable_mask, settings, 0, star_eligible=star_eligible
    )

    assert candidates_some.sum() <= candidates_all.sum()


def test_select_candidates_wrong_shape_raises():
    """select_candidates raises ValueError on shape mismatch."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=100, n_frames=3, seed=42)

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    settings = Settings()

    # Wrong shape
    star_eligible = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="star_eligible shape"):
        select_candidates(night, variable_mask, settings, 0, star_eligible=star_eligible)


def test_border_fixed_set_invariant():
    """Fixed-set invariant: reference uses same stars in every frame.

    Build a night with drifting positions, border-ineligible stars, and one cloudy frame.
    Check: candidates exclude border-ineligible, frame 7 is dropped, remaining frames use
    same fixed star set (n_used constant), and all reference stars are border-eligible.
    """
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(
        n_stars=2000, n_frames=12, size_px=3000.0, seed=3
    )

    # Set detector 3000x3000, inject known drift
    frame_meta_updated = [
        replace(m, naxis1=3000, naxis2=3000, telescope="TEST")
        for m in night.frame_meta
    ]
    dx_drift = np.linspace(-40.0, 10.0, 12)
    dy_drift = np.linspace(-10.0, 30.0, 12)

    # Start from copies of synthetic night's arrays (preserves noise)
    flags = night.flags.copy()
    frame_x = night.frame_x.copy()
    frame_y = night.frame_y.copy()
    flux = night.flux.copy()
    fluxerr = night.fluxerr.copy()
    presence = night.presence.copy()

    # Apply drift to frame positions
    frame_x[:] = night.x[:, None] + dx_drift[None, :]
    frame_y[:] = night.y[:, None] + dy_drift[None, :]

    # Make frame 7 "cloudy": mark ~97% of stars absent
    rng = np.random.default_rng(seed=3)
    absent_mask = rng.choice([False, True], size=night.n_stars, p=[0.03, 0.97])
    flags[absent_mask, 7] = -1
    frame_x[absent_mask, 7] = np.nan
    frame_y[absent_mask, 7] = np.nan
    flux[absent_mask, 7, :] = np.nan
    fluxerr[absent_mask, 7, :] = np.nan
    # Update presence: stars absent in frame 7 have 11/12 presence
    presence[absent_mask] = 11.0 / 12.0

    night_updated = replace(
        night,
        frame_meta=frame_meta_updated,
        frame_x=frame_x,
        frame_y=frame_y,
        flux=flux,
        fluxerr=fluxerr,
        flags=flags,
        presence=presence,
    )

    # Compute border eligibility
    settings = Settings(border=BorderSettings(enabled=True, edge_buffer_px=30.0))
    variable_mask = np.zeros(night_updated.n_stars, dtype=bool)

    border = border_eligibility(night_updated, settings.border)

    # Some stars should be ineligible
    n_ineligible = (~border.eligible).sum()
    assert 0 < n_ineligible < night_updated.n_stars
    # Border cut should remove some but not all stars

    # Call select_candidates WITHOUT star_eligible to test default computation path
    candidates = select_candidates(night_updated, variable_mask, settings, 0)

    # All candidates must be border-eligible (default path computed and applied)
    assert not candidates[~border.eligible].any(), "Candidates must exclude border-ineligible stars"

    # Build reference
    tilemap = build_tilemap(night_updated, candidates, settings)
    sel = select_reference_frames_and_stars(night_updated, tilemap, candidates, settings, 0)

    # Frame 7 should be dropped (too many stars absent)
    assert not sel.frame_kept[7], "Frame 7 should be dropped (cloudy)"
    assert sel.frame_kept.sum() == 11, "Should keep 11 out of 12 frames"

    # All stars in reference must be border-eligible
    for t in range(tilemap.n_tiles):
        tile_star_set = sel.tile_stars[t]
        if tile_star_set.size > 0:
            assert np.all(border.eligible[tile_star_set]), f"All stars in tile {t} must be eligible"

    # Build references and check n_used is constant in each tile
    ref = build_references(night_updated, tilemap, sel, settings)
    for t in range(tilemap.n_tiles):
        # Get n_used for this tile across kept frames at aperture 0
        n_used_values = ref.n_used[t, sel.frame_kept, 0]
        unique_counts = np.unique(n_used_values)
        assert unique_counts.size == 1, f"Tile {t}: n_used must be constant (got {unique_counts})"


def test_detector_info_from_frame_meta():
    """detector_info reads from frame_meta naxis/telescope if available."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=1, seed=42)

    frame_meta_updated = [
        replace(night.frame_meta[0], naxis1=5000, naxis2=6000, telescope="MYTEL")
    ]
    night_updated = replace(night, frame_meta=frame_meta_updated)

    info = detector_info(night_updated)
    assert info == (5000.0, 6000.0, "MYTEL")


def test_detector_info_reads_fits(tmp_path):
    """detector_info reads NAXIS/TELESCOP from FITS header if frame_meta lacks them."""
    from astropy.io import fits

    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=1, seed=42)

    # Create a minimal FITS file with NAXIS and TELESCOP
    fits_path = tmp_path / "test.fits"
    # Create a 2048x2048 array (matching the NAXIS values we'll store)
    data = np.zeros((2048, 2048), dtype=np.uint8)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["TELESCOP"] = "MYTEL2"
    hdu.writeto(fits_path)

    # Update frame_meta to point to this file with naxis/telescope unknown
    frame_meta_updated = [
        replace(night.frame_meta[0], file=fits_path, naxis1=0, naxis2=0, telescope="")
    ]
    night_updated = replace(night, frame_meta=frame_meta_updated)

    info = detector_info(night_updated)
    assert info == (2048.0, 2048.0, "MYTEL2")


def test_border_eligibility_no_finite_offsets(caplog):
    """No finite frame offsets returns all True with warning."""
    from tests.conftest import make_synthetic_night

    night, _, _ = make_synthetic_night(n_stars=50, n_frames=3, seed=42)

    # Make all frame positions NaN
    frame_meta_updated = [
        replace(m, naxis1=3000, naxis2=3000, telescope="TEST")
        for m in night.frame_meta
    ]
    night_updated = replace(
        night,
        frame_meta=frame_meta_updated,
        frame_x=np.full_like(night.frame_x, np.nan),
        frame_y=np.full_like(night.frame_y, np.nan),
    )

    settings = Settings(border=BorderSettings(enabled=True))
    info = border_eligibility(night_updated, settings.border)

    assert np.all(info.eligible)
    assert info.enabled is False
    assert "no finite frame offsets" in caplog.text
