"""Tests for tile_lc web utilities."""

import numpy as np

from relphot.web.tile_lc import (
    envelope,
    individual_ratio_curves,
    is_median_ensemble,
    member_rms,
    select_members,
)


class TestMemberRms:
    """Tests for member_rms utility."""

    def test_rms_simple(self):
        """RMS of simple data."""
        norm_flux = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        ens = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rms = member_rms(norm_flux, ens)
        assert rms.shape == (1,)
        assert rms[0] < 0.1  # Small RMS for constant data

    def test_rms_with_nan(self):
        """RMS with NaN values (but enough finite points)."""
        norm_flux = np.array([[1.0, np.nan, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        ens = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rms = member_rms(norm_flux, ens)
        assert rms.shape == (1,)
        assert np.isfinite(rms[0])

    def test_rms_few_frames(self):
        """RMS with fewer than 5 finite frames returns NaN."""
        norm_flux = np.array([[1.0, 2.0, 1.0, np.nan, np.nan]], dtype=np.float32)
        ens = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rms = member_rms(norm_flux, ens)
        assert np.isnan(rms[0])

    def test_rms_multiple_members(self):
        """RMS for multiple members."""
        norm_flux = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32
        )
        ens = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rms = member_rms(norm_flux, ens)
        assert rms.shape == (2,)
        assert np.all(np.isfinite(rms))


class TestEnvelope:
    """Tests for envelope percentile calculation."""

    def test_envelope_simple(self):
        """Envelope of simple data."""
        norm_flux = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        env = envelope(norm_flux)
        assert env is not None
        assert "median" in env
        assert "lo" in env
        assert "hi" in env
        assert len(env["median"]) == 5

    def test_envelope_few_members(self):
        """Envelope with fewer than 3 members returns None for that frame."""
        norm_flux = np.array([[1.0, np.nan], [2.0, 2.0]], dtype=np.float32)
        env = envelope(norm_flux)
        assert env is not None
        assert env["median"][0] is None  # Only 2 finite members in frame 0
        assert env["median"][1] is None  # Only 2 finite members in frame 1

    def test_envelope_nan_values(self):
        """Envelope with NaN values."""
        norm_flux = np.array(
            [[1.0, 1.1, 1.0], [1.0, np.nan, 1.0], [1.0, 0.9, 1.0]], dtype=np.float32
        )
        env = envelope(norm_flux)
        assert env is not None
        assert len(env["median"]) == 3
        assert env["median"][0] is not None
        assert env["median"][1] is None  # Only 2 finite values (< 3)
        assert env["median"][2] is not None


class TestSelectMembers:
    """Tests for member selection by magnitude, weight, or RMS."""

    def test_select_all_when_limit_too_high(self):
        """All members selected when limit >= n."""
        mag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = select_members("mag", limit=100, mag=mag)
        assert len(selected) == 5

    def test_select_by_mag_stratified(self):
        """Magnitude selection stratifies across range."""
        mag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = select_members("mag", limit=3, mag=mag)
        assert len(selected) <= 3
        # Should include roughly evenly spaced indices

    def test_select_by_weight_top(self):
        """Weight selection includes top members."""
        weight = np.array([0.1, 0.3, 0.5, 0.2, 0.1])
        selected = select_members("weight", limit=2, weight=weight)
        assert len(selected) == 2
        # Top 2 by weight are indices 2 and 1
        assert 2 in selected

    def test_select_by_rms_worst(self):
        """RMS selection includes worst (highest) RMS."""
        rms = np.array([0.1, 0.5, 0.2, 0.3, 0.4])
        selected = select_members("rms", limit=2, rms=rms)
        assert len(selected) == 2
        # Worst RMS is index 1 (0.5)
        assert 1 in selected

    def test_select_must_include(self):
        """Target must be included regardless of selection criteria."""
        mag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = select_members("mag", limit=2, mag=mag, must_include={4})
        assert 4 in selected

    def test_select_limit_zero_means_all(self):
        """Limit of 0 or negative means all members."""
        mag = np.array([1.0, 2.0, 3.0])
        selected = select_members("mag", limit=0, mag=mag)
        assert len(selected) == 3

    def test_select_with_nan_rms(self):
        """RMS selection handles NaN values."""
        rms = np.array([0.1, np.nan, 0.2, 0.3, np.nan])
        selected = select_members("rms", limit=3, rms=rms)
        # Should select the finite ones, NaN at end
        assert len(selected) <= 3
        # NaN indices should not be selected
        assert 1 not in selected
        assert 4 not in selected


class TestIndividualRatioCurves:
    """target / comp_i on the plotted curve's scale."""

    @staticmethod
    def _case(n_members: int, n_frames: int = 12, seed: int = 0):
        rng = np.random.default_rng(seed)
        c = 1.0 + 0.01 * rng.standard_normal((n_members, n_frames))
        ens = np.median(c, axis=0)
        # a plotted (decorrelated) curve that is not just 1: dip plus a frame-level correction
        lc = 1.0 - 0.02 * np.exp(-0.5 * ((np.arange(n_frames) - 6) / 1.5) ** 2)
        lc = lc * (1.0 + 0.003 * rng.standard_normal(n_frames))
        return lc, ens, c

    def test_median_of_individuals_is_the_plotted_curve_for_odd_members(self):
        lc, ens, c = self._case(31)
        ratios = individual_ratio_curves(lc, ens, c)
        assert ratios.shape == c.shape
        np.testing.assert_allclose(np.median(ratios, axis=0), lc, rtol=1e-12)

    def test_median_of_individuals_is_the_plotted_curve_for_even_members_to_second_order(self):
        lc, ens, c = self._case(30)
        ratios = individual_ratio_curves(lc, ens, c)
        np.testing.assert_allclose(np.median(ratios, axis=0), lc, rtol=1e-5)

    def test_each_ratio_is_target_flux_over_that_comparison(self):
        lc, ens, c = self._case(5)
        rel_target = lc * ens  # relative_flux of the target, before dividing by the ensemble
        ratios = individual_ratio_curves(lc, ens, c)
        np.testing.assert_allclose(ratios, rel_target[None, :] / c, rtol=1e-12)

    def test_nan_where_any_input_is_missing(self):
        lc, ens, c = self._case(5)
        lc[2] = np.nan
        ens[3] = np.nan
        c[1, 4] = np.nan
        c[0, 5] = 0.0
        ratios = individual_ratio_curves(lc, ens, c)
        assert np.all(np.isnan(ratios[:, 2])) and np.all(np.isnan(ratios[:, 3]))
        assert np.isnan(ratios[1, 4]) and np.isnan(ratios[0, 5])
        assert np.isfinite(ratios[2, 0])


class TestIsMedianEnsemble:
    def test_equal_weights_are_the_median_ensemble(self):
        assert is_median_ensemble(np.full(7, 1 / 7, dtype=np.float32), np.zeros(7))

    def test_unequal_weights_are_not(self):
        w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        assert not is_median_ensemble(w, np.zeros(3))

    def test_clipped_members_or_no_members_are_not(self):
        assert not is_median_ensemble(np.full(4, 0.25), np.array([0, 0, 1, 0]))
        assert not is_median_ensemble(np.array([]), np.array([]))
