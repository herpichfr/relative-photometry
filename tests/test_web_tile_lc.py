"""Tests for tile_lc web utilities."""

import numpy as np

from relphot.web.tile_lc import envelope, member_rms, select_members


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
