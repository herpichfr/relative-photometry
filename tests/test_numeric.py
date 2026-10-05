"""Tests for relphot.numeric: shared numeric utilities."""

from __future__ import annotations

import warnings

import numpy as np

from relphot.numeric import (
    fit_noise_floor,
    mad_sigma,
    nanmedian_quiet,
    quantile_bin_edges,
    unit_vectors,
    weighted_clipped_combine,
)


def test_nanmedian_quiet_all_nan_no_warning() -> None:
    """All-NaN slice gives NaN without RuntimeWarning."""
    arr = np.full(10, np.nan)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        result = nanmedian_quiet(arr)
        assert np.isnan(result)
        assert len(w) == 0, "nanmedian_quiet should suppress RuntimeWarning for all-NaN input"


def test_nanmedian_quiet_long_axis_matches_numpy_exactly() -> None:
    """Axes past numpy's slow-path threshold use the sorted implementation: same values."""
    rng = np.random.default_rng(1)
    for dtype in (np.float32, np.float64):
        for shape, axis in (((700, 9), 0), ((9, 650), 1), ((3, 700, 4), 1), ((600, 5), 0)):
            arr = rng.normal(size=shape).astype(dtype)
            arr[rng.random(shape) < 0.3] = np.nan
            lines = np.moveaxis(arr, axis, -1)  # a view: edits land in ``arr``
            lines[(0,) * (lines.ndim - 1)] = np.nan  # an all-NaN slice
            single = lines[(1,) * (lines.ndim - 1)]
            single[:] = np.nan
            single[3] = 3.5  # a slice with one valid value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                expected = np.nanmedian(arr, axis=axis)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=RuntimeWarning)
                got = nanmedian_quiet(arr, axis=axis)
            assert not caught
            assert got.dtype == expected.dtype
            np.testing.assert_array_equal(got, expected)


def test_mad_sigma_standard_normal() -> None:
    """MAD sigma of N(0,1) sample within 0.03 of 1.0."""
    rng = np.random.default_rng(seed=42)
    sample = rng.standard_normal(20000)
    sigma = mad_sigma(sample)
    assert np.isfinite(sigma)
    assert abs(sigma - 1.0) < 0.03, f"Expected MAD sigma ~1.0, got {sigma}"


def test_mad_sigma_all_nan() -> None:
    """MAD sigma of all-NaN array returns NaN."""
    arr = np.full(10, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = mad_sigma(arr)
        rows = mad_sigma(np.array([[np.nan, np.nan], [1.0, 3.0]]), axis=1)
    assert np.isnan(result)
    assert np.isnan(rows[0]) and np.isclose(rows[1], 1.4826)


def test_mad_sigma_zero_mad() -> None:
    """MAD sigma with zero MAD returns inf."""
    arr = np.full(10, 5.0)  # All same value -> MAD = 0
    result = mad_sigma(arr)
    assert np.isinf(result) and result > 0


def test_weighted_clipped_combine_axis0_basic() -> None:
    """Weighted clipped combine along axis 0 with basic outlier clipping."""
    values = np.array([[1.0, 2.0, 10.0], [1.0, 2.0, 2.0]], dtype=np.float64).T  # (3, 2)
    weights = np.ones_like(values)
    mean, sigma, n_used, mask = weighted_clipped_combine(
        values, weights, clip_sigma=3.0, max_iter=5, axis=0
    )
    # values is (3, 2): 3 rows, 2 columns. axis=0 reduces rows -> result shape (2,)
    assert mean.shape == (2,)
    assert sigma.shape == (2,)
    assert n_used.shape == (2,)
    assert mask.shape == (3, 2)
    # The outlier (10.0) should be clipped.
    assert mask[2, 0] is False or mask[2, 0] == np.bool_(False)


def test_weighted_clipped_combine_transpose_equivalence() -> None:
    """Axis 0 vs axis 1 equivalence via transpose."""
    values = np.array([[1.0, 2.0, 10.0], [1.0, 2.0, 2.0]], dtype=np.float64)  # (2, 3)
    weights = np.ones_like(values)

    # Combine along axis 0
    mean0, sigma0, n_used0, mask0 = weighted_clipped_combine(
        values, weights, clip_sigma=3.0, max_iter=5, axis=0
    )

    # Transpose, combine along axis 1, transpose back
    values_T = values.T  # (3, 2)
    weights_T = weights.T
    mean1, sigma1, n_used1, mask1 = weighted_clipped_combine(
        values_T, weights_T, clip_sigma=3.0, max_iter=5, axis=1
    )

    # After transposing back, results should match.
    np.testing.assert_allclose(mean0, mean1, rtol=1e-14)
    np.testing.assert_allclose(sigma0, sigma1, rtol=1e-14)
    np.testing.assert_array_equal(n_used0, n_used1)
    np.testing.assert_array_equal(mask0, mask1.T)


def test_weighted_clipped_combine_outlier_clipped() -> None:
    """Verify that an outlier is clipped (mask False)."""
    # Create 5 values: 4 normal and 1 outlier
    values = np.array([[1.0, 1.05, 0.95, 1.02, 10.0]], dtype=np.float64).T  # (5, 1)
    weights = np.ones_like(values)
    _mean, _sigma, _n_used, mask = weighted_clipped_combine(
        values, weights, clip_sigma=3.0, max_iter=10, axis=0
    )
    # The outlier (10.0) should be clipped.
    assert mask[4, 0] is False or mask[4, 0] == np.bool_(False), (
        f"Expected outlier (10.0) to be clipped, but mask={mask}"
    )


def test_weighted_clipped_combine_nan_handling() -> None:
    """NaN values are never used, regardless of weight."""
    values = np.array([[1.0, np.nan], [1.0, 2.0], [np.nan, 2.0]], dtype=np.float64)
    weights = np.ones_like(values)
    _mean, _sigma, _n_used, mask = weighted_clipped_combine(
        values, weights, clip_sigma=3.0, max_iter=5, axis=0
    )
    # All NaN entries should have mask False.
    assert mask[0, 1] is False or mask[0, 1] == np.bool_(False)
    assert mask[2, 0] is False or mask[2, 0] == np.bool_(False)


def test_quantile_bin_edges_basic() -> None:
    """Quantile bin edges for basic array."""
    mag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    edges = quantile_bin_edges(mag, n_bins=2)
    assert len(edges) >= 2
    assert edges[0] <= 1.0
    assert edges[-1] >= 5.0


def test_quantile_bin_edges_with_nan() -> None:
    """Quantile bin edges ignores NaN."""
    mag = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    edges = quantile_bin_edges(mag, n_bins=2)
    # Should only use finite values: [1, 2, 3, 4]
    assert len(edges) >= 2
    assert edges[0] <= 1.0
    assert edges[-1] >= 4.0


def test_quantile_bin_edges_empty() -> None:
    """Quantile bin edges with no finite values."""
    mag = np.array([np.nan, np.nan, np.nan])
    edges = quantile_bin_edges(mag, n_bins=2)
    assert len(edges) == 0


def test_fit_noise_floor_recovery() -> None:
    """Fit noise floor recovers a known floor within 10%."""
    # Create synthetic data: magnitude-dependent noise floor.
    # True floor: 10**(0.2*(m-15)) * 1e-3
    rng = np.random.default_rng(seed=123)
    mag = np.linspace(10, 20, 200)
    true_floor = 10.0 ** (0.2 * (mag - 15)) * 1e-3
    # Add scatter around the floor: sigma ~ floor * (1 + 0.1 * noise)
    sigma = true_floor * (1 + 0.1 * rng.standard_normal(200))
    valid = np.ones_like(mag, dtype=bool)

    floor_func = fit_noise_floor(mag, sigma, valid, n_bins=5, min_bin_stars=10)

    # Test at bin centres (roughly).
    test_mags = np.array([12.0, 14.0, 16.0, 18.0])
    predicted = floor_func(test_mags)
    expected = 10.0 ** (0.2 * (test_mags - 15)) * 1e-3

    # Should recover within 10%.
    rel_err = np.abs(predicted - expected) / expected
    assert np.all(rel_err < 0.1), f"Fit error > 10%: {rel_err}"


def test_fit_noise_floor_no_bins() -> None:
    """Fit noise floor with no surviving bins returns NaN."""
    mag = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.1, 0.1])
    valid = np.ones_like(mag, dtype=bool)
    # Require min_bin_stars=10 but only have 3 stars -> no bins survive.
    floor_func = fit_noise_floor(mag, sigma, valid, n_bins=5, min_bin_stars=10)
    result = floor_func(np.array([2.0]))
    assert np.isnan(result[0])


def test_fit_noise_floor_small_sample_caps_bins() -> None:
    """45 valid stars, n_bins=20, min_bin_stars=5: bins capped to 9, the floor is finite."""
    mag = np.linspace(10.0, 18.0, 45)
    sigma = 1e-3 * 10.0 ** (0.2 * (mag - 15.0))
    floor_func = fit_noise_floor(mag, sigma, np.ones(45, dtype=bool), n_bins=20, min_bin_stars=5)
    test_mags = np.array([11.0, 13.0, 15.0, 17.0])
    expected = 1e-3 * 10.0 ** (0.2 * (test_mags - 15.0))
    assert np.all(np.abs(floor_func(test_mags) - expected) / expected < 0.1)


def test_fit_noise_floor_cap_inactive_for_large_sample() -> None:
    """For n_valid >= n_bins * min_bin_stars the floor equals the explicit-edge (uncapped) fit."""
    rng = np.random.default_rng(seed=7)
    mag = rng.uniform(10.0, 20.0, 200)
    sigma = 1e-3 * 10.0 ** (0.2 * (mag - 15.0)) * (1 + 0.1 * rng.standard_normal(200))
    valid = np.ones(200, dtype=bool)
    valid[::10] = False  # 180 valid >= 8 * 10
    floor_func = fit_noise_floor(mag, sigma, valid, n_bins=8, min_bin_stars=10)
    edges = quantile_bin_edges(mag[valid], 8)
    centres, floors = [], []
    for i in range(len(edges) - 1):
        hi = (mag <= edges[i + 1]) if i == len(edges) - 2 else (mag < edges[i + 1])
        in_bin = valid & (mag >= edges[i]) & hi
        centres.append((edges[i] + edges[i + 1]) / 2.0)
        floors.append(nanmedian_quiet(np.log10(sigma[in_bin])))
    test_mags = np.linspace(9.0, 21.0, 25)
    expected = 10.0 ** np.interp(test_mags, centres, floors)
    np.testing.assert_array_equal(floor_func(test_mags), expected)


def test_fit_noise_floor_fewer_valid_than_min_bin_stars() -> None:
    """4 valid stars with min_bin_stars=5 give NaN; 5 valid stars give a finite floor."""
    mag = np.linspace(10.0, 18.0, 50)
    sigma = np.full_like(mag, 1e-3)
    valid = np.zeros(50, dtype=bool)
    valid[:4] = True
    m = np.array([10.5])
    assert np.isnan(fit_noise_floor(mag, sigma, valid, n_bins=20, min_bin_stars=5)(m)[0])
    valid[4] = True
    floor_func = fit_noise_floor(mag, sigma, valid, n_bins=20, min_bin_stars=5)
    np.testing.assert_allclose(floor_func(m), 1e-3)


def test_unit_vectors_basic() -> None:
    """Unit vectors have magnitude 1."""
    ra = np.array([0.0, 90.0, 180.0])
    dec = np.array([0.0, 0.0, 0.0])
    vecs = unit_vectors(ra, dec)
    assert vecs.shape == (3, 3)
    mags = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(mags, 1.0, rtol=1e-14)


def test_unit_vectors_north_pole() -> None:
    """Unit vector at north pole."""
    vecs = unit_vectors(np.array([0.0]), np.array([90.0]))
    expected = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(vecs, expected, atol=1e-14)
