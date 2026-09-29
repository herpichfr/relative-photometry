"""Unit tests for :mod:`relphot.multinight` (Unit A: cross-match and zero-point tie).

Synthetic nights are built directly as :class:`~relphot.multinight.NightProducts`
(no files) via :func:`_synthetic_tie_scenario`: a uniform 1x1 degree field of
stars with a known per-night zero-point surface ``Z_n(xi, eta, M)`` injected
(smooth spatial+magnitude poly2, anchor forced to zero), optional per-night
calibration-floor scatter, and a small fraction of comparison stars given a
one-night magnitude excursion that the whole-star clipping should reject.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from relphot.config import MultiNightSettings, settings_to_dict
from relphot.decorrelate import compute_crowding
from relphot.exceptions import ConfigError, MultiNightError
from relphot.multinight import (
    NightCrossMatch,
    NightProducts,
    build_multinight_lightcurves,
    check_compatible,
    core_crossmatch,
    crossmatch_nights,
    evaluate_zero_point,
    load_multinight,
    resolve_anchor_index,
    save_floor_report,
    save_multinight,
    save_multinight_tables,
    save_tie_report,
    split_loose_nights,
    tie_loose_nights,
    tie_nights,
)
from relphot.numeric import nanmedian_quiet


def _synthetic_tie_scenario(
    *,
    n_nights: int = 3,
    n_stars: int = 3000,
    frames_per_night: int = 100,
    n_aper: int = 3,
    seed: int = 0,
    field_deg: float = 1.0,
    ra0: float = 150.0,
    dec0: float = -30.0,
    anchor_index: int = 0,
    zp_coefs: list[tuple[float, ...]] | None = None,
    floor_mmag: list[float] | None = None,
    variable_frac: float = 0.0,
    variable_amp_range: tuple[float, float] = (0.15, 0.3),
    comparison_frac: float = 0.6,
    mag_range: tuple[float, float] = (13.0, 19.0),
    jitter_arcsec: float = 0.0,
    missing_frac: float = 0.0,
    stat_sigma_mag: tuple[float, ...] = (0.010, 0.005, 0.007),
    n_unmeasured: int = 0,
) -> tuple[list[NightProducts], dict]:
    """A synthetic multi-night scenario with a known injected zero-point tie."""
    rng = np.random.default_rng(seed)

    ra_true = ra0 + rng.uniform(-field_deg / 2, field_deg / 2, n_stars) / np.cos(np.radians(dec0))
    dec_true = dec0 + rng.uniform(-field_deg / 2, field_deg / 2, n_stars)
    mag_i = rng.uniform(mag_range[0], mag_range[1], n_stars)

    u = (ra_true - ra0) * np.cos(np.radians(dec0))
    v = dec_true - dec0
    mag0_truth = float(np.median(mag_i))
    dm = mag_i - mag0_truth

    if zp_coefs is None:
        zp_coefs = []
        for n in range(n_nights):
            if n == anchor_index:
                zp_coefs.append((0.0,) * 8)
                continue
            local_rng = np.random.default_rng(seed * 100 + n)
            # Term-specific caps (mag) so each term contributes at most ~10 mmag
            # over the field (|u|, |v| <~ 0.5 deg) and magnitude range (|dm| <~ 3
            # mag) used below -- a magnitude-degree term is far more sensitive to
            # its input range than a spatial one, so it needs a far smaller cap.
            term_cap = np.array([0.010, 0.020, 0.020, 0.040, 0.040, 0.040, 0.003, 0.0011])
            amp = local_rng.uniform(0.3, 1.0, 8) * term_cap
            sign = local_rng.choice([-1.0, 1.0], 8)
            zp_coefs.append(tuple(amp * sign))

    def _zp_truth(n: int) -> np.ndarray:
        c = zp_coefs[n]
        return (
            c[0] + c[1] * u + c[2] * v + c[3] * u**2 + c[4] * u * v + c[5] * v**2
            + c[6] * dm + c[7] * dm**2
        )

    comparison_mask_1d = rng.uniform(size=n_stars) < comparison_frac

    non_anchor_pool = np.array([n for n in range(n_nights) if n != anchor_index])
    comp_idx = np.nonzero(comparison_mask_1d)[0]
    n_variable = round(variable_frac * comp_idx.size)
    if n_variable and non_anchor_pool.size:
        variable_idx = rng.choice(comp_idx, size=n_variable, replace=False)
        variable_night = rng.choice(non_anchor_pool, size=n_variable)
        variable_amp = rng.uniform(*variable_amp_range, size=n_variable) * rng.choice(
            [-1.0, 1.0], size=n_variable
        )
    else:
        variable_idx = np.array([], dtype=np.int64)
        variable_night = np.array([], dtype=np.int64)
        variable_amp = np.array([], dtype=np.float64)
    variable_mask = np.zeros(n_stars, dtype=bool)
    variable_mask[variable_idx] = True

    if floor_mmag is None:
        floor_mmag = [0.0] * n_nights

    labels = [f"N{n}" for n in range(n_nights)]
    nights: list[NightProducts] = []
    for n in range(n_nights):
        present = rng.uniform(size=n_stars) >= missing_frac
        core_tile = np.where(present, 0, -1).astype(np.int64)

        if jitter_arcsec > 0:
            jitter_deg = jitter_arcsec / 3600.0
            ra_n = ra_true + rng.normal(0.0, jitter_deg, n_stars) / np.cos(np.radians(dec_true))
            dec_n = dec_true + rng.normal(0.0, jitter_deg, n_stars)
        else:
            ra_n = ra_true.copy()
            dec_n = dec_true.copy()

        bjd_tdb = 2460000.0 + 10.0 * n + np.linspace(0.0, 0.25, frames_per_night)
        airmass = np.clip(1.0 + 0.5 * np.linspace(-1, 1, frames_per_night) ** 2, 1.0, None)
        fwhm = np.full(frames_per_night, 2.5)
        frame_kept = np.ones(frames_per_night, dtype=bool)
        epoch_ok = np.ones((n_stars, frames_per_night), dtype=bool)

        zp_true_n = _zp_truth(n)
        floor_spec = floor_mmag[n]
        if callable(floor_spec):
            sigma_mag_arr = np.asarray(floor_spec(mag_i), dtype=np.float64) / 1000.0
            extra_shift = rng.normal(0.0, 1.0, n_stars) * sigma_mag_arr
        elif floor_spec and floor_spec > 0:
            extra_shift = rng.normal(0.0, floor_spec / 1000.0, n_stars)
        else:
            extra_shift = np.zeros(n_stars)
        var_shift = np.zeros(n_stars)
        this_night_var = variable_idx[variable_night == n]
        if this_night_var.size:
            amp_map = dict(zip(variable_idx.tolist(), variable_amp.tolist(), strict=True))
            var_shift[this_night_var] = [amp_map[i] for i in this_night_var]

        base_mag = mag_i + zp_true_n + extra_shift + var_shift

        lc = np.empty((n_stars, frames_per_night, n_aper), dtype=np.float32)
        lc_err = np.empty((n_stars, frames_per_night, n_aper), dtype=np.float32)
        rms = np.empty((n_stars, n_aper), dtype=np.float64)
        n_epochs = np.full((n_stars, n_aper), frames_per_night, dtype=np.int64)

        for a in range(n_aper):
            sigma_mag = stat_sigma_mag[a % len(stat_sigma_mag)]
            noise = rng.normal(0.0, sigma_mag, (n_stars, frames_per_night))
            mag_obs = base_mag[:, None] + noise
            flux = 10.0 ** (-0.4 * (mag_obs - 25.0))
            lc[:, :, a] = flux.astype(np.float32)
            lc_err[:, :, a] = (flux * sigma_mag / 1.0857).astype(np.float32)
            rms[:, a] = sigma_mag / 1.0857
        if n_unmeasured:
            rms[:n_unmeasured, :] = np.nan

        comparison_mask = np.repeat(comparison_mask_1d[:, None], n_aper, axis=1)
        star_best_aper = np.argmin(rms, axis=1).astype(np.int64)

        nights.append(
            NightProducts(
                label=labels[n],
                directory=Path(f"synthetic_{labels[n]}"),
                ra=ra_n,
                dec=dec_n,
                core_tile=core_tile,
                bjd_tdb=bjd_tdb,
                airmass=airmass,
                fwhm=fwhm,
                frame_kept=frame_kept,
                lc=lc,
                lc_err=lc_err,
                epoch_ok=epoch_ok,
                rms=rms,
                n_epochs=n_epochs,
                comparison_mask=comparison_mask,
                star_best_aper=star_best_aper,
                filter="R",
                object="synthetic",
                aperture_radii=tuple(float(i + 2) for i in range(n_aper)),
            )
        )

    truth = {
        "ra": ra_true,
        "dec": dec_true,
        "mag": mag_i,
        "u": u,
        "v": v,
        "mag0": mag0_truth,
        "zp_coefs": zp_coefs,
        "variable_mask": variable_mask,
        "variable_idx": variable_idx,
        "variable_night": variable_night,
        "comparison_mask": comparison_mask_1d,
        "floor_mmag": floor_mmag,
        "labels": labels,
        "anchor_index": anchor_index,
    }
    return nights, truth


def _truth_zp(truth: dict, n: int) -> np.ndarray:
    """Injected ``Z_n(xi, eta, M)`` truth (mag), evaluated at every star of ``truth``."""
    c = truth["zp_coefs"][n]
    u, v = truth["u"], truth["v"]
    dm = truth["mag"] - truth["mag0"]
    return (
        c[0] + c[1] * u + c[2] * v + c[3] * u**2 + c[4] * u * v + c[5] * v**2
        + c[6] * dm + c[7] * dm**2
    )


def _assert_crossmatch_self_consistent(nights, xmatch: NightCrossMatch) -> None:
    """Structural properties that must hold regardless of jitter-induced misses."""
    assert xmatch.index.shape[0] == len(nights)
    assert xmatch.index.shape[1] == xmatch.ra.shape[0] == xmatch.dec.shape[0]

    # Every core-tile star of every night lands in the global list exactly once
    # (one-to-one), and (since no shuffling happens in the synthetic
    # construction, local index == truth index) every column's non-negative
    # entries agree across nights.
    for n, night in enumerate(nights):
        assigned = xmatch.index[n]
        valid = assigned >= 0
        assert np.unique(assigned[valid]).size == int(valid.sum())
        assert int(valid.sum()) == int(np.count_nonzero(night.core_tile >= 0))

    valid = xmatch.index >= 0
    filled_max = np.where(valid, xmatch.index, -(10**9)).max(axis=0)
    filled_min = np.where(valid, xmatch.index, 10**9).min(axis=0)
    assert np.all(filled_max == filled_min)


def test_crossmatch_structure() -> None:
    # No astrometric jitter: every star present in >= 1 night merges into
    # exactly one global star, so n_global is exactly the union count.
    nights, _truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=1500, frames_per_night=5, n_aper=1, seed=1,
        jitter_arcsec=0.0, missing_frac=0.15,
    )
    xmatch = crossmatch_nights(nights, anchor_index=0, radius_arcsec=1.0)
    assert isinstance(xmatch, NightCrossMatch)
    any_present = np.any([n.core_tile >= 0 for n in nights], axis=0)
    assert xmatch.ra.shape[0] == int(np.count_nonzero(any_present))
    _assert_crossmatch_self_consistent(nights, xmatch)


def test_crossmatch_with_astrometric_jitter() -> None:
    # A little per-night astrometric jitter (well inside the match radius):
    # the global list stays close to the no-jitter union count, and every
    # structural property above still holds.
    nights, _truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=1500, frames_per_night=5, n_aper=1, seed=1,
        jitter_arcsec=0.2, missing_frac=0.15,
    )
    xmatch = crossmatch_nights(nights, anchor_index=0, radius_arcsec=1.0)
    any_present = np.any([n.core_tile >= 0 for n in nights], axis=0)
    n_union = int(np.count_nonzero(any_present))
    assert n_union <= xmatch.ra.shape[0] <= n_union + 20
    _assert_crossmatch_self_consistent(nights, xmatch)


def test_tie_recovery() -> None:
    nights, truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=3000, seed=2, variable_frac=0.03,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    assert anchor_index == truth["anchor_index"]

    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    tie = tie_nights(nights, xmatch, anchor_index, MultiNightSettings())

    assert np.all(tie.coef[anchor_index] == 0.0)

    a = 1  # lowest-noise aperture
    for n in range(3):
        if n == anchor_index:
            continue
        mask = tie.tie_star[n, :, a] & ~tie.rejected[:, a]
        assert np.count_nonzero(mask) > 500
        truth_z = _truth_zp(truth, n)[mask]
        resid = tie.zp[n, mask, a] - truth_z
        rms_mmag = float(np.sqrt(np.mean(resid**2))) * 1000.0
        assert rms_mmag < 1.0, f"night {n} aperture {a}: {rms_mmag:.3f} mmag"

    var_idx = truth["variable_idx"]
    assert var_idx.size > 0
    frac_rejected = float(np.mean(tie.rejected[var_idx, a]))
    assert frac_rejected >= 0.9, frac_rejected


def test_two_night_leave_one_out_unbiased() -> None:
    zp_coefs = [(0.0,) * 8, (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    nights, _truth = _synthetic_tie_scenario(
        n_nights=2, n_stars=2000, seed=3, zp_coefs=zp_coefs,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    tie = tie_nights(nights, xmatch, anchor_index, MultiNightSettings())

    other = 1 - anchor_index
    a = 1
    mask = tie.tie_star[other, :, a] & ~tie.rejected[:, a]
    assert np.count_nonzero(mask) > 500
    mean_z_mmag = float(np.mean(tie.zp[other, mask, a])) * 1000.0
    assert abs(mean_z_mmag - 50.0) < 1.0, mean_z_mmag


def test_floor_recovery_constant() -> None:
    # A constant (magnitude-independent) floor: every bin should recover
    # close to the same injected value.
    # Similar-sized floors across nights: with 3 nights the per-night floor
    # solves an exactly-determined 3x3 linear system per bin (f_n + f_k =
    # V_nk for each pair); when the true floors are very different in size,
    # differencing two much-larger V's to recover the smallest f amplifies
    # per-bin sampling noise well beyond the recovered value itself. Similar
    # sizes avoid that (purely a synthetic-test-design concern, not
    # something to work around in tie_nights itself).
    zp_coefs = [(0.0,) * 8, (0.0,) * 8, (0.0,) * 8]
    floor_mmag = [4.0, 5.0, 6.0]
    nights, _truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=3000, seed=4, zp_coefs=zp_coefs, floor_mmag=floor_mmag,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    tie = tie_nights(nights, xmatch, anchor_index, MultiNightSettings())

    a = 1
    for n in range(3):
        recovered_mmag = tie.floor[n, a, :] * 1000.0
        assert np.all(np.isfinite(recovered_mmag))
        assert np.max(np.abs(recovered_mmag - floor_mmag[n])) < 1.5, (n, recovered_mmag)
        chi2 = float(tie.chi2_after[n, a])
        assert 0.8 <= chi2 <= 1.25, (n, chi2)
        chi2_ho = float(tie.chi2_holdout[n, a])
        assert 0.8 <= chi2_ho <= 1.25, (n, chi2_ho)


def test_floor_recovery_magnitude_dependent() -> None:
    # A floor that rises with magnitude (2 mmag bright -> 10 mmag faint,
    # linear over the synthetic mag_range 13-19): recovered per-bin values
    # must track the injected function, and the held-out chi2 (fit on
    # even-global-id tie stars, evaluated on odd ones and vice versa) must
    # be consistent -- this is the real, non-tautological check.
    def truth_floor_mmag(mag: np.ndarray) -> np.ndarray:
        return 2.0 + (10.0 - 2.0) * (mag - 13.0) / (19.0 - 13.0)

    zp_coefs = [(0.0,) * 8, (0.0,) * 8, (0.0,) * 8]
    floor_mmag = [truth_floor_mmag, truth_floor_mmag, truth_floor_mmag]
    nights, _truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=12000, seed=13, zp_coefs=zp_coefs, floor_mmag=floor_mmag,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    tie = tie_nights(nights, xmatch, anchor_index, MultiNightSettings())

    a = 1
    centres = tie.floor_mag_centres[a]
    assert np.all(np.isfinite(centres))
    # NightTie's "M" is -2.5*log10(flux) with the synthetic flux generated
    # against a 25-mag zero point (see _synthetic_tie_scenario), i.e.
    # M == mag_i - 25; truth_floor_mmag is expressed in mag_i.
    expected_mmag = truth_floor_mmag(centres + 25.0)
    for n in range(3):
        recovered_mmag = tie.floor[n, a, :] * 1000.0
        assert np.all(np.isfinite(recovered_mmag))
        max_dev = np.max(np.abs(recovered_mmag - expected_mmag))
        assert max_dev < 1.5, (n, recovered_mmag, expected_mmag)
        # floor_at's interpolated form is forced non-decreasing bright -> faint
        # (the raw per-bin floor above is not -- a single noisy bin may dip).
        smoothed_mmag = tie.floor_at(n, a, centres) * 1000.0
        assert np.all(np.diff(smoothed_mmag) >= -1e-9)
        chi2_ho = float(tie.chi2_holdout[n, a])
        assert 0.8 <= chi2_ho <= 1.25, (n, chi2_ho)

    # floor_at interpolates between bins and stays constant beyond the ends
    # (comparing against its own value *at* the end centres, since the
    # smoothing accumulate-max can differ from the raw, possibly-dipping
    # per-bin value there).
    at_ends = tie.floor_at(0, a, np.array([centres[0], centres[-1]]))
    bright_extra = tie.floor_at(0, a, np.array([centres[0] - 5.0]))
    faint_extra = tie.floor_at(0, a, np.array([centres[-1] + 5.0]))
    assert np.isclose(bright_extra[0], at_ends[0])
    assert np.isclose(faint_extra[0], at_ends[1])


def test_min_tie_stars_violation_raises() -> None:
    nights, _truth = _synthetic_tie_scenario(
        n_nights=2, n_stars=300, seed=5, comparison_frac=0.6,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    settings = MultiNightSettings(min_tie_stars=10_000)
    with pytest.raises(MultiNightError):
        tie_nights(nights, xmatch, anchor_index, settings)


def test_incompatible_filters_raises() -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=2, n_stars=200, frames_per_night=5, seed=6)
    nights[1].filter = "V"
    with pytest.raises(MultiNightError):
        check_compatible(nights)


def test_incompatible_aperture_counts_raises() -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=2, n_stars=200, frames_per_night=5, seed=7)
    nights[1] = _synthetic_tie_scenario(
        n_nights=1, n_stars=200, frames_per_night=5, n_aper=2, seed=8,
    )[0][0]
    with pytest.raises(MultiNightError):
        check_compatible(nights)


def test_lightcurves() -> None:
    nights, truth = _synthetic_tie_scenario(
        n_nights=3, n_stars=1500, seed=9, n_unmeasured=5,
    )
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    settings = MultiNightSettings()
    tie = tie_nights(nights, xmatch, anchor_index, settings)
    mlc = build_multinight_lightcurves(nights, xmatch, tie, settings)

    # Reserved always-unmeasured stars get no usable aperture.
    assert np.all(mlc.aperture[:5] == -1)
    assert np.all(~np.isfinite(mlc.mag[:5]))

    # Every other star has a valid (measured) chosen aperture.
    assert np.all(mlc.aperture[5:] >= 0)

    # Calibrated nightly means of constant (non-variable, comparison) stars agree
    # across nights once the injected zero point has been removed.
    good = truth["comparison_mask"].copy()
    good[:5] = False
    n0, n1 = 0, 1
    both = good & np.isfinite(mlc.night_mean_mag[n0]) & np.isfinite(mlc.night_mean_mag[n1])
    assert np.count_nonzero(both) > 200
    diff_mmag = np.abs(mlc.night_mean_mag[n0, both] - mlc.night_mean_mag[n1, both]) * 1000.0
    assert float(np.median(diff_mmag)) < 1.0

    # flux_norm is per-night median-normalised to 1 for every measured star.
    frame_offsets = np.concatenate([[0], np.cumsum([n.n_frames for n in nights])])
    for n in range(3):
        lo, hi = int(frame_offsets[n]), int(frame_offsets[n + 1])
        block = mlc.flux_norm[5:, lo:hi]
        finite_rows = np.isfinite(block).any(axis=1)
        med = nanmedian_quiet(block[finite_rows], axis=1)
        np.testing.assert_allclose(med, 1.0, atol=1e-4)


def test_settings_aperture_override() -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=2, n_stars=800, seed=10)
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    settings = replace(MultiNightSettings(floor_min_bin_stars=20), aperture=0)
    tie = tie_nights(nights, xmatch, anchor_index, settings)
    mlc = build_multinight_lightcurves(nights, xmatch, tie, settings)
    assert np.all(mlc.aperture == 0)


def test_save_load_round_trip(tmp_path) -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=2, n_stars=500, seed=11)
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    settings = MultiNightSettings(floor_min_bin_stars=20)
    tie = tie_nights(nights, xmatch, anchor_index, settings)
    mlc = build_multinight_lightcurves(nights, xmatch, tie, settings)

    from relphot.config import Settings

    full_settings = replace(Settings(), multinight=settings)
    out_path = tmp_path / "multinight.npz"
    save_multinight(out_path, nights, xmatch, tie, mlc, full_settings)
    xmatch2, tie2, mlc2, night_info, settings2 = load_multinight(out_path)

    np.testing.assert_array_equal(xmatch.ra, xmatch2.ra)
    np.testing.assert_array_equal(xmatch.dec, xmatch2.dec)
    np.testing.assert_array_equal(xmatch.index, xmatch2.index)
    np.testing.assert_array_equal(xmatch.n_matched, xmatch2.n_matched)
    assert xmatch.labels == xmatch2.labels

    assert tie.anchor_index == tie2.anchor_index
    assert tie.basis_terms == tie2.basis_terms
    assert tie.seeing_basis_terms == tie2.seeing_basis_terms
    for field_name in (
        "coef", "xi", "eta", "mag0", "zp", "mean_mag", "night_mag", "night_mag_err",
        "tie_star", "rejected", "floor_mag_centres", "floor", "n_tie", "resid_mad",
        "resid_mad_bright", "chi2_after", "chi2_holdout", "chi2_holdout_bins", "n_iter",
        "seeing_coef", "seeing_mag0", "seeing_crowd0", "night_fwhm", "crowding", "loose",
    ):
        np.testing.assert_array_equal(getattr(tie, field_name), getattr(tie2, field_name))

    for field_name in (
        "night_of_frame", "frame_in_night", "bjd_tdb", "airmass", "fwhm", "aperture",
        "mag", "mag_err", "flux_norm", "flux_norm_err", "night_mean_mag",
        "night_mean_err", "mean_mag", "n_nights",
    ):
        np.testing.assert_array_equal(getattr(mlc, field_name), getattr(mlc2, field_name))

    assert len(night_info) == len(nights)
    assert night_info[0]["label"] == nights[0].label
    assert settings_to_dict(settings2) == settings_to_dict(full_settings)


def test_save_tie_report_and_tables(tmp_path) -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=2, n_stars=500, seed=12)
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)
    settings = MultiNightSettings(floor_min_bin_stars=20)
    tie = tie_nights(nights, xmatch, anchor_index, settings)
    mlc = build_multinight_lightcurves(nights, xmatch, tie, settings)

    tie_csv = tmp_path / "tie.csv"
    save_tie_report(tie_csv, tie)
    assert tie_csv.is_file()
    import csv as csv_mod

    with tie_csv.open() as handle:
        rows = list(csv_mod.DictReader(handle))
    assert len(rows) == 2 * tie.coef.shape[1]
    assert "xi" in rows[0]
    assert "floor_bright_mmag" in rows[0]
    assert "chi2_holdout" in rows[0]

    floor_csv = tmp_path / "floor.csv"
    save_floor_report(floor_csv, tie)
    assert floor_csv.is_file()
    with floor_csv.open() as handle:
        floor_rows = list(csv_mod.DictReader(handle))
    assert len(floor_rows) == 2 * tie.coef.shape[1] * settings.floor_n_bins
    assert "mag_centre" in floor_rows[0]

    stars_path, lc_path = save_multinight_tables(tmp_path / "mn", xmatch, tie, mlc, "auto")
    assert stars_path.is_file()
    assert lc_path.is_file()

    from astropy.table import Table

    stars_table = Table.read(stars_path)
    assert "mag_N0" in stars_table.colnames
    assert "star_id_N1" in stars_table.colnames
    assert len(stars_table) == xmatch.ra.shape[0]

    lc_table = Table.read(lc_path)
    assert "night_label" in lc_table.colnames
    assert len(lc_table) > 0


def _synthetic_seeing_scenario(
    *,
    n_nights: int = 3,
    n_stars: int = 4000,
    frames_per_night: int = 60,
    seed: int = 42,
    fwhm_by_night: list[float] | None = None,
    floor_mmag_true: list[float] | None = None,
    b_mag_true: float = 0.006,
    b_crowd_true: float = 0.05,
    anchor_index: int = 0,
    comparison_frac: float = 0.6,
    mag_range: tuple[float, float] = (13.0, 19.0),
    field_deg: float = 1.0,
    ra0: float = 150.0,
    dec0: float = -30.0,
    sigma_mag: float = 0.006,
) -> tuple[list[NightProducts], dict]:
    """A 1-aperture, no-jitter synthetic scenario with an injected pooled seeing term.

    Every star gets an injected shift ``beta_true(mag, crowding) * (F_n -
    F_anchor)`` -- linear in centred mean magnitude (coefficient
    ``b_mag_true``) and in centred crowding (``b_crowd_true``,
    :func:`~relphot.decorrelate.compute_crowding`, computed once directly on
    the (identical, unjittered) star positions) -- plus independent
    per-night Gaussian floor noise ``floor_mmag_true[n]``. No spatial/mag
    zero-point offset is injected (``Z_n`` truth is 0 everywhere) so every
    non-anchor night's residual against the anchor isolates the seeing
    term + floor.
    """
    rng = np.random.default_rng(seed)
    fwhm_by_night = fwhm_by_night or [2.13, 2.42, 2.97][:n_nights]
    floor_mmag_true = floor_mmag_true or [3.0, 4.0, 3.5][:n_nights]
    assert len(fwhm_by_night) == n_nights
    assert len(floor_mmag_true) == n_nights

    ra_true = ra0 + rng.uniform(-field_deg / 2, field_deg / 2, n_stars) / np.cos(np.radians(dec0))
    dec_true = dec0 + rng.uniform(-field_deg / 2, field_deg / 2, n_stars)
    mag_i = rng.uniform(mag_range[0], mag_range[1], n_stars)
    mag0_truth = float(np.median(mag_i))
    mag_c_true = mag_i - mag0_truth

    # compute_crowding only reads .ra/.dec/.n_stars -- reuse it directly on a
    # bare stand-in rather than duplicating its nearest-neighbour formula.
    crowd_true = compute_crowding(SimpleNamespace(ra=ra_true, dec=dec_true, n_stars=n_stars))
    crowd0_truth = float(np.median(crowd_true))
    crowd_c_true = crowd_true - crowd0_truth

    beta_true = b_mag_true * mag_c_true + b_crowd_true * crowd_c_true
    comparison_mask_1d = rng.uniform(size=n_stars) < comparison_frac
    labels = [f"N{n}" for n in range(n_nights)]

    nights: list[NightProducts] = []
    for n in range(n_nights):
        core_tile = np.zeros(n_stars, dtype=np.int64)
        bjd_tdb = 2460000.0 + 10.0 * n + np.linspace(0.0, 0.25, frames_per_night)
        airmass = np.clip(1.0 + 0.5 * np.linspace(-1, 1, frames_per_night) ** 2, 1.0, None)
        fwhm = np.full(frames_per_night, fwhm_by_night[n])
        frame_kept = np.ones(frames_per_night, dtype=bool)
        epoch_ok = np.ones((n_stars, frames_per_night), dtype=bool)

        d_fwhm_n = fwhm_by_night[n] - fwhm_by_night[anchor_index]
        seeing_shift = beta_true * d_fwhm_n
        floor_shift = rng.normal(0.0, floor_mmag_true[n] / 1000.0, n_stars)
        base_mag = mag_i + seeing_shift + floor_shift

        noise = rng.normal(0.0, sigma_mag, (n_stars, frames_per_night))
        mag_obs = base_mag[:, None] + noise
        flux = 10.0 ** (-0.4 * (mag_obs - 25.0))
        lc = flux.astype(np.float32)[:, :, None]
        lc_err = (flux * sigma_mag / 1.0857).astype(np.float32)[:, :, None]
        rms = np.full((n_stars, 1), sigma_mag / 1.0857, dtype=np.float64)
        n_epochs = np.full((n_stars, 1), frames_per_night, dtype=np.int64)
        comparison_mask = comparison_mask_1d[:, None]
        star_best_aper = np.zeros(n_stars, dtype=np.int64)

        nights.append(
            NightProducts(
                label=labels[n], directory=Path(f"synthetic_seeing_{labels[n]}"),
                ra=ra_true.copy(), dec=dec_true.copy(), core_tile=core_tile,
                bjd_tdb=bjd_tdb, airmass=airmass, fwhm=fwhm, frame_kept=frame_kept,
                lc=lc, lc_err=lc_err, epoch_ok=epoch_ok, rms=rms, n_epochs=n_epochs,
                comparison_mask=comparison_mask, star_best_aper=star_best_aper,
                filter="R", object="synthetic", aperture_radii=(3.0,),
            )
        )

    xmatch = NightCrossMatch(
        ra=ra_true, dec=dec_true,
        index=np.tile(np.arange(n_stars), (n_nights, 1)),
        labels=tuple(labels), n_matched=np.zeros(n_nights, dtype=np.int64),
    )
    truth = {
        "fwhm_by_night": fwhm_by_night,
        "floor_mmag_true": floor_mmag_true,
        "b_mag_true": b_mag_true,
        "b_crowd_true": b_crowd_true,
        "anchor_index": anchor_index,
    }
    return nights, xmatch, truth


def test_seeing_term_recovers_beta_and_floor_where_old_model_clips(caplog) -> None:
    # Reproduces the real T80S failure this feature targets: an unmodelled
    # star-dependent, seeing-dependent term makes the per-night calibration
    # floor's v_nk = f_n + f_k additivity assumption fail (v_04_06 >
    # v_04_05 + v_05_06 in the real data), so NNLS clips some night's floor
    # to exactly 0 in every bin. use_seeing_term=True must remove that
    # non-additivity (recovering the true, non-zero, per-night floor) by
    # absorbing the seeing x crowding term into the pooled beta surface.
    nights, xmatch, truth = _synthetic_seeing_scenario()
    anchor_index = truth["anchor_index"]
    settings_common = MultiNightSettings(floor_min_bin_stars=100)

    caplog.set_level("WARNING", logger="relphot.multinight")
    caplog.clear()
    settings_off = replace(settings_common, use_seeing_term=False)
    tie_off = tie_nights(nights, xmatch, anchor_index, settings_off)
    clip_messages_off = [r.message for r in caplog.records if "NNLS clipped" in r.message]
    assert clip_messages_off, "expected the unmodelled seeing term to force an NNLS floor clip"
    # Some night's floor is exactly 0 in (at least most of) every bin.
    assert np.any(np.all(tie_off.floor[:, 0, :] == 0.0, axis=1))
    assert tie_off.seeing_basis_terms == ()
    assert tie_off.seeing_coef.shape == (1, 0)

    caplog.clear()
    settings_on = replace(settings_common, use_seeing_term=True)
    tie_on = tie_nights(nights, xmatch, anchor_index, settings_on)
    clip_messages_on = [r.message for r in caplog.records if "NNLS clipped" in r.message]
    assert not clip_messages_on, clip_messages_on

    # No night's recovered floor collapses to (near) 0, and every night's
    # floor is close to its true, injected value in every bin.
    floor_on_mmag = tie_on.floor[:, 0, :] * 1000.0
    assert np.all(floor_on_mmag > 0.5)
    for n, true_mmag in enumerate(truth["floor_mmag_true"]):
        assert np.max(np.abs(floor_on_mmag[n] - true_mmag)) < 3.0, (n, floor_on_mmag[n])
    assert np.all((tie_on.chi2_after[:, 0] >= 0.7) & (tie_on.chi2_after[:, 0] <= 1.3))
    assert np.all((tie_on.chi2_holdout[:, 0] >= 0.6) & (tie_on.chi2_holdout[:, 0] <= 1.4))

    # The pooled beta surface's crowding coefficient is recovered; its
    # magnitude coefficient is not asserted -- with mag_degree >= 1 in the
    # per-night poly(M) (the default), a magnitude-only component of beta is
    # exactly degenerate with that night's own poly(M) coefficient for any
    # number of nights (see the module docstring), so it is absorbed there
    # instead, by design, not a recovery failure.
    cc_index = tie_on.seeing_basis_terms.index("cc")
    cc_coef = tie_on.seeing_coef[0, cc_index]
    assert abs(cc_coef - truth["b_crowd_true"]) < 0.02, tie_on.seeing_coef[0]

    np.testing.assert_array_equal(tie_on.night_fwhm, np.asarray(truth["fwhm_by_night"]))


def test_seeing_term_off_is_inert_without_true_seeing_signal() -> None:
    # _synthetic_tie_scenario's nights all share the same constant FWHM
    # (F_n - F_anchor == 0 for every non-anchor night), so the pooled
    # seeing-fit step contributes nothing to add (every night is skipped in
    # _fit_pooled_seeing) regardless of the toggle: on and off must be
    # bit-for-bit identical here.
    nights, _truth = _synthetic_tie_scenario(n_nights=3, n_stars=1500, seed=20)
    nights = check_compatible(nights)
    anchor_index = resolve_anchor_index(nights, "auto")
    xmatch = crossmatch_nights(nights, anchor_index, radius_arcsec=1.0)

    tie_off = tie_nights(nights, xmatch, anchor_index, MultiNightSettings(use_seeing_term=False))
    tie_on = tie_nights(nights, xmatch, anchor_index, MultiNightSettings(use_seeing_term=True))

    np.testing.assert_array_equal(tie_off.zp, tie_on.zp)
    np.testing.assert_array_equal(tie_off.coef, tie_on.coef)
    np.testing.assert_array_equal(tie_off.floor, tie_on.floor)
    np.testing.assert_array_equal(tie_off.chi2_after, tie_on.chi2_after)
    assert tie_off.seeing_basis_terms == ()
    assert tie_on.seeing_basis_terms != ()
    assert np.all(tie_on.seeing_coef == 0.0)


def test_seeing_term_two_nights_degeneracy_warning(caplog) -> None:
    nights, xmatch, truth = _synthetic_seeing_scenario(
        n_nights=2, fwhm_by_night=[2.13, 2.42], floor_mmag_true=[3.0, 4.0], seed=21,
    )
    anchor_index = truth["anchor_index"]

    caplog.set_level("WARNING", logger="relphot.multinight")
    caplog.clear()
    tie_nights(nights, xmatch, anchor_index, MultiNightSettings(seeing_crowding_degree=0))
    assert any("degenerate" in r.message for r in caplog.records)

    caplog.clear()
    tie_two = tie_nights(nights, xmatch, anchor_index, MultiNightSettings())
    assert not any("degenerate" in r.message for r in caplog.records)
    # Still runs and produces a usable (finite) crowding-surface coefficient.
    cc_index = tie_two.seeing_basis_terms.index("cc")
    assert np.isfinite(tie_two.seeing_coef[0, cc_index])


# --------------------------------------------------------------------------
# loose nights: tied to the fixed core frame, never moving the core tie
# --------------------------------------------------------------------------

_LOOSE_FLOOR_MMAG = 15.0
#: linear-only truth for the loose night (the loose surface is degree 1 / 1); no poly2 terms
_LOOSE_ZP_COEFS = (0.02, 0.03, -0.02, 0.0, 0.0, 0.0, 0.002, 0.0)


def _loose_scenario(seed: int = 21, missing_frac: float = 0.1):
    """Three core nights and a fourth, noisier, night with a linear zero-point surface."""
    zp_coefs = [(0.0,) * 8]
    for n in (1, 2):
        rng = np.random.default_rng(seed * 100 + n)
        cap = np.array([0.010, 0.020, 0.020, 0.040, 0.040, 0.040, 0.003, 0.0011])
        zp_coefs.append(tuple(rng.uniform(0.3, 1.0, 8) * cap * rng.choice([-1.0, 1.0], 8)))
    zp_coefs.append(_LOOSE_ZP_COEFS)
    nights, truth = _synthetic_tie_scenario(
        n_nights=4, n_stars=4000, seed=seed, zp_coefs=zp_coefs,
        floor_mmag=[3.0, 3.0, 3.0, _LOOSE_FLOOR_MMAG], missing_frac=missing_frac,
    )
    return nights, truth


def _tie_core_and_loose(nights, settings, n_core: int = 3):
    core, loose = split_loose_nights(nights, (nights[-1].label,))
    assert len(core) == n_core
    anchor_index = resolve_anchor_index(core, "auto")
    xmatch = crossmatch_nights(core + loose, anchor_index, 1.0)
    tie_core = tie_nights(core, core_crossmatch(xmatch, n_core), anchor_index, settings)
    tie = tie_loose_nights(core, loose, xmatch, tie_core, settings)
    return core, loose, xmatch, tie_core, tie


def test_loose_night_leaves_the_core_tie_bit_identical() -> None:
    nights, _truth = _loose_scenario()
    settings = MultiNightSettings()
    core, loose, xmatch, _tie_core, tie = _tie_core_and_loose(nights, settings)

    # the core alone, by the plain path a run without a loose night takes
    anchor_index = resolve_anchor_index(core, "auto")
    xmatch_alone = crossmatch_nights(core, anchor_index, 1.0)
    tie_alone = tie_nights(core, xmatch_alone, anchor_index, settings)
    mlc_alone = build_multinight_lightcurves(core, xmatch_alone, tie_alone, settings)
    mlc = build_multinight_lightcurves(core + loose, xmatch, tie, settings)

    n_g = xmatch_alone.ra.shape[0]
    assert xmatch.ra.shape[0] >= n_g
    np.testing.assert_array_equal(xmatch.index[:3, :n_g], xmatch_alone.index)
    np.testing.assert_array_equal(tie.loose, [False, False, False, True])
    assert tie.anchor_index == tie_alone.anchor_index

    for name in ("zp", "night_mag", "night_mag_err", "tie_star"):
        np.testing.assert_array_equal(
            getattr(tie, name)[:3, :n_g], getattr(tie_alone, name), err_msg=name
        )
    for name in ("mean_mag", "rejected", "xi", "eta", "crowding"):
        np.testing.assert_array_equal(
            getattr(tie, name)[:n_g], getattr(tie_alone, name), err_msg=name
        )
    for name in (
        "coef", "floor", "n_tie", "resid_mad", "resid_mad_bright", "chi2_after",
        "chi2_holdout", "chi2_holdout_bins", "night_fwhm",
    ):
        np.testing.assert_array_equal(
            getattr(tie, name)[:3], getattr(tie_alone, name), err_msg=name
        )
    np.testing.assert_array_equal(tie.floor_mag_centres, tie_alone.floor_mag_centres)
    np.testing.assert_array_equal(tie.seeing_coef, tie_alone.seeing_coef)

    np.testing.assert_array_equal(mlc.aperture[:n_g], mlc_alone.aperture)
    np.testing.assert_array_equal(mlc.night_mean_mag[:3, :n_g], mlc_alone.night_mean_mag)
    np.testing.assert_array_equal(mlc.night_mean_err[:3, :n_g], mlc_alone.night_mean_err)
    np.testing.assert_array_equal(mlc.mean_mag[:n_g], mlc_alone.mean_mag)
    n_epochs_core = mlc_alone.bjd_tdb.shape[0]
    np.testing.assert_array_equal(mlc.mag[:n_g, :n_epochs_core], mlc_alone.mag)
    # a star only the loose night contains has no aperture, hence no tied output
    assert np.all(mlc.aperture[n_g:] == -1)


def test_loose_night_floor_is_measured_and_inflates_its_errors() -> None:
    nights, truth = _loose_scenario()
    settings = MultiNightSettings()
    core, loose, xmatch, _tie_core, tie = _tie_core_and_loose(nights, settings)
    mlc = build_multinight_lightcurves(core + loose, xmatch, tie, settings)

    a = 1
    floor_mmag = tie.floor[3, a] * 1000.0
    assert np.all(np.isfinite(floor_mmag))
    assert np.max(np.abs(floor_mmag - _LOOSE_FLOOR_MMAG)) < 3.0, floor_mmag
    assert 0.8 <= float(tie.chi2_after[3, a]) <= 1.25
    assert 0.8 <= float(tie.chi2_holdout[3, a]) <= 1.25
    core_floor_mmag = tie.floor[:3, a] * 1000.0
    assert np.all(core_floor_mmag < 6.0)

    # the surface: its own 4 terms in the core basis, recovered to ~1 mmag on its fit stars
    coef = tie.coef[3, a]
    used = [i for i, t in enumerate(tie.basis_terms) if t in ("1", "xi", "eta", "dm")]
    assert np.all(coef[[i for i in range(coef.size) if i not in used]] == 0.0)
    g_fit = np.nonzero(tie.tie_star[3, :, a])[0]
    truth_i = xmatch.index[3, g_fit]  # a synthetic night's local star index is the truth index
    resid = tie.zp[3, g_fit, a] - _truth_zp(truth, 3)[truth_i]
    assert float(np.sqrt(np.mean(resid**2))) * 1000.0 < 2.0
    assert int(tie.n_tie[3, a]) == int(np.count_nonzero(tie.tie_star[3, :, a]))

    # its per-night errors carry the measured floor; the core nights' do not
    sel = (mlc.aperture == a) & np.isfinite(mlc.night_mean_err[3])
    assert np.count_nonzero(sel) > 1000
    assert float(np.median(mlc.night_mean_err[3, sel])) > 0.010
    core_sel = (mlc.aperture == a) & np.isfinite(mlc.night_mean_err[0])
    assert float(np.median(mlc.night_mean_err[0, core_sel])) < 0.006

    # evaluate_zero_point reproduces the stored zero point; no seeing term, no crowding needed
    g = np.nonzero(np.isfinite(tie.zp[3, :, a]))[0][:200]
    z = evaluate_zero_point(tie, 3, a, tie.xi[g], tie.eta[g], tie.mean_mag[g, a])
    np.testing.assert_allclose(z, tie.zp[3, g, a], atol=1e-12)


def test_loose_night_needs_enough_fit_stars() -> None:
    nights, _truth = _loose_scenario()
    core, loose = split_loose_nights(nights, (nights[-1].label,))
    anchor_index = resolve_anchor_index(core, "auto")
    xmatch = crossmatch_nights(core + loose, anchor_index, 1.0)
    settings = MultiNightSettings(min_tie_stars=50)
    tie_core = tie_nights(core, core_crossmatch(xmatch, 3), anchor_index, settings)
    with pytest.raises(MultiNightError, match="fit stars for loose night"):
        tie_loose_nights(core, loose, xmatch, tie_core, replace(settings, min_tie_stars=10**6))


def test_split_loose_nights_validation_and_settings() -> None:
    nights, _truth = _synthetic_tie_scenario(n_nights=3, n_stars=50, frames_per_night=5, n_aper=1)
    core, loose = split_loose_nights(nights, ("N2",))
    assert [n.label for n in core] == ["N0", "N1"] and [n.label for n in loose] == ["N2"]
    core, loose = split_loose_nights(nights, ())
    assert len(core) == 3 and loose == []
    with pytest.raises(MultiNightError, match="not found"):
        split_loose_nights(nights, ("nope",))
    with pytest.raises(MultiNightError, match="at least 2 core"):
        split_loose_nights(nights, ("N1", "N2"))

    with pytest.raises(ConfigError, match="cannot be a loose night"):
        MultiNightSettings(anchor="N1", loose_nights=("N1",))
    with pytest.raises(ConfigError, match="duplicate"):
        MultiNightSettings(loose_nights=("N1", "N1"))
    with pytest.raises(ConfigError, match="loose_spatial_degree"):
        MultiNightSettings(loose_spatial_degree=3)
    with pytest.raises(ConfigError, match="loose_mag_degree"):
        MultiNightSettings(loose_mag_degree=-1)
    with pytest.raises(ConfigError, match="loose_internight_support_p"):
        MultiNightSettings(loose_internight_support_p=0.0)
    default = MultiNightSettings()
    assert default.loose_nights == () and default.loose_internight_support_p == 1e-2
    assert (default.loose_spatial_degree, default.loose_mag_degree) == (1, 1)


def test_save_load_round_trip_with_loose_and_old_file_without_it(tmp_path) -> None:
    nights, _truth = _loose_scenario(seed=22)
    settings = MultiNightSettings()
    core, loose, xmatch, _tie_core, tie = _tie_core_and_loose(nights, settings)
    mlc = build_multinight_lightcurves(core + loose, xmatch, tie, settings)

    from relphot.config import Settings

    full_settings = replace(Settings(), multinight=replace(settings, loose_nights=("N3",)))
    path = tmp_path / "loose.npz"
    save_multinight(path, core + loose, xmatch, tie, mlc, full_settings)
    _xm, tie2, _mlc, _info, settings2 = load_multinight(path)
    np.testing.assert_array_equal(tie2.loose, [False, False, False, True])
    assert settings2.multinight.loose_nights == ("N3",)

    # a file written before `loose` existed has no tie_loose key: all nights are core
    with np.load(path, allow_pickle=False) as data:
        old = {k: data[k] for k in data.files if k != "tie_loose"}
    old_path = tmp_path / "old.npz"
    np.savez(old_path, **old)
    _xm, tie_old, _mlc, _info, _settings = load_multinight(old_path)
    assert tie_old.loose.dtype == bool and tie_old.loose.shape == (4,)
    assert not tie_old.loose.any()
    np.testing.assert_array_equal(tie_old.zp, tie.zp)
