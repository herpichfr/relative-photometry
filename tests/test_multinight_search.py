"""Unit tests for :mod:`relphot.multinight_search` (Unit B: cross-night search)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relphot.config import MultiNightSettings
from relphot.multinight import MultiNightLightCurves, NightCrossMatch, NightTie
from relphot.multinight_search import (
    bls_search,
    combined_periodogram,
    commensurate_periods,
    compute_internight_variability,
    cross_reference_nights,
    period_compatibility,
    run_multisearch,
)


def _toy_mlc(
    night_mean_mag: np.ndarray, night_mean_err: np.ndarray, labels: tuple[str, ...] | None = None,
) -> MultiNightLightCurves:
    """A minimal :class:`MultiNightLightCurves` with only the fields B2 needs."""
    n_nights, n_global = night_mean_mag.shape
    if labels is None:
        labels = tuple(f"N{n}" for n in range(n_nights))
    n_nights_arr = np.count_nonzero(np.isfinite(night_mean_mag), axis=0).astype(np.int64)
    return MultiNightLightCurves(
        labels=labels,
        night_of_frame=np.zeros(0, dtype=np.int64),
        frame_in_night=np.zeros(0, dtype=np.int64),
        bjd_tdb=np.zeros(0, dtype=np.float64),
        airmass=np.zeros(0, dtype=np.float64),
        fwhm=np.zeros(0, dtype=np.float64),
        aperture=np.zeros(n_global, dtype=np.int64),
        mag=np.zeros((n_global, 0), dtype=np.float32),
        mag_err=np.zeros((n_global, 0), dtype=np.float32),
        flux_norm=np.zeros((n_global, 0), dtype=np.float32),
        flux_norm_err=np.zeros((n_global, 0), dtype=np.float32),
        night_mean_mag=night_mean_mag,
        night_mean_err=night_mean_err,
        mean_mag=nanmean_ignore(night_mean_mag),
        n_nights=n_nights_arr,
    )


def nanmean_ignore(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0)


def test_internight_variability() -> None:
    rng = np.random.default_rng(0)
    n_nights, n_stars = 3, 2001
    base_mag = rng.uniform(13.0, 19.0, n_stars)
    err = 0.005
    night_mean_mag = base_mag[None, :] + rng.normal(0.0, err, (n_nights, n_stars))
    night_mean_err = np.full((n_nights, n_stars), err)

    # Star 0: a real +50 mmag offset in night 1 only.
    night_mean_mag[1, 0] += 0.05

    mlc = _toy_mlc(night_mean_mag, night_mean_err)
    settings = MultiNightSettings()
    result = compute_internight_variability(mlc, settings)

    assert result["candidate"][0]
    false_positive_rate = int(np.count_nonzero(result["candidate"][1:]))
    assert false_positive_rate <= 5, false_positive_rate


def test_period_compatibility_exclusion_support_unconstrained() -> None:
    settings = MultiNightSettings()
    t0 = 0.0
    depth = 0.01
    sigma_depth = 0.001
    duration_days = 0.05  # 1.2 h

    # Night 0 (the reference event's own night): a few points around t0.
    t_ref = t0 + np.linspace(-0.1, 0.1, 15)
    flux_ref = np.ones_like(t_ref)
    in_dip = np.abs(t_ref - t0) < duration_days / 2.0
    flux_ref[in_dip] -= depth
    err_ref = np.full_like(t_ref, 0.0008)
    night_id_ref = np.zeros_like(t_ref, dtype=np.int64)

    # Night 1: flat, densely sampled only over [t0 + 1.0, t0 + 1.3] days (a
    # window wide enough that the transit box below is a small fraction of
    # it, so the whole-night median is still a fair "out of transit" level).
    t_other = t0 + np.linspace(1.0, 1.3, 120)
    flux_other = np.ones_like(t_other) + np.random.default_rng(1).normal(0.0, 0.0005, t_other.size)
    err_other = np.full_like(t_other, 0.0008)
    night_id_other = np.ones_like(t_other, dtype=np.int64)

    t = np.concatenate([t_ref, t_other])
    flux = np.concatenate([flux_ref, flux_other])
    err = np.concatenate([err_ref, err_other])
    night_id = np.concatenate([night_id_ref, night_id_other])

    periods_excl = np.array([1.05, 0.525])
    pc = period_compatibility(
        t, flux, err, night_id, t0, depth, sigma_depth, duration_days, periods_excl, settings,
    )
    assert np.all(pc.status == -1), pc.status

    # A period whose predicted transit lands well outside the [1.0, 1.1] window
    # (night 1 has no data there at all) is unconstrained.
    pc_gap = period_compatibility(
        t, flux, err, night_id, t0, depth, sigma_depth, duration_days, np.array([2.0]), settings,
    )
    assert pc_gap.status[0] == 0

    # Now inject the same dip into night 1 at the predicted time (P=1.05, k=1).
    flux_other_dip = flux_other.copy()
    in_dip_other = np.abs(t_other - (t0 + 1.05)) < duration_days / 2.0
    assert np.any(in_dip_other)
    flux_other_dip[in_dip_other] -= depth
    flux_supported = np.concatenate([flux_ref, flux_other_dip])

    pc_supported = period_compatibility(
        t, flux_supported, err, night_id, t0, depth, sigma_depth, duration_days,
        np.array([1.05]), settings,
    )
    assert pc_supported.status[0] == 1


def test_commensurate_periods() -> None:
    events = [
        {
            "night": "N0", "tc": 0.0, "depth": 0.01, "sigma_depth": 0.001,
            "duration_hours": 1.2, "snr": 10.0,
        },
        {
            "night": "N1", "tc": 2.1, "depth": 0.0102, "sigma_depth": 0.0011,
            "duration_hours": 1.2, "snr": 9.0,
        },
    ]
    result = commensurate_periods(events, period_min_days=0.2, depth_consistency_sigma=3.0)
    assert len(result) == 1
    entry = result[0]
    assert entry["depth_consistent"]
    assert np.isclose(entry["dt_days"], 2.1)
    assert 2.1 in [pytest.approx(p) for p in entry["candidate_periods"]] or any(
        abs(p - 2.1) < 1e-9 for p in entry["candidate_periods"]
    )
    assert any(abs(p - 1.05) < 1e-9 for p in entry["candidate_periods"])


def _toy_tie(n_nights: int, n_aper: int = 1, labels: tuple[str, ...] | None = None) -> NightTie:
    """A minimal :class:`NightTie` with a zero calibration floor everywhere.

    Used only for B4's error term.
    """
    labels = labels or tuple(f"N{n}" for n in range(n_nights))
    n_bins = 2
    return NightTie(
        labels=labels,
        anchor_index=0,
        coef=np.zeros((n_nights, n_aper, 1)),
        basis_terms=("1",),
        xi=np.zeros(0),
        eta=np.zeros(0),
        centre_radec=(0.0, 0.0),
        scale_deg=1.0,
        mag0=np.zeros(n_aper),
        zp=np.zeros((n_nights, 0, n_aper)),
        mean_mag=np.zeros((0, n_aper)),
        night_mag=np.zeros((n_nights, 0, n_aper)),
        night_mag_err=np.zeros((n_nights, 0, n_aper)),
        tie_star=np.zeros((n_nights, 0, n_aper), dtype=bool),
        rejected=np.zeros((0, n_aper), dtype=bool),
        floor_mag_centres=np.tile(np.array([10.0, 20.0]), (n_aper, 1)),
        floor=np.zeros((n_nights, n_aper, n_bins)),
        n_tie=np.zeros((n_nights, n_aper), dtype=np.int64),
        resid_mad=np.zeros((n_nights, n_aper)),
        resid_mad_bright=np.zeros((n_nights, n_aper)),
        chi2_after=np.zeros((n_nights, n_aper)),
        chi2_holdout=np.zeros((n_nights, n_aper)),
        chi2_holdout_bins=np.zeros((n_nights, n_aper, n_bins)),
        n_iter=np.zeros(n_aper, dtype=np.int64),
        seeing_basis_terms=(),
        seeing_coef=np.zeros((n_aper, 0)),
        seeing_mag0=np.full(n_aper, np.nan),
        seeing_crowd0=np.full(n_aper, np.nan),
        night_fwhm=np.zeros(n_nights),
        crowding=np.zeros(0),
    )


def test_bls_recovers_injected_transit() -> None:
    # A handful of nights spaced exactly like a period (or a low multiple of
    # it) apart pathologically aliases box search (the period grid finds an
    # exact P/2 or P/3 sub-harmonic instead -- verified interactively while
    # writing this test, not a relphot bug: with only a few short windows,
    # aliases with identically-phased boxes can outscore the true period).
    # Irregular night spacing over a longer, denser baseline (10 nights of
    # 8 h, seeded) avoids that and recovers the injected period cleanly and
    # reproducibly across noise realisations.
    rng = np.random.default_rng(2)
    n_nights = 10
    n_per_night = 90
    period_true = 1.7
    duration_days = 2.0 / 24.0
    depth = 0.01

    night_starts = np.sort(np.random.default_rng(3000).uniform(0.0, 25.0, n_nights))
    bjd_list = []
    night_of_frame_list = []
    for n, start in enumerate(night_starts):
        bjd_list.append(start + np.linspace(0.0, 8.0 / 24.0, n_per_night))
        night_of_frame_list.append(np.full(n_per_night, n, dtype=np.int64))
    bjd_tdb = np.concatenate(bjd_list)
    night_of_frame = np.concatenate(night_of_frame_list)

    phase = ((bjd_tdb + period_true / 2.0) % period_true) - period_true / 2.0
    transit_flux = np.ones_like(bjd_tdb)
    transit_flux[np.abs(phase) < duration_days / 2.0] -= depth
    transit_flux += rng.normal(0.0, 0.0008, bjd_tdb.size)
    flat_flux = 1.0 + rng.normal(0.0, 0.0008, bjd_tdb.size)

    n_global = 2
    flux_norm = np.stack([flat_flux, transit_flux]).astype(np.float32)
    flux_norm_err = np.full((n_global, bjd_tdb.size), 0.0008, dtype=np.float32)

    mlc = MultiNightLightCurves(
        labels=tuple(f"N{n}" for n in range(n_nights)),
        night_of_frame=night_of_frame,
        frame_in_night=np.zeros_like(night_of_frame),
        bjd_tdb=bjd_tdb,
        airmass=np.ones_like(bjd_tdb),
        fwhm=np.ones_like(bjd_tdb),
        aperture=np.zeros(n_global, dtype=np.int64),
        mag=np.zeros((n_global, bjd_tdb.size), dtype=np.float32),
        mag_err=np.zeros((n_global, bjd_tdb.size), dtype=np.float32),
        flux_norm=flux_norm,
        flux_norm_err=flux_norm_err,
        night_mean_mag=np.zeros((n_nights, n_global)),
        night_mean_err=np.zeros((n_nights, n_global)),
        mean_mag=np.full(n_global, 15.0),
        n_nights=np.full(n_global, n_nights, dtype=np.int64),
    )

    settings = MultiNightSettings()
    result = bls_search(mlc, np.array([0, 1]), settings)

    assert not result["candidate"][0]
    assert result["candidate"][1]
    p_rec = result["period"][1]
    ratios = [p_rec / period_true, p_rec / (period_true / 2.0), p_rec / (period_true * 2.0)]
    assert any(abs(r - 1.0) < 0.01 for r in ratios), (p_rec, period_true)


def test_combined_periodogram_recovers_sinusoid() -> None:
    # Nights spaced exactly N days apart alias a Lomb-Scargle search onto
    # the window function's own period (verified interactively while
    # writing this test); irregular spacing avoids it.
    rng = np.random.default_rng(3)
    n_nights = 5
    n_per_night = 30
    period_true = 3.3
    amplitude = 0.010

    night_starts = np.sort(np.random.default_rng(4000).uniform(0.0, 20.0, n_nights))
    bjd_list = []
    night_of_frame_list = []
    for n, start in enumerate(night_starts):
        bjd_list.append(start + np.sort(rng.uniform(0.0, 6.0 / 24.0, n_per_night)))
        night_of_frame_list.append(np.full(n_per_night, n, dtype=np.int64))
    bjd_tdb = np.concatenate(bjd_list)
    night_of_frame = np.concatenate(night_of_frame_list)

    mag_base = 15.0
    noise_sigma = 0.002
    signal = amplitude * np.sin(2.0 * np.pi * bjd_tdb / period_true)
    mag = (mag_base + signal + rng.normal(0.0, noise_sigma, bjd_tdb.size)).astype(np.float32)
    mag_err = np.full(bjd_tdb.size, noise_sigma, dtype=np.float32)

    n_global = 1
    mlc = MultiNightLightCurves(
        labels=tuple(f"N{n}" for n in range(n_nights)),
        night_of_frame=night_of_frame,
        frame_in_night=np.zeros_like(night_of_frame),
        bjd_tdb=bjd_tdb,
        airmass=np.ones_like(bjd_tdb),
        fwhm=np.ones_like(bjd_tdb),
        aperture=np.zeros(n_global, dtype=np.int64),
        mag=mag[None, :],
        mag_err=mag_err[None, :],
        flux_norm=np.ones((n_global, bjd_tdb.size), dtype=np.float32),
        flux_norm_err=np.full((n_global, bjd_tdb.size), noise_sigma, dtype=np.float32),
        night_mean_mag=np.zeros((n_nights, n_global)),
        night_mean_err=np.zeros((n_nights, n_global)),
        mean_mag=np.full(n_global, mag_base),
        n_nights=np.full(n_global, n_nights, dtype=np.int64),
    )
    tie = _toy_tie(n_nights)
    settings = MultiNightSettings(
        periodogram_max_freq=10.0, periodogram_samples_per_peak=10, ls_fap_threshold=1e-3,
    )
    result = combined_periodogram(tie, mlc, np.array([0]), settings)

    assert result["candidate"][0]
    assert abs(result["period"][0] - period_true) / period_true < 0.02


def test_two_night_step_is_internight_but_not_periodic_or_bls() -> None:
    """A star constant within each of 2 nights but offset 50 mmag between them.

    With only 2 nights, ``min_nights_periodic`` (default 3) must keep both
    the combined periodogram and BLS from running on it at all (a period
    longer than a night's own span is unconstrained by 2 nights, and their
    naive FAPs/SNRs would otherwise happily fold this pure night-to-night
    step onto some phase and call it periodic/transiting). It must still be
    an inter-night variability candidate (B2 does not have this restriction).
    """
    rng = np.random.default_rng(42)
    n_nights = 2
    n_per_night = 40
    mag_level = [15.0, 15.05]  # a 50 mmag step between nights
    sigma = 0.003

    bjd_list = []
    night_of_frame_list = []
    mag_list = []
    flux_list = []
    for n in range(n_nights):
        start = n * 3.0
        t = start + np.linspace(0.0, 4.0 / 24.0, n_per_night)
        bjd_list.append(t)
        night_of_frame_list.append(np.full(n_per_night, n, dtype=np.int64))
        mag_list.append(mag_level[n] + rng.normal(0.0, sigma, n_per_night))
        # flux_norm is per-night median-normalised, so it never carries the
        # night-to-night step -- only the calibrated `mag` does.
        flux_list.append(1.0 + rng.normal(0.0, sigma / 1.0857, n_per_night))
    bjd_tdb = np.concatenate(bjd_list)
    night_of_frame = np.concatenate(night_of_frame_list)
    mag = np.concatenate(mag_list).astype(np.float32)
    mag_err = np.full(bjd_tdb.size, sigma, dtype=np.float32)
    flux_norm = np.concatenate(flux_list).astype(np.float32)
    flux_norm_err = np.full(bjd_tdb.size, sigma / 1.0857, dtype=np.float32)

    n_global = 1
    night_mean_mag = np.array([[mag_level[0]], [mag_level[1]]])
    night_mean_err = np.full((n_nights, n_global), sigma / np.sqrt(n_per_night))

    mlc = MultiNightLightCurves(
        labels=tuple(f"N{n}" for n in range(n_nights)),
        night_of_frame=night_of_frame,
        frame_in_night=np.zeros_like(night_of_frame),
        bjd_tdb=bjd_tdb,
        airmass=np.ones_like(bjd_tdb),
        fwhm=np.ones_like(bjd_tdb),
        aperture=np.zeros(n_global, dtype=np.int64),
        mag=mag[None, :],
        mag_err=mag_err[None, :],
        flux_norm=flux_norm[None, :],
        flux_norm_err=flux_norm_err[None, :],
        night_mean_mag=night_mean_mag,
        night_mean_err=night_mean_err,
        mean_mag=np.array([float(np.mean(mag_level))]),
        n_nights=np.array([n_nights], dtype=np.int64),
    )
    tie = _toy_tie(n_nights)
    settings = MultiNightSettings()  # default min_nights_periodic = 3

    internight = compute_internight_variability(mlc, settings)
    assert internight["candidate"][0]

    ls_result = combined_periodogram(tie, mlc, np.array([0]), settings)
    assert not ls_result["candidate"][0]
    assert not np.isfinite(ls_result["period"][0])

    bls_result = bls_search(mlc, np.array([0]), settings)
    assert not bls_result["candidate"][0]
    assert not np.isfinite(bls_result["period"][0])


# --------------------------------------------------------------------------
# loose nights: variability yes (with a core-support rule), transit tests no
# --------------------------------------------------------------------------


def test_internight_loose_night_alone_never_makes_a_candidate() -> None:
    rng = np.random.default_rng(5)
    n_nights, n_stars = 4, 1200
    base = rng.uniform(13.0, 19.0, n_stars)
    err = 0.005
    mag = base[None, :] + rng.normal(0.0, err, (n_nights, n_stars))
    errs = np.full((n_nights, n_stars), err)
    loose = np.array([False, False, False, True])

    # noise-free special stars, so the core-only test is exactly consistent (p_core = 1)
    mag[:, :5] = base[None, :5]
    mag[3, 0] += 0.05  # 0: a step in the loose night only
    mag[1, 1] += 0.05  # 1: a step in a core night
    mag[:3, 2] = np.nan  # 2: the loose night and just one core night ...
    mag[0, 2] = base[2]
    mag[3, 2] += 0.05  # ... which differ
    mag[3, 3] = np.nan  # 3: a step between core nights, no loose data at all
    mag[2, 3] += 0.05
    mlc = _toy_mlc(mag, errs)
    settings = MultiNightSettings()

    plain = compute_internight_variability(mlc, settings)
    assert plain["candidate"][:4].all()
    assert np.all(np.isnan(plain["p_core"])) and not plain["loose_gated"].any()

    res = compute_internight_variability(mlc, settings, loose)
    assert not res["candidate"][0] and res["loose_gated"][0]  # loose night alone: gated
    assert res["p_core"][0] == pytest.approx(1.0)
    assert res["candidate"][1] and not res["loose_gated"][1]  # core night supports it
    assert res["p_core"][1] < 1e-4
    assert not res["candidate"][2] and res["loose_gated"][2]  # < 2 core nights: never
    assert np.isnan(res["p_core"][2])
    assert res["candidate"][3] and not res["loose_gated"][3]  # no loose data: unchanged
    # everything else is untouched by the rule (same verdict, same p)
    np.testing.assert_array_equal(
        res["candidate"][4:] | res["loose_gated"][4:], plain["candidate"][4:]
    )
    np.testing.assert_array_equal(res["p"], plain["p"])


def _flat_mlc(
    n_nights: int, transit_nights: tuple[int, ...], depth: float
) -> MultiNightLightCurves:
    """One star, ``n_nights`` consecutive 8 h nights; a 2 h dip at each night's start in
    ``transit_nights`` only. ``night_mean_*`` are constant (no inter-night signal)."""
    rng = np.random.default_rng(9)
    n_per = 60
    bjd = np.concatenate([
        float(n) + np.linspace(0.05, 0.05 + 8.0 / 24.0, n_per) for n in range(n_nights)
    ])
    night = np.repeat(np.arange(n_nights), n_per)
    flux = 1.0 + rng.normal(0.0, 0.0008, bjd.size)
    dip = np.isin(night, transit_nights) & (bjd % 1.0 < 0.05 + 2.0 / 24.0)
    flux[dip] -= depth
    mag = 15.0 + rng.normal(0.0, 0.002, bjd.size)
    return MultiNightLightCurves(
        labels=tuple(f"N{n}" for n in range(n_nights)),
        night_of_frame=night.astype(np.int64),
        frame_in_night=np.zeros(bjd.size, dtype=np.int64),
        bjd_tdb=bjd,
        airmass=np.ones_like(bjd),
        fwhm=np.ones_like(bjd),
        aperture=np.zeros(1, dtype=np.int64),
        mag=mag[None, :].astype(np.float32),
        mag_err=np.full((1, bjd.size), 0.002, dtype=np.float32),
        flux_norm=flux[None, :].astype(np.float32),
        flux_norm_err=np.full((1, bjd.size), 0.0008, dtype=np.float32),
        night_mean_mag=np.full((n_nights, 1), 15.0),
        night_mean_err=np.full((n_nights, 1), 0.01),
        mean_mag=np.array([15.0]),
        n_nights=np.array([n_nights], dtype=np.int64),
    )


def test_bls_epoch_mask_keeps_loose_epochs_out() -> None:
    n_nights = 8
    loose = np.array([n >= 4 for n in range(n_nights)])
    mlc = _flat_mlc(n_nights, transit_nights=(4, 5, 6, 7), depth=0.03)  # dips only in loose nights
    settings = MultiNightSettings()

    unmasked = bls_search(mlc, np.array([0]), settings)
    assert unmasked["candidate"][0]  # the search does see the loose nights' dips

    masked = bls_search(mlc, np.array([0]), settings, ~loose[mlc.night_of_frame])
    assert not masked["candidate"][0]

    # too few core nights left for a periodic search: skipped outright
    too_few = np.array([n >= 2 for n in range(n_nights)])
    skipped = bls_search(mlc, np.array([0]), settings, ~too_few[mlc.night_of_frame])
    assert not np.isfinite(skipped["period"][0]) and not skipped["candidate"][0]


def _write_night_search(directory, *, event: bool) -> None:
    (directory / "lc").mkdir(parents=True)
    pd.DataFrame({
        "star_id": np.array([0]),
        "variability_candidate": np.array([False]),
        "transit_candidate": np.array([event]),
        "transit_tc_bjd_tdb": np.array([0.1]),
        "transit_depth": np.array([0.03]),
        "transit_snr": np.array([20.0]),
        "transit_duration_hours": np.array([2.0]),
    }).to_parquet(directory / "lc" / "night_lc_search_metrics.parquet")


def _multisearch_inputs(tmp_path, event_nights: tuple[int, ...], loose: tuple[int, ...]):
    n_nights = 6
    mlc = _flat_mlc(n_nights, transit_nights=tuple(range(n_nights)), depth=0.03)
    labels = mlc.labels
    night_info = []
    for n, label in enumerate(labels):
        directory = tmp_path / label
        _write_night_search(directory, event=n in event_nights)
        night_info.append({"label": label, "directory": str(directory)})
    xmatch = NightCrossMatch(
        ra=np.zeros(1), dec=np.zeros(1), index=np.zeros((n_nights, 1), dtype=np.int64),
        labels=labels, n_matched=np.zeros(n_nights, dtype=np.int64),
    )
    tie = _toy_tie(n_nights, labels=labels)
    tie.loose = np.array([n in loose for n in range(n_nights)])
    return xmatch, tie, mlc, night_info


def test_cross_reference_and_multisearch_drop_loose_nights_transit_events(tmp_path) -> None:
    settings = MultiNightSettings()

    # an event only in the loose night 5: no event, hence no period-compatibility run
    xmatch, tie, mlc, night_info = _multisearch_inputs(tmp_path / "a", (5,), (5,))
    plain = cross_reference_nights(night_info, xmatch)
    assert list(plain.events) == [0] and plain.events[0][0]["night"] == "N5"
    skipped = cross_reference_nights(night_info, xmatch, tie.loose)
    assert skipped.events == {}
    result = run_multisearch(xmatch, tie, mlc, night_info, settings)
    assert result["cross_ref"].events == {} and result["period_compat"] == {}

    # the same event in a core night is kept and gets its period-compatibility scan, which
    # (like the BLS search) sees only the core nights' epochs
    xmatch, tie, mlc, night_info = _multisearch_inputs(tmp_path / "b", (1,), (5,))
    result = run_multisearch(xmatch, tie, mlc, night_info, settings)
    assert list(result["cross_ref"].events) == [0]
    assert 0 in result["period_compat"]

    # without any loose night nothing changes
    xmatch, tie, mlc, night_info = _multisearch_inputs(tmp_path / "c", (5,), ())
    result = run_multisearch(xmatch, tie, mlc, night_info, settings)
    assert list(result["cross_ref"].events) == [0] and 0 in result["period_compat"]
