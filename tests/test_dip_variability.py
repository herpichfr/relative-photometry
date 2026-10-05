"""Tests for relphot.dip_variability: the catalogue-period dip tests of the VARIABILITY rule.

Pure numpy, no database (the rule in ``relphot db analyze`` is tested in
tests/test_db_variability.py).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from relphot.config import DbSettings
from relphot.dip_variability import (
    VariabilityVerdict,
    phase_pred_test,
    prep_lc,
    repeat_test,
    variability_verdict,
)
from relphot.exceptions import ConfigError

T0 = 2460000.0
PERIOD = 0.6
DUR_H = 1.44  # 0.06 d = 0.1 cycle
CADENCE = 2.5 / 1440.0
SIGMA = 0.003


def _night(k: int, flux_of, seed: int, start: float = 0.0, length: float = 0.36):
    """One night k days after T0: (bjd, flux, err) of ``flux_of(t)`` plus white noise."""
    rng = np.random.default_rng(seed)
    t = T0 + k + start + np.arange(0.0, length, CADENCE)
    flux = flux_of(t) + rng.normal(0.0, SIGMA, t.size)
    return t, flux, np.full(t.size, SIGMA)


def _eclipses(depth: float = 0.15, phase: float = 0.0):
    """Flux of an eclipsing binary: a box dip of ``DUR_H`` once per ``PERIOD`` at ``phase``."""

    def flux(t):
        ph = ((t - T0) / PERIOD - phase + 0.5) % 1.0 - 0.5  # cycles from the eclipse centre
        return 1.0 - depth * (np.abs(ph) * PERIOD <= 0.5 * DUR_H / 24.0)

    return flux


def _other_nights(flux_of, ks=(1, 2, 3, 4, 5, 6), starts=(0.0, 0.11, 0.23, 0.05, 0.31, 0.17)):
    return [_night(k, flux_of, seed=10 + k, start=st) for k, st in zip(ks, starts, strict=True)]


def _event_night(flux_of, tc: float, seed: int = 99):
    """The event night: a window of 0.36 d centred on ``tc`` (baseline on both sides)."""
    rng = np.random.default_rng(seed)
    t = tc - 0.18 + np.arange(0.0, 0.36, CADENCE)
    flux = flux_of(t) + rng.normal(0.0, SIGMA, t.size)
    return t, flux, np.full(t.size, SIGMA)


# --------------------------------------------------------------------------
# prep_lc
# --------------------------------------------------------------------------


def test_prep_lc_sorts_normalises_and_drops_spikes_and_bad_epochs() -> None:
    rng = np.random.default_rng(0)
    t = T0 + np.arange(40) * CADENCE
    flux = 50.0 * (1.0 + rng.normal(0.0, 0.002, 40))
    err = np.full(40, 0.1)
    flux[17] *= 1.2  # an isolated spike
    flux[3] = np.nan
    err[5] = 0.0
    order = rng.permutation(40)
    tt, yy, ee = prep_lc(t[order], flux[order], err[order])
    assert np.all(np.diff(tt) > 0)
    assert tt.size == 40 - 2 - 1  # the NaN epoch, the zero-error epoch and the spike
    assert abs(np.median(yy) - 1.0) < 1e-3
    assert np.allclose(ee * 50.0, 0.1, rtol=0.05)
    # fewer than 8 usable epochs: three empty arrays
    assert all(a.size == 0 for a in prep_lc(t[:7], flux[:7], err[:7]))


# --------------------------------------------------------------------------
# phase test
# --------------------------------------------------------------------------


def test_phase_fires_for_an_eclipse_at_the_predicted_phase_on_the_other_nights() -> None:
    flux_of = _eclipses()
    others = [prep_lc(*lc) for lc in _other_nights(flux_of)]
    tc = T0 + 8 * PERIOD  # an eclipse
    t, y, e = prep_lc(*_event_night(flux_of, tc))
    r = phase_pred_test(t, y, e, tc, DUR_H, others, PERIOD)
    assert r["cov"] >= 0.8 and r["n_nights"] == 6
    assert r["depth_obs"] == pytest.approx(0.15, rel=0.2)
    assert 0.7 <= r["ratio"] <= 1.3
    assert r["chi2_pred"] <= r["chi2_flat"]

    lcs = _other_nights(flux_of)
    verdict = variability_verdict(_event_night(flux_of, tc), lcs, tc, DUR_H, [], [], PERIOD)
    assert verdict.phase and not verdict.repeat and verdict.fired
    assert verdict.n_nights == 6 and verdict.cov >= 0.8 and verdict.ratio >= 0.7
    assert (verdict.n_match, verdict.p_tail) == (0, 1.0)


@pytest.mark.parametrize("phase", [0.15, 0.37, 0.62, 0.9])
def test_phase_does_not_fire_for_a_box_transit_at_a_random_phase_of_a_sinusoidal_variable(
    phase: float,
) -> None:
    def variable(t):
        return 1.0 + 0.01 * np.sin(2.0 * np.pi * (t - T0) / PERIOD)

    lcs = _other_nights(variable)
    tc = T0 + 8 * PERIOD + phase * PERIOD
    t, flux, err = _event_night(variable, tc)
    flux = flux * (1.0 - 0.03 * (np.abs(t - tc) <= 0.5 * DUR_H / 24.0))  # a real transit
    verdict = variability_verdict((t, flux, err), lcs, tc, DUR_H, [], [], PERIOD)
    assert not verdict.fired
    assert verdict.cov >= 0.8  # the phase was covered: the prediction simply has no dip
    assert not verdict.ratio >= 0.7


def test_the_phase_test_needs_other_nights_and_enough_epochs() -> None:
    flux_of = _eclipses()
    tc = T0 + 8 * PERIOD
    event = _event_night(flux_of, tc)
    assert not variability_verdict(event, [], tc, DUR_H, [], [], PERIOD).fired
    # nights of fewer than 8 epochs do not count
    short = [(t[:6], f[:6], e[:6]) for t, f, e in _other_nights(flux_of)]
    verdict = variability_verdict(event, short, tc, DUR_H, [], [], PERIOD)
    assert verdict.n_nights == 0 and not verdict.fired
    # an event night of fewer than 12 epochs is not judged, nor one without tc / duration / period
    t, f, e = event
    tiny = (t[:11], f[:11], e[:11])
    assert not variability_verdict(tiny, _other_nights(flux_of), tc, DUR_H, [], [], PERIOD).fired
    others = _other_nights(flux_of)
    for t_c, dur, period in (
        (float("nan"), DUR_H, PERIOD), (tc, float("nan"), PERIOD), (tc, DUR_H, 0.0),
    ):
        assert not variability_verdict(event, others, t_c, dur, [], [], period).fired


def test_phase_thresholds_come_from_the_settings() -> None:
    flux_of = _eclipses()
    tc = T0 + 8 * PERIOD
    args = (_event_night(flux_of, tc), _other_nights(flux_of), tc, DUR_H, [], [], PERIOD)
    assert variability_verdict(*args).phase
    assert not variability_verdict(*args, None, replace(DbSettings(), auto_var_phase_ratio_min=3.0)
                                   ).phase
    assert not variability_verdict(*args, None, replace(DbSettings(), auto_var_phase_cov_min=1.0,
                                                        auto_var_phase_ratio_min=3.0)).phase


# --------------------------------------------------------------------------
# repeat test
# --------------------------------------------------------------------------


def test_repeat_fires_for_other_events_at_multiples_of_the_period() -> None:
    p = 0.8
    others_tc = [T0 + p, T0 + 3 * p, T0 - 2 * p]
    r = repeat_test(T0, 1.0, others_tc, [1.0, 1.0, 1.0], p, 1e-4 * p)
    assert (r["n_other"], r["n_match_P"], r["n_match_half"], r["n_match"]) == (3, 3, 0, 3)
    assert r["p_tail"] < 1e-3 and r["best_resid_h"] < 1e-6
    # odd multiples of P/2 (the secondary eclipse) count too, even ones are the P matches
    r = repeat_test(T0, 1.0, [T0 + 1.5 * p, T0 + 0.5 * p], [1.0, 1.0], p, 1e-4 * p)
    assert (r["n_match_P"], r["n_match_half"], r["n_match"]) == (0, 2, 2)

    verdict = variability_verdict(
        _night(0, lambda t: np.ones_like(t), seed=1), [], T0 + 0.1, 1.0,
        [T0 + 0.1 + p, T0 + 0.1 + 3 * p, T0 + 0.1 - 2 * p], [1.0, 1.0, 1.0], p,
    )
    assert verdict.repeat and not verdict.phase and verdict.n_match == 3
    assert verdict.p_tail <= DbSettings().auto_var_repeat_p_max
    # a single match at this width is a chance coincidence one time in ten: not enough
    one = variability_verdict(
        _night(0, lambda t: np.ones_like(t), seed=1), [], T0 + 0.1, 1.0, [T0 + 0.1 + p], [1.0], p,
    )
    assert one.n_match == 1 and not one.repeat and 0.01 < one.p_tail < 0.2


def test_repeat_does_not_match_events_that_are_not_commensurate_with_the_period() -> None:
    p = 0.8
    r = repeat_test(T0, 1.0, [T0 + 0.37 * p, T0 + 1.23 * p, T0 - 2.71 * p], [1.0] * 3, p, 1e-4 * p)
    assert r["n_match"] == 0 and r["p_tail"] == 1.0
    assert np.isfinite(r["expected"]) and r["expected"] > 0
    # no other events, a bad period: nothing to say
    assert repeat_test(T0, 1.0, [], [], p, 1e-4 * p)["n_match"] == 0
    assert repeat_test(T0, 1.0, [T0 + p], [1.0], float("nan"), 1e-4)["n_match"] == 0
    assert repeat_test(T0, 1.0, [T0 + p], [1.0], -1.0, 1e-4)["n_match"] == 0


def test_repeat_tolerance_grows_with_the_period_uncertainty() -> None:
    p, dt = 0.8, 20 * 0.8 + 0.05  # 20 cycles later, 0.05 d (1.2 h) off
    assert repeat_test(T0, 1.0, [T0 + dt], [1.0], p, 1e-5)["n_match"] == 0
    assert repeat_test(T0, 1.0, [T0 + dt], [1.0], p, 3e-3)["n_match"] == 1
    # sigma_P NaN: 1e-4 * P is used (20 * 8e-5 d = 2.3 min on top of the half duration, 30 min)
    assert repeat_test(T0, 1.0, [T0 + 20 * p + 0.03], [1.0], p, float("nan"))["n_match"] == 0
    assert repeat_test(T0, 1.0, [T0 + 20 * p + 0.02], [1.0], p, float("nan"))["n_match"] == 1


# --------------------------------------------------------------------------
# the thresholds in DbSettings
# --------------------------------------------------------------------------


def test_variability_settings_defaults_and_validation() -> None:
    d = DbSettings()
    assert (d.auto_var_phase_cov_min, d.auto_var_phase_ratio_min, d.auto_var_repeat_p_max) == (
        0.8, 0.7, 0.01,
    )
    for name, bad in (
        ("auto_var_phase_cov_min", 1.5), ("auto_var_phase_cov_min", 0.0),
        ("auto_var_phase_ratio_min", -0.1), ("auto_var_phase_ratio_min", float("nan")),
        ("auto_var_repeat_p_max", 2.0), ("auto_var_repeat_p_max", True),
    ):
        with pytest.raises(ConfigError, match=name):
            replace(DbSettings(), **{name: bad})
    assert isinstance(VariabilityVerdict().fired, bool) and not VariabilityVerdict().fired
