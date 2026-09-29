"""Tests for relphot.web.phase: the Fourier model behind the phase diagram (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from relphot.web.phase import fourier_model, phase_coverage


def _series(period: float = 2.5, amp: float = 0.15, noise: float = 0.003, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = 2460000.0 + np.sort(np.concatenate([s + rng.uniform(0, 0.5, 100) for s in (0.0, 1.1, 2.3)]))
    night = np.repeat([1, 2, 3], 100)
    y = 15.0 + amp * np.sin(2 * np.pi * (t - 2460000.0) / period) + rng.normal(0, noise, t.size)
    return t, y, np.full(t.size, noise), night


def test_fourier_model_recovers_amplitude_and_phase_zero_of_a_long_period() -> None:
    t, y, dy, night = _series()
    m = fourier_model(t, y, dy, night, 2.5, per_night_offsets=False, brighter_is_lower=True)
    assert m is not None and len(m["coef"]) == 4
    assert m["amplitude"] == pytest.approx(0.30, rel=0.05)  # peak to peak of 0.15 sin
    assert 0.7 < m["chi2_red"] < 1.5
    # the faintest point (largest magnitude) of 15 + 0.15 sin(2 pi t / 2.5) is at t = 0.625 d
    phase = ((m["t_zero"] - 2460000.0 - 0.625) / 2.5) % 1.0
    assert min(phase, 1.0 - phase) < 0.03
    assert m["mean"] == pytest.approx(15.0, abs=0.03)


def test_per_night_offsets_option_is_honoured() -> None:
    t, y, dy, night = _series()
    free = fourier_model(t, y, dy, night, 2.5, per_night_offsets=False, brighter_is_lower=True)
    offsets = fourier_model(t, y, dy, night, 2.5, per_night_offsets=True, brighter_is_lower=True)
    # at the right period both find the signal; the offsets variant has 2 more parameters, so a
    # slightly smaller chi2, and its `mean` is the mean of three per-night offsets
    assert offsets["chi2_red"] <= free["chi2_red"] * (1 + 1e-9)
    assert offsets["amplitude"] == pytest.approx(free["amplitude"], rel=0.1)
    # a wrong period is not fitted as well by the offset-free model
    wrong = fourier_model(t, y, dy, night, 1.7, per_night_offsets=False, brighter_is_lower=True)
    assert wrong["chi2_red"] > 3 * free["chi2_red"]

    t2, y2, dy2, night2 = _series(period=0.31, amp=0.05)
    short = fourier_model(t2, y2, dy2, night2, 0.31, per_night_offsets=True, brighter_is_lower=True)
    assert short["amplitude"] == pytest.approx(0.10, rel=0.1)


def test_the_faint_point_is_the_minimum_of_a_flux_curve() -> None:
    t = 2460000.0 + np.linspace(0.0, 2.0, 200)
    y = 1.0 + 0.1 * np.sin(2 * np.pi * (t - 2460000.0) / 1.0)
    m = fourier_model(
        t, y, np.full_like(t, 0.001), np.ones(t.size), 1.0,
        per_night_offsets=True, brighter_is_lower=False,
    )
    phase = ((m["t_zero"] - 2460000.0 - 0.75) / 1.0) % 1.0  # sin is lowest at 0.75 cycles
    assert min(phase, 1.0 - phase) < 0.02


def test_fourier_model_refuses_what_it_cannot_fit() -> None:
    t = 2460000.0 + np.linspace(0.0, 1.0, 5)
    y, dy = np.ones(5), np.full(5, 0.01)
    night = np.ones(5)
    kw = {"per_night_offsets": False, "brighter_is_lower": True}
    assert fourier_model(t, y, dy, night, 1.0, **kw) is None  # 5 points, 5 parameters
    assert fourier_model(t, y, dy, night, -1.0, **kw) is None
    assert fourier_model(t, np.full(5, np.nan), dy, night, 1.0, **kw) is None


def test_phase_coverage_counts_bins_and_cycles() -> None:
    t = 2460000.0 + np.linspace(0.0, 0.5, 200)
    coverage, cycles = phase_coverage(t, 2.5, 20)
    assert cycles == pytest.approx(0.2) and coverage == pytest.approx(0.2, abs=0.05)
    coverage, cycles = phase_coverage(t, 0.1, 20)
    assert coverage == 1.0 and cycles == pytest.approx(5.0)
