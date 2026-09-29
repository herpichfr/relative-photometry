"""Phase-diagram helpers for the web API (numpy only; no database access).

:func:`fourier_model` fits the 2-harmonic Fourier series the period analysis uses
(:mod:`relphot.db.analyze`) at a fixed period by weighted linear least squares, for the
model curve drawn over the phase diagram. It is duplicated here, rather than imported,
because importing anything under ``relphot.db`` pulls in pandas and astropy, which the web
container does not install (see :mod:`relphot.web.db`).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["fourier_model", "phase_coverage"]


def phase_coverage(t: np.ndarray, period: float, n_bins: int = 20) -> tuple[float, float]:
    """``(fraction of n_bins phase bins holding a point, baseline in periods)`` of epochs ``t``."""
    t = np.asarray(t, dtype=np.float64)
    phase = ((t - float(t.min())) / period) % 1.0
    bins = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
    return float(np.unique(bins).size) / n_bins, float(t.max() - t.min()) / period


def fourier_model(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    night: np.ndarray,
    period: float,
    *,
    per_night_offsets: bool,
    brighter_is_lower: bool,
    n_grid: int = 720,
) -> dict | None:
    """Weighted 2-harmonic Fourier fit of ``y(t)`` at ``period``.

    ``y = c + sum_k (a_k cos(2 pi k (t - t_ref) / P) + b_k sin(...))``, with one constant per
    night when ``per_night_offsets`` (the mean of those constants is reported as ``mean``),
    else a single one. Returns ``None`` when the fit is under-determined, else a dict with
    ``period``, ``t_ref``, ``mean``, ``coef`` (``[a1, b1, a2, b2]``), ``chi2_red``,
    ``amplitude`` (peak to peak of the model over one cycle) and ``t_zero`` -- the epoch (BJD)
    of the model's faintest point (the deepest minimum of light: the maximum of a magnitude,
    the minimum of a flux; ``brighter_is_lower`` says which), the phase zero offered on the
    phase diagram.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    t, y, dy, night = t[good], y[good], dy[good], np.asarray(night)[good]
    if not (period > 0 and math.isfinite(period)) or t.size == 0:
        return None
    if per_night_offsets:
        _, inv = np.unique(night, return_inverse=True)
    else:
        inv = np.zeros(t.size, dtype=np.int64)
    n_off = int(inv.max()) + 1
    n_par = 4 + n_off
    if t.size - n_par < 1:
        return None
    t_ref = float(np.mean(t))
    phi = 2.0 * np.pi * (t - t_ref) / period
    offsets = np.zeros((t.size, n_off))
    offsets[np.arange(t.size), inv] = 1.0
    design = np.column_stack(
        [np.cos(phi), np.sin(phi), np.cos(2.0 * phi), np.sin(2.0 * phi), offsets]
    )
    w = 1.0 / dy
    beta, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    resid = (y - design @ beta) * w
    chi2_red = float(np.sum(resid**2)) / (t.size - n_par)
    mean = float(np.mean(beta[4:]))
    grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    g = 2.0 * np.pi * grid
    curve = (
        mean + beta[0] * np.cos(g) + beta[1] * np.sin(g)
        + beta[2] * np.cos(2.0 * g) + beta[3] * np.sin(2.0 * g)
    )
    faint = int(np.argmax(curve) if brighter_is_lower else np.argmin(curve))
    return {
        "period": float(period), "t_ref": t_ref, "mean": mean,
        "coef": [float(v) for v in beta[:4]], "chi2_red": chi2_red,
        "amplitude": float(curve.max() - curve.min()),
        "t_zero": t_ref + float(grid[faint]) * period,
    }
