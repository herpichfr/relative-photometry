"""R90 false-positive screen for the single-event transit search (informational flags).

A search event is a box fit to a decorrelated light curve; most false events are not
boxes at all but a few discrepant epochs, a seeing/background/centroid excursion that
the comparison-star CBVs do not carry, or noise the box happens to fit. This module
refits an event the way a person would vet it and returns the raw numbers
(:class:`R90Features`); :func:`relphot.transit_search.r90_flag_bits` turns them into the
``R90_*`` flag bits with the thresholds of :class:`relphot.config.SearchSettings`.

The model is a trapezoid (mid-time ``tc``, total duration ``T14``, ingress fraction
``f = T12 / T14``) fit *simultaneously* with the same nuisance model as the search
(polynomial in time + CBVs): a variable-projection fit, nonlinear in ``(tc, T14, f)``
and linear in the nuisance coefficients and the depth, started from a global grid.
The features are ported unchanged from the detection-improvement study
(``fitlib.features`` there) so that the thresholds derived on 1622 reviewed events
apply as they are:

* ``top1_share`` / ``top3_share`` -- share of the event's chi2 improvement carried by
  its largest one / three epochs (single-point influence);
* ``reg_dchi2_ratio`` / ``reg_depth_ratio`` -- refit with per-star FWHM (relative to
  the frame median), local background and centroid x/y added to the nuisance model:
  the retained fraction of the event's chi2 improvement and of its depth;
* ``clip3_dchi2`` -- chi2 improvement (scaled by the reduced chi2) after a 3-sigma
  residual clip and refit;
* ``dbic_flat`` -- the trapezoid against the nuisance-only model, penalised for the
  four extra parameters: ``dchi2 / max(1, chi2_red) - 4 ln N``.

These are *flags*, not vetoes: the regressor test in particular penalises a real
transit that coincides with a seeing change, so the cuts are reported for a person to
weigh and never touch candidacy or the tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from relphot.numeric import mad_sigma, nanmedian_quiet

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "R90Features",
    "compute_r90_features",
    "fit_trapezoid",
    "trap_shape",
]

#: Ingress fractions of the global starting grid.
_FGRID = (0.02, 0.08, 0.18, 0.3, 0.5)

#: Numerical failures of a fit that mean "this start did not work", never a bug to surface.
_FIT_ERRORS = (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError)


@dataclass(frozen=True, slots=True)
class R90Features:
    """Raw R90 numbers for one event; NaN where a number could not be computed.

    ``fit_ok`` is False when no trapezoid fit could be made at all (every feature NaN);
    ``success`` is the optimiser's own convergence flag for the final fit.
    """

    fit_ok: bool = False
    success: bool = False
    top1_share: float = np.nan
    top3_share: float = np.nan
    reg_dchi2_ratio: float = np.nan
    reg_depth_ratio: float = np.nan
    clip3_dchi2: float = np.nan
    dbic_flat: float = np.nan


def trap_shape(t: np.ndarray, tc: float, t14: float, f: float) -> np.ndarray:
    """Symmetric trapezoid, 1 on the flat bottom and 0 outside ``T14``, ingress ``f * T14``."""
    half = 0.5 * t14
    tau = np.maximum(f * t14, 1e-4 * t14)
    return np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _nuisance_matrix(t: np.ndarray, cbv: np.ndarray, poly: int) -> np.ndarray:
    """Design matrix of the search's nuisance model: poly(t - median) + CBVs."""
    t0 = np.median(t)
    cols = [np.ones_like(t)]
    for d in range(1, poly + 1):
        cols.append((t - t0) ** d)
    for row in cbv:
        cols.append(row)
    return np.column_stack(cols)


class _Projection:
    """Weighted projection of the nuisance space (the nuisance-only fit of ``y``)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.x = x
        self.w = w
        xtw = x.T * w[None, :]
        self.a_inv = np.linalg.pinv(xtw @ x)
        beta = self.a_inv @ (xtw @ y)
        self.yp = y - x @ beta
        self.chi2_0 = float(np.sum(w * self.yp**2))
        self.xtw = xtw

    def reduce(self, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(num, den)`` per column of ``shapes`` ``(N, G)``.

        The chi2 reduction is ``num**2 / den`` and the depth ``-num / den``.
        """
        sp = shapes - self.x @ (self.a_inv @ (self.xtw @ shapes))
        num = (self.w * self.yp) @ sp
        den = np.einsum("n,ng,ng->g", self.w, sp, sp)
        return num, den


def _duration_grid(span_d: float) -> np.ndarray:
    """20 log-spaced trial T14 from 0.25 h up to min(6 h, 3 x span)."""
    lo = 0.25 / 24.0
    hi = min(6.0 / 24.0, max(3.0 * span_d, 1.0 / 24.0))
    hi = max(hi, lo * 1.5)
    return np.geomspace(lo, hi, 20)


def _grid_trapezoid(
    t: np.ndarray, pj: _Projection, t14s: np.ndarray, fgrid: tuple[float, ...] = _FGRID
) -> dict:
    """Best (tc, T14, f) over a coarse grid, by chi2 reduction at depth > 0."""
    t0, t1 = t[0], t[-1]
    best: dict = {"chi2red": -1.0}
    cad = np.median(np.diff(t))
    res = []
    for dur in t14s:
        step = max(cad * 0.5, dur / 25.0)
        tcs = np.arange(t0 - 0.5 * dur, t1 + 0.5 * dur + step, step)
        for f in fgrid:
            shapes = trap_shape(t[:, None], tcs[None, :], dur, f)
            n_in = (shapes > 0).sum(0)
            num, den = pj.reduce(shapes)
            with np.errstate(invalid="ignore", divide="ignore"):
                cr = np.where((den > 1e-12) & (n_in >= 3) & (num < 0), num**2 / den, -1.0)
            k = int(np.argmax(cr))
            if cr[k] > best["chi2red"]:
                best = {
                    "chi2red": float(cr[k]), "tc": float(tcs[k]), "t14": float(dur),
                    "f": float(f), "depth": float(-num[k] / den[k]),
                }
            res.append((float(cr[k]), float(tcs[k]), float(dur), float(f)))
    best["all"] = res
    return best


def _variable_projection_fit(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    shape: Callable[..., np.ndarray] = trap_shape,
):
    """Least-squares over the shape parameters, linear coefficients eliminated each step."""
    sw = np.sqrt(w)
    yw = y * sw

    def resid(theta: np.ndarray) -> np.ndarray:
        s = shape(t, *theta)
        xf = np.column_stack([x, -s]) * sw[:, None]
        c, _, _, _ = np.linalg.lstsq(xf, yw, rcond=None)
        return yw - xf @ c

    th0 = np.clip(theta0, bounds[0] + 1e-9, bounds[1] - 1e-9)
    scale = np.array([max(th0[1], 1e-3) / 8.0, max(th0[1], 1e-3)] + [0.15] * (len(th0) - 2))
    return least_squares(resid, th0, bounds=bounds, x_scale=scale, diff_step=1e-3, max_nfev=80)


def _final_solution(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    theta: np.ndarray,
    shape: Callable[..., np.ndarray] = trap_shape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(coefficients, model, residuals, chi2)`` at fixed shape parameters."""
    s = shape(t, *theta)
    xf = np.column_stack([x, -s])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(xf * sw[:, None], y * sw, rcond=None)
    model = xf @ c
    resid = y - model
    return c, model, resid, float(np.sum(w * resid**2))


def _fixed_ingress_shape(f: float) -> Callable[..., np.ndarray]:
    """``trap_shape`` with the ingress fraction frozen at ``f`` (2-parameter shape)."""

    def shape(t: np.ndarray, tc: float, t14: float) -> np.ndarray:
        return trap_shape(t, tc, t14, f)

    return shape


def fit_trapezoid(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    start: tuple[float, float] | None = None,
) -> dict | None:
    """Global grid, then local refinement from several starts; the best chi2 wins.

    Starts: the grid optimum, the best grid point at each ingress fraction, the caller's
    ``(tc, T14)`` (with ``f = 0.15``) and the box (``f = 0.01``) and V (``f = 0.5``)
    refits. Only depth > 0 solutions count. Returns ``None`` when the grid finds no
    dimming at all; a dict with ``theta``, ``c`` (last entry = depth), ``resid``,
    ``chi2``, ``success`` and the nuisance-only projection ``pj`` otherwise.
    """
    pj = _Projection(x, y, w)
    span = t[-1] - t[0]
    t14s = _duration_grid(span)
    g = _grid_trapezoid(t, pj, t14s)
    if g["chi2red"] <= 0:
        return None
    starts = [(g["tc"], g["t14"], g["f"])]
    for fv in sorted({r_[3] for r_ in g["all"]}):
        cand = [r_ for r_ in g["all"] if r_[3] == fv and r_[0] > 0]
        if cand:
            b_ = max(cand)
            starts.append((b_[1], b_[2], fv))
    if start is not None:
        starts.append((start[0], start[1], 0.15))
    t14_max = max(t14s.max(), 0.5 / 24)
    lb2 = np.array([t[0] - 0.5 * t14_max, 0.2 / 24])
    ub2 = np.array([t[-1] + 0.5 * t14_max, t14_max])
    for fv in (0.01, 0.5):
        gg = _grid_trapezoid(t, pj, t14s, fgrid=(fv,))
        if gg["chi2red"] <= 0:
            continue
        try:
            r_ = _variable_projection_fit(
                t, y, w, x, np.array([gg["tc"], gg["t14"]]), (lb2, ub2),
                shape=_fixed_ingress_shape(fv),
            )
            starts.append((r_.x[0], r_.x[1], fv))
        except _FIT_ERRORS:
            pass
    lb = np.array([t[0] - 0.5 * t14_max, 0.2 / 24, 0.01])
    ub = np.array([t[-1] + 0.5 * t14_max, t14_max, 0.5])
    best = None
    for th0 in starts:
        try:
            r = _variable_projection_fit(t, y, w, x, np.array(th0, float), (lb, ub))
        except _FIT_ERRORS:
            continue
        c, _model, resid, chi2 = _final_solution(t, y, w, x, r.x)
        if c[-1] <= 0:
            continue
        if best is None or chi2 < best["chi2"]:
            best = {"theta": r.x, "c": c, "resid": resid, "chi2": chi2, "success": bool(r.success)}
    if best is None:
        th = np.array([g["tc"], g["t14"], g["f"]])
        c, _model, resid, chi2 = _final_solution(t, y, w, x, th)
        best = {"theta": th, "c": c, "resid": resid, "chi2": chi2, "success": False}
    best["pj"] = pj
    best["lb"] = lb
    best["ub"] = ub
    return best


def _regressor_matrix(regressors: np.ndarray) -> np.ndarray | None:
    """Standardised extra-regressor rows ``(n_reg, N)``, or ``None`` if none is usable.

    Each row (already evaluated at the event's own epochs) has its median removed and is
    divided by its standard deviation; non-finite entries become 0 (no information). A
    row that is entirely non-finite or constant carries nothing and is dropped.
    """
    rows = []
    for v in regressors:
        v = np.where(np.isfinite(v), v, np.nan)
        m = nanmedian_quiet(v)
        if not np.isfinite(m):
            continue
        v = v - m
        sd = float(np.nanstd(v))
        v = np.where(np.isfinite(v), v, 0.0)
        if sd > 0:
            rows.append(v / sd)
    return np.array(rows) if rows else None


def _chi2_scale(chi2: float, dof: int) -> float:
    """``max(1, chi2_red)``: errors are never shrunk, only inflated."""
    return max(1.0, chi2 / max(dof, 1))


def compute_r90_features(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    cbv: np.ndarray,
    tc: float,
    duration: float,
    regressors: np.ndarray | None,
    poly_degree: int,
) -> R90Features:
    """The R90 numbers of one event, from the light curve exactly as the search fit it.

    ``t``/``y``/``err`` are the time-sorted good epochs *after* the search's rolling
    clip, ``cbv`` ``(n_cbv, N)`` the CBVs at those epochs, ``(tc, duration)`` the
    search's best event (only a starting point of the refit). ``regressors`` is
    ``(4, N)``: FWHM relative to the frame median, local background, centroid x and y,
    at the same epochs (``None`` leaves the two ``reg_*`` ratios NaN). ``poly_degree``
    is the search's nuisance polynomial degree.

    A numerical failure anywhere is reported as ``fit_ok=False``, never raised: this is
    an informational screen and must not stop a night's search.
    """
    try:
        return _compute(t, y, err, cbv, tc, duration, regressors, poly_degree)
    except _FIT_ERRORS:
        return R90Features()


def _compute(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    cbv: np.ndarray,
    tc_start: float,
    dur_start: float,
    regressors: np.ndarray | None,
    poly_degree: int,
) -> R90Features:
    n = t.size
    w = 1.0 / err**2
    x = _nuisance_matrix(t, cbv, poly_degree)
    kn = x.shape[1]
    t_ref = t[0]
    tt = t - t_ref  # numerical conditioning of the shape parameters
    fit = fit_trapezoid(tt, y, w, x, start=(tc_start - t_ref, dur_start))
    if fit is None:
        return R90Features()
    tc, t14, f = fit["theta"]
    chi2 = fit["chi2"]
    depth = float(fit["c"][-1])
    s2 = _chi2_scale(chi2, n - (kn + 4))
    pj = fit["pj"]
    dchi2_flat_scaled = (pj.chi2_0 - chi2) / s2
    dbic_flat = dchi2_flat_scaled - 4 * np.log(n)  # trapezoid: 3 nonlinear + depth

    # Influence of single points: per-epoch chi2 improvement of the model over the
    # nuisance-only fit.
    res = fit["resid"]
    imp = (w * pj.yp**2) - (w * res**2)
    tot = imp.sum()
    top1 = top3 = np.nan
    if tot > 0:
        o = np.sort(imp)[::-1]
        top1 = float(o[0] / tot)
        top3 = float(o[:3].sum() / tot)

    # 3-sigma residual clip and refit.
    sig_pt = float(mad_sigma(res)) if n > 4 else np.nan
    sg = sig_pt if sig_pt > 0 else float(np.std(res))
    keep = np.abs(res) <= 3.0 * sg
    clip3 = np.nan
    if keep.sum() < n and keep.sum() > kn + 8:
        fit2 = fit_trapezoid(tt[keep], y[keep], w[keep], x[keep], start=(tc, t14))
        if fit2 is not None:
            s2c = _chi2_scale(fit2["chi2"], int(keep.sum()) - kn - 4)
            clip3 = float((fit2["pj"].chi2_0 - fit2["chi2"]) / s2c)
    else:
        clip3 = float(dchi2_flat_scaled)  # nothing (or too little left) to clip

    # Extra decorrelation regressors: seeing, background, centroid.
    reg_ratio_dchi2 = reg_ratio_depth = np.nan
    regs = None if regressors is None else _regressor_matrix(regressors)
    if regs is not None:
        xr = np.column_stack([x, regs.T])
        try:
            r = _variable_projection_fit(
                tt, y, w, xr, np.array([tc, t14, f]), (fit["lb"], fit["ub"])
            )
            cr, _model, _res, chi2r = _final_solution(tt, y, w, xr, r.x)
            pr = _Projection(xr, y, w)
            reg_dchi2 = (pr.chi2_0 - chi2r) / _chi2_scale(chi2r, n - xr.shape[1] - 3)
            if dchi2_flat_scaled > 0:
                reg_ratio_dchi2 = float(reg_dchi2 / dchi2_flat_scaled)
            if depth > 0:
                reg_ratio_depth = float(cr[-1] / depth)
        except _FIT_ERRORS:
            pass

    return R90Features(
        fit_ok=True, success=bool(fit["success"]), top1_share=top1, top3_share=top3,
        reg_dchi2_ratio=reg_ratio_dchi2, reg_depth_ratio=reg_ratio_depth,
        clip3_dchi2=clip3, dbic_flat=float(dbic_flat),
    )
