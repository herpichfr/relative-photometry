"""Is a per-night transit dip the star's own variability? (``relphot db analyze``)

The two period-based dip tests of :mod:`relphot.db.coincidence`'s ``VARIABILITY`` rule, for a star
whose CATALOGUE period is known. Pure numpy/scipy, no database. Every light curve is the plain
arrays of ONE night: ``t`` BJD_TDB (days), ``flux``, ``flux_err``; :func:`prep_lc` sorts it,
normalises it to its median and drops isolated single-epoch spikes. Depth > 0 is a dimming;
durations are in hours at the interface and days inside.

- :func:`phase_pred_test`: fold the OTHER nights of the object at the catalogue period ``P`` (phase
  binned mean curve, a free additive offset per night), predict the event night from it and
  compare the predicted dip with the observed one;
- :func:`repeat_test`: other transit events of the same object that fall at multiples of ``P``
  (or odd multiples of ``P / 2``, the secondary-eclipse positions) from this one;
- :func:`variability_verdict`: both tests on raw light curves and the decision of the rule.

The period is used as given (no refinement): a catalogue period is precise enough to fold a few
nights. Measured on the live database (2265 search events) this union of the two tests flags 15
events of stars with a literature period (14 eclipsing, 1 RR Lyrae) and none of the stars
without one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from relphot.config import DbSettings

__all__ = [
    "MIN_EVENT_EPOCHS",
    "MIN_OTHER_EPOCHS",
    "VariabilityVerdict",
    "fold_model",
    "phase_pred_test",
    "predict_phase",
    "prep_lc",
    "repeat_test",
    "variability_verdict",
]

#: An event night with fewer epochs (after :func:`prep_lc`) is not judged.
MIN_EVENT_EPOCHS = 12
#: Another night with fewer epochs (after :func:`prep_lc`) does not take part in the fold.
MIN_OTHER_EPOCHS = 8


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def prep_lc(t, flux, err, clip_sigma: float = 5.0):
    """Sort, drop non-finite, median-normalise, remove isolated single-epoch spikes.

    A spike is |y - running-median(5)| > clip_sigma * 1.4826 * MAD of those residuals.
    Returns (t, y, e) float64 arrays, or three empty arrays when < 8 epochs survive.
    """
    t = np.asarray(t, float)
    f = np.asarray(flux, float)
    s = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(s) & (s > 0) & (f > 0)
    t, f, s = t[ok], f[ok], s[ok]
    if t.size < 8:
        return np.empty(0), np.empty(0), np.empty(0)
    o = np.argsort(t)
    t, f, s = t[o], f[o], s[o]
    med = np.median(f)
    y, e = f / med, s / med
    r = y - median_filter(y, size=5, mode="nearest")
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    if sig > 0:
        keep = np.abs(r) <= clip_sigma * sig
        t, y, e = t[keep], y[keep], e[keep]
    if t.size < 8:
        return np.empty(0), np.empty(0), np.empty(0)
    return t, y, e


def _poly(t, deg, tref, scale):
    return np.vander((t - tref) / scale, deg + 1, increasing=True)



def _tscale(t):
    return 0.5 * (t[0] + t[-1]), max(0.5 * (t[-1] - t[0]), 1e-6)



def _box_fixed(t, y, sw, tc, dur_d, deg=2):
    """Fixed-(tc, dur) joint box + poly(deg) fit.  Returns (depth, resid_joint, resid_poly) where
    resids are (y - model) for the joint fit and for the baseline-only fit."""
    tref, scale = _tscale(t)
    P = _poly(t, deg, tref, scale)
    ind = (np.abs(t - tc) <= dur_d / 2.0).astype(float)
    Xj = np.column_stack([P, -ind])
    bj, *_ = np.linalg.lstsq(Xj * sw[:, None], y * sw, rcond=None)
    bp, *_ = np.linalg.lstsq(P * sw[:, None], y * sw, rcond=None)
    return float(bj[-1]), y - Xj @ bj, y - P @ bp


# ----------------------------------------------------------------------------------------------
# phase-prediction test
# ----------------------------------------------------------------------------------------------
def _stack(others):
    ts, ys, es, ks = [], [], [], []
    for k, (t, y, e) in enumerate(others):
        ts.append(t)
        ys.append(y)
        es.append(e)
        ks.append(np.full(len(t), k))
    if not ts:
        z = np.empty(0)
        return z, z, z, np.empty(0, int)
    return (np.concatenate(ts), np.concatenate(ys), np.concatenate(es),
            np.concatenate(ks).astype(int))


def fold_model(t, y, e, k, P, t0, bin_w, min_pts: int = 3, n_iter: int = 6):
    """Phase-binned mean curve g(phase) from several nights with a free additive offset per night.

    ``k`` = night index of each epoch.  Offsets are estimated leave-one-night-out (a night's offset
    is the median of y - g_{-night} over the bins the other nights cover; a night sharing no
    bin with the others keeps offset 0), n_iter alternations.  Returns dict(g, sg, n, nb, c,
    chi2_fold, n_fold) with g/sg/n per bin (g, sg NaN where n < min_pts).
    """
    nn = int(k.max()) + 1 if k.size else 0
    nb = round(1.0 / bin_w)
    phi = ((t - t0) / P) % 1.0
    b = np.minimum((phi * nb).astype(int), nb - 1)
    flat = k * nb + b
    c = np.zeros(nn)
    for _ in range(n_iter):
        z = y - c[k]
        S = np.bincount(flat, weights=z, minlength=nn * nb).reshape(nn, nb)
        N = np.bincount(flat, minlength=nn * nb).reshape(nn, nb)
        St, Nt = S.sum(0), N.sum(0)
        cn = c.copy()
        for j in range(nn):
            m = k == j
            Nj = Nt[b[m]] - N[j, b[m]]
            ok = Nj >= 2
            if ok.sum() >= 3:
                gj = (St[b[m]] - S[j, b[m]]) / np.maximum(Nj, 1)
                cn[j] = np.median((y[m] - gj)[ok])
        c = cn
    z = y - c[k]
    Nt = np.bincount(b, minlength=nb).astype(float)
    St = np.bincount(b, weights=z, minlength=nb)
    Qt = np.bincount(b, weights=z * z, minlength=nb)
    Et = np.bincount(b, weights=e * e, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = St / Nt
        var = np.maximum(Qt / Nt - g * g, 0.0)
        mean_e2 = Et / Nt
        sg = np.sqrt(np.maximum(var, mean_e2) / Nt)
    bad = Nt < min_pts
    gg = np.where(bad, np.nan, g)
    sgg = np.where(bad, np.nan, sg)
    sel = Nt[b] >= 2
    gp = np.where(np.isfinite(g[b]), g[b], 0.0)
    chi2_fold = float((((z - gp) / e) ** 2)[sel].sum())
    return {"g": gg, "sg": sgg, "n": Nt, "nb": nb, "c": c, "chi2_fold": chi2_fold,
            "n_fold": int(sel.sum()), "nb_used": int((Nt >= 2).sum()), "bin_w": bin_w}


def predict_phase(model, phi):
    """Periodic linear interpolation of the binned curve at phases ``phi`` in [0,1).

    Returns (g, sg, covered).  A query is covered when it lies inside a populated bin (nearest
    populated centre within 0.5 bin) or between two populated bins <= 2 bins apart; elsewhere g
    and sg are NaN.
    """
    g, sg, nb, w = model["g"], model["sg"], model["nb"], model["bin_w"]
    v = np.where(np.isfinite(g))[0]
    phi = np.asarray(phi, float)
    nan = np.full(phi.shape, np.nan)
    if v.size < 3:
        return nan, nan.copy(), np.zeros(phi.shape, bool)
    cen = (v + 0.5) / nb
    cen_x = np.concatenate([cen - 1.0, cen, cen + 1.0])
    g_x = np.tile(g[v], 3)
    s_x = np.tile(sg[v], 3)
    idx = np.searchsorted(cen_x, phi)
    lo = np.clip(idx - 1, 0, cen_x.size - 1)
    r = np.clip(idx, 0, cen_x.size - 1)
    dl = phi - cen_x[lo]
    dr = cen_x[r] - phi
    cov = (np.minimum(dl, dr) <= 0.5 * w + 1e-12) | (dl + dr <= 2.0 * w + 1e-12)
    tot = dl + dr
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(tot > 0, dl / tot, 0.0)
    gi = g_x[lo] + (g_x[r] - g_x[lo]) * a
    si = np.maximum(s_x[lo], s_x[r])
    return np.where(cov, gi, np.nan), np.where(cov, si, np.nan), cov


def _bin_width(dur_d, P):
    return float(np.clip(0.5 * dur_d / P, 0.01, 0.05))


def _phase_one(t, y, e, tc, dur_d, others, P):
    """Prediction for the event night at the period ``P``.  Returns dict (see phase_pred_test)."""
    res = {"cov": np.nan, "depth_obs": np.nan, "depth_pred": np.nan, "ratio": np.nan,
           "chi2_pred": np.nan, "chi2_box": np.nan, "chi2_flat": np.nan, "n_win": 0,
           "n_other": 0, "n_nights": 0, "g_ptp": np.nan, "c_event": np.nan}
    to, yo, eo, ko = _stack(others)
    res["n_other"] = int(to.size)
    res["n_nights"] = len(others)
    if to.size < 15 or P <= 0 or not np.isfinite(P):
        return res
    model = fold_model(to, yo, eo, ko, P, tc, _bin_width(dur_d, P))
    gv = model["g"][np.isfinite(model["g"])]
    if gv.size < 3:
        return res
    res["g_ptp"] = float(gv.max() - gv.min())
    phi = ((t - tc) / P) % 1.0
    g, sg, cov = predict_phase(model, phi)
    win = np.abs(t - tc) <= 0.5 * dur_d
    res["n_win"] = int(win.sum())
    if win.sum() < 3:
        return res
    res["cov"] = float(cov[win].mean())
    out = (np.abs(t - tc) >= 0.6 * dur_d) & cov
    if res["cov"] < 0.5 or out.sum() < 3:
        return res
    w = 1.0 / e**2
    c_ev = float(np.sum(w[out] * (y - g)[out]) / np.sum(w[out]))
    res["c_event"] = c_ev
    pred = c_ev + g
    use = cov
    sw = 1.0 / e[use]
    d_obs, r_joint, r_flat = _box_fixed(t[use], y[use], sw, tc, dur_d)
    d_pred, _, _ = _box_fixed(t[use], pred[use], sw, tc, dur_d)
    res["depth_obs"], res["depth_pred"] = d_obs, d_pred
    res["ratio"] = d_pred / d_obs if d_obs > 0 else np.nan
    # chi2 of the event window; common error scale from the joint-fit residuals
    dof = max(use.sum() - 4, 1)
    s2 = max(1.0, float(np.sum((r_joint * sw) ** 2) / dof))
    wu = win[use]
    ee2 = (e[use] ** 2) * s2
    res["chi2_box"] = float(np.sum(r_joint[wu] ** 2 / ee2[wu]))
    res["chi2_flat"] = float(np.sum(r_flat[wu] ** 2 / ee2[wu]))
    rp = (y - pred)[use]
    ep2 = ee2 + np.nan_to_num(sg[use], nan=0.0) ** 2
    res["chi2_pred"] = float(np.sum(rp[wu] ** 2 / ep2[wu]))
    res["s2"] = s2
    return res


def phase_pred_test(t, y, e, tc, dur_h, others, P):
    """Predict the event night from the other nights folded at ``P`` (no period refinement).

    ``others`` = list of (t, y, e) of the other nights (each night-median normalised; a free
    additive offset per night is fitted). The curve g(phase) is a phase-binned mean (bin width
    clip(0.5*dur/P, 0.01, 0.05) cycles, >= 3 points per bin, linear interpolation across at most
    one empty bin). The event night's own offset is fitted on its epochs outside |t-tc| < 0.6 dur
    where the phase is covered. depth_pred / depth_obs are both the joint box(tc, dur)+poly(2)
    depth coefficient evaluated on (prediction | data) at the covered epochs.

    Returns dict: cov (fraction of the in-window epochs whose phase is covered by the other
    nights; ratio etc. are NaN if < 0.5), ratio (= depth_pred / depth_obs), depth_obs, depth_pred,
    chi2_pred / chi2_box / chi2_flat (event window, common error scale), n_win, n_other,
    n_nights, g_ptp (peak-to-peak of the folded curve), c_event.
    """
    return _phase_one(t, y, e, tc, dur_h / 24.0, others, P)


# ----------------------------------------------------------------------------------------------
# repeat test
# ----------------------------------------------------------------------------------------------
def repeat_test(tc, dur_h, other_tc, other_dur_h, P, sigma_P):
    """Other events of the same object compatible with the ephemeris of this one.

    For each other event j: dt = tc_j - tc, and for step = P (integer multiples, n >= 1 in
    absolute value, i.e. before or after) and step = P/2 (odd multiples of P/2 only = the
    secondary-eclipse positions): n = round(dt/step); residual = dt - n step; match when
    |residual| <= tol = 0.5 * max(dur_i, dur_j) + |n| * sigma_step.  Chance probability of a
    match for event j: min(1, 2 tol / P) (the same for the P and the odd-P/2 positions).
    ``p_any`` = 1 - prod(1 - p_j): chance that >= 1 other event matches if events fall at random;
    ``p_tail`` = Poisson-binomial P(>= n_match matches by chance) (1.0 when n_match = 0) -- the
    significance of the observed number of repeats.

    Returns dict n_other, n_match_P, n_match_half, n_match (P or odd P/2), expected (sum p_j),
    p_any, p_tail, best_resid_h (smallest matching |residual|, hours).
    """
    other_tc = np.asarray(other_tc, float)
    other_dur_h = np.asarray(other_dur_h, float)
    res = {"n_other": int(other_tc.size), "n_match_P": 0, "n_match_half": 0, "n_match": 0,
           "expected": np.nan, "p_any": np.nan, "p_tail": np.nan, "best_resid_h": np.nan}
    if other_tc.size == 0 or not np.isfinite(P) or P <= 0:
        return res
    sP = sigma_P if np.isfinite(sigma_P) else 1e-4 * P
    dt = other_tc - tc
    tolbase = 0.5 * np.maximum(dur_h, other_dur_h) / 24.0
    p_all = np.zeros(other_tc.size)
    m_P = np.zeros(other_tc.size, bool)
    m_h = np.zeros(other_tc.size, bool)
    best = np.inf
    for j in range(other_tc.size):
        nP = np.round(dt[j] / P)
        rP = dt[j] - nP * P
        tolP = tolbase[j] + abs(nP) * sP
        nH = np.round(dt[j] / (0.5 * P))
        rH = dt[j] - nH * 0.5 * P
        tolH = tolbase[j] + abs(nH) * 0.5 * sP
        p_all[j] = min(1.0, 2.0 * max(tolP, tolH) / P)
        if nP != 0 and abs(rP) <= tolP:
            m_P[j] = True
            best = min(best, abs(rP))
        elif nH != 0 and int(nH) % 2 != 0 and abs(rH) <= tolH:
            m_h[j] = True
            best = min(best, abs(rH))
    res["n_match_P"] = int(m_P.sum())
    res["n_match_half"] = int(m_h.sum())
    res["n_match"] = int((m_P | m_h).sum())
    res["expected"] = float(p_all.sum())
    res["p_any"] = float(1.0 - np.prod(1.0 - p_all))
    pm = np.zeros(p_all.size + 1)  # Poisson-binomial pmf of the number of chance matches
    pm[0] = 1.0
    for p in p_all:
        pm[1:] = pm[1:] * (1 - p) + pm[:-1] * p
        pm[0] *= 1 - p
    res["p_tail"] = float(pm[res["n_match"]:].sum()) if res["n_match"] > 0 else 1.0
    res["best_resid_h"] = float(best * 24.0) if np.isfinite(best) else np.nan
    return res


# ----------------------------------------------------------------------------------------------
# the rule
# ----------------------------------------------------------------------------------------------
@dataclass(slots=True)
class VariabilityVerdict:
    """What :func:`variability_verdict` found for one event: both tests and whether each fired.

    ``cov`` / ``ratio`` / ``n_nights`` are the phase test's (NaN / 0 when it could not run),
    ``n_match`` / ``p_tail`` the repeat test's (``p_tail`` 1 without a match).
    """

    phase: bool = False
    repeat: bool = False
    n_nights: int = 0
    cov: float = float("nan")
    ratio: float = float("nan")
    n_match: int = 0
    p_tail: float = 1.0

    @property
    def fired(self) -> bool:
        """The rule fires on the phase test, the repeat test or both."""
        return self.phase or self.repeat


def variability_verdict(
    lc: tuple[Sequence[float], Sequence[float], Sequence[float]],
    other_lcs: Sequence[tuple[Sequence[float], Sequence[float], Sequence[float]]],
    tc: float,
    dur_h: float,
    other_tc: Sequence[float],
    other_dur_h: Sequence[float],
    period: float,
    period_err: float | None = None,
    settings: DbSettings | None = None,
) -> VariabilityVerdict:
    """Both dip tests of one event at the catalogue ``period`` (module docstring).

    ``lc`` is the raw (bjd_tdb, flux, flux_err) of the event's night, ``other_lcs`` those of the
    object's other nights (in night order; a night with fewer than ``MIN_OTHER_EPOCHS`` epochs
    after :func:`prep_lc` is left out), ``tc`` / ``dur_h`` the detection's centre time and
    duration, ``other_tc`` / ``other_dur_h`` those of the object's search events on the other
    nights. ``period_err`` NaN or ``None``: 1e-4 * ``period``. An event night with fewer than
    ``MIN_EVENT_EPOCHS`` epochs after :func:`prep_lc` or a non-finite ``tc`` / ``dur_h`` is not
    judged (nothing fires). The phase test fires iff the folded curve covers at least
    ``auto_var_phase_cov_min`` of the in-window epochs, predicts at least
    ``auto_var_phase_ratio_min`` of the observed depth and beats a flat baseline in the event
    window (chi2); the repeat test iff >= 1 other event matches the ephemeris and the chance
    probability of that many matches is at most ``auto_var_repeat_p_max``.
    """
    s = settings if settings is not None else DbSettings()
    verdict = VariabilityVerdict()
    t, y, e = prep_lc(*lc)
    if t.size < MIN_EVENT_EPOCHS or not (np.isfinite(tc) and np.isfinite(dur_h)):
        return verdict
    if not (np.isfinite(period) and period > 0):
        return verdict
    others = []
    for other in other_lcs:
        to, yo, eo = prep_lc(*other)
        if to.size >= MIN_OTHER_EPOCHS:
            others.append((to, yo, eo))
    verdict.n_nights = len(others)
    if others:
        r = phase_pred_test(t, y, e, tc, dur_h, others, period)
        verdict.cov, verdict.ratio = r["cov"], r["ratio"]
        verdict.phase = bool(
            r["cov"] >= s.auto_var_phase_cov_min
            and r["ratio"] >= s.auto_var_phase_ratio_min
            and r["chi2_pred"] <= r["chi2_flat"]
        )
    sigma = period_err if period_err and np.isfinite(period_err) else 1e-4 * period
    q = repeat_test(tc, dur_h, other_tc, other_dur_h, period, sigma)
    verdict.n_match = int(q["n_match"])
    verdict.p_tail = float(q["p_tail"]) if verdict.n_match else 1.0
    verdict.repeat = bool(q["n_match"] >= 1 and q["p_tail"] <= s.auto_var_repeat_p_max)
    return verdict
