"""Repeated transit events of one object (``relphot db analyze``, ``relphot db families``).

Two transit events of the SAME object that look alike -- same depth, duration, ingress -- may be
two transits of one planet. This module scores every pair of the object's *eligible* events, joins
mutually linked events into candidate *families*, and gives each family the periods its centre
times allow, so that future nights can be planned to double-check a candidate. Nothing is merged
and no status or class is changed (R4): a family is a scored candidate the person accepts or
rejects (``relphot.repeat_decision``), and an event outside every family stays a possible
additional signal (R1: several families, even overlapping ones, may share an object).

Eligible event: a converged trapezoid fit (``relphot.transit_shape``) of a per-night transit
detection that the person has not REJECTED and that the coincidence check has not auto-rejected
(unless the person CONFIRMED it) -- the rule of :mod:`relphot.objflags` -- and, unless
``db.repeat_include_loose``, that is not on a loose night. The other events of the object, and its
other nights, only enter the non-detection veto.

Per pair (:func:`_score_pairs`):

- ``p_match``: the existing :func:`relphot.db.analyze._match_pairs` probability (depth, T14 and
  ingress z-scores; a lower-limit T14 is one-sided, R6), with ``db.repeat_t14_sys_frac`` as T14
  floor and, for a pair with an event flagged NEIGHBOUR_BLEND / APERTURE_INCONSISTENT, the
  cross-telescope depth floor (dilution depends on the aperture);
- ``p_joint``: a likelihood-ratio test on the light curves, ``chi2`` of ONE common trapezoid
  (depth, T14, ingress; per-event centre time and baseline) minus the sum of the events' own fits,
  with one degree of freedom for the depth, plus one for T14 when no event is incomplete and one
  for the ingress when both have one. An incomplete event only constrains what was observed, so
  its lower-limit T14 is never over-penalised;
- ``n_alias`` / ``phys_ok``: how many periods ``dt / k`` are at least ``P_min``;
- ``linked``: both probabilities >= ``db.repeat_p_min`` (a missing ``p_joint`` is ignored), or the
  person said SAME, and not DIFFERENT. The decision is found by night and centre time, so it
  outlives a night reload.

Families (:func:`_families`) are the maximal cliques of the link graph whose centre times share an
integer-epoch ephemeris (a clique without one is split into its maximal consistent subsets).

Ephemeris (:func:`_ephemeris`): ``P = dt / k`` of the two earliest members for every k with
``P >= match_period_min_days``, refined by a weighted fit when there are more members (and dropped
when a member is off it by more than ``db.repeat_n_sigma_window`` sigma). ``P_min =
pi^2 G rho T14^3 / 3`` (``rho = db.repeat_rho_max_cgs``, T14 the longest member's, a valid bound
even as a lower limit) marks the shorter ones ``vetoed_density``. For the rest, every night of the
object whose light curve covers a predicted transit is tested: the night is flat-fitted and fitted
with the family's template (its smallest depth, its T14) at every timing within the window of
``db.repeat_n_sigma_window`` sigma; if the template is worse by more than ``db.repeat_veto_dchi2``
at EVERY timing the alias is ``vetoed_nondetection``. Families are kept whatever the alias statuses.

:func:`compute_families` only SELECTs; :func:`update_families` rewrites the rows of the given
objects and does not commit. :func:`predict_windows` turns the allowed aliases into the time
windows of the transits expected in a date range.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import psycopg

from relphot.config import DbSettings
from relphot.db.analyze import (
    _good_night_flux,
    _match_pairs,
    _NightData,
    _real_safe,
    _trapezoid_shape,
)
from relphot.repeat import (
    Window,
    load_families,
    match_decision,
    predict_windows,
)

# the decision matching is shared with the web API (relphot.repeat)
_decision_for = match_decision

logger = logging.getLogger(__name__)

__all__ = [
    "FamilyReport",
    "ObjectFamilies",
    "Window",
    "as_family_dicts",
    "compute_families",
    "load_families",
    "predict_windows",
    "summarize",
    "update_families",
]

#: Gravitational constant, cgs.
_G_CGS = 6.674e-8
_BLEND_FLAGS = ("NEIGHBOUR_BLEND", "APERTURE_INCONSISTENT")
_MIN_FIT_POINTS = 8
#: Most timings tried per predicted transit in the non-detection veto.
_MAX_SHIFTS = 41


@dataclass(slots=True)
class FamilyReport:
    """Counts from one :func:`update_families` / :func:`compute_families` call."""

    n_objects: int = 0
    n_events: int = 0
    n_links: int = 0
    n_linked: int = 0
    n_families: int = 0
    n_objects_with_families: int = 0
    #: number of ephemeris rows per status
    alias_status: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectFamilies:
    """Everything computed for one object (plain Python data, no database handles).

    ``links``: one dict per pair (the ``repeat_link`` columns plus ``tc_a`` / ``tc_b``);
    ``families``: one dict per family (``repeat_family`` columns plus ``det_ids`` and
    ``aliases``, the ``repeat_ephemeris`` rows).
    """

    obj_id: int
    n_events: int
    links: list[dict] = field(default_factory=list)
    families: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------


def _p_min_days(t14_h: float, rho_cgs: float, floor_days: float) -> float:
    """Shortest period (days) at which a star of density ``rho_cgs`` shows a ``t14_h``-hour transit.

    Central transit of a small planet: ``T14 = (P / pi) (R / a)`` with ``(a / R)^3 = G rho P^2 /
    (3 pi)``, i.e. ``P = pi^2 G rho T14^3 / 3``. An off-centre transit is shorter, so a longer
    period is needed: the bound holds for any impact parameter, and for a T14 that is only a lower
    limit. Never below ``floor_days``.
    """
    t14_s = t14_h * 3600.0
    return max(floor_days, math.pi**2 * _G_CGS * rho_cgs * t14_s**3 / 3.0 / 86400.0)


def _alias_ks(dt: float, floor_days: float, cap: int) -> np.ndarray:
    """The integers ``k`` with ``dt / k >= floor_days`` (at most ``cap``), ascending."""
    if not (dt > 0 and floor_days > 0 and dt >= floor_days):
        return np.zeros(0, dtype=np.int64)
    # the 1e-9 keeps dt = n * floor_days (0.2 * 5 is not exactly 1.0) from losing its last alias
    return np.arange(1, min(cap, math.floor(dt / floor_days * (1.0 + 1e-9))) + 1)


def _family_key(events: Sequence[dict]) -> str:
    """Stable text key of a family: its members' nights and centre times (to 1e-3 d)."""
    return "|".join(sorted(f"{e['night_id']}:{e['tc']:.3f}" for e in events))


def _event_sigma(ev: dict, s: DbSettings) -> float:
    """Centre-time error of an event (days), floored."""
    err = ev["tc_err"]
    return max(float(err) if err is not None and math.isfinite(err) else 0.0,
               s.repeat_tc_err_floor_days)


# --------------------------------------------------------------------------
# joint-fit likelihood-ratio test
# --------------------------------------------------------------------------


def _event_window(nd: tuple[np.ndarray, np.ndarray, np.ndarray], ev: dict) -> tuple | None:
    """``(t - tc, flux, error)`` of the event's night around its stored ``tc``, or ``None``.

    The same window as the event's own fit (:func:`relphot.db.analyze._fit_transit_shape`); the
    error carries the fit's ``sqrt(max(1, chi2_red))`` inflation.
    """
    t, y, e = nd
    d0 = max(ev["t14_h"], ev["det_duration_h"] or 0.0) / 24.0
    sel = np.abs(t - ev["tc"]) <= max(1.5 * d0, d0 + 1.0 / 24.0)
    if np.count_nonzero(sel) < _MIN_FIT_POINTS:
        return None
    scale = math.sqrt(max(1.0, ev["chi2_red"] or 1.0))
    return t[sel] - ev["tc"], y[sel], e[sel] * scale


def _fit_common(windows: list[tuple], events: list[dict], d0_max: float) -> float | None:
    """Chi2 of one trapezoid (shared depth, T14, ingress; own tc and baseline) over ``windows``.

    ``None`` when the fit fails. With one window this is the event's own five-parameter fit under
    the same bounds, so the difference of the two is a nested likelihood ratio.
    """
    from scipy.optimize import least_squares

    n = len(windows)
    depth0 = float(np.clip(np.mean([ev["depth"] for ev in events]), 1e-4, 0.9))
    ing = [ev["ingress_frac"] for ev in events if ev["ingress_frac"] is not None]
    ing0 = float(np.clip(np.mean(ing), 0.0, 0.5)) if ing else 0.2
    x0 = np.concatenate([[depth0, d0_max, ing0], np.zeros(n), np.ones(n)])
    lower = np.concatenate([[1e-6, 0.3 * d0_max, 0.0], np.full(n, -0.5 * d0_max), np.full(n, 0.1)])
    upper = np.concatenate([[1.0, 3.0 * d0_max, 0.5], np.full(n, 0.5 * d0_max), np.full(n, 10.0)])
    x_scale = np.concatenate([[depth0, d0_max, 0.25], np.full(n, 0.05 * d0_max), np.ones(n)])

    def resid(x: np.ndarray) -> np.ndarray:
        parts = []
        for i, (t_rel, y, e) in enumerate(windows):
            model = x[3 + n + i] * (
                1.0 - x[0] * _trapezoid_shape(t_rel, x[3 + i], x[1], x[2])
            )
            parts.append((y - model) / e)
        return np.concatenate(parts)

    try:
        res = least_squares(resid, x0, bounds=(lower, upper), x_scale=x_scale)
    except (ValueError, np.linalg.LinAlgError):
        logger.debug("common trapezoid fit failed", exc_info=True)
        return None
    chi2 = float(np.sum(res.fun**2))
    return chi2 if math.isfinite(chi2) else None


def _joint_test(
    a: dict, b: dict, night_lc: dict[int, tuple]
) -> tuple[float, int, float] | None:
    """``(delta chi2, dof, p_joint)`` of one common trapezoid for events ``a`` and ``b``.

    ``None`` when either night has no light curve or too few points around its event, or a fit
    fails. dof = 1 (depth) + 1 when neither T14 is a lower limit + 1 when both events have an
    ingress fraction.
    """
    from scipy.stats import chi2 as chi2_dist

    wins = []
    for ev in (a, b):
        nd = night_lc.get(ev["night_id"])
        win = None if nd is None else _event_window(nd, ev)
        if win is None:
            return None
        wins.append(win)
    d0_max = max(max(ev["t14_h"], ev["det_duration_h"] or 0.0) for ev in (a, b)) / 24.0
    c_joint = _fit_common(wins, [a, b], d0_max)
    c_a = _fit_common(wins[:1], [a], d0_max)
    c_b = _fit_common(wins[1:], [b], d0_max)
    if c_joint is None or c_a is None or c_b is None:
        return None
    dof = 1
    dof += int(not (a["t14_lower_limit"] or b["t14_lower_limit"]))
    dof += int(a["ingress_frac"] is not None and b["ingress_frac"] is not None)
    delta = max(c_joint - c_a - c_b, 0.0)
    return delta, dof, float(chi2_dist.sf(delta, dof))


# --------------------------------------------------------------------------
# links
# --------------------------------------------------------------------------


def _score_pairs(
    events: list[dict], night_lc: dict[int, tuple], decisions: list[dict], s: DbSettings
) -> list[dict]:
    """Every pair of ``events`` (ordered by detection id) with its link scores."""
    shapes = [
        {
            "det_id": ev["det_id"], "tc": ev["tc"], "depth": ev["depth"],
            "depth_err": ev["depth_err"], "t14_h": ev["t14_h"], "t14_err": ev["t14_err"],
            "t14_lower_limit": ev["t14_lower_limit"], "ingress_frac": ev["ingress_frac"],
            "ingress_err": ev["ingress_err"], "converged": True,
        }
        for ev in events
    ]
    # A blend-flagged event gets a telescope key of its own, so every pair it is in counts as
    # "cross-telescope" and carries the larger depth floor (the dilution differs per aperture).
    telescope_of = {
        ev["det_id"]: (
            f"{ev['telescope']}|blend|{ev['det_id']}" if ev["blend"] else ev["telescope"]
        )
        for ev in events
    }
    match_settings = replace(s, match_t14_sys_frac=s.repeat_t14_sys_frac)
    by_det = {ev["det_id"]: ev for ev in events}
    floor = s.match_period_min_days
    rows = []
    for m in _match_pairs(shapes, telescope_of, match_settings):
        a, b = by_det[m["det_a"]], by_det[m["det_b"]]
        dt = m["dt_days"]
        t14_max = max(a["t14_h"], b["t14_h"])
        p_min = _p_min_days(t14_max, s.repeat_rho_max_cgs, floor)
        ks = _alias_ks(dt, floor, s.repeat_max_aliases)
        n_alias = int(np.count_nonzero(dt / ks >= p_min)) if ks.size else 0
        joint = _joint_test(a, b, night_lc)
        decision = _decision_for(a, b, decisions)
        p_match = m["p_match"]
        ok = (
            p_match is not None and p_match >= s.repeat_p_min
            and (joint is None or joint[2] >= s.repeat_p_min)
        )
        linked = decision == "SAME" or (ok and decision != "DIFFERENT")
        frac = abs(dt - round(dt))
        rows.append({
            "det_a": a["det_id"], "det_b": b["det_id"], "obj_id": a["obj_id"],
            "night_a": a["night_id"], "night_b": b["night_id"],
            "tc_a": a["tc"], "tc_b": b["tc"], "dt_days": dt, "p_match": p_match,
            "p_joint": None if joint is None else joint[2],
            "chi2_joint": None if joint is None else joint[0],
            "dof_joint": None if joint is None else joint[1],
            "phys_ok": n_alias > 0, "n_alias": n_alias,
            "diurnal": bool(round(dt) >= 1 and frac <= 0.5 * t14_max / 24.0),
            "involves_loose": bool(a["loose"] or b["loose"]),
            "linked": bool(linked), "decision": decision,
        })
    return rows


# --------------------------------------------------------------------------
# ephemeris
# --------------------------------------------------------------------------


def _ephemeris(members: list[dict], s: DbSettings) -> list[dict]:
    """The period aliases whose integer-epoch ephemeris holds for every member (module docstring).

    ``members`` are ordered by ``tc``. Each result has ``k``, ``period``, ``period_err``, ``tc0``
    (the first member's epoch, fitted) and ``tc0_err``; empty when no ``P >= match_period_min_days``
    fits (two members closer than that) or a third member is off every alias.
    """
    tc = np.array([m["tc"] for m in members], dtype=float)
    sig = np.array([_event_sigma(m, s) for m in members], dtype=float)
    dt01 = tc[1] - tc[0]
    out = []
    for k in _alias_ks(dt01, s.match_period_min_days, s.repeat_max_aliases):
        period = dt01 / k
        period_err = math.hypot(sig[0], sig[1]) / k
        epoch = np.rint((tc - tc[0]) / period)
        if np.unique(epoch).size < tc.size:
            continue
        resid = tc - (tc[0] + epoch * period)
        tol = s.repeat_n_sigma_window * np.sqrt(sig**2 + sig[0] ** 2 + (epoch * period_err) ** 2)
        if np.any(np.abs(resid[2:]) > tol[2:]):
            continue
        if tc.size > 2:
            w = 1.0 / sig**2
            design = np.stack([np.ones_like(epoch), epoch], axis=1)
            cov = np.linalg.inv(design.T @ (design * w[:, None]))
            beta = cov @ (design.T @ (w * tc))
            chi2_red = float(np.sum(w * (tc - design @ beta) ** 2)) / (tc.size - 2)
            cov = cov * max(1.0, chi2_red)
            tc0, period = float(beta[0]), float(beta[1])
            tc0_err, period_err = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
        else:
            tc0, tc0_err = float(tc[0]), float(sig[0])
        out.append({
            "k": int(k), "period": period, "period_err": period_err, "tc0": tc0,
            "tc0_err": tc0_err,
        })
    return out


def _nondetection(
    alias: dict, template: dict, member_tcs: np.ndarray, night_lc: dict[int, tuple],
    s: DbSettings,
) -> tuple[bool, int | None, float | None, int]:
    """Whether the light curves exclude the alias: ``(vetoed, night_id, delta chi2, n tested)``.

    Per night and predicted transit that the night's frames cover (at least half the in-transit
    points of a full transit): ``dchi2(shift) = chi2(template at shift) - chi2(flat)`` for timings
    within ``repeat_n_sigma_window`` sigma of the prediction; the transit is excluded on that
    night when the minimum over the timings exceeds ``repeat_veto_dchi2``. A transit predicted on
    the member's own time (within its window) is the member itself and is skipped. The alias is
    vetoed by the night with the largest such minimum. A timing the frames do not cover counts 0.
    """
    period, tc0 = alias["period"], alias["tc0"]
    w14 = template["t14_h"] / 24.0
    best: tuple[float, int] | None = None
    n_tested = 0
    for night_id, (t, y, e) in night_lc.items():
        lo, hi = float(t.min()) - 0.5 * w14, float(t.max()) + 0.5 * w14
        m_lo, m_hi = math.ceil((lo - tc0) / period), math.floor((hi - tc0) / period)
        if m_hi - m_lo > 2000:
            continue
        cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        expected = w14 / cadence if cadence > 0 else 0.0
        w = 1.0 / e**2
        b_flat = float(np.sum(w * y) / np.sum(w))
        chi2_flat = float(np.sum(w * (y - b_flat) ** 2))
        for m in range(m_lo, m_hi + 1):
            centre = tc0 + m * period
            sigma = math.hypot(alias["tc0_err"] or 0.0, abs(m) * (alias["period_err"] or 0.0))
            half = s.repeat_n_sigma_window * sigma
            if np.any(np.abs(member_tcs - centre) <= max(half, 0.5 * w14)):
                continue
            step = max(w14 / 6.0, 2.0 * half / (_MAX_SHIFTS - 1))
            shifts = np.arange(-half, half + 0.5 * step, step) if half > 0 else np.zeros(1)
            shape = _trapezoid_shape(
                t[None, :], (centre + shifts)[:, None], w14, template["ingress_frac"]
            )
            covered = np.count_nonzero(shape > 0.5, axis=1) >= 0.5 * expected
            if expected < 1.0 or not np.any(covered):
                continue
            n_tested += 1
            model_f = 1.0 - template["depth"] * shape
            b = np.sum(w * y * model_f, axis=1) / np.sum(w * model_f**2, axis=1)
            chi2 = np.sum(w * (y - b[:, None] * model_f) ** 2, axis=1)
            dchi2 = np.where(covered, chi2 - chi2_flat, 0.0)
            worst = float(np.min(dchi2))
            if best is None or worst > best[0]:
                best = (worst, night_id)
    if best is not None and best[0] > s.repeat_veto_dchi2:
        return True, best[1], best[0], n_tested
    return False, None if best is None else best[1], None if best is None else best[0], n_tested


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------


def _cliques(nodes: list[int], adj: dict[int, set[int]]) -> list[frozenset[int]]:
    """Maximal cliques of the graph (Bron-Kerbosch with pivoting), size >= 2."""
    found: list[frozenset[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            if len(r) >= 2:
                found.append(frozenset(r))
            return
        pivot = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[pivot]):
            expand(r | {v}, p & adj[v], x & adj[v])
            p.discard(v)
            x.add(v)

    expand(set(), set(nodes), set())
    return found


def _consistent_subsets(
    clique: frozenset[int], events: list[dict], s: DbSettings
) -> list[frozenset[int]]:
    """The maximal subsets of ``clique`` (size >= 2) whose centre times share an ephemeris."""
    seen: set[frozenset[int]] = set()
    good: list[frozenset[int]] = []
    queue = [clique]
    while queue:
        cur = queue.pop()
        if cur in seen or len(cur) < 2:
            continue
        seen.add(cur)
        members = sorted((events[i] for i in cur), key=lambda e: e["tc"])
        if _ephemeris(members, s):
            good.append(cur)
        else:
            queue.extend(cur - {i} for i in cur)
    return [g for g in good if not any(g < other for other in good)]


def _summary(members: list[dict]) -> dict:
    """Depth / T14 / ingress summary of a family's members (see ``repeat_family``)."""
    depth = np.array([m["depth"] for m in members], dtype=float)
    err = np.array(
        [m["depth_err"] if m["depth_err"] and m["depth_err"] > 0 else np.nan for m in members],
        dtype=float,
    )
    if np.all(np.isfinite(err)):
        w = 1.0 / err**2
        mean, mean_err = float(np.sum(w * depth) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))
    else:
        mean, mean_err = float(np.mean(depth)), None
    longest = max(members, key=lambda m: m["t14_h"])
    ing = [m["ingress_frac"] for m in members if m["ingress_frac"] is not None]
    return {
        "depth": mean, "depth_err": mean_err, "t14_h": longest["t14_h"],
        "t14_lower_limit": bool(longest["t14_lower_limit"]),
        "ingress_frac": float(np.mean(ing)) if ing else None,
    }


def _object_families(
    obj_id: int, events: list[dict], night_lc: dict[int, tuple], decisions: list[dict],
    s: DbSettings,
) -> ObjectFamilies:
    """Links, families and aliases of one object's eligible ``events`` (no database access)."""
    result = ObjectFamilies(obj_id=obj_id, n_events=len(events))
    if len(events) < 2:
        return result
    if len(events) > s.repeat_max_family_events:
        logger.warning(
            "object %d: %d eligible transit events (> %d), repeated events skipped",
            obj_id, len(events), s.repeat_max_family_events,
        )
        return result
    events = sorted(events, key=lambda e: e["det_id"])
    result.links = _score_pairs(events, night_lc, decisions, s)
    index = {ev["det_id"]: i for i, ev in enumerate(events)}
    adj: dict[int, set[int]] = {i: set() for i in range(len(events))}
    link_of: dict[frozenset[int], dict] = {}
    for lk in result.links:
        i, j = index[lk["det_a"]], index[lk["det_b"]]
        link_of[frozenset((i, j))] = lk
        if lk["linked"]:
            adj[i].add(j)
            adj[j].add(i)

    subsets: list[frozenset[int]] = []
    for clique in _cliques(list(adj), adj):
        for sub in _consistent_subsets(clique, events, s):
            if sub not in subsets:
                subsets.append(sub)
    subsets = [g for g in subsets if not any(g < other for other in subsets)]

    for sub in sorted(subsets, key=lambda g: sorted(events[i]["tc"] for i in g)):
        members = sorted((events[i] for i in sub), key=lambda e: e["tc"])
        pair_links = [link_of[frozenset(p)] for p in _pairs(sorted(sub))]
        scores = [
            lk["p_joint"] if lk["p_joint"] is not None else lk["p_match"] for lk in pair_links
        ]
        summary = _summary(members)
        p_min = _p_min_days(summary["t14_h"], s.repeat_rho_max_cgs, s.match_period_min_days)
        template = {
            "depth": min(m["depth"] for m in members), "t14_h": summary["t14_h"],
            "ingress_frac": summary["ingress_frac"] if summary["ingress_frac"] is not None
            else 0.2,
        }
        member_tcs = np.array([m["tc"] for m in members], dtype=float)
        aliases = []
        for eph in _ephemeris(members, s):
            row = dict(eph, veto_night_id=None, veto_dchi2=None, n_nights_tested=None)
            if eph["period"] < p_min:
                row["status"] = "vetoed_density"
            else:
                vetoed, night_id, dchi2, n_tested = _nondetection(
                    eph, template, member_tcs, night_lc, s
                )
                row["status"] = "vetoed_nondetection" if vetoed else "allowed"
                row.update(veto_night_id=night_id, veto_dchi2=dchi2, n_nights_tested=n_tested)
            aliases.append(row)
        result.families.append({
            "family_key": _family_key(members), "n_members": len(members),
            "member_night_ids": sorted({m["night_id"] for m in members}),
            "involves_loose": any(m["loose"] for m in members),
            **summary,
            "score": min((x for x in scores if x is not None), default=None),
            "n_alias": len(aliases),
            "n_allowed": sum(1 for a in aliases if a["status"] == "allowed"),
            "accepted": all(lk["decision"] == "SAME" for lk in pair_links),
            "det_ids": [m["det_id"] for m in members], "aliases": aliases,
        })
    return result


def _pairs(ids: list[int]) -> list[tuple[int, int]]:
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

_EVENTS_SQL = """
    SELECT ts.obj_id, ts.det_id, d.night_id, n.telescope, ts.tc, ts.tc_err, ts.depth,
           ts.depth_err, ts.t14_h, ts.t14_err, ts.t14_lower_limit, ts.ingress_frac,
           ts.ingress_err, ts.chi2_red, d.duration_h, d.flags,
           d.night_id IN (SELECT unnest(loose_night_ids) FROM relphot.mn_run) AS loose
    FROM relphot.transit_shape ts
    JOIN relphot.detection d ON d.det_id = ts.det_id
    JOIN relphot.night n ON n.night_id = d.night_id
    WHERE ts.converged AND d.kind = 'transit'
      AND ts.tc IS NOT NULL AND ts.depth > 0 AND ts.t14_h > 0
      AND COALESCE(d.status, 'UNCONFIRMED') <> 'REJECTED'
      AND (d.auto_status IS DISTINCT FROM 'REJECTED' OR d.status = 'CONFIRMED')
      AND ts.obj_id IN (
          SELECT ts2.obj_id FROM relphot.transit_shape ts2
          JOIN relphot.detection d2 ON d2.det_id = ts2.det_id
          WHERE ts2.converged AND COALESCE(d2.status, 'UNCONFIRMED') <> 'REJECTED'
            AND (d2.auto_status IS DISTINCT FROM 'REJECTED' OR d2.status = 'CONFIRMED')
            AND (%(obj_ids)s::bigint[] IS NULL OR ts2.obj_id = ANY(%(obj_ids)s))
          GROUP BY ts2.obj_id HAVING count(*) >= 2
      )
      AND (%(obj_ids)s::bigint[] IS NULL OR ts.obj_id = ANY(%(obj_ids)s))
    ORDER BY ts.obj_id, ts.det_id
"""


def _fetch_events(
    conn: psycopg.Connection, obj_ids: Sequence[int] | None, s: DbSettings
) -> dict[int, list[dict]]:
    """Eligible events by object, for objects with at least two (SELECT only)."""
    events: dict[int, list[dict]] = {}
    with conn.cursor() as cur:
        cur.execute(_EVENTS_SQL, {"obj_ids": None if obj_ids is None else list(obj_ids)})
        for (
            obj_id, det_id, night_id, telescope, tc, tc_err, depth, depth_err, t14_h, t14_err,
            lower, ing, ing_err, chi2_red, det_duration_h, flags, loose,
        ) in cur.fetchall():
            if loose and not s.repeat_include_loose:
                continue
            tokens = set((flags or "").split("|"))
            events.setdefault(obj_id, []).append({
                "obj_id": obj_id, "det_id": det_id, "night_id": night_id,
                "telescope": telescope, "tc": float(tc), "tc_err": tc_err, "depth": float(depth),
                "depth_err": depth_err, "t14_h": float(t14_h), "t14_err": t14_err,
                "t14_lower_limit": bool(lower), "ingress_frac": ing, "ingress_err": ing_err,
                "chi2_red": chi2_red, "det_duration_h": det_duration_h,
                "blend": any(f in tokens for f in _BLEND_FLAGS), "loose": bool(loose),
            })
    return {o: evs for o, evs in events.items() if len(evs) >= 2}


def _fetch_night_lc(
    conn: psycopg.Connection, obj_ids: Sequence[int]
) -> dict[int, dict[int, tuple]]:
    """``{obj_id: {night_id: (t, flux / median, err / median)}}`` with the errors rescaled.

    The rescaling is ``max(1, p2p / median(err))`` with ``p2p`` the robust point-to-point scatter,
    so a night noisier than its errors says (a cloudy loose night) is not over-trusted.
    """
    out: dict[int, dict[int, tuple]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT lc.obj_id, lc.night_id, lc.bjd_tdb, lc.flux, lc.flux_err "
            "FROM relphot.lightcurve lc WHERE lc.obj_id = ANY(%s)",
            (list(obj_ids),),
        )
        for obj_id, night_id, bjd, flux, err in cur.fetchall():
            nd = _NightData(
                night_id=night_id, night_date=None, bjd=np.asarray(bjd, dtype=np.float64),
                flux=np.asarray(flux, dtype=np.float64),
                flux_err=np.asarray(err, dtype=np.float64),
            )
            got = _good_night_flux(nd)
            if got is None or got[0].size < _MIN_FIT_POINTS:
                continue
            t, y, e = got
            order = np.argsort(t)
            t, y, e = t[order], y[order], e[order]
            p2p = 1.4826 * float(np.median(np.abs(np.diff(y)))) / math.sqrt(2.0)
            scale = max(1.0, p2p / float(np.median(e))) if p2p > 0 else 1.0
            out.setdefault(obj_id, {})[night_id] = (t, y, e * scale)
    return out


def _fetch_decisions(conn: psycopg.Connection, obj_ids: Sequence[int]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, night_a, night_b, tc_a, tc_b, decision FROM relphot.repeat_decision "
            "WHERE obj_id = ANY(%s)",
            (list(obj_ids),),
        )
        for obj_id, na, nb, ta, tb, decision in cur.fetchall():
            out.setdefault(obj_id, []).append({
                "night_a": na, "night_b": nb, "tc_a": ta, "tc_b": tb, "decision": decision,
            })
    return out


def compute_families(
    conn: psycopg.Connection,
    obj_ids: Sequence[int] | None = None,
    settings: DbSettings | None = None,
) -> list[ObjectFamilies]:
    """The links, families and aliases of ``obj_ids`` (default: every object) -- SELECTs only.

    Objects with fewer than two eligible events are not in the result.
    """
    s = settings if settings is not None else DbSettings()
    events = _fetch_events(conn, obj_ids, s)
    if not events:
        return []
    ids = sorted(events)
    lcs = _fetch_night_lc(conn, ids)
    decisions = _fetch_decisions(conn, ids)
    return [
        _object_families(o, events[o], lcs.get(o, {}), decisions.get(o, []), s) for o in ids
    ]


def summarize(results: Sequence[ObjectFamilies]) -> FamilyReport:
    """The counts of :func:`compute_families` results."""
    report = FamilyReport(n_objects=len(results))
    for res in results:
        report.n_events += res.n_events
        report.n_links += len(res.links)
        report.n_linked += sum(1 for lk in res.links if lk["linked"])
        report.n_families += len(res.families)
        report.n_objects_with_families += int(bool(res.families))
        for fam in res.families:
            for al in fam["aliases"]:
                report.alias_status[al["status"]] = report.alias_status.get(al["status"], 0) + 1
    return report


_LINK_SQL = """
    INSERT INTO relphot.repeat_link
        (det_a, det_b, obj_id, night_a, night_b, dt_days, p_match, p_joint, chi2_joint,
         dof_joint, phys_ok, n_alias, diurnal, involves_loose, linked, decision, computed_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
"""

_FAMILY_SQL = """
    INSERT INTO relphot.repeat_family
        (obj_id, family_key, n_members, member_night_ids, involves_loose, depth, depth_err,
         t14_h, t14_lower_limit, ingress_frac, score, n_alias, n_allowed, accepted, computed_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
    RETURNING fam_id
"""

_EPHEMERIS_SQL = """
    INSERT INTO relphot.repeat_ephemeris
        (fam_id, obj_id, family_key, alias_k, period, period_err, tc0, tc0_err, status,
         veto_night_id, veto_dchi2, n_nights_tested, computed_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (obj_id, family_key, alias_k) DO UPDATE SET
        fam_id = EXCLUDED.fam_id, period = EXCLUDED.period, period_err = EXCLUDED.period_err,
        tc0 = EXCLUDED.tc0, tc0_err = EXCLUDED.tc0_err, status = EXCLUDED.status,
        veto_night_id = EXCLUDED.veto_night_id, veto_dchi2 = EXCLUDED.veto_dchi2,
        n_nights_tested = EXCLUDED.n_nights_tested, computed_at = now()
"""


def _real(value: float | None) -> float | None:
    """``value`` for a ``real`` column: ``None`` stays, an underflowing probability is floored."""
    return None if value is None else _real_safe(value)


def update_families(
    conn: psycopg.Connection,
    obj_ids: Sequence[int],
    settings: DbSettings | None = None,
) -> FamilyReport:
    """Recompute the repeated-event rows of every object in ``obj_ids`` (module docstring).

    The objects' ``repeat_link`` and ``repeat_family`` rows (members with them) are deleted and
    written afresh, also for an object that no longer has two eligible events; their
    ``repeat_ephemeris`` rows stay as history (``fam_id`` NULL) unless the family came back.
    ``repeat_decision`` is read, never written. Does not commit. The report counts the objects
    that have at least two eligible events.
    """
    s = settings if settings is not None else DbSettings()
    ids = sorted({int(o) for o in obj_ids})
    if not ids:
        return FamilyReport()
    results = compute_families(conn, ids, s)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.repeat_link WHERE obj_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM relphot.repeat_family WHERE obj_id = ANY(%s)", (ids,))
        for res in results:
            if res.links:
                cur.executemany(_LINK_SQL, [
                    (
                        lk["det_a"], lk["det_b"], res.obj_id, lk["night_a"], lk["night_b"],
                        lk["dt_days"], _real(lk["p_match"]), _real(lk["p_joint"]),
                        _real(lk["chi2_joint"]),
                        lk["dof_joint"], lk["phys_ok"], lk["n_alias"], lk["diurnal"],
                        lk["involves_loose"], lk["linked"], lk["decision"],
                    )
                    for lk in res.links
                ])
            for fam in res.families:
                cur.execute(_FAMILY_SQL, (
                    res.obj_id, fam["family_key"], fam["n_members"], fam["member_night_ids"],
                    fam["involves_loose"], _real(fam["depth"]), _real(fam["depth_err"]),
                    _real(fam["t14_h"]), fam["t14_lower_limit"], _real(fam["ingress_frac"]),
                    _real(fam["score"]), fam["n_alias"],
                    fam["n_allowed"], fam["accepted"],
                ))
                (fam_id,) = cur.fetchone()
                cur.executemany(
                    "INSERT INTO relphot.repeat_family_member (fam_id, det_id) VALUES (%s, %s)",
                    [(fam_id, det_id) for det_id in fam["det_ids"]],
                )
                if fam["aliases"]:
                    cur.executemany(_EPHEMERIS_SQL, [
                        (
                            fam_id, res.obj_id, fam["family_key"], al["k"], al["period"],
                            al["period_err"], al["tc0"], al["tc0_err"], al["status"],
                            al["veto_night_id"], _real(al["veto_dchi2"]), al["n_nights_tested"],
                        )
                        for al in fam["aliases"]
                    ])
    report = summarize(results)
    logger.info(
        "repeated events: %d objects, %d links (%d linked), %d families on %d objects",
        report.n_objects, report.n_links, report.n_linked, report.n_families,
        report.n_objects_with_families,
    )
    return report


def as_family_dicts(conn: psycopg.Connection, results: Sequence[ObjectFamilies]) -> list[dict]:
    """The families of :func:`compute_families` results in the shape of :func:`load_families`.

    ``fam_id`` is ``None`` (nothing is stored); ``obj_name`` is looked up (SELECT only).
    """
    ids = [res.obj_id for res in results if res.families]
    names: dict[int, str] = {}
    if ids:
        with conn.cursor() as cur:
            cur.execute("SELECT obj_id, name FROM relphot.object WHERE obj_id = ANY(%s)", (ids,))
            names = dict(cur.fetchall())
    return [
        dict(fam, fam_id=None, obj_id=res.obj_id, obj_name=names.get(res.obj_id))
        for res in results for fam in res.families
    ]
