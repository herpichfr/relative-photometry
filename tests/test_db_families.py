"""Tests for relphot.db.families: repeated transit events of one object, their families and periods.

The pure functions (joint-fit test, ephemeris, non-detection veto, cliques, prediction) need no
database. The rest (``update_families``, the ``analyze`` hook, the CLI, the loaders' rules) needs
RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring) and is skipped with an explicit
reason without it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import psycopg
import pytest

from relphot.cli import _parse_when, main
from relphot.config import DbSettings, Settings
from relphot.db import families as fam
from relphot.db.analyze import analyze
from relphot.db.connect import resolve_dsn
from relphot.db.families import (
    _alias_ks,
    _decision_for,
    _ephemeris,
    _joint_test,
    _nondetection,
    _object_families,
    _p_min_days,
    compute_families,
    load_families,
    predict_windows,
    update_families,
)
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError

T0 = 2460000.0
S = DbSettings()

# --------------------------------------------------------------------------
# synthetic events and light curves
# --------------------------------------------------------------------------


def _trap(t, tc, t14_h, q, depth):
    half = 0.5 * t14_h / 24.0
    tau = max(q * t14_h / 24.0, 1e-4)
    return 1.0 - depth * np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _lc(tc: float, *, t14_h=2.4, q=0.2, depth=0.02, noise=4e-4, seed=1, t_hi=None):
    """``(t, flux, err)`` of a night (0.4 d, 320 frames) holding one trapezoid transit at ``tc``."""
    rng = np.random.default_rng(seed)
    t = np.floor(tc) + np.linspace(0.0, 0.4, 320)
    if t_hi is not None:
        t = t[t <= t_hi]
    y = _trap(t, tc, t14_h, q, depth) + rng.normal(0.0, noise, t.size)
    return t, y, np.full(t.size, noise)


def _ev(det_id: int, night_id: int, tc: float, **kw) -> dict:
    ev = {
        "obj_id": 1, "det_id": det_id, "night_id": night_id, "telescope": "T80S", "tc": tc,
        "tc_err": 5e-4, "depth": 0.02, "depth_err": 6e-4, "t14_h": 2.4, "t14_err": 0.1,
        "t14_lower_limit": False, "ingress_frac": 0.2, "ingress_err": 0.05, "chi2_red": 1.0,
        "det_duration_h": 2.4, "blend": False, "loose": False,
    }
    ev.update(kw)
    return ev


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_p_min_days_scales_with_t14_cubed_density_and_has_a_floor() -> None:
    # the Earth around the Sun: a 13 h transit needs about a year (rho_sun = 1.41, b = 0)
    assert _p_min_days(13.0, 1.41, 0.2) == pytest.approx(365.0, rel=0.05)
    base = _p_min_days(10.0, 5.0, 0.0)
    assert _p_min_days(20.0, 5.0, 0.0) == pytest.approx(8 * base)
    assert _p_min_days(10.0, 10.0, 0.0) == pytest.approx(2 * base)
    assert _p_min_days(0.05, 5.0, 0.2) == 0.2


def test_alias_ks_keep_periods_at_or_above_the_floor() -> None:
    assert _alias_ks(1.0, 0.2, 500).tolist() == [1, 2, 3, 4, 5]
    assert _alias_ks(0.1, 0.2, 500).size == 0
    assert _alias_ks(10.0, 0.2, 7).tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert _alias_ks(-1.0, 0.2, 7).size == 0


def test_decision_is_found_by_nights_and_centre_times_either_way_round() -> None:
    a, b = _ev(1, 10, T0 + 0.2), _ev(2, 11, T0 + 1.2)
    decisions = [
        {"night_a": 10, "night_b": 11, "tc_a": T0 + 0.205, "tc_b": T0 + 1.198, "decision": "SAME"}
    ]
    assert _decision_for(a, b, decisions) == "SAME"
    assert _decision_for(b, a, decisions) == "SAME"
    # half a T14 is the tolerance; a different night pair never matches
    far = _ev(3, 11, T0 + 1.5)
    assert _decision_for(a, far, decisions) is None
    assert _decision_for(a, _ev(4, 12, T0 + 1.2), decisions) is None


# --------------------------------------------------------------------------
# joint-fit likelihood ratio
# --------------------------------------------------------------------------


def test_joint_test_accepts_the_same_shape_and_rejects_a_double_depth() -> None:
    a = _ev(1, 1, T0 + 0.2)
    b = _ev(2, 2, T0 + 3.2)
    lcs = {1: _lc(T0 + 0.2, seed=1), 2: _lc(T0 + 3.2, seed=2)}
    delta, dof, p = _joint_test(a, b, lcs)
    assert dof == 3
    assert p > 0.05
    assert delta < 10

    lcs_deep = {1: lcs[1], 2: _lc(T0 + 3.2, depth=0.04, seed=3)}
    b_deep = _ev(2, 2, T0 + 3.2, depth=0.04)
    delta, dof, p = _joint_test(a, b_deep, lcs_deep)
    assert p < 1e-6
    assert delta > 50


def test_joint_test_does_not_penalise_a_truncated_event_for_its_short_t14() -> None:
    a = _ev(1, 1, T0 + 0.2)
    # the second night stops at the transit centre: only ingress and the first half are seen
    lc_b = _lc(T0 + 3.2, seed=2, t_hi=T0 + 3.2)
    b = _ev(2, 2, T0 + 3.2, t14_h=1.2, t14_lower_limit=True, ingress_frac=None,
            det_duration_h=2.4)
    _delta, dof, p = _joint_test(a, b, {1: _lc(T0 + 0.2, seed=1), 2: lc_b})
    assert dof == 1  # the depth only: T14 is a lower limit, no ingress
    assert p > 0.05


def test_joint_test_is_none_without_a_light_curve_or_enough_points() -> None:
    a, b = _ev(1, 1, T0 + 0.2), _ev(2, 2, T0 + 3.2)
    assert _joint_test(a, b, {1: _lc(T0 + 0.2)}) is None
    thin = (np.array([T0 + 3.2]), np.array([1.0]), np.array([1e-3]))
    assert _joint_test(a, b, {1: _lc(T0 + 0.2), 2: thin}) is None


# --------------------------------------------------------------------------
# ephemeris and non-detection veto
# --------------------------------------------------------------------------


def test_ephemeris_of_a_pair_is_dt_over_k_down_to_the_period_floor() -> None:
    members = [_ev(1, 1, T0), _ev(2, 2, T0 + 2.0, tc_err=1e-3)]
    eph = _ephemeris(members, S)
    assert [e["k"] for e in eph] == list(range(1, 11))
    assert eph[1]["period"] == pytest.approx(1.0)
    assert eph[1]["period_err"] == pytest.approx(math.hypot(2e-3, 2e-3) / 2, rel=0.01)  # floored
    assert min(e["period"] for e in eph) >= S.match_period_min_days
    # too close for any period
    assert _ephemeris([_ev(1, 1, T0), _ev(2, 1, T0 + 0.1)], S) == []


def test_ephemeris_of_three_members_keeps_only_the_aliases_all_of_them_share() -> None:
    members = [_ev(1, 1, T0), _ev(2, 2, T0 + 1.0), _ev(3, 3, T0 + 1.5)]
    eph = _ephemeris(members, S)
    assert sorted(e["k"] for e in eph) == [2, 4]  # P = 0.5 and 0.25 (not 1, 1/3, 0.2)
    p_half = next(e for e in eph if e["k"] == 2)
    assert p_half["period"] == pytest.approx(0.5, abs=1e-3)
    assert 0 < p_half["period_err"] < 1e-3
    assert p_half["tc0"] == pytest.approx(T0, abs=1e-3)
    # a third member off every alias leaves nothing
    assert _ephemeris([*members[:2], _ev(3, 3, T0 + 1.37)], S) == []


_TEMPLATE = {"depth": 0.02, "t14_h": 2.4, "ingress_frac": 0.2}


def _alias(period=1.0, tc0=T0 + 0.2, err=1e-4) -> dict:
    return {"k": 1, "period": period, "period_err": err, "tc0": tc0, "tc0_err": err}


def test_nondetection_vetoes_a_flat_covered_night_and_keeps_a_transit_or_a_gap() -> None:
    members = np.array([T0 + 0.2, T0 + 3.2])
    # predicted at T0+1.2: a flat night there excludes the transit ...
    flat = _lc(T0 + 1.2, depth=0.0, seed=4)
    vetoed, night, dchi2, n = _nondetection(_alias(), _TEMPLATE, members, {7: flat}, S)
    assert (vetoed, night, n) == (True, 7, 1)
    assert dchi2 > S.repeat_veto_dchi2
    # ... a night with the transit in it does not ...
    seen = _lc(T0 + 1.2, depth=0.02, seed=5)
    vetoed, _night, dchi2, _n = _nondetection(_alias(), _TEMPLATE, members, {7: seen}, S)
    assert vetoed is False
    assert dchi2 < 0
    # ... and a night that does not reach the predicted transit tests nothing
    elsewhere = _lc(T0 + 1.2 + 0.45, depth=0.0, seed=6)
    elsewhere = (elsewhere[0] + 0.45 + 0.3, elsewhere[1], elsewhere[2])
    vetoed, night, dchi2, n = _nondetection(_alias(), _TEMPLATE, members, {7: elsewhere}, S)
    assert (vetoed, night, dchi2, n) == (False, None, None, 0)


def test_nondetection_skips_the_members_own_night_epoch() -> None:
    members = np.array([T0 + 0.2, T0 + 3.2])
    # the member's night holds its transit: predicted at T0+0.2 is the member, not a non-detection
    lcs = {1: _lc(T0 + 0.2, seed=1), 2: _lc(T0 + 3.2, seed=2)}
    vetoed, _night, _dchi2, n = _nondetection(_alias(), _TEMPLATE, members, lcs, S)
    assert vetoed is False
    assert n == 0


def test_nondetection_a_timing_uncertain_by_more_than_the_frames_cover_is_not_excluded() -> None:
    members = np.array([T0 + 0.2, T0 + 3.2])
    flat = _lc(T0 + 1.2, depth=0.0, seed=4)
    # +-0.3 d of timing: the transit may sit where the frames are (partly) missing -> tested,
    # but only a timing that fits inside the night can be excluded; an unlucky alias survives
    uncertain = _alias(err=0.2)
    t, y, e = flat
    short = (t[t <= T0 + 1.2 + 0.0], y[t <= T0 + 1.2], e[t <= T0 + 1.2])
    vetoed, *_ = _nondetection(uncertain, _TEMPLATE, members, {7: short}, S)
    assert vetoed is False


# --------------------------------------------------------------------------
# families of one object
# --------------------------------------------------------------------------


def _obj_families(events, decisions=(), lcs=None, s: DbSettings = S):
    return _object_families(1, events, lcs or {}, list(decisions), s)


def _lower(**kw):
    """Events whose T14 is a lower limit and ingress is unknown: the match is on depth alone."""
    return dict(t14_lower_limit=True, ingress_frac=None, ingress_err=None, **kw)


def test_two_alike_events_make_a_family_a_different_depth_does_not() -> None:
    res = _obj_families([
        _ev(1, 1, T0), _ev(2, 2, T0 + 1.0), _ev(3, 3, T0 + 2.0, depth=0.05, depth_err=6e-4),
    ])
    assert res.n_events == 3
    assert len(res.links) == 3
    assert len(res.families) == 1
    f = res.families[0]
    assert f["det_ids"] == [1, 2]
    assert f["member_night_ids"] == [1, 2]
    assert f["depth"] == pytest.approx(0.02)
    assert f["t14_h"] == 2.4
    assert f["t14_lower_limit"] is False
    assert f["accepted"] is False
    # 1.0 d aliases: T14 2.4 h needs P >= 8 d at 5 g/cc, so every alias is density-vetoed ...
    assert {a["status"] for a in f["aliases"]} == {"vetoed_density"}
    assert f["n_allowed"] == 0
    assert f["n_alias"] == len(f["aliases"]) == 5


def test_a_short_event_keeps_its_long_aliases_allowed() -> None:
    # P_min(1 h, 5 g/cc) = 0.59 d: of P = 1.0, 0.5, 0.33, 0.25, 0.2 only the first is long enough
    res = _obj_families([_ev(1, 1, T0, t14_h=1.0), _ev(2, 2, T0 + 1.0, t14_h=1.0)])
    (f,) = res.families
    assert [a["status"] for a in f["aliases"]] == ["allowed"] + ["vetoed_density"] * 4
    assert (f["n_alias"], f["n_allowed"]) == (5, 1)
    assert res.links[0]["phys_ok"] is True and res.links[0]["n_alias"] == 1


def test_overlapping_families_share_an_event_and_a_and_c_never_meet() -> None:
    # A ~ B and B ~ C but A and C differ by more than their errors: two families, one event shared
    events = [
        _ev(1, 1, T0, depth=0.0100, **_lower()),
        _ev(2, 2, T0 + 1.0, depth=0.0112, **_lower()),
        _ev(3, 3, T0 + 2.0, depth=0.0124, **_lower()),
    ]
    res = _obj_families(events)
    assert sorted(f["det_ids"] for f in res.families) == [[1, 2], [2, 3]]
    assert all(f["t14_lower_limit"] is True for f in res.families)
    linked = {(lk["det_a"], lk["det_b"]): lk["linked"] for lk in res.links}
    assert linked == {(1, 2): True, (1, 3): False, (2, 3): True}


def test_a_family_whose_clique_shares_no_ephemeris_is_split() -> None:
    # three alike events at 0, 1.0 and 1.37 d: no period fits all, but every pair is alike
    events = [
        _ev(1, 1, T0, **_lower()), _ev(2, 2, T0 + 1.0, **_lower()),
        _ev(3, 3, T0 + 1.37, **_lower()),
    ]
    res = _obj_families(events)
    assert sorted(f["det_ids"] for f in res.families) == [[1, 2], [1, 3], [2, 3]]


def test_user_decisions_block_or_force_a_link_and_accept_a_family() -> None:
    events = [_ev(1, 1, T0), _ev(2, 2, T0 + 1.0)]
    dec = {"night_a": 1, "night_b": 2, "tc_a": T0, "tc_b": T0 + 1.0}
    assert len(_obj_families(events).families) == 1
    blocked = _obj_families(events, [dict(dec, decision="DIFFERENT")])
    assert blocked.families == []
    assert blocked.links[0]["linked"] is False
    assert blocked.links[0]["decision"] == "DIFFERENT"
    # SAME forces the link although the depths disagree, and the family is accepted
    odd = [_ev(1, 1, T0), _ev(2, 2, T0 + 1.0, depth=0.06)]
    assert _obj_families(odd).families == []
    forced = _obj_families(odd, [dict(dec, decision="SAME")])
    assert len(forced.families) == 1
    assert forced.families[0]["accepted"] is True


def test_t14_lower_limit_is_never_a_reason_to_reject_a_longer_partner() -> None:
    events = [
        _ev(1, 1, T0, t14_h=2.4),
        _ev(2, 2, T0 + 1.0, t14_h=0.9, t14_lower_limit=True, ingress_frac=None),
    ]
    res = _obj_families(events)
    assert len(res.families) == 1
    assert res.families[0]["t14_h"] == 2.4
    assert res.families[0]["t14_lower_limit"] is False
    # the longest member is the incomplete one: the family T14 is a lower limit
    events[1]["t14_h"] = 3.0
    events[0]["t14_h"] = 3.4  # the partner must be no shorter than the lower limit
    events[0]["t14_lower_limit"] = True
    res = _obj_families(events)
    assert res.families[0]["t14_lower_limit"] is True


def test_a_blend_flag_raises_the_depth_floor() -> None:
    # 12 % depth difference: 0.0122 vs 0.0100 is z ~ 4 with the 5 % floor, ~ 2 with the blend one
    a = _ev(1, 1, T0, depth=0.0100, depth_err=2e-4, **_lower())
    b = _ev(2, 2, T0 + 1.0, depth=0.0112, depth_err=2e-4, **_lower())
    assert _obj_families([a, b]).links[0]["p_match"] < S.repeat_p_min
    b["blend"] = True
    assert _obj_families([a, b]).links[0]["p_match"] > S.repeat_p_min


def test_p_joint_uses_the_light_curves_and_can_unlink_a_pair() -> None:
    # equal stored shapes, but the night's data do not hold the same transit
    a, b = _ev(1, 1, T0 + 0.2), _ev(2, 2, T0 + 1.2)
    lcs = {1: _lc(T0 + 0.2, seed=1), 2: _lc(T0 + 1.2, depth=0.0, seed=2)}
    res = _obj_families([a, b], lcs=lcs)
    assert res.links[0]["p_joint"] < 1e-6
    assert res.links[0]["linked"] is False
    assert res.families == []
    ok = _obj_families([a, b], lcs={1: lcs[1], 2: _lc(T0 + 1.2, seed=2)})
    assert ok.links[0]["p_joint"] > 0.05
    assert len(ok.families) == 1


def test_diurnal_flag_and_too_many_events_are_handled() -> None:
    res = _obj_families([_ev(1, 1, T0), _ev(2, 2, T0 + 1.0), _ev(3, 3, T0 + 2.41)])
    flags = {(lk["det_a"], lk["det_b"]): lk["diurnal"] for lk in res.links}
    assert flags[(1, 2)] is True
    assert flags[(1, 3)] is False  # 0.41 d off a whole number of days
    crowded = [_ev(k, k, T0 + k) for k in range(1, 6)]
    skipped = _obj_families(crowded, s=replace(S, repeat_max_family_events=4))
    assert skipped.links == [] and skipped.families == []


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def _family(aliases, **kw) -> dict:
    base = {"obj_id": 7, "fam_id": 3, "depth": 0.02, "t14_h": 2.4, "t14_lower_limit": False,
            "aliases": aliases}
    base.update(kw)
    return base


def _a(period: float, status="allowed", err=1e-4) -> dict:
    return {"k": 1, "period": period, "period_err": err, "tc0": T0, "tc0_err": err,
            "status": status}


def test_predict_windows_counts_the_aliases_that_predict_each_window() -> None:
    family = _family([_a(1.0), _a(0.5), _a(2.0, "vetoed_density")])
    # 20.25-20.75 holds only the 0.5 d alias's transit (at 20.5); the other two are vetoed or off
    (w,) = predict_windows([family], T0 + 20.25, T0 + 20.75)
    assert (w.n_aliases, w.n_aliases_total) == (1, 2)
    assert w.start < T0 + 20.5 < w.end
    # epoch 41: sigma = hypot(1e-4, 41 * 1e-4); half = T14 / 2 + 3 sigma
    expected_half = 1.2 / 24 + 3 * math.hypot(1e-4, 41e-4)
    assert 0.5 * (w.end - w.start) == pytest.approx(expected_half, rel=0.01)
    # 20.9-21.1 holds the transits of both allowed aliases (at 21.0), merged into one window
    (w,) = predict_windows([family], T0 + 20.9, T0 + 21.1)
    assert (w.n_aliases, w.n_aliases_total) == (2, 2)
    assert (w.obj_id, w.fam_id) == (7, 3)
    assert (w.depth, w.t14_h, w.t14_lower_limit) == (0.02, 2.4, False)


def test_predict_windows_widen_with_the_epoch_and_respect_the_status_filter() -> None:
    family = _family([_a(1.0, err=1e-3)])
    near = predict_windows([family], T0 + 1.0, T0 + 1.0)[0]
    far = predict_windows([family], T0 + 101.0, T0 + 101.0)[0]
    assert (far.end - far.start) > (near.end - near.start)
    assert predict_windows([family], T0 + 1.0, T0 + 1.0, statuses=("vetoed_density",)) == []
    assert predict_windows([_family([])], T0, T0 + 5) == []
    windows = predict_windows([family], T0 + 1.0, T0 + 4.0)
    assert len(windows) == 4  # transits at +1, +2, +3, +4
    assert [w.start for w in windows] == sorted(w.start for w in windows)


def test_parse_when_reads_bjd_dates_and_end_of_day() -> None:
    assert _parse_when("2460000.25", end=False) == 2460000.25
    start = _parse_when("2025-12-01", end=False)
    assert _parse_when("2025-12-01", end=True) == pytest.approx(start + 1.0)
    assert _parse_when("2025-12-01T12:00", end=True) == pytest.approx(start + 0.5)
    with pytest.raises(ValueError):
        _parse_when("not a date", end=False)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_repeat_settings_defaults_and_validation() -> None:
    d = DbSettings()
    assert (d.repeat_rho_max_cgs, d.repeat_include_loose, d.repeat_p_min) == (5.0, True, 0.05)
    for bad in (
        {"repeat_p_min": 0.0}, {"repeat_p_min": 1.5}, {"repeat_rho_max_cgs": -1.0},
        {"repeat_veto_dchi2": float("nan")}, {"repeat_max_family_events": 1},
        {"repeat_max_aliases": 2.5}, {"repeat_t14_sys_frac": -0.1},
        {"repeat_n_sigma_window": True},
    ):
        with pytest.raises(ConfigError):
            DbSettings(**bad)


# --------------------------------------------------------------------------
# the database
# --------------------------------------------------------------------------


@pytest.fixture
def test_conn():
    try:
        dsn = resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    init_schema(conn)
    yield conn
    conn.close()


def _night(conn, label: str, telescope: str = "T80S") -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES (%s, '2025-01-01', %s, %s, now()) RETURNING night_id",
            (telescope, label, f"/tmp/{label}"),
        )
        (night_id,) = cur.fetchone()
    return night_id


def _object(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, data_updated_at) "
            "VALUES (%s, 10.0, -20.0, now()) RETURNING obj_id",
            (name,),
        )
        (obj_id,) = cur.fetchone()
    return obj_id


def _lightcurve(conn, obj_id: int, night_id: int, star_id: int, lc: tuple) -> None:
    t, y, e = lc
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night (obj_id, night_id, star_id, tile, mag, best_aperture, "
            "n_epochs) VALUES (%s, %s, %s, 0, 15.0, 1, %s)",
            (obj_id, night_id, star_id, t.size),
        )
        cur.execute(
            "INSERT INTO relphot.lightcurve (obj_id, night_id, frame_index, bjd_tdb, flux, "
            "flux_err, flux_raw) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                obj_id, night_id, list(range(t.size)), [float(v) for v in t],
                [float(v) for v in y], [float(v) for v in e], [float(v) for v in y],
            ),
        )


def _event(
    conn, obj_id: int, night_id: int, tc: float, *, t14_h: float = 1.0, depth: float = 0.02,
    status: str | None = None, auto_status: str | None = None, flags: str | None = None,
    origin: str = "search", converged: bool = True, tc_err: float = 5e-4,
    lower_limit: bool = False, ingress: float | None = 0.2,
) -> int:
    """A per-night transit detection of ``obj_id`` with its trapezoid-fit row."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, origin, status, auto_status, flags) VALUES (%s, %s, 'transit', "
            "10.0, %s, %s, %s, 1, %s, %s, %s, %s) RETURNING det_id",
            (obj_id, night_id, depth, tc, t14_h, origin, status, auto_status, flags),
        )
        (det_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, tc_err, depth, depth_err, "
            "t14_h, t14_err, t14_lower_limit, ingress_frac, ingress_err, chi2_red, n_points, "
            "input, converged, computed_at) VALUES (%s, %s, %s, %s, %s, 0.0006, %s, 0.1, %s, %s, "
            "0.05, 1.0, 80, 'night', %s, now())",
            (det_id, obj_id, tc, tc_err, depth, t14_h, lower_limit, ingress, converged),
        )
    return det_id


def _pair_object(conn, name: str = "target", *, dt: float = 3.0, with_lc: bool = True, **kw):
    """An object with one eligible event on each of two nights, ``dt`` days apart."""
    obj_id = _object(conn, name)
    n1, n2 = _night(conn, f"{name}_1"), _night(conn, f"{name}_2")
    if with_lc:
        _lightcurve(conn, obj_id, n1, 0, _lc(T0 + 0.2, t14_h=1.0, seed=1))
        _lightcurve(conn, obj_id, n2, 1, _lc(T0 + 0.2 + dt, t14_h=1.0, seed=2))
    d1 = _event(conn, obj_id, n1, T0 + 0.2, **kw)
    d2 = _event(conn, obj_id, n2, T0 + 0.2 + dt, **kw)
    conn.commit()
    return obj_id, (n1, n2), (d1, d2)


def _rows(conn, sql: str, params=()) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def test_update_families_stores_links_families_members_and_aliases(test_conn) -> None:
    obj_id, nights, (d1, d2) = _pair_object(test_conn)

    report = update_families(test_conn, [obj_id])
    test_conn.commit()

    assert (report.n_objects, report.n_events, report.n_links, report.n_linked) == (1, 2, 1, 1)
    assert (report.n_families, report.n_objects_with_families) == (1, 1)
    # dt = 3.0 d, T14 1 h: P_min(5 g/cc) = 0.59 d, so k = 1..5 are allowed, k = 6..15 too short
    assert report.alias_status == {"allowed": 5, "vetoed_density": 10}

    (link,) = _rows(
        test_conn,
        "SELECT det_a, det_b, night_a, night_b, dt_days, p_match, p_joint, dof_joint, phys_ok, "
        "n_alias, linked, decision, involves_loose FROM relphot.repeat_link",
    )
    assert link[:4] == (d1, d2, *nights)
    assert link[4] == pytest.approx(3.0)
    assert link[5] > 0.05 and link[6] > 0.05 and link[7] == 3
    assert link[8:] == (True, 5, True, None, False)

    (family,) = _rows(
        test_conn,
        "SELECT fam_id, n_members, member_night_ids, depth, t14_h, t14_lower_limit, n_alias, "
        "n_allowed, accepted, involves_loose, family_key FROM relphot.repeat_family",
    )
    assert family[1:3] == (2, list(nights))
    assert family[3] == pytest.approx(0.02) and family[4] == pytest.approx(1.0)
    assert family[5:10] == (False, 15, 5, False, False)
    assert family[10] == f"{nights[0]}:{T0 + 0.2:.3f}|{nights[1]}:{T0 + 3.2:.3f}"
    assert _rows(test_conn, "SELECT det_id FROM relphot.repeat_family_member ORDER BY det_id") \
        == [(d1,), (d2,)]

    eph = _rows(
        test_conn,
        "SELECT alias_k, period, period_err, tc0, status, fam_id FROM relphot.repeat_ephemeris "
        "ORDER BY alias_k",
    )
    assert len(eph) == 15
    assert eph[0][1] == pytest.approx(3.0) and eph[0][5] == family[0]
    assert eph[2][1] == pytest.approx(1.0)
    assert eph[0][3] == pytest.approx(T0 + 0.2)
    assert [e[4] for e in eph] == ["allowed"] * 5 + ["vetoed_density"] * 10

    # recomputing rewrites the same rows (the ephemeris is keyed, not duplicated)
    update_families(test_conn, [obj_id])
    test_conn.commit()
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_ephemeris") == [(15,)]
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(1,)]

    # and no status or class was touched
    assert _rows(
        test_conn,
        "SELECT count(*) FROM relphot.detection WHERE COALESCE(status, 'UNCONFIRMED') = "
        "'UNCONFIRMED' AND auto_status IS NULL",
    ) == [(2,)]


def test_an_underflowing_joint_probability_is_stored_as_the_real_floor(test_conn) -> None:
    obj_id = _object(test_conn, "underflow")
    n1, n2 = _night(test_conn, "u1"), _night(test_conn, "u2")
    _lightcurve(test_conn, obj_id, n1, 0, _lc(T0 + 0.2, t14_h=1.0, depth=0.05, noise=1e-4, seed=1))
    _lightcurve(test_conn, obj_id, n2, 1, _lc(T0 + 3.2, t14_h=1.0, depth=0.0, noise=1e-4, seed=2))
    _event(test_conn, obj_id, n1, T0 + 0.2, depth=0.05)
    _event(test_conn, obj_id, n2, T0 + 3.2, depth=0.05)
    test_conn.commit()

    update_families(test_conn, [obj_id])  # a float4 column would raise "underflow" otherwise
    test_conn.commit()

    ((p_joint, chi2, linked),) = _rows(
        test_conn, "SELECT p_joint, chi2_joint, linked FROM relphot.repeat_link"
    )
    assert p_joint == pytest.approx(1e-30, rel=1e-3) and chi2 > 1000
    assert linked is False


def test_only_eligible_events_take_part(test_conn) -> None:
    obj_id = _object(test_conn, "elig")
    nights = [_night(test_conn, f"e{k}") for k in range(6)]
    dets = {
        "ok1": _event(test_conn, obj_id, nights[0], T0 + 0.2),
        "ok2": _event(test_conn, obj_id, nights[1], T0 + 3.2),
        "rejected": _event(test_conn, obj_id, nights[2], T0 + 6.2, status="REJECTED"),
        "auto": _event(test_conn, obj_id, nights[3], T0 + 9.2, auto_status="REJECTED"),
        "unconverged": _event(test_conn, obj_id, nights[4], T0 + 12.2, converged=False),
        "confirmed_auto": _event(
            test_conn, obj_id, nights[5], T0 + 15.2, status="CONFIRMED", auto_status="REJECTED"
        ),
    }
    test_conn.commit()

    results = compute_families(test_conn, [obj_id])

    (res,) = results
    assert res.n_events == 3  # the two plain events and the CONFIRMED auto-rejected one
    in_links = {lk["det_a"] for lk in res.links} | {lk["det_b"] for lk in res.links}
    assert in_links == {dets["ok1"], dets["ok2"], dets["confirmed_auto"]}
    assert dets["rejected"] not in in_links and dets["auto"] not in in_links
    # an object with a single eligible event has no result at all
    assert compute_families(test_conn, [_object(test_conn, "lonely")]) == []


def test_loose_night_events_are_members_flagged_unless_excluded(test_conn) -> None:
    obj_id, (_n1, n2), _dets = _pair_object(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, labels, anchor, loaded_at, loose_night_ids) "
            "VALUES ('run', %s, 'a', now(), %s)", (["a", "b"], [n2]),
        )
    test_conn.commit()

    update_families(test_conn, [obj_id])
    test_conn.commit()
    assert _rows(test_conn, "SELECT involves_loose FROM relphot.repeat_link") == [(True,)]
    assert _rows(test_conn, "SELECT involves_loose FROM relphot.repeat_family") == [(True,)]

    update_families(test_conn, [obj_id], replace(S, repeat_include_loose=False))
    test_conn.commit()
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_link") == [(0,)]
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(0,)]


def test_the_nondetection_veto_uses_the_other_nights_of_the_object(test_conn) -> None:
    # events 2 d apart on nights 1 and 3; night 2 (1 d after night 1) is flat where P = 1 d and
    # P = 2/3 d would put a transit (P = 2 d and P = 0.5 d would not be tested there)
    obj_id = _object(test_conn, "veto")
    n1, n2, n3 = (_night(test_conn, f"v{k}") for k in range(3))
    _lightcurve(test_conn, obj_id, n1, 0, _lc(T0 + 0.2, t14_h=1.0, seed=1))
    _lightcurve(test_conn, obj_id, n2, 1, _lc(T0 + 1.2, t14_h=1.0, depth=0.0, seed=2))
    _lightcurve(test_conn, obj_id, n3, 2, _lc(T0 + 2.2, t14_h=1.0, seed=3))
    _event(test_conn, obj_id, n1, T0 + 0.2)
    _event(test_conn, obj_id, n3, T0 + 2.2)
    test_conn.commit()

    update_families(test_conn, [obj_id])
    test_conn.commit()

    status = dict(_rows(test_conn, "SELECT alias_k, status FROM relphot.repeat_ephemeris"))
    assert status[1] == "allowed"  # P = 2 d: transits at 0.2 and 2.2 only, none on night 2
    assert status[2] == "vetoed_nondetection"  # P = 1 d: a transit at 1.2 on the flat night 2
    assert status[3] == "allowed"  # P = 2/3 d: 0.87 and 1.53 are outside night 2's frames
    assert status[4] == "vetoed_density"  # P = 0.5 d < P_min = 0.59 d
    (veto,) = _rows(
        test_conn,
        "SELECT veto_night_id, veto_dchi2, n_nights_tested FROM relphot.repeat_ephemeris "
        "WHERE alias_k = 2",
    )
    assert veto[0] == n2 and veto[1] > S.repeat_veto_dchi2
    # the family is still stored
    assert _rows(test_conn, "SELECT n_alias, n_allowed FROM relphot.repeat_family")[0][0] > 0


def test_a_decision_outlives_a_reload_and_is_attached_by_night_and_time(test_conn) -> None:
    obj_id, (n1, n2), _dets = _pair_object(test_conn, depth=0.02)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.repeat_decision (obj_id, night_a, night_b, tc_a, tc_b, decision)"
            " VALUES (%s, %s, %s, %s, %s, 'DIFFERENT')",
            (obj_id, n1, n2, T0 + 0.2 + 0.004, T0 + 3.2 - 0.003),
        )
    test_conn.commit()
    update_families(test_conn, [obj_id])
    test_conn.commit()
    assert _rows(test_conn, "SELECT linked, decision FROM relphot.repeat_link") == [
        (False, "DIFFERENT")
    ]
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(0,)]

    # a reload replaces the detections; the decision is keyed by night and tc, not detection id
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.detection WHERE night_id IN (%s, %s)", (n1, n2))
    _event(test_conn, obj_id, n1, T0 + 0.2 + 0.002, t14_h=1.0)
    _event(test_conn, obj_id, n2, T0 + 3.2 + 0.001, t14_h=1.0)
    test_conn.commit()
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_link") == [(0,)]  # cascaded
    update_families(test_conn, [obj_id])
    test_conn.commit()
    assert _rows(test_conn, "SELECT linked, decision FROM relphot.repeat_link") == [
        (False, "DIFFERENT")
    ]

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.repeat_decision SET decision = 'SAME'")
    test_conn.commit()
    update_families(test_conn, [obj_id])
    test_conn.commit()
    assert _rows(test_conn, "SELECT accepted FROM relphot.repeat_family") == [(True,)]


def test_ephemeris_history_survives_a_family_that_goes_away_and_comes_back(test_conn) -> None:
    obj_id, _nights, (d1, _d2) = _pair_object(test_conn)
    update_families(test_conn, [obj_id])
    test_conn.commit()
    (fam_id,) = _rows(test_conn, "SELECT fam_id FROM relphot.repeat_family")[0]

    with test_conn.cursor() as cur:  # the person rejects one event: the family is gone
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED' WHERE det_id = %s", (d1,))
    test_conn.commit()
    report = update_families(test_conn, [obj_id])
    test_conn.commit()
    assert report.n_objects == 0
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(0,)]
    assert _rows(test_conn, "SELECT count(*), count(fam_id) FROM relphot.repeat_ephemeris") \
        == [(15, 0)]  # kept as history, no family

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'UNCONFIRMED' WHERE det_id = %s", (d1,))
    test_conn.commit()
    update_families(test_conn, [obj_id])
    test_conn.commit()
    rows = _rows(
        test_conn, "SELECT count(*), count(fam_id), min(fam_id) FROM relphot.repeat_ephemeris"
    )
    assert rows[0][:2] == (15, 15) and rows[0][2] != fam_id  # the same rows, attached again


def test_load_families_and_predict_from_the_stored_rows(test_conn) -> None:
    obj_id, _nights, _dets = _pair_object(test_conn, "predict_me")
    update_families(test_conn, [obj_id])
    test_conn.commit()

    (family,) = load_families(test_conn)
    assert family["obj_name"] == "predict_me" and family["n_members"] == 2
    assert len(family["aliases"]) == 15
    assert load_families(test_conn, telescope="T80S") == [family]
    assert load_families(test_conn, telescope="ROBO43") == []
    assert load_families(test_conn, accepted_only=True) == []

    # allowed aliases: P = 3, 1.5, 1, 0.75, 0.6 from tc0 = T0+0.2; a window on +6.2 holds all five
    windows = predict_windows([family], T0 + 6.0, T0 + 6.4)
    assert [(w.n_aliases, w.n_aliases_total) for w in windows] == [(5, 5)]
    # +5.15 .. +5.3 holds the transit of P = 1.0 d (at 5.2) only
    windows = predict_windows([family], T0 + 5.15, T0 + 5.3)
    assert [(w.n_aliases, w.n_aliases_total) for w in windows] == [(1, 5)]


def test_analyze_computes_the_families_after_the_shapes(test_conn) -> None:
    obj_id = _object(test_conn, "via_analyze")
    n1, n2 = _night(test_conn, "a1"), _night(test_conn, "a2")
    for night_id, star, tc, seed in ((n1, 0, 0.2, 21), (n2, 1, 3.2, 22)):
        t, y, e = _lc(T0 + tc, t14_h=1.0, seed=seed)
        _lightcurve(test_conn, obj_id, night_id, star, (t, y, e))
        with test_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
                "duration_h, tier) VALUES (%s, %s, 'transit', 12.0, 0.02, %s, 1.0, 1)",
                (obj_id, night_id, T0 + tc + 0.002),
            )
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=Settings(), workers=1)
    test_conn.commit()

    assert report.n_transit_shapes == 2
    assert (report.n_repeat_links, report.n_repeat_families) == (1, 1)
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family_member") == [(2,)]
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_ephemeris") == [(15,)]


def test_dry_run_and_predict_cli(test_conn, capsys) -> None:
    obj_id, _nights, _dets = _pair_object(test_conn, "cli_obj")
    dsn = resolve_dsn(env_var="RELPHOT_TEST_DSN")

    assert main(["db", "families", "--dry-run", "--dsn", dsn]) == 0
    out = capsys.readouterr().out
    assert "families=1" in out and "links=1 linked=1" in out
    assert "aliases[allowed=5 vetoed_density=10]" in out and "dry_run=True" in out
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(0,)]  # no write

    # predict computes in memory with --recompute: +6.2 d is the transit of all five aliases
    start = "2460006.0"
    assert main([
        "db", "predict", "--recompute", "--start", start, "--end", "2460006.4", "--csv",
        "--dsn", dsn,
    ]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("obj_id,obj_name,fam_id,start_utc")
    assert len(lines) == 2
    assert lines[1].split(",")[:2] == [str(obj_id), "cli_obj"]
    assert lines[1].split(",")[7:9] == ["5", "5"]
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(0,)]

    assert main(["db", "families", "--dsn", dsn]) == 0
    assert "dry_run=False" in capsys.readouterr().out
    assert _rows(test_conn, "SELECT count(*) FROM relphot.repeat_family") == [(1,)]
    # the stored rows give the same windows, and a filter on the aliases' share drops them
    assert main(["db", "predict", "--start", start, "--end", "2460006.4", "--dsn", dsn]) == 0
    assert "cli_obj" in capsys.readouterr().out
    assert main([
        "db", "predict", "--start", start, "--end", "2460006.4", "--accepted-only", "--dsn", dsn,
    ]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 1  # the header only
    assert main([
        "db", "predict", "--start", "2460006.4", "--end", "2460006.0", "--dsn", dsn,
    ]) == 1


def test_helper_module_names_are_exported() -> None:
    assert {"update_families", "compute_families", "predict_windows", "load_families"} <= set(
        fam.__all__
    )
