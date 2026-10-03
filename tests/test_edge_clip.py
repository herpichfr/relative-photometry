"""Tests for the edge clip: ``edge_outlier_mask``, its use in the search and in the analyze shape
fit, the ``EDGE_OUTLIER`` flag and the shape rules of the automatic verdict (no database)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from test_transit_search import _run_search, _scenario

from relphot.config import DbSettings, SearchSettings, Settings, load_settings
from relphot.db.analyze import _fit_transit_shape, _NightData, _TransitDet, _trapezoid_shape
from relphot.db.coincidence import shape_reasons
from relphot.numeric import edge_outlier_mask, robust_clip_series
from relphot.transit_search import (
    FLAG_EDGE_OUTLIER,
    FLAG_NAMES,
    HARD_REJECT_FLAGS,
    depth_at_other_aperture,
    flags_to_string,
    search_one_star,
    tier_for_flags,
)

T0 = 2460930.5
CADENCE = 2.2 / 1440.0  # days


def _night(n: int = 42, sigma: float = 0.003, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = T0 + np.arange(n) * CADENCE
    return t, 1.0 + rng.normal(0.0, sigma, n), np.full(n, sigma)


def _dropped(t, y, e, **kw) -> list[int]:
    return np.flatnonzero(~edge_outlier_mask(t, y, e, **kw)).tolist()


# --------------------------------------------------------------------------
# 1. edge_outlier_mask
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({-1: 1.22}, [41]),
        ({0: 1.13}, [0]),
        ({0: 1.15, 1: 1.2}, [0, 1]),
        ({-2: 1.2, -1: 1.25}, [40, 41]),
        ({0: 0.8}, [0]),  # two-sided: a low edge epoch too
        ({0: 1.2, -1: 0.82}, [0, 41]),  # both ends, opposite signs
    ],
)
def test_isolated_edge_epochs_are_masked(values, expected) -> None:
    t, y, e = _night()
    for k, v in values.items():
        y[k] = v
    assert _dropped(t, y, e) == expected


def test_a_run_of_three_deviant_epochs_is_never_masked() -> None:
    t, y, e = _night()
    y[-3:] = 1.2
    assert _dropped(t, y, e) == []
    t, y, e = _night()
    y[:3] = 0.8
    assert _dropped(t, y, e) == []


def test_a_real_partial_transit_at_the_night_start_is_not_masked() -> None:
    t, y, e = _night(sigma=0.003)
    y *= 1.0 - 0.03 * _trapezoid_shape(t, t[0] + 3 * CADENCE, 12 * CADENCE, 0.2)
    assert (y[:6] < 0.99).sum() >= 5  # six epochs of the dip, all deep
    assert _dropped(t, y, e) == []


def test_trends_and_ramps_are_not_masked() -> None:
    t, _y, e = _night()
    rng = np.random.default_rng(5)
    for ramp in (0.02, 0.40):
        y = 1.0 + np.linspace(ramp / 2, -ramp / 2, t.size) + rng.normal(0.0, 0.003, t.size)
        assert _dropped(t, y, e) == []
    # an edge outlier on a trend is still found (the reference is the median of the next 6)
    y = 1.0 + np.linspace(0.01, -0.01, t.size) + rng.normal(0.0, 0.003, t.size)
    y[-1] += 0.2
    assert _dropped(t, y, e) == [41]


def test_degenerate_inputs_are_a_no_op() -> None:
    t, y, e = _night(n=14)
    y[-1] = 1.5
    assert edge_outlier_mask(t, y, e).all()  # fewer than min_n epochs
    assert edge_outlier_mask(t, y, e, min_n=10)[-1] == np.False_  # unless allowed
    t, y, e = _night()
    assert edge_outlier_mask(t, np.ones_like(y), e).all()  # constant series
    y[-1] = 1.3
    assert edge_outlier_mask(t, y, e, k_max=0).all()  # switched off
    assert edge_outlier_mask(t[:0], y[:0]).size == 0


def test_nan_epochs_are_kept_and_unsorted_input_gives_the_same_mask() -> None:
    t, y, e = _night()
    y[-1] = 1.25
    ref = edge_outlier_mask(t, y, e)
    assert (~ref).sum() == 1
    perm = np.random.default_rng(1).permutation(t.size)
    assert (edge_outlier_mask(t[perm], y[perm], e[perm]) == ref[perm]).all()
    # a NaN flux/time is never judged nor dropped; the outlier at the finite end is still found
    y2, t2 = y.copy(), t.copy()
    y2[3] = np.nan
    t2[7] = np.nan
    mask = edge_outlier_mask(t2, y2, e)
    assert mask[3] and mask[7]
    assert np.flatnonzero(~mask).tolist() == [41]
    # a NaN as the last epoch: the last finite epoch is the end
    y3 = y.copy()
    y3[-1] = np.nan
    y3[-2] = 1.25
    mask = edge_outlier_mask(t, y3, e)
    assert mask[-1] and not mask[-2]


def test_symmetry_under_flux_inversion_and_the_error_floor() -> None:
    t, y, e = _night()
    y[0], y[-1] = 1.2, 0.85
    assert (edge_outlier_mask(t, y, e) == edge_outlier_mask(t, 2.0 - y, e)).all()
    # errors larger than the deviation raise the scale: nothing is an outlier then
    y = np.ones(t.size)
    y[-1] = 1.05
    assert not edge_outlier_mask(t, y, np.full(t.size, 0.001))[-1]
    assert edge_outlier_mask(t, y, np.full(t.size, 0.02))[-1]


def test_robust_clip_series_is_blind_to_the_edge_epochs() -> None:
    """Characterisation of why the edge clip exists: the same outlier is kept at either end and
    clipped in the middle of the series by the rolling-median clip."""
    t, y, e = _night()
    for k, clipped_mid in ((0, False), (-1, False), (20, True)):
        yy = y.copy()
        yy[k] = 1.22
        keep = robust_clip_series(yy, 6.0, 15)
        assert bool(keep[k]) is not clipped_mid
    yy = y.copy()
    yy[-1] = 1.22
    assert not (robust_clip_series(yy, 6.0, 15) & edge_outlier_mask(t, yy, e))[-1]


# --------------------------------------------------------------------------
# 2. the search
# --------------------------------------------------------------------------

_SEARCH = SearchSettings(min_epochs=20)
_NO_EDGE = replace(_SEARCH, edge_clip_max_epochs=0)


def _star_search(y, e, settings):
    n = y.size
    t = T0 + np.arange(n) * CADENCE
    return t, search_one_star(
        t, y, e, np.ones(n, dtype=bool), np.zeros((0, n)), settings
    )


@pytest.mark.parametrize("where", ["first", "last", "two"])
def test_an_edge_artefact_is_no_longer_a_candidate(where) -> None:
    _, y, e = _night(n=45, sigma=0.004, seed=3)
    if where == "first":
        y[0] = 1.3
    elif where == "last":
        y[-1] = 1.3
    else:
        y[-2:] = [1.25, 1.3]
    t, off = _star_search(y, e, _NO_EDGE)
    assert off.snr > 20  # the artefact manufactures a "dip" out of the rest of the night
    assert off.edge_clip_t.size == 0
    _, on = _star_search(y, e, _SEARCH)
    assert on.snr < _SEARCH.snr_threshold
    expected = {"first": t[:1], "last": t[-1:], "two": t[-2:]}[where]
    assert np.array_equal(on.edge_clip_t, expected)


def test_a_transit_at_the_night_start_is_recovered_unchanged() -> None:
    t, y, e = _night(n=45, sigma=0.003, seed=4)
    y *= 1.0 - 0.02 * _trapezoid_shape(t, t[0] + 2 * CADENCE, 10 * CADENCE, 0.2)
    _, off = _star_search(y, e, _NO_EDGE)
    _, on = _star_search(y, e, _SEARCH)
    assert on.edge_clip_t.size == 0
    assert (on.tc, on.depth, on.snr) == (off.tc, off.depth, off.snr)
    assert on.snr > _SEARCH.snr_threshold


def test_search_transits_sets_edge_outlier_flag_times_and_leaves_the_tier_alone() -> None:
    n_stars, target = 100, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, best, bjd = _scenario(
        n_stars=n_stars, n_frames=60, seed=8, cadence_s=132.0, sigma=0.004
    )
    lc[target, -1, 0] *= 1.3
    mask = np.zeros((n_stars, 1), dtype=bool)
    mask[:80, 0] = True
    result = _run_search(night, tilemap, lc, lc_err, epoch_ok, frame_kept, best, mask)
    assert result.flags[target] & FLAG_EDGE_OUTLIER
    assert result.edge_clip_bjd[target] == (float(bjd[-1]),)
    assert not result.candidate[target]
    others = np.arange(n_stars) != target
    assert sum(1 for i in np.flatnonzero(others) if result.edge_clip_bjd[i]) <= 2
    off = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, best, mask,
        replace(Settings(), search=replace(SearchSettings(), edge_clip_max_epochs=0)),
    )
    assert off.snr[target] > result.snr[target]
    assert not off.flags[target] & FLAG_EDGE_OUTLIER


def test_depth_at_another_aperture_excludes_the_edge_artefact() -> None:
    t, y, e = _night(n=45, sigma=0.003, seed=6)
    tc = t[0] + 0.5 * (t[-1] - t[0])
    y *= 1.0 - 0.02 * _trapezoid_shape(t, tc, 12 * CADENCE, 0.2)
    clean, _ = depth_at_other_aperture(
        t, y, e, np.ones(45, dtype=bool), np.zeros((0, 45)), tc, 12 * CADENCE, _SEARCH
    )
    y[-1] = 1.4
    args = (t, y, e, np.ones(45, dtype=bool), np.zeros((0, 45)), tc, 12 * CADENCE)
    on, _ = depth_at_other_aperture(*args, _SEARCH)
    off, _ = depth_at_other_aperture(*args, _NO_EDGE)
    assert on == pytest.approx(clean, abs=5e-4)
    assert abs(off - clean) > 5 * abs(on - clean)


# --------------------------------------------------------------------------
# flag bit, settings
# --------------------------------------------------------------------------


def test_edge_outlier_flag_bit_is_free_informational_and_named() -> None:
    assert FLAG_EDGE_OUTLIER == 1 << 16
    bits = [b for b, _ in FLAG_NAMES]
    assert len(set(bits)) == len(bits)
    assert all(bin(b).count("1") == 1 for b in bits)
    assert dict(FLAG_NAMES)[FLAG_EDGE_OUTLIER] == "EDGE_OUTLIER"
    assert FLAG_EDGE_OUTLIER & HARD_REJECT_FLAGS == 0
    for flags in (0, 1 << 3, 1 << 8, 1 << 9):
        assert tier_for_flags(flags | FLAG_EDGE_OUTLIER) == tier_for_flags(flags)
    assert flags_to_string(FLAG_EDGE_OUTLIER | (1 << 9)) == "ON_VARIABLE|EDGE_OUTLIER"


def test_edge_clip_settings_defaults_and_toml(tmp_path) -> None:
    s = SearchSettings()
    assert (s.edge_clip_max_epochs, s.edge_clip_ref_epochs, s.lc_clip_sigma) == (2, 6, 6.0)
    d = DbSettings()
    assert (d.auto_nodip_depth, d.auto_nobaseline_min_epochs, d.auto_nobaseline_min_depth) == (
        1e-3, 5, 0.2,
    )
    cfg = tmp_path / "s.toml"
    cfg.write_text(
        "[search]\nedge_clip_max_epochs = 1\nedge_clip_ref_epochs = 8\n"
        "[db]\nauto_nodip_depth = 2e-3\nauto_nobaseline_min_epochs = 4\n"
    )
    loaded = load_settings(cfg)
    assert (loaded.search.edge_clip_max_epochs, loaded.search.edge_clip_ref_epochs) == (1, 8)
    assert (loaded.db.auto_nodip_depth, loaded.db.auto_nobaseline_min_epochs) == (2e-3, 4)


# --------------------------------------------------------------------------
# 3. the analyze shape fit
# --------------------------------------------------------------------------


def _fit(y, tc, dur_h, search=None, n=42, sigma=0.004):
    t = T0 + np.arange(n) * CADENCE
    nd = _NightData(1, date(2025, 1, 1), t, y, np.full(n, sigma))
    det = _TransitDet(det_id=1, night_id=1, tc=tc, depth=0.3, duration_h=dur_h)
    return t, _fit_transit_shape(nd, det, None, None, search)


def test_shape_fit_with_a_hot_last_epoch_has_no_open_box() -> None:
    """The labelled 'bad phot edge' pattern: a +22 % last epoch and a box that stops next to it."""
    t, y, _ = _night(sigma=0.004, seed=0)
    y[-1] = 1.22
    tc = 0.5 * (t[0] + t[-2])
    dur_h = (t[-2] - t[0]) * 24.0
    _, off = _fit(y, tc, dur_h, replace(SearchSettings(), edge_clip_max_epochs=0))
    assert off["converged"] and off["depth"] > 0.3  # the open box
    span_h = (t[-1] - t[0]) * 24.0
    _, on = _fit(y, tc, dur_h)
    assert on["edge_clip_bjd"] == [float(t[-1])]
    assert on["edge_adjacent"] is True
    assert on["t14_h"] <= span_h
    assert on["depth"] < 0.1
    assert not (on["converged"] and on["depth"] > 0.1)


def test_shape_fit_recovers_an_injected_trapezoid_despite_an_edge_outlier() -> None:
    n, depth, tc_off = 42, 0.03, 0.45
    t = T0 + np.arange(n) * CADENCE
    tc = t[0] + tc_off * (t[-1] - t[0])
    rng = np.random.default_rng(1)
    y = (1.0 + rng.normal(0.0, 0.003, n)) * (1.0 - depth * _trapezoid_shape(t, tc, 0.5 / 24, 0.2))
    y[-1] = 1.22
    _, on = _fit(y, tc, 0.5, sigma=0.003)
    assert on["converged"] and on["edge_adjacent"] is False
    assert on["depth"] == pytest.approx(depth, rel=0.1)
    assert on["n_outside"] > 20
    assert shape_reasons(
        on["depth"], on["converged"], on["edge_clip_bjd"], on["edge_adjacent"], on["n_outside"]
    ) == {}
    _, off = _fit(y, tc, 0.5, replace(SearchSettings(), edge_clip_max_epochs=0), sigma=0.003)
    assert abs(off["depth"] - depth) > abs(on["depth"] - depth)  # the unclipped fit is bent


# --------------------------------------------------------------------------
# shape rules, golden synthetic cases
# --------------------------------------------------------------------------


def test_shape_reasons_rules_and_thresholds() -> None:
    assert set(shape_reasons(0.01, True, None, None, 30)) == set()
    assert set(shape_reasons(5e-4, True, None, None, 30)) == {"NO_DIP"}
    assert set(shape_reasons(1e-3, True, None, None, 30)) == set()  # strictly below
    assert set(shape_reasons(0.3, True, None, None, 4)) == {"NO_BASELINE"}
    assert set(shape_reasons(0.3, True, None, None, 5)) == set()
    assert set(shape_reasons(0.2, True, None, None, 0)) == set()  # strictly above
    assert set(shape_reasons(0.3, False, None, None, 0)) == set()  # unconverged fit: no verdict
    assert set(shape_reasons(None, True, None, None, None)) == set()
    r = shape_reasons(0.01, False, [T0, T0 + 1], True, 30)
    assert set(r) == {"EDGE_OUTLIER"} and r["EDGE_OUTLIER"].startswith("edge outlier: 2 ")
    assert shape_reasons(0.01, True, [T0], False, 30) == {}
    assert shape_reasons(0.01, True, [], True, 30) == {}
    both = shape_reasons(5e-4, True, [T0], True, 2)
    assert set(both) == {"EDGE_OUTLIER", "NO_DIP"}
    loose = replace(DbSettings(), auto_nodip_depth=1e-2, auto_nobaseline_min_epochs=40)
    assert set(shape_reasons(5e-3, True, None, None, 30, loose)) == {"NO_DIP"}
    assert shape_reasons(0.3, True, None, None, 30, loose).keys() == {"NO_BASELINE"}


def _golden(kind: str):
    """Synthetic stand-ins of the labelled cases: ``(t, y, tc, det_duration_h)``."""
    rng = np.random.default_rng(11)
    if kind == "11114-like":  # 42 epochs, +22 % last epoch, search box to 0.01 h before the end
        t = T0 + np.arange(42) * CADENCE
        y = 1.0 + rng.normal(0.0, 0.004, 42)
        y[-1] = 1.22
        return t, y, 0.5 * (t[0] + t[-2]), (t[-2] - t[0]) * 24.0
    if kind == "27931-like":  # 66 epochs, +33 % first epoch, short box right after it
        t = T0 + np.arange(66) * CADENCE
        y = 1.0 + rng.normal(0.0, 0.004, 66)
        y[0] = 1.33
        return t, y, 0.5 * (t[1] + t[25]), (t[25] - t[1]) * 24.0
    if kind == "5276-like":  # 69 epochs, a real 5 % transit with baseline on both sides
        t = T0 + np.arange(69) * CADENCE
        tc = t[0] + 0.5 * (t[-1] - t[0])
        dip = 1.0 - 0.05 * _trapezoid_shape(t, tc, 1.1 / 24, 0.2)
        y = (1.0 + rng.normal(0.0, 0.004, 69)) * dip
        return t, y, tc, 1.1
    # 64813-like: 310 epochs of 32 s, a 2 % transit
    t = T0 + np.arange(310) * 32.0 / 86400.0
    tc = t[0] + 0.5 * (t[-1] - t[0])
    dip = 1.0 - 0.02 * _trapezoid_shape(t, tc, 0.61 / 24, 0.27)
    y = (1.0 + rng.normal(0.0, 0.004, 310)) * dip
    return t, y, tc, 0.61


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("11114-like", True), ("27931-like", True), ("5276-like", False), ("64813-like", False)],
)
def test_golden_cases_edge_cases_are_h1_and_real_transits_are_not(kind, expected) -> None:
    t, y, tc, dur_h = _golden(kind)
    nd = _NightData(1, date(2025, 1, 1), t, y, np.full(t.size, 0.004))
    det = _TransitDet(det_id=1, night_id=1, tc=tc, depth=0.3, duration_h=dur_h)
    shape = _fit_transit_shape(nd, det, None, None)
    reasons = shape_reasons(
        shape["depth"], shape["converged"], shape["edge_clip_bjd"], shape["edge_adjacent"],
        shape["n_outside"],
    )
    assert ("EDGE_OUTLIER" in reasons) is expected
    if not expected:
        assert reasons == {} and shape["converged"] and shape["edge_clip_bjd"] is None


# --------------------------------------------------------------------------
# 6. injection / recovery (slow, seeded)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_injection_recovery_loss_and_base_rate() -> None:
    """Edge clip at z = 6: it costs <= 0.2 % of the injected transits that are recovered without
    it, and it clips < 0.5 % of stars with no artefact."""
    rng = np.random.default_rng(2026)
    n_inject = n_clean = 700
    recovered_off = recovered_on = lost = clipped_clean = 0
    for k in range(n_inject + n_clean):
        n = int(rng.integers(40, 70))
        sigma = float(rng.uniform(0.003, 0.006))
        t = T0 + np.arange(n) * CADENCE
        y = 1.0 + rng.normal(0.0, sigma, n)
        e = np.full(n, sigma)
        good = np.ones(n, dtype=bool)
        cbv = np.zeros((0, n))
        if k < n_inject:
            dur = float(rng.choice([0.5, 1.0, 2.0])) / 24.0
            tc = t[0] + float(rng.uniform(-0.3, 1.3)) * (t[-1] - t[0])  # centres incl. partial
            y *= 1.0 - float(rng.choice([0.01, 0.02, 0.05])) * _trapezoid_shape(t, tc, dur, 0.2)
            on = search_one_star(t, y, e, good, cbv, _SEARCH)
            off = search_one_star(t, y, e, good, cbv, _NO_EDGE)
            hit = [
                bool(r.ok and r.snr >= _SEARCH.snr_threshold and abs(r.tc - tc) < dur)
                for r in (off, on)
            ]
            recovered_off += hit[0]
            recovered_on += hit[1]
            lost += hit[0] and not hit[1]
        else:
            clipped_clean += bool(
                search_one_star(t, y, e, good, cbv, _SEARCH).edge_clip_t.size
            )
    assert recovered_off > 150
    assert lost <= 0.002 * recovered_off
    assert clipped_clean < 0.005 * n_clean
