"""Tests for relphot.transit_neighbour: the informational NEIGHBOUR_SHARED_EVENT flag."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from relphot.config import Settings
from relphot.cotrend import compute_cbvs
from relphot.transit_neighbour import NeighbourSharedResult, flag_neighbour_shared_events
from relphot.transit_search import (
    FLAG_NAMES,
    FLAG_NEIGHBOUR_SHARED_EVENT,
    FLAG_ON_VARIABLE,
    FLAG_R90_FIT_FAIL,
    HARD_REJECT_FLAGS,
    flags_to_string,
    search_transits,
    tier_for_flags,
)
from tests.test_transit_search import (
    _FakeComparisonResult,
    _limb_darkened_transit,
    _scenario,
)

N_STARS, N_COMP = 100, 80
SOURCE, VICTIM = 98, 99
ARCSEC = 1.0 / 3600.0


def _pair_night(
    sep_arcsec: float = 5.0,
    victim_offset_frac: float = 0.0,
    victim_depth: float = 0.04,
    source_depth: float = 0.02,
    victim_flux: float = 1000.0,
    source_flux: float = 4500.0,
    seed: int = 7,
):
    """A one-tile night with a star pair: SOURCE eclipses, VICTIM dims at the same time.

    All other stars sit 3 arcmin apart (RA steps). The victim's dip is displaced from the
    source's by ``victim_offset_frac`` of the night span.
    """
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=N_STARS, seed=seed
    )
    span = bjd[-1] - bjd[0]
    tc_source = bjd[0] + 0.30 * span
    tc_victim = tc_source + victim_offset_frac * span
    duration = 0.5 / 24.0
    lc[SOURCE, :, 0] *= _limb_darkened_transit(bjd, tc_source, source_depth, duration)
    lc[VICTIM, :, 0] *= _limb_darkened_transit(bjd, tc_victim, victim_depth, duration)

    night.ra = 10.0 + np.arange(N_STARS) * 0.05
    night.dec = np.full(N_STARS, -30.0)
    night.dec[VICTIM] = night.dec[SOURCE] + sep_arcsec * ARCSEC
    night.ra[VICTIM] = night.ra[SOURCE]
    night.flux = np.full((N_STARS, bjd.size, 1), 2000.0)
    night.flux[SOURCE] = source_flux
    night.flux[VICTIM] = victim_flux
    mask = np.zeros((N_STARS, 1), dtype=bool)
    mask[:N_COMP, 0] = True
    return night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask


def _run(scenario, settings=None):
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask = scenario
    settings = settings or Settings()
    comparison_result = _FakeComparisonResult(mask)
    cotrend_result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    transit = search_transits(
        night, tilemap, cotrend_result, comparison_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, settings,
    )
    flags_before = transit.flags.copy()
    shared = flag_neighbour_shared_events(
        night, tilemap, cotrend_result, lc, lc_err, epoch_ok, frame_kept, star_best_aper,
        transit, settings,
    )
    return transit, shared, flags_before


def test_flag_bit_is_informational_and_named() -> None:
    names = dict(FLAG_NAMES)
    assert FLAG_NEIGHBOUR_SHARED_EVENT == 1 << 15
    assert names[FLAG_NEIGHBOUR_SHARED_EVENT] == "NEIGHBOUR_SHARED_EVENT"
    assert FLAG_NEIGHBOUR_SHARED_EVENT & HARD_REJECT_FLAGS == 0
    assert flags_to_string(FLAG_NEIGHBOUR_SHARED_EVENT | FLAG_ON_VARIABLE) == (
        "ON_VARIABLE|NEIGHBOUR_SHARED_EVENT"
    )
    for flags in (0, FLAG_ON_VARIABLE, FLAG_R90_FIT_FAIL):
        assert tier_for_flags(flags | FLAG_NEIGHBOUR_SHARED_EVENT) == tier_for_flags(flags)


def test_settings_defaults_and_toml(tmp_path) -> None:
    from relphot.config import load_settings, settings_from_dict, settings_to_dict

    default = Settings().search
    assert default.neighbour_event_enabled is True
    assert default.neighbour_event_radius_arcsec == 12.0
    assert default.neighbour_event_dip_sigma == 3.0
    path = tmp_path / "nbr.toml"
    path.write_text(
        "[search]\nneighbour_event_enabled = false\nneighbour_event_radius_arcsec = 9.5\n"
    )
    search = load_settings(path).search
    assert search.neighbour_event_enabled is False
    assert search.neighbour_event_radius_arcsec == 9.5
    assert search.neighbour_event_dip_sigma == 3.0
    assert settings_from_dict(settings_to_dict(load_settings(path))).search == search


def test_close_pair_flags_both_members_and_names_the_source() -> None:
    transit, shared, before = _run(_pair_night())
    assert transit.candidate[SOURCE] and transit.candidate[VICTIM]
    for star, other in ((SOURCE, VICTIM), (VICTIM, SOURCE)):
        assert transit.flags[star] & FLAG_NEIGHBOUR_SHARED_EVENT
        assert shared.partner[star] == other
        assert abs(shared.sep_arcsec[star] - 5.0) < 0.01
        assert shared.dip_sigma[star] >= 3.0 and shared.depth[star] > 0
        assert abs(shared.dtc_hours[star]) < 0.25
    # source = larger flux deficit (0.02 * 4500 = 90 against 0.04 * 1000 = 40), one answer per pair
    assert shared.is_source[SOURCE] and not shared.is_source[VICTIM]
    assert np.isclose(shared.deficit_ratio[SOURCE] * shared.deficit_ratio[VICTIM], 1.0)
    assert shared.deficit_ratio[SOURCE] < 1.0
    # nothing else touched: only the two bits, never candidacy or tier
    changed = np.nonzero(transit.flags != before)[0]
    assert changed.tolist() == [SOURCE, VICTIM]
    assert ((transit.flags ^ before)[changed] == FLAG_NEIGHBOUR_SHARED_EVENT).all()
    assert (shared.partner[:SOURCE] == -1).all()
    for star in (SOURCE, VICTIM):
        assert tier_for_flags(int(transit.flags[star])) == tier_for_flags(int(before[star]))


def test_pair_source_is_the_one_with_the_larger_flux_deficit_not_the_deeper_one() -> None:
    # the faint star is deeper (5 % against 1.5 %) but loses fewer counts: 50 against 67.5
    transit, shared, _ = _run(_pair_night(victim_depth=0.05, source_depth=0.015))
    assert transit.candidate[SOURCE] and transit.candidate[VICTIM]
    assert shared.is_source[SOURCE] and not shared.is_source[VICTIM]
    # swap the fluxes and the verdict follows the flux, not the star index
    transit, shared, _ = _run(
        _pair_night(victim_depth=0.05, source_depth=0.015, victim_flux=4500.0, source_flux=1000.0)
    )
    assert shared.is_source[VICTIM] and not shared.is_source[SOURCE]


def test_non_pair_is_not_flagged() -> None:
    # same two events, but the second star is 3 arcmin from the first: independent stars
    scenario = _pair_night()
    scenario[0].ra[VICTIM] = scenario[0].ra[SOURCE] + 0.05
    transit, shared, before = _run(scenario)
    assert transit.candidate[SOURCE] and transit.candidate[VICTIM]
    assert np.array_equal(transit.flags, before)
    assert (shared.partner == -1).all()
    assert not shared.is_source.any()


def test_far_pair_outside_the_radius_is_not_flagged_and_inside_it_is() -> None:
    transit, shared, before = _run(_pair_night(sep_arcsec=30.0))
    assert np.array_equal(transit.flags, before)
    assert (shared.partner == -1).all()
    # the same pair is flagged once the radius reaches it: the radius is the only difference
    wide = Settings()
    wide = replace(wide, search=replace(wide.search, neighbour_event_radius_arcsec=40.0))
    transit, shared, _ = _run(_pair_night(sep_arcsec=30.0), wide)
    assert transit.flags[SOURCE] & transit.flags[VICTIM] & FLAG_NEIGHBOUR_SHARED_EVENT
    assert abs(shared.sep_arcsec[SOURCE] - 30.0) < 0.01


def test_time_offset_pair_is_not_flagged() -> None:
    # two close stars with dips of the same size but 40 % of the night apart (~5 durations)
    transit, shared, before = _run(_pair_night(victim_offset_frac=0.4))
    assert transit.candidate[SOURCE] and transit.candidate[VICTIM]
    assert np.array_equal(transit.flags, before)
    for star in (SOURCE, VICTIM):
        # the neighbour is measured (raw numbers are kept) but is flat in this star's window
        assert shared.partner[star] >= 0
        assert shared.dip_sigma[star] < 3.0
        assert shared.dtc_hours[star] != 0.0


def test_flat_neighbour_is_not_flagged() -> None:
    transit, shared, before = _run(_pair_night(victim_depth=1e-9))
    assert transit.candidate[SOURCE] and not transit.candidate[VICTIM]
    assert np.array_equal(transit.flags, before)
    assert shared.partner[SOURCE] == VICTIM and shared.dip_sigma[SOURCE] < 3.0
    # a non-candidate neighbour is never screened itself
    assert shared.partner[VICTIM] == -1


def test_dim_neighbour_below_the_candidate_threshold_is_still_found() -> None:
    # 0.4 % dip on the neighbour: well under snr_threshold in its own search, obvious at the
    # source's known window
    transit, shared, _ = _run(_pair_night(victim_depth=0.004, sep_arcsec=4.0))
    assert transit.candidate[SOURCE] and not transit.candidate[VICTIM]
    assert transit.flags[SOURCE] & FLAG_NEIGHBOUR_SHARED_EVENT
    assert shared.partner[SOURCE] == VICTIM and shared.dip_sigma[SOURCE] >= 3.0
    assert shared.is_source[SOURCE]
    assert not transit.flags[VICTIM] & FLAG_NEIGHBOUR_SHARED_EVENT  # not a candidate


def test_disabled_is_a_no_op() -> None:
    settings = Settings()
    settings = replace(settings, search=replace(settings.search, neighbour_event_enabled=False))
    transit, shared, before = _run(_pair_night(), settings)
    assert np.array_equal(transit.flags, before)
    empty = NeighbourSharedResult.empty(N_STARS)
    for name in ("partner", "sep_arcsec", "depth", "dip_sigma", "dtc_hours", "deficit_ratio",
                 "is_source"):
        assert np.array_equal(getattr(shared, name), getattr(empty, name), equal_nan=True), name


def test_no_candidates_is_a_no_op() -> None:
    transit, shared, before = _run(_pair_night(source_depth=1e-9, victim_depth=1e-9))
    assert not transit.candidate[[SOURCE, VICTIM]].any()
    assert np.array_equal(transit.flags, before)
    assert (shared.partner == -1).all()
