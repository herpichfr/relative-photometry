"""The mapping between a night's EXOP verdict and the status of its transit events (pure part)."""

from __future__ import annotations

import pytest

from relphot.objflags import (
    competing_events,
    events_overlap,
    night_state,
    plan_keep,
    plan_night_exop_to_events,
    transit_night_verdict,
)


@pytest.mark.parametrize(
    ("statuses", "verdict"),
    [
        ([], None),
        (["UNCONFIRMED"], None),
        ([None], None),
        (["REJECTED"], "REJECTED"),
        (["REJECTED", "REJECTED"], "REJECTED"),
        (["REJECTED", "UNCONFIRMED"], None),
        (["REJECTED", None], None),
        (["CONFIRMED"], "CONFIRMED"),
        (["CONFIRMED", "REJECTED"], "CONFIRMED"),
        (["CONFIRMED", "UNCONFIRMED", "REJECTED"], "CONFIRMED"),
    ],
)
def test_transit_night_verdict(statuses, verdict) -> None:
    assert transit_night_verdict(statuses) == verdict


def test_plan_night_exop_to_events_rejected_rejects_every_event() -> None:
    events = [(1, "UNCONFIRMED"), (2, "CONFIRMED"), (3, "REJECTED"), (4, None)]
    assert plan_night_exop_to_events(events, None, "REJECTED") == ([1, 2, 4], "REJECTED")
    assert plan_night_exop_to_events(events, "CONFIRMED", "REJECTED") == ([1, 2, 4], "REJECTED")


def test_plan_night_exop_to_events_confirmed_keeps_a_person_s_rejection() -> None:
    events = [(1, "UNCONFIRMED"), (2, "REJECTED"), (3, "CONFIRMED"), (4, None)]
    assert plan_night_exop_to_events(events, None, "CONFIRMED") == ([1, 4], "CONFIRMED")
    # every event rejected: the night cannot be CONFIRMED without one, so all are confirmed
    assert plan_night_exop_to_events([(1, "REJECTED"), (2, "REJECTED")], "REJECTED", "CONFIRMED") \
        == ([1, 2], "CONFIRMED")


def test_plan_night_exop_to_events_clearing_resets_what_carried_the_verdict() -> None:
    events = [(1, "REJECTED"), (2, "CONFIRMED"), (3, "UNCONFIRMED"), (4, "REJECTED")]
    assert plan_night_exop_to_events(events, "REJECTED", None) == ([1, 4], "UNCONFIRMED")
    assert plan_night_exop_to_events(events, "CONFIRMED", None) == ([2], "UNCONFIRMED")


def test_plan_night_exop_to_events_without_a_change_or_events_does_nothing() -> None:
    events = [(1, "REJECTED")]
    for verdict in (None, "CONFIRMED", "REJECTED"):
        assert plan_night_exop_to_events(events, verdict, verdict) == ([], "")
    assert plan_night_exop_to_events([], None, "REJECTED") == ([], "REJECTED")
    assert plan_night_exop_to_events([], "REJECTED", None) == ([], "UNCONFIRMED")


# --------------------------------------------------------------------------
# superseded events: one active event per transit of one light curve
# --------------------------------------------------------------------------


def _ev(det_id, tc, duration_h=2.0, superseded_by=None):
    return {"det_id": det_id, "tc": tc, "duration_h": duration_h, "superseded_by": superseded_by}


def test_events_overlap_is_half_the_longer_duration() -> None:
    day = 2460000.0
    # 2 h and 0.5 h events: the limit is 0.5 * 2 h = 1 h = 0.041667 d
    assert events_overlap(day, 2.0, day + 0.99 / 24.0, 0.5) is True
    assert events_overlap(day, 2.0, day + 1.01 / 24.0, 0.5) is False
    assert events_overlap(day + 0.99 / 24.0, 0.5, day, 2.0) is True  # symmetric
    assert events_overlap(day, 2.0, day, 2.0) is True
    # unknown centre time: overlaps nothing; unknown duration counts as zero
    assert events_overlap(None, 2.0, day, 2.0) is False
    assert events_overlap(day, None, day, None) is True
    assert events_overlap(day, None, day + 0.001, None) is False
    assert events_overlap(day, None, day + 0.5 / 24.0, 1.5) is True


def test_plan_keep_supersedes_the_overlapping_events_and_leaves_other_transits_alone() -> None:
    # 1 = search, 2 = RERUN of the same transit, 3 = another transit of the night
    events = [_ev(1, 100.00), _ev(2, 100.01), _ev(3, 100.30)]
    assert plan_keep(2, events) == {1: 2}
    # the person's other transit is not part of the group
    assert 3 not in plan_keep(1, events)


def test_plan_keep_swaps_back_and_keeps_links_flat() -> None:
    # 2 supersedes 1 and 3 (all one transit)
    events = [_ev(1, 100.00, superseded_by=2), _ev(2, 100.01), _ev(3, 100.02, superseded_by=2)]
    assert plan_keep(1, events) == {1: None, 2: 1, 3: 1}
    assert plan_keep(2, events) == {}  # already the active one
    # a new RERUN (4) overlapping only the active event takes the whole group over
    events.append(_ev(4, 100.011))
    assert plan_keep(4, events) == {1: 4, 2: 4, 3: 4}


def test_plan_keep_group_follows_links_even_without_overlap() -> None:
    # 1 was superseded by 2 although they no longer overlap (e.g. fitted differently): pressing
    # keep on 1 still swaps them back
    events = [_ev(1, 100.00, superseded_by=2), _ev(2, 100.20)]
    assert plan_keep(1, events) == {1: None, 2: 1}
    assert plan_keep(99, events) == {}  # not an event of this light curve


def test_competing_events_lists_each_events_group_without_itself() -> None:
    events = [_ev(1, 100.00, superseded_by=2), _ev(2, 100.01), _ev(3, 100.30)]
    assert competing_events(events) == {1: [2], 2: [1], 3: []}
    assert competing_events([]) == {}


def _d(kind="transit", status=None, origin="search", det_id=None, superseded_by=None, auto=None):
    return {
        "kind": kind, "status": status, "origin": origin, "det_id": det_id,
        "superseded_by": superseded_by, "auto_status": auto,
    }


def test_night_state_superseded_search_event_is_stood_for_by_the_rerun() -> None:
    dets = [_d(det_id=1, superseded_by=2), _d(origin="user", det_id=2)]
    state = night_state(dets, None, None)
    # the RERUN event is the night's open automatic evidence: the night keeps its place
    assert (state["auto_exop"], state["exop_open"], state["pending"]) == (True, True, True)
    # confirmed: no longer open; rejected: no evidence
    assert night_state([dets[0], _d(origin="user", det_id=2, status="CONFIRMED")], None, None)[
        "exop_open"] is False
    rejected = night_state([dets[0], _d(origin="user", det_id=2, status="REJECTED")], None, None)
    assert (rejected["auto_exop"], rejected["pending"]) == (False, False)


def test_night_state_swapped_back_counts_the_search_event_again() -> None:
    dets = [_d(det_id=1), _d(origin="user", det_id=2, superseded_by=1)]
    state = night_state(dets, None, None)
    assert (state["auto_exop"], state["exop_open"]) == (True, True)
    # the superseded status does not count: only the active event's does
    dets[1]["status"] = "CONFIRMED"
    assert night_state(dets, None, None)["exop_open"] is True
    dets[0]["status"] = "REJECTED"
    assert night_state(dets, None, None)["auto_exop"] is False


def test_night_state_a_rerun_of_nothing_the_search_found_is_no_evidence() -> None:
    assert night_state([_d(origin="user", det_id=2)], None, None)["auto_exop"] is False
    # a rerun superseding an auto-rejected search event is no evidence either ...
    dets = [_d(det_id=1, superseded_by=2, auto="REJECTED"), _d(origin="user", det_id=2)]
    assert night_state(dets, None, None)["auto_exop"] is False
    # ... unless a person CONFIRMED that search event
    dets[0]["status"] = "CONFIRMED"
    assert night_state(dets, None, None)["auto_exop"] is True
    # events without ids (the older callers) behave as before
    assert night_state([_d(origin="user")], None, None)["auto_exop"] is False
