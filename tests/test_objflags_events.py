"""The mapping between a night's EXOP verdict and the status of its transit events (pure part)."""

from __future__ import annotations

import pytest

from relphot.objflags import plan_night_exop_to_events, transit_night_verdict


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
