"""Tests (live test database, see tests/test_db_schema.py) for the ``VARIABILITY`` rule of the
automatic verdict pass of relphot.db.coincidence: a search transit event whose star's CATALOGUE
period explains the dip (the other nights folded at the period predict it, or other events of the
star repeat at multiples of the period) is auto-rejected. The tests of the numerics are in
tests/test_dip_variability.py."""

from __future__ import annotations

import numpy as np
import psycopg
import pytest
from test_db_analyze import _insert_night, _insert_object, _insert_star_night_and_lc
from test_db_coincidence import T0 as DB_T0
from test_db_coincidence import _detection, _event, _night, _object
from test_dip_variability import DUR_H, PERIOD, T0, _eclipses, _event_night
from test_dip_variability import _night as _lc_night

from relphot.config import Settings
from relphot.db.analyze import analyze
from relphot.db.coincidence import update_auto_verdicts, variability_reasons
from relphot.db.connect import resolve_dsn
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError
from relphot.web.app import _effective_status

assert DB_T0 == T0  # the two test modules share one time origin

TC = T0 + 8 * PERIOD  # an eclipse of the synthetic binary
_STARTS = (0.0, 0.11, 0.23, 0.05, 0.31)


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


def _nights(conn, n: int = 5) -> list[int]:
    """``n`` nights (no frames: the coincidence rule has nothing to judge); the first is the one
    the event is on."""
    return [_night(conn, f"2025010{k + 1}", frames=False) for k in range(n)]


def _catalog(conn, obj_id: int, catalog: str, name: str, type_: str | None = None,
             period: float | None = None, period_err: float | None = None,
             sep: float | None = 1.0) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.catalog_match (obj_id, catalog, name, type, period, period_err,"
            " sep_arcsec) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (obj_id, catalog, name, type_, period, period_err, sep),
        )


def _put_lc(conn, obj_id: int, night_id: int, star_id: int, lc) -> None:
    _insert_star_night_and_lc(conn, obj_id, night_id, star_id, *lc)


def _binary(conn, nights: list[int], name: str, *, period: float | None = PERIOD,
            catalog: str = "VSX", extra=(), status: str | None = None) -> tuple[int, int]:
    """An eclipsing binary (dip of ``DUR_H`` every ``PERIOD``) with a light curve on every night
    and a search event at an eclipse on the first one; ``period`` / ``catalog`` give its
    catalogue row (``period=None``: no row). Returns (obj_id, det_id)."""
    obj_id = _object(conn, name)
    flux_of = _eclipses()
    _put_lc(conn, obj_id, nights[0], obj_id * 10, _event_night(flux_of, TC))
    for k, (night_id, start) in enumerate(zip(nights[1:], _STARTS, strict=False), start=1):
        _put_lc(conn, obj_id, night_id, obj_id * 10 + k,
                _lc_night(k, flux_of, seed=10 + k, start=start))
    if period is not None:
        _catalog(conn, obj_id, catalog, f"{name} cat", "EA", period)
    for row in extra:
        _catalog(conn, obj_id, *row)
    det_id = _event(conn, obj_id, nights[0], TC, DUR_H, 0.15, status=status)
    conn.commit()
    return obj_id, det_id


def _flat_events(conn, nights: list[int], name: str, period: float) -> tuple[int, list[int]]:
    """A star with flat light curves and one search event per night, at multiples of ``period``
    from each other (nights are whole days apart, ``period`` divides a day)."""
    obj_id = _object(conn, name)
    dets = []
    for k, night_id in enumerate(nights[:4]):
        _put_lc(conn, obj_id, night_id, obj_id * 10 + k,
                _lc_night(k, lambda t: np.ones_like(t), seed=30 + k))
        dets.append(_event(conn, obj_id, night_id, T0 + k + 0.2, 1.0, 0.01,
                           status="REJECTED" if k == 1 else None))
    _catalog(conn, obj_id, "VSX", f"{name} cat", "RRAB", period)
    conn.commit()
    return obj_id, dets


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def test_the_catalogue_period_rejects_an_eclipse_and_only_that_event(test_conn) -> None:
    nights = _nights(test_conn)
    obj, det = _binary(test_conn, nights, "binary")
    _plain_obj, plain = _binary(test_conn, nights, "no catalogue row", period=None)

    report = update_auto_verdicts(test_conn, nights)
    test_conn.commit()

    status, auto, reason = _detection(test_conn, det)
    assert (status, auto) == (None, "REJECTED")  # the person's status is never touched
    assert reason.startswith(
        "variability: the catalogue period P=0.6000 d (VSX) predicts this dip from 4 other nights "
        "folded (predicted/observed depth "
    )
    assert reason.endswith(", phase coverage 100 %)")
    assert "repeats" not in reason
    assert _detection(test_conn, plain) == (None, None, None)
    assert report.n_variability == 1 and report.n_rejected == 1
    assert report.n_coincidence == report.n_no_dip == report.n_no_baseline == 0
    assert report.changed_obj_ids == [obj]


def test_the_period_of_the_nearest_eligible_row_is_used(test_conn) -> None:
    nights = _nights(test_conn)
    # the nearest row (0.5") has the right period, a farther Gaia row a wrong one
    _obj, near_right = _binary(test_conn, nights, "near right", catalog="Gaia DR3",
                               extra=(("VSX", "far", "EA", 0.9, None, 3.0),))
    assert update_auto_verdicts(test_conn, nights).n_variability == 1
    assert "(Gaia DR3)" in _detection(test_conn, near_right)[2]


def test_a_wrong_nearest_period_does_not_reject(test_conn) -> None:
    nights = _nights(test_conn)
    _obj, det = _binary(test_conn, nights, "near wrong", period=0.9,
                        extra=(("Gaia DR3", "far", "EA", PERIOD, None, 3.0),))
    report = update_auto_verdicts(test_conn, nights)
    assert report.n_variability == 0 and _detection(test_conn, det) == (None, None, None)


@pytest.mark.parametrize(
    "extra",
    [
        (("NASA Exoplanet Archive", "planet b", None, None, None, 0.5),),
        (("TOI", "TOI-1", None, None, None, 0.5),),
        (("NASA Exoplanet Archive", "planet b", None, PERIOD, None, 0.5),),
    ],
)
def test_a_known_planet_host_is_never_judged(test_conn, extra) -> None:
    nights = _nights(test_conn)
    _obj, det = _binary(test_conn, nights, "host", extra=extra)
    report = update_auto_verdicts(test_conn, nights)
    test_conn.commit()
    assert report.n_variability == 0 and report.n_rejected == 0
    assert _detection(test_conn, det) == (None, None, None)
    assert report.changed_obj_ids == []
    with test_conn.cursor() as cur:
        assert variability_reasons(cur, nights[0]) == {}


@pytest.mark.parametrize(
    "rows",
    [
        [("VSX", "x", "EP", PERIOD, None, 1.0)],  # an exoplanet transit: a transit period
        [("VSX", "x", "ROT|EP", PERIOD, None, 1.0)],
        [("VSX", "x", "EA", None, None, 1.0)],  # no period
        [("VSX", "x", "EA", 0.0, None, 1.0)],
        [("VSX", "x", "EA", -PERIOD, None, 1.0)],
    ],
)
def test_a_row_without_a_usable_period_or_with_a_planet_type_is_ignored(test_conn, rows) -> None:
    nights = _nights(test_conn)
    _obj, det = _binary(test_conn, nights, "unusable", period=None, extra=rows)
    assert update_auto_verdicts(test_conn, nights).n_variability == 0
    assert _detection(test_conn, det) == (None, None, None)


def test_a_type_that_only_contains_the_letters_ep_is_not_a_planet_type(test_conn) -> None:
    nights = _nights(test_conn)
    _obj, det = _binary(test_conn, nights, "rep", period=None,
                        extra=(("VSX", "x", "REP|DEP", PERIOD, None, 1.0),))
    assert update_auto_verdicts(test_conn, nights).n_variability == 1
    assert _detection(test_conn, det)[1] == "REJECTED"


def test_repeated_events_at_multiples_of_the_period_reject_every_one_of_them(test_conn) -> None:
    nights = _nights(test_conn)
    obj, dets = _flat_events(test_conn, nights, "repeater", 0.5)

    report = update_auto_verdicts(test_conn, nights)
    test_conn.commit()

    assert report.n_variability == 4 and report.n_rejected == 4
    for det in dets:
        _status, auto, reason = _detection(test_conn, det)
        assert auto == "REJECTED"
        assert reason.startswith(
            "variability: the dip repeats at multiples of the catalogue period P=0.5000 d (VSX): "
            "3 other events at n*P (or odd n*P/2), chance p="
        )
    # a person's REJECTED stays theirs; the verdict is only annotated
    assert _detection(test_conn, dets[1])[:2] == ("REJECTED", "REJECTED")
    assert report.changed_obj_ids == [obj]


def test_phase_and_repeat_are_both_named_when_both_fire(test_conn) -> None:
    nights = _nights(test_conn)
    obj, det = _binary(test_conn, nights, "both", period=PERIOD)
    # other eclipses of the star on the other nights, one period multiple apart
    for k in (1, 2, 3):
        _event(test_conn, obj, nights[k], TC + k * 6 * PERIOD, DUR_H, 0.15)
    test_conn.commit()
    update_auto_verdicts(test_conn, nights)
    reason = _detection(test_conn, det)[2]
    assert reason.startswith("variability: the catalogue period P=0.6000 d (VSX) predicts this "
                             "dip from 4 other nights folded (predicted/observed depth ")
    assert " and the dip repeats at multiples of it: 3 other events at n*P (or odd n*P/2), " \
           "chance p=" in reason


def test_a_vetted_event_is_not_judged_but_still_counts_as_another_event(test_conn) -> None:
    nights = _nights(test_conn)
    _obj, dets = _flat_events(test_conn, nights, "vetted repeater", 0.5)
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET auto_status = 'REJECTED', auto_reason = 'old' "
                    "WHERE det_id = %s", (dets[0],))
    test_conn.commit()

    report = update_auto_verdicts(test_conn, nights, skip_det_ids={dets[0]})
    test_conn.commit()

    assert _detection(test_conn, dets[0]) == (None, "REJECTED", "old")  # exactly as stored
    assert report.n_variability == 3
    # the vetted event is one of the three others the remaining events repeat with
    assert "3 other events" in _detection(test_conn, dets[2])[2]


def test_a_user_event_is_not_judged_and_does_not_count_as_a_repeat(test_conn) -> None:
    nights = _nights(test_conn)
    obj, det = _binary(test_conn, nights, "user event", period=PERIOD, status=None)
    user = _event(test_conn, obj, nights[1], TC + 6 * PERIOD, DUR_H, 0.15, origin="user")
    test_conn.commit()
    update_auto_verdicts(test_conn, nights)
    assert _detection(test_conn, user) == (None, None, None)
    assert "repeats" not in _detection(test_conn, det)[2]


def test_the_rule_is_idempotent_and_its_reason_goes_when_it_stops_firing(test_conn) -> None:
    nights = _nights(test_conn)
    obj, det = _binary(test_conn, nights, "idempotent")

    first = update_auto_verdicts(test_conn, nights)
    test_conn.commit()
    snapshot = _detection(test_conn, det)
    second = update_auto_verdicts(test_conn, [nights[0], nights[0], *nights[1:]])
    test_conn.commit()
    assert second.changed_obj_ids == [] and _detection(test_conn, det) == snapshot
    assert (second.n_variability, second.n_rejected) == (first.n_variability, first.n_rejected)

    # a shape rule firing too: its reason comes first (rule order), the variability one stays
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.transit_shape SET depth = 5e-4 WHERE det_id = %s", (det,))
    test_conn.commit()
    update_auto_verdicts(test_conn, nights)
    both = _detection(test_conn, det)[2]
    assert both.startswith("no dip: ") and "; variability: the catalogue period" in both
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.transit_shape SET depth = 0.15 WHERE det_id = %s", (det,))

    # the period no longer explains the dip: only the reason of this rule goes, and the verdict
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.catalog_match SET period = 0.7777 WHERE obj_id = %s", (obj,))
    test_conn.commit()
    third = update_auto_verdicts(test_conn, nights)
    test_conn.commit()
    assert _detection(test_conn, det) == (None, None, None)
    assert third.n_variability == 0 and third.changed_obj_ids == [obj]


def test_a_persons_confirmed_verdict_wins_over_the_variability_annotation(test_conn) -> None:
    nights = _nights(test_conn)
    _obj, det = _binary(test_conn, nights, "confirmed", status="CONFIRMED")
    update_auto_verdicts(test_conn, nights)
    status, auto, reason = _detection(test_conn, det)
    assert (status, auto) == ("CONFIRMED", "REJECTED") and reason.startswith("variability: ")
    assert _effective_status(status, auto) == "CONFIRMED"


def test_an_event_night_without_a_light_curve_or_enough_epochs_is_not_judged(test_conn) -> None:
    nights = _nights(test_conn)
    obj, det = _binary(test_conn, nights, "no lc")
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.lightcurve WHERE obj_id = %s AND night_id = %s",
                    (obj, nights[0]))
    test_conn.commit()
    assert update_auto_verdicts(test_conn, nights).n_variability == 0
    assert _detection(test_conn, det) == (None, None, None)
    # a night of 11 epochs
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.star_night WHERE obj_id = %s AND night_id = %s",
                    (obj, nights[0]))
    _put_lc(test_conn, obj, nights[0], 9, tuple(a[:11] for a in _event_night(_eclipses(), TC)))
    test_conn.commit()
    assert update_auto_verdicts(test_conn, nights).n_variability == 0


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def test_analyze_applies_the_rule_and_reports_it(test_conn) -> None:
    night = _insert_night(test_conn, "20250101")
    others = [_insert_night(test_conn, f"2025010{k + 2}") for k in range(4)]
    obj_id = _insert_object(test_conn, "analyzed binary")
    flux_of = _eclipses()
    _insert_star_night_and_lc(test_conn, obj_id, night, 0, *_event_night(flux_of, TC))
    for k, (other, start) in enumerate(zip(others, _STARTS, strict=False), start=1):
        _insert_star_night_and_lc(test_conn, obj_id, other, k,
                                  *_lc_night(k, flux_of, seed=10 + k, start=start))
    _catalog(test_conn, obj_id, "VSX", "analyzed binary cat", "EA", PERIOD)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, flags) VALUES (%s, %s, 'transit', 12.0, 0.15, %s, %s, 1, 'OK') "
            "RETURNING det_id", (obj_id, night, TC, DUR_H),
        )
        (det_id,) = cur.fetchone()
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=Settings(), workers=1)
    test_conn.commit()

    status, auto, reason = _detection(test_conn, det_id)
    assert (status, auto) == ("UNCONFIRMED", "REJECTED")
    assert "variability: the catalogue period P=0.6000 d (VSX) predicts this dip" in reason
    assert report.n_variability == 1 and report.n_auto_rejected == 1

    again = analyze(test_conn, obj_ids=[obj_id], settings=Settings(), workers=1)
    assert (again.n_variability, again.n_auto_rejected) == (1, 1)
    assert _detection(test_conn, det_id) == (status, auto, reason)
