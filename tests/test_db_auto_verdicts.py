"""Tests (live test database, see tests/test_db_schema.py) for the automatic verdict pass of
relphot.db.coincidence (coincidence + EDGE_OUTLIER / NO_DIP / NO_BASELINE unioned), the edge clip
in ``relphot db analyze``, ``analyze(keep_vetted=True)``, ``relphot db reload-search`` and the
vetted-event definition."""

from __future__ import annotations

import numpy as np
import psycopg
import pytest
from test_db_analyze import _insert_night, _insert_object, _insert_star_night_and_lc
from test_db_coincidence import (
    T0,
    _auto_rejected,
    _background,
    _cluster,
    _detection,
    _event,
    _night,
)
from test_db_coincidence import _object as _co_object
from test_db_load_night import _SETTINGS, _edit_metrics, _write_night1

from relphot.config import Settings
from relphot.db.analyze import analyze
from relphot.db.coincidence import update_auto_verdicts, update_coincidence
from relphot.db.connect import resolve_dsn
from relphot.db.load_night import load_night
from relphot.db.reload_search import reload_search_detections
from relphot.db.schema import init_schema
from relphot.db.vetted import vetted_det_ids
from relphot.exceptions import ConfigError, NightLoadError
from relphot.web.app import _effective_status


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


def _set_shape(conn, det_id: int, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE relphot.transit_shape SET {sets} WHERE det_id = %s", (*cols.values(), det_id)
        )


def _reason(conn, det_id: int) -> str | None:
    return _detection(conn, det_id)[2]


# --------------------------------------------------------------------------
# the verdict pass
# --------------------------------------------------------------------------


def test_reasons_from_different_rules_are_unioned_and_counted(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night)
    objs = [_co_object(test_conn, f"deep{k}") for k in range(8)]
    cluster = [_event(test_conn, o, night, T0 + 0.2, 2.0, 0.3) for o in objs]
    solo = [_event(test_conn, _co_object(test_conn, f"solo{k}"), night, T0 + 0.05 + 0.01 * k)
            for k in range(4)]
    clean = solo[3]
    # cluster[0]: coincidence + edge outlier + no baseline; solo[0]: edge only; solo[1]: no dip;
    # solo[2]: no baseline; solo[3] stays clean
    _set_shape(test_conn, cluster[0], edge_clip_bjd=[T0], edge_adjacent=True, n_outside=2)
    _set_shape(test_conn, solo[0], edge_clip_bjd=[T0 + 0.4], edge_adjacent=True, n_outside=40)
    _set_shape(test_conn, solo[1], depth=5e-4, n_outside=40)
    _set_shape(test_conn, solo[2], depth=0.35, n_outside=3)
    test_conn.commit()

    report = update_auto_verdicts(test_conn, [night])
    test_conn.commit()

    assert report.n_coincidence == len(cluster)
    assert (report.n_edge_outlier, report.n_no_dip, report.n_no_baseline) == (2, 1, 2)
    assert report.n_rejected == len(cluster) + 3
    assert _auto_rejected(test_conn) == set(cluster) | set(solo[:3])
    both = _reason(test_conn, cluster[0])
    assert both.startswith("too many similar events: 7 other events")
    assert "; edge outlier: 1 isolated edge epoch was excluded from the fit and the only " in both
    assert both.endswith("; no baseline: only 2 epochs outside the fitted trapezoid (< 5) for a "
                         "depth of 0.30 (> 0.2)")
    assert _reason(test_conn, cluster[1]).startswith("too many similar events")
    assert "edge outlier" not in _reason(test_conn, cluster[1])
    assert _reason(test_conn, solo[0]).startswith("edge outlier: 1 isolated edge epoch was")
    assert _reason(test_conn, solo[1]) == "no dip: the fitted trapezoid depth is 5.0e-04 (< 1e-03)"
    assert _reason(test_conn, solo[2]).startswith("no baseline: only 3 epochs outside")
    assert _detection(test_conn, clean) == (None, None, None)
    assert report.changed_obj_ids == sorted(set(objs) | {
        _owner(test_conn, d) for d in solo[:3]
    })


def _owner(conn, det_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT obj_id FROM relphot.detection WHERE det_id = %s", (det_id,))
        return cur.fetchone()[0]


def test_the_pass_is_idempotent_and_clears_a_reason_only_when_its_rule_stops_firing(
    test_conn,
) -> None:
    night = _night(test_conn)
    _background(test_conn, night)
    _objs, cluster = _cluster(test_conn, night)
    target = cluster[0]
    _set_shape(test_conn, target, edge_clip_bjd=[T0], edge_adjacent=True)
    test_conn.commit()

    first = update_auto_verdicts(test_conn, [night])
    test_conn.commit()
    snapshot = {d: _detection(test_conn, d) for d in cluster}
    second = update_auto_verdicts(test_conn, [night, night])
    test_conn.commit()
    assert second.changed_obj_ids == []
    assert (second.n_rejected, second.n_edge_outlier) == (first.n_rejected, first.n_edge_outlier)
    assert {d: _detection(test_conn, d) for d in cluster} == snapshot
    assert "edge outlier" in _reason(test_conn, target)

    # the edge rule stops firing: only its reason goes, the coincidence verdict stays
    _set_shape(test_conn, target, edge_adjacent=False)
    test_conn.commit()
    third = update_auto_verdicts(test_conn, [night])
    test_conn.commit()
    reason = _reason(test_conn, target)
    assert reason.startswith("too many similar events") and "edge outlier" not in reason
    assert third.changed_obj_ids == []  # still auto-rejected: the verdict did not change

    # the coincidence stops (the others move away), the edge rule fires again: the other way round
    _set_shape(test_conn, target, edge_adjacent=True)
    with test_conn.cursor() as cur:
        for k, det_id in enumerate(cluster[1:]):
            cur.execute("UPDATE relphot.transit_shape SET tc = %s WHERE det_id = %s",
                        (T0 + 0.03 + 0.045 * k, det_id))
    test_conn.commit()
    fourth = update_auto_verdicts(test_conn, [night])
    test_conn.commit()
    assert _reason(test_conn, target).startswith("edge outlier")
    assert _auto_rejected(test_conn) == {target}
    assert len(fourth.changed_obj_ids) == len(cluster) - 1

    # nothing fires any more: the verdict is cleared
    _set_shape(test_conn, target, edge_adjacent=False)
    test_conn.commit()
    update_auto_verdicts(test_conn, [night])
    test_conn.commit()
    assert _detection(test_conn, target) == (None, None, None)
    assert update_coincidence is update_auto_verdicts


def test_a_persons_verdict_always_wins_and_user_events_are_not_judged(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night, n=20)
    rejected = _event(test_conn, _co_object(test_conn, "r"), night, T0 + 0.05, status="REJECTED")
    confirmed = _event(test_conn, _co_object(test_conn, "c"), night, T0 + 0.07, status="CONFIRMED")
    plain = _event(test_conn, _co_object(test_conn, "p"), night, T0 + 0.09)
    user = _event(test_conn, _co_object(test_conn, "u"), night, T0 + 0.11, origin="user")
    for d in (rejected, confirmed, plain, user):
        _set_shape(test_conn, d, depth=1e-4)
    test_conn.commit()

    update_auto_verdicts(test_conn, [night])
    test_conn.commit()
    assert _detection(test_conn, rejected)[:2] == ("REJECTED", "REJECTED")
    assert _detection(test_conn, confirmed)[:2] == ("CONFIRMED", "REJECTED")  # annotated only
    assert _detection(test_conn, user) == (None, None, None)
    assert _effective_status("CONFIRMED", "REJECTED") == "CONFIRMED"
    assert _effective_status("REJECTED", "REJECTED") == "REJECTED"
    assert _effective_status(None, "REJECTED") == "REJECTED (auto)"
    assert _detection(test_conn, plain)[:2] == (None, "REJECTED")


def test_skip_det_ids_leave_a_vetted_event_exactly_as_stored(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night, n=20)
    vetted = _event(test_conn, _co_object(test_conn, "v"), night, T0 + 0.05, status="REJECTED")
    other = _event(test_conn, _co_object(test_conn, "o"), night, T0 + 0.07)
    for d in (vetted, other):
        _set_shape(test_conn, d, depth=1e-4)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.transit_coincidence (det_id, night_id, n_similar, rejected) "
            "VALUES (%s, %s, 99, false)", (vetted, night)
        )
    test_conn.commit()

    report = update_auto_verdicts(test_conn, [night], skip_det_ids={vetted})
    test_conn.commit()
    assert _detection(test_conn, vetted) == ("REJECTED", None, None)
    assert _detection(test_conn, other)[1] == "REJECTED"
    assert report.n_no_dip == 1 and report.n_rejected == 1
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT n_similar FROM relphot.transit_coincidence WHERE det_id = %s", (vetted,)
        )
        assert cur.fetchone() == (99,)


# --------------------------------------------------------------------------
# analyze: the edge clip and the stored columns; keep_vetted
# --------------------------------------------------------------------------


def _artefact_object(conn, name: str, night_id: int, star_id: int, seed: int) -> tuple[int, int]:
    """A star with a flat 42-epoch night and a +22 % last epoch, whose search event is a box over
    all but the last epoch: the labelled 'bad phot edge' pattern. Returns (obj_id, det_id)."""
    obj_id = _insert_object(conn, name)
    rng = np.random.default_rng(seed)
    t = 2460000.0 + np.arange(42) * 2.2 / 1440.0
    flux = 1.0 + rng.normal(0.0, 0.004, 42)
    flux[-1] = 1.22
    _insert_star_night_and_lc(conn, obj_id, night_id, star_id, t, flux, np.full(42, 0.004))
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, flags) VALUES (%s, %s, 'transit', 12.0, 0.3, %s, %s, 1, 'OK') "
            "RETURNING det_id",
            (obj_id, night_id, 0.5 * (t[0] + t[-2]), (t[-2] - t[0]) * 24.0),
        )
        (det_id,) = cur.fetchone()
    return obj_id, det_id


def _shape_row(conn, det_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT depth, converged, edge_clip_bjd, edge_adjacent, n_outside "
            "FROM relphot.transit_shape WHERE det_id = %s", (det_id,)
        )
        return cur.fetchone()


def test_analyze_excludes_the_edge_epoch_stores_it_and_auto_rejects_idempotently(test_conn) -> None:
    night = _insert_night(test_conn, "20250101")
    obj_id, det_id = _artefact_object(test_conn, "edge", night, 0, seed=0)
    test_conn.commit()

    report = analyze(test_conn, obj_ids=[obj_id], settings=Settings(), workers=1)
    test_conn.commit()

    depth, _converged, clip, adjacent, _n_out = _shape_row(test_conn, det_id)
    assert clip == [pytest.approx(2460000.0 + 41 * 2.2 / 1440.0)]
    assert adjacent is True
    assert depth < 0.1  # without the clip this was the open box (depth 0.36)
    status, auto, reason = _detection(test_conn, det_id)
    assert (status, auto) == ("UNCONFIRMED", "REJECTED")
    assert reason.startswith("edge outlier: 1 isolated edge epoch was excluded")
    assert (report.n_edge_outlier, report.n_auto_rejected, report.n_coincidence_rejected) == (
        1, 1, 0,
    )

    again = analyze(test_conn, obj_ids=[obj_id], settings=Settings(), workers=1)
    test_conn.commit()
    assert (again.n_edge_outlier, again.n_auto_rejected) == (1, 1)
    assert _detection(test_conn, det_id) == (status, auto, reason)


def test_analyze_keep_vetted_does_not_refit_or_rejudge_a_vetted_event(test_conn) -> None:
    night = _insert_night(test_conn, "20250101")
    vetted_obj, vetted = _artefact_object(test_conn, "vetted", night, 0, seed=1)
    open_obj, unvetted = _artefact_object(test_conn, "open", night, 1, seed=2)
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET status = 'REJECTED', notes = 'Bad phot edge' "
            "WHERE det_id = %s", (vetted,)
        )
        # the stored fit the person looked at (an old open-box fit)
        cur.execute(
            "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, depth, t14_h, input, converged,"
            " computed_at) VALUES (%s, %s, 2460000.03, 0.36, 1.5, 'night', true, now())",
            (vetted, vetted_obj),
        )
    test_conn.commit()

    analyze(test_conn, all_candidates=False, obj_ids=[vetted_obj, open_obj],
            settings=Settings(), workers=1, keep_vetted=True)
    test_conn.commit()

    assert _shape_row(test_conn, vetted) == (pytest.approx(0.36), True, None, None, None)
    assert _detection(test_conn, vetted) == ("REJECTED", None, None)
    assert _shape_row(test_conn, unvetted)[3] is True  # refit with the clip
    assert _detection(test_conn, unvetted)[1] == "REJECTED"

    # without the option the vetted event is refit and annotated, as before
    analyze(test_conn, obj_ids=[vetted_obj], settings=Settings(), workers=1)
    test_conn.commit()
    assert _shape_row(test_conn, vetted)[3] is True
    assert _detection(test_conn, vetted)[:2] == ("REJECTED", "REJECTED")


# --------------------------------------------------------------------------
# vetted events
# --------------------------------------------------------------------------


def test_vetted_det_ids_definition(test_conn) -> None:
    night = _night(test_conn, frames=False)
    other_night = _night(test_conn, "20250102", frames=False)
    objs = [_co_object(test_conn, f"o{k}") for k in range(9)]
    ev = {
        "plain": _event(test_conn, objs[0], night, T0),
        "unconfirmed": _event(test_conn, objs[1], night, T0, status="UNCONFIRMED"),
        "confirmed": _event(test_conn, objs[2], night, T0, status="CONFIRMED"),
        "rejected": _event(test_conn, objs[3], night, T0, status="REJECTED"),
        "notes": _event(test_conn, objs[4], night, T0),
        "user": _event(test_conn, objs[5], night, T0, origin="user"),
        "reviewed": _event(test_conn, objs[6], night, T0),
        "reviewed_other_night": _event(test_conn, objs[7], night, T0),
        "linked": _event(test_conn, objs[8], night, T0),
    }
    competing = _event(test_conn, objs[8], night, T0 + 0.001)
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET notes = 'x' WHERE det_id = %s", (ev["notes"],))
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, note) VALUES (%s, %s, 'n')",
            (objs[6], night),
        )
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, note) VALUES (%s, %s, 'n')",
            (objs[7], other_night),
        )
        cur.execute(
            "UPDATE relphot.detection SET superseded_by = %s WHERE det_id = %s",
            (competing, ev["linked"]),
        )
    test_conn.commit()
    with test_conn.cursor() as cur:
        got = vetted_det_ids(cur)
        assert got == {ev[k] for k in ("confirmed", "rejected", "notes", "user", "reviewed")} | {
            ev["linked"], competing,
        }
        assert vetted_det_ids(cur, night_ids=[other_night]) == set()
        assert vetted_det_ids(cur, obj_ids=[objs[2], objs[0]]) == {ev["confirmed"]}


# --------------------------------------------------------------------------
# reload-search
# --------------------------------------------------------------------------


def _counts(conn) -> tuple:
    with conn.cursor() as cur:
        out = []
        for table in ("frame", "star_night", "lightcurve", "detection_review_orphan"):
            cur.execute(f"SELECT count(*) FROM relphot.{table}")
            out.append(cur.fetchone()[0])
        return tuple(out)


def _transits(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT det_id, obj_id, tc_bjd_tdb, status, flags, extra FROM relphot.detection "
            "WHERE kind = 'transit' AND origin = 'search' ORDER BY det_id"
        )
        return cur.fetchall()


def test_reload_search_replaces_an_unvetted_event_and_keeps_everything_else(
    test_conn, tmp_path,
) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    report = load_night(test_conn, root, settings=_SETTINGS)
    (old,) = _transits(test_conn)
    counts = _counts(test_conn)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, depth, tc_bjd_tdb, duration_h,"
            " flags, origin) VALUES (%s, %s, 'transit', 0.02, 2460000.55, 1.5, 'USER', 'user')",
            (old[1], report.night_id),
        )
    test_conn.commit()

    def edit(sm) -> None:
        sm["transit_tc_bjd_tdb"] = sm["transit_tc_bjd_tdb"] + 0.2  # a different event now
        sm["transit_flags_str"] = np.array(["", "", "", "", "EDGE_OUTLIER", ""], dtype=object)
        sm["transit_edge_clip_bjd"] = np.array(["", "", "", "", "2460000.5006,2460000.51", ""],
                                               dtype=object)

    _edit_metrics(root, edit)
    rep = reload_search_detections(test_conn, root, settings=_SETTINGS)

    assert (rep.night_id, rep.n_deleted, rep.n_inserted, rep.n_vetted_kept) == (
        report.night_id, 1, 1, 0,
    )
    assert _counts(test_conn) == counts  # frames, stars, light curves untouched; no orphans
    (new,) = _transits(test_conn)
    assert new[0] != old[0] and new[1] == old[1]
    assert new[2] == pytest.approx(2460000.75) and new[4] == "EDGE_OUTLIER"
    assert new[5]["transit_edge_clip_bjd"] == [2460000.5006, 2460000.51]
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.detection WHERE origin = 'user'")
        assert cur.fetchone() == (1,)


def test_reload_search_keeps_a_vetted_event_with_its_id_and_never_orphans_it(
    test_conn, tmp_path,
) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    load_night(test_conn, root, settings=_SETTINGS)
    (old,) = _transits(test_conn)
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED', notes = 'Bad fit' "
                    "WHERE det_id = %s", (old[0],))
    test_conn.commit()

    # the new search no longer finds the event at all
    _edit_metrics(root, lambda sm: sm.__setitem__(
        "transit_candidate", np.zeros(len(sm), dtype=bool)))
    rep = reload_search_detections(test_conn, root, settings=_SETTINGS)
    assert (rep.n_deleted, rep.n_inserted, rep.n_vetted_kept) == (0, 0, 1)
    (kept,) = _transits(test_conn)
    assert kept[0] == old[0] and kept[3] == "REJECTED"
    assert _counts(test_conn)[3] == 0

    # it finds it again, slightly moved: still the same single event, no duplicate
    _edit_metrics(root, lambda sm: sm.__setitem__("transit_candidate", sm["transit_snr"] > 5))
    _edit_metrics(root, lambda sm: sm.__setitem__(
        "transit_tc_bjd_tdb", sm["transit_tc_bjd_tdb"] + 0.02))
    rep = reload_search_detections(test_conn, root, settings=_SETTINGS)
    assert (rep.n_deleted, rep.n_inserted, rep.n_skipped_vetted) == (0, 0, 1)
    assert [r[0] for r in _transits(test_conn)] == [old[0]]

    # an object with a night review is left alone entirely, even for a different event
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = NULL, notes = NULL")
        cur.execute("INSERT INTO relphot.user_night_review (obj_id, night_id, note) "
                    "VALUES (%s, %s, 'looked at')", (old[1], rep.night_id))
    test_conn.commit()
    _edit_metrics(root, lambda sm: sm.__setitem__(
        "transit_tc_bjd_tdb", sm["transit_tc_bjd_tdb"] + 0.2))
    rep = reload_search_detections(test_conn, root, settings=_SETTINGS)
    assert (rep.n_deleted, rep.n_inserted, rep.n_vetted_kept, rep.n_skipped_vetted) == (0, 0, 1, 1)
    assert [r[0] for r in _transits(test_conn)] == [old[0]]


def test_reload_search_needs_a_loaded_night_and_search_metrics(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    with pytest.raises(NightLoadError, match="not loaded"):
        reload_search_detections(test_conn, root, settings=_SETTINGS)
    (root / "lc" / "night_lc_search_metrics.parquet").unlink()
    with pytest.raises(NightLoadError, match="search_metrics"):
        reload_search_detections(test_conn, root, settings=_SETTINGS)


def test_load_night_keeps_the_edge_clip_epochs_in_the_detection_extra(test_conn, tmp_path) -> None:
    root = tmp_path / "T80S_reduced" / "20250101" / "relphot"
    _write_night1(root)
    _edit_metrics(root, lambda sm: sm.__setitem__(
        "transit_edge_clip_bjd", np.array(["", "", "", "", "2460000.500600", ""], dtype=object)))
    load_night(test_conn, root, settings=_SETTINGS)
    (row,) = _transits(test_conn)
    assert row[5]["transit_edge_clip_bjd"] == [2460000.5006]
    # an older product without the column loads without the key
    _write_night1(tmp_path / "T80S_reduced" / "20250102" / "relphot")
    load_night(test_conn, tmp_path / "T80S_reduced" / "20250102" / "relphot", settings=_SETTINGS)
    assert all("transit_edge_clip_bjd" not in r[5] for r in _transits(test_conn)[1:])
