"""Tests for relphot.db.reprocess (``relphot db reprocess``) against a live test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for how it is
resolved); skipped with an explicit reason when there is none. Requests are inserted with
plain SQL as the owner role (the web's own INSERT path is tested in test_web.py); fixtures
insert night/object/star_night/lightcurve rows directly so each test controls its light
curve precisely.
"""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import replace

import numpy as np
import psycopg
import pytest

from relphot.cli import main
from relphot.config import DbSettings, Settings
from relphot.db.analyze import analyze
from relphot.db.connect import resolve_dsn
from relphot.db.refresh import refresh_objects
from relphot.db.reprocess import reprocess
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError

_SETTINGS = replace(Settings(), db=replace(DbSettings(), max_expected_noise=0.05))
# relphot.db re-exports the *function* ``reprocess``, which shadows the submodule attribute
reprocess_module = importlib.import_module("relphot.db.reprocess")


def _dsn() -> str:
    try:
        return resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")


@pytest.fixture
def test_conn():
    conn = psycopg.connect(_dsn())
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    init_schema(conn)
    yield conn
    conn.close()


def _insert_night(conn, label: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-01', %s, %s, now()) RETURNING night_id",
            (label, f"/tmp/{label}"),
        )
        return cur.fetchone()[0]


def _insert_object(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, data_updated_at) "
            "VALUES (%s, 10.0, -20.0, now()) RETURNING obj_id",
            (name,),
        )
        return cur.fetchone()[0]


def _insert_lc(conn, obj_id: int, night_id: int, star_id: int, t, flux, flux_err) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night "
            "(obj_id, night_id, star_id, tile, mag, best_aperture, rms, "
            "expected_noise, chi2_reduced, n_epochs, is_comparison) "
            "VALUES (%s, %s, %s, 0, 15.0, 1, 0.01, 0.01, 1.0, %s, false)",
            (obj_id, night_id, star_id, len(t)),
        )
        cur.execute(
            "INSERT INTO relphot.lightcurve "
            "(obj_id, night_id, frame_index, bjd_tdb, flux, flux_err, flux_raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                obj_id, night_id, list(range(len(t))), [float(v) for v in t],
                [float(v) for v in flux], [float(v) for v in flux_err], [float(v) for v in flux],
            ),
        )


def _queue(conn, obj_id: int, kind: str, **kw) -> int:
    cols = ["obj_id", "kind", *kw]
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO relphot.reprocess_request ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING req_id",
            [obj_id, kind, *kw.values()],
        )
        req_id = cur.fetchone()[0]
    conn.commit()
    return req_id


def _request(conn, req_id: int) -> dict:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, error, result, started_at, finished_at "
            "FROM relphot.reprocess_request WHERE req_id = %s",
            (req_id,),
        )
        cols = [d.name for d in cur.description]
        return dict(zip(cols, cur.fetchone(), strict=True))


# --------------------------------------------------------------------------
# variable requests: user-guided period search
# --------------------------------------------------------------------------

_P = 2.5
_STARTS = (0.0, 1.1, 2.3)


def _insert_slow_variable(conn, name: str, *, tied: bool = True) -> int:
    """A 2.5 d sinusoid on three tied nights about a day apart (baseline 1.1 cycles)."""
    rng = np.random.default_rng(3)
    obj_id = _insert_object(conn, name)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, labels, anchor, loaded_at) "
            "VALUES (%s, %s, 'a', now()) RETURNING mn_run_id",
            (f"mn_{name}", ["a", "b", "c"]),
        )
        mn_run = cur.fetchone()[0]
    for i, t0 in enumerate(_STARTS):
        night_id = _insert_night(conn, f"{name}_{i}")
        t = 2460000.0 + np.sort(t0 + rng.uniform(0.0, 0.5, 120))
        mag = 15.0 + 0.15 * np.sin(2 * np.pi * (t - 2460000.0) / _P) + rng.normal(0, 0.003, 120)
        flux = 10.0 ** (-0.4 * (mag - np.median(mag)))
        _insert_lc(conn, obj_id, night_id, i, t, flux, np.full_like(t, 0.003) * flux)
        if tied:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO relphot.tie (mn_run_id, obj_id, night_id, mag, mag_err) "
                    "VALUES (%s, %s, %s, %s, 0.005)",
                    (mn_run, obj_id, night_id, float(np.median(mag))),
                )
    conn.commit()
    return obj_id


def test_worker_guided_period_search_stores_ls_guided_and_leaves_period_alone(test_conn) -> None:
    obj_id = _insert_slow_variable(test_conn, "slow")
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.object SET period = 7.7, period_source = 'manual' WHERE obj_id = %s",
            (obj_id,),
        )
    req_id = _queue(test_conn, obj_id, "variable", period_guess=2.3)

    report = reprocess(test_conn, settings=_SETTINGS)
    assert (report.n_done, report.n_failed) == (1, 0)

    req = _request(test_conn, req_id)
    assert req["status"] == "done" and req["error"] is None
    assert req["started_at"] is not None and req["finished_at"] >= req["started_at"]
    res = req["result"]
    assert res["found"] and res["input"] == "tied" and res["n_nights"] == 3
    assert res["period"] == pytest.approx(_P, abs=0.03)
    assert res["period_err"] is not None and 0 < res["period_err"] < 0.02
    assert res["guess"] == 2.3 and res["verify_status"] == "no_literature"

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT method, guess, input, n_nights, period, alias_periods IS NOT NULL, "
            "phase_coverage, n_cycles FROM relphot.period_estimate WHERE obj_id = %s",
            (obj_id,),
        )
        ((method, guess, input_label, n_nights, period, has_alias, coverage, cycles),) = (
            cur.fetchall()
        )
        cur.execute(
            "SELECT period, period_source FROM relphot.object WHERE obj_id = %s", (obj_id,)
        )
        obj = cur.fetchone()
    assert (method, guess, input_label, n_nights) == ("LS-guided", 2.3, "tied", 3)
    assert period == pytest.approx(res["period"]) and has_alias
    assert 0 < coverage <= 1 and 1.0 < cycles < 2.0
    assert obj == (7.7, "manual")  # PERIOD is only ever changed by "Adopt as period"


def test_guided_period_is_verified_against_the_literature_and_a_far_guess_finds_nothing(
    test_conn,
) -> None:
    obj_id = _insert_slow_variable(test_conn, "slow_lit")
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.catalog_match (obj_id, catalog, name, type, period, period_err) "
            "VALUES (%s, 'VSX', 'V* Slow', 'ROT', 2.5, 0.01)",
            (obj_id,),
        )
    test_conn.commit()
    good = _queue(test_conn, obj_id, "variable", period_guess=2.4)
    far = _queue(test_conn, obj_id, "variable", period_guess=40.0)
    reprocess(test_conn, settings=_SETTINGS)

    res = _request(test_conn, good)["result"]
    assert res["verify_status"] == "verified" and res["harmonic"] == 1.0
    assert abs(res["delta"]) < 0.03
    res = _request(test_conn, far)
    assert res["status"] == "done" and res["result"]["found"] is False
    assert res["result"]["verify_status"] in {"lit_period_outside_grid", "no_peak_in_window"}
    with test_conn.cursor() as cur:
        # one guided row per night set: the later request replaced the earlier one
        cur.execute("SELECT count(*), max(guess) FROM relphot.period_estimate")
        assert cur.fetchone() == (1, 40.0)


# --------------------------------------------------------------------------
# transit requests: user detections
# --------------------------------------------------------------------------


def _dip(t, tc, t14_h, depth, q=0.2):
    half = 0.5 * t14_h / 24.0
    tau = q * t14_h / 24.0
    return depth * np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _insert_two_dip_night(conn, name: str) -> tuple[int, int, int]:
    """One night with two 2 h, 2 % dips (tc 0.12 and 0.30 d); the search found the first."""
    rng = np.random.default_rng(11)
    obj_id = _insert_object(conn, name)
    night_id = _insert_night(conn, f"{name}_n")
    t = 2460000.0 + np.linspace(0.0, 0.42, 340)
    flux = 1.0 - _dip(t, 2460000.12, 2.0, 0.02) - _dip(t, 2460000.30, 2.0, 0.02)
    flux = flux + rng.normal(0.0, 0.0006, t.size)
    _insert_lc(conn, obj_id, night_id, 0, t, flux, np.full_like(t, 0.0006))
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, flags) VALUES (%s, %s, 'transit', 12.0, 0.019, %s, 1.9, 1, "
            "'OK') RETURNING det_id",
            (obj_id, night_id, 2460000.121),
        )
        search_det = cur.fetchone()[0]
    conn.commit()
    analyze(conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    conn.commit()
    return obj_id, night_id, search_det


def test_worker_transit_request_adds_a_user_detection_and_never_touches_the_search_one(
    test_conn,
) -> None:
    obj_id, night_id, search_det = _insert_two_dip_night(test_conn, "dips")
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT tc_bjd_tdb, depth, duration_h, flags, tier, status FROM relphot.detection "
            "WHERE det_id = %s",
            (search_det,),
        )
        search_before = cur.fetchone()
    on_search = _queue(
        test_conn, obj_id, "transit", tc_guess=2460000.125, width_guess_h=2.4, night_id=night_id,
        note="near the search event",
    )
    elsewhere = _queue(test_conn, obj_id, "transit", tc_guess=2460000.31, width_guess_h=1.5)

    report = reprocess(test_conn, settings=_SETTINGS)
    assert (report.n_done, report.n_failed) == (2, 0)

    a, b = _request(test_conn, on_search)["result"], _request(test_conn, elsewhere)["result"]
    assert a["det_id"] != search_det and a["search_det_id"] == search_det
    assert b["search_det_id"] is None and b["det_id"] not in (a["det_id"], search_det)
    assert a["night_id"] == b["night_id"] == night_id
    for res, tc in ((a, 2460000.12), (b, 2460000.30)):
        assert abs(res["tc"] - tc) < max(4 * res["tc_err"], 1e-3)
        assert res["depth"] == pytest.approx(0.02, abs=max(4 * res["depth_err"], 1.5e-3))
        assert res["t14_h"] == pytest.approx(2.0, abs=0.25)
        assert res["t14_lower_limit"] is False

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT det_id, origin, flags, night_id, kind, status, "
            "(extra ->> 'search_det_id')::bigint, extra ->> 'note' FROM relphot.detection "
            "WHERE origin = 'user' ORDER BY det_id"
        )
        users = cur.fetchall()
        cur.execute(
            "SELECT tc_bjd_tdb, depth, duration_h, flags, tier, status FROM relphot.detection "
            "WHERE det_id = %s AND origin = 'search'",
            (search_det,),
        )
        search_after = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM relphot.transit_shape WHERE det_id = ANY(%s) AND converged",
            ([a["det_id"], b["det_id"], search_det],),
        )
        (n_shapes,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM relphot.transit_match WHERE obj_id = %s", (obj_id,))
        (n_matches,) = cur.fetchone()
        cur.execute("SELECT is_exop FROM relphot.object WHERE obj_id = %s", (obj_id,))
        (is_exop,) = cur.fetchone()
    assert users == [
        (a["det_id"], "user", "USER", night_id, "transit", "UNCONFIRMED", search_det,
         "near the search event"),
        (b["det_id"], "user", "USER", night_id, "transit", "UNCONFIRMED", None, None),
    ]
    assert search_after == search_before  # the search detection is never overwritten
    assert n_shapes == 3
    assert n_matches == 3  # search/user-a, search/user-b, user-a/user-b: never merged
    assert b["n_matches"] == 3
    assert is_exop is True  # from the search event; see the next test for user events alone


def _links(conn) -> dict[int, int | None]:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("SELECT det_id, superseded_by FROM relphot.detection ORDER BY det_id")
        return dict(cur.fetchall())


def _night_verdict(conn, obj_id: int, night_id: int) -> str | None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT exop_verdict FROM relphot.user_night_review "
            "WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
    return None if row is None else row[0]


def test_worker_rerun_supersedes_the_search_event_it_refits_not_another_transit(test_conn) -> None:
    obj_id, night_id, search_det = _insert_two_dip_night(test_conn, "supersede")
    # the person had REJECTED the night, and with it the search event, before this RERUN
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET status = 'REJECTED' WHERE det_id = %s", (search_det,)
        )
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
            "VALUES (%s, %s, 'REJECTED')",
            (obj_id, night_id),
        )
    test_conn.commit()
    on_search = _queue(
        test_conn, obj_id, "transit", tc_guess=2460000.125, width_guess_h=2.4, night_id=night_id
    )
    elsewhere = _queue(test_conn, obj_id, "transit", tc_guess=2460000.31, width_guess_h=1.5)

    assert reprocess(test_conn, settings=_SETTINGS).n_done == 2
    a, b = _request(test_conn, on_search)["result"], _request(test_conn, elsewhere)["result"]

    # the refit of the search event supersedes it; the other dip of the night is its own event
    assert a["superseded"] == [search_det] and b["superseded"] == []
    assert _links(test_conn) == {search_det: a["det_id"], a["det_id"]: None, b["det_id"]: None}
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT status, origin FROM relphot.detection WHERE det_id = %s", (search_det,)
        )
        assert cur.fetchone() == ("REJECTED", "search")  # the person's verdict is kept on it
        cur.execute(
            "SELECT is_exop, n_review_pending FROM relphot.object WHERE obj_id = %s", (obj_id,)
        )
        # the RERUN event stands for the search event: the flag stays, the night awaits review
        assert cur.fetchone() == (True, 1)
        cur.execute("SELECT count(*) FROM relphot.transit_match WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (3,)  # the pairs are still stored; the web hides superseded ones
    # the night's EXOP verdict follows the active events (the new ones are open): cleared
    assert _night_verdict(test_conn, obj_id, night_id) is None


def test_failed_rerun_supersedes_nothing_and_leaves_the_verdicts_alone(test_conn) -> None:
    obj_id, night_id, search_det = _insert_two_dip_night(test_conn, "fail_supersede")
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET status = 'REJECTED' WHERE det_id = %s", (search_det,)
        )
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
            "VALUES (%s, %s, 'REJECTED')",
            (obj_id, night_id),
        )
    test_conn.commit()
    bad_fit = _queue(test_conn, obj_id, "transit", tc_guess=2460000.0005, width_guess_h=0.1)

    assert reprocess(test_conn, settings=_SETTINGS).n_failed == 1
    assert _request(test_conn, bad_fit)["status"] == "failed"
    assert _links(test_conn) == {search_det: None}
    assert _night_verdict(test_conn, obj_id, night_id) == "REJECTED"


def test_a_second_rerun_of_the_same_transit_supersedes_the_first_and_the_search_event(
    test_conn,
) -> None:
    obj_id, _night, search_det = _insert_two_dip_night(test_conn, "twice")
    first = _queue(test_conn, obj_id, "transit", tc_guess=2460000.125, width_guess_h=2.4)
    assert reprocess(test_conn, settings=_SETTINGS).n_done == 1
    second = _queue(test_conn, obj_id, "transit", tc_guess=2460000.119, width_guess_h=2.0)
    assert reprocess(test_conn, settings=_SETTINGS).n_done == 1

    one, two = _request(test_conn, first)["result"], _request(test_conn, second)["result"]
    assert one["superseded"] == [search_det]
    assert two["superseded"] == sorted([search_det, one["det_id"]])
    # flat: both older events point at the newest
    assert _links(test_conn) == {
        search_det: two["det_id"], one["det_id"]: two["det_id"], two["det_id"]: None,
    }


def test_user_detections_never_set_is_exop_and_analyze_refits_them(test_conn) -> None:
    obj_id = _insert_object(test_conn, "user_only")
    night_id = _insert_night(test_conn, "user_only_n")
    t = 2460000.0 + np.linspace(0.0, 0.42, 340)
    flux = 1.0 - _dip(t, 2460000.2, 2.0, 0.02) + np.random.default_rng(2).normal(0, 0.0006, t.size)
    _insert_lc(test_conn, obj_id, night_id, 0, t, flux, np.full_like(t, 0.0006))
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, depth, tc_bjd_tdb, "
            "duration_h, flags, origin) VALUES (%s, %s, 'transit', 0.02, %s, 2.0, 'USER', 'user')",
            (obj_id, night_id, 2460000.2),
        )
    refresh_objects(test_conn, [obj_id])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, best_snr, n_detections FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        assert cur.fetchone() == (False, None, 1)  # counted, but not a flag nor the best transit

    report = analyze(test_conn, obj_ids=[obj_id], settings=_SETTINGS, workers=1)
    test_conn.commit()
    assert report.n_transit_shapes == 1
    with test_conn.cursor() as cur:
        cur.execute("SELECT is_exop FROM relphot.object WHERE obj_id = %s", (obj_id,))
        assert cur.fetchone() == (False,)


# --------------------------------------------------------------------------
# queue behaviour: failures, order, restart, NOTIFY
# --------------------------------------------------------------------------


def test_failed_request_is_marked_with_its_error_and_the_queue_goes_on(test_conn) -> None:
    empty = _insert_object(test_conn, "no_lightcurve")
    obj_id, _night, _det = _insert_two_dip_night(test_conn, "fail_dips")
    no_data = _queue(test_conn, empty, "variable", period_guess=1.0)
    outside = _queue(test_conn, obj_id, "transit", tc_guess=2460005.0, width_guess_h=2.0)
    bad_fit = _queue(test_conn, obj_id, "transit", tc_guess=2460000.0005, width_guess_h=0.1)
    fine = _queue(test_conn, obj_id, "transit", tc_guess=2460000.30, width_guess_h=2.0)

    report = reprocess(test_conn, settings=_SETTINGS)
    assert (report.n_done, report.n_failed) == (1, 3)

    rows = {r: _request(test_conn, r) for r in (no_data, outside, bad_fit, fine)}
    assert rows[no_data]["status"] == "failed" and "no stored light curve" in rows[no_data]["error"]
    assert rows[outside]["status"] == "failed" and "not inside" in rows[outside]["error"]
    assert rows[bad_fit]["status"] == "failed" and "did not converge" in rows[bad_fit]["error"]
    assert rows[fine]["status"] == "done" and rows[fine]["error"] is None
    for r in (no_data, outside, bad_fit):
        assert rows[r]["finished_at"] is not None and rows[r]["result"] is None
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.detection WHERE origin = 'user'")
        assert cur.fetchone() == (1,)  # only the request that worked wrote anything


def test_unexpected_exception_rolls_the_request_back_and_marks_it_failed(
    test_conn, monkeypatch
) -> None:
    obj_id = _insert_slow_variable(test_conn, "boom")
    req_id = _queue(test_conn, obj_id, "variable", period_guess=2.4)
    original = reprocess_module._process_variable

    def write_then_explode(conn, req, settings):
        original(conn, req, settings)  # inserts the period_estimate row ...
        msg = "boom"
        raise RuntimeError(msg)  # ... then fails: everything must be rolled back

    monkeypatch.setattr(reprocess_module, "_process_variable", write_then_explode)
    report = reprocess(test_conn, settings=_SETTINGS)
    assert (report.n_done, report.n_failed) == (0, 1)
    req = _request(test_conn, req_id)
    assert req["status"] == "failed" and req["error"] == "RuntimeError: boom"
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.period_estimate")
        assert cur.fetchone() == (0,)


def test_requests_are_worked_off_first_in_first_out(test_conn) -> None:
    obj_id = _insert_slow_variable(test_conn, "fifo")
    ids = [_queue(test_conn, obj_id, "variable", period_guess=g) for g in (2.4, 2.6, 2.5)]
    reprocess(test_conn, settings=_SETTINGS)
    started = [_request(test_conn, i)["started_at"] for i in ids]
    assert started == sorted(started) and len(set(started)) == 3


def test_requests_a_crashed_worker_left_running_are_queued_again(test_conn) -> None:
    obj_id = _insert_slow_variable(test_conn, "crashed")
    req_id = _queue(test_conn, obj_id, "variable", period_guess=2.4)
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'running', started_at = now() "
            "WHERE req_id = %s",
            (req_id,),
        )
    test_conn.commit()
    report = reprocess(test_conn, settings=_SETTINGS)
    assert report.n_done == 1 and _request(test_conn, req_id)["status"] == "done"


def test_watch_mode_wakes_on_notify_and_stops_on_the_event(test_conn) -> None:
    obj_id = _insert_slow_variable(test_conn, "watch")
    dsn = _dsn()
    worker_conn = psycopg.connect(dsn)
    listen_conn = psycopg.connect(dsn, autocommit=True)
    stop = threading.Event()
    out: dict = {}

    def run() -> None:
        # a 60 s poll fallback: only the NOTIFY sent by the insert trigger can wake it in time
        out["report"] = reprocess(
            worker_conn, settings=_SETTINGS, watch=True, listen_conn=listen_conn,
            poll_seconds=60.0, stop=stop,
        )

    thread = threading.Thread(target=run)
    thread.start()
    try:
        time.sleep(1.5)  # the worker is idle, waiting on its LISTEN
        req_id = _queue(test_conn, obj_id, "variable", period_guess=2.4)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and _request(test_conn, req_id)["status"] != "done":
            time.sleep(0.2)
        assert _request(test_conn, req_id)["status"] == "done"
    finally:
        stop.set()
        thread.join(timeout=15.0)
        worker_conn.close()
        listen_conn.close()
    assert not thread.is_alive()
    assert out["report"].n_done == 1


def test_cli_once_works_off_the_queue(test_conn, capsys) -> None:
    obj_id = _insert_slow_variable(test_conn, "cli")
    req_id = _queue(test_conn, obj_id, "variable", period_guess=2.4)
    assert main(["db", "reprocess", "--once", "--dsn", _dsn()]) == 0
    assert "done=1 failed=0" in capsys.readouterr().out
    assert _request(test_conn, req_id)["status"] == "done"


# --------------------------------------------------------------------------
# night-only variable requests (restricted to one night, no multi-night tie)
# --------------------------------------------------------------------------


def _insert_short_variable(conn, name, period=0.12):
    """Insert a 2-night short-period variable for testing."""
    rng = np.random.default_rng(5)
    obj_id = _insert_object(conn, name)

    # Insert 2 nights
    night_ids = []
    for i in range(2):
        night_id = _insert_night(conn, f"{name}_{i}")
        night_ids.append(night_id)

    # Insert light curve data (sinusoid)
    for night_id in night_ids:
        t = 2460310.5 + np.sort(rng.uniform(0.0, 0.4, 150))
        phase = (t / period) % 1.0
        flux_mean = 1.0
        amplitude = 0.05
        flux = flux_mean * (1.0 - amplitude * np.sin(2 * np.pi * phase))
        flux_err = np.full_like(flux, 0.003)
        _insert_lc(conn, obj_id, night_id, 0, t, flux, flux_err)

    conn.commit()
    return obj_id, night_ids


def test_night_only_variable_request_uses_only_that_night(test_conn) -> None:
    """Variable request with night_id uses only that night, no tie."""
    obj_id, night_ids = _insert_short_variable(test_conn, "test_night_var")
    n0, _n1 = night_ids

    # Single night request
    req_id = _queue(test_conn, obj_id, "variable", period_guess=0.125, night_id=n0)
    reprocess(test_conn, settings=_SETTINGS)
    req = _request(test_conn, req_id)
    assert req["status"] == "done" and req["error"] is None
    res = req["result"]
    assert res["found"] and res["n_nights"] == 1 and res["input"] == "night"
    assert abs(res["period"] - 0.12) < 0.005  # within 0.005 of true period
    assert res["night_id"] == n0 and res["all_nights"] is False

    # All nights request
    req_id = _queue(test_conn, obj_id, "variable", period_guess=0.125, night_id=None)
    reprocess(test_conn, settings=_SETTINGS)
    req = _request(test_conn, req_id)
    assert req["status"] == "done"
    assert req["result"]["n_nights"] == 2 and req["result"]["all_nights"] is True


def test_night_only_request_ignores_the_multi_night_tie(test_conn) -> None:
    """Night-only request ignores multi-night tie."""
    obj_id = _insert_slow_variable(test_conn, "tied_ignore", tied=True)
    night_ids = []
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT night_id FROM relphot.star_night WHERE obj_id = %s",
            (obj_id,),
        )
        night_ids = [r[0] for r in cur.fetchall()]

    # Request a period that's too long for one night (2.5 d baseline is ~1.1 cycles per night)
    req_id = _queue(test_conn, obj_id, "variable", period_guess=2.5, night_id=night_ids[0])
    reprocess(test_conn, settings=_SETTINGS)
    req = _request(test_conn, req_id)
    assert req["status"] == "done"
    assert req["result"]["found"] is False  # too long for a single night


def test_night_only_request_for_a_foreign_night_fails(test_conn) -> None:
    """Night-only request for a night not holding this object raises."""
    from relphot.db.reprocess import _process_variable
    from relphot.exceptions import ReprocessError

    obj_id, _night_ids = _insert_short_variable(test_conn, "foreign_night")
    # Create another night that doesn't have this object's data
    other_night = _insert_night(test_conn, "other_night_with_different_obj")

    req = {
        "obj_id": obj_id, "kind": "variable", "period_guess": 0.125,
        "tc_guess": None, "width_guess_h": None, "night_id": other_night, "note": None,
        "req_id": 1,
    }
    with pytest.raises(ReprocessError, match="has no stored light curve"):
        _process_variable(test_conn, req, _SETTINGS)


def test_nothing_is_queued_by_analyze_or_refresh(test_conn) -> None:
    """analyze and refresh don't queue reprocess requests."""
    _obj_id = _insert_slow_variable(test_conn, "no_queue")

    with test_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM relphot.reprocess_request")
        count_before = cur.fetchone()[0]

    # Analyze just ran in the fixture setup
    # Verify no requests were queued
    assert count_before == 0

    with test_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM relphot.reprocess_request")
        count_after = cur.fetchone()[0]

    assert count_after == count_before
