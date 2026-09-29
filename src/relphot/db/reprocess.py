"""User-guided reprocessing worker (``relphot db reprocess``).

A person asks the web for a re-run of one object with their own guesses; the web inserts a
row into ``relphot.reprocess_request`` (a trigger sends ``NOTIFY relphot_reprocess``) and this
worker, which runs on the host with the owner role like the loader, carries it out:

- ``kind = 'variable'``: a Lomb-Scargle period search around ``period_guess`` -- the windows
  ``guess * h * (1 +/- db.guided_period_window_frac)`` for h in (0.5, 1, 2) -- refined by the
  Fourier fit and stored as a ``period_estimate`` row with ``method = 'LS-guided'`` and the
  guess in ``guess`` (tie-calibrated magnitudes, and no per-night offsets, for a period beyond
  ``db.long_period_days``; see :func:`relphot.db.analyze._period_estimate`). It is verified
  against the literature period as usual and never changes the object's PERIOD: the web offers
  an "Adopt as period" button. ``night_id`` set: only that night is used (no tie); NULL: all
  of the object's nights.
- ``kind = 'transit'``: the trapezoid fit of :func:`relphot.db.analyze._fit_transit_shape` (same
  window ``tc +/- max(1.5 w, w + 1 h)``, bounds and duration lower-limit rules) started at
  ``tc_guess`` / ``width_guess_h``. The result is a new ``transit_shape`` linked to a NEW
  ``detection`` with ``origin = 'user'``; a search detection of that night within ``0.5 w`` of
  ``tc_guess`` is never overwritten (its id is kept in the new detection's ``extra``). User
  detections take part in ``transit_match`` (events are never merged), survive a night reload
  (which deletes ``origin = 'search'`` only) and never set ``is_exop`` by themselves.

Both kinds then rebuild the object's transit matches and refresh its summary fields.

Requests are worked off first-in first-out, one at a time; each is claimed in its own short
transaction (``status = 'running'``, visible to the web at once) and carried out in a second one
that also stores the result (``status = 'done'``, ``result`` jsonb). An exception rolls the work
back and marks the request ``failed`` with the error text. :func:`reprocess` assumes it is the
only worker: requests a crashed worker left ``running`` are queued again when it starts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from relphot.config import Settings
from relphot.db.analyze import (
    _ESTIMATE_INSERT_SQL,
    _SHAPE_INSERT_SQL,
    _estimate_values,
    _fetch_chunk_data,
    _fit_transit_shape,
    _NightData,
    _ObjTask,
    _period_estimate,
    _shape_values,
    _TransitDet,
    recompute_matches,
)
from relphot.db.refresh import refresh_objects
from relphot.exceptions import ReprocessError

logger = logging.getLogger(__name__)

__all__ = ["CHANNEL", "ReprocessReport", "process_request", "reprocess"]

#: The NOTIFY channel ``relphot.reprocess_request``'s insert trigger sends on.
CHANNEL = "relphot_reprocess"

_REQUEST_COLUMNS = (
    "req_id", "obj_id", "kind", "period_guess", "tc_guess", "width_guess_h", "night_id", "note",
)


@dataclass(slots=True)
class ReprocessReport:
    """Counts and elapsed time from one :func:`reprocess` call."""

    n_done: int
    n_failed: int
    elapsed_s: float


def _load_task(conn: psycopg.Connection, req: dict, settings: Settings) -> _ObjTask:
    task = _fetch_chunk_data(conn, [req["obj_id"]], settings.db).get(req["obj_id"])
    if task is None:
        msg = f"object {req['obj_id']} not found"
        raise ReprocessError(msg)
    if not task.nights:
        msg = "the object has no stored light curve"
        raise ReprocessError(msg)
    return task


def _restrict_to_night(task: _ObjTask, night_id: int) -> _ObjTask:
    """Restrict a task to one night; no multi-night tie (per-night normalised flux)."""
    nights = [nd for nd in task.nights if nd.night_id == night_id]
    if not nights:
        msg = f"night {night_id} has no stored light curve for this object"
        raise ReprocessError(msg)
    # one night: no multi-night tie applies (input 'night', per-night-normalised flux)
    return replace(task, nights=nights, tie=None, night_ties={})


def _process_variable(conn: psycopg.Connection, req: dict, settings: Settings) -> dict:
    task = _load_task(conn, req, settings)
    if req["night_id"] is not None:
        task = _restrict_to_night(task, req["night_id"])
    est = _period_estimate(task, guess=req["period_guess"])
    if est is None:
        msg = "no usable data for a period search"
        raise ReprocessError(msg)
    with conn.cursor() as cur:
        cur.execute(_ESTIMATE_INSERT_SQL, _estimate_values(req["obj_id"], est))
        (est_id,) = cur.fetchone()
    return {
        "est_id": est_id, "guess": est["guess"], "found": est["period"] is not None,
        "period": est["period"], "period_err": est["period_err"], "input": est["input"],
        "n_nights": est["n_nights"], "baseline_days": est["baseline_days"],
        "harmonic": est["harmonic"], "delta": est["delta"], "delta_err": est["delta_err"],
        "verify_status": est["verify_status"], "verify_note": est["verify_note"],
        "phase_coverage": est["phase_coverage"], "n_cycles": est["n_cycles"],
        "night_id": req["night_id"], "all_nights": req["night_id"] is None,
    }


def _night_of(task: _ObjTask, tc: float, night_id: int | None) -> _NightData:
    """The night holding ``tc`` (the requested one when ``night_id`` is given)."""
    for nd in task.nights:
        if night_id is not None and nd.night_id != night_id:
            continue
        t = nd.bjd[np.isfinite(nd.bjd)]
        if t.size and float(t.min()) <= tc <= float(t.max()):
            return nd
    where = f"night {night_id}" if night_id is not None else "any observed night"
    msg = f"tc_guess {tc:.5f} is not inside {where} of this object"
    raise ReprocessError(msg)


def _process_transit(conn: psycopg.Connection, req: dict, settings: Settings) -> dict:
    obj_id = req["obj_id"]
    task = _load_task(conn, req, settings)
    tc, width_h = float(req["tc_guess"]), float(req["width_guess_h"])
    nd = _night_of(task, tc, req["night_id"])
    tie_mags = [entry[0] for entry in task.night_ties.values()]
    tie_ref = float(np.mean(tie_mags)) if tie_mags else None
    start = _TransitDet(det_id=0, night_id=nd.night_id, tc=tc, depth=None, duration_h=width_h)
    shape = _fit_transit_shape(nd, start, task.night_ties.get(nd.night_id), tie_ref)
    if not shape["converged"]:
        msg = (
            f"the trapezoid fit did not converge ({shape['n_points']} points within "
            f"tc +/- max(1.5 w, w + 1 h) of {tc:.5f})"
        )
        raise ReprocessError(msg)

    with conn.cursor() as cur:
        # the search's own detection of this event, if any: linked in `extra`, never touched
        cur.execute(
            "SELECT det_id FROM relphot.detection WHERE obj_id = %s AND night_id = %s "
            "AND kind = 'transit' AND origin = 'search' AND tc_bjd_tdb IS NOT NULL "
            "AND abs(tc_bjd_tdb - %s) <= %s ORDER BY abs(tc_bjd_tdb - %s) LIMIT 1",
            (obj_id, nd.night_id, tc, 0.5 * width_h / 24.0, tc),
        )
        found = cur.fetchone()
        search_det_id = None if found is None else found[0]
        extra = {
            "req_id": req["req_id"], "tc_guess": tc, "width_guess_h": width_h,
            "search_det_id": search_det_id, "note": req["note"],
        }
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, depth, tc_bjd_tdb, "
            "duration_h, flags, extra, duration_lower_limit, origin) "
            "VALUES (%s, %s, 'transit', %s, %s, %s, 'USER', %s, %s, 'user') RETURNING det_id",
            (
                obj_id, nd.night_id, shape["depth"], shape["tc"], shape["t14_h"], Jsonb(extra),
                bool(shape["t14_lower_limit"]),
            ),
        )
        (det_id,) = cur.fetchone()
        shape["det_id"] = det_id
        cur.execute(_SHAPE_INSERT_SQL, _shape_values(obj_id, shape))

    return {
        "det_id": det_id, "search_det_id": search_det_id, "night_id": nd.night_id,
        **{
            key: shape[key] for key in (
                "tc", "tc_err", "depth", "depth_err", "t14_h", "t14_err", "t14_lower_limit",
                "incomplete_reason", "ingress_frac", "ingress_err", "chi2_red", "n_points",
                "input",
            )
        },
    }


def process_request(conn: psycopg.Connection, req: dict, settings: Settings | None = None) -> dict:
    """Carry out one request (a dict of :data:`_REQUEST_COLUMNS`) in ``conn``'s open transaction.

    Returns the ``result`` summary; commits nothing. Raises
    :class:`~relphot.exceptions.ReprocessError` (or whatever the analysis raises) on failure.
    """
    settings = settings if settings is not None else Settings()
    if req["kind"] == "variable":
        if not (req["period_guess"] and req["period_guess"] > 0):
            msg = "a variable request needs a positive period_guess"
            raise ReprocessError(msg)
        result = _process_variable(conn, req, settings)
    elif req["kind"] == "transit":
        if req["tc_guess"] is None or not (req["width_guess_h"] and req["width_guess_h"] > 0):
            msg = "a transit request needs tc_guess and a positive width_guess_h"
            raise ReprocessError(msg)
        result = _process_transit(conn, req, settings)
    else:
        msg = f"unknown request kind {req['kind']!r}"
        raise ReprocessError(msg)
    # both kinds: the object's matching-transit pairs and summary fields are brought up to date
    db = settings.db
    result["n_matches"] = recompute_matches(conn, req["obj_id"], db)
    refresh_objects(
        conn, [req["obj_id"]], bls_min_snr=db.bls_min_snr, ls_fap_threshold=db.ls_fap_threshold,
        class_multinight_kinds=db.class_multinight_kinds,
    )
    return result


def _claim_next(conn: psycopg.Connection) -> dict | None:
    """Mark the oldest queued request ``running`` (own transaction) and return it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE relphot.reprocess_request r
            SET status = 'running', started_at = now(), finished_at = NULL, error = NULL
            WHERE r.req_id = (
                SELECT req_id FROM relphot.reprocess_request WHERE status = 'queued'
                ORDER BY requested_at, req_id FOR UPDATE SKIP LOCKED LIMIT 1
            )
            RETURNING r.req_id, r.obj_id, r.kind, r.period_guess, r.tc_guess, r.width_guess_h,
                      r.night_id, r.note
            """
        )
        row = cur.fetchone()
    conn.commit()
    return None if row is None else dict(zip(_REQUEST_COLUMNS, row, strict=True))


def _run_one(conn: psycopg.Connection, req: dict, settings: Settings) -> bool:
    """Carry out a claimed request and record ``done`` / ``failed``. ``True`` on success."""
    try:
        result = process_request(conn, req, settings)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE relphot.reprocess_request SET status = 'done', finished_at = now(), "
                "result = %s, error = NULL WHERE req_id = %s",
                (Jsonb(result), req["req_id"]),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.exception("reprocess request %d failed", req["req_id"])
        text = str(exc) if isinstance(exc, ReprocessError) else f"{type(exc).__name__}: {exc}"
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE relphot.reprocess_request SET status = 'failed', finished_at = now(), "
                "error = %s WHERE req_id = %s",
                (text, req["req_id"]),
            )
        conn.commit()
        return False
    logger.info("reprocess request %d (%s) done", req["req_id"], req["kind"])
    return True


def _wait_for_notify(
    listen_conn: psycopg.Connection | None, poll_seconds: float, stop: threading.Event
) -> None:
    """Block until a NOTIFY arrives, ``poll_seconds`` pass, or ``stop`` is set."""
    if listen_conn is None:
        stop.wait(poll_seconds)
        return
    deadline = time.monotonic() + poll_seconds
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        for _ in listen_conn.notifies(timeout=min(remaining, 1.0), stop_after=1):
            return


def reprocess(
    conn: psycopg.Connection,
    *,
    settings: Settings | None = None,
    watch: bool = False,
    listen_conn: psycopg.Connection | None = None,
    poll_seconds: float = 60.0,
    stop: threading.Event | None = None,
) -> ReprocessReport:
    """Work off the queued reprocess requests, first-in first-out.

    Without ``watch`` it returns once the queue is empty. With ``watch`` it then waits for a
    NOTIFY on :data:`CHANNEL` (sent by the insert trigger) on ``listen_conn`` -- a separate
    connection, put into autocommit mode here; without one it only polls -- and polls anyway
    every ``poll_seconds``, until ``stop`` is set (the CLI sets it on SIGTERM / SIGINT).
    """
    t0 = time.monotonic()
    settings = settings if settings is not None else Settings()
    stop = stop if stop is not None else threading.Event()

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'queued', started_at = NULL "
            "WHERE status = 'running'"
        )
        if cur.rowcount:
            logger.warning("%d interrupted request(s) queued again", cur.rowcount)
    conn.commit()
    if watch and listen_conn is not None:
        listen_conn.autocommit = True
        listen_conn.execute(f"LISTEN {CHANNEL}")

    n_done = n_failed = 0
    while True:
        while not stop.is_set():
            req = _claim_next(conn)
            if req is None:
                break
            if _run_one(conn, req, settings):
                n_done += 1
            else:
                n_failed += 1
        if not watch or stop.is_set():
            break
        _wait_for_notify(listen_conn, poll_seconds, stop)
    return ReprocessReport(n_done=n_done, n_failed=n_failed, elapsed_s=time.monotonic() - t0)
