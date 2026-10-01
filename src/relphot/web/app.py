"""FastAPI application for the relphot results-database web API (see
docs/DB_PLAN.md, "Web (requirements 10-15)").

Every read goes through the ``relphot_ro`` role (SELECT only); the manual-edit
endpoints (``PATCH /api/object/{obj_id}``, ``PATCH /api/detection/{det_id}``,
``PUT /api/object/{obj_id}/night/{night_id}/review``,
``PUT`` / ``DELETE /api/object/{obj_id}/repeat_link``, and
``POST /api/object/{obj_id}/adopt_period``, ``POST /api/detections/review``) and the
reprocess-request endpoint (``POST /api/object/{obj_id}/reprocess``, an INSERT into the queue
``relphot db reprocess`` works off) go through ``relphot_web`` (SELECT plus UPDATE on a
fixed set of ``relphot.object`` / ``relphot.detection`` / ``relphot.user_night_review``
columns, plus INSERT on a fixed set of ``relphot.reprocess_request`` columns, plus
INSERT/UPDATE/DELETE on ``relphot.repeat_decision``). No query
ever interpolates a user-supplied *value* into SQL -- only a handful of fixed,
code-controlled identifiers (column names from a whitelist, ``ASC``/``DESC``) are
ever placed directly in a query string.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date
from importlib import resources
from typing import Literal

import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from relphot.config import DbSettings
from relphot.objflags import PLANET_CATALOGS, night_state, refresh_flags
from relphot.repeat import (
    decision_tol_days,
    load_families,
    match_decision,
    parse_when,
    predict_windows,
)
from relphot.web.db import column_exists, get_ro_conn, get_rw_conn, resolve_ro_dsn
from relphot.web.phase import fourier_model, phase_coverage
from relphot.web.tile_lc import (
    envelope,
    individual_ratio_curves,
    is_median_ensemble,
    member_rms,
    select_members,
)

__all__ = ["app"]

app = FastAPI(title="relphot results database")

# --------------------------------------------------------------------------
# JSON helpers: FastAPI's default encoder handles date/datetime/Decimal, but
# not NaN/Infinity (Python's json module renders them as the bare tokens
# NaN/Infinity, which is not valid JSON and breaks JSON.parse in the
# browser) -- every float column here (mag, period, snr, depth, the
# light-curve arrays, ...) can hold NaN, so every response is sanitised.
# --------------------------------------------------------------------------


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(value) for value in obj]
    return obj


def _json(data, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=_sanitize(jsonable_encoder(data)), status_code=status_code)


def _rows_to_dicts(cur, rows) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _fmt_duration(hours: float | None, lower_limit: bool | None) -> str | None:
    """A duration for display; an incomplete transit's is only a minimum, shown as ``>= x.xx h``."""
    if hours is None or not math.isfinite(hours):
        return None
    return f"\u2265 {hours:.2f} h" if lower_limit else f"{hours:.2f} h"


def _effective_status(status: str | None, auto_status: str | None) -> str:
    """A detection's status as it counts: a person's CONFIRMED / REJECTED stands, else
    ``'REJECTED (auto)'`` when the cross-candidate check rejected it, else UNCONFIRMED."""
    status = status or "UNCONFIRMED"
    if status == "UNCONFIRMED" and auto_status == "REJECTED":
        return "REJECTED (auto)"
    return status


#: Most look-alike events listed with a transit event (its ``n_similar`` is the full count).
_MAX_SIMILAR_SHOWN = 20
#: Look-alike events one stack plot carries (default and hard cap), and one bulk review changes.
_STACK_DEFAULT = 40
_STACK_MAX = 100


def _select_columns(
    conn: psycopg.Connection, table: str, base: list[str], optional: list[str]
) -> list[str]:
    cols = list(base)
    for c in optional:
        if column_exists(conn, table, c):
            cols.append(c)
    return cols


# --------------------------------------------------------------------------
# /api/search, /api/search.csv
# --------------------------------------------------------------------------

_SEARCH_COLUMNS = [
    "obj_id", "name", "ra", "dec", "class", "is_exop", "is_var", "known", "source_db",
    "known_name", "known_type", "period", "period_err", "period_source", "known_period",
    "mean_mag", "n_nights", "best_snr", "depth", "duration_h", "duration_lower_limit",
    "amplitude", "status", "first_night", "last_night", "n_review_pending", "n_nights_reviewed",
]


def _app_mag_sql(obj_ref: str) -> tuple[str, str, str]:
    """Correlated subqueries for an object's apparent mean magnitude, where it comes from and
    the zero point it rests on.

    relphot's instrumental magnitude plus the night's zero point (``night.zp``: the Gaia
    calibration of the frame headers where they carry one, else the telescope's measured
    zero point, else the assumed 20 mag), averaged over the object's nights like
    ``object.mean_mag``. The source is the ``zp_source`` all the object's nights share
    (``'gaia'``, ``'measured'`` or ``'assumed'``), else ``'mixed'``; the zero point is the
    value they share, NULL when the nights differ. ``obj_ref`` is ``o.obj_id`` or a ``%s``
    placeholder.
    """
    body = (
        "FROM relphot.star_night zs JOIN relphot.night zn ON zn.night_id = zs.night_id "
        f"WHERE zs.obj_id = {obj_ref} AND zs.mag IS NOT NULL)"
    )
    mean = f"(SELECT avg(zs.mag + zn.zp) {body}"
    source = (
        "(SELECT CASE WHEN count(DISTINCT zn.zp_source) = 1 THEN min(zn.zp_source) "
        f"WHEN count(*) > 0 THEN 'mixed' END {body}"
    )
    zp = f"(SELECT CASE WHEN min(zn.zp) = max(zn.zp) THEN min(zn.zp) END {body}"
    return mean, source, zp


_MEAN_MAG_APP_SQL, _MAG_ZP_SOURCE_SQL, _ = _app_mag_sql("o.obj_id")

#: Result columns computed per object: transit-event count, best "matching transits"
#: probability, the latest period verification (delta = P_obs/harmonic - P_lit, with
#: the status and note saying whether/why it could be verified), the apparent mean magnitude
#: and its zero-point source, and the RERUN requests (how many are still queued/running, and
#: the status/finish time of the newest one; correlated subqueries on
#: ``reprocess_request_obj_idx (obj_id, requested_at)``).
_SEARCH_DERIVED_SQL = {
    "n_transit_events": (
        "(SELECT count(*) FROM relphot.detection d "
        "WHERE d.obj_id = o.obj_id AND d.kind = 'transit')"
    ),
    "max_p_match": (
        "(SELECT max(m.p_match) FROM relphot.transit_match m WHERE m.obj_id = o.obj_id)"
    ),
    "n_repeat_families": (
        "(SELECT count(*) FROM relphot.repeat_family rf WHERE rf.obj_id = o.obj_id)"
    ),
    "period_delta": "pv.delta",
    "period_delta_err": "pv.delta_err",
    "period_verify_status": "pv.verify_status",
    "period_verify_note": "pv.verify_note",
    "mean_mag_app": _MEAN_MAG_APP_SQL,
    "mag_zp_source": _MAG_ZP_SOURCE_SQL,
    "n_rerun_pending": (
        "(SELECT count(*) FROM relphot.reprocess_request rr "
        "WHERE rr.obj_id = o.obj_id AND rr.status IN ('queued', 'running'))"
    ),
    "last_rerun_status": (
        "(SELECT rr.status FROM relphot.reprocess_request rr WHERE rr.obj_id = o.obj_id "
        "ORDER BY rr.requested_at DESC, rr.req_id DESC LIMIT 1)"
    ),
    "last_rerun_finished_at": (
        "(SELECT rr.finished_at FROM relphot.reprocess_request rr WHERE rr.obj_id = o.obj_id "
        "ORDER BY rr.requested_at DESC, rr.req_id DESC LIMIT 1)"
    ),
}
_SEARCH_JOINS = (
    " LEFT JOIN LATERAL (SELECT pe.delta, pe.delta_err, pe.verify_status, pe.verify_note "
    "FROM relphot.period_estimate pe "
    "WHERE pe.obj_id = o.obj_id AND pe.lit_period IS NOT NULL AND pe.method = 'LS' "
    "ORDER BY pe.last_night DESC NULLS LAST, pe.n_nights DESC, pe.computed_at DESC "
    "LIMIT 1) pv ON true"
)
_SEARCH_OUTPUT_COLUMNS = [*_SEARCH_COLUMNS, *_SEARCH_DERIVED_SQL]
_SEARCH_SELECT_SQL = ", ".join(
    [f"o.{c}" for c in _SEARCH_COLUMNS]
    + [f"{expr} AS {name}" for name, expr in _SEARCH_DERIVED_SQL.items()]
)
_SORT_WHITELIST = set(_SEARCH_OUTPUT_COLUMNS)
_CLASS_LABELS = {"UNC", "EXOP", "VAR", "EXOP+VAR"}
#: relphot.detection.kind's CHECK constraint values (schema/001_init.sql).
_DETECTION_KINDS = {"transit", "variable", "internight", "ls_periodic", "bls", "recurrent"}
_DETECTION_SCOPES = {"night", "multinight"}


def _search_filters(
    class_: list[str] | None = Query(default=None, alias="class"),  # noqa: B008
    is_exop: bool | None = Query(default=None),
    is_var: bool | None = Query(default=None),
    min_p_match: float | None = Query(
        default=None, description="objects with a transit pair matching with p >= this"
    ),
    known: bool | None = Query(default=None),
    source_db: str | None = Query(default=None),
    status: str | None = Query(default=None),
    needs_review: bool | None = Query(default=None),
    user_reviewed: bool | None = Query(default=None),
    rerun_pending: bool | None = Query(
        default=None, description="objects with a RERUN request still queued or running"
    ),
    has_repeat_family: bool | None = Query(
        default=None, description="objects with (or without) a repeated-event family"
    ),
    telescope: str | None = Query(default=None),
    name: str | None = Query(default=None),
    gaia_id: str | None = Query(default=None),
    ra: float | None = Query(default=None, description="cone search centre RA, degrees"),
    dec: float | None = Query(default=None, description="cone search centre Dec, degrees"),
    radius: float | None = Query(default=None, description="cone search radius, arcsec"),
    mag_min: float | None = Query(default=None),
    mag_max: float | None = Query(default=None),
    period_min: float | None = Query(default=None),
    period_max: float | None = Query(default=None),
    snr_min: float | None = Query(default=None),
    depth_min: float | None = Query(default=None),
    tier_max: int | None = Query(default=None),
    n_nights_min: int | None = Query(default=None),
    night_from: date | None = Query(default=None),  # noqa: B008
    night_to: date | None = Query(default=None),  # noqa: B008
    has_periodogram: bool | None = Query(default=None),
    detection_kind: list[str] | None = Query(default=None),  # noqa: B008
    detection_scope: str | None = Query(default=None),
) -> dict:
    return {
        "class_": class_, "is_exop": is_exop, "is_var": is_var, "min_p_match": min_p_match,
        "known": known, "source_db": source_db, "status": status,
        "needs_review": needs_review, "user_reviewed": user_reviewed,
        "rerun_pending": rerun_pending, "has_repeat_family": has_repeat_family,
        "telescope": telescope, "name": name, "gaia_id": gaia_id, "ra": ra, "dec": dec,
        "radius": radius, "mag_min": mag_min, "mag_max": mag_max, "period_min": period_min,
        "period_max": period_max, "snr_min": snr_min, "depth_min": depth_min,
        "tier_max": tier_max, "n_nights_min": n_nights_min, "night_from": night_from,
        "night_to": night_to, "has_periodogram": has_periodogram,
        "detection_kind": detection_kind, "detection_scope": detection_scope,
    }


def _build_where(f: dict) -> tuple[str, dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}

    if f["class_"]:
        invalid_class = sorted(set(f["class_"]) - _CLASS_LABELS)
        if invalid_class:
            raise HTTPException(status_code=400, detail=f"invalid class: {invalid_class!r}")
        clauses.append("o.class = ANY(%(class_list)s)")
        params["class_list"] = list(f["class_"])
    if f["is_exop"] is not None:
        clauses.append("o.is_exop = %(is_exop)s")
        params["is_exop"] = f["is_exop"]
    if f["is_var"] is not None:
        clauses.append("o.is_var = %(is_var)s")
        params["is_var"] = f["is_var"]
    if f["min_p_match"] is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM relphot.transit_match m WHERE m.obj_id = o.obj_id "
            "AND m.p_match >= %(min_p_match)s)"
        )
        params["min_p_match"] = f["min_p_match"]
    if f["known"] is not None:
        clauses.append("o.known = %(known)s")
        params["known"] = f["known"]
    if f["source_db"]:
        clauses.append("o.source_db ILIKE %(source_db)s")
        params["source_db"] = f"%{f['source_db']}%"
    if f["status"]:
        clauses.append("o.status = %(status)s")
        params["status"] = f["status"]
    if f["needs_review"] is not None:
        if f["needs_review"]:
            clauses.append("o.n_review_pending > 0")
        else:
            clauses.append("o.n_review_pending = 0")
    if f["user_reviewed"] is not None:
        if f["user_reviewed"]:
            clauses.append("o.n_nights_reviewed > 0")
        else:
            clauses.append("o.n_nights_reviewed = 0")
    if f["rerun_pending"] is not None:
        pending = (
            "EXISTS (SELECT 1 FROM relphot.reprocess_request rr WHERE rr.obj_id = o.obj_id "
            "AND rr.status IN ('queued', 'running'))"
        )
        clauses.append(pending if f["rerun_pending"] else f"NOT {pending}")
    if f["has_repeat_family"] is not None:
        family = "EXISTS (SELECT 1 FROM relphot.repeat_family rf WHERE rf.obj_id = o.obj_id)"
        clauses.append(family if f["has_repeat_family"] else f"NOT {family}")
    if f["telescope"]:
        clauses.append(
            "EXISTS (SELECT 1 FROM relphot.star_night sn "
            "JOIN relphot.night n ON n.night_id = sn.night_id "
            "WHERE sn.obj_id = o.obj_id AND n.telescope = %(telescope)s)"
        )
        params["telescope"] = f["telescope"]
    if f["name"]:
        clauses.append("o.name ILIKE %(name)s")
        params["name"] = f"%{f['name']}%"
    if f["gaia_id"]:
        clauses.append("o.gaia_id ILIKE %(gaia_id)s")
        params["gaia_id"] = f"%{f['gaia_id']}%"
    if f["ra"] is not None and f["dec"] is not None and f["radius"] is not None:
        clauses.append(
            "q3c_radial_query(o.ra, o.dec, %(cone_ra)s, %(cone_dec)s, %(cone_radius_deg)s)"
        )
        params["cone_ra"] = f["ra"]
        params["cone_dec"] = f["dec"]
        params["cone_radius_deg"] = f["radius"] / 3600.0
    if f["mag_min"] is not None:
        clauses.append("o.mean_mag >= %(mag_min)s")
        params["mag_min"] = f["mag_min"]
    if f["mag_max"] is not None:
        clauses.append("o.mean_mag <= %(mag_max)s")
        params["mag_max"] = f["mag_max"]
    if f["period_min"] is not None:
        clauses.append("o.period >= %(period_min)s")
        params["period_min"] = f["period_min"]
    if f["period_max"] is not None:
        clauses.append("o.period <= %(period_max)s")
        params["period_max"] = f["period_max"]
    if f["snr_min"] is not None:
        clauses.append("o.best_snr >= %(snr_min)s")
        params["snr_min"] = f["snr_min"]
    if f["depth_min"] is not None:
        clauses.append("o.depth >= %(depth_min)s")
        params["depth_min"] = f["depth_min"]
    if f["tier_max"] is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM relphot.detection d WHERE d.obj_id = o.obj_id "
            "AND d.kind = 'transit' AND d.tier <= %(tier_max)s)"
        )
        params["tier_max"] = f["tier_max"]
    if f["n_nights_min"] is not None:
        clauses.append("o.n_nights >= %(n_nights_min)s")
        params["n_nights_min"] = f["n_nights_min"]
    if f["night_from"] is not None or f["night_to"] is not None:
        sub = (
            "EXISTS (SELECT 1 FROM relphot.star_night sn "
            "JOIN relphot.night n ON n.night_id = sn.night_id "
            "WHERE sn.obj_id = o.obj_id"
        )
        if f["night_from"] is not None:
            sub += " AND n.night_date >= %(night_from)s"
            params["night_from"] = f["night_from"]
        if f["night_to"] is not None:
            sub += " AND n.night_date <= %(night_to)s"
            params["night_to"] = f["night_to"]
        sub += ")"
        clauses.append(sub)
    if f["has_periodogram"] is not None:
        exists = "EXISTS (SELECT 1 FROM relphot.periodogram p WHERE p.obj_id = o.obj_id)"
        clauses.append(exists if f["has_periodogram"] else f"NOT {exists}")
    if f["detection_scope"] is not None and f["detection_scope"] not in _DETECTION_SCOPES:
        raise HTTPException(
            status_code=400, detail=f"invalid detection_scope: {f['detection_scope']!r}"
        )
    if f["detection_kind"]:
        invalid = sorted(set(f["detection_kind"]) - _DETECTION_KINDS)
        if invalid:
            raise HTTPException(status_code=400, detail=f"invalid detection_kind: {invalid!r}")
        scope_sql = ""
        if f["detection_scope"] == "night":
            scope_sql = " AND d.night_id IS NOT NULL"
        elif f["detection_scope"] == "multinight":
            scope_sql = " AND d.mn_run_id IS NOT NULL"
        clauses.append(
            "EXISTS (SELECT 1 FROM relphot.detection d WHERE d.obj_id = o.obj_id "
            "AND d.kind = ANY(%(detection_kind_list)s)" + scope_sql + ")"
        )
        params["detection_kind_list"] = list(f["detection_kind"])

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def _validate_sort(sort: str, order: str) -> tuple[str, str]:
    if sort not in _SORT_WHITELIST:
        raise HTTPException(status_code=400, detail=f"invalid sort column: {sort!r}")
    order_norm = order.lower()
    if order_norm not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail=f"invalid order: {order!r}")
    return sort, order_norm


def _order_by_sql(sort_col: str, order_norm: str) -> str:
    """``ORDER BY`` clause for a whitelisted column (an object column or a derived output name)."""
    target = sort_col if sort_col in _SEARCH_DERIVED_SQL else f"o.{sort_col}"
    return f"ORDER BY {target} {order_norm.upper()} NULLS LAST"


@app.get("/api/search")
def search(
    filters: dict = Depends(_search_filters),  # noqa: B008
    sort: str = Query(default="obj_id"),
    order: str = Query(default="asc"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    sort_col, order_norm = _validate_sort(sort, order)
    where_sql, params = _build_where(filters)
    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM relphot.object o{where_sql}", params)
        (total,) = cur.fetchone()
        query = (
            f"SELECT {_SEARCH_SELECT_SQL} FROM relphot.object o{_SEARCH_JOINS}{where_sql} "
            f"{_order_by_sql(sort_col, order_norm)} "
            "LIMIT %(limit)s OFFSET %(offset)s"
        )
        cur.execute(query, {**params, "limit": limit, "offset": offset})
        rows = _rows_to_dicts(cur, cur.fetchall())
    for row in rows:
        row["duration_display"] = _fmt_duration(row["duration_h"], row["duration_lower_limit"])
    return _json({"total": total, "rows": rows})


@app.get("/api/search.csv")
def search_csv(
    filters: dict = Depends(_search_filters),  # noqa: B008
    sort: str = Query(default="obj_id"),
    order: str = Query(default="asc"),
    limit: int = Query(default=1000, ge=1, le=100000),
):
    sort_col, order_norm = _validate_sort(sort, order)
    where_sql, params = _build_where(filters)
    with get_ro_conn() as conn, conn.cursor() as cur:
        query = (
            f"SELECT {_SEARCH_SELECT_SQL} FROM relphot.object o{_SEARCH_JOINS}{where_sql} "
            f"{_order_by_sql(sort_col, order_norm)} LIMIT %(limit)s"
        )
        cur.execute(query, {**params, "limit": limit})
        rows = _rows_to_dicts(cur, cur.fetchall())
    buf = io.StringIO()
    writer = csv.writer(buf)
    csv_columns = [*_SEARCH_OUTPUT_COLUMNS, "duration_display"]
    writer.writerow(csv_columns)
    for row in rows:
        row["duration_display"] = _fmt_duration(row["duration_h"], row["duration_lower_limit"])
        writer.writerow([row[c] for c in csv_columns])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="search.csv"'},
    )


# --------------------------------------------------------------------------
# /api/sql
# --------------------------------------------------------------------------

_SQL_MAX_ROWS = 5000


class SqlBody(BaseModel):
    sql: str


@app.post("/api/sql")
def run_sql(body: SqlBody):
    conn = psycopg.connect(resolve_ro_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '30000'")
            cur.execute(body.sql)
            if cur.description is None:
                columns: list[str] = []
                rows: list[tuple] = []
            else:
                columns = [d.name for d in cur.description]
                rows = cur.fetchmany(_SQL_MAX_ROWS + 1)
    except psycopg.Error as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.rollback()
        conn.close()
    truncated = len(rows) > _SQL_MAX_ROWS
    if truncated:
        rows = rows[:_SQL_MAX_ROWS]
    return _json({"columns": columns, "rows": [list(r) for r in rows], "truncated": truncated})


# --------------------------------------------------------------------------
# /api/object/{obj_id} and its sub-resources
# --------------------------------------------------------------------------

_OBJECT_BASE_COLUMNS = [
    "obj_id", "name", "ra", "dec", "gaia_id", "mean_mag", "n_nights", "class", "class_source",
    "is_exop", "is_var", "exop_source", "var_source", "period", "period_err",
    "period_n_nights", "period_source", "known", "source_db", "known_name", "known_type",
    "known_period", "status", "best_snr", "depth", "duration_h", "amplitude", "n_detections",
    "first_night", "last_night", "neighbour_sep_arcsec", "notes", "updated_at",
    "duration_lower_limit",
]
_OBJECT_OPTIONAL_COLUMNS = ["data_updated_at", "n_review_pending", "n_nights_reviewed"]


def _similar_info(cur, det_ids: list[int], *, stack: bool = False) -> dict[int, dict]:
    """The transit events behind ``det_ids``, keyed by det_id: object, centre time, depth and T14
    (the shape fit's where there is one, else the detection's own).

    ``stack`` adds what the similar-events plot and its bulk review need: the night, the
    verdicts and notes, and the trapezoid's ingress fraction and convergence.
    """
    if not det_ids:
        return {}
    extra = (
        ", d.night_id, n.label AS night_label, d.status, d.auto_status, d.notes, "
        "ts.ingress_frac, ts.converged, ts.incomplete_reason"
        if stack else ""
    )
    night_join = "LEFT JOIN relphot.night n ON n.night_id = d.night_id " if stack else ""
    cur.execute(
        "SELECT d.det_id, d.obj_id, o.name AS obj_name, "
        "COALESCE(ts.tc, d.tc_bjd_tdb) AS tc, COALESCE(ts.depth, d.depth) AS depth, "
        "COALESCE(ts.t14_h, d.duration_h) AS t14_h, "
        "(COALESCE(ts.t14_lower_limit, false) OR d.duration_lower_limit) "
        f"AS t14_lower_limit{extra} "
        "FROM relphot.detection d JOIN relphot.object o ON o.obj_id = d.obj_id "
        f"{night_join}"
        "LEFT JOIN relphot.transit_shape ts ON ts.det_id = d.det_id "
        "WHERE d.det_id = ANY(%s) AND d.kind = 'transit'",
        (det_ids,),
    )
    info: dict[int, dict] = {}
    for sim in _rows_to_dicts(cur, cur.fetchall()):
        sim["duration_display"] = _fmt_duration(sim["t14_h"], sim["t14_lower_limit"])
        if stack:
            sim["effective_status"] = _effective_status(sim["status"], sim["auto_status"])
        info[sim["det_id"]] = sim
    return info


def _rejected_ids(cur, det_ids: list[int]) -> set[int]:
    """The ``det_ids`` a person REJECTED (``status``; the automatic ``auto_status`` is not one)."""
    if not det_ids:
        return set()
    cur.execute(
        "SELECT det_id FROM relphot.detection WHERE det_id = ANY(%s) AND status = 'REJECTED'",
        (det_ids,),
    )
    return {row[0] for row in cur.fetchall()}


def _visible_similar(
    det_ids: list[int], rejected: set[int], *, limit: int, include_rejected: bool = False
) -> tuple[list[int], int]:
    """``(the look-alikes to list, how many rejected ones are hidden)``.

    ``det_ids`` is the stored list, nearest in time first. A look-alike a person REJECTED leaves
    the list (unless ``include_rejected``); the first ``limit`` of the rest are kept. The
    coincidence veto's own counts (``n_similar``, ``auto_status``) are not touched by this.
    """
    if include_rejected:
        return det_ids[:limit], 0
    kept = [j for j in det_ids if j not in rejected]
    return kept[:limit], len(det_ids) - len(kept)


@app.get("/api/object/{obj_id}")
def object_detail(obj_id: int):
    with get_ro_conn() as conn:
        obj_cols = _select_columns(conn, "object", _OBJECT_BASE_COLUMNS, _OBJECT_OPTIONAL_COLUMNS)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(obj_cols)} FROM relphot.object WHERE obj_id = %s", (obj_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="object not found")
            obj = dict(zip(obj_cols, row, strict=True))
            obj["duration_display"] = _fmt_duration(
                obj["duration_h"], obj["duration_lower_limit"]
            )
            mean_app_sql, zp_source_sql, zp_sql = _app_mag_sql("%s")
            cur.execute(
                f"SELECT {mean_app_sql}, {zp_source_sql}, {zp_sql}", (obj_id, obj_id, obj_id)
            )
            obj["mean_mag_app"], obj["mag_zp_source"], mag_zp = cur.fetchone()
            obj["mag_zp"] = None if mag_zp is None else round(mag_zp, 4)

            cur.execute(
                "SELECT catalog, name, type, period, period_err, sep_arcsec, reference "
                "FROM relphot.catalog_match WHERE obj_id = %s ORDER BY catalog",
                (obj_id,),
            )
            catalog_matches = _rows_to_dicts(cur, cur.fetchall())

            # Compute lit_exop and lit_var from catalog matches
            lit_exop = any(cm["catalog"] in PLANET_CATALOGS for cm in catalog_matches)
            lit_var = any(cm["catalog"] not in PLANET_CATALOGS for cm in catalog_matches)

            cur.execute(
                "SELECT d.det_id, d.kind, d.snr, d.depth, d.tc_bjd_tdb, d.duration_h, d.tier, "
                "d.flags, d.amplitude, d.excess, d.period, d.fap, d.extra, d.night_id, "
                "n.label AS night_label, n.telescope, d.mn_run_id, mr.stem AS mn_run_stem, "
                "d.status, d.notes, d.duration_lower_limit, d.origin, "
                "d.auto_status, d.auto_reason "
                "FROM relphot.detection d "
                "LEFT JOIN relphot.night n ON n.night_id = d.night_id "
                "LEFT JOIN relphot.mn_run mr ON mr.mn_run_id = d.mn_run_id "
                "WHERE d.obj_id = %s ORDER BY d.tier NULLS LAST, d.snr DESC NULLS LAST",
                (obj_id,),
            )
            detections = _rows_to_dicts(cur, cur.fetchall())
            for det in detections:
                det["duration_display"] = _fmt_duration(
                    det["duration_h"], det["duration_lower_limit"]
                )
                det["effective_status"] = _effective_status(det["status"], det["auto_status"])

            cur.execute(
                "SELECT sn.night_id, n.label, n.telescope, n.night_date, sn.n_epochs, sn.rms, "
                "sn.expected_noise, sn.mag, sn.best_aperture, sn.err_scale, sn.blended, "
                "n.zp, n.zp_source, sn.mag + n.zp AS mag_app, "
                "(lc.obj_id IS NOT NULL) AS has_lc, "
                "r.exop_verdict AS review_exop, r.var_verdict AS review_var, "
                "r.note AS review_note, r.updated_at AS review_updated_at, "
                "sn.tile, sn.star_id, sn.is_comparison, "
                "(tl.night_id IS NOT NULL) AS has_reference, "
                "COALESCE(tl.n_comp,0) > 0 AS has_comparison "
                "FROM relphot.star_night sn "
                "JOIN relphot.night n ON n.night_id = sn.night_id "
                "LEFT JOIN relphot.lightcurve lc "
                "ON lc.obj_id = sn.obj_id AND lc.night_id = sn.night_id "
                "LEFT JOIN relphot.user_night_review r "
                "ON r.obj_id = sn.obj_id AND r.night_id = sn.night_id "
                "LEFT JOIN relphot.tile_lc tl "
                "ON tl.night_id = sn.night_id AND tl.tile = sn.tile "
                "AND tl.aperture = sn.best_aperture "
                "WHERE sn.obj_id = %s ORDER BY n.night_date",
                (obj_id,),
            )
            nights = _rows_to_dicts(cur, cur.fetchall())
            # Guard: if schema v11 not loaded, set has_reference/has_comparison to false
            if not column_exists(conn, "tile_lc", "night_id"):
                for night in nights:
                    night["has_reference"] = False
                    night["has_comparison"] = False

            pg_cols = _select_columns(
                conn, "periodogram", ["scope", "method", "peak_period", "peak_power", "fap"],
                ["input", "coarsened", "extra"],
            )
            cur.execute(
                f"SELECT {', '.join(pg_cols)} FROM relphot.periodogram "
                "WHERE obj_id = %s ORDER BY scope, method",
                (obj_id,),
            )
            periodograms = _rows_to_dicts(cur, cur.fetchall())

            cur.execute(
                "SELECT t.mn_run_id, mr.stem, mr.anchor, mr.labels, "
                "array_agg(t.night_id ORDER BY t.night_id) AS night_ids "
                "FROM relphot.tie t JOIN relphot.mn_run mr ON mr.mn_run_id = t.mn_run_id "
                "WHERE t.obj_id = %s GROUP BY t.mn_run_id, mr.stem, mr.anchor, mr.labels "
                "ORDER BY mr.stem",
                (obj_id,),
            )
            ties = _rows_to_dicts(cur, cur.fetchall())

            cur.execute(
                "SELECT d.det_id, d.night_id, n.label AS night_label, n.telescope, "
                "n.night_date, d.tc_bjd_tdb AS det_tc, d.depth AS det_depth, "
                "d.duration_h AS det_duration_h, d.snr, d.tier, d.flags, d.status, d.notes, "
                "d.duration_lower_limit AS det_duration_lower_limit, d.origin, "
                "d.auto_status, d.auto_reason, "
                "tcx.n_similar, tcx.n_expected, tcx.p_chance, tcx.similar_det_ids, "
                "ts.tc, ts.tc_err, ts.depth, ts.depth_err, ts.t14_h, ts.t14_err, "
                "ts.t14_lower_limit, ts.incomplete_reason, "
                "ts.ingress_frac, ts.ingress_err, ts.chi2_red, ts.n_points, ts.input, "
                "ts.converged "
                "FROM relphot.detection d "
                "JOIN relphot.night n ON n.night_id = d.night_id "
                "LEFT JOIN relphot.transit_shape ts ON ts.det_id = d.det_id "
                "LEFT JOIN relphot.transit_coincidence tcx ON tcx.det_id = d.det_id "
                "WHERE d.obj_id = %s AND d.kind = 'transit' "
                "ORDER BY COALESCE(ts.tc, d.tc_bjd_tdb), d.det_id",
                (obj_id,),
            )
            transit_events = _rows_to_dicts(cur, cur.fetchall())
            for ev in transit_events:
                # the fitted T14 when there is one, else the detection's own duration; either
                # is a lower limit if the shape fit or the search flagged the event incomplete
                lower = bool(ev["t14_lower_limit"]) or bool(ev["det_duration_lower_limit"])
                hours = ev["t14_h"] if ev["t14_h"] is not None else ev["det_duration_h"]
                ev["duration_lower_limit"] = lower
                ev["duration_display"] = _fmt_duration(hours, lower)
                ev["effective_status"] = _effective_status(ev["status"], ev["auto_status"])

            # the look-alikes of each event (other objects' events of the same night): the
            # nearest in time, capped; n_similar is the full count. One a person REJECTED is no
            # longer listed (n_similar_rejected counts those); the veto's own numbers stand.
            all_similar = {ev["det_id"]: ev.pop("similar_det_ids") or [] for ev in transit_events}
            rejected = _rejected_ids(cur, sorted({j for ids in all_similar.values() for j in ids}))
            similar_ids: dict[int, list[int]] = {}
            for ev in transit_events:
                similar_ids[ev["det_id"]], ev["n_similar_rejected"] = _visible_similar(
                    all_similar[ev["det_id"]], rejected, limit=_MAX_SIMILAR_SHOWN
                )
            similar_info = _similar_info(
                cur, sorted({j for ids in similar_ids.values() for j in ids})
            )
            for ev in transit_events:
                ev["similar_events"] = [
                    similar_info[j] for j in similar_ids[ev["det_id"]] if j in similar_info
                ]

            cur.execute(
                "SELECT m.det_a, m.det_b, na.label AS night_a, nb.label AS night_b, "
                "m.dt_days, m.depth_z, m.t14_z, m.ingress_z, m.chi2, m.dof, m.p_match, "
                "m.same_telescope, m.commensurate_periods, "
                "sa.t14_h AS t14_a_h, (COALESCE(sa.t14_lower_limit, false) "
                "OR da.duration_lower_limit) AS t14_a_lower_limit, "
                "sb.t14_h AS t14_b_h, (COALESCE(sb.t14_lower_limit, false) "
                "OR db.duration_lower_limit) AS t14_b_lower_limit "
                "FROM relphot.transit_match m "
                "JOIN relphot.detection da ON da.det_id = m.det_a "
                "JOIN relphot.night na ON na.night_id = da.night_id "
                "JOIN relphot.detection db ON db.det_id = m.det_b "
                "JOIN relphot.night nb ON nb.night_id = db.night_id "
                "LEFT JOIN relphot.transit_shape sa ON sa.det_id = m.det_a "
                "LEFT JOIN relphot.transit_shape sb ON sb.det_id = m.det_b "
                "WHERE m.obj_id = %s "
                "ORDER BY m.p_match DESC NULLS LAST, m.det_a, m.det_b",
                (obj_id,),
            )
            transit_matches = _rows_to_dicts(cur, cur.fetchall())
            for m in transit_matches:
                m["t14_a_display"] = _fmt_duration(m["t14_a_h"], m["t14_a_lower_limit"])
                m["t14_b_display"] = _fmt_duration(m["t14_b_h"], m["t14_b_lower_limit"])

            repeat_families, repeat_decisions = _object_repeat(conn, cur, obj_id)

            cur.execute(
                "SELECT est_id, computed_at, method, input, night_ids, n_nights, last_night, "
                "baseline_days, period, period_err, power, fap, lit_period, lit_period_err, "
                "lit_catalog, harmonic, delta, delta_err, delta_z, verify_status, verify_note, "
                "guess, phase_coverage, n_cycles, alias_periods, alias_powers "
                "FROM relphot.period_estimate WHERE obj_id = %s "
                "ORDER BY last_night, n_nights, computed_at",
                (obj_id,),
            )
            period_estimates = _rows_to_dicts(cur, cur.fetchall())

        # Enrich nights with night_state by grouping detections by night_id
        night_dets_map: dict[int, list[dict]] = {}
        for det in detections:
            if det["night_id"] is not None:
                if det["night_id"] not in night_dets_map:
                    night_dets_map[det["night_id"]] = []
                night_dets_map[det["night_id"]].append(det)

        for night in nights:
            nid = night["night_id"]
            ns = night_state(
                night_dets_map.get(nid, []),
                night["review_exop"],
                night["review_var"],
            )
            night.update(ns)

        # Add literature flags to object
        obj["lit_exop"] = lit_exop
        obj["lit_var"] = lit_var

    return _json(
        {
            "object": obj,
            "catalog_matches": catalog_matches,
            "detections": detections,
            "nights": nights,
            "periodograms": periodograms,
            "ties": ties,
            "transit_events": transit_events,
            "transit_matches": transit_matches,
            "repeat_families": repeat_families,
            "repeat_decisions": repeat_decisions,
            "period_estimates": period_estimates,
        }
    )


def _stack_lcs(cur, det_ids: list[int]) -> dict[int, dict]:
    """The light curve of each event's own night, keyed by det_id: ``flux`` and ``flux_err`` over
    the curve's median flux (rounded), one query for all. An event without a usable curve is
    left out."""
    cur.execute(
        "SELECT d.det_id, lc.bjd_tdb, lc.flux, lc.flux_err FROM relphot.detection d "
        "JOIN relphot.lightcurve lc ON lc.obj_id = d.obj_id AND lc.night_id = d.night_id "
        "WHERE d.det_id = ANY(%s)",
        (det_ids,),
    )
    curves: dict[int, dict] = {}
    for det_id, bjd_tdb, flux, flux_err in cur.fetchall():
        flux = np.asarray(flux, dtype=float)
        finite = flux[np.isfinite(flux)]
        median = float(np.median(finite)) if finite.size else float("nan")
        if not median > 0:
            continue
        curves[det_id] = {
            "bjd_tdb": np.round(np.asarray(bjd_tdb, dtype=float), 6).tolist(),
            "flux": np.round(flux / median, 5).tolist(),
            "flux_err": np.round(np.asarray(flux_err, dtype=float) / median, 5).tolist(),
        }
    return curves


@app.get("/api/detection/{det_id}/similar")
def detection_similar(
    det_id: int,
    limit: int = Query(default=_STACK_DEFAULT, ge=1, le=_STACK_MAX),
    include_rejected: bool = Query(default=False),
):
    """A transit event (the ``anchor``) and its coincidence look-alikes, nearest in time first,
    each with its own night's light curve over its median -- the data of the stack plot.

    Look-alikes a person REJECTED are left out unless ``include_rejected`` (``n_rejected_hidden``
    says how many); the anchor is always returned.
    """
    with get_ro_conn() as conn, conn.cursor() as cur:
        anchor = _similar_info(cur, [det_id], stack=True).get(det_id)
        if anchor is None:
            raise HTTPException(status_code=404, detail="transit detection not found")
        cur.execute(
            "SELECT similar_det_ids FROM relphot.transit_coincidence WHERE det_id = %s", (det_id,)
        )
        row = cur.fetchone()
        similar_ids = (row[0] if row else None) or []
        rejected = _rejected_ids(cur, similar_ids)
        shown, hidden = _visible_similar(
            similar_ids, rejected, limit=limit, include_rejected=include_rejected
        )
        info = _similar_info(cur, shown, stack=True)
        events = [info[j] for j in shown if j in info]
        curves = _stack_lcs(cur, [det_id, *shown])
    for ev in (anchor, *events):
        ev["lc"] = curves.get(ev["det_id"])
    return _json(
        {
            "anchor": anchor,
            "n_total": len(similar_ids),
            "n_rejected": len(rejected),
            "n_rejected_hidden": hidden,
            "n_returned": len(events),
            "events": events,
        }
    )


@app.get("/api/object/{obj_id}/lc")
def object_lc(obj_id: int, night_id: int):
    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT frame_index, bjd_tdb, flux, flux_err, flux_raw "
            "FROM relphot.lightcurve WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no light curve for this object/night")
        frame_index, bjd_tdb, flux, flux_err, flux_raw = row

        cur.execute(
            "SELECT n.label, n.zp, n.zp_source, sn.mag + n.zp FROM relphot.night n "
            "LEFT JOIN relphot.star_night sn ON sn.night_id = n.night_id AND sn.obj_id = %s "
            "WHERE n.night_id = %s",
            (obj_id, night_id),
        )
        night_label, zp, zp_source, app_mag = cur.fetchone() or (None, None, None, None)

        cur.execute(
            "SELECT frame_index, file_name, airmass FROM relphot.frame "
            "WHERE night_id = %s AND frame_index = ANY(%s)",
            (night_id, list(frame_index)),
        )
        frame_map = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    file_name = [frame_map.get(i, (None, None))[0] for i in frame_index]
    airmass = [frame_map.get(i, (None, None))[1] for i in frame_index]

    return _json(
        {
            "bjd_tdb": list(bjd_tdb),
            "flux": list(flux),
            "flux_err": list(flux_err),
            "flux_raw": list(flux_raw),
            "frame_index": list(frame_index),
            "file_name": file_name,
            "airmass": airmass,
            "night_label": night_label,
            "zp": zp,
            "zp_source": zp_source,
            "app_mag": None if app_mag is None else round(app_mag, 4),
        }
    )


def _combined_lc(cur, obj_id: int) -> dict:
    """Every night's light curve of one object on one scale: the "All nights" series.

    ``mode`` is ``'tied-mag'`` when a multi-night run ties >= 2 of the object's nights (each
    night's flux is put on that night's tied magnitude; nights the run does not tie are left
    out), else ``'night-normalised'`` (each night divided by its own median: the nights are
    NOT tied). ``value_err`` is the photometric error alone, ``tie_err`` (``None`` untied) the
    night's tie error. ``app_mag`` is each point's night's apparent mean magnitude (``None`` if
    unknown); tied, ``zp`` (``zp_source``) is the zero point that turns the tied magnitudes
    into apparent ones (the anchor night's: the tie puts every night on its scale), untied
    ``zp_source`` sums up the nights' zero points. Raises a 404 for an unknown object.
    """
    cur.execute("SELECT 1 FROM relphot.object WHERE obj_id = %s", (obj_id,))
    if cur.fetchone() is None:
        raise HTTPException(status_code=404, detail="object not found")

    cur.execute(
        "SELECT t.mn_run_id, COUNT(DISTINCT t.night_id) AS n_tied, mr.loaded_at "
        "FROM relphot.tie t JOIN relphot.mn_run mr ON mr.mn_run_id = t.mn_run_id "
        "WHERE t.obj_id = %s GROUP BY t.mn_run_id, mr.loaded_at "
        "HAVING COUNT(DISTINCT t.night_id) >= 2 "
        "ORDER BY n_tied DESC, mr.loaded_at DESC LIMIT 1",
        (obj_id,),
    )
    best_tie = cur.fetchone()

    if best_tie is not None:
        mode = "tied-mag"
        mn_run_id = best_tie[0]
        cur.execute(
            "SELECT night_id, mag, mag_err FROM relphot.tie "
            "WHERE mn_run_id = %s AND obj_id = %s ORDER BY night_id",
            (mn_run_id, obj_id),
        )
        tie_rows = cur.fetchall()
    else:
        mode = "night-normalised"
        cur.execute(
            "SELECT night_id FROM relphot.lightcurve WHERE obj_id = %s ORDER BY night_id",
            (obj_id,),
        )
        tie_rows = [(r[0], None, None) for r in cur.fetchall()]

    zp = zp_source = None
    if mode == "tied-mag":
        cur.execute(
            "SELECT n.zp, n.zp_source FROM relphot.tie t "
            "JOIN relphot.night n ON n.night_id = t.night_id "
            "JOIN relphot.mn_run mr ON mr.mn_run_id = t.mn_run_id "
            "WHERE t.mn_run_id = %s AND t.obj_id = %s "
            "ORDER BY (n.label = mr.anchor) DESC, n.night_date LIMIT 1",
            (mn_run_id, obj_id),
        )
        zp, zp_source = cur.fetchone()
    sources: set[str] = set()
    out: dict = {
        "mode": mode, "bjd_tdb": [], "value": [], "value_err": [], "tie_err": [],
        "night_id": [], "night_label": [], "file_name": [], "app_mag": [],
        "zp": zp, "zp_source": zp_source,
    }
    for night_id, tie_mag, tie_mag_err in tie_rows:
        if mode == "tied-mag" and tie_mag is None:
            continue  # a night the run could not calibrate
        cur.execute(
            "SELECT frame_index, bjd_tdb, flux, flux_err FROM relphot.lightcurve "
            "WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        lc_row = cur.fetchone()
        if lc_row is None:
            continue
        frame_index, night_bjd, night_flux, night_flux_err = lc_row

        cur.execute(
            "SELECT n.label, n.zp_source, sn.mag + n.zp FROM relphot.night n "
            "LEFT JOIN relphot.star_night sn ON sn.night_id = n.night_id AND sn.obj_id = %s "
            "WHERE n.night_id = %s",
            (obj_id, night_id),
        )
        night_label, night_zp_source, night_app_mag = cur.fetchone()
        sources.add(night_zp_source)

        cur.execute(
            "SELECT frame_index, file_name FROM relphot.frame "
            "WHERE night_id = %s AND frame_index = ANY(%s)",
            (night_id, list(frame_index)),
        )
        fname_map = {r[0]: r[1] for r in cur.fetchall()}

        flux_arr = np.asarray(night_flux, dtype=float)
        flux_err_arr = np.asarray(night_flux_err, dtype=float)
        median_flux = float(np.nanmedian(flux_arr))

        if mode == "tied-mag":
            point_value = tie_mag - 2.5 * np.log10(flux_arr / median_flux)
            point_err = 1.0857 * flux_err_arr / flux_arr
        else:
            point_value = flux_arr / median_flux
            point_err = flux_err_arr / median_flux

        n = len(frame_index)
        out["bjd_tdb"].extend(float(x) for x in night_bjd)
        out["value"].extend(float(x) for x in point_value)
        out["value_err"].extend(float(x) for x in point_err)
        out["tie_err"].extend([None if tie_mag_err is None else float(tie_mag_err)] * n)
        out["night_id"].extend([night_id] * n)
        out["night_label"].extend([night_label] * n)
        out["file_name"].extend(fname_map.get(i) for i in frame_index)
        out["app_mag"].extend([None if night_app_mag is None else round(night_app_mag, 4)] * n)
    if mode != "tied-mag" and sources:
        out["zp_source"] = sources.pop() if len(sources) == 1 else "mixed"
    return out


@app.get("/api/object/{obj_id}/lc/combined")
def object_lc_combined(obj_id: int):
    with get_ro_conn() as conn, conn.cursor() as cur:
        data = _combined_lc(cur, obj_id)
    return _json({k: v for k, v in data.items() if k != "tie_err"})


# --------------------------------------------------------------------------
# /api/night, /api/object/{obj_id}/night/{night_id} -- reference/comparison members
# --------------------------------------------------------------------------


def _tile_frames(cur: psycopg.Cursor, night_id: int) -> list[dict]:
    """Fetch frame info for a night."""
    cur.execute(
        "SELECT frame_index, bjd_tdb, kept, file_name, airmass "
        "FROM relphot.frame WHERE night_id = %s ORDER BY frame_index",
        (night_id,),
    )
    return _rows_to_dicts(cur, cur.fetchall())


def _reference_payload(
    cur: psycopg.Cursor, night_id: int, tile: int, aperture: int
) -> dict | None:
    """Build reference light-curve payload for a tile."""
    if not column_exists(cur.connection, "tile_lc", "night_id"):
        return None
    cur.execute(
        "SELECT ref_flux, ref_flux_err, ens_flux, ens_flux_err, n_ensemble, n_comp "
        "FROM relphot.tile_lc WHERE night_id = %s AND tile = %s AND aperture = %s",
        (night_id, tile, aperture),
    )
    row = cur.fetchone()
    if not row:
        return None
    ref_flux, ref_flux_err, ens_flux, ens_flux_err, _, n_comp = row
    # Count reference members
    cur.execute(
        "SELECT COUNT(*) FROM relphot.reference_member WHERE night_id = %s AND tile = %s",
        (night_id, tile),
    )
    (n_ref,) = cur.fetchone()
    # Get night info
    cur.execute("SELECT zp, zp_source FROM relphot.night WHERE night_id = %s", (night_id,))
    (zp, zp_source) = cur.fetchone()
    # Get tile info
    cur.execute(
        "SELECT x_min, x_max, y_min, y_max FROM relphot.night_tile "
        "WHERE night_id = %s AND tile = %s",
        (night_id, tile),
    )
    tile_info_row = cur.fetchone()
    tile_info = (
        {
            "x_min": tile_info_row[0],
            "x_max": tile_info_row[1],
            "y_min": tile_info_row[2],
            "y_max": tile_info_row[3],
        }
        if tile_info_row
        else {}
    )
    # Get apertures stored for this tile
    cur.execute(
        "SELECT array_agg(DISTINCT aperture ORDER BY aperture) FROM relphot.tile_lc "
        "WHERE night_id = %s AND tile = %s",
        (night_id, tile),
    )
    (apertures,) = cur.fetchone()
    apertures = apertures or []

    frames = _tile_frames(cur, night_id)
    return {
        "night_id": night_id,
        "tile": tile,
        "aperture": aperture,
        "apertures": apertures,
        "tile_info": tile_info,
        "frames": frames,
        "ref_flux": ref_flux,
        "ref_flux_err": ref_flux_err,
        "ens_flux": ens_flux,
        "ens_flux_err": ens_flux_err,
        "n_ref": n_ref,
        "n_comp": n_comp,
        "zp": zp,
        "zp_source": zp_source,
    }


def _comparison_payload(
    cur: psycopg.Cursor,
    night_id: int,
    tile: int,
    aperture: int,
    limit: int = 200,
    order: str = "mag",
    target_obj_id: int | None = None,
) -> dict | None:
    """Build comparison light-curve payload for a tile."""
    if not column_exists(cur.connection, "tile_lc", "night_id"):
        return None

    # Get tile_lc info
    cur.execute(
        "SELECT ens_flux, ens_flux_err, n_comp FROM relphot.tile_lc "
        "WHERE night_id = %s AND tile = %s AND aperture = %s",
        (night_id, tile, aperture),
    )
    row = cur.fetchone()
    if not row:
        return None
    ens_flux, ens_flux_err, _ = row

    # Fetch all comparison members for this (tile, aperture)
    cur.execute(
        "SELECT cm.star_id, cm.obj_id, cm.mag, cm.weight, cm.n_clipped, "
        "cm.clipped_frames, cm.norm_flux, o.name "
        "FROM relphot.comparison_member cm "
        "LEFT JOIN relphot.object o ON o.obj_id = cm.obj_id "
        "WHERE cm.night_id = %s AND cm.tile = %s AND cm.aperture = %s "
        "ORDER BY cm.star_id",
        (night_id, tile, aperture),
    )
    member_rows = cur.fetchall()

    if not member_rows:
        return {
            "tile": tile,
            "aperture": aperture,
            "n_members": 0,
            "n_shown": 0,
            "order": order,
            "limit": limit,
            "frames": _tile_frames(cur, night_id),
            "ens_flux": ens_flux,
            "ens_flux_err": ens_flux_err,
            "envelope": None,
            "members": [],
        }

    # Get night info
    cur.execute("SELECT zp FROM relphot.night WHERE night_id = %s", (night_id,))
    (zp,) = cur.fetchone()

    # Get target info if applicable
    target_info = None
    target_star_id = None
    if target_obj_id is not None:
        cur.execute(
            "SELECT sn.star_id FROM relphot.star_night sn "
            "WHERE sn.obj_id = %s AND sn.night_id = %s AND sn.tile = %s",
            (target_obj_id, night_id, tile),
        )
        target_row = cur.fetchone()
        if target_row:
            (target_star_id,) = target_row
            # Get target's lc info (sparse: frame_index + flux_raw)
            cur.execute(
                "SELECT lc.frame_index, lc.flux_raw FROM relphot.lightcurve lc "
                "WHERE lc.obj_id = %s AND lc.night_id = %s",
                (target_obj_id, night_id),
            )
            lc_row = cur.fetchone()
            if lc_row:
                frame_indices, flux_raw_sparse = lc_row
                # Scatter sparse flux into full-length array
                if ens_flux and flux_raw_sparse and frame_indices:
                    n_frames = len(ens_flux)
                    flux_raw_full = np.full(n_frames, np.nan, dtype=np.float32)
                    frame_idx_arr = np.asarray(frame_indices, dtype=np.int32)
                    flux_raw_arr = np.asarray(flux_raw_sparse, dtype=np.float32)
                    flux_raw_full[frame_idx_arr] = flux_raw_arr
                    ens_flux_arr = np.asarray(ens_flux, dtype=np.float32)
                    prod = flux_raw_full * ens_flux_arr
                    finite = np.isfinite(prod)
                    if np.any(finite):
                        med = np.nanmedian(prod[finite])
                        norm_flux_target = prod / med
                        flux_raw_finite = flux_raw_full[np.isfinite(flux_raw_full)]
                        if len(flux_raw_finite) > 0:
                            resid_flux = flux_raw_full / np.nanmedian(flux_raw_finite)
                            # Get is_member and weight from comparison_member
                            cur.execute(
                                (
                                    "SELECT EXISTS(SELECT 1 FROM "
                                    "relphot.comparison_member WHERE night_id=%s AND tile=%s "
                                    "AND aperture=%s AND star_id=%s), COALESCE((SELECT weight "
                                    "FROM relphot.comparison_member WHERE night_id=%s AND "
                                    "tile=%s AND aperture=%s AND star_id=%s), NULL) FROM "
                                    "(SELECT 1) dummy"
                                ),
                                (
                                    night_id, tile, aperture, target_star_id,
                                    night_id, tile, aperture, target_star_id,
                                ),
                            )
                            member_row = cur.fetchone()
                            is_member = member_row[0] if member_row else False
                            weight = member_row[1] if member_row else None
                            target_info = {
                                "obj_id": target_obj_id,
                                "star_id": target_star_id,
                                "is_member": is_member,
                                "weight": weight,
                                "frame_index": list(range(n_frames)),
                                "norm_flux": [
                                    float(v) if np.isfinite(v) else None
                                    for v in norm_flux_target
                                ],
                                "resid_flux": [
                                    float(v) if np.isfinite(v) else None
                                    for v in resid_flux
                                ],
                            }

    # Parse member data
    members_data = []
    norm_flux_all = []
    for star_id, obj_id, mag, weight, n_clipped, clipped_frames, norm_flux, name in member_rows:
        norm_flux_arr = np.asarray(norm_flux, dtype=np.float32)
        norm_flux_all.append(norm_flux_arr)
        members_data.append(
            {
                "star_id": star_id,
                "obj_id": obj_id,
                "name": name,
                "mag": mag,
                "mag_app": mag + zp if mag is not None else None,
                "weight": weight,
                "n_clipped": n_clipped,
                "clipped_frames": clipped_frames,
                "norm_flux": norm_flux_arr,
            }
        )

    # Compute RMS and envelope
    if norm_flux_all:
        norm_flux_2d = np.array(norm_flux_all)
        ens_flux_arr = np.asarray(ens_flux, dtype=np.float32) if ens_flux else None
        if ens_flux_arr is not None:
            rms_vals = member_rms(norm_flux_2d, ens_flux_arr)
            for i, m in enumerate(members_data):
                m["rms"] = float(rms_vals[i]) if np.isfinite(rms_vals[i]) else None
        env = envelope(norm_flux_2d)
    else:
        env = None

    # Select members to show
    n_members = len(members_data)
    must_include = (
        {i for i, m in enumerate(members_data) if m["star_id"] == target_star_id}
        if target_star_id
        else set()
    )

    if order == "mag":
        mag_arr = np.array(
            [m["mag"] if m["mag"] is not None else np.nan for m in members_data]
        )
        selected_indices = select_members(
            order, limit, mag=mag_arr, must_include=must_include
        )
    elif order == "weight":
        weight_arr = np.array(
            [m["weight"] if m["weight"] is not None else 0.0 for m in members_data]
        )
        selected_indices = select_members(
            order, limit, weight=weight_arr, must_include=must_include
        )
    elif order == "rms":
        rms_arr = np.array([m.get("rms", np.nan) or np.nan for m in members_data])
        selected_indices = select_members(
            order, limit, rms=rms_arr, must_include=must_include
        )
    else:
        selected_indices = select_members(order, limit, must_include=must_include)

    # Build member list with shown flag
    shown_set = {int(i) for i in selected_indices}
    for i, m in enumerate(members_data):
        m["shown"] = i in shown_set
        if m["shown"]:
            m["norm_flux"] = [
                round(float(v), 5) if np.isfinite(v) else None for v in m["norm_flux"]
            ]
        else:
            # Don't include flux arrays for non-shown members
            m.pop("norm_flux", None)
            m.pop("clipped_frames", None)

    frames = _tile_frames(cur, night_id)
    return {
        "tile": tile,
        "aperture": aperture,
        "n_members": n_members,
        "n_shown": len(shown_set),
        "order": order,
        "limit": limit,
        "frames": frames,
        "ens_flux": ens_flux,
        "ens_flux_err": ens_flux_err,
        "envelope": env,
        "members": members_data,
        "target": target_info,
    }


def _members_schema() -> bool:
    """Whether the schema-v11 member tables exist (cached after the first check)."""
    with get_ro_conn() as conn:
        return column_exists(conn, "tile_lc", "night_id")


@app.get("/api/nights")
def list_nights():
    """List nights with basic info and member status."""
    if not _members_schema():
        return _json({
            "nights": [],
            "note": "members not available (schema v11 / not loaded)"
        })

    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT n.night_id, n.label, n.telescope, n.night_date, n.n_frames, "
            "(SELECT count(*) FROM relphot.frame f "
            " WHERE f.night_id = n.night_id AND f.kept) AS n_kept, "
            "n.zp, n.zp_source, "
            "(SELECT count(*) FROM relphot.night_tile nt "
            " WHERE nt.night_id = n.night_id) AS n_tiles, "
            "EXISTS (SELECT 1 FROM relphot.night_tile nt "
            " WHERE nt.night_id = n.night_id) AS has_members "
            "FROM relphot.night n "
            "ORDER BY n.night_date DESC"
        )
        nights = _rows_to_dicts(cur, cur.fetchall())
    return _json({"nights": nights})


@app.get("/api/night/{night_id}/tiles")
def night_tiles(night_id: int):
    """List tiles of a night with their metadata."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Check night exists
        cur.execute("SELECT night_id FROM relphot.night WHERE night_id = %s", (night_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"night {night_id} not found")

        # Get night info
        cur.execute(
            "SELECT label, telescope, night_date, n_frames FROM relphot.night "
            "WHERE night_id = %s",
            (night_id,),
        )
        night_row = cur.fetchone()
        night_info = {
            "night_id": night_id,
            "label": night_row[0],
            "telescope": night_row[1],
            "night_date": night_row[2],
            "n_frames": night_row[3],
        } if night_row else {}

        # Get tiles
        cur.execute(
            "SELECT tile, x_min, x_max, y_min, y_max, n_core, n_extended, "
            "n_ref_stars, ref_aperture, best_apertures "
            "FROM relphot.night_tile WHERE night_id = %s ORDER BY tile",
            (night_id,),
        )
        tile_rows = cur.fetchall()

        tiles = []
        for (
            tile,
            x_min,
            x_max,
            y_min,
            y_max,
            n_core,
            n_extended,
            n_ref_stars,
            ref_aperture,
            best_apertures,
        ) in tile_rows:
            # Count stored comparison members for this tile
            cur.execute(
                (
                    "SELECT COUNT(DISTINCT aperture) FROM relphot.tile_lc "
                    "WHERE night_id = %s AND tile = %s"
                ),
                (night_id, tile),
            )
            (n_comp,) = cur.fetchone()
            tiles.append({
                "tile": tile,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "n_core": n_core,
                "n_extended": n_extended,
                "n_ref_stars": n_ref_stars,
                "ref_aperture": ref_aperture,
                "best_apertures": best_apertures or [],
                "n_comp": n_comp or 0,
            })

    return _json({"night": night_info, "tiles": tiles})


@app.get("/api/night/{night_id}/tiles/references")
def night_tiles_references(night_id: int, aperture: int = Query(default=None)):
    """Get reference light curves for all tiles of a night at a given aperture."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Check night exists
        cur.execute("SELECT night_id FROM relphot.night WHERE night_id = %s", (night_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"night {night_id} not found")

        frames = _tile_frames(cur, night_id)

        # Get tiles and their reference fluxes at the given aperture
        cur.execute(
            "SELECT tl.tile, tl.aperture, tl.ref_flux "
            "FROM relphot.tile_lc tl "
            "WHERE tl.night_id = %s AND (aperture = %s OR %s IS NULL) "
            "ORDER BY tl.tile, tl.aperture",
            (night_id, aperture, aperture),
        )
        tile_rows = cur.fetchall()

        tiles_data = []
        for tile, aper, ref_flux in tile_rows:
            tiles_data.append({
                "tile": tile,
                "aperture": aper,
                "n_ref": len([f for f in ref_flux if f is not None]),
                "ref_flux": ref_flux,
            })

    return _json({
        "frames": frames,
        "aperture": aperture,
        "tiles": tiles_data,
    })


@app.get("/api/night/{night_id}/tile/{tile}/reference")
def night_tile_reference(night_id: int, tile: int, aperture: int = Query(default=None)):
    """Get reference light curve for a tile."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Check night and tile exist
        cur.execute("SELECT night_id FROM relphot.night WHERE night_id = %s", (night_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"night {night_id} not found")

        # Get best_apertures for this tile, use default aperture if needed
        cur.execute(
            (
                "SELECT ref_aperture, best_apertures FROM relphot.night_tile "
                "WHERE night_id = %s AND tile = %s"
            ),
            (night_id, tile),
        )
        nt_row = cur.fetchone()
        if not nt_row:
            raise HTTPException(
                status_code=404,
                detail=f"tile {tile} not found in night {night_id}",
            )

        ref_aperture, best_apertures = nt_row
        if aperture is None:
            # Use first of best_apertures, or ref_aperture if empty
            aperture = (best_apertures[0] if best_apertures else ref_aperture) or 0

        payload = _reference_payload(cur, night_id, tile, aperture)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"reference not found for tile {tile} aperture "
                    f"{aperture}"
                ),
            )

    return _json(payload)


@app.get("/api/night/{night_id}/tile/{tile}/reference/members")
def night_tile_reference_members(night_id: int, tile: int):
    """Get reference members for a tile."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Get reference aperture for this tile
        cur.execute(
            "SELECT ref_aperture FROM relphot.night_tile WHERE night_id = %s AND tile = %s",
            (night_id, tile),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"tile {tile} not found")

        (ref_aperture,) = row
        if ref_aperture is None:
            ref_aperture = 0

        # Get night's zero point
        cur.execute("SELECT zp FROM relphot.night WHERE night_id = %s", (night_id,))
        (zp,) = cur.fetchone()

        # Get reference members ordered by weight desc
        cur.execute(
            "SELECT rm.star_id, rm.obj_id, o.name, rm.ra, rm.dec, rm.mag, rm.weight, rm.in_core "
            "FROM relphot.reference_member rm "
            "LEFT JOIN relphot.object o ON o.obj_id = rm.obj_id "
            "WHERE rm.night_id = %s AND rm.tile = %s "
            "ORDER BY rm.weight DESC, rm.star_id",
            (night_id, tile),
        )
        rows = cur.fetchall()

        members = []
        for star_id, obj_id, name, ra, dec, mag, weight, in_core in rows:
            members.append({
                "star_id": star_id,
                "obj_id": obj_id,
                "name": name,
                "ra": ra,
                "dec": dec,
                "mag": mag,
                "mag_app": mag + zp if mag is not None else None,
                "weight": weight,
                "in_core": in_core,
            })

    return _json({"members": members})


@app.get("/api/night/{night_id}/tile/{tile}/comparison")
def night_tile_comparison(
    night_id: int,
    tile: int,
    aperture: int = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    order: str = Query(default="mag"),
):
    """Get comparison members for a tile."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Validate parameters
        if order not in ("mag", "weight", "rms"):
            order = "mag"

        # Check tile exists
        cur.execute(
            "SELECT best_apertures FROM relphot.night_tile WHERE night_id = %s AND tile = %s",
            (night_id, tile),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"tile {tile} not found")

        best_apertures, = row
        if aperture is None:
            aperture = best_apertures[0] if best_apertures else 0

        payload = _comparison_payload(cur, night_id, tile, aperture, limit, order)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"comparison not found for tile {tile} aperture "
                    f"{aperture}"
                ),
            )

    return _json(payload)


@app.get("/api/object/{obj_id}/night/{night_id}/reference")
def object_night_reference(obj_id: int, night_id: int):
    """Get reference light curve for an object on a given night."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Get object's star_night for this night
        cur.execute(
            "SELECT sn.tile, sn.best_aperture FROM relphot.star_night sn "
            "WHERE sn.obj_id = %s AND sn.night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"object {obj_id} not in night {night_id}")

        tile, best_aperture = row

        payload = _reference_payload(cur, night_id, tile, best_aperture)
        if payload is None:
            raise HTTPException(status_code=404, detail="reference not found")

    return _json(payload)


@app.get("/api/object/{obj_id}/night/{night_id}/comparison")
def object_night_comparison(
    obj_id: int,
    night_id: int,
    limit: int = Query(default=200, ge=1, le=1000),
    order: str = Query(default="mag"),
):
    """Get comparison members for an object on a given night."""
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        # Validate parameters
        if order not in ("mag", "weight", "rms"):
            order = "mag"

        # Get object's star_night for this night
        cur.execute(
            "SELECT sn.tile, sn.best_aperture FROM relphot.star_night sn "
            "WHERE sn.obj_id = %s AND sn.night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"object {obj_id} not in night {night_id}")

        tile, best_aperture = row

        payload = _comparison_payload(
            cur, night_id, tile, best_aperture, limit, order,
            target_obj_id=obj_id,
        )
        if payload is None:
            raise HTTPException(status_code=404, detail="comparison not found")

    return _json(payload)


@app.get("/api/object/{obj_id}/night/{night_id}/lc_ratios")
def object_night_lc_ratios(
    obj_id: int,
    night_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Individual target / comparison-i light curves behind an object's per-night curve.

    Each ratio is ``lc * ens / c_i`` (see :func:`relphot.web.tile_lc.individual_ratio_curves`)
    at the target's best aperture, on the scale of the plotted curve and aligned to the
    target's own ``frame_index`` list. The target itself is left out when it is a member.
    ``ensemble`` is ``"median"`` when the tile's ensemble was the median (the per-epoch median
    of the returned ratios is then the plotted curve) and ``"weighted"`` otherwise.
    """
    if not _members_schema():
        msg = "members not available (schema v11 / not loaded)"
        raise HTTPException(status_code=404, detail=msg)

    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT sn.tile, sn.best_aperture, sn.star_id, n.zp FROM relphot.star_night sn "
            "JOIN relphot.night n ON n.night_id = sn.night_id "
            "WHERE sn.obj_id = %s AND sn.night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"object {obj_id} not in night {night_id}")
        tile, aperture, target_star_id, zp = row

        cur.execute(
            "SELECT frame_index, flux FROM relphot.lightcurve WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        lc_row = cur.fetchone()
        cur.execute(
            "SELECT ens_flux FROM relphot.tile_lc "
            "WHERE night_id = %s AND tile = %s AND aperture = %s",
            (night_id, tile, aperture),
        )
        ens_row = cur.fetchone()
        if not lc_row or not ens_row or not ens_row[0]:
            raise HTTPException(status_code=404, detail="comparison ratios not available")
        frame_index, flux = lc_row
        ens_flux = ens_row[0]

        cur.execute(
            "SELECT cm.star_id, cm.obj_id, o.name, cm.mag, cm.weight, cm.n_clipped, cm.norm_flux "
            "FROM relphot.comparison_member cm "
            "LEFT JOIN relphot.object o ON o.obj_id = cm.obj_id "
            "WHERE cm.night_id = %s AND cm.tile = %s AND cm.aperture = %s ORDER BY cm.star_id",
            (night_id, tile, aperture),
        )
        member_rows = cur.fetchall()

    frame_idx = np.asarray(frame_index, dtype=np.int64)
    n_frames = len(ens_flux)
    keep = (frame_idx >= 0) & (frame_idx < n_frames)
    frame_idx = frame_idx[keep]
    lc_sparse = np.asarray(flux, dtype=np.float64)[keep]
    lc_full = np.full(n_frames, np.nan)
    lc_full[frame_idx] = lc_sparse

    weights = np.array([r[4] if r[4] is not None else np.nan for r in member_rows])
    n_clipped = np.array([r[5] for r in member_rows])
    ensemble = "median" if is_median_ensemble(weights, n_clipped) else "weighted"
    target_is_member = any(r[0] == target_star_id for r in member_rows)
    others = [r for r in member_rows if r[0] != target_star_id]

    members: list[dict] = []
    if others:
        norm_flux = np.array([r[6] for r in others], dtype=np.float64)
        ratios = individual_ratio_curves(lc_full, np.asarray(ens_flux, dtype=np.float64), norm_flux)
        at_target = ratios[:, frame_idx]
        usable = np.count_nonzero(np.isfinite(at_target), axis=1) >= 3
        mag = np.array([r[3] if r[3] is not None else np.nan for r in others])
        idx = np.nonzero(usable)[0]
        if idx.size:
            # select_members returns floats when it has no must_include members
            picked = np.asarray(select_members("mag", limit, mag=mag[idx]), dtype=np.int64)
            chosen = idx[picked]
            for i in chosen:
                star_id, o_id, name, m_mag, _w, _nc, _nf = others[i]
                members.append(
                    {
                        "star_id": star_id,
                        "obj_id": o_id,
                        "name": name,
                        "mag_app": None if m_mag is None or zp is None else m_mag + zp,
                        "ratio": [
                            round(float(v), 5) if np.isfinite(v) else None for v in at_target[i]
                        ],
                    }
                )

    return _json(
        {
            "night_id": night_id,
            "tile": tile,
            "aperture": aperture,
            "ensemble": ensemble,
            "target_is_member": target_is_member,
            "n_members": len(others),
            "n_shown": len(members),
            "limit": limit,
            "frame_index": [int(f) for f in frame_idx],
            "members": members,
        }
    )


# --------------------------------------------------------------------------
# /api/object/{obj_id}/phase -- the phase diagram of all nights
# --------------------------------------------------------------------------

#: Periods beyond this (days) are fitted without a free offset per night on tied magnitudes
#: (mirrors ``DbSettings.long_period_days``; the web cannot import :mod:`relphot.config`'s
#: database settings without ``relphot.db``, so the default is repeated here).
_LONG_PERIOD_DAYS = 1.0
_PHASE_BINS = 20


@app.get("/api/object/{obj_id}/phase")
def object_phase(obj_id: int, period: float | None = Query(default=None, gt=0)):
    """All nights of one object as one phase-diagram payload.

    Points are tie-calibrated magnitudes when a multi-night run ties >= 2 nights
    (``tied = true``), else per-night-normalised flux (``tied = false``: the nights are NOT
    tied, and the diagram is labelled untied). ``value_err`` is the photometric error (already
    inflated for blended stars at the light-curve stage) plus, tied, the night's tie error in
    quadrature. ``period_candidates`` lists the object's PERIOD, its latest LS estimate, its
    latest user-guided one and the literature period; ``aliases`` the latest estimate's alias
    candidates. With ``period`` (default: the first candidate) the payload carries the
    2-harmonic Fourier ``model`` at that period (per-night offsets only when untied or when the
    period is not longer than a night: a longer one lives in the night-to-night changes) and the
    phase coverage of the points.
    """
    with get_ro_conn() as conn, conn.cursor() as cur:
        data = _combined_lc(cur, obj_id)
        cur.execute(
            "SELECT period, period_err, period_source, known_period FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        obj_period, obj_period_err, obj_period_source, known_period = cur.fetchone()
        cur.execute(
            "SELECT est_id, method, period, period_err, n_nights, computed_at, verify_status, "
            "phase_coverage, n_cycles, alias_periods, alias_powers, guess "
            "FROM relphot.period_estimate WHERE obj_id = %s AND period IS NOT NULL "
            "ORDER BY last_night DESC NULLS LAST, n_nights DESC, computed_at DESC",
            (obj_id,),
        )
        estimates = _rows_to_dicts(cur, cur.fetchall())

    tied = data["mode"] == "tied-mag"
    candidates: list[dict] = []
    if obj_period is not None:
        candidates.append({
            "key": "period", "label": f"PERIOD ({obj_period_source or 'unknown'})",
            "period": obj_period, "period_err": obj_period_err,
        })
    latest_ls = next((e for e in estimates if e["method"] == "LS"), None)
    latest_guided = max(
        (e for e in estimates if e["method"] == "LS-guided"),
        key=lambda e: e["computed_at"], default=None,
    )
    if latest_ls is not None:
        candidates.append({
            "key": "estimate", "label": f"latest estimate ({latest_ls['n_nights']} nights)",
            "period": latest_ls["period"], "period_err": latest_ls["period_err"],
        })
    if latest_guided is not None:
        guess = latest_guided['guess']
        n_nights = latest_guided['n_nights']
        candidates.append({
            "key": "guided", "label": f"latest guided (guess {guess:g} d, {n_nights} night(s))",
            "period": latest_guided["period"], "period_err": latest_guided["period_err"],
        })
    if known_period is not None:
        candidates.append({
            "key": "literature", "label": "literature", "period": known_period,
            "period_err": None,
        })
    alias_src = next((e for e in estimates if e["method"] == "LS" and e["alias_periods"]), None)
    aliases = []
    if alias_src is not None:
        powers = alias_src["alias_powers"] or []
        aliases = [
            {"period": p, "power": powers[i] if i < len(powers) else None}
            for i, p in enumerate(alias_src["alias_periods"])
        ]

    chosen = period if period is not None else (candidates[0]["period"] if candidates else None)
    t = np.asarray(data["bjd_tdb"], dtype=float)
    value_err = np.asarray(data["value_err"], dtype=float)
    if tied:
        tie_err = np.asarray(
            [0.0 if e is None else e for e in data["tie_err"]], dtype=float
        )
        value_err = np.hypot(value_err, tie_err)
    model = None
    coverage = n_cycles = None
    if chosen is not None and t.size:
        coverage, n_cycles = phase_coverage(t, chosen, _PHASE_BINS)
        model = fourier_model(
            t, np.asarray(data["value"], dtype=float), value_err,
            np.asarray(data["night_id"]), chosen,
            per_night_offsets=(not tied) or chosen <= _LONG_PERIOD_DAYS,
            brighter_is_lower=tied,
        )
    return _json({
        "mode": data["mode"], "tied": tied,
        "label": "tie-calibrated magnitudes" if tied else "untied: per-night normalised flux",
        "zp": data["zp"], "zp_source": data["zp_source"], "app_mag": data["app_mag"],
        "bjd_tdb": data["bjd_tdb"], "value": data["value"], "value_err": value_err.tolist(),
        "night_id": data["night_id"], "night_label": data["night_label"],
        "file_name": data["file_name"],
        "t_first": float(t.min()) if t.size else None,
        "baseline_days": float(t.max() - t.min()) if t.size else None,
        "period_candidates": candidates, "aliases": aliases,
        "period": chosen, "phase_coverage": coverage, "n_cycles": n_cycles,
        "model": model, "long_period_days": _LONG_PERIOD_DAYS,
    })


@app.get("/api/object/{obj_id}/periodogram")
def object_periodogram(obj_id: int, scope: str, method: str):
    with get_ro_conn() as conn:
        cols = _select_columns(
            conn, "periodogram", ["fmin", "df", "n", "power", "peak_period", "peak_power", "fap"],
            ["extra"],
        )
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(cols)} FROM relphot.periodogram "
                "WHERE obj_id = %s AND scope = %s AND method = %s",
                (obj_id, scope, method),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no periodogram for this object/scope/method")
    return _json(dict(zip(cols, row, strict=True)))


# --------------------------------------------------------------------------
# PATCH /api/object/{obj_id} -- manual edits (relphot_web role)
# --------------------------------------------------------------------------

_STATUS_VALUES = {"UNCONFIRMED", "CONFIRMED", "REJECTED"}
_SOURCE_VALUES = {"auto", "manual"}


class ObjectPatchBody(BaseModel):
    """Manual edits of one object (shorthand for per-night verdicts).

    ``is_exop`` / ``is_var`` is a shorthand that upserts CONFIRMED/REJECTED verdicts
    on all nights the object currently has data for; ``exop_source`` / ``var_source``
    = ``'auto'`` resets those verdicts to NULL on all nights and deletes empty rows.
    The two flags are independent. The ``class`` / ``class_source`` fields work as
    shorthand: ``class`` sets the flags it names to true (``'UNC'`` clears both) and
    leaves the other flag alone; ``class_source`` = ``'auto'`` resets both flags.
    Verdicts are then re-evaluated per-night via ``refresh_flags``, and ``class`` /
    ``class_source`` on the object row are re-derived from the result (never set directly).
    """

    model_config = ConfigDict(populate_by_name=True)

    class_: str | None = Field(default=None, alias="class")
    is_exop: bool | None = None
    is_var: bool | None = None
    exop_source: str | None = None
    var_source: str | None = None
    status: str | None = None
    notes: str | None = None
    period: float | None = None
    class_source: str | None = None
    period_source: str | None = None


def _flag_edits(provided: dict) -> tuple[dict[str, bool | None], dict[str, str | None]]:
    """Resolve the flag/source fields of a PATCH into ``({flag: new value}, {flag: new source})``.

    Keys are ``'exop'`` / ``'var'``; a missing key means "leave alone". Raises a 400 on an
    invalid value or on a flag set by hand and reset to auto in the same request.
    """
    flags: dict[str, bool | None] = {}
    sources: dict[str, str | None] = {}

    if "class_" in provided:
        value = provided["class_"]
        if value not in _CLASS_LABELS:
            raise HTTPException(status_code=400, detail=f"invalid class: {value!r}")
        if value == "UNC":
            flags["exop"] = flags["var"] = False
        else:
            if "EXOP" in value:
                flags["exop"] = True
            if "VAR" in value:
                flags["var"] = True
    if "class_source" in provided:
        value = provided["class_source"]
        if value not in _SOURCE_VALUES:
            raise HTTPException(status_code=400, detail=f"invalid class_source: {value!r}")
        if "class_" not in provided:
            sources["exop"] = sources["var"] = value

    for name, key in (("is_exop", "exop"), ("is_var", "var")):
        if name in provided:
            if not isinstance(provided[name], bool):
                raise HTTPException(status_code=400, detail=f"invalid {name}: must be a boolean")
            flags[key] = provided[name]
    for name, key in (("exop_source", "exop"), ("var_source", "var")):
        if name in provided:
            if provided[name] not in _SOURCE_VALUES:
                raise HTTPException(
                    status_code=400, detail=f"invalid {name}: {provided[name]!r}"
                )
            sources[key] = provided[name]

    for key in flags:
        if sources.get(key) == "auto":
            raise HTTPException(
                status_code=400,
                detail=f"cannot set is_{key} by hand and reset it to auto in one request",
            )
    return flags, sources


@app.patch("/api/object/{obj_id}")
def patch_object(obj_id: int, body: ObjectPatchBody):
    provided = body.model_dump(exclude_unset=True, by_alias=False)
    set_parts: list[str] = []
    params: dict[str, object] = {"obj_id": obj_id}

    # flag edits are a shorthand for per-night verdicts on the nights loaded now
    flags, sources = _flag_edits(provided)
    with get_rw_conn() as conn, conn.cursor() as cur:
        if flags or sources:
            for key in ("exop", "var"):
                if key in flags:
                    verdict = "CONFIRMED" if flags[key] else "REJECTED"
                    cur.execute(
                        f"INSERT INTO relphot.user_night_review (obj_id, night_id, {key}_verdict) "
                        "SELECT %(obj_id)s, sn.night_id, %(verdict)s FROM relphot.star_night sn "
                        "WHERE sn.obj_id = %(obj_id)s "
                        "ON CONFLICT (obj_id, night_id) DO UPDATE SET "
                        f"{key}_verdict = %(verdict)s, updated_at = now()",
                        {"obj_id": obj_id, "verdict": verdict},
                    )
                elif sources.get(key) == "auto":
                    cur.execute(
                        f"UPDATE relphot.user_night_review SET {key}_verdict = NULL, "
                        "updated_at = now() WHERE obj_id = %(obj_id)s",
                        {"obj_id": obj_id},
                    )
            cur.execute(
                "DELETE FROM relphot.user_night_review WHERE obj_id = %(obj_id)s "
                "AND exop_verdict IS NULL AND var_verdict IS NULL AND note IS NULL",
                {"obj_id": obj_id},
            )
            refresh_flags(
                conn, [obj_id], class_multinight_kinds=DbSettings().class_multinight_kinds
            )

        # Handle period and status updates
        if "period" in provided:
            set_parts.append("period = %(period)s")
            set_parts.append("period_source = 'manual'")
            set_parts.append("period_err = NULL")
            set_parts.append("period_n_nights = NULL")
            params["period"] = provided["period"]
        elif "period_source" in provided:
            value = provided["period_source"]
            if value not in _SOURCE_VALUES:
                raise HTTPException(status_code=400, detail=f"invalid period_source: {value!r}")
            set_parts.append("period_source = %(period_source)s")
            params["period_source"] = value

        if "status" in provided:
            value = provided["status"]
            if value not in _STATUS_VALUES:
                raise HTTPException(status_code=400, detail=f"invalid status: {value!r}")
            set_parts.append("status = %(status)s")
            params["status"] = value

        if "notes" in provided:
            set_parts.append("notes = %(notes)s")
            params["notes"] = provided["notes"]

        if not set_parts and not (flags or sources):
            raise HTTPException(status_code=400, detail="no recognised fields to update")

        if set_parts:
            set_parts.append("updated_at = now()")
            sql = (
                f"UPDATE relphot.object SET {', '.join(set_parts)} "
                "WHERE obj_id = %(obj_id)s RETURNING *"
            )
            cur.execute(sql, params)
        else:
            # Only flag edits, no direct object update; fetch the object row
            cur.execute("SELECT * FROM relphot.object WHERE obj_id = %(obj_id)s", params)

        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="object not found")
        columns = [d.name for d in cur.description]
        conn.commit()

    return _json(dict(zip(columns, row, strict=True)))


# --------------------------------------------------------------------------
# PATCH /api/detection/{det_id} -- a person's verdict on one event (relphot_web role)
# --------------------------------------------------------------------------


class DetectionPatchBody(BaseModel):
    status: str | None = None
    notes: str | None = None


@app.patch("/api/detection/{det_id}")
def patch_detection(det_id: int, body: DetectionPatchBody):
    provided = body.model_dump(exclude_unset=True)
    set_parts: list[str] = []
    params: dict[str, object] = {"det_id": det_id}
    if "status" in provided:
        if provided["status"] not in _STATUS_VALUES:
            raise HTTPException(status_code=400, detail=f"invalid status: {provided['status']!r}")
        set_parts.append("status = %(status)s")
        params["status"] = provided["status"]
    if "notes" in provided:
        set_parts.append("notes = %(notes)s")
        params["notes"] = provided["notes"]
    if not set_parts:
        raise HTTPException(status_code=400, detail="no recognised fields to update")

    sql = (
        f"UPDATE relphot.detection SET {', '.join(set_parts)} WHERE det_id = %(det_id)s "
        "RETURNING det_id, obj_id, night_id, mn_run_id, kind, status, notes"
    )
    with get_rw_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="detection not found")
        columns = [d.name for d in cur.description]
        # a verdict on an event changes the night's automatic evidence: re-derive the flags
        if "status" in provided:
            refresh_flags(
                conn, [row[columns.index("obj_id")]],
                class_multinight_kinds=DbSettings().class_multinight_kinds,
            )
        conn.commit()
    return _json(dict(zip(columns, row, strict=True)))


# --------------------------------------------------------------------------
# PUT /api/object/{obj_id}/night/{night_id}/review - per-night user verdicts (relphot_web write)
# --------------------------------------------------------------------------


_REVIEW_NOTE_MAX = 2000


class NightReviewBody(BaseModel):
    """Per-night user verdicts on exoplanet and variable classification.

    All-null (and note null/empty) deletes the row; otherwise INSERT ... ON CONFLICT UPDATE.
    Verdicts must be 'CONFIRMED', 'REJECTED', or null (auto). Note at most 2000 characters.
    """

    exop: str | None = None
    var: str | None = None
    note: str | None = None


@app.put("/api/object/{obj_id}/night/{night_id}/review")
def put_night_review(obj_id: int, night_id: int, body: NightReviewBody):
    """Upsert or delete a per-night review verdict.

    All null (and note null/empty) -> DELETE row, else INSERT ... ON CONFLICT DO UPDATE.
    Returns {review, night: night_state dict, object: object row with updated flags/counts}.
    """
    provided = body.model_dump(exclude_unset=True)

    for key in ("exop", "var"):
        if provided.get(key) not in (None, "CONFIRMED", "REJECTED"):
            raise HTTPException(
                status_code=400, detail=f"invalid {key}: must be 'CONFIRMED', 'REJECTED', or null"
            )
    note = provided.get("note")
    if note is not None and len(note) > _REVIEW_NOTE_MAX:
        raise HTTPException(
            status_code=400, detail=f"note must be at most {_REVIEW_NOTE_MAX} characters"
        )

    with get_rw_conn() as conn, conn.cursor() as cur:
        # Check object exists
        cur.execute("SELECT 1 FROM relphot.object WHERE obj_id = %s", (obj_id,))
        if cur.fetchone() is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="object not found")

        # Check object has data on this night
        cur.execute(
            "SELECT 1 FROM relphot.star_night WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        if cur.fetchone() is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="object has no data on this night")

        # Determine action: all null (and note null/empty) -> delete, else upsert
        exop_v = provided.get("exop")
        var_v = provided.get("var")
        note_v = provided.get("note") or ""

        if exop_v is None and var_v is None and not note_v:
            # Delete
            cur.execute(
                "DELETE FROM relphot.user_night_review WHERE obj_id = %s AND night_id = %s",
                (obj_id, night_id),
            )
            review = None
        else:
            # Upsert
            # relphot_web may INSERT only (obj_id, night_id, verdicts, note): updated_at defaults
            cur.execute(
                "INSERT INTO relphot.user_night_review "
                "(obj_id, night_id, exop_verdict, var_verdict, note) "
                "VALUES (%(obj_id)s, %(night_id)s, %(exop_v)s, %(var_v)s, NULLIF(%(note_v)s, '')) "
                "ON CONFLICT (obj_id, night_id) DO UPDATE SET "
                "exop_verdict = EXCLUDED.exop_verdict, var_verdict = EXCLUDED.var_verdict, "
                "note = EXCLUDED.note, updated_at = now() "
                "RETURNING exop_verdict, var_verdict, note, updated_at",
                {
                    "obj_id": obj_id, "night_id": night_id, "exop_v": exop_v, "var_v": var_v,
                    "note_v": note_v,
                },
            )
            row = cur.fetchone()
            if row:
                review = {
                    "exop": row[0],
                    "var": row[1],
                    "note": row[2],
                    "updated_at": row[3].isoformat() if row[3] else None,
                }
            else:
                review = None

        # Compute night_state for this night
        cur.execute(
            "SELECT d.kind, d.status, d.origin, d.auto_status FROM relphot.detection d "
            "WHERE d.obj_id = %s AND d.night_id = %s",
            (obj_id, night_id),
        )
        night_dets = [
            {"kind": k, "status": s, "origin": o, "auto_status": a}
            for k, s, o, a in cur.fetchall()
        ]
        ns = night_state(night_dets, exop_v, var_v)

        refresh_flags(conn, [obj_id], class_multinight_kinds=DbSettings().class_multinight_kinds)

        # Fetch updated object row
        cur.execute(
            "SELECT obj_id, is_exop, is_var, exop_source, var_source, class, class_source, "
            "n_review_pending, n_nights_reviewed FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        obj_row = cur.fetchone()
        if obj_row:
            obj_data = {
                "is_exop": obj_row[1],
                "is_var": obj_row[2],
                "exop_source": obj_row[3],
                "var_source": obj_row[4],
                "class": obj_row[5],
                "class_source": obj_row[6],
                "n_review_pending": obj_row[7],
                "n_nights_reviewed": obj_row[8],
            }
        else:
            obj_data = {}

        conn.commit()

    return _json({
        "review": review,
        "night": ns,
        "object": obj_data,
    })


@app.delete("/api/object/{obj_id}/night/{night_id}/review")
def delete_night_review(obj_id: int, night_id: int):
    """Delete a per-night review (same as PUT with all null)."""
    body = NightReviewBody(exop=None, var=None, note=None)
    return put_night_review(obj_id, night_id, body)


# --------------------------------------------------------------------------
# POST /api/detections/review - one verdict (and note) on several transit events at once
# --------------------------------------------------------------------------

_BULK_VERBS = {"CONFIRMED": "CONFIRM", "REJECTED": "REJECT", "UNCONFIRMED": "UNCONFIRM"}


class DetectionsReviewBody(BaseModel):
    """A bulk verdict from the similar-events window.

    ``det_ids`` (1..100) get ``status`` and, when ``note`` is not empty, the note: appended on a
    new line of their ``notes`` (``note_mode = 'append'``) or replacing them (``'replace'``). The
    viewed event ``anchor_det_id`` may be among them. Either way a summary line of the action is
    appended to the anchor's notes (after the user note, if the anchor got one).
    """

    anchor_det_id: int
    det_ids: list[int] = Field(min_length=1, max_length=_STACK_MAX)
    status: str
    note: str | None = None
    note_mode: Literal["append", "replace"] = "append"


def _append_note_sql(param: str) -> str:
    """SQL of ``notes`` with the text of ``%(param)s`` on a new last line."""
    return (
        f"CASE WHEN notes IS NULL OR notes = '' THEN %({param})s "
        f"ELSE notes || chr(10) || %({param})s END"
    )


def _bulk_summary(
    status: str, others: list[tuple[int, str]], with_anchor: bool, note: str | None
) -> str:
    """The line recorded on the viewed event: the date, the action, who it covered (the viewed
    event itself and/or its look-alikes by name and det_id) and the user's note after an em dash."""
    who = ["viewed event"] if with_anchor else []
    if others:
        who.append(f"{len(others)} look-alike{'s' if len(others) != 1 else ''}")
    text = f"{date.today().isoformat()} {_BULK_VERBS[status]} ALL {' + '.join(who)}"
    if others:
        text += ": " + ", ".join(f"{name} (det {det_id})" for det_id, name in others)
    if note:
        text += f" \u2014 {note}"
    return text


@app.post("/api/detections/review")
def post_detections_review(body: DetectionsReviewBody):
    if body.status not in _STATUS_VALUES:
        raise HTTPException(status_code=400, detail=f"invalid status: {body.status!r}")
    note = (body.note or "").strip() or None
    if note is not None and len(note) > _REVIEW_NOTE_MAX:
        raise HTTPException(
            status_code=400, detail=f"note too long (max {_REVIEW_NOTE_MAX} characters)"
        )
    order = list(dict.fromkeys(body.det_ids))  # as sent (nearest first), without repeats
    anchor_id = body.anchor_det_id
    set_parts = ["status = %(status)s"]
    params: dict[str, object] = {"status": body.status, "ids": sorted(order)}
    if note is not None:
        set_parts.append(
            "notes = %(note)s" if body.note_mode == "replace"
            else f"notes = {_append_note_sql('note')}"
        )
        params["note"] = note

    with get_rw_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE relphot.detection SET {', '.join(set_parts)} "
            "WHERE det_id = ANY(%(ids)s) AND kind = 'transit' "
            "RETURNING det_id, obj_id, night_id, status, auto_status, notes",
            params,
        )
        updated = {row["det_id"]: row for row in _rows_to_dicts(cur, cur.fetchall())}
        missing = [j for j in order if j not in updated]
        anchor_obj = updated[anchor_id]["obj_id"] if anchor_id in updated else None
        if anchor_obj is None:
            cur.execute(
                "SELECT obj_id FROM relphot.detection WHERE det_id = %s AND kind = 'transit'",
                (anchor_id,),
            )
            found = cur.fetchone()
            if found is None:
                missing.append(anchor_id)
            else:
                anchor_obj = found[0]
        if missing:
            conn.rollback()
            raise HTTPException(
                status_code=404, detail=f"transit detections not found: {missing}"
            )
        other_ids = [j for j in order if j != anchor_id]
        names: dict[int, str] = {}
        if other_ids:
            cur.execute(
                "SELECT d.det_id, o.name FROM relphot.detection d "
                "JOIN relphot.object o ON o.obj_id = d.obj_id WHERE d.det_id = ANY(%s)",
                (other_ids,),
            )
            names = dict(cur.fetchall())
        line = _bulk_summary(
            body.status, [(j, names.get(j, "?")) for j in other_ids], anchor_id in updated, note
        )
        cur.execute(
            f"UPDATE relphot.detection SET notes = {_append_note_sql('line')} "
            "WHERE det_id = %(det_id)s RETURNING status, notes",
            {"line": line, "det_id": anchor_id},
        )
        anchor_status, anchor_notes = cur.fetchone()
        if anchor_id in updated:
            updated[anchor_id]["notes"] = anchor_notes
        # a verdict on an event changes the night's automatic evidence: re-derive the flags
        refresh_flags(
            conn, sorted({row["obj_id"] for row in updated.values()} | {anchor_obj}),
            class_multinight_kinds=DbSettings().class_multinight_kinds,
        )
        conn.commit()
    rows = [updated[j] for j in sorted(updated)]
    for row in rows:
        row["effective_status"] = _effective_status(row["status"], row["auto_status"])
    return _json(
        {
            "updated": rows,
            "anchor": {
                "det_id": anchor_id, "obj_id": anchor_obj, "status": anchor_status,
                "notes": anchor_notes,
            },
        }
    )


# --------------------------------------------------------------------------
# Repeated transit events: /api/object/{obj_id}/repeat_link, /api/repeat/predict
# --------------------------------------------------------------------------

#: ``relphot db analyze`` recomputes the families; until then a web verdict only marks them stale.
_REPEAT_NOTE = (
    "Families and their periods are recomputed at the next `relphot db analyze`; "
    "until then the affected families are marked stale."
)
_STALE_TEXT = "stale \u2014 recomputed at next analyze"
_PREDICT_DEFAULT_DAYS = 10.0


def _has_repeat_tables(conn: psycopg.Connection) -> bool:
    """Whether migration 012 (``relphot.repeat_family``) is applied."""
    return column_exists(conn, "repeat_family", "fam_id")


def _object_repeat(
    conn: psycopg.Connection, cur: psycopg.Cursor, obj_id: int
) -> tuple[list[dict], list[dict]]:
    """``(repeat_families, repeat_decisions)`` of one object for ``/api/object``.

    Every stored family is returned, whatever its alias statuses. Members the person has since
    rejected (or the coincidence check auto-rejected, unless confirmed) are dropped and the family
    is marked ``stale``, as it is when a decision was made on one of its pairs since it was
    computed: the families themselves only change at the next ``relphot db analyze``. Links are
    the stored pair scores (``decision`` is the person's current one, ``decision_applied`` the one
    the family was computed with).
    """
    if not _has_repeat_tables(conn):
        return [], []
    cur.execute(
        "SELECT fam_id, family_key, n_members, member_night_ids, involves_loose, depth, "
        "depth_err, t14_h, t14_lower_limit, ingress_frac, score, n_alias, n_allowed, accepted, "
        "computed_at FROM relphot.repeat_family WHERE obj_id = %s ORDER BY fam_id",
        (obj_id,),
    )
    families = _rows_to_dicts(cur, cur.fetchall())
    cur.execute(
        "SELECT r.night_a, na.label AS label_a, r.night_b, nb.label AS label_b, r.tc_a, r.tc_b, "
        "r.decision, r.note, r.updated_at "
        "FROM relphot.repeat_decision r "
        "JOIN relphot.night na ON na.night_id = r.night_a "
        "JOIN relphot.night nb ON nb.night_id = r.night_b "
        "WHERE r.obj_id = %s ORDER BY r.night_a, r.night_b, r.tc_a",
        (obj_id,),
    )
    decisions = _rows_to_dicts(cur, cur.fetchall())
    if not families:
        return [], decisions
    fam_ids = [f["fam_id"] for f in families]

    cur.execute(
        "SELECT m.fam_id, d.det_id, d.night_id, n.label AS night_label, n.telescope, "
        "ts.tc, ts.tc_err, ts.depth, ts.depth_err, ts.t14_h, ts.t14_err, ts.t14_lower_limit, "
        "ts.ingress_frac, ts.ingress_err, d.status, d.auto_status, d.flags, "
        "(d.night_id IN (SELECT unnest(loose_night_ids) FROM relphot.mn_run)) AS loose "
        "FROM relphot.repeat_family_member m "
        "JOIN relphot.detection d ON d.det_id = m.det_id "
        "JOIN relphot.transit_shape ts ON ts.det_id = m.det_id "
        "JOIN relphot.night n ON n.night_id = d.night_id "
        "WHERE m.fam_id = ANY(%s) ORDER BY m.fam_id, ts.tc",
        (fam_ids,),
    )
    members_by_fam: dict[int, list[dict]] = {}
    for mem in _rows_to_dicts(cur, cur.fetchall()):
        mem["effective_status"] = _effective_status(mem["status"], mem["auto_status"])
        mem["duration_display"] = _fmt_duration(mem["t14_h"], mem["t14_lower_limit"])
        members_by_fam.setdefault(mem.pop("fam_id"), []).append(mem)

    cur.execute(
        "SELECT det_a, det_b, dt_days, p_match, p_joint, chi2_joint, dof_joint, phys_ok, "
        "n_alias, diurnal, involves_loose, linked, decision FROM relphot.repeat_link "
        "WHERE obj_id = %s",
        (obj_id,),
    )
    link_of = {(r["det_a"], r["det_b"]): r for r in _rows_to_dicts(cur, cur.fetchall())}

    cur.execute(
        "SELECT fam_id, alias_k, period, period_err, tc0, tc0_err, status, veto_night_id, "
        "veto_dchi2, n_nights_tested FROM relphot.repeat_ephemeris WHERE fam_id = ANY(%s) "
        "ORDER BY fam_id, alias_k",
        (fam_ids,),
    )
    aliases_by_fam: dict[int, list[dict]] = {}
    for al in _rows_to_dicts(cur, cur.fetchall()):
        aliases_by_fam.setdefault(al.pop("fam_id"), []).append(al)
    night_label = {}
    if any(a["veto_night_id"] is not None for al in aliases_by_fam.values() for a in al):
        cur.execute("SELECT night_id, label FROM relphot.night")
        night_label = dict(cur.fetchall())

    for fam in families:
        all_members = members_by_fam.get(fam["fam_id"], [])
        members = [
            m for m in all_members
            if m["status"] != "REJECTED"
            and not (m["auto_status"] == "REJECTED" and m["status"] != "CONFIRMED")
        ]
        reasons = []
        if len(members) < len(all_members):
            reasons.append(f"{len(all_members) - len(members)} member(s) rejected since")
        links = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                first, second = sorted((a, b), key=lambda e: e["det_id"])
                stored = link_of.get((first["det_id"], second["det_id"]))
                if stored is None:
                    continue
                live = match_decision(a, b, decisions)
                if live != stored["decision"]:
                    reasons.append("a decision changed since")
                links.append({
                    **stored, "decision_applied": stored["decision"], "decision": live,
                    "night_label_a": first["night_label"], "night_label_b": second["night_label"],
                })
        fam["members"] = members
        fam["links"] = links
        fam["aliases"] = aliases_by_fam.get(fam["fam_id"], [])
        for al in fam["aliases"]:
            al["veto_night_label"] = night_label.get(al["veto_night_id"])
        fam["stale"] = bool(reasons)
        fam["stale_reason"] = (
            f"{_STALE_TEXT} ({'; '.join(sorted(set(reasons)))})" if reasons else None
        )
        fam["n_members_stored"] = len(all_members)
    return families, decisions


class RepeatLinkBody(BaseModel):
    """The person's verdict on a pair of one object's transit events: SAME (one planet) or
    DIFFERENT. The events are never merged and are kept either way; ``det_a`` / ``det_b`` are
    the pair's detection ids (either order)."""

    det_a: int
    det_b: int
    decision: str
    note: str | None = None


def _repeat_pair(cur: psycopg.Cursor, obj_id: int, det_a: int, det_b: int) -> tuple[dict, dict]:
    """The two events (night id, fitted tc, T14) of a pair, ordered ``(night, tc)``."""
    if det_a == det_b:
        raise HTTPException(status_code=400, detail="det_a and det_b must differ")
    cur.execute(
        "SELECT d.det_id, d.night_id, ts.tc, ts.t14_h FROM relphot.detection d "
        "JOIN relphot.transit_shape ts ON ts.det_id = d.det_id "
        "WHERE d.obj_id = %s AND d.kind = 'transit' AND d.det_id = ANY(%s) "
        "AND ts.tc IS NOT NULL AND ts.t14_h IS NOT NULL",
        (obj_id, [det_a, det_b]),
    )
    events = _rows_to_dicts(cur, cur.fetchall())
    if len(events) != 2:
        raise HTTPException(
            status_code=404, detail="both events must be transit events of this object with a fit"
        )
    first, second = sorted(events, key=lambda e: (e["night_id"], e["tc"]))
    return first, second


def _delete_repeat_decision(cur: psycopg.Cursor, obj_id: int, first: dict, second: dict) -> int:
    """Delete the stored decision(s) on the pair (same nights, centre times within tolerance)."""
    tol = decision_tol_days(first["t14_h"], second["t14_h"])
    cur.execute(
        "DELETE FROM relphot.repeat_decision WHERE obj_id = %s AND night_a = %s AND night_b = %s "
        "AND abs(tc_a - %s) <= %s AND abs(tc_b - %s) <= %s",
        (obj_id, first["night_id"], second["night_id"], first["tc"], tol, second["tc"], tol),
    )
    return cur.rowcount


@app.put("/api/object/{obj_id}/repeat_link")
def put_repeat_link(obj_id: int, body: RepeatLinkBody):
    """Record SAME / DIFFERENT on a pair of events (replacing an earlier decision on it).

    Written with the ``relphot_web`` role into ``relphot.repeat_decision`` only; nothing else is
    touched and no event is merged or rejected. The families change at the next ``relphot db
    analyze`` (the response says so).
    """
    if body.decision not in ("SAME", "DIFFERENT"):
        raise HTTPException(status_code=400, detail="decision must be 'SAME' or 'DIFFERENT'")
    if body.note is not None and len(body.note) > _REVIEW_NOTE_MAX:
        raise HTTPException(
            status_code=400, detail=f"note must be at most {_REVIEW_NOTE_MAX} characters"
        )
    with get_rw_conn() as conn, conn.cursor() as cur:
        first, second = _repeat_pair(cur, obj_id, body.det_a, body.det_b)
        _delete_repeat_decision(cur, obj_id, first, second)
        cur.execute(
            "INSERT INTO relphot.repeat_decision "
            "(obj_id, night_a, night_b, tc_a, tc_b, decision, note) "
            "VALUES (%s, %s, %s, %s, %s, %s, NULLIF(%s, '')) "
            "RETURNING night_a, night_b, tc_a, tc_b, decision, note, updated_at",
            (
                obj_id, first["night_id"], second["night_id"], first["tc"], second["tc"],
                body.decision, body.note or "",
            ),
        )
        row = _rows_to_dicts(cur, cur.fetchall())[0]
        conn.commit()
    return _json({"decision": row, "det_a": first["det_id"], "det_b": second["det_id"],
                  "note": _REPEAT_NOTE})


@app.delete("/api/object/{obj_id}/repeat_link")
def delete_repeat_link(obj_id: int, det_a: int = Query(...), det_b: int = Query(...)):
    """Forget the decision on a pair of events (no decision = the automatic link stands)."""
    with get_rw_conn() as conn, conn.cursor() as cur:
        first, second = _repeat_pair(cur, obj_id, det_a, det_b)
        deleted = _delete_repeat_decision(cur, obj_id, first, second)
        conn.commit()
    return _json({"deleted": deleted, "det_a": first["det_id"], "det_b": second["det_id"],
                  "note": _REPEAT_NOTE})


@app.get("/api/repeat/predict")
def repeat_predict(
    start: str | None = Query(default=None, description="UTC date/date-time or BJD; default now"),
    end: str | None = Query(
        default=None, description="UTC date (whole day) / date-time or BJD; default start + 10 d"
    ),
    telescope: str | None = Query(default=None),
    obj_id: int | None = Query(default=None),
    min_alias_frac: float = Query(default=0.0, ge=0.0, le=1.0),
    accepted_only: bool = Query(default=False),
):
    """Windows in which the stored repeated-event families may transit (allowed aliases only).

    Read-only: :func:`relphot.repeat.predict_windows` over the stored ephemerides. A window lists
    how many of the family's allowed aliases predict a transit in it; ``stale`` marks a family
    with a member rejected (or a decision made) since it was computed.
    """
    from astropy.time import Time

    try:
        start_jd = parse_when(start, end=False) if start else float(Time.now().tdb.jd)
        end_jd = parse_when(end, end=True) if end else start_jd + _PREDICT_DEFAULT_DAYS
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"cannot read start / end: {exc}") from exc
    if not end_jd > start_jd:
        raise HTTPException(status_code=400, detail="end must be after start")
    if end_jd - start_jd > 400.0:
        raise HTTPException(status_code=400, detail="the range is limited to 400 days")

    def utc(jd: float) -> str:
        return Time(jd, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")

    windows: list[dict] = []
    n_families = 0
    with get_ro_conn() as conn:
        available = _has_repeat_tables(conn)
        if available:
            families = load_families(
                conn, None if obj_id is None else [obj_id], telescope=telescope,
                accepted_only=accepted_only,
            )
            n_families = len(families)
            names = {f["obj_id"]: f["obj_name"] for f in families}
            for w in predict_windows(families, start_jd, end_jd, DbSettings()):
                if w.n_aliases < min_alias_frac * w.n_aliases_total:
                    continue
                windows.append({
                    "obj_id": w.obj_id, "obj_name": names.get(w.obj_id), "fam_id": w.fam_id,
                    "start_bjd": w.start, "end_bjd": w.end, "start_utc": utc(w.start),
                    "end_utc": utc(w.end), "n_aliases": w.n_aliases,
                    "n_aliases_total": w.n_aliases_total, "depth": w.depth, "t14_h": w.t14_h,
                    "t14_lower_limit": w.t14_lower_limit, "duration_display":
                    _fmt_duration(w.t14_h, w.t14_lower_limit), "stale": w.stale,
                })
    return _json({
        "available": available, "start_bjd": start_jd, "end_bjd": end_jd,
        "start_utc": utc(start_jd), "end_utc": utc(end_jd), "n_families": n_families,
        "windows": windows,
    })


# --------------------------------------------------------------------------
# User-guided reprocessing: /api/object/{obj_id}/reprocess (relphot_web INSERT)
# --------------------------------------------------------------------------

_WIDTH_RANGE_H = (0.1, 12.0)
_PERIOD_GUESS_MAX = 1.0e4
_MAX_ENTRIES = 20
_REPROCESS_COLUMNS = (
    "r.req_id, r.kind, r.period_guess, r.tc_guess, r.width_guess_h, r.night_id, r.note, "
    "r.status, r.requested_at, r.started_at, r.finished_at, r.error, r.result, "
    "n.label AS night_label, "
    "CASE WHEN r.status IN ('queued', 'running') THEN "
    "(SELECT count(*) FROM relphot.reprocess_request q "
    "WHERE q.status IN ('queued', 'running') "
    "AND (q.requested_at, q.req_id) < (r.requested_at, r.req_id)) END AS requests_ahead"
)


class ReprocessExop(BaseModel):
    tc_guess: float | None = None
    width_guess_h: float | None = None


class ReprocessVar(BaseModel):
    period_guess: float | None = None
    all_nights: bool = False


class ReprocessEntry(BaseModel):
    night_id: int | None = None
    exop: ReprocessExop | None = None
    var: ReprocessVar | None = None


class ReprocessBody(BaseModel):
    entries: list[ReprocessEntry] = Field(min_length=1, max_length=_MAX_ENTRIES)
    note: str | None = Field(default=None, max_length=2000)


def _lightcurve_spans(cur, obj_id: int) -> dict[int, tuple[float | None, float | None]]:
    """Light curve time span (t_min, t_max) keyed by night_id, or (None, None) if empty."""
    cur.execute(
        "SELECT night_id, (SELECT min(x) FROM unnest(bjd_tdb) x), "
        "(SELECT max(x) FROM unnest(bjd_tdb) x) FROM relphot.lightcurve WHERE obj_id = %s "
        "ORDER BY night_id",
        (obj_id,),
    )
    return {nid: (t_min, t_max) for nid, t_min, t_max in cur.fetchall()}


def _span_holds(
    spans: dict[int, tuple[float | None, float | None]], night_id: int, tc: float
) -> bool:
    """Whether tc is within the span of the given night_id."""
    if night_id not in spans:
        return False
    t_min, t_max = spans[night_id]
    return t_min is not None and t_min <= tc <= t_max


def _plan_reprocess(
    entries: list[ReprocessEntry], spans: dict[int, tuple[float | None, float | None]]
) -> list[tuple[str, float | None, float | None, float | None, int | None]]:
    """Validate and plan reprocess entries; raise HTTPException(400) on error.

    Returns list of (kind, period_guess, tc_guess, width_guess_h, night_id) tuples.
    """
    rows: list[tuple[str, float | None, float | None, float | None, int | None]] = []
    seen_keys: set[tuple[str, int | None]] = set()

    for i, entry in enumerate(entries, start=1):
        # Neither exop nor var
        if entry.exop is None and entry.var is None:
            raise HTTPException(status_code=400, detail=f"entry {i}: tick EXOP and/or VAR")

        # Check exop
        if entry.exop is not None:
            if entry.exop.tc_guess is not None and not math.isfinite(entry.exop.tc_guess):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: tc_guess "
                            f"must be a finite number"),
                )
            if entry.exop.width_guess_h is not None and not math.isfinite(entry.exop.width_guess_h):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: width_guess_h "
                            f"must be a finite number"),
                )

            if entry.exop.tc_guess is None or entry.exop.width_guess_h is None:
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: EXOP needs "
                            f"tc_guess and width_guess_h"),
                )
            low, high = _WIDTH_RANGE_H
            if not (low <= entry.exop.width_guess_h <= high):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: width_guess_h must be "
                            f"between {low:g} and {high:g} hours"),
                )
            if entry.night_id is None:
                raise HTTPException(status_code=400, detail=f"entry {i}: EXOP needs a night")
            if entry.night_id not in spans:
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: night {entry.night_id} has "
                            f"no light curve of this object"),
                )
            if not _span_holds(spans, entry.night_id, entry.exop.tc_guess):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: tc_guess is not inside "
                            f"night {entry.night_id} of this object"),
                )

            exop_row = (
                "transit", None, entry.exop.tc_guess,
                entry.exop.width_guess_h, entry.night_id,
            )
            key = (exop_row[0], exop_row[4])
            if key in seen_keys:
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: repeats an earlier entry "
                            f"(same kind and night)"),
                )
            rows.append(exop_row)
            seen_keys.add(key)

        # Check var
        if entry.var is not None:
            if entry.var.period_guess is not None and not math.isfinite(entry.var.period_guess):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: period_guess "
                            f"must be a finite number"),
                )

            if (entry.var.period_guess is None or
                    not (0.0 < entry.var.period_guess <= _PERIOD_GUESS_MAX)):
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: a variable request needs "
                            f"0 < period_guess <= {_PERIOD_GUESS_MAX:g} days"),
                )

            if not entry.var.all_nights:
                if entry.night_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"entry {i}: VAR needs a night unless "
                                f'"All nights rerun" is ticked'),
                    )
                if entry.night_id not in spans:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"entry {i}: night {entry.night_id} has "
                                f"no light curve of this object"),
                    )
                night_key = entry.night_id
            else:
                night_key = None

            var_row = ("variable", entry.var.period_guess, None, None, night_key)
            key = (var_row[0], var_row[4])
            if key in seen_keys:
                raise HTTPException(
                    status_code=400,
                    detail=(f"entry {i}: repeats an earlier entry "
                            f"(same kind and night)"),
                )
            rows.append(var_row)
            seen_keys.add(key)

    return rows


@app.post("/api/object/{obj_id}/reprocess")
def post_reprocess(obj_id: int, body: ReprocessBody):
    """Queue a user-guided re-run of one object (worked off by ``relphot db reprocess``)."""
    with get_ro_conn() as ro, ro.cursor() as cur:
        cur.execute("SELECT 1 FROM relphot.object WHERE obj_id = %s", (obj_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="object not found")
        spans = _lightcurve_spans(cur, obj_id)

    # Validate all entries first; insert nothing on error
    rows = _plan_reprocess(body.entries, spans)

    with get_rw_conn() as conn, conn.cursor() as cur:
        results = []
        for kind, period_guess, tc_guess, width_guess_h, night_id in rows:
            cur.execute(
                ("INSERT INTO relphot.reprocess_request "
                 "(obj_id, kind, period_guess, tc_guess, width_guess_h, night_id, note) "
                 "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                 "RETURNING req_id, kind, night_id, status, requested_at"),
                (obj_id, kind, period_guess, tc_guess, width_guess_h, night_id, body.note),
            )
            req_id, kind_ret, night_id_ret, status, requested_at = cur.fetchone()
            results.append({
                "req_id": req_id, "kind": kind_ret, "night_id": night_id_ret,
                "status": status, "requested_at": requested_at,
            })
        conn.commit()

    return _json({"requests": results}, status_code=201)


@app.get("/api/object/{obj_id}/reprocess")
def get_reprocess(obj_id: int):
    """The object's reprocess requests (newest first) and the global queue depth."""
    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM relphot.object WHERE obj_id = %s", (obj_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="object not found")
        cur.execute(
            f"SELECT {_REPROCESS_COLUMNS} FROM relphot.reprocess_request r "
            "LEFT JOIN relphot.night n ON n.night_id = r.night_id "
            "WHERE r.obj_id = %s ORDER BY r.requested_at DESC, r.req_id DESC LIMIT 50",
            (obj_id,),
        )
        requests = _rows_to_dicts(cur, cur.fetchall())
        queue = _queue_depth(cur)
    return _json({"requests": requests, "queue": queue})


def _queue_depth(cur) -> dict[str, int]:
    """Number of queued and running reprocess requests of all objects."""
    cur.execute(
        "SELECT status, count(*) FROM relphot.reprocess_request "
        "WHERE status IN ('queued', 'running') GROUP BY status"
    )
    return {"queued": 0, "running": 0, **dict(cur.fetchall())}


_REPROCESS_STATUSES = {"queued", "running", "done", "failed"}
_MAX_WATCH = 100


@app.get("/api/reprocess")
def list_reprocess(
    status: str = Query(
        default="queued,running", description="comma-separated request statuses to list"
    ),
    watch: list[int] | None = Query(  # noqa: B008
        default=None, max_length=_MAX_WATCH,
        description="req_ids to report whatever their status (how the page sees one finish)",
    ),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Reprocess requests of all objects, newest first: those with one of ``status``
    (default: still queued or running), the ``watch``-ed ones whatever their status, and
    the global queue depth.

    The page polls this while it knows of pending requests; a request it had seen pending
    that comes back ``done``/``failed`` under ``watched`` is a finished RERUN (``finished_at``
    is the worker transaction's start, so a timestamp cannot tell). Read-only.
    """
    wanted = sorted({s for s in status.split(",") if s})
    invalid = sorted(set(wanted) - _REPROCESS_STATUSES)
    if invalid or not wanted:
        raise HTTPException(status_code=400, detail=f"invalid status: {invalid or status!r}")
    select = (
        f"SELECT r.obj_id, o.name AS obj_name, {_REPROCESS_COLUMNS} "
        "FROM relphot.reprocess_request r JOIN relphot.object o ON o.obj_id = r.obj_id "
        "LEFT JOIN relphot.night n ON n.night_id = r.night_id "
    )
    newest_first = "ORDER BY r.requested_at DESC, r.req_id DESC"
    with get_ro_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"{select}WHERE r.status = ANY(%s) {newest_first} LIMIT %s", (wanted, limit)
        )
        requests = _rows_to_dicts(cur, cur.fetchall())
        watched: list[dict] = []
        if watch:
            cur.execute(f"{select}WHERE r.req_id = ANY(%s) {newest_first}", (list(watch),))
            watched = _rows_to_dicts(cur, cur.fetchall())
        queue = _queue_depth(cur)
    return _json({"requests": requests, "watched": watched, "queue": queue})


class AdoptPeriodBody(BaseModel):
    est_id: int


@app.post("/api/object/{obj_id}/adopt_period")
def adopt_period(obj_id: int, body: AdoptPeriodBody):
    """Adopt one period estimate (typically a user-guided one) as the object's manual PERIOD.

    Sets ``period`` to the estimate's period, ``period_err`` and ``period_n_nights`` from it
    and ``period_source = 'manual'``, so ``relphot db analyze`` never replaces it; ``reset
    period to auto`` (``PATCH`` with ``period_source: "auto"``) hands it back.
    """
    with get_rw_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_err, n_nights, verify_status FROM relphot.period_estimate "
            "WHERE est_id = %s AND obj_id = %s",
            (body.est_id, obj_id),
        )
        est = cur.fetchone()
        if est is None:
            raise HTTPException(status_code=404, detail="period estimate not found")
        period, period_err, n_nights, verify_status = est
        if period is None:
            raise HTTPException(status_code=400, detail="this estimate has no period")
        if verify_status == "long_period_needs_tie":
            raise HTTPException(
                status_code=400,
                detail="long period needs a multi-night tie: this estimate cannot be adopted",
            )
        cur.execute(
            "UPDATE relphot.object SET period = %s, period_err = %s, period_n_nights = %s, "
            "period_source = 'manual', updated_at = now() WHERE obj_id = %s RETURNING *",
            (period, period_err, n_nights, obj_id),
        )
        row = cur.fetchone()
        columns = [d.name for d in cur.description]
        conn.commit()
    return _json(dict(zip(columns, row, strict=True)))


# --------------------------------------------------------------------------
# Static front end
# --------------------------------------------------------------------------

_STATIC_DIR = resources.files("relphot.web") / "static"


@app.get("/static/plotly.min.js")
def plotly_js() -> FileResponse:
    path = resources.files("plotly").joinpath("package_data", "plotly.min.js")
    return FileResponse(str(path), media_type="application/javascript")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))
