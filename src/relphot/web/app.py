"""FastAPI application for the relphot results-database web API (see
docs/DB_PLAN.md, "Web (requirements 10-15)").

Every read goes through the ``relphot_ro`` role (SELECT only); the manual-edit
endpoints (``PATCH /api/object/{obj_id}``, ``PATCH /api/detection/{det_id}``,
``PUT /api/object/{obj_id}/night/{night_id}/review``, and
``POST /api/object/{obj_id}/adopt_period``) and the reprocess-request endpoint
(``POST /api/object/{obj_id}/reprocess``, an INSERT into the queue
``relphot db reprocess`` works off) go through ``relphot_web`` (SELECT plus UPDATE on a
fixed set of ``relphot.object`` / ``relphot.detection`` / ``relphot.user_night_review``
columns, plus INSERT on a fixed set of ``relphot.reprocess_request`` columns). No query
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

import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from relphot.config import DbSettings
from relphot.objflags import PLANET_CATALOGS, night_state, refresh_flags
from relphot.web.db import column_exists, get_ro_conn, get_rw_conn, resolve_ro_dsn
from relphot.web.phase import fourier_model, phase_coverage

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
        "rerun_pending": rerun_pending,
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
                "r.note AS review_note, r.updated_at AS review_updated_at "
                "FROM relphot.star_night sn "
                "JOIN relphot.night n ON n.night_id = sn.night_id "
                "LEFT JOIN relphot.lightcurve lc "
                "ON lc.obj_id = sn.obj_id AND lc.night_id = sn.night_id "
                "LEFT JOIN relphot.user_night_review r "
                "ON r.obj_id = sn.obj_id AND r.night_id = sn.night_id "
                "WHERE sn.obj_id = %s ORDER BY n.night_date",
                (obj_id,),
            )
            nights = _rows_to_dicts(cur, cur.fetchall())

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
            # nearest in time, capped; n_similar is the full count
            similar_ids = {
                ev["det_id"]: (ev.pop("similar_det_ids") or [])[:_MAX_SIMILAR_SHOWN]
                for ev in transit_events
            }
            wanted = sorted({j for ids in similar_ids.values() for j in ids})
            similar_info: dict[int, dict] = {}
            if wanted:
                cur.execute(
                    "SELECT d.det_id, d.obj_id, o.name AS obj_name, "
                    "COALESCE(ts.tc, d.tc_bjd_tdb) AS tc, COALESCE(ts.depth, d.depth) AS depth, "
                    "COALESCE(ts.t14_h, d.duration_h) AS t14_h, "
                    "(COALESCE(ts.t14_lower_limit, false) OR d.duration_lower_limit) "
                    "AS t14_lower_limit "
                    "FROM relphot.detection d JOIN relphot.object o ON o.obj_id = d.obj_id "
                    "LEFT JOIN relphot.transit_shape ts ON ts.det_id = d.det_id "
                    "WHERE d.det_id = ANY(%s)",
                    (wanted,),
                )
                for sim in _rows_to_dicts(cur, cur.fetchall()):
                    sim["duration_display"] = _fmt_duration(sim["t14_h"], sim["t14_lower_limit"])
                    similar_info[sim["det_id"]] = sim
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
            "period_estimates": period_estimates,
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
