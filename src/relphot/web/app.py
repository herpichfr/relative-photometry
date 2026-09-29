"""FastAPI application for the relphot results-database web API (see
docs/DB_PLAN.md, "Web (requirements 10-15)").

Every read goes through the ``relphot_ro`` role (SELECT only); the two
manual-edit endpoints (``PATCH /api/object/{obj_id}`` and ``PATCH
/api/detection/{det_id}``) go through ``relphot_web`` (SELECT plus UPDATE on a
fixed set of ``relphot.object`` / ``relphot.detection`` columns). No query
ever interpolates a user-supplied *value* into SQL --
only a handful of fixed, code-controlled identifiers (column names from a
whitelist, ``ASC``/``DESC``) are ever placed directly in a query string.
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

from relphot.web.db import column_exists, get_ro_conn, get_rw_conn, resolve_ro_dsn

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


def _json(data) -> JSONResponse:
    return JSONResponse(content=_sanitize(jsonable_encoder(data)))


def _rows_to_dicts(cur, rows) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _fmt_duration(hours: float | None, lower_limit: bool | None) -> str | None:
    """A duration for display; an incomplete transit's is only a minimum, shown as ``>= x.xx h``."""
    if hours is None or not math.isfinite(hours):
        return None
    return f"\u2265 {hours:.2f} h" if lower_limit else f"{hours:.2f} h"


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
    "amplitude", "status", "first_night", "last_night",
]
#: Result columns computed per object: transit-event count, best "matching transits"
#: probability, and the latest period verification (delta = P_obs/harmonic - P_lit, with
#: the status and note saying whether/why it could be verified).
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
}
_SEARCH_JOINS = (
    " LEFT JOIN LATERAL (SELECT pe.delta, pe.delta_err, pe.verify_status, pe.verify_note "
    "FROM relphot.period_estimate pe "
    "WHERE pe.obj_id = o.obj_id AND pe.lit_period IS NOT NULL "
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
_OBJECT_OPTIONAL_COLUMNS = ["data_updated_at"]


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

            cur.execute(
                "SELECT catalog, name, type, period, period_err, sep_arcsec, reference "
                "FROM relphot.catalog_match WHERE obj_id = %s ORDER BY catalog",
                (obj_id,),
            )
            catalog_matches = _rows_to_dicts(cur, cur.fetchall())

            cur.execute(
                "SELECT d.det_id, d.kind, d.snr, d.depth, d.tc_bjd_tdb, d.duration_h, d.tier, "
                "d.flags, d.amplitude, d.excess, d.period, d.fap, d.extra, d.night_id, "
                "n.label AS night_label, n.telescope, d.mn_run_id, mr.stem AS mn_run_stem, "
                "d.status, d.notes, d.duration_lower_limit "
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

            cur.execute(
                "SELECT sn.night_id, n.label, n.telescope, n.night_date, sn.n_epochs, sn.rms, "
                "sn.expected_noise, sn.mag, sn.best_aperture, (lc.obj_id IS NOT NULL) AS has_lc "
                "FROM relphot.star_night sn "
                "JOIN relphot.night n ON n.night_id = sn.night_id "
                "LEFT JOIN relphot.lightcurve lc "
                "ON lc.obj_id = sn.obj_id AND lc.night_id = sn.night_id "
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
                "d.duration_lower_limit AS det_duration_lower_limit, "
                "ts.tc, ts.tc_err, ts.depth, ts.depth_err, ts.t14_h, ts.t14_err, "
                "ts.t14_lower_limit, ts.incomplete_reason, "
                "ts.ingress_frac, ts.ingress_err, ts.chi2_red, ts.n_points, ts.input, "
                "ts.converged "
                "FROM relphot.detection d "
                "JOIN relphot.night n ON n.night_id = d.night_id "
                "LEFT JOIN relphot.transit_shape ts ON ts.det_id = d.det_id "
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
                "lit_catalog, harmonic, delta, delta_err, delta_z, verify_status, verify_note "
                "FROM relphot.period_estimate WHERE obj_id = %s "
                "ORDER BY last_night, n_nights, computed_at",
                (obj_id,),
            )
            period_estimates = _rows_to_dicts(cur, cur.fetchall())

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

        cur.execute("SELECT label FROM relphot.night WHERE night_id = %s", (night_id,))
        night_row = cur.fetchone()
        night_label = night_row[0] if night_row else None

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
        }
    )


@app.get("/api/object/{obj_id}/lc/combined")
def object_lc_combined(obj_id: int):
    with get_ro_conn() as conn, conn.cursor() as cur:
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

        bjd_tdb: list[float] = []
        value: list[float] = []
        value_err: list[float] = []
        night_ids: list[int] = []
        night_labels: list = []
        file_names: list = []

        for night_id, tie_mag, _tie_mag_err in tie_rows:
            cur.execute(
                "SELECT frame_index, bjd_tdb, flux, flux_err FROM relphot.lightcurve "
                "WHERE obj_id = %s AND night_id = %s",
                (obj_id, night_id),
            )
            lc_row = cur.fetchone()
            if lc_row is None:
                continue
            frame_index, night_bjd, night_flux, night_flux_err = lc_row

            cur.execute("SELECT label FROM relphot.night WHERE night_id = %s", (night_id,))
            (night_label,) = cur.fetchone()

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
            bjd_tdb.extend(float(x) for x in night_bjd)
            value.extend(float(x) for x in point_value)
            value_err.extend(float(x) for x in point_err)
            night_ids.extend([night_id] * n)
            night_labels.extend([night_label] * n)
            file_names.extend(fname_map.get(i) for i in frame_index)

    return _json(
        {
            "mode": mode,
            "bjd_tdb": bjd_tdb,
            "value": value,
            "value_err": value_err,
            "night_id": night_ids,
            "night_label": night_labels,
            "file_name": file_names,
        }
    )


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
    """Manual edits of one object.

    ``is_exop`` / ``is_var`` set that flag by hand (its ``*_source`` becomes
    ``'manual'``, so ``relphot db analyze`` never changes it); the two flags are
    independent. ``exop_source`` / ``var_source`` = ``'auto'`` hands a flag back to
    the pipeline. The older ``class`` / ``class_source`` fields still work as
    shorthand: ``class`` sets the flags it names to true (``'UNC'`` clears both) and
    leaves the other flag alone; ``class_source`` = ``'auto'`` resets both flags.
    ``class`` / ``class_source`` on the row are always re-derived from the flags.
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

    flags, sources = _flag_edits(provided)
    if flags or sources:
        eff_sources: dict[str, str | None] = {}
        for key in ("exop", "var"):
            if key in flags:
                set_parts.append(f"is_{key} = %(is_{key})s")
                set_parts.append(f"{key}_source = 'manual'")
                eff_sources[key] = "manual"
            elif key in sources:
                set_parts.append(f"{key}_source = %({key}_source)s")
                eff_sources[key] = sources[key]
            else:
                eff_sources[key] = None
            params[f"is_{key}"] = flags.get(key)
            params[f"{key}_source"] = eff_sources[key]
        # class / class_source are derived labels: recompute them from the row as it will be
        set_parts.append(
            "class = CASE "
            "WHEN COALESCE(%(is_exop)s::boolean, is_exop) "
            "AND COALESCE(%(is_var)s::boolean, is_var) THEN 'EXOP+VAR' "
            "WHEN COALESCE(%(is_exop)s::boolean, is_exop) THEN 'EXOP' "
            "WHEN COALESCE(%(is_var)s::boolean, is_var) THEN 'VAR' ELSE 'UNC' END"
        )
        set_parts.append(
            "class_source = CASE "
            "WHEN COALESCE(%(exop_source)s::text, exop_source) = 'manual' "
            "OR COALESCE(%(var_source)s::text, var_source) = 'manual' "
            "THEN 'manual' ELSE 'auto' END"
        )

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

    if not set_parts:
        raise HTTPException(status_code=400, detail="no recognised fields to update")

    set_parts.append("updated_at = now()")
    sql = (
        f"UPDATE relphot.object SET {', '.join(set_parts)} "
        "WHERE obj_id = %(obj_id)s RETURNING *"
    )
    with get_rw_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
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
