"""Load one relphot night's results into the results database.

See docs/DB_PLAN.md for the schema and the noise cut this implements.
:func:`load_night` reads ``night.npz``, ``ref.npz``, and the
``lc/<stem>_starstats.parquet`` / ``_lightcurves.parquet`` /
``_search_metrics.parquet`` triple written by ``relphot lightcurves`` and
``relphot search``, and inserts or replaces that night's rows in
``relphot.night``/``frame``/``star_night``/``lightcurve``/``detection``/
``catalog_match``, matching stars to existing ``relphot.object`` rows (or
creating new ones) by position. Everything happens in one transaction.

A reload of the same night keeps its ``user_night_review`` rows (per-night user verdicts
survive the reload); a new night is always evaluated automatically. The night's
detections are deleted, but a person's ``status``/``notes`` on them are saved first and
re-attached to the matching new detection (or, when none matches, kept in
``relphot.detection_review_orphan``); an object with user_night_review rows is never
dropped as an orphan. A transit detection whose flags include EDGE or PARTIAL is stored
with ``duration_lower_limit`` set (its duration is only a minimum). Only ``origin = 'search'``
detections are deleted and re-created: a detection a person created by reprocessing
(``origin = 'user'``, see :mod:`relphot.db.reprocess`) survives a reload.

``night.zp`` / ``zp_source``: the median Gaia ``ZPABS`` of the kept frames (``'gaia'``); without
one, the telescope's measured zero point in ``settings.db.telescope_zp`` (``'measured'``; T80S
27.85 is the Gaia DR3 G scale, uniform across telescopes); else ``settings.db.assumed_zp``
(``'assumed'``, 20 mag).

``star_night.err_scale`` / ``blended`` come from ``*_starstats.parquet`` (the factor
``lc_err`` was inflated by and the neighbour-flag verdict); a product written before error
inflation existed has neither column and loads them as NULL (read as a factor of 1).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from relphot.config import DbSettings, Settings
from relphot.db.refresh import refresh_objects
from relphot.exceptions import NightLoadError

logger = logging.getLogger(__name__)

__all__ = ["LoadReport", "load_night"]

_LABEL_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


@dataclass(slots=True)
class LoadReport:
    """Counts and identifiers from one :func:`load_night` call."""

    night_id: int
    telescope: str
    label: str
    n_frames: int
    n_kept: int
    n_stars: int
    n_passed_cut: int
    n_candidates_forced: int
    n_stored: int
    n_new_objects: int
    n_matched_objects: int
    lightcurve_stars: int
    lightcurve_points: int
    n_transit_detections: int
    n_variable_detections: int
    n_catalog_matches: int
    n_reference_members: int = 0
    n_comparison_members: int = 0
    elapsed_s: float = 0.0


def _format_ra_hms(ra_deg: float) -> str:
    """``ra_deg`` (0-360) as ``hhmmss.ss``, carrying a rounded 60.00s/60m up."""
    hours = (ra_deg / 15.0) % 24.0
    h = int(hours)
    m_full = (hours - h) * 60.0
    m = int(m_full)
    s = round((m_full - m) * 60.0, 2)
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    h %= 24
    return f"{h:02d}{m:02d}{s:05.2f}"


def _format_dec_dms(dec_deg: float) -> str:
    """``dec_deg`` as ``sddmmss.s`` (sign, degrees, minutes, seconds), with carry."""
    sign = "-" if dec_deg < 0 else "+"
    adeg = abs(dec_deg)
    d = int(adeg)
    m_full = (adeg - d) * 60.0
    m = int(m_full)
    s = round((m_full - m) * 60.0, 1)
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return f"{sign}{d:02d}{m:02d}{s:04.1f}"


def _object_base_name(ra_deg: float, dec_deg: float) -> str:
    return f"RP J{_format_ra_hms(ra_deg)}{_format_dec_dms(dec_deg)}"


def _unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        existing.add(base)
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    name = f"{base}-{i}"
    existing.add(name)
    return name


def _infer_telescope(night_dir: Path) -> str | None:
    """Telescope from a ``<TEL>_reduced`` (SSD staging) or ``<TEL>/reduced`` (permanent storage)."""
    parts = night_dir.parts
    for i, part in enumerate(parts):
        if part.endswith("_reduced") and len(part) > len("_reduced"):
            return part[: -len("_reduced")]
        if part == "reduced" and i > 1:
            return parts[i - 1]
    return None


def _night_date_from_label(label: str, frame_meta: list[dict]) -> date:
    m = _LABEL_DATE_RE.match(label)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return date(y, mo, d)
    return datetime.fromisoformat(frame_meta[0]["date_obs"]).date()


def _night_zero_point(
    frame_meta: list[dict], frame_kept: np.ndarray, telescope: str, settings: DbSettings
) -> tuple[float, str]:
    """(zp, source) of a night: the median Gaia ``zp`` of its kept frames when at least half of
    them carry one (``'gaia'``); else the telescope's measured zero point from
    ``settings.telescope_zp`` (``'measured'``; T80S 27.85 on the Gaia G scale, uniform across
    telescopes); else ``settings.assumed_zp`` (``'assumed'``, robo43's ``instrumental_zp`` of
    20 mag)."""
    zps = np.array(
        [np.nan if m.get("zp") is None else float(m["zp"]) for m in frame_meta], dtype=float
    )
    good = np.isfinite(zps) & frame_kept
    if good.any() and 2 * int(good.sum()) >= int(frame_kept.sum()):
        return float(np.median(zps[good])), "gaia"
    if telescope in settings.telescope_zp:
        return float(settings.telescope_zp[telescope]), "measured"
    return float(settings.assumed_zp), "assumed"


def _mode(values: list[str]) -> str | None:
    counts = Counter(v for v in values if v)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _json_safe(value: object) -> object:
    """``value`` as a JSON-serialisable native Python scalar (NaN/inf -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        return value
    return value


def _row_extra(row: dict, cols: list[str]) -> dict[str, object]:
    return {c: _json_safe(row[c]) for c in cols if c in row}


def _nan_to_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _is_incomplete_transit(flags_str: str, partial: bool) -> bool:
    """Whether a transit event is incomplete (EDGE/PARTIAL), so its duration is a minimum."""
    tokens = set(flags_str.split("|"))
    return partial or "EDGE" in tokens or "PARTIAL" in tokens


def _restore_detection_reviews(
    cur: psycopg.Cursor, night_id: int, saved: list[tuple]
) -> tuple[int, int]:
    """Re-attach saved detection reviews to the reloaded night's detections.

    ``saved`` holds ``(obj_id, kind, tc, duration_h, status, notes)`` rows.

    A saved transit goes to the new detection of the same object and kind whose ``tc``
    differs by at most half the transit duration (the nearest one); any other kind goes
    to the same object and kind. What cannot be matched is logged as a warning and
    kept in ``relphot.detection_review_orphan``. Returns ``(n_restored, n_orphaned)``.
    """
    cur.execute(
        "SELECT det_id, obj_id, kind, tc_bjd_tdb, duration_h FROM relphot.detection "
        "WHERE night_id = %s AND origin = 'search' ORDER BY det_id",
        (night_id,),
    )
    candidates: dict[tuple[int, str], list[tuple[int, float | None, float | None]]] = {}
    for det_id, obj_id, kind, tc, dur in cur.fetchall():
        candidates.setdefault((obj_id, kind), []).append((det_id, tc, dur))

    used: set[int] = set()
    n_restored = n_orphaned = 0
    for obj_id, kind, tc, duration_h, status, notes in saved:
        pool = [c for c in candidates.get((obj_id, kind), []) if c[0] not in used]
        if kind == "transit" and tc is not None:
            pool = [
                c for c in pool
                if c[1] is not None
                and abs(c[1] - tc)
                <= 0.5 * (duration_h if duration_h is not None else (c[2] or 0.0)) / 24.0
            ]
            pool.sort(key=lambda c: abs(c[1] - tc))
        if pool:
            used.add(pool[0][0])
            cur.execute(
                "UPDATE relphot.detection SET status = %s, notes = %s WHERE det_id = %s",
                (status, notes, pool[0][0]),
            )
            n_restored += 1
        else:
            logger.warning(
                "night %d reload: no detection matches the saved review (obj_id=%d, kind=%s, "
                "tc=%s, status=%s); kept in relphot.detection_review_orphan",
                night_id, obj_id, kind, tc, status,
            )
            cur.execute(
                "INSERT INTO relphot.detection_review_orphan "
                "(obj_id, night_id, kind, tc_bjd_tdb, status, notes) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (obj_id, night_id, kind, tc, status, notes),
            )
            n_orphaned += 1
    return n_restored, n_orphaned


def _restore_supersede_links(
    cur: psycopg.Cursor, night_id: int, saved: list[tuple]
) -> tuple[int, int]:
    """Re-attach the saved ``detection.superseded_by`` links of the reloaded night.

    ``saved`` holds ``(det_id, obj_id, origin, tc, duration_h, superseded_by)`` of every transit
    event that was linked (either end) before the reload. A person's (``origin = 'user'``) event
    keeps its id; a search event was re-created, so it goes to the new search transit event of
    the same object whose ``tc`` differs by at most half the saved duration (the nearest one),
    like a saved review. A link whose either end cannot be found is dropped with a warning: both
    events are then active again. Returns ``(n_restored, n_lost)``.
    """
    cur.execute(
        "SELECT det_id, obj_id, tc_bjd_tdb, duration_h FROM relphot.detection "
        "WHERE night_id = %s AND kind = 'transit' AND origin = 'search' ORDER BY det_id",
        (night_id,),
    )
    pool: dict[int, list[tuple[int, float | None, float | None]]] = {}
    for det_id, obj_id, tc, dur in cur.fetchall():
        pool.setdefault(obj_id, []).append((det_id, tc, dur))

    now_id: dict[int, int] = {}
    used: set[int] = set()
    for old_id, obj_id, origin, tc, duration_h, _ in saved:
        if origin != "search":
            now_id[old_id] = old_id
            continue
        if tc is None:
            continue
        near = [
            c for c in pool.get(obj_id, [])
            if c[0] not in used and c[1] is not None
            and abs(c[1] - tc)
            <= 0.5 * (duration_h if duration_h is not None else (c[2] or 0.0)) / 24.0
        ]
        if near:
            best = min(near, key=lambda c: abs(c[1] - tc))
            used.add(best[0])
            now_id[old_id] = best[0]

    n_restored = n_lost = 0
    for old_id, obj_id, _origin, tc, _dur, old_parent in saved:
        if old_parent is None:
            continue
        child, parent = now_id.get(old_id), now_id.get(old_parent)
        if child is None or parent is None:
            logger.warning(
                "night %d reload: a transit event of obj_id=%d (tc=%s) superseded by an event "
                "that no longer matches; both are active again",
                night_id, obj_id, tc,
            )
            n_lost += 1
            continue
        cur.execute(
            "UPDATE relphot.detection SET superseded_by = %s WHERE det_id = %s", (parent, child)
        )
        n_restored += 1
    return n_restored, n_lost


def _optional_col(row: object, name: str) -> float | None:
    """``row.<name>`` as a finite float, or ``None`` when the column is absent or NaN.

    For metrics columns that only newer ``relphot search`` outputs (or another
    producer) carry, e.g. a catalogue period uncertainty.
    """
    value = getattr(row, name, None)
    return None if value is None else _nan_to_none(value)


#: PostgreSQL's ``real`` (float4) underflows on a nonzero magnitude below this
#: (see :mod:`relphot.db.load_multinight`'s own ``_REAL_UNDERFLOW_FLOOR``,
#: guarding the same kind of column): a transit SNR/depth/duration or a
#: variability amplitude/excess/Lomb-Scargle FAP on very clean data can
#: compute far below it.
_REAL_UNDERFLOW_FLOOR = 1e-30


def _real_safe(value: float | None) -> float | None:
    """``value`` as a value ``relphot.detection``'s ``real`` columns can store.

    Non-finite maps to ``None``; a nonzero magnitude below
    ``_REAL_UNDERFLOW_FLOOR`` is clamped up to that floor (keeping its sign)
    instead of being passed through to Postgres, which raises
    ``NumericValueOutOfRange`` on an underflowing ``real``.
    """
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if 0.0 < abs(value) < _REAL_UNDERFLOW_FLOOR:
        return math.copysign(_REAL_UNDERFLOW_FLOOR, value)
    return value


def _infer_variable_catalog(name: str) -> str:
    """Best-effort source catalogue for a merged ``known_variable_name``.

    :func:`relphot.catalogs.match_known_variables` merges VSX, Gaia DR3
    variability, and ASAS-SN hits into one name (the first non-empty one, in
    query order) with no catalogue column kept per star. Only the Gaia DR3
    fallback-naming format (``"Gaia DR3 <source_id>"``, used only when no
    catalogue-native name column matched) is reliably recoverable from the
    name string alone; anything else is reported under the merged label.
    """
    if name.startswith("Gaia DR3 "):
        return "Gaia DR3"
    return "VSX|Gaia DR3|ASAS-SN (merged)"


def _discover_lc_files(
    night_dir: Path, lc_stem: str | None
) -> tuple[Path, Path, Path | None]:
    lc_dir = night_dir / "lc"
    if lc_stem is not None:
        starstats_path = lc_dir / f"{lc_stem}_starstats.parquet"
        if not starstats_path.is_file():
            msg = f"no such file: {starstats_path}"
            raise NightLoadError(msg)
    else:
        candidates = sorted(lc_dir.glob("*_starstats.parquet"))
        if len(candidates) == 0:
            msg = f"no *_starstats.parquet found under {lc_dir}"
            raise NightLoadError(msg)
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            msg = f"multiple *_starstats.parquet found under {lc_dir}: {names} (pass --lc-stem)"
            raise NightLoadError(msg)
        starstats_path = candidates[0]
        lc_stem = starstats_path.name[: -len("_starstats.parquet")]

    lightcurves_path = lc_dir / f"{lc_stem}_lightcurves.parquet"
    if not lightcurves_path.is_file():
        msg = f"no such file: {lightcurves_path}"
        raise NightLoadError(msg)

    search_metrics_path = lc_dir / f"{lc_stem}_search_metrics.parquet"
    if not search_metrics_path.is_file():
        search_metrics_path = None
    return starstats_path, lightcurves_path, search_metrics_path


#: transit_*/variability_* search-metrics columns not already mapped to a
#: scalar ``detection`` column, kept verbatim (NaN -> null) in ``extra``.
_TRANSIT_EXTRA_BASE = [
    "transit_searched", "transit_n_in", "transit_beta", "transit_coverage",
    "transit_partial", "transit_frame_error_scale_at_tc", "transit_dchi2_box_vs_flat",
    "transit_dchi2_box_vs_step", "transit_coincidence_count", "transit_flags",
    # R90 screen (relphot.transit_r90): absent from search-metrics files written before it.
    "transit_r90_evaluated", "transit_r90_pass", "transit_top1_share", "transit_top3_share",
    "transit_reg_dchi2_ratio", "transit_reg_depth_ratio", "transit_clip3_dchi2",
    "transit_dbic_flat",
    # NEIGHBOUR_SHARED_EVENT partner (relphot.transit_neighbour): same, absent before it.
    "transit_shared_partner", "transit_shared_sep_arcsec", "transit_shared_depth",
    "transit_shared_dip_sigma", "transit_shared_dtc_hours", "transit_shared_deficit_ratio",
    "transit_shared_is_source", "transit_shared_gaia_id",
]
_VARIABLE_EXTRA_COLS = [
    "variability_searched", "variability_rms_robust", "variability_rms_std",
    "variability_von_neumann", "variability_von_neumann_significance",
    "variability_ls_power", "variability_trend_slope", "variability_trend_significance",
    "variability_systematic_excluded", "variability_class",
]


def load_night(
    conn: psycopg.Connection,
    night_dir: Path | str,
    *,
    telescope: str | None = None,
    label: str | None = None,
    lc_stem: str | None = None,
    settings: Settings | None = None,
) -> LoadReport:
    """Load one night's relphot outputs from ``night_dir`` into the database.

    ``night_dir`` is a night's ``relphot/`` directory (holding ``night.npz``,
    ``ref.npz``, and ``lc/``). ``settings`` supplies the noise cut
    (``settings.db``, ``settings.search``) uniformly across every night
    loaded this way, regardless of the settings snapshot recorded inside
    that night's own files; it defaults to :class:`~relphot.config.Settings`
    with every field at its default. Runs as one transaction: commits on
    success, and the caller should roll back on any raised exception.
    """
    t0 = time.monotonic()
    settings = settings if settings is not None else Settings()
    night_dir = Path(night_dir).resolve()

    resolved_telescope = telescope or _infer_telescope(night_dir)
    if not resolved_telescope:
        msg = (
            f"cannot infer telescope from {night_dir} "
            "(no '<TEL>_reduced' or '<TEL>/reduced' path component); pass --telescope"
        )
        raise NightLoadError(msg)
    resolved_label = label or night_dir.parent.name

    night_npz = night_dir / "night.npz"
    ref_npz = night_dir / "ref.npz"
    if not night_npz.is_file():
        msg = f"no such file: {night_npz}"
        raise NightLoadError(msg)
    if not ref_npz.is_file():
        msg = f"no such file: {ref_npz}"
        raise NightLoadError(msg)

    with np.load(night_npz, allow_pickle=False) as data:
        frame_meta: list[dict] = json.loads(str(data["frame_meta_json"]))
        config_raw: dict = json.loads(str(data["config_json"]))
    with np.load(ref_npz, allow_pickle=False) as data:
        frame_kept = np.asarray(data["frame_kept"], dtype=bool)

    n_frames = len(frame_meta)
    if frame_kept.shape[0] != n_frames:
        msg = (
            f"{ref_npz} frame_kept has {frame_kept.shape[0]} entries, "
            f"{night_npz} has {n_frames} frames"
        )
        raise NightLoadError(msg)
    n_kept = int(frame_kept.sum())
    night_zp, night_zp_source = _night_zero_point(
        frame_meta, frame_kept, resolved_telescope, settings.db
    )

    night_date = _night_date_from_label(resolved_label, frame_meta)
    object_name = _mode([m["object"] for m in frame_meta])
    filt = _mode([m["filter"] for m in frame_meta])
    site_cfg = config_raw.get("site") or {}
    site_lat = site_cfg.get("latitude_deg")
    site_lon = site_cfg.get("longitude_deg")
    site_elev = site_cfg.get("elevation_m")

    starstats_path, lightcurves_path, search_metrics_path = _discover_lc_files(
        night_dir, lc_stem
    )
    df_ss = pd.read_parquet(starstats_path)
    n_stars = len(df_ss)

    df_sm: pd.DataFrame | None = None
    if search_metrics_path is not None:
        df_sm = pd.read_parquet(search_metrics_path)
    else:
        logger.warning(
            "%s: no *_search_metrics.parquet found; loading without detections", night_dir
        )

    min_epochs = settings.search.effective_min_epochs(n_kept)
    max_expected_noise = settings.db.max_expected_noise
    keep_candidates = settings.db.keep_candidates

    if df_sm is not None:
        cand_bool = (
            df_sm["transit_candidate"].to_numpy() | df_sm["variability_candidate"].to_numpy()
        )
        cand_series = pd.Series(cand_bool, index=df_sm["star_id"].to_numpy())
        is_candidate = df_ss["star_id"].map(cand_series).fillna(False).to_numpy(dtype=bool)
    else:
        is_candidate = np.zeros(n_stars, dtype=bool)

    rms = df_ss["rms"].to_numpy(dtype=np.float64)
    n_epochs = df_ss["n_epochs"].to_numpy(dtype=np.int64)
    expected_noise = df_ss["expected_noise"].to_numpy(dtype=np.float64)
    passes_normal = (
        np.isfinite(rms) & (n_epochs >= min_epochs) & (expected_noise <= max_expected_noise)
    )
    forced = keep_candidates & is_candidate & ~passes_normal
    stored_mask = passes_normal | (keep_candidates & is_candidate)

    n_passed_cut = int(np.count_nonzero(passes_normal))
    n_candidates_forced = int(np.count_nonzero(forced))
    n_stored = int(np.count_nonzero(stored_mask))

    noise_cut = {
        "max_expected_noise": max_expected_noise,
        "min_epochs": int(min_epochs),
        "keep_candidates": bool(keep_candidates),
        "n_stars": n_stars,
        "n_passed_cut": n_passed_cut,
        "n_candidates_forced": n_candidates_forced,
        "n_stored": n_stored,
    }

    df_store = df_ss.loc[stored_mask].copy()
    store_star_ids = df_store["star_id"].to_numpy()
    store_id_set = {int(s) for s in store_star_ids}

    n_new_objects = 0
    n_matched_objects = 0
    lightcurve_stars = 0
    lightcurve_points = 0
    n_transit_detections = 0
    n_variable_detections = 0
    n_catalog_matches = 0
    n_reference_members = 0
    n_comparison_members = 0

    try:
        with conn.cursor() as cur:
            # --- night upsert (reuses night_id on a reload) ---
            cur.execute(
                """
                INSERT INTO relphot.night
                    (telescope, night_date, label, site_lat, site_lon, site_elev,
                     object, filter, n_frames, n_kept, source_dir, settings,
                     noise_cut, zp, zp_source, loaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (source_dir) DO UPDATE SET
                    telescope = EXCLUDED.telescope,
                    night_date = EXCLUDED.night_date,
                    label = EXCLUDED.label,
                    site_lat = EXCLUDED.site_lat,
                    site_lon = EXCLUDED.site_lon,
                    site_elev = EXCLUDED.site_elev,
                    object = EXCLUDED.object,
                    filter = EXCLUDED.filter,
                    n_frames = EXCLUDED.n_frames,
                    n_kept = EXCLUDED.n_kept,
                    settings = EXCLUDED.settings,
                    noise_cut = EXCLUDED.noise_cut,
                    zp = EXCLUDED.zp,
                    zp_source = EXCLUDED.zp_source,
                    loaded_at = EXCLUDED.loaded_at
                RETURNING night_id
                """,
                (
                    resolved_telescope, night_date, resolved_label, site_lat, site_lon,
                    site_elev, object_name, filt, n_frames, n_kept, str(night_dir),
                    Jsonb(config_raw), Jsonb(noise_cut), night_zp, night_zp_source,
                ),
            )
            (night_id,) = cur.fetchone()

            # --- reload: drop this night's previous frame/star_night/detection rows ---
            # a person's status/notes on a detection are saved first and re-attached below
            cur.execute(
                "SELECT obj_id, kind, tc_bjd_tdb, duration_h, status, notes "
                "FROM relphot.detection WHERE night_id = %s AND origin = 'search' "
                "AND (status IS DISTINCT FROM 'UNCONFIRMED' OR notes IS NOT NULL)",
                (night_id,),
            )
            saved_reviews = cur.fetchall()
            # the supersede links (a RERUN's event replacing another of the same light curve) of
            # this night: the search events are re-created below with new ids, and the person's
            # events lose a link to one of them, so both ends are saved and re-attached
            cur.execute(
                "SELECT det_id, obj_id, origin, tc_bjd_tdb, duration_h, superseded_by "
                "FROM relphot.detection WHERE night_id = %(night_id)s AND kind = 'transit' "
                "AND (superseded_by IS NOT NULL OR det_id IN ("
                "SELECT superseded_by FROM relphot.detection "
                "WHERE night_id = %(night_id)s AND superseded_by IS NOT NULL))",
                {"night_id": night_id},
            )
            saved_links = cur.fetchall()
            # the repeated-event families with an event of this night go too (their ephemeris
            # rows stay as history); `db analyze` recomputes them
            cur.execute(
                "DELETE FROM relphot.repeat_family WHERE fam_id IN ("
                "SELECT m.fam_id FROM relphot.repeat_family_member m "
                "JOIN relphot.detection d ON d.det_id = m.det_id "
                "WHERE d.night_id = %s AND d.origin = 'search')",
                (night_id,),
            )
            # a user's own detections (origin = 'user') are never reloaded away
            cur.execute(
                "DELETE FROM relphot.detection WHERE night_id = %s AND origin = 'search'",
                (night_id,),
            )
            cur.execute("DELETE FROM relphot.star_night WHERE night_id = %s", (night_id,))
            cur.execute("DELETE FROM relphot.night_tile WHERE night_id = %s", (night_id,))
            cur.execute("DELETE FROM relphot.frame WHERE night_id = %s", (night_id,))

            # --- frames ---
            with cur.copy(
                "COPY relphot.frame "
                "(night_id, frame_index, file_name, file_path, date_obs, jd_utc, "
                "bjd_tdb, exptime, airmass, fwhm, n_sources, kept) FROM STDIN"
            ) as copy:
                for idx, m in enumerate(frame_meta):
                    dt = datetime.fromisoformat(m["date_obs"]).replace(tzinfo=UTC)
                    copy.write_row(
                        (
                            night_id, idx, Path(m["file"]).name, m["file"], dt,
                            m["jd_utc"], m["bjd_tdb"], m["exptime"], m["airmass"],
                            m["median_fwhm"], m["n_sources"], bool(frame_kept[idx]),
                        )
                    )

            # --- cross-match stored stars to existing objects (bulk, q3c) ---
            cur.execute(
                "CREATE TEMP TABLE tmp_stars "
                "(star_id bigint, ra double precision, dec double precision) "
                "ON COMMIT DROP"
            )
            with cur.copy("COPY tmp_stars (star_id, ra, dec) FROM STDIN") as copy:
                for star_id, ra, dec in zip(
                    df_store["star_id"], df_store["ra"], df_store["dec"], strict=True
                ):
                    copy.write_row((int(star_id), float(ra), float(dec)))

            radius_deg = settings.db.match_radius_arcsec / 3600.0
            cur.execute(
                """
                SELECT t.star_id, o.obj_id, o.dist_arcsec
                FROM tmp_stars t
                LEFT JOIN LATERAL (
                    SELECT obj_id, q3c_dist(t.ra, t.dec, o2.ra, o2.dec) * 3600.0 AS dist_arcsec
                    FROM relphot.object o2
                    WHERE q3c_join(t.ra, t.dec, o2.ra, o2.dec, %s)
                    ORDER BY q3c_dist(t.ra, t.dec, o2.ra, o2.dec) ASC
                    LIMIT 1
                ) o ON true
                """,
                (radius_deg,),
            )
            match_rows = cur.fetchall()

            by_obj: dict[int, list[tuple[int, float]]] = {}
            unmatched_star_ids: list[int] = []
            star_to_obj: dict[int, int] = {}
            for star_id, obj_id, dist in match_rows:
                if obj_id is None:
                    unmatched_star_ids.append(star_id)
                else:
                    by_obj.setdefault(obj_id, []).append((star_id, dist))
            for obj_id, claims in by_obj.items():
                claims.sort(key=lambda c: c[1])
                winner_star_id, _ = claims[0]
                star_to_obj[winner_star_id] = obj_id
                for loser_star_id, _ in claims[1:]:
                    unmatched_star_ids.append(loser_star_id)
            n_matched_objects = len(star_to_obj)

            # --- create new objects for unmatched stars ---
            if unmatched_star_ids:
                cur.execute("SELECT name FROM relphot.object")
                existing_names = {row[0] for row in cur.fetchall()}

                star_lookup = df_store.set_index("star_id")
                new_rows = []
                for star_id in sorted(unmatched_star_ids):
                    ra = float(star_lookup.loc[star_id, "ra"])
                    dec = float(star_lookup.loc[star_id, "dec"])
                    name = _unique_name(_object_base_name(ra, dec), existing_names)
                    new_rows.append((star_id, name, ra, dec))

                cur.execute(
                    "CREATE TEMP TABLE tmp_new_obj "
                    "(star_id bigint, name text, ra double precision, dec double precision) "
                    "ON COMMIT DROP"
                )
                with cur.copy("COPY tmp_new_obj (star_id, name, ra, dec) FROM STDIN") as copy:
                    for row in new_rows:
                        copy.write_row(row)
                cur.execute(
                    "INSERT INTO relphot.object (name, ra, dec) "
                    "SELECT name, ra, dec FROM tmp_new_obj RETURNING obj_id, name"
                )
                name_to_obj = {name: obj_id for obj_id, name in cur.fetchall()}
                for star_id, name, _ra, _dec in new_rows:
                    star_to_obj[star_id] = name_to_obj[name]
                n_new_objects = len(new_rows)

            # --- star_night ---
            with cur.copy(
                "COPY relphot.star_night "
                "(obj_id, night_id, star_id, tile, mag, best_aperture, rms, "
                "expected_noise, chi2_reduced, n_epochs, is_comparison, err_scale, blended) "
                "FROM STDIN"
            ) as copy:
                for row in df_store.itertuples(index=False):
                    blended = getattr(row, "blended", None)
                    copy.write_row(
                        (
                            star_to_obj[row.star_id], night_id, int(row.star_id), int(row.tile),
                            _nan_to_none(row.mag), int(row.best_aperture), _nan_to_none(row.rms),
                            _nan_to_none(row.expected_noise), _nan_to_none(row.chi2_reduced),
                            int(row.n_epochs), bool(row.is_comparison),
                            _optional_col(row, "err_scale"),
                            None if blended is None or pd.isna(blended) else bool(blended),
                        )
                    )

            # --- reference and comparison members (optional) ---
            members_path = lightcurves_path.with_name(
                lightcurves_path.name[: -len("_lightcurves.parquet")] + "_members.npz"
            )
            if members_path.is_file():
                from relphot.db.load_members import insert_members
                from relphot.members import load_members_npz

                try:
                    members_product = load_members_npz(members_path)
                    members_report = insert_members(
                        cur,
                        night_id,
                        members_product,
                        star_to_obj,
                        match_radius_deg=settings.db.match_radius_arcsec / 3600.0,
                        n_frames=n_frames,
                    )
                    # Update report fields
                    n_reference_members = members_report.n_reference_members
                    n_comparison_members = members_report.n_comparison_members
                except Exception:
                    logger.exception(
                        "error loading members from %s; skipping",
                        members_path,
                    )
            else:
                logger.info("no %s; reference/comparison members not loaded", members_path.name)

            # --- light curves: numpy groupby, not a per-row Python loop over the parquet ---
            df_lc = pd.read_parquet(lightcurves_path)
            df_lc = df_lc[df_lc["star_id"].isin(store_id_set)]
            df_lc = df_lc.sort_values(["star_id", "frame"], kind="stable")

            n_lc_rows = len(df_lc)
            if n_lc_rows:
                lc_star_ids = df_lc["star_id"].to_numpy()
                boundaries = np.flatnonzero(np.diff(lc_star_ids)) + 1
                group_starts = np.concatenate(([0], boundaries))
                frame_arr = df_lc["frame"].to_numpy(dtype=np.int64)
                bjd_arr = df_lc["bjd_tdb"].to_numpy(dtype=np.float64)
                flux_arr = df_lc["lc"].to_numpy(dtype=np.float32)
                fluxerr_arr = df_lc["lc_err"].to_numpy(dtype=np.float32)
                fluxraw_arr = df_lc["lc_raw"].to_numpy(dtype=np.float32)

                frame_groups = np.split(frame_arr, group_starts[1:])
                bjd_groups = np.split(bjd_arr, group_starts[1:])
                flux_groups = np.split(flux_arr, group_starts[1:])
                fluxerr_groups = np.split(fluxerr_arr, group_starts[1:])
                fluxraw_groups = np.split(fluxraw_arr, group_starts[1:])
                group_star_ids = lc_star_ids[group_starts]

                with cur.copy(
                    "COPY relphot.lightcurve "
                    "(obj_id, night_id, frame_index, bjd_tdb, flux, flux_err, flux_raw) "
                    "FROM STDIN"
                ) as copy:
                    for sid, frames, bjds, fluxes, fluxerrs, fluxraws in zip(
                        group_star_ids, frame_groups, bjd_groups, flux_groups,
                        fluxerr_groups, fluxraw_groups, strict=True,
                    ):
                        copy.write_row(
                            (
                                star_to_obj[int(sid)], night_id,
                                [int(f) for f in frames],
                                [float(b) for b in bjds],
                                [float(f) for f in fluxes],
                                [float(f) for f in fluxerrs],
                                [float(f) for f in fluxraws],
                            )
                        )
                lightcurve_stars = len(group_star_ids)
                lightcurve_points = n_lc_rows

            # --- detections + catalog matches + gaia/neighbour from search_metrics ---
            if df_sm is not None:
                df_sm_store = df_sm[df_sm["star_id"].isin(store_id_set)]

                n_aper = 0
                aper_re = re.compile(r"^transit_depth_aper(\d+)$")
                for col in df_sm.columns:
                    m = aper_re.match(col)
                    if m:
                        n_aper = max(n_aper, int(m.group(1)) + 1)
                transit_extra_cols = [
                    *_TRANSIT_EXTRA_BASE,
                    *[f"transit_depth_aper{a}" for a in range(n_aper)],
                    *[f"transit_sigma_depth_aper{a}" for a in range(n_aper)],
                ]

                detection_rows = []
                catalog_rows = []
                gaia_rows = []
                for row in df_sm_store.itertuples(index=False):
                    obj_id = star_to_obj[int(row.star_id)]
                    is_cand = row.transit_candidate or row.variability_candidate
                    row_d = row._asdict() if is_cand else None

                    if row.transit_candidate:
                        detection_rows.append(
                            (
                                obj_id, night_id, "transit",
                                _real_safe(row.transit_snr), _real_safe(row.transit_depth),
                                _nan_to_none(row.transit_tc_bjd_tdb),
                                _real_safe(row.transit_duration_hours), int(row.transit_tier),
                                str(row.transit_flags_str), None, None, None, None,
                                Jsonb(_row_extra(row_d, transit_extra_cols)),
                                _is_incomplete_transit(
                                    str(row.transit_flags_str), bool(row.transit_partial)
                                ),
                            )
                        )
                        n_transit_detections += 1
                    if row.variability_candidate:
                        detection_rows.append(
                            (
                                obj_id, night_id, "variable",
                                None, None, None, None, None, None,
                                _real_safe(row.variability_amplitude),
                                _real_safe(row.variability_excess),
                                _nan_to_none(row.variability_ls_period_days),
                                _real_safe(row.variability_ls_fap),
                                Jsonb(_row_extra(row_d, _VARIABLE_EXTRA_COLS)),
                                False,
                            )
                        )
                        n_variable_detections += 1

                    var_name = str(row.known_variable_name) if row.known_variable else ""
                    if var_name:
                        var_type = str(row.known_variable_type)
                        catalog_rows.append(
                            (
                                obj_id, _infer_variable_catalog(var_name), var_name,
                                var_type or None,
                                _nan_to_none(row.known_variable_period_days),
                                _optional_col(row, "known_variable_period_err_days"),
                            )
                        )
                    planet_name = str(row.known_planet_name) if row.known_planet else ""
                    if planet_name:
                        catalog_rows.append(
                            (
                                obj_id,
                                "TOI" if row.known_planet_is_toi else "NASA Exoplanet Archive",
                                planet_name, None,
                                _nan_to_none(row.known_planet_period_days),
                                _optional_col(row, "known_planet_period_err_days"),
                            )
                        )

                    gaia_id = str(row.gaia_id) if row.gaia_id else ""
                    sep = _nan_to_none(row.neighbour_sep_arcsec)
                    if gaia_id or sep is not None:
                        gaia_rows.append((obj_id, gaia_id or None, sep))

                if detection_rows:
                    with cur.copy(
                        "COPY relphot.detection "
                        "(obj_id, night_id, kind, snr, depth, tc_bjd_tdb, duration_h, tier, "
                        "flags, amplitude, excess, period, fap, extra, duration_lower_limit) "
                        "FROM STDIN"
                    ) as copy:
                        for r in detection_rows:
                            copy.write_row(r)

                if catalog_rows:
                    cur.executemany(
                        """
                        INSERT INTO relphot.catalog_match
                            (obj_id, catalog, name, type, period, period_err)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (obj_id, catalog, name) DO UPDATE SET
                            type = EXCLUDED.type, period = EXCLUDED.period,
                            period_err = EXCLUDED.period_err
                        """,
                        catalog_rows,
                    )
                    n_catalog_matches = len(catalog_rows)

                if gaia_rows:
                    cur.execute(
                        "CREATE TEMP TABLE tmp_gaia "
                        "(obj_id bigint, gaia_id text, neighbour_sep_arcsec real) "
                        "ON COMMIT DROP"
                    )
                    with cur.copy(
                        "COPY tmp_gaia (obj_id, gaia_id, neighbour_sep_arcsec) FROM STDIN"
                    ) as copy:
                        for r in gaia_rows:
                            copy.write_row(r)
                    cur.execute(
                        """
                        UPDATE relphot.object o SET gaia_id = t.gaia_id
                        FROM tmp_gaia t
                        WHERE o.obj_id = t.obj_id AND o.gaia_id IS NULL AND t.gaia_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        UPDATE relphot.object o SET neighbour_sep_arcsec = t.neighbour_sep_arcsec
                        FROM tmp_gaia t
                        WHERE o.obj_id = t.obj_id AND o.neighbour_sep_arcsec IS NULL
                            AND t.neighbour_sep_arcsec IS NOT NULL
                        """
                    )

            if saved_reviews:
                _restore_detection_reviews(cur, night_id, saved_reviews)
            if saved_links:
                _restore_supersede_links(cur, night_id, saved_links)

            # --- drop objects this reload (or a prior one) left with no star_night, except
            # those a person has touched: any manual flag or period, a status, notes, a
            # detection of their own (origin 'user') or a reprocess request ---
            cur.execute(
                """
                DELETE FROM relphot.object o
                WHERE o.class_source IS DISTINCT FROM 'manual'
                    AND o.exop_source IS DISTINCT FROM 'manual'
                    AND o.var_source IS DISTINCT FROM 'manual'
                    AND o.period_source IS DISTINCT FROM 'manual'
                    AND COALESCE(o.status, 'UNCONFIRMED') = 'UNCONFIRMED'
                    AND o.notes IS NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM relphot.star_night sn WHERE sn.obj_id = o.obj_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM relphot.detection d
                        WHERE d.obj_id = o.obj_id AND d.origin = 'user'
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM relphot.reprocess_request r WHERE r.obj_id = o.obj_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM relphot.user_night_review r WHERE r.obj_id = o.obj_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM relphot.repeat_decision r WHERE r.obj_id = o.obj_id
                    )
                """
            )

            touched_obj_ids = sorted(set(star_to_obj.values()))
            cur.execute(
                "UPDATE relphot.object SET data_updated_at = now() WHERE obj_id = ANY(%(obj_ids)s)",
                {"obj_ids": touched_obj_ids},
            )
            refresh_objects(
                conn, touched_obj_ids,
                bls_min_snr=settings.db.bls_min_snr, ls_fap_threshold=settings.db.ls_fap_threshold,
                class_multinight_kinds=settings.db.class_multinight_kinds,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return LoadReport(
        night_id=night_id,
        telescope=resolved_telescope,
        label=resolved_label,
        n_frames=n_frames,
        n_kept=n_kept,
        n_stars=n_stars,
        n_passed_cut=n_passed_cut,
        n_candidates_forced=n_candidates_forced,
        n_stored=n_stored,
        n_new_objects=n_new_objects,
        n_matched_objects=n_matched_objects,
        lightcurve_stars=lightcurve_stars,
        lightcurve_points=lightcurve_points,
        n_transit_detections=n_transit_detections,
        n_variable_detections=n_variable_detections,
        n_catalog_matches=n_catalog_matches,
        n_reference_members=n_reference_members,
        n_comparison_members=n_comparison_members,
        elapsed_s=time.monotonic() - t0,
    )
