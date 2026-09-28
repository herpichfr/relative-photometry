"""Load one relphot multi-night tie run's products into the results database.

See docs/DB_PLAN.md for the schema (``mn_run``/``tie``/``detection``) this
implements. :func:`load_multinight` reads ``STEM.npz`` (written by
``relphot multinight``, via :func:`relphot.multinight.load_multinight`) and,
when ``search_dir`` is given, ``search_dir/multinight_search_metrics.parquet``
(written by ``relphot multisearch``), and inserts or replaces that run's rows
in ``relphot.mn_run``/``tie``/``detection``, mapping each global star to the
``relphot.object`` row its per-night ``star_id`` already resolves to in
``relphot.star_night``. Everything happens in one transaction.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from relphot.config import Settings, settings_to_dict
from relphot.db.refresh import refresh_objects
from relphot.exceptions import NightLoadError
from relphot.multinight import load_multinight as _read_multinight_npz

__all__ = ["MultiNightLoadReport", "load_multinight"]


@dataclass(slots=True)
class MultiNightLoadReport:
    """Counts and identifiers from one :func:`load_multinight` call."""

    mn_run_id: int
    stem: str
    labels: list[str]
    anchor: str
    n_tie_rows: int
    n_objects_mapped: int
    n_globals_unmapped: int
    n_conflicts: int
    n_detections_internight: int
    n_detections_ls_periodic: int
    n_detections_bls: int
    n_detections_recurrent: int
    elapsed_s: float


def _nan_to_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


#: PostgreSQL's ``real`` (float4) underflows on a nonzero magnitude below this
#: (see :mod:`relphot.db.analyze`'s own ``_REAL_UNDERFLOW_FLOOR``, guarding
#: the same periodogram FAP/power columns): an inter-night or Lomb-Scargle
#: false-alarm probability on very clean data can compute far below it.
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


def _resolve_night_ids(
    cur: psycopg.Cursor, labels: list[str], night_info: list[dict]
) -> list[int]:
    """``night_id`` for each of ``labels``, in order.

    Matched by ``relphot.night.source_dir`` against ``night_info``'s own
    recorded directory for that label, falling back to a match on
    ``relphot.night.label`` alone when exactly one night has that label.
    Raises :class:`~relphot.exceptions.NightLoadError` listing every label
    that resolves to no night.
    """
    dir_by_label = {ni["label"]: ni["directory"] for ni in night_info}

    cur.execute("SELECT night_id, source_dir, label FROM relphot.night")
    night_rows = cur.fetchall()
    night_id_by_source_dir = {
        source_dir: night_id for night_id, source_dir, _label in night_rows if source_dir
    }
    night_ids_by_label: dict[str, list[int]] = defaultdict(list)
    for night_id, _source_dir, label in night_rows:
        night_ids_by_label[label].append(night_id)

    night_ids: list[int] = []
    missing: list[str] = []
    for label in labels:
        directory = dir_by_label.get(label)
        resolved_dir = str(Path(directory).resolve()) if directory else None
        night_id = night_id_by_source_dir.get(resolved_dir) if resolved_dir else None
        if night_id is None:
            candidates = night_ids_by_label.get(label, [])
            if len(candidates) == 1:
                night_id = candidates[0]
        if night_id is None:
            missing.append(label)
        else:
            night_ids.append(night_id)

    if missing:
        msg = (
            f"nights not loaded in the results database: {', '.join(missing)} "
            "-- load these nights first with `relphot db load-night`"
        )
        raise NightLoadError(msg)
    return night_ids


def load_multinight(
    conn: psycopg.Connection,
    stem: Path | str,
    *,
    search_dir: Path | str | None = None,
    settings: Settings | None = None,
) -> MultiNightLoadReport:
    """Load one multi-night tie run's products from ``STEM.npz`` (+ search metrics).

    ``stem`` is the output stem passed to ``relphot multinight --out`` (no
    extension); ``STEM.npz`` is read via
    :func:`relphot.multinight.load_multinight`. ``search_dir`` is the output
    directory of a matching ``relphot multisearch`` run (holding
    ``multinight_search_metrics.parquet``); when omitted, no
    ``relphot.detection`` rows are written. ``settings`` supplies the
    :func:`~relphot.db.refresh.refresh_objects` thresholds (``settings.db``);
    it defaults to :class:`~relphot.config.Settings` with every field at its
    default. Runs as one transaction: commits on success, and the caller
    should roll back on any raised exception.

    Every night the run covers must already be loaded (``relphot db
    load-night``) -- see :func:`_resolve_night_ids`.

    A global star's per-night ``star_id`` may resolve (via
    ``relphot.star_night``) to different ``relphot.object`` rows across the
    nights it appears in (a rare cross-match disagreement). When that
    happens, the object with the most agreeing nights wins (ties broken by
    the anchor night's object, else the lowest object id); the count of
    globals with such a disagreement is reported, and the run's
    disagreeing nights get no ``tie`` row for that star.
    """
    t0 = time.monotonic()
    settings = settings if settings is not None else Settings()
    stem_path = Path(stem)
    npz_path = stem_path.with_suffix(".npz")
    if not npz_path.is_file():
        msg = f"no such file: {npz_path}"
        raise NightLoadError(msg)
    resolved_stem = str(stem_path.resolve())

    xmatch, tie, mlc, night_info, mn_settings = _read_multinight_npz(npz_path)
    labels = list(xmatch.labels)
    anchor_label = labels[tie.anchor_index]

    n_det_internight = n_det_ls_periodic = n_det_bls = n_det_recurrent = 0

    try:
        with conn.cursor() as cur:
            night_ids = _resolve_night_ids(cur, labels, night_info)
            night_ids_arr = np.asarray(night_ids, dtype=np.int64)
            anchor_night_id = int(night_ids_arr[tie.anchor_index])

            # --- mn_run upsert (reuses mn_run_id on a reload) ---
            cur.execute(
                """
                INSERT INTO relphot.mn_run (stem, labels, anchor, settings, loaded_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (stem) DO UPDATE SET
                    labels = EXCLUDED.labels,
                    anchor = EXCLUDED.anchor,
                    settings = EXCLUDED.settings,
                    loaded_at = EXCLUDED.loaded_at
                RETURNING mn_run_id
                """,
                (resolved_stem, labels, anchor_label, Jsonb(settings_to_dict(mn_settings))),
            )
            (mn_run_id,) = cur.fetchone()

            # --- reload: drop this run's previous tie/detection rows ---
            cur.execute("DELETE FROM relphot.detection WHERE mn_run_id = %s", (mn_run_id,))
            cur.execute("DELETE FROM relphot.tie WHERE mn_run_id = %s", (mn_run_id,))

            # --- global star -> obj_id, via each night's own star_id -> star_night ---
            valid_n, valid_g = np.nonzero(xmatch.index >= 0)
            valid_star_id = xmatch.index[valid_n, valid_g]
            valid_night_id = night_ids_arr[valid_n]

            cur.execute(
                "CREATE TEMP TABLE tmp_mn_xmatch "
                "(g bigint, night_id integer, star_id integer) ON COMMIT DROP"
            )
            with cur.copy("COPY tmp_mn_xmatch (g, night_id, star_id) FROM STDIN") as copy:
                for g, night_id, star_id in zip(
                    valid_g.tolist(), valid_night_id.tolist(), valid_star_id.tolist(), strict=True
                ):
                    copy.write_row((int(g), int(night_id), int(star_id)))

            cur.execute(
                "SELECT t.g, t.night_id, sn.obj_id FROM tmp_mn_xmatch t "
                "JOIN relphot.star_night sn ON sn.night_id = t.night_id AND sn.star_id = t.star_id"
            )
            matched_rows = cur.fetchall()

            matched_by_g: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for g, night_id, obj_id in matched_rows:
                matched_by_g[g].append((night_id, obj_id))

            night_id_to_n = {night_id: n for n, night_id in enumerate(night_ids)}

            winner_by_g: dict[int, int] = {}
            n_conflicts = 0
            tie_rows: list[tuple[int, int, float, float | None]] = []
            for g, entries in matched_by_g.items():
                counts = Counter(obj_id for _night_id, obj_id in entries)
                max_count = max(counts.values())
                top = [oid for oid, c in counts.items() if c == max_count]
                if len(top) == 1:
                    winner = top[0]
                else:
                    anchor_obj = next(
                        (
                            obj_id for night_id, obj_id in entries
                            if night_id == anchor_night_id and obj_id in top
                        ),
                        None,
                    )
                    winner = anchor_obj if anchor_obj is not None else min(top)
                if len(counts) > 1:
                    n_conflicts += 1
                winner_by_g[g] = winner

                for night_id, obj_id in entries:
                    if obj_id != winner:
                        continue
                    n_idx = night_id_to_n[night_id]
                    mag = float(mlc.night_mean_mag[n_idx, g])
                    if not math.isfinite(mag):
                        continue
                    mag_err = _nan_to_none(float(mlc.night_mean_err[n_idx, g]))
                    tie_rows.append((winner, night_id, mag, mag_err))

            n_globals_unmapped = xmatch.index.shape[1] - len(matched_by_g)

            if tie_rows:
                with cur.copy(
                    "COPY relphot.tie (mn_run_id, obj_id, night_id, mag, mag_err) FROM STDIN"
                ) as copy:
                    for obj_id, night_id, mag, mag_err in tie_rows:
                        copy.write_row((mn_run_id, obj_id, night_id, mag, mag_err))

            # --- multi-night detections (only when search_dir is given) ---
            if search_dir is not None:
                metrics_path = Path(search_dir) / "multinight_search_metrics.parquet"
                if not metrics_path.is_file():
                    msg = f"no such file: {metrics_path}"
                    raise NightLoadError(msg)
                df_sm = pd.read_parquet(metrics_path)

                detection_rows = []
                for row in df_sm.itertuples(index=False):
                    obj_id = winner_by_g.get(int(row.global_id))
                    if obj_id is None:
                        continue

                    if row.internight_candidate:
                        detection_rows.append((
                            obj_id, mn_run_id, "internight",
                            None, None, None, None, None, None,
                            _real_safe(row.internight_amplitude), None, None, None,
                            Jsonb({
                                "chi2": _json_safe(row.internight_chi2),
                                "p": _json_safe(row.internight_p),
                            }),
                        ))
                        n_det_internight += 1

                    if row.periodic_candidate:
                        detection_rows.append((
                            obj_id, mn_run_id, "ls_periodic",
                            None, None, None, None, None, None, None, None,
                            _nan_to_none(row.ls_period_days), _real_safe(row.ls_fap),
                            Jsonb({
                                "power": _json_safe(row.ls_power),
                                "second_period_days": _json_safe(row.ls_second_period_days),
                            }),
                        ))
                        n_det_ls_periodic += 1

                    if row.bls_candidate:
                        bls_duration_days = _nan_to_none(row.bls_duration_days)
                        duration_h = (
                            _real_safe(bls_duration_days * 24.0)
                            if bls_duration_days is not None else None
                        )
                        flags = str(row.bls_flags) if row.bls_flags else None
                        detection_rows.append((
                            obj_id, mn_run_id, "bls",
                            _real_safe(row.bls_depth_snr), _real_safe(row.bls_depth),
                            _nan_to_none(row.bls_t0), duration_h, None, flags,
                            None, None, _nan_to_none(row.bls_period_days), None,
                            Jsonb({
                                "nights_in_transit": _json_safe(row.bls_nights_in_transit),
                                "period_compat_frac_excluded": _json_safe(
                                    row.period_compat_frac_excluded
                                ),
                                "period_compat_frac_supported": _json_safe(
                                    row.period_compat_frac_supported
                                ),
                                "period_compat_frac_unconstrained": _json_safe(
                                    row.period_compat_frac_unconstrained
                                ),
                                "period_compat_allowed_intervals": _json_safe(
                                    row.period_compat_allowed_intervals
                                ),
                                "period_compat_best_period_days": _json_safe(
                                    row.period_compat_best_period_days
                                ),
                                "period_compat_best_snr": _json_safe(row.period_compat_best_snr),
                            }),
                        ))
                        n_det_bls += 1

                    if row.recurrent_variable:
                        detection_rows.append((
                            obj_id, mn_run_id, "recurrent",
                            None, None, None, None, None, None, None, None, None, None,
                            Jsonb({
                                "n_nights_var_candidate": _json_safe(row.n_nights_var_candidate),
                            }),
                        ))
                        n_det_recurrent += 1

                if detection_rows:
                    with cur.copy(
                        "COPY relphot.detection "
                        "(obj_id, mn_run_id, kind, snr, depth, tc_bjd_tdb, duration_h, tier, "
                        "flags, amplitude, excess, period, fap, extra) FROM STDIN"
                    ) as copy:
                        for r in detection_rows:
                            copy.write_row(r)

            # --- touch mapped objects and recompute their derived summaries ---
            touched_obj_ids = sorted(set(winner_by_g.values()))
            if touched_obj_ids:
                cur.execute(
                    "UPDATE relphot.object SET data_updated_at = now() "
                    "WHERE obj_id = ANY(%(obj_ids)s)",
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

    return MultiNightLoadReport(
        mn_run_id=mn_run_id,
        stem=resolved_stem,
        labels=labels,
        anchor=anchor_label,
        n_tie_rows=len(tie_rows),
        n_objects_mapped=len(set(winner_by_g.values())),
        n_globals_unmapped=n_globals_unmapped,
        n_conflicts=n_conflicts,
        n_detections_internight=n_det_internight,
        n_detections_ls_periodic=n_det_ls_periodic,
        n_detections_bls=n_det_bls,
        n_detections_recurrent=n_det_recurrent,
        elapsed_s=time.monotonic() - t0,
    )
