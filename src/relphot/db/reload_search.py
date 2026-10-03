"""Replace one loaded night's unvetted search transit events (``relphot db reload-search``).

``relphot db load-night`` of an already-loaded night rebuilds everything of it (frames, star_night,
light curves, every ``origin = 'search'`` detection with new ids) and re-attaches a person's
status/notes by the event's centre time; an event the new search no longer finds becomes an orphan
review. After a change of the transit search that is more than needed and it moves vetted events.
:func:`reload_search_detections` only swaps the night's *unvetted* search transit events for the
ones in the night's current ``*_search_metrics.parquet`` (from a re-run ``relphot search``):

- the night's frames, stars and light curves are not touched (the search does not change them), nor
  are its ``origin = 'user'`` detections, its variable detections, the multi-night tie rows and
  multi-night detections (``mn_run``) or any ``user_night_review`` row;
- a *vetted* event (:mod:`relphot.db.vetted`: a person's CONFIRMED / REJECTED status or notes, a
  RERUN event, a supersede link, any event of an object with a ``user_night_review`` row for the
  night) is kept as it is, same ``det_id``, shape, auto verdict: nothing is re-attached, nothing is
  orphaned;
- the unvetted events (and their ``transit_shape`` / ``transit_coincidence`` / ``transit_match``
  rows, and the repeated-event families they belong to) are deleted and the new candidate events
  are inserted for every star of the night that has a ``star_night`` row, except those of an object
  with a vetted event of this night that overlaps in time (``|tc - tc_vetted|`` within half the
  larger duration) or with a ``user_night_review`` row for the night;
- the new events have no shape yet: run ``relphot db analyze --all --keep-vetted``.

A candidate star without a ``star_night`` row of the night (not stored when the night was loaded) is
skipped and counted; only a full ``load-night`` can add it. Everything is one transaction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg

from relphot.config import Settings
from relphot.db.load_night import (
    _discover_lc_files,
    _transit_detection_row,
    _transit_extra_cols,
)
from relphot.db.refresh import refresh_objects
from relphot.db.vetted import vetted_det_ids
from relphot.exceptions import NightLoadError

logger = logging.getLogger(__name__)

__all__ = ["ReloadSearchReport", "reload_search_detections"]


@dataclass(slots=True)
class ReloadSearchReport:
    """What one :func:`reload_search_detections` call did."""

    night_id: int
    label: str
    #: unvetted search transit events deleted / vetted ones left alone / new events inserted
    n_deleted: int
    n_vetted_kept: int
    n_inserted: int
    #: new candidates not inserted: the star's object has a review row for the night or a vetted
    #: event at that time, or the star has no ``star_night`` row
    n_skipped_vetted: int
    n_skipped_no_star: int
    elapsed_s: float


def reload_search_detections(
    conn: psycopg.Connection,
    night_dir: Path | str,
    *,
    lc_stem: str | None = None,
    settings: Settings | None = None,
) -> ReloadSearchReport:
    """Swap one loaded night's unvetted search transit events for the current search metrics.

    ``night_dir`` is the night's ``relphot/`` directory exactly as given to ``load-night`` (the
    night is found by its resolved ``source_dir``); raises :class:`NightLoadError` when the night
    is not loaded or ``lc/<stem>_search_metrics.parquet`` is missing. Commits on success.
    """
    t0 = time.monotonic()
    settings = settings if settings is not None else Settings()
    night_dir = Path(night_dir).resolve()
    _, _, metrics_path = _discover_lc_files(night_dir, lc_stem)
    if metrics_path is None:
        msg = f"no *_search_metrics.parquet found under {night_dir / 'lc'}"
        raise NightLoadError(msg)
    df_sm = pd.read_parquet(metrics_path)

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT night_id, label FROM relphot.night WHERE source_dir = %s",
                (str(night_dir),),
            )
            found = cur.fetchone()
            if found is None:
                msg = f"night {night_dir} is not loaded (use `relphot db load-night` first)"
                raise NightLoadError(msg)
            night_id, label = found

            cur.execute(
                "SELECT star_id, obj_id FROM relphot.star_night WHERE night_id = %s", (night_id,)
            )
            star_to_obj = dict(cur.fetchall())
            vetted = vetted_det_ids(cur, night_ids=[night_id])
            cur.execute(
                "SELECT obj_id FROM relphot.user_night_review WHERE night_id = %s", (night_id,)
            )
            reviewed_objs = {row[0] for row in cur.fetchall()}
            cur.execute(
                "SELECT det_id, obj_id, tc_bjd_tdb, duration_h FROM relphot.detection "
                "WHERE night_id = %s AND kind = 'transit' AND origin = 'search'",
                (night_id,),
            )
            search_events = cur.fetchall()
            doomed = [row[0] for row in search_events if row[0] not in vetted]
            doomed_set = set(doomed)
            kept: dict[int, list[tuple[float | None, float | None]]] = {}
            for det_id, obj_id, tc, dur in search_events:
                if det_id in vetted:
                    kept.setdefault(obj_id, []).append((tc, dur))

            # the repeated-event families with a deleted event go too (`db analyze` recomputes
            # them; the people's repeat_decision rows are not touched)
            cur.execute(
                "DELETE FROM relphot.repeat_family WHERE fam_id IN ("
                "SELECT m.fam_id FROM relphot.repeat_family_member m "
                "WHERE m.det_id = ANY(%s))",
                (doomed,),
            )
            cur.execute("DELETE FROM relphot.detection WHERE det_id = ANY(%s)", (doomed,))

            extra_cols = _transit_extra_cols(df_sm.columns)
            rows = []
            touched = {row[1] for row in search_events if row[0] in doomed_set}
            n_skipped_vetted = n_skipped_no_star = 0
            for row in df_sm[df_sm["transit_candidate"].astype(bool)].itertuples(index=False):
                obj_id = star_to_obj.get(int(row.star_id))
                if obj_id is None:
                    n_skipped_no_star += 1
                    continue
                tc = row.transit_tc_bjd_tdb
                dur = row.transit_duration_hours
                overlaps = any(
                    k_tc is not None and tc == tc
                    and abs(k_tc - tc) <= 0.5 * max(k_dur or 0.0, dur if dur == dur else 0.0) / 24.0
                    for k_tc, k_dur in kept.get(obj_id, [])
                )
                if obj_id in reviewed_objs or overlaps:
                    n_skipped_vetted += 1
                    continue
                rows.append(
                    _transit_detection_row(row, row._asdict(), obj_id, night_id, extra_cols)
                )
                touched.add(obj_id)
            if rows:
                with cur.copy(
                    "COPY relphot.detection "
                    "(obj_id, night_id, kind, snr, depth, tc_bjd_tdb, duration_h, tier, "
                    "flags, amplitude, excess, period, fap, extra, duration_lower_limit) "
                    "FROM STDIN"
                ) as copy:
                    for r in rows:
                        copy.write_row(r)
            if touched:
                cur.execute(
                    "UPDATE relphot.object SET data_updated_at = now() WHERE obj_id = ANY(%s)",
                    (sorted(touched),),
                )
                refresh_objects(
                    conn, sorted(touched),
                    bls_min_snr=settings.db.bls_min_snr,
                    ls_fap_threshold=settings.db.ls_fap_threshold,
                    class_multinight_kinds=settings.db.class_multinight_kinds,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    report = ReloadSearchReport(
        night_id=night_id, label=label, n_deleted=len(doomed),
        n_vetted_kept=sum(1 for row in search_events if row[0] in vetted), n_inserted=len(rows),
        n_skipped_vetted=n_skipped_vetted, n_skipped_no_star=n_skipped_no_star,
        elapsed_s=time.monotonic() - t0,
    )
    logger.info(
        "night %s (%d): %d unvetted search events replaced by %d, %d vetted kept, "
        "%d new skipped (vetted object), %d (no star_night)",
        label, night_id, report.n_deleted, report.n_inserted, report.n_vetted_kept,
        report.n_skipped_vetted, report.n_skipped_no_star,
    )
    return report
