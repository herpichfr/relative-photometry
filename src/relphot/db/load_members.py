"""Load reference and comparison members into the results database.

:func:`load_members` reads a ``*_members.npz`` product written by
:mod:`relphot.members` and inserts the reference and comparison member
records (night_tile, tile_lc, reference_member, comparison_member)
for a night already loaded via :func:`~relphot.db.load_night.load_night`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg

from relphot.db.load_night import _discover_lc_files
from relphot.exceptions import NightLoadError
from relphot.members import MembersProduct, load_members_npz

logger = logging.getLogger(__name__)

__all__ = ["MembersLoadReport", "insert_members", "load_members"]


@dataclass(slots=True)
class MembersLoadReport:
    """Counts and identifiers from one :func:`load_members` call."""

    night_id: int
    n_tiles: int
    n_tile_lc: int
    n_reference_members: int
    n_comparison_members: int
    n_linked: int
    n_unlinked: int
    elapsed_s: float


def insert_members(
    cur: psycopg.Cursor,
    night_id: int,
    product: MembersProduct,
    star_to_obj: dict[int, int],
    match_radius_deg: float,
    n_frames: int,
) -> MembersLoadReport:
    """Insert members from a MembersProduct into the database.

    Parameters
    ----------
    cur : psycopg.Cursor
        Cursor for executing queries.
    night_id : int
        The night_id to load members for.
    product : MembersProduct
        The members product loaded from _members.npz.
    star_to_obj : dict[int, int]
        Mapping from star_id to obj_id (from star_night).
    match_radius_deg : float
        Radius in degrees for q3c position matching (for stars not in star_to_obj).
    n_frames : int
        Expected number of frames for validation.

    Returns
    -------
    MembersLoadReport
        Counts of inserted rows.

    Raises
    ------
    NightLoadError
        If product.n_frames != n_frames.
    """
    if product.n_frames != n_frames:
        msg = f"product.n_frames {product.n_frames} != night.n_frames {n_frames}"
        raise NightLoadError(msg)

    # --- Delete existing members for this night ---
    cur.execute("DELETE FROM relphot.night_tile WHERE night_id = %s", (night_id,))

    # --- night_tile ---
    n_tiles = len(product.tile_xmin)
    n_tile_lc = 0

    with cur.copy(
        "COPY relphot.night_tile "
        "(night_id, tile, x_min, x_max, y_min, y_max, n_core, n_extended, "
        "n_ref_stars, ref_aperture, best_apertures) FROM STDIN"
    ) as copy:
        for t in range(n_tiles):
            n_ref_stars = int(product.ref_offsets[t + 1] - product.ref_offsets[t])
            ref_aper = product.ref_aper if product.ref_aper >= 0 else None

            # Find best_apertures for this tile from used_pairs
            best_apertures = []
            for pair_t, pair_a in product.used_pairs:
                if pair_t == t:
                    best_apertures.append(int(pair_a))

            copy.write_row(
                (
                    night_id, int(t),
                    float(product.tile_xmin[t]) if np.isfinite(product.tile_xmin[t]) else None,
                    float(product.tile_xmax[t]) if np.isfinite(product.tile_xmax[t]) else None,
                    float(product.tile_ymin[t]) if np.isfinite(product.tile_ymin[t]) else None,
                    float(product.tile_ymax[t]) if np.isfinite(product.tile_ymax[t]) else None,
                    int(product.tile_n_core[t]),
                    int(product.tile_n_extended[t]),
                    n_ref_stars,
                    ref_aper,
                    best_apertures,
                )
            )

    # --- tile_lc: one row for EVERY (tile, aperture) pair ---
    # Build a lookup for used_pairs to get n_comp
    used_pairs_dict = {}
    for pair_idx, (t, a) in enumerate(product.used_pairs):
        used_pairs_dict[(int(t), int(a))] = pair_idx

    with cur.copy(
        "COPY relphot.tile_lc "
        "(night_id, tile, aperture, ref_flux, ref_flux_err, ens_flux, ens_flux_err, "
        "n_ensemble, n_comp, n_rounds) FROM STDIN"
    ) as copy:
        for t in range(n_tiles):
            for a in range(product.n_aper):
                # ref_flux and ref_flux_err from product
                ref_flux = [
                    float(v) if np.isfinite(v) else float('nan')
                    for v in product.tile_R[t, :, a]
                ]
                ref_flux_err = [
                    float(v) if np.isfinite(v) else float('nan')
                    for v in product.tile_sigma_R[t, :, a]
                ]

                ens_flux = None
                ens_flux_err = None
                if product.tile_ens is not None:
                    ens_flux = [
                        float(v) if np.isfinite(v) else float('nan')
                        for v in product.tile_ens[t, :, a]
                    ]
                    ens_flux_err = [
                        float(v) if np.isfinite(v) else float('nan')
                        for v in product.tile_sigma_ens[t, :, a]
                    ]

                n_ensemble = (
                    int(product.tile_n_ensemble[t, a])
                    if product.tile_n_ensemble is not None
                    else None
                )
                # n_comp = number of comparison members for this (t, a)
                # 0 for unused pairs, otherwise from comp_offsets
                pair_key = (int(t), int(a))
                if pair_key in used_pairs_dict:
                    pair_idx = used_pairs_dict[pair_key]
                    n_comp = int(
                        product.comp_offsets[pair_idx + 1]
                        - product.comp_offsets[pair_idx]
                    )
                else:
                    n_comp = 0

                n_rounds = (
                    int(product.tile_n_rounds[t, a]) if product.tile_n_rounds is not None else None
                )

                copy.write_row(
                    (
                        night_id, int(t), int(a),
                        ref_flux, ref_flux_err,
                        ens_flux, ens_flux_err,
                        n_ensemble, n_comp, n_rounds,
                    )
                )
                n_tile_lc += 1

    # --- Build star_to_obj mapping for members not in star_night ---
    # Get unique star IDs from members that aren't already in star_to_obj
    ref_stars = set()
    for i in range(len(product.ref_offsets) - 1):
        start, end = product.ref_offsets[i], product.ref_offsets[i + 1]
        for j in range(start, end):
            ref_stars.add(int(product.ref_star[j]))

    comp_stars = set()
    for i in range(len(product.comp_offsets) - 1):
        start, end = product.comp_offsets[i], product.comp_offsets[i + 1]
        for j in range(start, end):
            comp_stars.add(int(product.comp_star[j]))

    all_member_stars = ref_stars | comp_stars
    unmatched_stars = [s for s in all_member_stars if s not in star_to_obj]

    members_star_to_obj = dict(star_to_obj)

    if unmatched_stars:
        # Create temp table and run q3c query
        cur.execute(
            "CREATE TEMP TABLE tmp_member_stars "
            "(star_id integer, ra double precision, dec double precision) "
            "ON COMMIT DROP"
        )

        with cur.copy("COPY tmp_member_stars (star_id, ra, dec) FROM STDIN") as copy:
            for sid in sorted(unmatched_stars):
                idx = np.where(product.ref_star == sid)[0]
                if len(idx) == 0:
                    idx = np.where(product.comp_star == sid)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    if sid in ref_stars:
                        ra = float(product.ref_ra[idx])
                        dec = float(product.ref_dec[idx])
                    else:
                        ra = float(product.comp_ra[idx])
                        dec = float(product.comp_dec[idx])
                    copy.write_row((int(sid), ra, dec))

        cur.execute(
            """
            SELECT t.star_id, o.obj_id
            FROM tmp_member_stars t
            LEFT JOIN LATERAL (
                SELECT obj_id
                FROM relphot.object o2
                WHERE q3c_join(t.ra, t.dec, o2.ra, o2.dec, %s)
                ORDER BY q3c_dist(t.ra, t.dec, o2.ra, o2.dec) ASC
                LIMIT 1
            ) o ON true
            """,
            (match_radius_deg,),
        )

        for star_id, obj_id in cur.fetchall():
            if obj_id is not None:
                members_star_to_obj[int(star_id)] = int(obj_id)

    # --- reference_member ---
    n_reference_members = 0
    with cur.copy(
        "COPY relphot.reference_member "
        "(night_id, tile, star_id, obj_id, ra, dec, mag, weight, in_core) FROM STDIN"
    ) as copy:
        for t in range(n_tiles):
            start, end = product.ref_offsets[t], product.ref_offsets[t + 1]
            for j in range(start, end):
                star_id = int(product.ref_star[j])
                obj_id = members_star_to_obj.get(star_id)
                ra_v = product.ref_ra[j]
                ra = float(ra_v) if np.isfinite(ra_v) else None
                dec_v = product.ref_dec[j]
                dec = float(dec_v) if np.isfinite(dec_v) else None
                mag_v = product.ref_mag[j]
                mag = float(mag_v) if np.isfinite(mag_v) else None
                weight_v = product.ref_weight[j]
                weight = float(weight_v) if np.isfinite(weight_v) else None
                in_core = bool(product.ref_in_core[j])

                copy.write_row(
                    (night_id, int(t), star_id, obj_id, ra, dec, mag, weight, in_core)
                )
                n_reference_members += 1

    # --- comparison_member ---
    n_comparison_members = 0
    with cur.copy(
        "COPY relphot.comparison_member "
        "(night_id, tile, aperture, star_id, obj_id, ra, dec, mag, weight, "
        "n_clipped, clipped_frames, norm_flux) FROM STDIN"
    ) as copy:
        for pair_idx, (t, a) in enumerate(product.used_pairs):
            start, end = product.comp_offsets[pair_idx], product.comp_offsets[pair_idx + 1]
            for j in range(start, end):
                star_id = int(product.comp_star[j])
                obj_id = members_star_to_obj.get(star_id)
                ra_v = product.comp_ra[j]
                ra = float(ra_v) if np.isfinite(ra_v) else None
                dec_v = product.comp_dec[j]
                dec = float(dec_v) if np.isfinite(dec_v) else None
                mag_v = product.comp_mag[j]
                mag = float(mag_v) if np.isfinite(mag_v) else None
                weight_v = product.comp_weight[j]
                weight = float(weight_v) if np.isfinite(weight_v) else None
                n_clipped = int(product.comp_n_clipped[j])
                clip_start = product.comp_clip_offsets[j]
                clip_end = product.comp_clip_offsets[j + 1]
                clipped_frames = [
                    int(f) for f in product.comp_clip_frames[clip_start:clip_end]
                ]
                norm_flux = [
                    float(v) if np.isfinite(v) else float('nan')
                    for v in product.comp_norm_flux[j, :]
                ]

                copy.write_row(
                    (
                        night_id, int(t), int(a), star_id, obj_id, ra, dec, mag, weight,
                        n_clipped, clipped_frames, norm_flux,
                    )
                )
                n_comparison_members += 1

    # Count n_linked and n_unlinked from member rows with obj_id not null / null
    n_linked = n_reference_members + n_comparison_members  # Will be refined below
    n_unlinked = 0

    # Count actual linked/unlinked by querying the inserted members
    cur.execute(
        "SELECT COUNT(*) FROM relphot.reference_member "
        "WHERE night_id = %s AND obj_id IS NOT NULL",
        (night_id,),
    )
    ref_linked = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM relphot.reference_member "
        "WHERE night_id = %s AND obj_id IS NULL",
        (night_id,),
    )
    ref_unlinked = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM relphot.comparison_member "
        "WHERE night_id = %s AND obj_id IS NOT NULL",
        (night_id,),
    )
    comp_linked = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM relphot.comparison_member "
        "WHERE night_id = %s AND obj_id IS NULL",
        (night_id,),
    )
    comp_unlinked = cur.fetchone()[0]

    n_linked = ref_linked + comp_linked
    n_unlinked = ref_unlinked + comp_unlinked

    return MembersLoadReport(
        night_id=night_id,
        n_tiles=n_tiles,
        n_tile_lc=n_tile_lc,
        n_reference_members=n_reference_members,
        n_comparison_members=n_comparison_members,
        n_linked=n_linked,
        n_unlinked=n_unlinked,
        elapsed_s=0.0,  # Will be set by caller
    )


def load_members(
    conn: psycopg.Connection,
    night_dir: Path | str,
    *,
    lc_stem: str | None = None,
) -> MembersLoadReport:
    """Load members for a night already loaded into the database.

    Parameters
    ----------
    conn : psycopg.Connection
        Database connection.
    night_dir : Path | str
        Night directory containing _members.npz product.
    lc_stem : str | None, optional
        Light curve stem. If None, discovered automatically.

    Returns
    -------
    MembersLoadReport
        Counts of loaded members.

    Raises
    ------
    NightLoadError
        If the night has not been loaded, or if _members.npz is not found.
    """
    t0 = time.monotonic()
    night_dir = Path(night_dir).resolve()

    # Discover lc files (for members path discovery)
    _discover_lc_files(night_dir, lc_stem)

    # Get night_id from database
    with conn.cursor() as cur:
        cur.execute(
            "SELECT night_id, n_frames FROM relphot.night WHERE source_dir = %s",
            (str(night_dir),),
        )
        row = cur.fetchone()
        if row is None:
            msg = f"night not loaded; run 'relphot db load-night' first for {night_dir}"
            raise NightLoadError(msg)
        night_id, n_frames = row

        # Load star_to_obj from star_night
        cur.execute(
            "SELECT star_id, obj_id FROM relphot.star_night WHERE night_id = %s",
            (night_id,),
        )
        star_to_obj = {int(sid): int(oid) for sid, oid in cur.fetchall()}

    # Discover members file
    from relphot.db.load_night import _discover_lc_files as discover_lc
    starstats_path, _, _ = discover_lc(night_dir, lc_stem)
    stem = starstats_path.name[: -len("_starstats.parquet")]
    members_path = night_dir / "lc" / f"{stem}_members.npz"

    if not members_path.is_file():
        msg = f"no such file: {members_path}"
        raise NightLoadError(msg)

    # Load members product
    product = load_members_npz(members_path)

    # Insert into database
    with conn.cursor() as cur:
        report = insert_members(
            cur,
            night_id,
            product,
            star_to_obj,
            match_radius_deg=1.0 / 3600.0,  # 1 arcsec
            n_frames=n_frames,
        )

    elapsed = time.monotonic() - t0
    report.elapsed_s = elapsed
    conn.commit()

    return report
