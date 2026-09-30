"""Reference and comparison member traceability.

Builds detailed records of which stars contributed to each tile's reference
and comparison ensembles, with their weights, positions, magnitudes, and
normalised light curves.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from relphot.exceptions import MembersError, RelphotError
from relphot.numeric import nanmedian_quiet, weighted_clipped_combine
from relphot.reference import reference_member_weights

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import Settings
    from relphot.match import MatchedNight
    from relphot.reference import ReferenceResult
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "MembersProduct",
    "build_members",
    "load_members_npz",
    "members_path_for",
    "recover_reference_stars",
    "save_members_npz",
]


@dataclass(slots=True)
class MembersProduct:
    """Reference and comparison members with their contributions to tile ensembles.

    Attributes
    ----------
    n_frames : int
        Total number of frames in the night.
    n_aper : int
        Number of apertures.
    ref_aper : int
        Aperture index used to build reference (used for reference members only).
    meta : dict
        Metadata dictionary with 'version', 'ref_method', 'comp_method'.
    tile_xmin, tile_xmax, tile_ymin, tile_ymax : np.ndarray
        Tile boundaries (T,), float64.
    tile_n_core : np.ndarray
        Number of core stars per tile (T,), int64.
    tile_n_extended : np.ndarray
        Number of extended stars per tile (T,), int64.
    ref_offsets : np.ndarray
        Offsets into flat ref_star arrays, (T+1,) int64.
    ref_star : np.ndarray
        Flat array of reference star indices, int64.
    ref_ra, ref_dec : np.ndarray
        RA/Dec of reference members, float64.
    ref_mag : np.ndarray
        Magnitude of reference members, float32.
    ref_weight : np.ndarray
        Normalised weight of reference members, float32.
    ref_in_core : np.ndarray
        Whether reference member is in core tile, bool.
    tile_R, tile_sigma_R : np.ndarray
        Reference light curves per tile and aperture (T,F,A), float64.
    tile_ens, tile_sigma_ens : np.ndarray
        Comparison ensemble per tile and aperture (T,F,A), float64.
    tile_n_ensemble : np.ndarray
        Number of comparison members per (tile, aperture) (T,A), int64.
    tile_n_rounds : np.ndarray
        Number of selection rounds per (tile, aperture) (T,A), int16.
    used_pairs : np.ndarray
        (tile, aperture) pairs with stored comparison members, (P,2) int32.
    comp_offsets : np.ndarray
        Offsets into flat comp_star arrays per used pair, (P+1,) int64.
    comp_star : np.ndarray
        Flat array of comparison star indices, int64.
    comp_ra, comp_dec : np.ndarray
        RA/Dec of comparison members, float64.
    comp_mag : np.ndarray
        Magnitude of comparison members, float32.
    comp_weight : np.ndarray
        Normalised weight of comparison members, float32.
    comp_n_clipped : np.ndarray
        Number of frames where member was clipped, int16.
    comp_clip_offsets : np.ndarray
        Offsets into comp_clip_frames per comparison member, (M+1,) int64.
    comp_clip_frames : np.ndarray
        Frame indices where members were clipped, int16.
    comp_norm_flux : np.ndarray
        Normalised comparison fluxes (M,F), float32. NaN for clipped frames.
    """

    n_frames: int
    n_aper: int
    ref_aper: int
    meta: dict
    tile_xmin: np.ndarray
    tile_xmax: np.ndarray
    tile_ymin: np.ndarray
    tile_ymax: np.ndarray
    tile_n_core: np.ndarray
    tile_n_extended: np.ndarray
    ref_offsets: np.ndarray
    ref_star: np.ndarray
    ref_ra: np.ndarray
    ref_dec: np.ndarray
    ref_mag: np.ndarray
    ref_weight: np.ndarray
    ref_in_core: np.ndarray
    tile_R: np.ndarray
    tile_sigma_R: np.ndarray
    tile_ens: np.ndarray
    tile_sigma_ens: np.ndarray
    tile_n_ensemble: np.ndarray
    tile_n_rounds: np.ndarray
    used_pairs: np.ndarray
    comp_offsets: np.ndarray
    comp_star: np.ndarray
    comp_ra: np.ndarray
    comp_dec: np.ndarray
    comp_mag: np.ndarray
    comp_weight: np.ndarray
    comp_n_clipped: np.ndarray
    comp_clip_offsets: np.ndarray
    comp_clip_frames: np.ndarray
    comp_norm_flux: np.ndarray


def members_path_for(lc_npz_path: Path | str) -> Path:
    """Return the members .npz path for a given light-curves .npz path.

    Parameters
    ----------
    lc_npz_path : Path or str
        Path to lc/night_lc.npz or similar.

    Returns
    -------
    Path
        Path to lc/night_lc_members.npz.
    """
    lc_path = Path(lc_npz_path)
    stem = lc_path.name
    if stem.endswith("_lightcurves.npz"):
        new_stem = stem[:-len("_lightcurves.npz")] + "_members.npz"
    else:
        new_stem = stem[:-4] + "_members.npz"
    return lc_path.with_name(new_stem)


def save_members_npz(product: MembersProduct, path: Path | str) -> None:
    """Save MembersProduct as uncompressed .npz.

    Parameters
    ----------
    product : MembersProduct
        The members product to save.
    path : Path or str
        Output .npz file path.
    """
    path = Path(path)
    meta_json = json.dumps(product.meta)

    np.savez(
        path,
        n_frames=np.int64(product.n_frames),
        n_aper=np.int64(product.n_aper),
        ref_aper=np.int64(product.ref_aper),
        meta_json=meta_json,
        tile_xmin=product.tile_xmin,
        tile_xmax=product.tile_xmax,
        tile_ymin=product.tile_ymin,
        tile_ymax=product.tile_ymax,
        tile_n_core=product.tile_n_core,
        tile_n_extended=product.tile_n_extended,
        ref_offsets=product.ref_offsets,
        ref_star=product.ref_star,
        ref_ra=product.ref_ra,
        ref_dec=product.ref_dec,
        ref_mag=product.ref_mag,
        ref_weight=product.ref_weight,
        ref_in_core=product.ref_in_core,
        tile_R=product.tile_R,
        tile_sigma_R=product.tile_sigma_R,
        tile_ens=product.tile_ens,
        tile_sigma_ens=product.tile_sigma_ens,
        tile_n_ensemble=product.tile_n_ensemble,
        tile_n_rounds=product.tile_n_rounds,
        used_pairs=product.used_pairs,
        comp_offsets=product.comp_offsets,
        comp_star=product.comp_star,
        comp_ra=product.comp_ra,
        comp_dec=product.comp_dec,
        comp_mag=product.comp_mag,
        comp_weight=product.comp_weight,
        comp_n_clipped=product.comp_n_clipped,
        comp_clip_offsets=product.comp_clip_offsets,
        comp_clip_frames=product.comp_clip_frames,
        comp_norm_flux=product.comp_norm_flux,
    )
    logger.info("wrote %s", path)


def load_members_npz(path: Path | str) -> MembersProduct:
    """Load MembersProduct from .npz.

    Parameters
    ----------
    path : Path or str
        Path to members .npz file.

    Returns
    -------
    MembersProduct
        The loaded product.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta_json']))
        product = MembersProduct(
            n_frames=int(data['n_frames']),
            n_aper=int(data['n_aper']),
            ref_aper=int(data['ref_aper']),
            meta=meta,
            tile_xmin=data['tile_xmin'],
            tile_xmax=data['tile_xmax'],
            tile_ymin=data['tile_ymin'],
            tile_ymax=data['tile_ymax'],
            tile_n_core=data['tile_n_core'],
            tile_n_extended=data['tile_n_extended'],
            ref_offsets=data['ref_offsets'],
            ref_star=data['ref_star'],
            ref_ra=data['ref_ra'],
            ref_dec=data['ref_dec'],
            ref_mag=data['ref_mag'],
            ref_weight=data['ref_weight'],
            ref_in_core=data['ref_in_core'],
            tile_R=data['tile_R'],
            tile_sigma_R=data['tile_sigma_R'],
            tile_ens=data['tile_ens'],
            tile_sigma_ens=data['tile_sigma_ens'],
            tile_n_ensemble=data['tile_n_ensemble'],
            tile_n_rounds=data['tile_n_rounds'],
            used_pairs=data['used_pairs'],
            comp_offsets=data['comp_offsets'],
            comp_star=data['comp_star'],
            comp_ra=data['comp_ra'],
            comp_dec=data['comp_dec'],
            comp_mag=data['comp_mag'],
            comp_weight=data['comp_weight'],
            comp_n_clipped=data['comp_n_clipped'],
            comp_clip_offsets=data['comp_clip_offsets'],
            comp_clip_frames=data['comp_clip_frames'],
            comp_norm_flux=data['comp_norm_flux'],
        )
    return product


def recover_reference_stars(
    night: MatchedNight,
    tilemap: TileMap,
    reference_result: ReferenceResult,
    settings: Settings,
    *,
    aper: int | None = None,
    use_variables: str | None = None,
    star_eligible: np.ndarray | None = None,
) -> tuple[list[np.ndarray], int]:
    """Recover reference star set and aperture from an existing reference result.

    Tries variable-mask options and apertures until one matches the reference
    result's R and frame_kept. Raises MembersError if no match found.

    Parameters
    ----------
    night : MatchedNight
        The matched night.
    tilemap : TileMap
        The tile map.
    reference_result : ReferenceResult
        The reference result to match.
    settings : Settings
        The settings.
    aper : int, optional
        If provided, only try this aperture. Otherwise tries default then all.
    use_variables : {'auto', 'yes', 'no'}, optional
        If 'auto', tries both. Otherwise uses only the specified option.
    star_eligible : np.ndarray, optional
        Star eligibility mask, (n_stars,) bool. If None, computes from settings.

    Returns
    -------
    tile_stars : list[np.ndarray]
        Reference star set per tile.
    ref_aper : int
        The aperture used.

    Raises
    ------
    MembersError
        If no combination matches the reference result.
    """
    from relphot.reference import (
        build_references,
        select_candidates,
        select_reference_frames_and_stars,
    )
    from relphot.variables import flag_known_variables

    # Determine apertures to try
    if aper is not None:
        apertures = [aper]
    else:
        n_aper = reference_result.n_aper
        default_aper = 1 if n_aper >= 2 else 0
        apertures = [default_aper] + [
            a for a in range(n_aper) if a != default_aper
        ]

    # Determine variable masks to try
    variable_masks = []
    if use_variables == 'auto':
        variable_masks.append(np.zeros(night.n_stars, dtype=bool))
        variable_masks.append(flag_known_variables(night, settings))
    elif use_variables == 'yes':
        variable_masks.append(flag_known_variables(night, settings))
    elif use_variables == 'no':
        variable_masks.append(np.zeros(night.n_stars, dtype=bool))
    else:
        variable_masks.append(np.zeros(night.n_stars, dtype=bool))
        variable_masks.append(flag_known_variables(night, settings))

    # Try each combination
    attempted = []
    for variable_mask in variable_masks:
        for a in apertures:
            try:
                candidates = select_candidates(night, variable_mask, settings, a, star_eligible)
                frame_sel = select_reference_frames_and_stars(
                    night, tilemap, candidates, settings, a
                )
                result2 = build_references(night, tilemap, frame_sel, settings)

                # Check match
                if np.array_equal(frame_sel.frame_kept, reference_result.frame_kept) and \
                   np.allclose(result2.R, reference_result.R, rtol=1e-12, atol=0, equal_nan=True):
                    return frame_sel.tile_stars, a

                attempted.append((a, type(variable_mask).__name__, "no match"))
            except RelphotError as e:
                attempted.append((a, type(variable_mask).__name__, str(e)))

    # No match found
    msg_parts = [f"aperture {a}: {reason}" for a, _, reason in attempted]
    raise MembersError(f"cannot recover reference stars; tried {', '.join(msg_parts)}")


def build_members(
    night: MatchedNight,
    tilemap: TileMap,
    reference_result: ReferenceResult,
    tile_stars: list[np.ndarray],
    ref_aper: int,
    comparison_result: ComparisonResult,
    best_aper_per_tile: np.ndarray,
    settings: Settings,
    *,
    apertures: str = "used",
) -> MembersProduct:
    """Build reference and comparison member records.

    Parameters
    ----------
    night : MatchedNight
        The matched night.
    tilemap : TileMap
        The tile map.
    reference_result : ReferenceResult
        The reference result.
    tile_stars : list[np.ndarray]
        Reference star sets per tile.
    ref_aper : int
        Aperture used for reference.
    comparison_result : ComparisonResult
        The comparison result.
    best_aper_per_tile : np.ndarray
        Best aperture per (tile, bin), shape (n_tiles, n_bins).
    settings : Settings
        The settings.
    apertures : {'used', 'all'}, optional
        'used': only store pairs in best_aper_per_tile. 'all': every pair with
        n_comparison > 0.

    Returns
    -------
    MembersProduct
        The members product.

    Raises
    ------
    MembersError
        If ensemble reconstruction fails for any tile/aperture pair.
    """
    n_tiles = tilemap.n_tiles
    n_frames = night.n_frames
    n_aper = night.n_aper
    frame_kept = reference_result.frame_kept
    kept_idx = np.nonzero(frame_kept)[0]

    # Collect reference members
    ref_star_list = []
    ref_ra_list = []
    ref_dec_list = []
    ref_mag_list = []
    ref_weight_list = []
    ref_in_core_list = []
    ref_offsets = [0]

    for t in range(n_tiles):
        s_t = tile_stars[t]
        if len(s_t) < 2:
            ref_offsets.append(ref_offsets[-1])
            continue

        # Compute reference member weights
        f_s = night.flux[s_t, :, ref_aper][:, frame_kept].astype(np.float64)
        sigma_s = night.fluxerr[s_t, :, ref_aper][:, frame_kept].astype(np.float64)
        bjd_kept = np.array([night.frame_meta[i].bjd_tdb for i in kept_idx], dtype=np.float64)

        method = reference_result.method
        if method == "weighted_fixed_mean":
            baseline_i, w_i = reference_member_weights(f_s, sigma_s, bjd_kept, settings.reference)
            w_i = w_i / w_i.sum()  # Normalise to sum to 1
        else:  # median_fixed
            baseline_i = nanmedian_quiet(f_s, axis=1)
            baseline_i = np.where(np.isfinite(baseline_i) & (baseline_i > 0), baseline_i, np.nan)
            w_i = np.ones(len(s_t)) / len(s_t)

        # Compute magnitudes and positions
        mag_i = np.where(
            np.isfinite(baseline_i) & (baseline_i > 0),
            -2.5 * np.log10(baseline_i),
            np.nan
        )
        ra_i = night.ra[s_t]
        dec_i = night.dec[s_t]
        in_core_i = tilemap.core_tile[s_t] == t

        ref_star_list.append(s_t)
        ref_ra_list.append(ra_i)
        ref_dec_list.append(dec_i)
        ref_mag_list.append(mag_i.astype(np.float32))
        ref_weight_list.append(w_i.astype(np.float32))
        ref_in_core_list.append(in_core_i)
        ref_offsets.append(ref_offsets[-1] + len(s_t))

    # Flatten reference members
    if ref_star_list:
        ref_star = np.concatenate(ref_star_list).astype(np.int64)
        ref_ra = np.concatenate(ref_ra_list)
        ref_dec = np.concatenate(ref_dec_list)
        ref_mag = np.concatenate(ref_mag_list)
        ref_weight = np.concatenate(ref_weight_list)
        ref_in_core = np.concatenate(ref_in_core_list)
    else:
        ref_star = np.empty(0, dtype=np.int64)
        ref_ra = np.empty(0, dtype=np.float64)
        ref_dec = np.empty(0, dtype=np.float64)
        ref_mag = np.empty(0, dtype=np.float32)
        ref_weight = np.empty(0, dtype=np.float32)
        ref_in_core = np.empty(0, dtype=bool)

    ref_offsets = np.array(ref_offsets, dtype=np.int64)

    # Determine used (tile, aperture) pairs
    if apertures == "used":
        used_pairs_set = set()
        for t in range(n_tiles):
            best_a_in_tile = best_aper_per_tile[t, :]
            for a in best_a_in_tile:
                if a >= 0:
                    used_pairs_set.add((t, int(a)))
        used_pairs_list = sorted(used_pairs_set)
    else:  # "all"
        used_pairs_list = []
        for t in range(n_tiles):
            for a in range(n_aper):
                if comparison_result.n_comparison[t, a] > 0:
                    used_pairs_list.append((t, a))

    if used_pairs_list:
        used_pairs = np.array(used_pairs_list, dtype=np.int32)
    else:
        used_pairs = np.empty((0, 2), dtype=np.int32)

    # Collect comparison members
    comp_star_list = []
    comp_ra_list = []
    comp_dec_list = []
    comp_mag_list = []
    comp_weight_list = []
    comp_n_clipped_list = []
    comp_clip_offset_list = [0]
    comp_clip_frames_list = []
    comp_norm_flux_list = []
    comp_offsets = [0]

    comp = settings.comparison

    for _pair_idx, (t, a) in enumerate(used_pairs_list):
        # Recompute ensemble for this pair to get member selection
        core_mask = tilemap.core_tile == t
        sel = np.nonzero(comparison_result.mask[:, a] & core_mask)[0]

        if len(sel) == 0:
            comp_offsets.append(comp_offsets[-1])
            continue

        r_sel = reference_result.relative_flux[sel, :, a]
        baseline_sel = nanmedian_quiet(r_sel, axis=1)
        baseline_sel = np.where(
            np.isfinite(baseline_sel) & (baseline_sel > 0), baseline_sel, np.nan
        )

        c = r_sel / baseline_sel[:, None]
        fluxerr_rel = night.fluxerr[sel, :, a] / reference_result.R[t, :, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = baseline_sel[:, None] ** 2 / (fluxerr_rel**2)
        w = np.where(np.isfinite(c) & np.isfinite(w) & (w > 0), w, 0.0)
        c = np.where(np.isfinite(c), c, np.nan)

        if comparison_result.method == "weighted_clipped_mean":
            ens, _sig, _, m = weighted_clipped_combine(c, w, comp.clip_sigma, comp.max_iter, axis=0)
            # Compute weights once per pair: sum_j(where(m, w, 0)) / sum_ij(where(m, w, 0))
            w_used = np.where(m, w, 0.0)
            total_w = w_used.sum()
            use_ensemble_weights = True
        else:  # median
            ens = nanmedian_quiet(c, axis=0)
            m = np.isfinite(c)
            # _sig computed but not stored per spec
            use_ensemble_weights = False
            median_weight = 1.0 / len(sel)

        # Verify ensemble matches
        if not np.allclose(
            ens, comparison_result.ensemble[t, :, a],
            rtol=1e-9, atol=1e-12, equal_nan=True
        ):
            raise MembersError(f"ensemble mismatch for tile {t}, aperture {a}")

        # Extract member info
        for member_idx, star_idx in enumerate(sel):
            baseline = baseline_sel[member_idx]
            if np.isfinite(baseline) and baseline > 0:
                mag_val = -2.5 * np.log10(baseline)
            else:
                mag_val = np.nan

            # Weight per member
            if use_ensemble_weights:
                weight_val = w_used[member_idx].sum() / total_w if total_w > 0 else 0.0
            else:  # median
                weight_val = median_weight

            # Clipped frames: where valid & (w > 0) & ~m
            valid_c = np.isfinite(c[member_idx, :]) & (w[member_idx, :] > 0)
            clipped = np.where(valid_c & ~m[member_idx, :])[0]

            comp_star_list.append(star_idx)
            comp_ra_list.append(night.ra[star_idx])
            comp_dec_list.append(night.dec[star_idx])
            comp_mag_list.append(mag_val)
            comp_weight_list.append(weight_val)
            comp_n_clipped_list.append(len(clipped))
            comp_clip_frames_list.extend(clipped.astype(np.int16))
            comp_clip_offset_list.append(comp_clip_offset_list[-1] + len(clipped))

            # Normalised flux
            norm_f = c[member_idx, :].astype(np.float32)
            comp_norm_flux_list.append(norm_f)

        comp_offsets.append(comp_offsets[-1] + len(sel))

    # Flatten comparison members
    if comp_star_list:
        comp_star = np.array(comp_star_list, dtype=np.int64)
        comp_ra = np.array(comp_ra_list, dtype=np.float64)
        comp_dec = np.array(comp_dec_list, dtype=np.float64)
        comp_mag = np.array(comp_mag_list, dtype=np.float32)
        comp_weight = np.array(comp_weight_list, dtype=np.float32)
        comp_n_clipped = np.array(comp_n_clipped_list, dtype=np.int16)
        if comp_clip_frames_list:
            comp_clip_frames = np.array(comp_clip_frames_list, dtype=np.int16)
        else:
            comp_clip_frames = np.empty(0, dtype=np.int16)
        comp_norm_flux = np.array(comp_norm_flux_list, dtype=np.float32)
    else:
        comp_star = np.empty(0, dtype=np.int64)
        comp_ra = np.empty(0, dtype=np.float64)
        comp_dec = np.empty(0, dtype=np.float64)
        comp_mag = np.empty(0, dtype=np.float32)
        comp_weight = np.empty(0, dtype=np.float32)
        comp_n_clipped = np.empty(0, dtype=np.int16)
        comp_clip_frames = np.empty(0, dtype=np.int16)
        comp_norm_flux = np.empty((0, n_frames), dtype=np.float32)

    comp_offsets = np.array(comp_offsets, dtype=np.int64)
    comp_clip_offset_list = np.array(comp_clip_offset_list, dtype=np.int64)

    # Build product
    product = MembersProduct(
        n_frames=n_frames,
        n_aper=n_aper,
        ref_aper=ref_aper,
        meta={
            "version": 1,
            "ref_method": reference_result.method,
            "comp_method": comparison_result.method,
        },
        tile_xmin=tilemap.xmin,
        tile_xmax=tilemap.xmax,
        tile_ymin=tilemap.ymin,
        tile_ymax=tilemap.ymax,
        tile_n_core=np.array(
            [len(tilemap.core_indices[t]) for t in range(n_tiles)], dtype=np.int64
        ),
        tile_n_extended=np.array(
            [len(tilemap.extended_indices[t]) for t in range(n_tiles)], dtype=np.int64
        ),
        ref_offsets=ref_offsets,
        ref_star=ref_star,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
        ref_mag=ref_mag,
        ref_weight=ref_weight,
        ref_in_core=ref_in_core,
        tile_R=reference_result.R,
        tile_sigma_R=reference_result.sigma_R,
        tile_ens=comparison_result.ensemble,
        tile_sigma_ens=comparison_result.sigma_ensemble,
        tile_n_ensemble=comparison_result.n_comparison,
        tile_n_rounds=comparison_result.n_rounds_used,
        used_pairs=used_pairs,
        comp_offsets=comp_offsets,
        comp_star=comp_star,
        comp_ra=comp_ra,
        comp_dec=comp_dec,
        comp_mag=comp_mag,
        comp_weight=comp_weight,
        comp_n_clipped=comp_n_clipped,
        comp_clip_offsets=comp_clip_offset_list,
        comp_clip_frames=comp_clip_frames,
        comp_norm_flux=comp_norm_flux,
    )

    return product
