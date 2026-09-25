"""Cross-match per-frame catalogues into one master star list.

:func:`select_master` picks the master frame (most sources among frames with
good seeing); :func:`match_night` matches every other frame to it with a
:class:`~scipy.spatial.cKDTree` on 3-D sky unit vectors, growing the master
list with each frame's unmatched sources, then assembles dense
``[star, frame(, aper)]`` arrays with NaN/-1 for missing epochs and applies
the minimum-presence cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from relphot.exceptions import MatchError

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.ingest import FrameCatalog, FrameMeta

logger = logging.getLogger(__name__)

__all__ = ["MatchReport", "MatchedNight", "match_night", "select_master"]


@dataclass(frozen=True, slots=True)
class MatchReport:
    """Match outcome for one frame against the (then-current) master list."""

    file: str
    n_sources: int
    n_matched_to_master: int
    match_fraction: float
    median_sep_arcsec: float


@dataclass(slots=True)
class MatchedNight:
    """Dense star x frame(x aper) arrays for one night, after the presence cut.

    ``flux``/``fluxerr`` are ``(n_stars, n_frames, n_aper)``; ``fwhm``/
    ``snr``/``background``/``flags`` are ``(n_stars, n_frames)``. Missing
    epochs are NaN, except ``flags`` which uses ``-1`` (a valid FLAGS value
    is always >= 0). ``ra``/``dec``/``x``/``y`` are ``(n_stars,)`` master
    positions; ``frame_x``/``frame_y`` are ``(n_stars, n_frames)`` per-frame
    detector positions (NaN where the star is missing in that frame).
    """

    ra: np.ndarray
    dec: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frame_x: np.ndarray
    frame_y: np.ndarray
    flux: np.ndarray
    fluxerr: np.ndarray
    fwhm: np.ndarray
    snr: np.ndarray
    background: np.ndarray
    flags: np.ndarray
    presence: np.ndarray
    frame_meta: list[FrameMeta]
    reports: list[MatchReport]
    master_frame_index: int
    n_stars_before_cut: int
    n_stars_after_cut: int

    @property
    def n_stars(self) -> int:
        return int(self.ra.shape[0])

    @property
    def n_frames(self) -> int:
        return len(self.frame_meta)

    @property
    def n_aper(self) -> int:
        return int(self.flux.shape[2]) if self.flux.ndim == 3 else 0


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors on the sky sphere for an array of RA/Dec in degrees."""
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)])


def select_master(catalogs: list[FrameCatalog]) -> int:
    """Index of the master frame: most sources among frames with good seeing.

    "Good seeing" means median FWHM <= 1.2x the night's median FWHM (frames
    with a non-finite median FWHM are excluded from that comparison). If no
    frame qualifies, every frame is eligible.
    """
    if not catalogs:
        msg = "no catalogues to select a master frame from"
        raise MatchError(msg)

    fwhms = np.array([c.meta.median_fwhm for c in catalogs], dtype=np.float64)
    finite = np.isfinite(fwhms)
    night_median = np.nanmedian(fwhms) if finite.any() else np.nan

    if np.isfinite(night_median):
        threshold = 1.2 * night_median
        eligible = [i for i, ok in enumerate(finite) if ok and fwhms[i] <= threshold]
    else:
        eligible = []
    if not eligible:
        eligible = list(range(len(catalogs)))

    return max(eligible, key=lambda i: catalogs[i].n_sources)


def _resolve_duplicates(
    dist: np.ndarray, idx: np.ndarray, within: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised "closest wins" duplicate resolution.

    Several frame sources may fall within the match radius of the same
    master star; only the closest is kept as matched, the rest are treated
    as unmatched (and become new master stars). Returns
    ``(matched_src, matched_master, matched_dist)``, each aligned by index.
    """
    cand_src = np.nonzero(within)[0]
    if cand_src.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    cand_master = idx[cand_src]
    cand_dist = dist[cand_src]
    order = np.argsort(cand_dist, kind="stable")
    src_sorted = cand_src[order]
    master_sorted = cand_master[order]
    dist_sorted = cand_dist[order]

    # np.unique's "first occurrence" is the first index in the *given* order;
    # since that order is ascending distance, the first occurrence of each
    # master id is exactly its closest claimant.
    uniq_master, first_pos = np.unique(master_sorted, return_index=True)
    matched_src = src_sorted[first_pos]
    matched_dist = dist_sorted[first_pos]
    return matched_src, uniq_master, matched_dist


def match_night(catalogs: list[FrameCatalog], settings: Settings) -> MatchedNight:
    """Match every frame in ``catalogs`` to a master list and build dense arrays."""
    if not catalogs:
        msg = "no catalogues to match"
        raise MatchError(msg)

    n_frames = len(catalogs)
    n_aper_values = {c.n_aper for c in catalogs}
    if len(n_aper_values) != 1:
        msg = f"inconsistent aperture counts across frames: {sorted(n_aper_values)}"
        raise MatchError(msg)
    n_aper = n_aper_values.pop()

    master_idx = select_master(catalogs)
    master_cat = catalogs[master_idx]
    master_wcs = master_cat.meta.wcs
    if master_wcs is None or not master_wcs.has_celestial:
        msg = f"master frame {master_cat.meta.file} has no usable celestial WCS"
        raise MatchError(msg)
    logger.info(
        "master frame: %s (%d sources, FWHM=%.2f px)",
        master_cat.meta.file, master_cat.n_sources, master_cat.meta.median_fwhm,
    )

    radius_rad = np.radians(settings.catalog.match_radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)

    master_ra: list[float] = list(np.asarray(master_cat.ra, dtype=np.float64))
    master_dec: list[float] = list(np.asarray(master_cat.dec, dtype=np.float64))

    frame_star_ids: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_frames)]
    frame_star_ids[master_idx] = np.arange(master_cat.n_sources, dtype=np.int64)

    reports: list[MatchReport | None] = [None] * n_frames
    reports[master_idx] = MatchReport(
        file=str(master_cat.meta.file),
        n_sources=master_cat.n_sources,
        n_matched_to_master=master_cat.n_sources,
        match_fraction=1.0,
        median_sep_arcsec=0.0,
    )

    for f in range(n_frames):
        if f == master_idx:
            continue
        cat = catalogs[f]
        n = cat.n_sources
        ids = np.empty(n, dtype=np.int64)

        if n == 0:
            frame_star_ids[f] = ids
            reports[f] = MatchReport(
                file=str(cat.meta.file), n_sources=0, n_matched_to_master=0,
                match_fraction=0.0, median_sep_arcsec=float("nan"),
            )
            continue

        master_vec = _unit_vectors(np.asarray(master_ra), np.asarray(master_dec))
        tree = cKDTree(master_vec)
        frame_vec = _unit_vectors(cat.ra, cat.dec)
        dist, idx = tree.query(frame_vec, k=1)
        within = dist <= chord_radius

        matched_src, matched_master, matched_dist = _resolve_duplicates(dist, idx, within)

        matched_mask = np.zeros(n, dtype=bool)
        matched_mask[matched_src] = True
        ids[matched_src] = matched_master

        unmatched_src = np.nonzero(~matched_mask)[0]
        n_new = unmatched_src.size
        if n_new:
            new_ids = np.arange(len(master_ra), len(master_ra) + n_new, dtype=np.int64)
            ids[unmatched_src] = new_ids
            master_ra.extend(np.asarray(cat.ra[unmatched_src], dtype=np.float64).tolist())
            master_dec.extend(np.asarray(cat.dec[unmatched_src], dtype=np.float64).tolist())

        frame_star_ids[f] = ids

        theta = 2.0 * np.arcsin(np.clip(matched_dist / 2.0, 0.0, 1.0))
        seps_arcsec = np.degrees(theta) * 3600.0
        n_matched = int(matched_src.size)
        reports[f] = MatchReport(
            file=str(cat.meta.file),
            n_sources=n,
            n_matched_to_master=n_matched,
            match_fraction=n_matched / n,
            median_sep_arcsec=float(np.median(seps_arcsec)) if n_matched else float("nan"),
        )
        logger.info(
            "matched %s: %d/%d (%.1f%%), median sep %.3f arcsec, %d new stars",
            cat.meta.file, n_matched, n, 100.0 * n_matched / n,
            reports[f].median_sep_arcsec, n_new,
        )

    n_stars_total = len(master_ra)
    master_ra_arr = np.asarray(master_ra, dtype=np.float64)
    master_dec_arr = np.asarray(master_dec, dtype=np.float64)

    # Master x/y for every star (including the master frame's own) come from
    # projecting RA/Dec through the master frame's own WCS -- never a
    # source's native X_IMAGE/Y_IMAGE from whichever frame first saw it --
    # so every star's master pixel position is on one consistent grid.
    # +1 converts astropy's 0-based world_to_pixel output to the 1-based
    # FITS pixel convention the catalogues themselves use (X_IMAGE/Y_IMAGE).
    sky = SkyCoord(ra=master_ra_arr * u.deg, dec=master_dec_arr * u.deg)
    px, py = master_wcs.world_to_pixel(sky)
    master_x = np.asarray(px, dtype=np.float64) + 1.0
    master_y = np.asarray(py, dtype=np.float64) + 1.0

    flux = np.full((n_stars_total, n_frames, n_aper), np.nan, dtype=np.float32)
    fluxerr = np.full((n_stars_total, n_frames, n_aper), np.nan, dtype=np.float32)
    frame_x = np.full((n_stars_total, n_frames), np.nan, dtype=np.float64)
    frame_y = np.full((n_stars_total, n_frames), np.nan, dtype=np.float64)
    fwhm = np.full((n_stars_total, n_frames), np.nan, dtype=np.float32)
    snr = np.full((n_stars_total, n_frames), np.nan, dtype=np.float32)
    background = np.full((n_stars_total, n_frames), np.nan, dtype=np.float32)
    flags = np.full((n_stars_total, n_frames), -1, dtype=np.int32)

    for f, cat in enumerate(catalogs):
        ids = frame_star_ids[f]
        if ids.size == 0:
            continue
        flux[ids, f, :] = cat.flux
        fluxerr[ids, f, :] = cat.fluxerr
        frame_x[ids, f] = cat.x
        frame_y[ids, f] = cat.y
        fwhm[ids, f] = cat.fwhm
        snr[ids, f] = cat.snr
        background[ids, f] = cat.background
        flags[ids, f] = cat.flags

    presence = np.count_nonzero(flags != -1, axis=1) / n_frames
    n_before_cut = n_stars_total
    keep = presence >= settings.catalog.min_presence
    n_after_cut = int(np.count_nonzero(keep))
    logger.info(
        "presence cut (>= %.0f%%): %d -> %d stars",
        100.0 * settings.catalog.min_presence, n_before_cut, n_after_cut,
    )

    return MatchedNight(
        ra=master_ra_arr[keep],
        dec=master_dec_arr[keep],
        x=master_x[keep],
        y=master_y[keep],
        frame_x=frame_x[keep],
        frame_y=frame_y[keep],
        flux=flux[keep],
        fluxerr=fluxerr[keep],
        fwhm=fwhm[keep],
        snr=snr[keep],
        background=background[keep],
        flags=flags[keep],
        presence=presence[keep],
        frame_meta=[c.meta for c in catalogs],
        reports=[r for r in reports if r is not None],
        master_frame_index=master_idx,
        n_stars_before_cut=n_before_cut,
        n_stars_after_cut=n_after_cut,
    )
