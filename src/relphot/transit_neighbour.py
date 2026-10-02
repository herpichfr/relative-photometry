"""Neighbour-shared-event screen: one star's eclipse leaking into a neighbour's aperture.

A real eclipse of star A that falls inside the photometric aperture of a close neighbour B
dims B as well, at the same time and with a (diluted-by-flux-ratio) depth that can even be
larger than A's own, because B is fainter. Both then look like transit candidates, and the
coincident-transit veto (``SHARED_EPOCH``, many stars at one epoch) never sees a pair of
two. :func:`flag_neighbour_shared_events` finds them: for every transit candidate it
measures every other star within ``neighbour_event_radius_arcsec`` at the candidate's own
``(tc, duration)`` window, with the same joint nuisance + box fit the cross-aperture depth
check uses (:func:`relphot.transit_search.depth_at_other_aperture`), and sets the
informational ``NEIGHBOUR_SHARED_EVENT`` bit when the best neighbour dims there at
``neighbour_event_dip_sigma`` or more.

The partner is *measured* in the candidate's window rather than looked up among the other
candidates: a leakage dip is shallower than the source's by the leak fraction and can sit
below ``snr_threshold`` while still being unmistakable at a known epoch. The window is also
the time-coincidence test: a neighbour dip displaced by about one duration or more falls
outside it and is not seen. Only candidates are screened, so the cost is a handful of box
fits per candidate (a few ms each), not a pass over the night.

Which star is the probable source is decided by the absolute flux deficit
``depth * median flux`` in each star's own best aperture: the star that loses more
counts is the one that eclipses, the other one only receives the leaked deficit
(victim depth ~ source depth * source flux * leak fraction / victim flux, leak fraction < 1).
When both members of a pair are candidates each one measured the pair at its own window, which
can disagree for a near tie, so the pair is decided once, at the window of the higher-SNR
member (its timing is the better one), and the other member gets the complementary answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from relphot.numeric import nanmedian_quiet, unit_vectors
from relphot.transit_search import FLAG_NEIGHBOUR_SHARED_EVENT, depth_at_other_aperture

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.cotrend import CotrendResult
    from relphot.match import MatchedNight
    from relphot.tiles import TileMap
    from relphot.transit_search import TransitSearchResult

logger = logging.getLogger(__name__)

__all__ = ["NeighbourSharedResult", "flag_neighbour_shared_events"]


@dataclass(slots=True)
class NeighbourSharedResult:
    """Per-star neighbour-shared-event measurements, ``(n_stars,)``.

    Only transit candidates are screened; every other star keeps the defaults (``partner``
    -1, NaN numbers, ``is_source`` False). For a screened star ``partner`` is the
    ``star_id`` of the neighbour dimming most significantly at this star's window (-1 when
    no neighbour within the radius could be measured), whether or not that dip reached
    the threshold, so the threshold can be re-tuned from the stored numbers; the star is
    flagged exactly when ``dip_sigma >= neighbour_event_dip_sigma``.

    ``sep_arcsec`` is the star-partner separation; ``depth``/``dip_sigma`` the partner's
    depth and depth/error at this star's ``(tc, duration)``; ``dtc_hours`` the partner's own
    best-event tc minus this star's tc (NaN when the partner was not searched);
    ``deficit_ratio`` the partner's flux deficit over this star's (depth * median flux, each
    in its own best aperture); ``is_source`` (set only on a flagged star) True when this
    star's deficit is at least the partner's, i.e. this star is the probable source of the
    shared event, False when the partner is.
    """

    partner: np.ndarray
    sep_arcsec: np.ndarray
    depth: np.ndarray
    dip_sigma: np.ndarray
    dtc_hours: np.ndarray
    deficit_ratio: np.ndarray
    is_source: np.ndarray

    @classmethod
    def empty(cls, n_stars: int) -> NeighbourSharedResult:
        return cls(
            partner=np.full(n_stars, -1, dtype=np.int64),
            sep_arcsec=np.full(n_stars, np.nan),
            depth=np.full(n_stars, np.nan),
            dip_sigma=np.full(n_stars, np.nan),
            dtc_hours=np.full(n_stars, np.nan),
            deficit_ratio=np.full(n_stars, np.nan),
            is_source=np.zeros(n_stars, dtype=bool),
        )


def flag_neighbour_shared_events(
    night: MatchedNight,
    tilemap: TileMap,
    cotrend_result: CotrendResult,
    lc: np.ndarray,
    lc_err: np.ndarray,
    epoch_ok: np.ndarray,
    frame_kept: np.ndarray,
    star_best_aper: np.ndarray,
    transit_result: TransitSearchResult,
    settings: Settings,
) -> NeighbourSharedResult:
    """Set ``FLAG_NEIGHBOUR_SHARED_EVENT`` in place on candidates a neighbour also dips with.

    Informational exactly like ``ON_VARIABLE``: the bit is not in ``HARD_REJECT_FLAGS``,
    never changes the tier or ``transit_result.candidate``. ``lc``/``lc_err``/``epoch_ok``
    are the decorrelated light curves given to :func:`relphot.transit_search.search_transits`
    and ``transit_result`` is its result (its ``frame_error_scale`` rescales every
    partner's errors exactly as for the search). Returns the raw per-star measurements.
    """
    search = settings.search
    n_stars = night.n_stars
    result = NeighbourSharedResult.empty(n_stars)
    cand = np.nonzero(transit_result.candidate)[0]
    if not search.neighbour_event_enabled or cand.size == 0:
        return result

    search = replace(
        search, min_epochs=search.effective_min_epochs(int(np.count_nonzero(frame_kept)))
    )
    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    core_tile = tilemap.core_tile

    ids = np.nonzero(np.isfinite(night.ra) & np.isfinite(night.dec))[0]
    row_of = np.full(n_stars, -1, dtype=np.int64)
    row_of[ids] = np.arange(ids.size)
    vec = unit_vectors(night.ra[ids], night.dec[ids])
    tree = cKDTree(vec)
    radius_rad = np.radians(search.neighbour_event_radius_arcsec / 3600.0)
    chord = 2.0 * np.sin(radius_rad / 2.0)

    def median_flux(k: int) -> float:
        a = int(star_best_aper[k])
        return float(nanmedian_quiet(night.flux[k, :, a])) if a >= 0 else np.nan

    flagged: list[int] = []
    for i in cand:
        if row_of[i] < 0:
            continue
        tc_i = float(transit_result.tc[i])
        dur_i = float(transit_result.duration[i])
        best_sigma = -np.inf
        best = None
        for jj in tree.query_ball_point(vec[row_of[i]], chord):
            j = int(ids[jj])
            t_tile = int(core_tile[j])
            a = int(star_best_aper[j])
            if j == i or t_tile < 0 or a < 0:
                continue
            good = epoch_ok[j] & frame_kept
            good &= np.isfinite(lc[j, :, a]) & np.isfinite(lc_err[j, :, a])
            if np.count_nonzero(good) < search.min_epochs:
                continue
            med = nanmedian_quiet(np.where(good, lc[j, :, a], np.nan))
            if not np.isfinite(med) or med == 0:
                continue
            n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a, :, 0])))
            err = lc_err[j, :, a] * transit_result.frame_error_scale[t_tile, a, :]
            depth_j, sigma_depth_j = depth_at_other_aperture(
                bjd, lc[j, :, a] / med, err / med, good,
                cotrend_result.basis[t_tile, a, :n_cbv, :], tc_i, dur_i, search,
            )
            if not (np.isfinite(depth_j) and np.isfinite(sigma_depth_j) and sigma_depth_j > 0):
                continue
            sigma = depth_j / sigma_depth_j
            if sigma > best_sigma:
                best_sigma = sigma
                best = (j, depth_j, np.linalg.norm(vec[row_of[i]] - vec[jj]))
        if best is None:
            continue

        j, depth_j, chord_ij = best
        result.partner[i] = j
        result.sep_arcsec[i] = np.degrees(2.0 * np.arcsin(chord_ij / 2.0)) * 3600.0
        result.depth[i] = depth_j
        result.dip_sigma[i] = best_sigma
        if transit_result.searched[j] and np.isfinite(transit_result.tc[j]):
            result.dtc_hours[i] = (float(transit_result.tc[j]) - tc_i) * 24.0
        deficit_i = float(transit_result.depth[i]) * median_flux(i)
        deficit_j = depth_j * median_flux(j)
        has_deficits = np.isfinite(deficit_i) and np.isfinite(deficit_j) and deficit_i != 0
        if has_deficits:
            result.deficit_ratio[i] = deficit_j / deficit_i
        if best_sigma >= search.neighbour_event_dip_sigma:
            transit_result.flags[i] |= FLAG_NEIGHBOUR_SHARED_EVENT
            result.is_source[i] = bool(has_deficits and deficit_i >= deficit_j)
            flagged.append(int(i))

    # A pair of flagged candidates that name each other is decided once, by the higher-SNR one.
    flagged_set = set(flagged)
    for i in flagged:
        j = int(result.partner[i])
        if j <= i or j not in flagged_set or result.partner[j] != i:
            continue
        ref, other = (i, j) if transit_result.snr[i] >= transit_result.snr[j] else (j, i)
        result.is_source[other] = not result.is_source[ref]
        if np.isfinite(result.deficit_ratio[ref]) and result.deficit_ratio[ref] != 0:
            result.deficit_ratio[other] = 1.0 / result.deficit_ratio[ref]

    logger.info(
        "neighbour shared events: %d of %d candidates flagged (radius %.1f arcsec, >= %.1f sigma)",
        len(flagged), cand.size,
        search.neighbour_event_radius_arcsec, search.neighbour_event_dip_sigma,
    )
    return result
