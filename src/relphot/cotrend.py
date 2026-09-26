"""Stage 6a: per-tile, per-aperture cotrending basis vectors (CBVs).

A tile's CBVs are the leading principal components of its comparison-star
ensemble's normalised, sigma-clipped, gap-filled decorrelated light curves
(:class:`~relphot.decorrelate.DecorrelationResult.lc_corrected` /
:class:`~relphot.lightcurve.LightCurveResult.lc`), restricted to kept frames.
That input is already free of any star's own free-fit systematics -- the
per-star seeing/airmass correction in :mod:`relphot.decorrelate` is a
*population-level* surface prediction from the comparison ensemble, not a
free fit to each star's own light curve -- so a CBV regression built on top
of it stays transit-safe (:mod:`relphot.transit_search` never fits a star's
own light curve freely before searching it).

:func:`detect_systematic_frames` flags frames where a large fraction of a
tile's comparison ensemble moves together beyond its own scatter (an
end-of-night ramp, a step-like glitch, ...); :mod:`relphot.variability` uses
it to avoid crediting a shared systematic as genuine stellar variability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relphot.numeric import mad_sigma, nanmedian_quiet

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import SearchSettings
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "CotrendResult",
    "compute_cbvs",
    "compute_frame_error_scale",
    "detect_systematic_frames",
    "select_star_epochs",
]


@dataclass(slots=True)
class CotrendResult:
    """Per-(tile, aperture) cotrending basis vectors on the full frame axis.

    ``basis`` is ``(n_tiles, n_aper, n_cbv, n_frames)`` float64, NaN outside
    kept frames and beyond however many components a given (tile, aperture)
    actually produced (see ``cbv_explained_variance``/``n_cbv``).
    ``explained_variance_ratio`` is ``(n_tiles, n_aper, n_cbv)``, NaN where
    ``basis`` is NaN for that component. ``n_used`` is ``(n_tiles, n_aper)``,
    the number of comparison stars actually used after sigma-clipping.
    """

    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    n_used: np.ndarray

    @property
    def n_cbv(self) -> int:
        return int(self.basis.shape[2])


def _normalised_matrix(sub: np.ndarray, settings: SearchSettings) -> np.ndarray | None:
    """(n_rows_kept, n_kept_frames) normalised, clipped, gap-filled residual matrix.

    ``sub`` is ``(n_members, n_kept_frames)``, already restricted to a tile's
    comparison-star ensemble at one aperture and the night's kept frames.
    Rows with fewer than 3 finite points, or a non-finite/zero median, are
    dropped. Returns ``None`` if fewer than 3 rows survive.
    """
    good = np.isfinite(sub)
    med = nanmedian_quiet(np.where(good, sub, np.nan), axis=1)
    ok_row = np.isfinite(med) & (med != 0) & (np.count_nonzero(good, axis=1) >= 3)
    if np.count_nonzero(ok_row) < 3:
        return None

    sub = sub[ok_row]
    good = good[ok_row]
    med = med[ok_row]

    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(good, sub / med[:, None] - 1.0, np.nan)

    for _ in range(max(int(settings.cbv_max_iter), 1)):
        sig = mad_sigma(norm, axis=1)
        with np.errstate(invalid="ignore"):
            keep = np.isfinite(norm) & (np.abs(norm) <= settings.cbv_clip_sigma * sig[:, None])
        new_norm = np.where(keep, norm, np.nan)
        if np.array_equal(np.isfinite(new_norm), np.isfinite(norm)):
            norm = new_norm
            break
        norm = new_norm

    # Gap-fill any still-missing entry (a bad epoch for that star) with 0 --
    # the star's own robust mean in the normalised residual -- so the SVD
    # sees a complete matrix. Only kept frames are ever columns here.
    return np.where(np.isfinite(norm), norm, 0.0)


def compute_cbvs(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    lc: np.ndarray,
    frame_kept: np.ndarray,
    settings: SearchSettings,
) -> CotrendResult:
    """Leading principal components of each tile's comparison ensemble, per aperture.

    ``lc`` is the decorrelated light curve, ``(n_stars, n_frames, n_aper)``.
    """
    n_tiles = tilemap.n_tiles
    n_frames = lc.shape[1]
    n_aper = lc.shape[2]
    k = max(int(settings.n_cbv), 1)

    basis = np.full((n_tiles, n_aper, k, n_frames), np.nan, dtype=np.float64)
    evr = np.full((n_tiles, n_aper, k), np.nan, dtype=np.float64)
    n_used = np.zeros((n_tiles, n_aper), dtype=np.int64)

    kept_idx = np.nonzero(frame_kept)[0]
    if kept_idx.size == 0:
        logger.warning("no kept frames; CBVs are all-NaN")
        return CotrendResult(basis=basis, explained_variance_ratio=evr, n_used=n_used)

    for a in range(n_aper):
        for t in range(n_tiles):
            members = np.nonzero(comparison_result.mask[:, a] & (tilemap.core_tile == t))[0]
            if members.size < 3:
                logger.warning(
                    "tile %d, aperture %d: %d comparison stars, need >= 3; no CBVs",
                    t, a, members.size,
                )
                continue

            matrix = _normalised_matrix(lc[members][:, kept_idx, a], settings)
            if matrix is None:
                logger.warning(
                    "tile %d, aperture %d: too few usable comparison stars for CBVs", t, a
                )
                continue

            n_used[t, a] = matrix.shape[0]
            try:
                _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.warning("tile %d, aperture %d: SVD failed; no CBVs", t, a)
                continue

            var = s**2
            total_var = float(var.sum())
            if total_var <= 0:
                continue
            ratio = var / total_var
            cum = np.cumsum(ratio)
            n_keep = min(k, vt.shape[0])
            over = np.nonzero(cum >= settings.cbv_explained_variance)[0]
            if over.size:
                n_keep = min(n_keep, int(over[0]) + 1)

            for c in range(n_keep):
                basis[t, a, c, kept_idx] = vt[c]
                evr[t, a, c] = ratio[c]

    return CotrendResult(basis=basis, explained_variance_ratio=evr, n_used=n_used)


def detect_systematic_frames(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    lc: np.ndarray,
    frame_kept: np.ndarray,
    settings: SearchSettings,
) -> np.ndarray:
    """Boolean mask, ``(n_frames,)``, of frames where a comparison ensemble moves together.

    A frame is flagged for a given (tile, aperture) when more than
    ``settings.systematic_frame_fraction`` of that tile's comparison stars
    are simultaneously a > 3-sigma outlier from their own robust scatter
    there; the returned mask is the OR of every (tile, aperture) result.
    Used to keep a shared instrumental glitch (an end-of-night ramp, a
    step) from being credited as genuine per-star variability.
    """
    n_frames = lc.shape[1]
    n_aper = lc.shape[2]
    flagged = np.zeros(n_frames, dtype=bool)

    kept_idx = np.nonzero(frame_kept)[0]
    if kept_idx.size == 0:
        return flagged

    for a in range(n_aper):
        for t in range(tilemap.n_tiles):
            members = np.nonzero(comparison_result.mask[:, a] & (tilemap.core_tile == t))[0]
            if members.size < 5:
                continue

            sub = lc[members][:, kept_idx, a]
            good = np.isfinite(sub)
            med = nanmedian_quiet(np.where(good, sub, np.nan), axis=1)
            ok_row = np.isfinite(med) & (med != 0)
            if np.count_nonzero(ok_row) < 5:
                continue

            sub = sub[ok_row]
            good = good[ok_row]
            med = med[ok_row]
            with np.errstate(invalid="ignore", divide="ignore"):
                norm = np.where(good, sub / med[:, None] - 1.0, np.nan)
            sig = mad_sigma(norm, axis=1)
            with np.errstate(invalid="ignore"):
                is_outlier = np.isfinite(norm) & (np.abs(norm) > 3.0 * sig[:, None])
            valid_count = np.count_nonzero(np.isfinite(norm), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(
                    valid_count > 0, np.count_nonzero(is_outlier, axis=0) / valid_count, 0.0
                )
            bad = frac > settings.systematic_frame_fraction
            flagged[kept_idx[bad]] = True

    logger.info("systematic frames flagged: %d/%d", int(flagged.sum()), n_frames)
    return flagged


def select_star_epochs(
    tilemap: TileMap,
    cotrend_result: CotrendResult,
    bjd: np.ndarray,
    lc: np.ndarray,
    lc_err: np.ndarray,
    epoch_ok: np.ndarray,
    frame_kept: np.ndarray,
    star_best_aper: np.ndarray,
    min_epochs: int,
    star: int,
) -> tuple[bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One star's time-sorted, normalised, good-epoch light curve and CBV rows.

    A convenience wrapper around the epoch-selection logic every Stage-6
    per-star routine (:mod:`relphot.transit_search`, :mod:`relphot.variables`)
    repeats, kept here for callers -- such as CLI plotting -- that need to
    reproduce exactly the same star/epoch selection those routines used,
    without re-deriving it. Returns ``(ok, aper, t_good, y_norm, err_norm,
    cbv_rows, idx_good)``; ``idx_good`` indexes the full frame axis
    (time-sorted). ``ok=False`` (and every array empty, ``aper=-1``) when the
    star has no core tile, no best aperture, fewer than ``min_epochs`` good
    epochs, or a non-finite median flux.
    """
    empty_f = np.array([], dtype=np.float64)
    empty_i = np.array([], dtype=np.int64)
    t_tile = int(tilemap.core_tile[star])
    a = int(star_best_aper[star])
    if t_tile < 0 or a < 0 or not (0 <= a < lc.shape[2]):
        return False, -1, empty_f, empty_f, empty_f, np.empty((0, 0)), empty_i

    good = (
        epoch_ok[star] & frame_kept & np.isfinite(lc[star, :, a]) & np.isfinite(lc_err[star, :, a])
    )
    if np.count_nonzero(good) < min_epochs:
        return False, a, empty_f, empty_f, empty_f, np.empty((0, 0)), empty_i

    idx = np.nonzero(good)[0]
    order = np.argsort(bjd[idx])
    idx = idx[order]
    t_g = bjd[idx]
    y = lc[star, idx, a].astype(np.float64)
    err = lc_err[star, idx, a].astype(np.float64)
    med = nanmedian_quiet(y)
    if not np.isfinite(med) or med == 0:
        return False, a, empty_f, empty_f, empty_f, np.empty((0, 0)), empty_i

    n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a, :, 0])))
    cbv_rows = cotrend_result.basis[t_tile, a, :n_cbv, :][:, idx]
    return True, a, t_g, y / med, err / med, cbv_rows, idx


def compute_frame_error_scale(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    lc: np.ndarray,
    lc_err: np.ndarray,
    frame_kept: np.ndarray,
    settings: SearchSettings,
) -> np.ndarray:
    """Per-(tile, aperture, frame) error-inflation factor from the comparison ensemble.

    For each kept frame, ``f_j = 1.4826 * MAD_i((y_ij - median_i) / err_ij)``
    over that tile's comparison stars ``i`` -- how far off, in formal sigma,
    the comparison ensemble collectively sits at that epoch. Normalised so
    the median over kept frames is 1, then clipped to
    ``[settings.frame_error_scale_min, settings.frame_error_scale_max]``
    (a value below 1 is kept, not floored: a frame quieter than typical
    should not be penalised, only one noisier than typical inflated).
    Multiplying every star's per-point error by this factor before a search
    (see :func:`relphot.transit_search.search_transits`) down-weights a
    genuinely noisy or glitch-affected epoch -- rising airmass near the end
    of a run, a shared instrumental dropout -- without any per-star fit, so
    it stays transit-safe. Defaults to 1 (no rescaling) for a
    (tile, aperture) with fewer than 3 usable comparison stars, or for any
    frame where the ensemble itself could not be scored.

    Parameters
    ----------
    tilemap : TileMap
        Tile mapping.
    comparison_result : ComparisonResult
        Comparison star selection.
    lc : np.ndarray
        Decorrelated light curves, ``(n_stars, n_frames, n_aper)``.
    lc_err : np.ndarray
        Light curve errors, same shape.
    frame_kept : np.ndarray
        Boolean mask of kept frames, ``(n_frames,)``.
    settings : SearchSettings
        Settings with ``frame_error_scale_min``/``frame_error_scale_max``.

    Returns
    -------
    np.ndarray
        ``(n_tiles, n_aper, n_frames)``, all-1 outside kept frames or where
        undeterminable.
    """
    n_tiles = tilemap.n_tiles
    n_aper = lc.shape[2]
    n_frames = lc.shape[1]
    scale = np.ones((n_tiles, n_aper, n_frames), dtype=np.float64)

    kept_idx = np.nonzero(frame_kept)[0]
    if kept_idx.size == 0:
        return scale

    for a in range(n_aper):
        for t in range(n_tiles):
            members = np.nonzero(comparison_result.mask[:, a] & (tilemap.core_tile == t))[0]
            if members.size < 3:
                logger.warning(
                    "tile %d, aperture %d: %d comparison stars, need >= 3; no error rescaling",
                    t, a, members.size,
                )
                continue

            sub_y = lc[members][:, kept_idx, a]
            sub_err = lc_err[members][:, kept_idx, a]
            good = np.isfinite(sub_y) & np.isfinite(sub_err) & (sub_err > 0)
            med = nanmedian_quiet(np.where(good, sub_y, np.nan), axis=1)
            ok_row = np.isfinite(med) & (med != 0) & (np.count_nonzero(good, axis=1) >= 3)
            if np.count_nonzero(ok_row) < 3:
                logger.warning(
                    "tile %d, aperture %d: too few usable comparison stars for error rescaling",
                    t, a,
                )
                continue

            sub_y = sub_y[ok_row]
            sub_err = sub_err[ok_row]
            good = good[ok_row]
            med = med[ok_row]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(good, (sub_y - med[:, None]) / sub_err, np.nan)
            f = 1.4826 * mad_sigma(r, axis=0)
            good_f = np.isfinite(f) & (f > 0)
            if not np.any(good_f):
                continue
            med_f = nanmedian_quiet(f[good_f])
            if not np.isfinite(med_f) or med_f <= 0:
                continue
            f_norm = f / med_f
            f_clipped = np.clip(
                f_norm, settings.frame_error_scale_min, settings.frame_error_scale_max
            )
            scale[t, a, kept_idx] = np.where(good_f, f_clipped, 1.0)

    return scale
