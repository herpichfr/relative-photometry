"""Tile-level seeing/airmass decorrelation for light curves.

Fits per-star models to remove frame-level and stellar-property-dependent
scatter before output. The decorrelation model uses seeing (FWHM) and airmass
as frame regressors, and stellar magnitude and crowding as spatial regressors
to predict the per-star correction factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from relphot.numeric import mad_sigma, nanmedian_quiet, unit_vectors

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import DecorrelationSettings
    from relphot.match import MatchedNight
    from relphot.multinight import NightProducts
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "DecorrelationResult",
    "compute_crowding",
    "fit_coefficient_surface",
    "fit_decorrelation",
    "fit_star_seeing_airmass",
    "predict_surface",
]


@dataclass(slots=True)
class DecorrelationResult:
    """Decorrelation model outputs and corrected light curves.

    ``beta_star`` is ``(n_stars, n_aper, 2)`` with predicted ``[b_fwhm, b_air]``;
    NaN where not applied. ``lc_corrected`` is ``(n_stars, n_frames, n_aper)``
    after decorrelation. ``surface_coef`` is ``(n_aper, 2, n_terms)`` pooled coefficients
    for [fwhm, airmass]. ``tile_surface_coef`` is ``(n_tiles, n_aper, 2, n_terms)`` for
    per-tile fits (None if scope="pooled"). ``centring`` holds medians used in the
    surface fit: ``median_mag``, ``median_crowd``, ``median_fwhm``, ``median_airmass``,
    each ``(n_aper,)``. ``method`` is per aperture: "pooled" | "per_tile" | "disabled".
    ``diagnostics`` holds ``n_comparison_used`` and ``n_fallback_tiles``, each ``(n_aper,)``.
    """

    beta_star: np.ndarray
    lc_corrected: np.ndarray
    surface_coef: np.ndarray
    tile_surface_coef: np.ndarray | None
    centring: dict[str, np.ndarray]
    method: list[str]
    diagnostics: dict[str, np.ndarray]


def compute_crowding(night: MatchedNight | NightProducts) -> np.ndarray:
    """Crowding index for each star: log10(nearest-neighbor separation in arcsec).

    Uses a 3-D unit-vector cKDTree to find k=2 nearest neighbours (the first
    is the star itself). Separation is computed as arcsec on the sphere, bounded
    below at 0.05 arcsec. Stars with non-finite RA/Dec get NaN.

    Only ``night.ra``/``night.dec``/``night.n_stars`` are used, so a
    :class:`~relphot.multinight.NightProducts` works here too (reused, with
    the same definition, as the crowding regressor of the multi-night tie's
    pooled seeing surface -- see :mod:`relphot.multinight`).

    Parameters
    ----------
    night : MatchedNight | NightProducts
        Matched night data, or one night's multi-night-tie products.

    Returns
    -------
    np.ndarray
        Crowding indices, shape (n_stars,), in log10(arcsec).
    """
    n = night.n_stars
    crowding = np.full(n, np.nan, dtype=np.float64)

    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        return crowding

    vec = unit_vectors(night.ra[finite], night.dec[finite])
    tree = cKDTree(vec)
    # Query for k=2 to get self + nearest neighbor
    dist_rad, _idx = tree.query(vec, k=2)
    # dist_rad is (n_finite, 2); the second column is the nearest neighbor
    dist_arcsec = np.degrees(dist_rad[:, 1]) * 3600.0
    dist_arcsec = np.maximum(dist_arcsec, 0.05)

    local_idx = np.nonzero(finite)[0]
    crowding[local_idx] = np.log10(dist_arcsec)

    return crowding


def fit_star_seeing_airmass(
    lc: np.ndarray,
    good: np.ndarray,
    fwhm_c: np.ndarray,
    air_c: np.ndarray,
    settings: DecorrelationSettings,
    use_airmass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched per-star fit of seeing and airmass corrections.

    Fits y = b0 + b_fwhm*fwhm_c + [b_air*air_c] to each star's normalized light
    curve via normal equations. y is computed as -2.5*log10(LC / nanmedian(LC))
    over "good" epochs (finite LC and frame_ok). Iteratively clips outliers
    (MAD-based) up to max_iter_star rounds.

    Parameters
    ----------
    lc : np.ndarray
        Light curve, shape (n_stars, n_frames).
    good : np.ndarray
        Boolean mask of valid epochs, shape (n_stars, n_frames), True where
        LC is finite and frame_ok.
    fwhm_c : np.ndarray
        Centered FWHM, shape (n_frames,).
    air_c : np.ndarray
        Centered airmass, shape (n_frames,).
    settings : DecorrelationSettings
        Settings with min_epochs_per_star, clip_sigma_star, max_iter_star.
    use_airmass : bool, optional
        Whether to include the airmass term. Default True.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients, shape (n_stars, 3) = [b0, b_fwhm, b_air]; NaN
        if fewer than min_epochs_per_star good epochs. If use_airmass=False,
        b_air is NaN.
    se : np.ndarray
        Standard errors of fitted coefficients, shape (n_stars, 3). NaN where
        no fit, and NaN for airmass column when airmass is unused.
    n_used : np.ndarray
        Number of epochs used in the final fit, shape (n_stars,), int.
    ok : np.ndarray
        Boolean mask of successful fits, shape (n_stars,), True if n_used >=
        min_epochs_per_star.
    """
    n_stars, n_frames = lc.shape
    beta = np.full((n_stars, 3), np.nan, dtype=np.float64)
    se = np.full((n_stars, 3), np.nan, dtype=np.float64)

    lc_median = nanmedian_quiet(np.where(good, lc, np.nan), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = -2.5 * np.log10(np.where(good, lc / lc_median[:, None], np.nan))
    usable = good & np.isfinite(y)
    y0 = np.where(usable, y, 0.0)

    cols = [np.ones(n_frames), fwhm_c] + ([air_c] if use_airmass else [])
    x = np.column_stack(cols)  # (n_frames, n_params), shared by every star
    n_params = x.shape[1]
    xx = np.einsum("jk,jl->jkl", x, x)  # (n_frames, n_params, n_params)

    def _solve(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched normal equations for every star; singular systems -> NaN."""
        xtwx = np.einsum("sj,jkl->skl", weight, xx)
        xtwy = np.einsum("sj,jk,sj->sk", weight, x, y0)
        n = weight.sum(axis=1)
        fit = n >= max(settings.min_epochs_per_star, n_params + 1)
        with np.errstate(invalid="ignore"):
            fit &= np.abs(np.linalg.det(xtwx)) > 1e-12 * np.maximum(
                np.einsum("skk->s", xtwx) ** n_params, 1e-300
            )
        b = np.full((weight.shape[0], n_params), np.nan)
        inv = np.full_like(xtwx, np.nan)
        if np.any(fit):
            inv[fit] = np.linalg.inv(xtwx[fit])
            b[fit] = np.einsum("skl,sl->sk", inv[fit], xtwy[fit])
        return b, inv, fit

    weight = usable.astype(np.float64)
    for _ in range(max(int(settings.max_iter_star), 1)):
        b, _inv, fit = _solve(weight)
        resid = np.where(usable, y - b @ x.T, np.nan)
        used_resid = np.where(weight > 0, resid, np.nan)
        med = nanmedian_quiet(used_resid, axis=1)
        sig = mad_sigma(used_resid, axis=1)
        with np.errstate(invalid="ignore"):
            keep = usable & (
                ~np.isfinite(sig)[:, None]
                | (np.abs(resid - med[:, None]) <= settings.clip_sigma_star * sig[:, None])
            )
        new_weight = np.where(fit[:, None], keep, usable).astype(np.float64)
        if np.array_equal(new_weight, weight):
            break
        weight = new_weight

    b, inv, fit = _solve(weight)
    n_used = weight.sum(axis=1).astype(np.int64)
    resid = np.where(weight > 0, y0 - b @ x.T, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (resid**2).sum(axis=1) / (n_used - n_params)
        err = np.sqrt(s2[:, None] * np.einsum("skk->sk", inv))
    ok = fit
    beta[ok, :n_params] = b[ok]
    se[ok, :n_params] = err[ok]
    n_used = np.where(ok, n_used, 0)
    return beta, se, n_used, ok


def fit_coefficient_surface(
    mag_c: np.ndarray,
    crowd_c: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    settings: DecorrelationSettings,
) -> np.ndarray | None:
    """Fit a polynomial surface over stellar properties.

    Builds a design matrix Z with polynomial terms in mag_c and crowd_c:
    Z = [1, mag_c, ..., mag_c^mag_degree, crowd_c, ..., crowd_c^crowding_degree].
    Fits z = Z @ coef via weighted normal equations, iteratively clipping
    outliers. Weights typically come from 1/se^2 (inverse squared standard errors).

    Parameters
    ----------
    mag_c : np.ndarray
        Centered magnitudes, shape (n_stars,).
    crowd_c : np.ndarray
        Centered crowding, shape (n_stars,).
    y : np.ndarray
        Dependent variable (beta_fwhm or beta_air), shape (n_stars,).
    w : np.ndarray
        Weights (typically 1/se^2 from per-star fit), shape (n_stars,).
    settings : DecorrelationSettings
        Settings with mag_degree, crowding_degree, clip_sigma_surface, max_iter_surface,
        use_crowding, min_comparison_stars_total.

    Returns
    -------
    np.ndarray or None
        Fitted coefficients, shape (n_terms,), or None if insufficient data.
    """
    # Build design matrix
    terms = [np.ones_like(mag_c)]

    for deg in range(1, settings.mag_degree + 1):
        terms.append(mag_c ** deg)

    if settings.use_crowding:
        for deg in range(1, settings.crowding_degree + 1):
            terms.append(crowd_c ** deg)

    Z = np.column_stack(terms)
    n_terms = Z.shape[1]

    # Check if we have enough data: exclude rows where Z is non-finite
    Z_finite = np.all(np.isfinite(Z), axis=1)
    valid = Z_finite & np.isfinite(y) & np.isfinite(w) & (w > 0)
    n_valid = np.count_nonzero(valid)

    if n_valid < n_terms + 3 or n_valid < settings.min_comparison_stars_total:
        return None

    # Iteratively clip outliers
    mask = valid.copy()
    for _ in range(max(int(settings.max_iter_surface), 1)):
        Z_masked = Z[mask]
        y_masked = y[mask]
        w_masked = w[mask]

        # Weighted normal equations: (Z^T W Z) coef = Z^T W y
        ZtWZ = Z_masked.T @ (w_masked[:, None] * Z_masked)
        ZtWy = Z_masked.T @ (w_masked * y_masked)

        try:
            coef = np.linalg.solve(ZtWZ, ZtWy)
        except np.linalg.LinAlgError:
            return None

        # Compute residuals and MAD-based clipping
        resid = y - (Z @ coef)
        med = nanmedian_quiet(resid[mask])
        mad = mad_sigma(resid[mask])

        if np.isfinite(mad) and mad > 0:
            new_mask = valid & (np.abs(resid - med) <= settings.clip_sigma_surface * mad)
        else:
            new_mask = mask.copy()

        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    # Final fit
    Z_final = Z[mask]
    y_final = y[mask]
    w_final = w[mask]

    if np.count_nonzero(mask) < n_terms + 3:
        return None

    try:
        ZtWZ = Z_final.T @ (w_final[:, None] * Z_final)
        ZtWy = Z_final.T @ (w_final * y_final)
        return np.linalg.solve(ZtWZ, ZtWy)
    except np.linalg.LinAlgError:
        return None


def predict_surface(
    mag_c: np.ndarray, crowd_c: np.ndarray, coef: np.ndarray, settings: DecorrelationSettings
) -> np.ndarray:
    """Evaluate the surface at given (mag_c, crowd_c) values.

    Builds the same design matrix as fit_coefficient_surface and computes
    Z @ coef. Returns NaN for non-finite inputs.

    Parameters
    ----------
    mag_c : np.ndarray
        Centered magnitudes (broadcast-compatible).
    crowd_c : np.ndarray
        Centered crowding (broadcast-compatible).
    coef : np.ndarray
        Fitted coefficients from fit_coefficient_surface.
    settings : DecorrelationSettings
        Settings with mag_degree, crowding_degree, use_crowding.

    Returns
    -------
    np.ndarray
        Predicted values, same shape as mag_c and crowd_c (after broadcasting).
    """
    mag_c = np.asarray(mag_c, dtype=np.float64)
    crowd_c = np.asarray(crowd_c, dtype=np.float64)

    terms = [np.ones_like(mag_c)]
    for deg in range(1, settings.mag_degree + 1):
        terms.append(mag_c ** deg)
    if settings.use_crowding:
        for deg in range(1, settings.crowding_degree + 1):
            terms.append(crowd_c ** deg)

    Z = np.stack(terms, axis=-1)
    pred = np.tensordot(Z, coef, axes=([-1], [0]))
    return np.where(np.isfinite(pred), pred, np.nan)


def fit_decorrelation(
    night: MatchedNight,
    _tilemap: TileMap,
    comparison: ComparisonResult,
    lc: np.ndarray,
    frame_kept: np.ndarray,
    settings: DecorrelationSettings,
) -> DecorrelationResult:
    """Fit and apply decorrelation to light curves.

    Performs per-star fitting to remove frame-level FWHM and airmass dependence,
    then fits a surface model across stellar properties (magnitude, crowding) to
    generalize to all stars. Returns the corrected light curves and all
    intermediate products for diagnostics.

    Parameters
    ----------
    night : MatchedNight
        Matched night data.
    tilemap : TileMap
        Tile information.
    comparison : ComparisonResult
        Comparison star selection and ensemble.
    lc : np.ndarray
        Raw light curves (n_stars, n_frames, n_aper).
    frame_kept : np.ndarray
        Boolean mask of kept frames.
    settings : DecorrelationSettings
        Decorrelation settings.

    Returns
    -------
    DecorrelationResult
        Decorrelation model and corrected light curves.
    """
    n_stars, n_frames, n_aper = lc.shape

    # Initialize outputs
    beta_star = np.full((n_stars, n_aper, 2), np.nan, dtype=np.float64)
    lc_corrected = lc.copy()
    surface_coef = np.full((n_aper, 2, 0), np.nan, dtype=np.float64)  # Will resize
    tile_surface_coef = None
    method_list = []
    centring_dict = {}
    n_comparison_used = np.zeros(n_aper, dtype=np.int64)
    n_fallback_tiles = np.zeros(n_aper, dtype=np.int64)

    # Compute frame-level regressors: FWHM and airmass
    fwhm_frame = nanmedian_quiet(night.fwhm, axis=0)
    # Fallback to frame_meta FWHM if needed
    for j in range(n_frames):
        if not np.isfinite(fwhm_frame[j]):
            fwhm_frame[j] = float(night.frame_meta[j].median_fwhm)

    air_frame = np.array([
        (night.frame_meta[j].airmass if night.frame_meta[j].airmass is not None else np.nan)
        for j in range(n_frames)
    ], dtype=np.float64)

    # Check airmass availability
    n_finite_air = np.count_nonzero(np.isfinite(air_frame))
    missing_air_frac = 1.0 - (n_finite_air / max(n_frames, 1))
    use_airmass = (
        settings.use_airmass and missing_air_frac <= settings.max_missing_airmass_frac
    )
    if not use_airmass and settings.use_airmass:
        logger.warning(
            "airmass missing in %.1f%% of frames; dropping airmass term",
            100.0 * missing_air_frac,
        )

    # Center regressors
    fwhm_c = fwhm_frame - nanmedian_quiet(fwhm_frame)
    air_c = air_frame - nanmedian_quiet(air_frame) if use_airmass else np.zeros(n_frames)

    # Frame validity for fitting
    frame_ok = frame_kept & np.isfinite(fwhm_c) & (
        np.isfinite(air_c) if use_airmass else np.ones(n_frames, dtype=bool)
    )

    # Compute crowding for all stars
    crowding = compute_crowding(night)

    # Per aperture
    for a in range(n_aper):
        lc_a = lc[:, :, a]
        good_epochs = frame_ok[None, :] & np.isfinite(lc_a)

        # Per-star fitting on comparison stars
        comp_mask = comparison.mask[:, a]
        beta_full, se_full, _n_used_full, ok_full = fit_star_seeing_airmass(
            lc_a[comp_mask], good_epochs[comp_mask], fwhm_c, air_c, settings, use_airmass
        )

        # Extract b_fwhm and b_air from beta_full (which is [b0, b_fwhm, b_air])
        comp_indices = np.nonzero(comp_mask)[0]
        beta_star[comp_indices[ok_full], a, 0] = beta_full[ok_full, 1]  # b_fwhm
        beta_star[comp_indices[ok_full], a, 1] = beta_full[ok_full, 2]  # b_air

        # Surface fitting
        valid_beta = ok_full
        if not np.any(valid_beta):
            logger.warning("aperture %d: no valid per-star beta; disabling decorrelation", a)
            method_list.append("disabled")
            lc_corrected[:, :, a] = lc_a
            surface_coef = np.full((n_aper, 2, 1), np.nan, dtype=np.float64)
            n_comparison_used[a] = 0
            continue

        # Prepare data for surface fit
        comp_idx_valid = comp_indices[valid_beta]
        mag_comp = comparison.mag[comp_idx_valid, a]
        crowd_comp = crowding[comp_idx_valid]
        mag_c_comp = mag_comp - nanmedian_quiet(mag_comp)
        crowd_c_comp = crowd_comp - nanmedian_quiet(crowd_comp)

        # Weights based on standard errors: 1/se^2
        se_fwhm = se_full[valid_beta, 1]
        se_air = se_full[valid_beta, 2]

        # Non-finite or zero se -> weight 0
        w_fwhm = np.where((np.isfinite(se_fwhm)) & (se_fwhm > 0), 1.0 / (se_fwhm ** 2), 0.0)
        w_air = np.where((np.isfinite(se_air)) & (se_air > 0), 1.0 / (se_air ** 2), 0.0)

        # Fit surfaces for b_fwhm and b_air
        coef_fwhm = fit_coefficient_surface(
            mag_c_comp,
            crowd_c_comp,
            beta_full[valid_beta, 1],  # b_fwhm
            w_fwhm,
            settings,
        )
        coef_air = fit_coefficient_surface(
            mag_c_comp,
            crowd_c_comp,
            beta_full[valid_beta, 2] if use_airmass else np.zeros(len(comp_idx_valid)),
            w_air,
            settings,
        )

        # Determine method
        if coef_fwhm is None or coef_air is None:
            logger.warning("aperture %d: insufficient comparison stars for surface fit", a)
            method_list.append("disabled")
            lc_corrected[:, :, a] = lc_a
            surface_coef = np.full((n_aper, 2, 1), np.nan, dtype=np.float64)
            n_comparison_used[a] = 0
        else:
            method_list.append(settings.scope)
            n_comparison_used[a] = len(comp_idx_valid)

            # Store surface coefficients
            max_terms = max(len(coef_fwhm), len(coef_air))
            if surface_coef.shape[2] < max_terms:
                new_coef = np.full((n_aper, 2, max_terms), np.nan, dtype=np.float64)
                if surface_coef.shape[2] > 0:
                    new_coef[:, :, :surface_coef.shape[2]] = surface_coef
                surface_coef = new_coef

            surface_coef[a, 0, :len(coef_fwhm)] = coef_fwhm
            surface_coef[a, 1, :len(coef_air)] = coef_air

            # Predict beta_star for all stars
            mag_all = comparison.mag[:, a]
            crowd_all = crowding
            mag_c_all = mag_all - nanmedian_quiet(mag_comp)
            crowd_c_all = crowd_all - nanmedian_quiet(crowd_comp)

            pred_fwhm = predict_surface(mag_c_all, crowd_c_all, coef_fwhm, settings)
            pred_air = (
                predict_surface(mag_c_all, crowd_c_all, coef_air, settings)
                if use_airmass
                else np.zeros(n_stars)
            )

            # Apply correction with broadcasting
            # corr = 10**(0.4*(pred_fwhm[:,None]*fwhm_c[None,:] +
            #                  pred_air[:,None]*air_c[None,:]))
            pred_fwhm_col = pred_fwhm[:, None]
            pred_air_col = pred_air[:, None]
            fwhm_c_row = fwhm_c[None, :]
            air_c_row = air_c[None, :]

            correction_exp = 0.4 * (pred_fwhm_col * fwhm_c_row + pred_air_col * air_c_row)
            correction = 10.0 ** correction_exp

            # Apply where finite(pred) & frame_ok
            pred_finite = np.isfinite(pred_fwhm) & np.isfinite(pred_air)
            lc_corrected[:, :, a] = np.where(
                pred_finite[:, None] & frame_ok[None, :],
                lc_a * correction,
                lc_a
            )

            # Store predictions in beta_star
            beta_star[pred_finite, a, 0] = pred_fwhm[pred_finite]
            beta_star[pred_finite, a, 1] = pred_air[pred_finite]

    # Store centring values
    centring_dict["median_mag"] = np.array(
        [nanmedian_quiet(comparison.mag[:, a]) for a in range(n_aper)], dtype=np.float64
    )
    centring_dict["median_crowd"] = np.nanmedian(crowding)
    centring_dict["median_fwhm"] = nanmedian_quiet(fwhm_frame)
    centring_dict["median_airmass"] = nanmedian_quiet(air_frame)

    return DecorrelationResult(
        beta_star=beta_star,
        lc_corrected=lc_corrected,
        surface_coef=surface_coef,
        tile_surface_coef=tile_surface_coef,
        centring=centring_dict,
        method=method_list,
        diagnostics={
            "n_comparison_used": n_comparison_used,
            "n_fallback_tiles": n_fallback_tiles,
        },
    )
