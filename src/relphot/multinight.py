"""Phase 2: cross-match and zero-point tie two or more already-processed nights.

Each night processed by :mod:`relphot.cli` (``ingest`` -> ``reference`` ->
``lightcurves``) yields a self-consistent set of instrumental magnitudes, but
the flux level between two nights differs by a zero point that depends on
sky position and stellar magnitude (differential seeing, differential
extinction correction, different reference-star populations). This module
cross-matches the nights' stars into one global list (:func:`crossmatch_nights`),
then fits a per-(night, aperture) zero-point surface
``Z_n(xi, eta, M) = poly2(xi, eta) + poly2(M)`` -- fixed to zero for one
anchor night -- to comparison stars common to at least two nights
(:func:`tie_nights`), and finally assembles multi-night calibrated light
curves (:func:`build_multinight_lightcurves`). ``xi``/``eta`` are gnomonic
tangent-plane coordinates of the field; the anchor is the night with the
most kept frames by default. Optionally (``settings.use_seeing_term``, on by
default) the model also carries a pooled seeing term ``beta(M, crowding) *
(F_n - F_anchor)``, ``F_n`` the night's median FWHM (px) over kept frames and
``beta`` one magnitude+crowding surface shared by every night -- fit jointly
with the per-night polynomials inside the same alternating fit -- so that a
star-dependent, seeing-dependent excess (crowding/aperture loss growing with
the seeing difference between two nights) is absorbed by the tie itself
rather than left for the per-night calibration floor below to (mis)represent
as a per-night constant. An empirical per-night calibration floor is
estimated from the night-to-night scatter of tie stars after convergence and
added in quadrature to the reported errors, so that ``chi2_after`` comes out
close to 1 when the tie is self-consistent.

A night can instead be tied *loosely* (``settings.loose_nights``,
:func:`split_loose_nights`, :func:`tie_loose_nights`): the other (core) nights
are tied by the unchanged code on the core subset alone, and each loose night
is then fit only to that fixed core frame -- the core nights' calibrated
weighted mean -- by a low-order surface, with a calibration floor measured
against the frame. A loose night therefore cannot move any core night's zero
point, floor, mean magnitude or aperture, and its own offset and errors stay
loose (a night that would break the tie, e.g. a cloudy one, is still usable for
variability work). :attr:`NightTie.loose` marks these nights; the multi-night
search (:mod:`relphot.multinight_search`) keeps them out of every transit test.

Vectorised over stars throughout; the only Python-level loops are over
nights (2-10), apertures, and alternating-fit iterations, per the project's
usual convention.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

from relphot.config import MultiNightSettings, Settings, settings_from_dict, settings_to_dict
from relphot.decorrelate import compute_crowding
from relphot.exceptions import MultiNightError
from relphot.io import load_lightcurves_npz, load_night, load_reference
from relphot.match import _resolve_duplicates
from relphot.numeric import mad_sigma, nanmedian_quiet, unit_vectors

logger = logging.getLogger(__name__)

__all__ = [
    "MultiNightLightCurves",
    "NightCrossMatch",
    "NightProducts",
    "NightTie",
    "build_multinight_lightcurves",
    "check_compatible",
    "core_crossmatch",
    "crossmatch_nights",
    "evaluate_zero_point",
    "load_multinight",
    "load_night_products",
    "plot_tie_diagnostics",
    "resolve_anchor_index",
    "save_multinight",
    "save_multinight_tables",
    "save_tie_report",
    "split_loose_nights",
    "tie_loose_nights",
    "tie_nights",
]

#: median(chi^2_1) -- see the calibration-floor bisection in :func:`tie_nights`.
_MEDIAN_CHI2_1 = 0.45494

#: A fitted zero point (mag) of a tied star beyond this is a runaway of the alternating
#: fit, not a calibration: the surfaces are mmag to sub-mag (the largest, a faint-end
#: ``dm**2`` extrapolation, reaches ~2 mag), so :func:`tie_nights` drops such a star
#: (:attr:`NightTie.rejected`, NaN ``mean_mag`` and ``zp``). A loose night's comparison
#: star further than this from the night's median offset to the core frame is not fit.
_ZP_RUNAWAY_MAG = 5.0
#: A tie row (night, star) whose magnitude differs from the other nights' weighted mean
#: by more than this (mag), beyond the night's own median offset, is not a calibrator:
#: a nightly magnitude that far off (a garbage or blended measurement, typically with a
#: huge error) has no weight by its error alone but, as a high-leverage row of the
#: ``dm**2`` fit, wrecks the surface. Such rows are left out of the tie fit.
_TIE_DISCREPANCY_MAG = 3.0
#: A loose night's evaluated zero point (mag) beyond this, or non-finite, is not a
#: number a surface can mean (a degenerate fit): the star's ``zp`` is NaN.
_ZP_ABSURD_MAG = 50.0


@dataclass(slots=True)
class NightProducts:
    """One night's products, trimmed to what multi-night tying needs.

    ``core_tile`` is -1 for a star with no core tile (never a tie/reference
    candidate). ``lc``/``lc_err`` are ``(n_stars, n_frames, n_aper)``;
    ``epoch_ok`` is ``(n_stars, n_frames)``; ``rms``/``n_epochs``/
    ``comparison_mask`` are ``(n_stars, n_aper)``; ``star_best_aper`` is
    ``(n_stars,)``. ``airmass`` is NaN for a frame with no airmass recorded.
    """

    label: str
    directory: Path
    ra: np.ndarray
    dec: np.ndarray
    core_tile: np.ndarray
    bjd_tdb: np.ndarray
    airmass: np.ndarray
    fwhm: np.ndarray
    frame_kept: np.ndarray
    lc: np.ndarray
    lc_err: np.ndarray
    epoch_ok: np.ndarray
    rms: np.ndarray
    n_epochs: np.ndarray
    comparison_mask: np.ndarray
    star_best_aper: np.ndarray
    filter: str
    object: str
    aperture_radii: tuple[float, ...]

    @property
    def n_stars(self) -> int:
        return int(self.ra.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.bjd_tdb.shape[0])

    @property
    def n_aper(self) -> int:
        return int(self.lc.shape[2]) if self.lc.ndim == 3 else 0


def load_night_products(
    directory: Path | str,
    *,
    label: str | None = None,
    night_file: str = "night.npz",
    ref_file: str = "ref.npz",
    lc_file: str = "lc/night_lc.npz",
) -> NightProducts:
    """Load one night's ``night.npz``/``ref.npz``/``lc/night_lc.npz`` as a :class:`NightProducts`.

    ``directory`` is the night's ``relphot/`` directory. ``label`` defaults
    to the name of ``directory``'s parent (e.g. a night processed at
    ``.../20251104/relphot`` gets label ``"20251104"``). Only the arrays
    :mod:`relphot.multinight` needs are kept; the (much larger) per-frame
    flux/flag/tile-index arrays are dropped once this function returns.

    Raises
    ------
    MultiNightError
        If the star counts recorded in ``night_file``, ``ref_file`` and
        ``lc_file`` disagree, or their frame counts disagree.
    """
    directory = Path(directory)
    if label is None:
        label = directory.parent.name

    night, _night_settings = load_night(directory / night_file)
    tilemap, ref_result, _ref_settings = load_reference(directory / ref_file)
    (
        lc_result, star_stats, _best_aper_per_tile, _bin_edges, star_best_aper,
        comparison_result, _lc_settings, _decorrelation,
    ) = load_lightcurves_npz(directory / lc_file)

    n_stars_night = night.n_stars
    n_stars_ref = int(tilemap.core_tile.shape[0])
    n_stars_lc = int(lc_result.lc.shape[0])
    if not (n_stars_night == n_stars_ref == n_stars_lc):
        msg = (
            f"{directory}: inconsistent star counts across products "
            f"(night={n_stars_night}, ref={n_stars_ref}, lc={n_stars_lc})"
        )
        raise MultiNightError(msg)

    bjd_tdb = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    airmass = np.array(
        [np.nan if m.airmass is None else m.airmass for m in night.frame_meta], dtype=np.float64
    )
    fwhm = np.array([m.median_fwhm for m in night.frame_meta], dtype=np.float64)
    if bjd_tdb.shape[0] != lc_result.lc.shape[1]:
        msg = (
            f"{directory}: frame count mismatch (night={bjd_tdb.shape[0]}, "
            f"lc={lc_result.lc.shape[1]})"
        )
        raise MultiNightError(msg)

    aperture_radii = (
        tuple(night.frame_meta[0].aperture_radii_px) if night.frame_meta else ()
    )
    filt = night.frame_meta[0].filter if night.frame_meta else ""
    obj = night.frame_meta[0].object if night.frame_meta else ""

    return NightProducts(
        label=str(label),
        directory=directory,
        ra=np.asarray(night.ra, dtype=np.float64),
        dec=np.asarray(night.dec, dtype=np.float64),
        core_tile=np.asarray(tilemap.core_tile, dtype=np.int64),
        bjd_tdb=bjd_tdb,
        airmass=airmass,
        fwhm=fwhm,
        frame_kept=np.asarray(ref_result.frame_kept, dtype=bool),
        lc=np.asarray(lc_result.lc, dtype=np.float32),
        lc_err=np.asarray(lc_result.lc_err, dtype=np.float32),
        epoch_ok=np.asarray(lc_result.epoch_ok, dtype=bool),
        rms=np.asarray(star_stats.rms, dtype=np.float64),
        n_epochs=np.asarray(star_stats.n_epochs, dtype=np.int64),
        comparison_mask=np.asarray(comparison_result.mask, dtype=bool),
        star_best_aper=np.asarray(star_best_aper, dtype=np.int64),
        filter=filt,
        object=obj,
        aperture_radii=aperture_radii,
    )


def check_compatible(nights: list[NightProducts]) -> list[NightProducts]:
    """Validate cross-night compatibility and return ``nights`` sorted by median ``bjd_tdb``.

    Every night must share the same filter and aperture count, else
    :class:`~relphot.exceptions.MultiNightError`; a differing ``object`` or
    aperture-radii set only logs a warning (the field name/setup may
    legitimately differ while the same photometric system is used).
    """
    filters = {n.filter for n in nights}
    if len(filters) > 1:
        msg = f"nights have different filters: {sorted(filters)}"
        raise MultiNightError(msg)

    n_apers = {n.n_aper for n in nights}
    if len(n_apers) > 1:
        details = {n.label: n.n_aper for n in nights}
        msg = f"nights have different aperture counts: {details}"
        raise MultiNightError(msg)

    objects = {n.object for n in nights}
    if len(objects) > 1:
        logger.warning("nights have different 'object' header values: %s", sorted(objects))

    radii = {n.aperture_radii for n in nights}
    if len(radii) > 1:
        logger.warning("nights have different aperture radii: %s", sorted(radii))

    return sorted(nights, key=lambda n: float(nanmedian_quiet(n.bjd_tdb)))


def resolve_anchor_index(nights: list[NightProducts], anchor: str) -> int:
    """Index into ``nights`` of the zero-point anchor night (``settings.multinight.anchor``).

    ``"auto"`` picks the night with the most kept frames; ties go to the
    earliest night, i.e. the smallest index -- callers pass ``nights``
    already sorted by time (see :func:`check_compatible`).
    """
    if anchor == "auto":
        n_kept = np.array(
            [int(np.count_nonzero(n.frame_kept)) for n in nights], dtype=np.int64
        )
        return int(np.argmax(n_kept))

    for i, night in enumerate(nights):
        if night.label == anchor:
            return i
    labels = [n.label for n in nights]
    msg = f"anchor night {anchor!r} not found among {labels}"
    raise MultiNightError(msg)


def split_loose_nights(
    nights: list[NightProducts], loose_labels: tuple[str, ...]
) -> tuple[list[NightProducts], list[NightProducts]]:
    """``(core, loose)``: ``nights`` split by ``settings.loose_nights``, each in its own order.

    Callers tie ``core + loose`` (core first), so that every core night's rows of the
    global star list come before any star only a loose night contains (see
    :func:`core_crossmatch`).

    Raises
    ------
    MultiNightError
        If a label in ``loose_labels`` is not among ``nights``, or fewer than 2 core
        nights remain while some night is loose.
    """
    labels = [n.label for n in nights]
    missing = [label for label in loose_labels if label not in labels]
    if missing:
        msg = f"loose night(s) {missing} not found among {labels}"
        raise MultiNightError(msg)
    core = [n for n in nights if n.label not in loose_labels]
    loose = [n for n in nights if n.label in loose_labels]
    if loose and len(core) < 2:
        msg = f"loose nights need at least 2 core nights, got {[n.label for n in core]}"
        raise MultiNightError(msg)
    return core, loose


@dataclass(frozen=True, slots=True)
class NightCrossMatch:
    """Global star list built by cross-matching every night against the anchor first.

    ``ra``/``dec`` are the global list's positions (position of first
    sighting; the anchor night's own stars come first). ``index[n, g]`` is
    star ``g``'s index within night ``n``, or -1 if night ``n`` does not
    contain it. ``n_matched[n]`` is the number of night ``n``'s own
    (core-tile) stars that matched an already-known global star (0 for the
    anchor night, whose stars all *become* the initial global list).
    """

    ra: np.ndarray
    dec: np.ndarray
    index: np.ndarray
    labels: tuple[str, ...]
    n_matched: np.ndarray


def crossmatch_nights(
    nights: list[NightProducts], anchor_index: int, radius_arcsec: float
) -> NightCrossMatch:
    """Cross-match every night's core-tile stars into one global star list.

    The global list starts as the anchor night's own core-tile stars; every
    other night is then matched against the (growing) global list in the
    order given, one-to-one "closest wins" (:func:`relphot.match._resolve_duplicates`),
    with unmatched stars appended as new global stars. Only stars with
    ``core_tile >= 0`` take part -- a star never used as a reference/comparison
    candidate in its own night carries no useful photometric information for
    the tie.
    """
    n_nights = len(nights)
    labels = tuple(n.label for n in nights)
    radius_rad = np.radians(radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)

    anchor = nights[anchor_index]
    anchor_local = np.nonzero(anchor.core_tile >= 0)[0]
    global_ra: list[float] = list(np.asarray(anchor.ra[anchor_local], dtype=np.float64))
    global_dec: list[float] = list(np.asarray(anchor.dec[anchor_local], dtype=np.float64))

    star_index = np.full((n_nights, anchor_local.size), -1, dtype=np.int64)
    star_index[anchor_index] = anchor_local
    n_matched = np.zeros(n_nights, dtype=np.int64)

    other_nights = [n for n in range(n_nights) if n != anchor_index]
    for n in other_nights:
        night = nights[n]
        local = np.nonzero(night.core_tile >= 0)[0]
        if local.size == 0:
            logger.info("crossmatch %s: 0 core-tile stars to match", night.label)
            continue

        n_global_before = star_index.shape[1]
        global_vec = unit_vectors(np.asarray(global_ra), np.asarray(global_dec))
        tree = cKDTree(global_vec)
        night_vec = unit_vectors(night.ra[local], night.dec[local])
        dist, idx = tree.query(night_vec, k=1)
        within = dist <= chord_radius
        matched_src, matched_master, matched_dist = _resolve_duplicates(dist, idx, within)

        col = np.full(n_global_before, -1, dtype=np.int64)
        col[matched_master] = local[matched_src]

        matched_mask = np.zeros(local.size, dtype=bool)
        matched_mask[matched_src] = True
        unmatched_local = local[~matched_mask]
        n_new = int(unmatched_local.size)
        if n_new:
            global_ra.extend(np.asarray(night.ra[unmatched_local], dtype=np.float64).tolist())
            global_dec.extend(np.asarray(night.dec[unmatched_local], dtype=np.float64).tolist())
            new_cols = np.full((n_nights, n_new), -1, dtype=np.int64)
            new_cols[n] = unmatched_local
            star_index = np.concatenate([star_index, new_cols], axis=1)

        star_index[n, :n_global_before] = col
        n_matched[n] = matched_src.size

        if matched_dist.size:
            theta = 2.0 * np.arcsin(np.clip(np.median(matched_dist) / 2.0, 0.0, 1.0))
            median_sep_arcsec = float(np.degrees(theta) * 3600.0)
        else:
            median_sep_arcsec = float("nan")
        logger.info(
            "crossmatch %s: %d/%d matched to the global list (median sep %.3f arcsec), "
            "%d new stars",
            night.label, int(matched_src.size), int(local.size), median_sep_arcsec, n_new,
        )

    return NightCrossMatch(
        ra=np.asarray(global_ra, dtype=np.float64),
        dec=np.asarray(global_dec, dtype=np.float64),
        index=star_index,
        labels=labels,
        n_matched=n_matched,
    )


def core_crossmatch(xmatch: NightCrossMatch, n_core: int) -> NightCrossMatch:
    """The first ``n_core`` nights' cross-match, without the stars only later nights contain.

    ``xmatch`` must come from :func:`crossmatch_nights` with the core nights first:
    that function appends a night's unmatched stars after every earlier night's
    stars, so the global stars any core night contains are a prefix of the list, and
    this is that prefix -- identical to the cross-match of the core nights alone.

    Raises
    ------
    MultiNightError
        If the columns any core night contains are not a prefix of the list.
    """
    index = xmatch.index[:n_core]
    present = (index >= 0).any(axis=0)
    n_core_global = int(np.count_nonzero(present))
    if not np.all(present[:n_core_global]):
        msg = "core nights' stars are not a prefix of the global list (core must come first)"
        raise MultiNightError(msg)
    return NightCrossMatch(
        ra=xmatch.ra[:n_core_global],
        dec=xmatch.dec[:n_core_global],
        index=index[:, :n_core_global],
        labels=xmatch.labels[:n_core],
        n_matched=xmatch.n_matched[:n_core],
    )


def _project_to_global(index: np.ndarray, local_values: list[np.ndarray], fill) -> np.ndarray:
    """Per-night local arrays projected onto the global star axis (axis 0 -> night).

    ``local_values[n]`` has shape ``(n_stars_n, ...)``; the result has shape
    ``(n_nights, n_global, ...)``, filled with ``fill`` where night ``n``
    does not contain that global star.
    """
    n_nights, n_global = index.shape
    trailing_shape = local_values[0].shape[1:]
    out = np.full((n_nights, n_global, *trailing_shape), fill, dtype=local_values[0].dtype)
    for n in range(n_nights):
        idx_n = index[n]
        present = idx_n >= 0
        if np.any(present):
            out[n, present] = local_values[n][idx_n[present]]
    return out


def _gnomonic(
    ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (tangent-plane) standard coordinates of (ra_deg, dec_deg) about (ra0_deg, dec0_deg).

    Returns ``(xi, eta)`` in degrees.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra0 = np.radians(ra0_deg)
    dec0 = np.radians(dec0_deg)
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return np.degrees(xi), np.degrees(eta)


def _spatial_term_name(p: int, q: int) -> str:
    parts = []
    if p == 1:
        parts.append("xi")
    elif p > 1:
        parts.append(f"xi^{p}")
    if q == 1:
        parts.append("eta")
    elif q > 1:
        parts.append(f"eta^{q}")
    return "*".join(parts)


def _make_basis_terms(spatial_degree: int, mag_degree: int) -> tuple[str, ...]:
    """Human-readable basis-term names, e.g. ``("1", "xi", "eta", "xi^2", ..., "dm", "dm^2")``."""
    names = ["1"]
    for deg in range(1, spatial_degree + 1):
        for p in range(deg, -1, -1):
            q = deg - p
            names.append(_spatial_term_name(p, q))
    for k in range(1, mag_degree + 1):
        names.append("dm" if k == 1 else f"dm^{k}")
    return tuple(names)


def _make_seeing_basis_terms(mag_degree: int, crowding_degree: int) -> tuple[str, ...]:
    """Basis-term names of the pooled seeing surface ``beta(M, crowding)``.

    E.g. ``("1", "mc", "mc^2", "cc")`` for ``mag_degree=2, crowding_degree=1``
    -- ``"mc"``/``"cc"`` are the centred mean-magnitude/crowding regressors
    (see :func:`_evaluate_basis_term`'s ``variables``), additive (no
    mag*crowding cross terms), matching :func:`relphot.decorrelate.fit_coefficient_surface`'s
    convention.
    """
    names = ["1"]
    for k in range(1, mag_degree + 1):
        names.append("mc" if k == 1 else f"mc^{k}")
    for k in range(1, crowding_degree + 1):
        names.append("cc" if k == 1 else f"cc^{k}")
    return tuple(names)


def _evaluate_basis_term(name: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """``name`` (e.g. ``"xi^2"``, ``"xi*eta"``, ``"mc"``) evaluated against ``variables``."""
    reference = next(iter(variables.values()))
    if name == "1":
        return np.ones_like(reference, dtype=np.float64)
    value = np.ones_like(reference, dtype=np.float64)
    for factor in name.split("*"):
        base, _, power_str = factor.partition("^")
        power = int(power_str) if power_str else 1
        value = value * variables[base] ** power
    return value


def _design_matrix(variables: dict[str, np.ndarray], basis_terms: tuple[str, ...]) -> np.ndarray:
    """``(len(next(iter(variables.values()))), len(basis_terms))`` design matrix."""
    return np.column_stack([_evaluate_basis_term(t, variables) for t in basis_terms])


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Weighted normal-equations solve; ``None`` on a singular system."""
    xtwx = x.T @ (w[:, None] * x)
    xtwy = x.T @ (w * y)
    try:
        return np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        return None


def _fit_pooled_seeing(
    seeing_design: np.ndarray, d_fwhm: np.ndarray, residual: np.ndarray, weight: np.ndarray,
    rows_mask: np.ndarray,
) -> np.ndarray | None:
    """Pooled seeing-surface coefficients ``alpha``, stacked over every non-anchor night.

    One shared ``alpha`` (not per night -- "beta shared by all nights", see
    the module docstring) is fit by weighted least squares on the stacked
    rows ``{(n, star): rows_mask[n, star]}`` of every night ``n`` with
    ``d_fwhm[n] != 0`` (the anchor, and degenerately any other night whose
    ``F_n`` happens to equal ``F_anchor``, contribute an all-zero regressor
    row and are skipped rather than sent into a singular solve).
    ``seeing_design`` is ``(n_global, n_seeing_terms)`` (the star-only part
    ``beta(M, crowding)``'s design matrix, the same for every night);
    ``residual``/``weight``/``rows_mask`` are ``(n_nights, n_global)``.
    Returns ``None`` if no night contributes, or the stacked system is
    singular or under-determined.
    """
    n_nights = residual.shape[0]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for n in range(n_nights):
        if d_fwhm[n] == 0.0:
            continue
        rows = rows_mask[n]
        if not np.any(rows):
            continue
        xs.append(seeing_design[rows] * d_fwhm[n])
        ys.append(residual[n, rows])
        ws.append(weight[n, rows])
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)
    if x.shape[0] < x.shape[1] + 1:
        return None
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    return _weighted_lstsq(x, y, w)


def _weighted_mean_and_loo(
    w: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean over axis 0 (nights), and its leave-one-night-out version per row.

    ``w``/``v`` are ``(n_nights, n_global)``; entries with zero weight do not
    contribute. Returns ``(mean, loo_mean, loo_total_weight)``, each
    ``(n_global,)``/``(n_nights, n_global)``/``(n_nights, n_global)``.
    ``loo_total_weight`` doubles as ``1 / Var(loo_mean)``.
    """
    wtot = w.sum(axis=0)
    wsum = (w * v).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wtot > 0, wsum / wtot, np.nan)

    wtot_loo = wtot[None, :] - w
    wsum_loo = wsum[None, :] - w * v
    with np.errstate(invalid="ignore", divide="ignore"):
        loo_mean = np.where(wtot_loo > 0, wsum_loo / wtot_loo, np.nan)
    return mean, loo_mean, wtot_loo


def _bright_decile_mask(m_values: np.ndarray, pool_mask: np.ndarray) -> np.ndarray:
    """Boolean mask, same shape as ``pool_mask``, of its brightest 10% (by ``m_values``)."""
    idx = np.nonzero(pool_mask)[0]
    mask = np.zeros_like(pool_mask)
    if idx.size == 0:
        return mask
    order = idx[np.argsort(m_values[idx])]
    n_bright = max(1, int(np.ceil(0.1 * idx.size)))
    mask[order[:n_bright]] = True
    return mask


def _solve_floor_variance(d2: np.ndarray, s2sum: np.ndarray) -> float:
    """``V >= 0`` such that ``median(d2 / (s2sum + V)) == _MEDIAN_CHI2_1``, bisecting on
    ``[0, 1]`` mag^2.

    Returns 0 if the population is already at or below the target with
    ``V = 0``; returns 1.0 (mag^2) unreduced if the target cannot be reached
    within that bound (an unusually large night-to-night inconsistency).
    """

    def _median_stat(v: float) -> float:
        return float(np.median(d2 / (s2sum + v)))

    if _median_stat(0.0) <= _MEDIAN_CHI2_1:
        return 0.0
    lo, hi = 0.0, 1.0
    if _median_stat(hi) > _MEDIAN_CHI2_1:
        logger.warning("calibration-floor bisection did not reach its target within [0, 1] mag^2")
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _median_stat(mid) > _MEDIAN_CHI2_1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fit_floor_bins(
    mn: np.ndarray, z: np.ndarray, sn: np.ndarray, tie_fit: np.ndarray, tie_eval: np.ndarray,
    bin_idx: np.ndarray, n_bins: int, n_nights: int, min_bin_stars: int,
    *, warn_night_labels: tuple[str, ...] | None = None, aperture: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin calibration-floor variance solved on ``tie_fit``, evaluated (chi2) on ``tie_eval``.

    ``tie_fit``/``tie_eval`` are ``(n_nights, n_global)`` boolean masks (the
    same array gives the tautological in-sample chi2 used for
    ``chi2_after``; disjoint even/odd-global-id halves give the real,
    held-out ``chi2_holdout``). Per bin, every night pair with >=
    ``min_bin_stars`` common ``tie_fit`` stars in that bin contributes one
    variance estimate (:func:`_solve_floor_variance`); those are split into
    per-night floors by equal division (2 nights) or non-negative least
    squares (:func:`scipy.optimize.nnls`, >2 nights). A bin/night with no
    contributing pair is NaN.

    When ``warn_night_labels`` is given (the primary, non-holdout call
    only -- the holdout calls leave it ``None`` so they do not double-log),
    a night whose >2-nights NNLS solve clips its floor to exactly 0 despite
    contributing a pair in that bin logs a
    :func:`~logging.Logger.warning`: NNLS forces a non-negative solution, so
    an exact 0 here can mean the night-pair excess variances in that bin are
    not well described by the additive ``v_nk = f_n + f_k`` model (e.g. one
    pair's excess exceeds the sum of the other two), not that the night
    truly needs no floor -- a diagnostic for model inadequacy.

    Returns ``(floor[n_nights, n_bins]`` (mag, i.e. already square-rooted),
    ``chi2_by_night[n_nights]``, ``chi2_by_bin[n_nights, n_bins])``.
    """
    floor = np.full((n_nights, n_bins), np.nan)
    chi2_by_bin = np.full((n_nights, n_bins), np.nan)
    night_terms: list[list[np.ndarray]] = [[] for _ in range(n_nights)]

    for b in range(n_bins):
        pairs_b: list[tuple[int, int, float]] = []
        for n in range(n_nights):
            for k in range(n + 1, n_nights):
                common_fit = tie_fit[n] & tie_fit[k] & (bin_idx == b)
                if np.count_nonzero(common_fit) < min_bin_stars:
                    continue
                d = (mn[n, common_fit] - z[n, common_fit]) - (mn[k, common_fit] - z[k, common_fit])
                s2sum = sn[n, common_fit] ** 2 + sn[k, common_fit] ** 2
                v_nk = _solve_floor_variance(d**2, s2sum)
                pairs_b.append((n, k, v_nk))
        if not pairs_b:
            continue

        if n_nights == 2:
            _n0, _k0, v01 = pairs_b[0]
            f_b = np.array([v01 / 2.0, v01 / 2.0])
        else:
            design = np.zeros((len(pairs_b), n_nights))
            b_vec = np.zeros(len(pairs_b))
            for row, (n, k, v_nk) in enumerate(pairs_b):
                design[row, n] = 1.0
                design[row, k] = 1.0
                b_vec[row] = v_nk
            f_b, _resid_norm = nnls(design, b_vec)
            has_pair = design.sum(axis=0) > 0
            if warn_night_labels is not None:
                clipped = has_pair & (f_b == 0.0)
                for n_clip in np.nonzero(clipped)[0]:
                    logger.warning(
                        "aperture %s, bin %d: NNLS clipped night %s's calibration floor "
                        "to 0 (possible model inadequacy -- non-additive night-pair "
                        "excess variance)",
                        aperture, b, warn_night_labels[n_clip],
                    )
            f_b = np.where(has_pair, f_b, np.nan)
        floor[:, b] = f_b

        for n in range(n_nights):
            for k in range(n + 1, n_nights):
                if not (np.isfinite(f_b[n]) and np.isfinite(f_b[k])):
                    continue
                common_eval = tie_eval[n] & tie_eval[k] & (bin_idx == b)
                if np.count_nonzero(common_eval) == 0:
                    continue
                dd = (
                    (mn[n, common_eval] - z[n, common_eval])
                    - (mn[k, common_eval] - z[k, common_eval])
                )
                s2sum = sn[n, common_eval] ** 2 + sn[k, common_eval] ** 2
                terms = dd**2 / (s2sum + f_b[n] + f_b[k])
                night_terms[n].append(terms)
                night_terms[k].append(terms)
                chi2_bin_val = float(np.median(terms)) / _MEDIAN_CHI2_1
                chi2_by_bin[n, b] = chi2_bin_val
                chi2_by_bin[k, b] = chi2_bin_val

    with np.errstate(invalid="ignore"):
        floor = np.sqrt(floor)
    chi2_by_night = np.full(n_nights, np.nan)
    for n in range(n_nights):
        if night_terms[n]:
            pooled = np.concatenate(night_terms[n])
            chi2_by_night[n] = float(np.median(pooled)) / _MEDIAN_CHI2_1
    return floor, chi2_by_night, chi2_by_bin


def _nightly_mag_err(night: NightProducts) -> tuple[np.ndarray, np.ndarray]:
    """Nightly magnitude and its statistical error, per local star and aperture.

    ``m = -2.5 log10(nanmedian_t lc)`` over epochs with ``epoch_ok &
    frame_kept`` and finite ``lc``; NaN if fewer than 3 such epochs or the
    median is non-positive. ``s = 1.0857 * 1.2533 * rms / sqrt(n_epochs)``,
    NaN if ``n_epochs < 3``.
    """
    ok = night.epoch_ok & night.frame_kept[None, :]
    lc = np.where(ok[:, :, None], night.lc.astype(np.float64), np.nan)
    n_good = np.count_nonzero(np.isfinite(lc), axis=1)
    med = nanmedian_quiet(lc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(med)
    valid_mag = (n_good >= 3) & np.isfinite(med) & (med > 0)
    mag = np.where(valid_mag, mag, np.nan)

    n_epochs = night.n_epochs
    with np.errstate(invalid="ignore", divide="ignore"):
        err = 1.0857 * 1.2533 * night.rms / np.sqrt(n_epochs)
    err = np.where(n_epochs >= 3, err, np.nan)
    return mag, err.astype(np.float64)


@dataclass(slots=True)
class NightTie:
    """Per-(night, aperture) zero-point tie: fitted model and every intermediate product.

    ``coef`` is ``(n_nights, n_aper, n_coef)``; the anchor night's row is
    all zero. ``zp`` is ``Z_n(i)`` evaluated per star, NaN where that star
    is absent from night ``n``. ``mean_mag`` is the calibrated,
    all-nights-weighted mean magnitude ``M``; ``night_mag``/``night_mag_err``
    are the raw (uncalibrated) per-night magnitude ``m``/its statistical
    error ``s``. ``tie_star`` marks which (night, star) entries were used in
    the converged fit; ``rejected`` marks a star clipped at the whole-star
    level (excluded from every night), including one whose zero point ran away
    (``|Z| >`` :data:`_ZP_RUNAWAY_MAG`; its ``zp`` and ``mean_mag`` are NaN, never a
    diverged number). ``floor`` is the post-convergence
    empirical calibration floor, ``(n_nights, n_aper, n_bins)``, fit
    independently in each of ``floor_mag_centres``'s equal-count magnitude
    bins (NaN where a bin had too few common tie stars) -- see
    :meth:`floor_at` for the usable, interpolated/monotonic form.
    ``chi2_after`` evaluates that floor on the *same* stars used to fit it,
    so for 2 nights it is 1 by construction (a single pair/bin exactly
    determines its own floor) and is only a convergence sanity check, not
    independent evidence; ``chi2_holdout``/``chi2_holdout_bins`` are the
    real, non-tautological version -- floor fit on even-global-id tie
    stars and evaluated on odd-global-id ones and vice versa, averaged.
    ``resid_mad``/``resid_mad_bright`` are the tie-star leave-one-out
    residual scatter (mag), overall and for the brightest 10% by
    ``mean_mag``.

    ``seeing_coef`` is the pooled seeing-surface ``beta(M, crowding)``
    coefficients, ``(n_aper, n_seeing_terms)`` -- one surface shared by
    every night, not per-night (empty last axis if
    ``settings.use_seeing_term`` was ``False``). ``seeing_mag0``/
    ``seeing_crowd0`` are its per-aperture centring constants (subtracted
    from ``mean_mag``/``crowding`` before evaluating ``seeing_basis_terms``,
    NaN if unused). ``night_fwhm`` is each night's seeing measure ``F_n``
    (median FWHM, px, over kept frames); ``crowding`` is each global star's
    pooled (median-over-nights) crowding index
    (:func:`relphot.decorrelate.compute_crowding`). See
    :func:`evaluate_zero_point` for how these combine into
    ``beta(M, crowding) * (F_n - F_anchor)``.

    ``loose`` is ``(n_nights,)``: True for a night tied loosely to the fixed core
    frame by :func:`tie_loose_nights` (all False when there is none, and for a
    file written before the field existed). A loose night's ``coef`` row is its
    low-order surface written into the core basis (other terms zero), it has no
    seeing term, its ``floor`` is measured against the frame in the core nights'
    magnitude bins, and ``tie_star`` marks its fit stars; the core nights' entries,
    ``mean_mag``, ``rejected`` and ``floor_mag_centres`` are the core tie's own.
    """

    labels: tuple[str, ...]
    anchor_index: int
    coef: np.ndarray
    basis_terms: tuple[str, ...]
    xi: np.ndarray
    eta: np.ndarray
    centre_radec: tuple[float, float]
    scale_deg: float
    mag0: np.ndarray
    zp: np.ndarray
    mean_mag: np.ndarray
    night_mag: np.ndarray
    night_mag_err: np.ndarray
    tie_star: np.ndarray
    rejected: np.ndarray
    floor_mag_centres: np.ndarray
    floor: np.ndarray
    n_tie: np.ndarray
    resid_mad: np.ndarray
    resid_mad_bright: np.ndarray
    chi2_after: np.ndarray
    chi2_holdout: np.ndarray
    chi2_holdout_bins: np.ndarray
    n_iter: np.ndarray
    seeing_basis_terms: tuple[str, ...]
    seeing_coef: np.ndarray
    seeing_mag0: np.ndarray
    seeing_crowd0: np.ndarray
    night_fwhm: np.ndarray
    crowding: np.ndarray
    loose: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.loose is None:
            self.loose = np.zeros(len(self.labels), dtype=bool)

    def floor_at(self, night_index: int, aperture: int, mag) -> np.ndarray:
        """Calibration floor (mag) at magnitude ``mag``, night ``night_index``, ``aperture``.

        Linear interpolation between the finite entries of
        ``floor_mag_centres[aperture]``/``floor[night_index, aperture]``,
        constant beyond the end bins (:func:`numpy.interp`'s ordinary
        extrapolation), and forced non-decreasing from bright to faint
        (``np.maximum.accumulate``) so a single noisy bin cannot dip below
        its brighter neighbours. NaN everywhere if every bin is NaN.
        """
        mag_arr = np.asarray(mag, dtype=np.float64)
        centres = self.floor_mag_centres[aperture]
        values = self.floor[night_index, aperture]
        finite = np.isfinite(centres) & np.isfinite(values)
        if not np.any(finite):
            return np.full(mag_arr.shape, np.nan)
        order = np.argsort(centres[finite])
        c = centres[finite][order]
        v = np.maximum.accumulate(values[finite][order])
        return np.interp(mag_arr, c, v)


def tie_nights(
    nights: list[NightProducts], xmatch: NightCrossMatch, anchor_index: int,
    settings: MultiNightSettings,
) -> NightTie:
    """Fit the per-(night, aperture) zero-point tie described in the module docstring.

    Two guards keep a star whose nights disagree wildly (typically one night a
    few-mag-fainter measurement with a mag-sized error, as forced photometry gives) from
    breaking the fit: at the first iteration a tie row more than
    :data:`_TIE_DISCREPANCY_MAG` from the other nights' mean (beyond the night's median
    offset) is not a tie row, and a star whose zero point ever exceeds
    :data:`_ZP_RUNAWAY_MAG` is rejected with NaN ``zp``/``mean_mag``.

    Raises
    ------
    MultiNightError
        If, for some aperture, any night has fewer than ``settings.min_tie_stars``
        tie stars -- either before the alternating fit starts, or (after
        whole-star clipping) once it has converged.
    """
    n_nights = len(nights)
    n_global = xmatch.ra.shape[0]
    n_aper = nights[0].n_aper
    labels = xmatch.labels

    ra0 = float(nanmedian_quiet(xmatch.ra))
    dec0 = float(nanmedian_quiet(xmatch.dec))
    xi_raw, eta_raw = _gnomonic(xmatch.ra, xmatch.dec, ra0, dec0)
    scale_deg = float(max(np.max(np.abs(xi_raw)), np.max(np.abs(eta_raw)), 1e-12))
    xi = xi_raw / scale_deg
    eta = eta_raw / scale_deg

    basis_terms = _make_basis_terms(settings.spatial_degree, settings.mag_degree)
    n_coef = len(basis_terms)
    seeing_terms = (
        _make_seeing_basis_terms(settings.seeing_mag_degree, settings.seeing_crowding_degree)
        if settings.use_seeing_term
        else ()
    )
    n_seeing_coef = len(seeing_terms)

    local_mag_err = [_nightly_mag_err(night) for night in nights]
    night_mag = _project_to_global(
        xmatch.index, [m for m, _s in local_mag_err], np.nan
    ).astype(np.float64)
    night_mag_err = _project_to_global(
        xmatch.index, [s for _m, s in local_mag_err], np.nan
    ).astype(np.float64)
    night_pool = _project_to_global(
        xmatch.index, [night.comparison_mask for night in nights], False
    )

    # Per-night seeing measure F_n (median FWHM, px, over kept frames) and each
    # global star's pooled crowding index (median over the nights it appears
    # in) -- the two regressors of the pooled seeing term beta(M, crowding) *
    # (F_n - F_anchor); see the module docstring. Computed unconditionally
    # (cheap; a KDTree per night) so they always round-trip in NightTie, even
    # with settings.use_seeing_term=False.
    night_fwhm = np.array(
        [float(nanmedian_quiet(night.fwhm[night.frame_kept])) for night in nights],
        dtype=np.float64,
    )
    d_fwhm = night_fwhm - night_fwhm[anchor_index]
    local_crowding = [compute_crowding(night) for night in nights]
    crowding_global = nanmedian_quiet(
        _project_to_global(xmatch.index, local_crowding, np.nan), axis=0
    )
    logger.info(
        "per-night seeing F_n (median FWHM, px, kept frames): %s",
        {label: round(float(f), 3) for label, f in zip(labels, night_fwhm, strict=True)},
    )
    if settings.use_seeing_term and n_nights == 2 and settings.seeing_crowding_degree < 1:
        logger.warning(
            "use_seeing_term with exactly 2 nights and seeing_crowding_degree=0: the "
            "seeing term beta(M)*(F_n-F_anchor) is then fit from a single non-anchor "
            "night's tie stars against the same magnitude the per-night poly(M) term "
            "already uses, and is nearly degenerate with it; increase "
            "seeing_crowding_degree (crowding is what separates them) or set "
            "use_seeing_term=False",
        )

    finite = np.isfinite(night_mag) & np.isfinite(night_mag_err)
    count_finite = finite.sum(axis=0)  # (n_global, n_aper)

    coef = np.zeros((n_nights, n_aper, n_coef), dtype=np.float64)
    zp = np.full((n_nights, n_global, n_aper), np.nan, dtype=np.float64)
    mean_mag = np.full((n_global, n_aper), np.nan, dtype=np.float64)
    tie_star = np.zeros((n_nights, n_global, n_aper), dtype=bool)
    rejected = np.zeros((n_global, n_aper), dtype=bool)
    n_bins = settings.floor_n_bins
    floor_mag_centres = np.full((n_aper, n_bins), np.nan, dtype=np.float64)
    floor = np.full((n_nights, n_aper, n_bins), np.nan, dtype=np.float64)
    n_tie = np.zeros((n_nights, n_aper), dtype=np.int64)
    resid_mad = np.full((n_nights, n_aper), np.nan, dtype=np.float64)
    resid_mad_bright = np.full((n_nights, n_aper), np.nan, dtype=np.float64)
    chi2_after = np.full((n_nights, n_aper), np.nan, dtype=np.float64)
    chi2_holdout = np.full((n_nights, n_aper), np.nan, dtype=np.float64)
    chi2_holdout_bins = np.full((n_nights, n_aper, n_bins), np.nan, dtype=np.float64)
    n_iter = np.zeros(n_aper, dtype=np.int64)
    mag0 = np.full(n_aper, np.nan, dtype=np.float64)
    seeing_coef = np.zeros((n_aper, n_seeing_coef), dtype=np.float64)
    seeing_mag0 = np.full(n_aper, np.nan, dtype=np.float64)
    seeing_crowd0 = np.full(n_aper, np.nan, dtype=np.float64)

    for a in range(n_aper):
        mn = night_mag[:, :, a]
        sn = night_mag_err[:, :, a]
        pool = night_pool[:, :, a]
        finite_a = finite[:, :, a]

        tie_base = pool & finite_a & (count_finite[:, a] >= 2)
        base_counts = tie_base.sum(axis=1)
        short = [labels[n] for n in range(n_nights) if base_counts[n] < settings.min_tie_stars]
        if short:
            msg = (
                f"aperture {a}: fewer than {settings.min_tie_stars} candidate tie stars "
                f"for night(s) {short} (before rejection)"
            )
            raise MultiNightError(msg)

        w = np.where(finite_a, 1.0 / (sn**2 + settings.tie_err_floor_mag**2), 0.0)

        z = np.zeros((n_nights, n_global), dtype=np.float64)
        seeing_z = np.zeros((n_nights, n_global), dtype=np.float64)
        rej = np.zeros(n_global, dtype=bool)
        runaway_star = np.zeros(n_global, dtype=bool)
        mag0_a: float | None = None
        seeing_mag0_a: float | None = None
        seeing_crowd0_a: float | None = None
        alpha_a: np.ndarray | None = None
        it_used = 0
        delta = np.inf

        for it in range(1, settings.max_iter + 1):
            it_used = it
            v = np.where(finite_a, mn - z, 0.0)
            m_all, m_loo, wtot_loo = _weighted_mean_and_loo(w, v)

            if mag0_a is None:
                d_loo = np.where(tie_base, mn - m_loo, np.nan)
                off_n = nanmedian_quiet(d_loo, axis=1)
                tie_base = tie_base & ~(np.abs(d_loo - off_n[:, None]) > _TIE_DISCREPANCY_MAG)
                any_tie0 = tie_base.any(axis=0)
                mag0_a = float(nanmedian_quiet(np.where(any_tie0, m_all, np.nan)))
                mag0[a] = mag0_a
                if settings.use_seeing_term:
                    seeing_mag0_a = mag0_a
                    seeing_crowd0_a = float(
                        nanmedian_quiet(np.where(any_tie0, crowding_global, np.nan))
                    )

            tie_eff = tie_base & ~rej[None, :]
            z_poly = np.zeros((n_nights, n_global), dtype=np.float64)
            coef_a = coef[:, a, :].copy()
            for n in range(n_nights):
                if n == anchor_index:
                    continue
                dm_all = m_loo[n] - mag0_a
                design_all = _design_matrix({"xi": xi, "eta": eta, "dm": dm_all}, basis_terms)
                rows = tie_eff[n]
                if np.count_nonzero(rows) < n_coef + 1:
                    continue
                y = mn[n, rows] - m_loo[n, rows] - seeing_z[n, rows]
                fit = _weighted_lstsq(design_all[rows], y, w[n, rows])
                if fit is None:
                    continue
                coef_a[n] = fit
                z_poly[n] = design_all @ fit
            coef[:, a, :] = coef_a

            if settings.use_seeing_term:
                mag_c_pool = m_all - seeing_mag0_a
                crowd_c_pool = crowding_global - seeing_crowd0_a
                seeing_design_all = _design_matrix(
                    {"mc": mag_c_pool, "cc": crowd_c_pool}, seeing_terms
                )
                residual_pool = mn - m_loo - z_poly
                fit_alpha = _fit_pooled_seeing(seeing_design_all, d_fwhm, residual_pool, w, tie_eff)
                if fit_alpha is not None:
                    alpha_a = fit_alpha
                if alpha_a is not None:
                    beta_pool = seeing_design_all @ alpha_a
                    seeing_z_new = d_fwhm[:, None] * beta_pool[None, :]
                    seeing_z_new[anchor_index, :] = 0.0
                else:
                    seeing_z_new = np.zeros((n_nights, n_global), dtype=np.float64)
            else:
                seeing_z_new = np.zeros((n_nights, n_global), dtype=np.float64)

            z_new = z_poly + seeing_z_new
            # A star whose other nights disagree wildly with this one feeds its own
            # magnitude back through the dm**2 term and diverges; drop it, with every
            # night's zero point, before the number spreads.
            runaway_star |= (np.abs(z_new) > _ZP_RUNAWAY_MAG).any(axis=0)
            z_new = np.where(runaway_star[None, :], 0.0, z_new)
            seeing_z_new = np.where(runaway_star[None, :], 0.0, seeing_z_new)
            seeing_z = seeing_z_new

            active = tie_eff & ~runaway_star[None, :]
            if np.any(active):
                delta = float(np.nanmax(np.abs(np.where(active, z_new - z, 0.0))))
            else:
                delta = 0.0
            z = z_new

            resid = mn - z - m_loo
            with np.errstate(invalid="ignore", divide="ignore"):
                var_loo = np.where(wtot_loo > 0, 1.0 / wtot_loo, np.nan)
                denom = np.sqrt(sn**2 + var_loo + settings.tie_err_floor_mag**2)
                z_score = resid / denom
            z_masked = np.where(tie_eff, z_score, np.nan)
            any_tie_i = tie_eff.any(axis=0)
            pop = z_masked[tie_eff]
            pop_sigma = float(mad_sigma(pop)) if pop.size else np.nan
            if np.isfinite(pop_sigma) and pop_sigma > 0:
                chi = np.full(n_global, np.nan)
                if np.any(any_tie_i):
                    chi[any_tie_i] = np.nanmax(np.abs(z_masked[:, any_tie_i]), axis=0)
                robust_chi = chi / pop_sigma
                newly_rejected = (
                    any_tie_i & np.isfinite(robust_chi) & (robust_chi > settings.clip_sigma)
                )
            else:
                newly_rejected = np.zeros(n_global, dtype=bool)
            rej = rej | newly_rejected | runaway_star

            if delta < settings.tol_mag:
                break
        else:
            logger.warning(
                "aperture %d: tie fit did not converge in %d iterations (last max|dZ|=%.2e)",
                a, settings.max_iter, delta,
            )

        # Final, self-consistent pass at the converged Z and rejected set.
        v = np.where(finite_a, mn - z, 0.0)
        m_all, m_loo, wtot_loo = _weighted_mean_and_loo(w, v)
        tie_eff = tie_base & ~rej[None, :]
        final_counts = tie_eff.sum(axis=1)
        short_final = [
            labels[n] for n in range(n_nights) if final_counts[n] < settings.min_tie_stars
        ]
        if short_final:
            msg = (
                f"aperture {a}: fewer than {settings.min_tie_stars} tie stars "
                f"for night(s) {short_final} after whole-star rejection"
            )
            raise MultiNightError(msg)

        resid = mn - z - m_loo
        absent = xmatch.index < 0  # (n_nights, n_global)
        if np.any(runaway_star):
            logger.warning(
                "aperture %d: %d star(s) whose zero point ran away (|Z| > %.1f mag) are "
                "excluded (rejected, NaN zp and mean_mag)", a, int(np.count_nonzero(runaway_star)),
                _ZP_RUNAWAY_MAG,
            )
        zp[:, :, a] = np.where(absent | runaway_star[None, :], np.nan, z)
        mean_mag[:, a] = np.where(runaway_star, np.nan, m_all)
        tie_star[:, :, a] = tie_eff
        rejected[:, a] = rej
        n_tie[:, a] = tie_eff.sum(axis=1)
        n_iter[a] = it_used
        seeing_coef[a, :] = alpha_a if alpha_a is not None else 0.0
        seeing_mag0[a] = seeing_mag0_a if seeing_mag0_a is not None else np.nan
        seeing_crowd0[a] = seeing_crowd0_a if seeing_crowd0_a is not None else np.nan

        for n in range(n_nights):
            rows = tie_eff[n]
            resid_n = resid[n, rows]
            resid_mad[n, a] = float(mad_sigma(resid_n)) if resid_n.size else np.nan
            bright_mask = _bright_decile_mask(mean_mag[:, a], rows)
            resid_bright = resid[n, bright_mask]
            resid_mad_bright[n, a] = float(mad_sigma(resid_bright)) if resid_bright.size else np.nan

        # Calibration floor: a magnitude-dependent additive term per night,
        # fit in equal-count M bins (see _fit_floor_bins). chi2_after
        # evaluates that SAME (fit-on-everything) floor on the SAME stars
        # used to fit it -- close to 1 by construction (see NightTie's
        # docstring) -- while chi2_holdout below is the real, held-out
        # check (fit on one global-id parity half, evaluate on the other,
        # averaged both ways).
        any_tie_i = tie_eff.any(axis=0)
        if np.any(any_tie_i):
            edges = np.quantile(mean_mag[any_tie_i, a], np.linspace(0.0, 1.0, n_bins + 1))
        else:
            edges = np.linspace(-1.0, 1.0, n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        floor_mag_centres[a, :] = centres
        bin_idx = np.clip(np.searchsorted(edges, mean_mag[:, a], side="right") - 1, 0, n_bins - 1)

        floor_a, chi2_after_a, _chi2_bin_full = _fit_floor_bins(
            mn, z, sn, tie_eff, tie_eff, bin_idx, n_bins, n_nights, settings.floor_min_bin_stars,
            warn_night_labels=labels, aperture=a,
        )
        floor[:, a, :] = floor_a
        chi2_after[:, a] = chi2_after_a
        if not np.any(np.isfinite(floor_a)):
            logger.warning(
                "aperture %d: no (night pair, magnitude bin) has >= %d common tie stars; "
                "calibration floor unavailable", a, settings.floor_min_bin_stars,
            )
        else:
            mean_floor_mmag = np.nanmean(floor_a, axis=0) * 1000.0
            logger.info(
                "aperture %d: calibration floor, brightest -> faintest bin (mmag): %s",
                a, np.array2string(mean_floor_mmag, precision=2),
            )

        parity_even = (np.arange(n_global) % 2) == 0
        tie_even = tie_eff & parity_even[None, :]
        tie_odd = tie_eff & ~parity_even[None, :]
        min_split = max(10, settings.floor_min_bin_stars // 2)
        _floor_e, chi2_e, chi2_bin_e = _fit_floor_bins(
            mn, z, sn, tie_even, tie_odd, bin_idx, n_bins, n_nights, min_split,
        )
        _floor_o, chi2_o, chi2_bin_o = _fit_floor_bins(
            mn, z, sn, tie_odd, tie_even, bin_idx, n_bins, n_nights, min_split,
        )
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            chi2_holdout[:, a] = np.nanmean(np.stack([chi2_e, chi2_o]), axis=0)
            chi2_holdout_bins[:, a, :] = np.nanmean(np.stack([chi2_bin_e, chi2_bin_o]), axis=0)

    return NightTie(
        labels=labels,
        anchor_index=anchor_index,
        coef=coef,
        basis_terms=basis_terms,
        xi=xi,
        eta=eta,
        centre_radec=(ra0, dec0),
        scale_deg=scale_deg,
        mag0=mag0,
        zp=zp,
        mean_mag=mean_mag,
        night_mag=night_mag,
        night_mag_err=night_mag_err,
        tie_star=tie_star,
        rejected=rejected,
        floor_mag_centres=floor_mag_centres,
        floor=floor,
        n_tie=n_tie,
        resid_mad=resid_mad,
        resid_mad_bright=resid_mad_bright,
        chi2_after=chi2_after,
        chi2_holdout=chi2_holdout,
        chi2_holdout_bins=chi2_holdout_bins,
        n_iter=n_iter,
        seeing_basis_terms=seeing_terms,
        seeing_coef=seeing_coef,
        seeing_mag0=seeing_mag0,
        seeing_crowd0=seeing_crowd0,
        night_fwhm=night_fwhm,
        crowding=crowding_global,
    )


def _pad_stars(arr: np.ndarray, axis: int, n_pad: int, fill) -> np.ndarray:
    """``arr`` extended by ``n_pad`` entries of ``fill`` at the end of ``axis`` (the star axis)."""
    shape = list(arr.shape)
    shape[axis] = n_pad
    return np.concatenate([arr, np.full(shape, fill, dtype=arr.dtype)], axis=axis)


def _loose_floor_bins(
    r: np.ndarray, s2: np.ndarray, bin_idx: np.ndarray, n_bins: int, min_bin_stars: int
) -> np.ndarray:
    """Per-bin calibration floor (mag) of one loose night against the core frame.

    ``r`` are its residuals against the frame, ``s2`` their variance without any floor
    (its own statistical variance plus the frame's). Per bin the floor variance ``V >=
    0`` solves ``median(r**2 / (s2 + V)) == median(chi^2_1)``
    (:func:`_solve_floor_variance`), the same statistic the core floor uses. NaN for a
    bin with fewer than ``min_bin_stars`` stars.
    """
    floor = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = bin_idx == b
        if np.count_nonzero(sel) < min_bin_stars:
            continue
        floor[b] = np.sqrt(_solve_floor_variance(r[sel] ** 2, s2[sel]))
    return floor


def _loose_chi2(
    r: np.ndarray, s2: np.ndarray, bin_idx: np.ndarray, floor: np.ndarray, n_bins: int
) -> tuple[float, np.ndarray]:
    """Median ``r**2 / (s2 + floor**2)`` over the stars, ``/ median(chi^2_1)``: pooled, per bin.

    A bin with no finite floor or no star is left out (NaN per bin; NaN pooled if every
    bin is).
    """
    chi2_bins = np.full(n_bins, np.nan)
    pooled: list[np.ndarray] = []
    for b in range(n_bins):
        sel = bin_idx == b
        if not np.isfinite(floor[b]) or not np.any(sel):
            continue
        terms = r[sel] ** 2 / (s2[sel] + floor[b] ** 2)
        pooled.append(terms)
        chi2_bins[b] = float(np.median(terms)) / _MEDIAN_CHI2_1
    if not pooled:
        return float("nan"), chi2_bins
    return float(np.median(np.concatenate(pooled))) / _MEDIAN_CHI2_1, chi2_bins


def _core_frame(tie_core: NightTie, aperture: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The core nights' fixed frame at ``aperture``: ``(mag, err, n_nights)`` per core star.

    ``mag`` is the weighted mean over the core nights of ``night_mag - zp`` with
    weights ``1 / (s**2 + floor**2)`` (``floor`` = :meth:`NightTie.floor_at` at the
    star's ``mean_mag``; 0 where unavailable), ``err = 1 / sqrt(sum of weights)``,
    ``n_nights`` the number of core nights that contribute. NaN/0 where none does.
    """
    n_core = tie_core.zp.shape[0]
    a = aperture
    cal = tie_core.night_mag[:, :, a] - tie_core.zp[:, :, a]
    s = tie_core.night_mag_err[:, :, a]
    m = tie_core.mean_mag[:, a]
    floor = np.stack([tie_core.floor_at(n, a, m) for n in range(n_core)])
    floor = np.where(np.isfinite(floor), floor, 0.0)
    var = s**2 + floor**2
    ok = np.isfinite(cal) & np.isfinite(var) & (var > 0)
    w = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)
    wtot = w.sum(axis=0)
    has = wtot > 0
    safe_wtot = np.where(has, wtot, 1.0)
    frame = np.where(has, (w * np.where(ok, cal, 0.0)).sum(axis=0) / safe_wtot, np.nan)
    err = np.where(has, 1.0 / np.sqrt(safe_wtot), np.nan)
    return frame, err, ok.sum(axis=0)


def tie_loose_nights(
    core: list[NightProducts], loose: list[NightProducts], xmatch: NightCrossMatch,
    tie_core: NightTie, settings: MultiNightSettings,
) -> NightTie:
    """Tie ``loose`` nights to the fixed core frame of ``tie_core``; return the extended tie.

    ``xmatch`` is the full cross-match of ``core + loose`` (core first; the core
    tie ``tie_core`` came from :func:`tie_nights` on ``core`` and
    :func:`core_crossmatch`). The core nights' entries of the result are
    ``tie_core``'s own, padded with NaN/False for the stars only a loose night
    contains -- nothing about them is refit. For each loose night and aperture:

    1. The frame is :func:`_core_frame`; the fit stars are the loose night's
       comparison stars with a finite nightly magnitude and error, at least 2 core
       nights in the frame and no whole-star rejection in the core tie.
    2. ``m_loose - frame`` is fit by weighted least squares (weights ``1 / (s**2 +
       frame_err**2 + tie_err_floor_mag**2)``) with the low-order polynomial
       ``settings.loose_spatial_degree`` / ``loose_mag_degree`` in the core tie's
       ``xi``, ``eta`` and ``dm = mean_mag - mag0``, clipping stars beyond
       ``settings.clip_sigma`` robust sigmas until the set is stable. No seeing term.
       Before the first fit, a star more than :data:`_ZP_RUNAWAY_MAG` from the night's
       median offset to the frame (or with a non-finite one) is left out, and a star
       whose evaluated zero point is non-finite or beyond :data:`_ZP_ABSURD_MAG` gets
       a NaN ``zp`` and is not a fit star.
       The surface is stored in the core basis (unused terms zero), so
       :func:`evaluate_zero_point` evaluates it unchanged.
    3. The calibration floor is measured per bin of the core tie's magnitude bins
       (edges recomputed from ``tie_core``) by :func:`_loose_floor_bins` on the
       residuals, with the frame's error counted in the noise, not by the pairwise
       non-negative solve of the core: the frame is fixed, so the loose night's floor
       is the only unknown.

    Raises
    ------
    MultiNightError
        If a loose surface term is not in the core basis, or a loose night has fewer
        than ``settings.min_tie_stars`` fit stars (or a singular fit) in some aperture.
    """
    n_loose = len(loose)
    if n_loose == 0:
        return tie_core
    n_core = len(core)
    n_global = int(xmatch.ra.shape[0])
    n_core_global = int(tie_core.mean_mag.shape[0])
    n_pad = n_global - n_core_global
    n_aper = int(tie_core.coef.shape[1])
    n_coef = int(tie_core.coef.shape[2])
    n_bins = int(tie_core.floor.shape[2])

    loose_terms = _make_basis_terms(settings.loose_spatial_degree, settings.loose_mag_degree)
    missing_terms = [t for t in loose_terms if t not in tie_core.basis_terms]
    if missing_terms:
        msg = (
            f"loose surface terms {missing_terms} are not in the core basis "
            f"{tie_core.basis_terms}"
        )
        raise MultiNightError(msg)
    term_cols = [tie_core.basis_terms.index(t) for t in loose_terms]
    n_fit_min = max(settings.min_tie_stars, len(loose_terms) + 1)
    min_split = max(10, settings.floor_min_bin_stars // 2)

    ra0, dec0 = tie_core.centre_radec
    xi_raw, eta_raw = _gnomonic(xmatch.ra[n_core_global:], xmatch.dec[n_core_global:], ra0, dec0)
    xi = np.concatenate([tie_core.xi, xi_raw / tie_core.scale_deg])
    eta = np.concatenate([tie_core.eta, eta_raw / tie_core.scale_deg])

    coef = np.concatenate([tie_core.coef, np.zeros((n_loose, n_aper, n_coef))], axis=0)
    zp = np.concatenate(
        [_pad_stars(tie_core.zp, 1, n_pad, np.nan),
         np.full((n_loose, n_global, n_aper), np.nan)], axis=0,
    )
    night_mag = np.concatenate(
        [_pad_stars(tie_core.night_mag, 1, n_pad, np.nan),
         np.full((n_loose, n_global, n_aper), np.nan)], axis=0,
    )
    night_mag_err = np.concatenate(
        [_pad_stars(tie_core.night_mag_err, 1, n_pad, np.nan),
         np.full((n_loose, n_global, n_aper), np.nan)], axis=0,
    )
    tie_star = np.concatenate(
        [_pad_stars(tie_core.tie_star, 1, n_pad, False),
         np.zeros((n_loose, n_global, n_aper), dtype=bool)], axis=0,
    )
    mean_mag = _pad_stars(tie_core.mean_mag, 0, n_pad, np.nan)
    rejected = _pad_stars(tie_core.rejected, 0, n_pad, False)
    floor = np.concatenate([tie_core.floor, np.full((n_loose, n_aper, n_bins), np.nan)], axis=0)
    n_tie = np.concatenate([tie_core.n_tie, np.zeros((n_loose, n_aper), dtype=np.int64)], axis=0)
    nan_rows = np.full((n_loose, n_aper), np.nan)
    resid_mad = np.concatenate([tie_core.resid_mad, nan_rows], axis=0)
    resid_mad_bright = np.concatenate([tie_core.resid_mad_bright, nan_rows], axis=0)
    chi2_after = np.concatenate([tie_core.chi2_after, nan_rows], axis=0)
    chi2_holdout = np.concatenate([tie_core.chi2_holdout, nan_rows], axis=0)
    chi2_holdout_bins = np.concatenate(
        [tie_core.chi2_holdout_bins, np.full((n_loose, n_aper, n_bins), np.nan)], axis=0
    )
    night_fwhm = np.concatenate([
        tie_core.night_fwhm,
        [float(nanmedian_quiet(night.fwhm[night.frame_kept])) for night in loose],
    ])
    crowding = _pad_stars(tie_core.crowding, 0, n_pad, np.nan)

    for j, night in enumerate(loose):
        n = n_core + j
        index_n = xmatch.index[n : n + 1]
        m_local, s_local = _nightly_mag_err(night)
        night_mag[n] = _project_to_global(index_n, [m_local], np.nan)[0]
        night_mag_err[n] = _project_to_global(index_n, [s_local], np.nan)[0]
        comparison = _project_to_global(index_n, [night.comparison_mask], False)[0]
        present = xmatch.index[n] >= 0

        for a in range(n_aper):
            frame_core, frame_err_core, n_frame_core = _core_frame(tie_core, a)
            frame = _pad_stars(frame_core, 0, n_pad, np.nan)
            frame_err = _pad_stars(frame_err_core, 0, n_pad, np.nan)
            n_frame = _pad_stars(n_frame_core, 0, n_pad, 0)
            dm_all = mean_mag[:, a] - tie_core.mag0[a]

            zmask = present & np.isfinite(frame) & np.isfinite(dm_all)
            gid = np.nonzero(zmask)[0]
            x_z = _design_matrix(
                {"xi": xi[gid], "eta": eta[gid], "dm": dm_all[gid]}, loose_terms
            )
            mn = night_mag[n, gid, a]
            sn = night_mag_err[n, gid, a]
            fr = frame[gid]
            fe = frame_err[gid]
            var0 = sn**2 + fe**2
            err2 = var0 + settings.tie_err_floor_mag**2
            base = (
                comparison[gid, a] & np.isfinite(mn) & np.isfinite(sn) & (n_frame[gid] >= 2)
                & ~rejected[gid, a]
            )
            d_off = mn - fr
            base &= np.isfinite(d_off)
            if np.any(base):
                # a star far from the night's own offset to the frame (or with a garbage
                # frame) is not a calibrator, and one such row ruins the first fit
                off = float(np.median(d_off[base]))
                base &= np.abs(d_off - off) <= _ZP_RUNAWAY_MAG
            rows = base.copy()
            fit = None
            for _ in range(settings.max_iter):
                if np.count_nonzero(rows) < n_fit_min:
                    msg = (
                        f"aperture {a}: fewer than {n_fit_min} fit stars for loose night "
                        f"{night.label} against the core frame"
                    )
                    raise MultiNightError(msg)
                fit = _weighted_lstsq(x_z[rows], (mn - fr)[rows], 1.0 / err2[rows])
                if fit is None or not np.all(np.isfinite(fit)):
                    msg = f"aperture {a}: singular loose-night fit for {night.label}"
                    raise MultiNightError(msg)
                r = mn - fr - x_z @ fit
                z_score = r / np.sqrt(err2)
                sigma = float(mad_sigma(z_score[rows]))
                if not (np.isfinite(sigma) and sigma > 0):
                    break
                new_rows = base & (np.abs(z_score) < settings.clip_sigma * sigma)
                if np.array_equal(new_rows, rows):
                    break
                rows = new_rows
            if np.count_nonzero(rows) < n_fit_min:
                msg = (
                    f"aperture {a}: fewer than {n_fit_min} fit stars for loose night "
                    f"{night.label} against the core frame"
                )
                raise MultiNightError(msg)

            coef[n, a, term_cols] = fit
            zp_fit = x_z @ fit
            sane = np.isfinite(zp_fit) & (np.abs(zp_fit) <= _ZP_ABSURD_MAG)
            if not np.all(sane):
                logger.warning(
                    "aperture %d: loose night %s: %d star(s) with a non-finite or absurd "
                    "zero point (|Z| > %.0f mag) are excluded (NaN zp)",
                    a, night.label, int(np.count_nonzero(~sane)), _ZP_ABSURD_MAG,
                )
            rows = rows & sane
            zp[n, gid, a] = np.where(sane, zp_fit, np.nan)
            tie_star[n, gid[rows], a] = True
            n_tie[n, a] = int(np.count_nonzero(rows))

            r = mn - fr - x_z @ fit
            m_z = mean_mag[gid, a]
            resid_mad[n, a] = float(mad_sigma(r[rows]))
            bright = _bright_decile_mask(m_z, rows)
            resid_mad_bright[n, a] = float(mad_sigma(r[bright])) if np.any(bright) else np.nan

            any_tie = tie_core.tie_star[:, :, a].any(axis=0)
            if np.any(any_tie):
                edges = np.quantile(
                    tie_core.mean_mag[any_tie, a], np.linspace(0.0, 1.0, n_bins + 1)
                )
            else:
                edges = np.linspace(-1.0, 1.0, n_bins + 1)
            bin_z = np.clip(np.searchsorted(edges, m_z, side="right") - 1, 0, n_bins - 1)

            r_f, s2_f, bin_f, gid_f = r[rows], var0[rows], bin_z[rows], gid[rows]
            floor_a = _loose_floor_bins(r_f, s2_f, bin_f, n_bins, settings.floor_min_bin_stars)
            floor[n, a] = floor_a
            chi2_after[n, a], _chi2_bins = _loose_chi2(r_f, s2_f, bin_f, floor_a, n_bins)
            if not np.any(np.isfinite(floor_a)):
                logger.warning(
                    "aperture %d: loose night %s has no magnitude bin with >= %d fit stars; "
                    "calibration floor unavailable", a, night.label, settings.floor_min_bin_stars,
                )
            else:
                logger.info(
                    "aperture %d: loose night %s: %d fit stars, calibration floor, brightest "
                    "-> faintest bin (mmag): %s", a, night.label, n_tie[n, a],
                    np.array2string(floor_a * 1000.0, precision=2),
                )

            even = (gid_f % 2) == 0
            floor_e = _loose_floor_bins(r_f[even], s2_f[even], bin_f[even], n_bins, min_split)
            floor_o = _loose_floor_bins(r_f[~even], s2_f[~even], bin_f[~even], n_bins, min_split)
            chi2_e, chi2_bin_e = _loose_chi2(
                r_f[~even], s2_f[~even], bin_f[~even], floor_e, n_bins
            )
            chi2_o, chi2_bin_o = _loose_chi2(r_f[even], s2_f[even], bin_f[even], floor_o, n_bins)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                chi2_holdout[n, a] = np.nanmean([chi2_e, chi2_o])
                chi2_holdout_bins[n, a] = np.nanmean(np.stack([chi2_bin_e, chi2_bin_o]), axis=0)

    return NightTie(
        labels=xmatch.labels,
        anchor_index=tie_core.anchor_index,
        coef=coef,
        basis_terms=tie_core.basis_terms,
        xi=xi,
        eta=eta,
        centre_radec=tie_core.centre_radec,
        scale_deg=tie_core.scale_deg,
        mag0=tie_core.mag0,
        zp=zp,
        mean_mag=mean_mag,
        night_mag=night_mag,
        night_mag_err=night_mag_err,
        tie_star=tie_star,
        rejected=rejected,
        floor_mag_centres=tie_core.floor_mag_centres,
        floor=floor,
        n_tie=n_tie,
        resid_mad=resid_mad,
        resid_mad_bright=resid_mad_bright,
        chi2_after=chi2_after,
        chi2_holdout=chi2_holdout,
        chi2_holdout_bins=chi2_holdout_bins,
        n_iter=tie_core.n_iter,
        seeing_basis_terms=tie_core.seeing_basis_terms,
        seeing_coef=tie_core.seeing_coef,
        seeing_mag0=tie_core.seeing_mag0,
        seeing_crowd0=tie_core.seeing_crowd0,
        night_fwhm=night_fwhm,
        crowding=crowding,
        loose=np.concatenate([np.zeros(n_core, dtype=bool), np.ones(n_loose, dtype=bool)]),
    )


def evaluate_zero_point(
    tie: NightTie, night_index: int, aperture: int, xi: np.ndarray, eta: np.ndarray,
    mag: np.ndarray, crowding: np.ndarray | None = None,
) -> np.ndarray:
    """``Z_n(xi, eta, mag)`` of the fitted model in ``tie``, for arbitrary star
    positions/magnitudes.

    Zero for the anchor night. ``xi``/``eta`` must already be in the same
    scaled tangent-plane system as ``tie.xi``/``tie.eta`` (divide raw
    gnomonic coordinates, in degrees, by ``tie.scale_deg``).

    ``crowding`` (:func:`relphot.decorrelate.compute_crowding`'s
    log10-arcsec index, one per star) adds the pooled seeing term
    ``beta(mag, crowding) * (F_n - F_anchor)`` -- see the module docstring
    and :func:`tie_nights`. It is required whenever ``tie.seeing_basis_terms``
    is non-empty (``settings.use_seeing_term`` was ``True``); with an empty
    ``tie.seeing_basis_terms`` (``use_seeing_term=False``) it is ignored. A loose
    night (``tie.loose[night_index]``) has no seeing term either, so ``crowding``
    is ignored for it.
    """
    xi = np.asarray(xi, dtype=np.float64)
    if night_index == tie.anchor_index:
        return np.zeros_like(xi)
    eta = np.asarray(eta, dtype=np.float64)
    dm = np.asarray(mag, dtype=np.float64) - tie.mag0[aperture]
    design = _design_matrix({"xi": xi, "eta": eta, "dm": dm}, tie.basis_terms)
    z = design @ tie.coef[night_index, aperture]

    if tie.seeing_basis_terms and not tie.loose[night_index]:
        if crowding is None:
            msg = "tie has a fitted seeing term (use_seeing_term=True); crowding is required"
            raise MultiNightError(msg)
        mag_c = np.asarray(mag, dtype=np.float64) - tie.seeing_mag0[aperture]
        crowd_c = np.asarray(crowding, dtype=np.float64) - tie.seeing_crowd0[aperture]
        seeing_design = _design_matrix({"mc": mag_c, "cc": crowd_c}, tie.seeing_basis_terms)
        beta = seeing_design @ tie.seeing_coef[aperture]
        d_fwhm = tie.night_fwhm[night_index] - tie.night_fwhm[tie.anchor_index]
        z = z + beta * d_fwhm
    return z


@dataclass(slots=True)
class MultiNightLightCurves:
    """Multi-night calibrated light curves at each star's chosen (single, cross-night) aperture.

    ``aperture[i] = -1`` means star ``i`` has no usable aperture (no finite
    RMS in any night); every per-star output is NaN for it.
    ``night_of_frame``/``frame_in_night`` locate each concatenated epoch in
    ``(night, frame index within that night)``. ``mag`` is the fully
    calibrated epoch magnitude (zero-point corrected); ``flux_norm`` is only
    per-night median-normalised (no zero-point applied), for time-domain
    (e.g. transit) work where a per-night additive offset in flux ratio
    would matter more than in log-magnitude.
    """

    labels: tuple[str, ...]
    night_of_frame: np.ndarray
    frame_in_night: np.ndarray
    bjd_tdb: np.ndarray
    airmass: np.ndarray
    fwhm: np.ndarray
    aperture: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray
    flux_norm: np.ndarray
    flux_norm_err: np.ndarray
    night_mean_mag: np.ndarray
    night_mean_err: np.ndarray
    mean_mag: np.ndarray
    n_nights: np.ndarray


def _choose_aperture(
    nights: list[NightProducts], xmatch: NightCrossMatch, settings: MultiNightSettings,
    loose: np.ndarray,
) -> np.ndarray:
    """Per global star, the single aperture used in every night (see the module docstring).

    Loose nights do not vote: the choice (the aperture of the lowest median RMS over
    nights) uses the core nights only, so it is the one a run without the loose
    nights makes, and a star only a loose night contains has no aperture.
    """
    n_aper = nights[0].n_aper
    n_global = xmatch.ra.shape[0]
    if settings.aperture >= 0:
        if settings.aperture >= n_aper:
            msg = f"settings.multinight.aperture={settings.aperture} out of range [0, {n_aper})"
            raise MultiNightError(msg)
        return np.full(n_global, settings.aperture, dtype=np.int64)

    night_rms = _project_to_global(xmatch.index, [night.rms for night in nights], np.nan)
    night_rms[loose] = np.nan
    med_rms = nanmedian_quiet(night_rms, axis=0)  # (n_global, n_aper)
    has_finite = np.isfinite(med_rms).any(axis=1)
    med_rms_filled = np.where(np.isfinite(med_rms), med_rms, np.inf)
    aperture = np.argmin(med_rms_filled, axis=1)
    return np.where(has_finite, aperture, -1).astype(np.int64)


def build_multinight_lightcurves(
    nights: list[NightProducts], xmatch: NightCrossMatch, tie: NightTie,
    settings: MultiNightSettings,
) -> MultiNightLightCurves:
    """Assemble multi-night calibrated light curves from a converged :class:`NightTie`."""
    n_global = xmatch.ra.shape[0]
    labels = xmatch.labels

    aperture = _choose_aperture(nights, xmatch, settings, tie.loose)
    has_aper = aperture >= 0
    aper_safe = np.where(has_aper, aperture, 0)

    night_mag_sel = np.take_along_axis(tie.night_mag, aper_safe[None, :, None], axis=2)[:, :, 0]
    zp_sel = np.take_along_axis(tie.zp, aper_safe[None, :, None], axis=2)[:, :, 0]
    err_sel = np.take_along_axis(tie.night_mag_err, aper_safe[None, :, None], axis=2)[:, :, 0]
    night_mag_sel = np.where(has_aper[None, :], night_mag_sel, np.nan)
    zp_sel = np.where(has_aper[None, :], zp_sel, np.nan)
    err_sel = np.where(has_aper[None, :], err_sel, np.nan)

    mean_mag = np.where(has_aper, tie.mean_mag[np.arange(n_global), aper_safe], np.nan)

    n_nights = len(nights)
    floor_sel = np.full((n_nights, n_global), np.nan, dtype=np.float64)
    for a_val in np.unique(aper_safe[has_aper]):
        cols = has_aper & (aperture == a_val)
        for n in range(n_nights):
            floor_sel[n, cols] = tie.floor_at(n, int(a_val), mean_mag[cols])

    with np.errstate(invalid="ignore"):
        night_mean_mag = night_mag_sel - zp_sel
        night_mean_err = np.sqrt(err_sel**2 + floor_sel**2)
    night_mean_mag = np.where(has_aper[None, :], night_mean_mag, np.nan)
    night_mean_err = np.where(has_aper[None, :], night_mean_err, np.nan)

    n_nights_out = np.count_nonzero(np.isfinite(night_mag_sel), axis=0).astype(np.int64)

    frame_counts = [night.n_frames for night in nights]
    frame_offsets = np.concatenate([[0], np.cumsum(frame_counts)])
    n_total_frames = int(frame_offsets[-1])

    night_of_frame = np.empty(n_total_frames, dtype=np.int64)
    frame_in_night = np.empty(n_total_frames, dtype=np.int64)
    bjd_tdb = np.empty(n_total_frames, dtype=np.float64)
    airmass = np.empty(n_total_frames, dtype=np.float64)
    fwhm = np.empty(n_total_frames, dtype=np.float64)
    mag = np.full((n_global, n_total_frames), np.nan, dtype=np.float32)
    mag_err = np.full((n_global, n_total_frames), np.nan, dtype=np.float32)
    flux_norm = np.full((n_global, n_total_frames), np.nan, dtype=np.float32)
    flux_norm_err = np.full((n_global, n_total_frames), np.nan, dtype=np.float32)

    for n, night in enumerate(nights):
        lo, hi = int(frame_offsets[n]), int(frame_offsets[n + 1])
        night_of_frame[lo:hi] = n
        frame_in_night[lo:hi] = np.arange(night.n_frames)
        bjd_tdb[lo:hi] = night.bjd_tdb
        airmass[lo:hi] = night.airmass
        fwhm[lo:hi] = night.fwhm

        idx_n = xmatch.index[n]
        present = (idx_n >= 0) & has_aper
        if not np.any(present):
            continue
        local_idx = idx_n[present]
        aper_local = aper_safe[present]
        zp_local = zp_sel[n, present]

        lc_local = night.lc[local_idx, :, aper_local].astype(np.float64)
        lc_err_local = night.lc_err[local_idx, :, aper_local].astype(np.float64)
        epoch_ok_local = night.epoch_ok[local_idx, :]

        good = epoch_ok_local & np.isfinite(lc_local) & (lc_local > 0)
        lc_masked = np.where(good, lc_local, np.nan)
        med_local = nanmedian_quiet(lc_masked, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mag_local = -2.5 * np.log10(lc_masked) - zp_local[:, None]
            flux_norm_local = lc_masked / med_local[:, None]
            err_masked = np.where(good, lc_err_local, np.nan)
            flux_norm_err_local = err_masked / med_local[:, None]
            mag_err_local = 1.0857 * err_masked / lc_masked

        global_rows = np.nonzero(present)[0]
        mag[global_rows, lo:hi] = mag_local.astype(np.float32)
        mag_err[global_rows, lo:hi] = mag_err_local.astype(np.float32)
        flux_norm[global_rows, lo:hi] = flux_norm_local.astype(np.float32)
        flux_norm_err[global_rows, lo:hi] = flux_norm_err_local.astype(np.float32)

    return MultiNightLightCurves(
        labels=labels,
        night_of_frame=night_of_frame,
        frame_in_night=frame_in_night,
        bjd_tdb=bjd_tdb,
        airmass=airmass,
        fwhm=fwhm,
        aperture=aperture,
        mag=mag,
        mag_err=mag_err,
        flux_norm=flux_norm,
        flux_norm_err=flux_norm_err,
        night_mean_mag=night_mean_mag,
        night_mean_err=night_mean_err,
        mean_mag=mean_mag,
        n_nights=n_nights_out,
    )


def save_multinight(
    path: Path | str, nights: list[NightProducts], xmatch: NightCrossMatch, tie: NightTie,
    mlc: MultiNightLightCurves, settings: Settings,
) -> None:
    """Write every cross-match/tie/light-curve array and ``settings`` as uncompressed .npz."""
    path = Path(path)
    night_info = [
        {
            "label": n.label,
            "directory": str(n.directory.resolve()),
            "filter": n.filter,
            "object": n.object,
            "n_frames": int(n.n_frames),
            "n_kept": int(np.count_nonzero(n.frame_kept)),
        }
        for n in nights
    ]

    np.savez(
        path,
        labels_json=json.dumps(list(xmatch.labels)),
        xmatch_ra=xmatch.ra,
        xmatch_dec=xmatch.dec,
        xmatch_index=xmatch.index,
        xmatch_n_matched=xmatch.n_matched,
        tie_anchor_index=np.int64(tie.anchor_index),
        tie_coef=tie.coef,
        tie_basis_terms_json=json.dumps(list(tie.basis_terms)),
        tie_xi=tie.xi,
        tie_eta=tie.eta,
        tie_centre_ra=np.float64(tie.centre_radec[0]),
        tie_centre_dec=np.float64(tie.centre_radec[1]),
        tie_scale_deg=np.float64(tie.scale_deg),
        tie_mag0=tie.mag0,
        tie_zp=tie.zp,
        tie_mean_mag=tie.mean_mag,
        tie_night_mag=tie.night_mag,
        tie_night_mag_err=tie.night_mag_err,
        tie_tie_star=tie.tie_star,
        tie_rejected=tie.rejected,
        tie_floor_mag_centres=tie.floor_mag_centres,
        tie_floor=tie.floor,
        tie_n_tie=tie.n_tie,
        tie_resid_mad=tie.resid_mad,
        tie_resid_mad_bright=tie.resid_mad_bright,
        tie_chi2_after=tie.chi2_after,
        tie_chi2_holdout=tie.chi2_holdout,
        tie_chi2_holdout_bins=tie.chi2_holdout_bins,
        tie_n_iter=tie.n_iter,
        tie_seeing_basis_terms_json=json.dumps(list(tie.seeing_basis_terms)),
        tie_seeing_coef=tie.seeing_coef,
        tie_seeing_mag0=tie.seeing_mag0,
        tie_seeing_crowd0=tie.seeing_crowd0,
        tie_night_fwhm=tie.night_fwhm,
        tie_crowding=tie.crowding,
        tie_loose=tie.loose,
        mlc_night_of_frame=mlc.night_of_frame,
        mlc_frame_in_night=mlc.frame_in_night,
        mlc_bjd_tdb=mlc.bjd_tdb,
        mlc_airmass=mlc.airmass,
        mlc_fwhm=mlc.fwhm,
        mlc_aperture=mlc.aperture,
        mlc_mag=mlc.mag,
        mlc_mag_err=mlc.mag_err,
        mlc_flux_norm=mlc.flux_norm,
        mlc_flux_norm_err=mlc.flux_norm_err,
        mlc_night_mean_mag=mlc.night_mean_mag,
        mlc_night_mean_err=mlc.night_mean_err,
        mlc_mean_mag=mlc.mean_mag,
        mlc_n_nights=mlc.n_nights,
        night_info_json=json.dumps(night_info),
        settings_json=json.dumps(settings_to_dict(settings)),
    )
    logger.info("wrote %s (%d nights, %d global stars)", path, len(nights), xmatch.ra.shape[0])


def load_multinight(
    path: Path | str,
) -> tuple[NightCrossMatch, NightTie, MultiNightLightCurves, list[dict], Settings]:
    """Read back the products written by :func:`save_multinight`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        labels = tuple(json.loads(str(data["labels_json"])))

        xmatch = NightCrossMatch(
            ra=data["xmatch_ra"],
            dec=data["xmatch_dec"],
            index=data["xmatch_index"],
            labels=labels,
            n_matched=data["xmatch_n_matched"],
        )
        tie = NightTie(
            labels=labels,
            anchor_index=int(data["tie_anchor_index"]),
            coef=data["tie_coef"],
            basis_terms=tuple(json.loads(str(data["tie_basis_terms_json"]))),
            xi=data["tie_xi"],
            eta=data["tie_eta"],
            centre_radec=(float(data["tie_centre_ra"]), float(data["tie_centre_dec"])),
            scale_deg=float(data["tie_scale_deg"]),
            mag0=data["tie_mag0"],
            zp=data["tie_zp"],
            mean_mag=data["tie_mean_mag"],
            night_mag=data["tie_night_mag"],
            night_mag_err=data["tie_night_mag_err"],
            tie_star=data["tie_tie_star"],
            rejected=data["tie_rejected"],
            floor_mag_centres=data["tie_floor_mag_centres"],
            floor=data["tie_floor"],
            n_tie=data["tie_n_tie"],
            resid_mad=data["tie_resid_mad"],
            resid_mad_bright=data["tie_resid_mad_bright"],
            chi2_after=data["tie_chi2_after"],
            chi2_holdout=data["tie_chi2_holdout"],
            chi2_holdout_bins=data["tie_chi2_holdout_bins"],
            n_iter=data["tie_n_iter"],
            seeing_basis_terms=tuple(json.loads(str(data["tie_seeing_basis_terms_json"]))),
            seeing_coef=data["tie_seeing_coef"],
            seeing_mag0=data["tie_seeing_mag0"],
            seeing_crowd0=data["tie_seeing_crowd0"],
            night_fwhm=data["tie_night_fwhm"],
            crowding=data["tie_crowding"],
            loose=data["tie_loose"] if "tie_loose" in data.files else None,
        )
        mlc = MultiNightLightCurves(
            labels=labels,
            night_of_frame=data["mlc_night_of_frame"],
            frame_in_night=data["mlc_frame_in_night"],
            bjd_tdb=data["mlc_bjd_tdb"],
            airmass=data["mlc_airmass"],
            fwhm=data["mlc_fwhm"],
            aperture=data["mlc_aperture"],
            mag=data["mlc_mag"],
            mag_err=data["mlc_mag_err"],
            flux_norm=data["mlc_flux_norm"],
            flux_norm_err=data["mlc_flux_norm_err"],
            night_mean_mag=data["mlc_night_mean_mag"],
            night_mean_err=data["mlc_night_mean_err"],
            mean_mag=data["mlc_mean_mag"],
            n_nights=data["mlc_n_nights"],
        )
        night_info = json.loads(str(data["night_info_json"]))
        settings = settings_from_dict(json.loads(str(data["settings_json"])))
    return xmatch, tie, mlc, night_info, settings


def save_tie_report(path_csv: Path | str, tie: NightTie) -> None:
    """One row per (night, aperture): tie diagnostics and every fitted coefficient.

    ``floor_bright_mmag`` is the calibration floor's brightest magnitude bin
    only (a scalar summary); the full per-bin floor, with its magnitude
    centres and held-out chi2, is in :func:`save_floor_report`.
    ``chi2_after`` is close to 1 by construction for 2 nights (see
    :class:`NightTie`'s docstring) -- ``chi2_holdout`` is the real check.
    ``night_fwhm``/``d_fwhm_anchor`` and the ``seeing_*`` columns are the
    pooled seeing term's inputs/fitted surface (one shared ``beta`` per
    aperture, so the ``seeing_*`` values repeat identically across a given
    aperture's night rows); all-empty/zero when ``settings.use_seeing_term``
    was ``False``.
    """
    path_csv = Path(path_csv)
    n_nights = len(tie.labels)
    n_aper = tie.coef.shape[1]
    seeing_fieldnames = [f"seeing_{name}" for name in tie.seeing_basis_terms]
    fieldnames = [
        "label", "aperture", "is_anchor", "n_tie", "n_rejected",
        "resid_mad_mmag", "resid_mad_bright_mmag", "floor_bright_mmag",
        "chi2_after", "chi2_holdout", "n_iter",
        "night_fwhm", "d_fwhm_anchor", "seeing_mag0", "seeing_crowd0",
        *tie.basis_terms, *seeing_fieldnames,
    ]
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for a in range(n_aper):
            n_rejected = int(np.count_nonzero(tie.rejected[:, a]))
            for n in range(n_nights):
                row = {
                    "label": tie.labels[n],
                    "aperture": a,
                    "is_anchor": n == tie.anchor_index,
                    "n_tie": int(tie.n_tie[n, a]),
                    "n_rejected": n_rejected,
                    "resid_mad_mmag": float(tie.resid_mad[n, a]) * 1000.0,
                    "resid_mad_bright_mmag": float(tie.resid_mad_bright[n, a]) * 1000.0,
                    "floor_bright_mmag": float(tie.floor[n, a, 0]) * 1000.0,
                    "chi2_after": float(tie.chi2_after[n, a]),
                    "chi2_holdout": float(tie.chi2_holdout[n, a]),
                    "n_iter": int(tie.n_iter[a]),
                    "night_fwhm": float(tie.night_fwhm[n]),
                    "d_fwhm_anchor": float(tie.night_fwhm[n] - tie.night_fwhm[tie.anchor_index]),
                    "seeing_mag0": float(tie.seeing_mag0[a]),
                    "seeing_crowd0": float(tie.seeing_crowd0[a]),
                }
                for t, name in enumerate(tie.basis_terms):
                    row[name] = float(tie.coef[n, a, t])
                for t, name in enumerate(seeing_fieldnames):
                    row[name] = float(tie.seeing_coef[a, t])
                writer.writerow(row)
    logger.info("wrote %s (%d rows)", path_csv, n_nights * n_aper)


def save_floor_report(path_csv: Path | str, tie: NightTie) -> None:
    """Long-format calibration floor: one row per (night, aperture, magnitude bin).

    Columns: ``label, aperture, bin, mag_centre, floor_mmag, chi2_holdout``
    (the last two NaN where that bin had no directly-fit/held-out-evaluated
    pair -- see :meth:`NightTie.floor_at` for the usable, gap-filled form).
    """
    path_csv = Path(path_csv)
    n_nights = len(tie.labels)
    n_aper, n_bins = tie.floor.shape[1], tie.floor.shape[2]
    fieldnames = ["label", "aperture", "bin", "mag_centre", "floor_mmag", "chi2_holdout"]
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for a in range(n_aper):
            for b in range(n_bins):
                for n in range(n_nights):
                    writer.writerow({
                        "label": tie.labels[n],
                        "aperture": a,
                        "bin": b,
                        "mag_centre": float(tie.floor_mag_centres[a, b]),
                        "floor_mmag": float(tie.floor[n, a, b]) * 1000.0,
                        "chi2_holdout": float(tie.chi2_holdout_bins[n, a, b]),
                    })
    logger.info("wrote %s (%d rows)", path_csv, n_nights * n_aper * n_bins)


def _resolve_table_format(fmt: str) -> str:
    """``fmt`` as given, or (for ``"auto"``) ``"parquet"`` if pyarrow is importable, else
    ``"fits"``.
    """
    if fmt != "auto":
        return fmt
    return "parquet" if importlib.util.find_spec("pyarrow") is not None else "fits"


def _write_table(columns: dict, path: Path, fmt: str) -> Path:
    """``columns`` (a plain dict of equal-length 1-D arrays) written to ``path`` as parquet
    or FITS.
    """
    from astropy.table import Table

    actual_fmt = _resolve_table_format(fmt)
    if actual_fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
        msg = "pyarrow is required for parquet output"
        raise MultiNightError(msg)

    columns = {
        key: (np.asarray(value, dtype=str) if np.asarray(value).dtype == object else value)
        for key, value in columns.items()
    }
    table = Table(columns)
    if actual_fmt == "parquet":
        out_path = path.with_suffix(".parquet")
        table.write(out_path, format="parquet", overwrite=True)
    else:
        out_path = path.with_suffix(".fits")
        table.write(out_path, format="fits", overwrite=True)
    logger.info("wrote %s (%d rows)", out_path, len(table))
    return out_path


def save_multinight_tables(
    stem: Path | str, xmatch: NightCrossMatch, tie: NightTie, mlc: MultiNightLightCurves,
    fmt: str = "auto",
) -> tuple[Path, Path]:
    """Write ``{stem}_stars.*`` (one row per star) and ``{stem}_lightcurves.*`` (long format)."""
    stem = Path(stem)
    n_global = xmatch.ra.shape[0]

    star_columns: dict = {
        "global_id": np.arange(n_global, dtype=np.int64),
        "ra": xmatch.ra,
        "dec": xmatch.dec,
        "aperture": mlc.aperture,
        "mean_mag": mlc.mean_mag,
        "n_nights": mlc.n_nights,
    }
    for n, label in enumerate(tie.labels):
        star_columns[f"mag_{label}"] = mlc.night_mean_mag[n]
        star_columns[f"err_{label}"] = mlc.night_mean_err[n]
        star_columns[f"star_id_{label}"] = xmatch.index[n]
    stars_path = _write_table(star_columns, stem.with_name(f"{stem.name}_stars"), fmt)

    aper_safe = np.where(mlc.aperture >= 0, mlc.aperture, 0)
    gi, fj = np.nonzero(np.isfinite(mlc.mag))
    labels_arr = np.asarray(mlc.labels, dtype=object)
    lc_columns = {
        "global_id": gi.astype(np.int64),
        "night_label": labels_arr[mlc.night_of_frame[fj]],
        "frame_in_night": mlc.frame_in_night[fj].astype(np.int64),
        "bjd_tdb": mlc.bjd_tdb[fj],
        "aperture": aper_safe[gi].astype(np.int64),
        "mag": mlc.mag[gi, fj],
        "mag_err": mlc.mag_err[gi, fj],
        "flux_norm": mlc.flux_norm[gi, fj],
        "flux_norm_err": mlc.flux_norm_err[gi, fj],
    }
    lc_path = _write_table(lc_columns, stem.with_name(f"{stem.name}_lightcurves"), fmt)
    return stars_path, lc_path


def plot_tie_diagnostics(path_png: Path | str, tie: NightTie, aperture: int) -> None:
    """Per non-anchor night: residual-vs-magnitude and residual-map (xi, eta) panels for tie stars.

    Logs a message and returns without writing anything if matplotlib is not
    installed (the ``lightcurve`` extra).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping tie diagnostics plot")
        return

    path_png = Path(path_png)
    non_anchor = [n for n in range(len(tie.labels)) if n != tie.anchor_index]
    if not non_anchor:
        logger.warning("no non-anchor night to plot tie diagnostics for")
        return

    fig, axes = plt.subplots(len(non_anchor), 2, figsize=(10, 4 * len(non_anchor)), squeeze=False)
    for row, n in enumerate(non_anchor):
        mask = tie.tie_star[n, :, aperture]
        m = tie.mean_mag[mask, aperture]
        resid_mmag = (
            (tie.night_mag[n, mask, aperture] - tie.zp[n, mask, aperture]) - m
        ) * 1000.0

        ax0 = axes[row, 0]
        ax0.scatter(m, resid_mmag, s=4, alpha=0.5)
        ax0.axhline(0.0, color="grey", lw=0.5)
        ax0.set_xlabel("M (mag)")
        ax0.set_ylabel("residual (mmag)")
        ax0.set_title(f"{tie.labels[n]}: residual vs M")

        ax1 = axes[row, 1]
        sc = ax1.scatter(tie.xi[mask], tie.eta[mask], c=resid_mmag, s=4, cmap="coolwarm")
        ax1.set_xlabel("xi")
        ax1.set_ylabel("eta")
        ax1.set_title(f"{tie.labels[n]}: residual map")
        fig.colorbar(sc, ax=ax1, label="residual (mmag)")

    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path_png)
