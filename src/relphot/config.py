"""Settings dataclasses for relphot, with defaults for every field.

An optional TOML file can override any subset of the defaults; a value not
present in the file simply keeps its dataclass default. An unknown top-level
or nested key in the TOML file is an error rather than a typo that silently
does nothing.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from relphot.exceptions import ConfigError

__all__ = [
    "BorderSettings",
    "CatalogSettings",
    "ColumnMap",
    "ComparisonSettings",
    "DbSettings",
    "DecorrelationSettings",
    "LightcurveSettings",
    "MultiNightSettings",
    "ReferenceSettings",
    "SearchSettings",
    "Settings",
    "SiteSettings",
    "TailSettings",
    "TileSettings",
    "VariableSettings",
    "load_settings",
    "settings_from_dict",
    "settings_to_dict",
]


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Column-name mapping from a source catalogue to relphot's internal fields.

    Defaults match the robo43 SExtractor ``CATALOG``/``*_proc_catalog.csv``
    convention. A catalogue produced by another photometry method is read by
    pointing these names at its own columns; nothing else in
    :mod:`relphot.ingest` needs to change.
    """

    ra: str = "RA"
    dec: str = "DEC"
    x: str = "X_IMAGE"
    y: str = "Y_IMAGE"
    #: Name of the per-aperture flux vector column in a FITS BinTableHDU
    #: (e.g. SExtractor's ``FLUX_APER``, shape (n_sources, n_aper)).
    flux: str = "FLUX_APER"
    fluxerr: str = "FLUXERR_APER"
    flags: str = "FLAGS"
    snr: str = "SNR"
    fwhm: str = "FWHM"
    background: str = "BACKGROUND"
    #: The robo43 CSV convention splits the vector columns above into
    #: ``<prefix>1``, ``<prefix>2``, ... instead of one vector column. Used
    #: only by the CSV adapter, which probes ``f"{flux_prefix}{i}"`` for
    #: i = 1, 2, ... until a name is missing to discover n_aper.
    flux_prefix: str = "FLUX_APER_"
    fluxerr_prefix: str = "FLUXERR_APER_"


@dataclass(frozen=True, slots=True)
class CatalogSettings:
    """How a per-frame catalogue is read and which sources are trusted."""

    #: Name of the BinTableHDU holding the source catalogue in a FITS file.
    hdu_name: str = "CATALOG"
    #: Cross-match radius, in arcsec, used by relphot.match.
    match_radius_arcsec: float = 1.0
    #: Minimum fraction of frames a star must appear in to be kept.
    min_presence: float = 0.8
    #: FLAGS bit meaning "saturated" / "non-linear" (never set together with
    #: a trustworthy flux); OR'd against a source's FLAGS value.
    saturation_flag_bit: int = 4
    nonlinear_flag_bit: int = 64
    columns: ColumnMap = field(default_factory=ColumnMap)


@dataclass(frozen=True, slots=True)
class SiteSettings:
    """Fallback observatory geodetic position.

    Header keys are tried first: ``LATITUDE``/``LONGITUD``/``ALTITUDE`` (T80S),
    then ``SITELAT``/``SITELONG``/``SITEELEV`` (N.I.N.A./ASCOM, as written for ROBO43).
    This fallback is used only when both key sets are missing -- a last resort
    for other data.
    """

    latitude_deg: float | None = None
    longitude_deg: float | None = None
    elevation_m: float = 0.0


@dataclass(frozen=True, slots=True)
class TileSettings:
    """Adaptive rectangular tiling of the master frame's pixel projection.

    See :mod:`relphot.tiles`. ``tile_size_px`` is the nominal edge length of
    the regular starting grid; ``overlap_px`` extends each tile's candidate
    catchment beyond its core rectangle without moving any star's own light
    curve out of its core tile. The adaptive loop merges tiles short of
    ``min_ref_candidates`` clean reference candidates until every tile meets
    it (or only one tile is left); a final tile below
    ``hard_min_ref_candidates`` raises :class:`~relphot.exceptions.TilingError`.
    """

    tile_size_px: float = 1000.0
    overlap_px: float = 100.0
    min_ref_candidates: int = 50
    hard_min_ref_candidates: int = 20


@dataclass(frozen=True, slots=True)
class VariableSettings:
    """Known-variable-star cross-match, used to exclude reference candidates.

    ``catalogs`` are VizieR catalogue identifiers queried through
    ``astroquery.vizier`` (an optional dependency -- see the ``variables``
    extra); ``extra_catalogs`` appends further VizieR IDs (e.g. OGLE for
    Magellanic fields) without displacing the defaults. Results are cached
    per (catalogue, rounded field centre, radius) as ECSV under
    ``cache_dir``. Disabled, offline, or with astroquery missing, the
    cross-match logs a warning and flags nothing -- see
    :func:`relphot.variables.flag_known_variables`.
    """

    enabled: bool = True
    #: AAVSO VSX; Gaia DR3 variable-classification results; ASAS-SN variables
    #: (Jayasinghe et al. catv2021 table -- "II/366/catalog" does not exist).
    catalogs: tuple[str, ...] = ("B/vsx/vsx", "I/358/vclassre", "II/366/catv2021")
    extra_catalogs: tuple[str, ...] = ()
    match_radius_arcsec: float = 2.0
    cache_dir: str = "~/.cache/relphot/variables"


@dataclass(frozen=True, slots=True)
class MultiNightSettings:
    """Phase 2: cross-match and zero-point tie of two or more already-processed nights.

    See :mod:`relphot.multinight`. Every star's reference magnitude is tied
    across nights by a per-(night, aperture) zero-point surface
    ``Z_n(xi, eta, M)``, fixed to zero for the anchor night; ``xi``/``eta``
    are tangent-plane coordinates of the field, ``M`` the star's calibrated
    mean magnitude. ``min_tie_stars`` guards against fitting that surface to
    too few comparison stars.
    """

    #: Cross-match radius, in arcsec, used to link the same star across nights.
    match_radius_arcsec: float = 1.0
    #: Night label of the zero-point anchor (Z ≡ 0); "auto" picks the night
    #: with the most kept frames (ties: earliest).
    anchor: str = "auto"
    #: Aperture index used for every star; -1 chooses per star (see
    #: :func:`relphot.multinight.build_multinight_lightcurves`).
    aperture: int = -1
    #: Degree of the tangent-plane spatial polynomial in the zero-point model.
    spatial_degree: int = 2
    #: Degree of the magnitude-dependent polynomial in the zero-point model.
    mag_degree: int = 2
    #: Added in quadrature to per-star errors when weighting the tie fit.
    tie_err_floor_mag: float = 0.003
    #: Star-level clipping threshold (robust sigma) in the zero-point tie.
    clip_sigma: float = 4.0
    #: Maximum alternating-fit iterations.
    max_iter: int = 20
    #: Alternating-fit convergence: max |ΔZ| change over tie stars.
    tol_mag: float = 1e-5
    #: Minimum tie stars required for a (night, aperture); fewer raises
    #: :class:`~relphot.exceptions.MultiNightError`.
    min_tie_stars: int = 50
    #: Equal-count magnitude bins for the per-night calibration floor
    #: (see :func:`relphot.multinight.tie_nights`/:meth:`~relphot.multinight.NightTie.floor_at`).
    floor_n_bins: int = 6
    #: Minimum common tie stars in a (night pair, magnitude bin) for that
    #: bin's calibration-floor variance to be fit directly (an unmet bin is
    #: filled by :meth:`~relphot.multinight.NightTie.floor_at`'s interpolation).
    floor_min_bin_stars: int = 100
    #: Add a pooled seeing term ``beta(M, crowding) * (F_n - F_anchor)`` to
    #: the zero-point model, ``F_n`` the night's median FWHM (px) over kept
    #: frames -- see :mod:`relphot.multinight`'s module docstring. ``beta``
    #: is one surface shared by every night (never a free per-star fit).
    #: ``False`` reproduces the model without this term.
    use_seeing_term: bool = True
    #: Degree of the pooled seeing-surface polynomial in centred mean
    #: magnitude.
    seeing_mag_degree: int = 1
    #: Degree of the pooled seeing-surface polynomial in centred crowding
    #: (:func:`relphot.decorrelate.compute_crowding`). With exactly 2 nights
    #: this is what separates ``beta`` from the per-night ``poly(M)`` term,
    #: which is otherwise nearly degenerate with it (see the module
    #: docstring); 0 disables the crowding dependence and should only be
    #: used with >= 3 nights.
    seeing_crowding_degree: int = 1

    # --- loose nights (relphot.multinight.tie_loose_nights) ---
    #: Labels of nights tied "loosely", e.g. a cloudy night that would distort the tie of
    #: the good ones. The other (core) nights are tied by exactly the same code as without
    #: this setting -- run on the core subset, so their zero points, calibration floors,
    #: mean magnitudes and aperture choice are bit-identical to a run without the loose
    #: nights. Each loose night is then fit only to that fixed core frame (the core nights'
    #: calibrated weighted mean) with a low-order surface (the two degrees below, at most
    #: ``spatial_degree``/``mag_degree``; no seeing term) and gets its own calibration floor
    #: measured against the frame, which inflates its errors everywhere the floor is used.
    #: A loose night takes part in the variability tests (inter-night chi2, Lomb-Scargle,
    #: per-night verdict recurrence) but never in the transit ones (BLS, period
    #: compatibility, per-night transit events): see :mod:`relphot.multinight_search`.
    loose_nights: tuple[str, ...] = ()
    #: Degree of the tangent-plane polynomial of a loose night's zero-point surface.
    loose_spatial_degree: int = 1
    #: Degree of the magnitude polynomial of a loose night's zero-point surface.
    loose_mag_degree: int = 1

    # --- Unit B: cross-night search (relphot.multinight_search) ---
    #: Inter-night (long-term) variability chi2 p-value threshold.
    internight_p_threshold: float = 1e-4
    #: Minimum max-min nightly-mean spread (mag) also required for an
    #: inter-night variability candidate.
    internight_min_amplitude_mag: float = 0.005
    #: A loose night alone never makes an inter-night candidate. A star with a loose
    #: night among its nights is a candidate only if the test on all its nights passes
    #: (``internight_p_threshold``, ``internight_min_amplitude_mag``) AND the same chi2
    #: test on its core nights alone (loose night dropped) has p below this; a star with
    #: fewer than 2 core nights is then never a candidate. Stars without a loose night
    #: are unaffected.
    loose_internight_support_p: float = 1e-2
    #: Which stars get the combined Lomb-Scargle periodogram (B4)/BLS search
    #: (B6): "candidates" (union of variability/transit-flagged stars) or "all".
    periodogram_stars: str = "candidates"
    periodogram_max_freq: float = 50.0
    periodogram_samples_per_peak: int = 10
    ls_fap_threshold: float = 1e-3
    period_min_days: float = 0.2
    period_max_days: float = 30.0
    period_grid_oversample: int = 5
    bls_durations_hours: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    #: A trial duration longer than this fraction of the median night span
    #: is dropped before the BLS search -- a "transit" close to or longer
    #: than a night's own span is a night-to-night step, not a transit.
    bls_max_duration_fraction: float = 0.5
    #: Minimum out-of-transit epochs a night must have, alongside >= 1
    #: in-transit epoch, to count toward that candidate's ``nights_in_transit``.
    bls_min_out_of_transit: int = 5
    bls_snr_threshold: float = 7.0
    bls_min_nights_in_transit: int = 2
    #: Sigma thresholds for :func:`relphot.multinight_search.period_compatibility`.
    event_exclusion_sigma: float = 3.0
    event_min_box_coverage: float = 0.5
    depth_consistency_sigma: float = 3.0
    #: Minimum nights with a finite nightly magnitude for a star to enter
    #: the inter-night variability search.
    min_nights: int = 2
    #: Minimum nights of finite epoch data for a star to enter the combined
    #: Lomb-Scargle periodogram (B4) or BLS (B6) search at all -- a period
    #: longer than a night's own span is unconstrained by fewer nights than
    #: this, and their false-alarm probabilities are unreliable besides
    #: (red/systematic, not white, noise). A star with fewer nights gets
    #: NaN/False LS and BLS columns; skipped stars are logged once, not
    #: per star.
    min_nights_periodic: int = 3

    def __post_init__(self) -> None:
        if len(set(self.loose_nights)) != len(self.loose_nights):
            msg = f"multinight.loose_nights has duplicate labels: {self.loose_nights!r}"
            raise ConfigError(msg)
        if self.anchor in self.loose_nights:
            msg = f"multinight.anchor {self.anchor!r} cannot be a loose night"
            raise ConfigError(msg)
        for name, cap_name in (
            ("loose_spatial_degree", "spatial_degree"), ("loose_mag_degree", "mag_degree"),
        ):
            value = getattr(self, name)
            cap = getattr(self, cap_name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= cap:
                msg = (
                    f"multinight.{name} must be an integer in [0, {cap_name}={cap}], "
                    f"got {value!r}"
                )
                raise ConfigError(msg)
        value = self.loose_internight_support_p
        if (
            isinstance(value, bool) or not isinstance(value, int | float)
            or not 0.0 < value <= 1.0
        ):
            msg = f"multinight.loose_internight_support_p must be in (0, 1], got {value!r}"
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    """Reference-candidate selection and per-tile reference construction.

    Per-tile reference is built from a fixed star set S_t valid in every kept frame
    of the night. A frame with too few reference stars is flagged and dropped from
    the whole night, not per-tile. See :func:`relphot.reference.select_reference_frames_and_stars`
    for frame-dropping logic. ``method`` selects ``"weighted_fixed_mean"`` (default,
    inverse-variance weighted mean) or ``"median_fixed"`` (plain median).
    """

    min_snr: float = 15.0
    isolation_radius_arcsec: float = 5.0
    baseline_iter: int = 2
    frame_outlier_sigma: float = 3.0
    #: Minimum number of reference stars in S_t for every tile in every kept frame.
    min_ref_stars: int = 20
    #: Maximum fraction of frames allowed to be dropped from the whole night.
    max_dropped_frame_fraction: float = 0.3
    #: Sigma threshold for per-star outlier rejection in reference star set selection.
    star_outlier_sigma: float = 5.0
    #: Maximum iterations for per-star outlier rejection loop.
    star_reject_iter: int = 5
    method: str = "weighted_fixed_mean"


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    """Stage-4 comparison-star selection and ensemble construction.

    Pool criteria mirror ReferenceSettings (presence, FLAGS==0, not a known
    variable) but relax the SNR floor and drop the isolation cut by default
    -- see comparison.py's module docstring.
    """

    min_snr: float = 5.0
    require_isolation: bool = False
    isolation_radius_arcsec: float = 5.0
    k_floor: float = 2.0
    k_floor_growth: float = 1.5
    max_k_floor: float = 8.0
    n_rounds: int = 3
    n_mag_bins: int = 20
    min_bin_stars: int = 5
    min_comparison_stars: int = 5
    clip_sigma: float = 3.0
    max_iter: int = 5
    ensemble_statistic: str = "weighted_clipped_mean"


@dataclass(frozen=True, slots=True)
class DbSettings:
    """Results-database noise cut and cross-match (see docs/DB_PLAN.md).

    A star of a night is stored in the results database iff its rms is
    finite, it passes the night's epoch cut, and its predicted per-point
    noise at the best aperture is at most ``max_expected_noise`` -- unless
    it is a transit or variability candidate of that night's search, which
    is always stored when ``keep_candidates`` is true. ``assumed_zp`` and
    ``telescope_zp`` give the zero point of a night without a Gaia calibration.
    """

    #: Maximum predicted per-point noise (fractional) at the best aperture
    #: for a non-candidate star to be stored.
    max_expected_noise: float = 0.05
    #: Always store a night's transit/variability search candidates, even
    #: when they fail the noise cut.
    keep_candidates: bool = True
    #: Cross-match radius, in arcsec, used to link a star to an existing
    #: ``object`` row when loading a night.
    match_radius_arcsec: float = 1.0
    #: Lomb-Scargle period-grid floor (days) for ``relphot db analyze``.
    ls_min_period_days: float = 0.02
    #: Lomb-Scargle period-grid ceiling (days); capped by the data's own time span.
    ls_max_period_days: float = 100.0
    #: Frequency-grid oversampling factor for both the LS and BLS grids
    #: (same convention as ``MultiNightSettings.periodogram_samples_per_peak``).
    ls_samples_per_peak: int = 10
    #: Frequency-grid size above which a periodogram's ``df`` is coarsened
    #: (increased) to fit, and ``periodogram.coarsened`` is set.
    max_periodogram_points: int = 200000
    #: False-alarm-probability threshold for a combined Lomb-Scargle PERIOD (VAR).
    ls_fap_threshold: float = 0.01
    #: BLS period-grid floor (days).
    bls_min_period_days: float = 0.2
    #: BLS period-grid ceiling (days); also capped by combined_span / 1.5.
    bls_max_period_days: float = 30.0
    #: Trial transit durations for the combined BLS search; a duration not
    #: shorter than ``bls_min_period_days`` is dropped.
    bls_durations_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    #: Minimum BLS depth SNR (``periodogram.extra['depth_snr']``) for a
    #: combined BLS PERIOD (EXOP).
    bls_min_snr: float = 7.0
    #: Multi-night detection kinds that count towards CLASS; 'bls',
    #: 'ls_periodic', 'internight' are stored and queryable but excluded
    #: until their thresholds are calibrated.
    class_multinight_kinds: tuple[str, ...] = ("recurrent",)
    #: Systematic floor on the transit-depth difference of a matching-transit
    #: pair, as a fraction of the pair's mean depth.
    match_depth_sys_frac: float = 0.05
    #: Extra depth-difference floor (fraction of mean depth) added when the two
    #: events were observed with different telescopes (dilution, pixel scale).
    match_depth_sys_frac_cross_telescope: float = 0.15
    #: Systematic floor on the T14 difference, as a fraction of the mean T14.
    match_t14_sys_frac: float = 0.0
    #: Systematic floor on the ingress-fraction (T12/T14) difference (absolute).
    match_ingress_sys: float = 0.0
    #: Smallest commensurate period (days) listed for a matching-transit pair
    #: (same as ``MultiNightSettings.period_min_days``).
    match_period_min_days: float = 0.2
    #: Most commensurate periods stored per matching-transit pair.
    match_max_commensurate: int = 50
    #: Half-width, as a fraction of ``lit_period * harmonic``, of the window in
    #: which a literature period is verified against the data.
    lit_period_window_frac: float = 0.05
    #: Periods longer than this (days) cannot be measured on per-night-normalised
    #: flux, nor with a free offset per night: both absorb variability slower than
    #: a night. Their period analysis uses tie-calibrated magnitudes and no
    #: per-night offsets, and needs a multi-night tie covering >= 2 nights.
    long_period_days: float = 1.0
    #: Half-width, as a fraction of ``period_guess * harmonic``, of the windows a
    #: user-guided period search (``relphot db reprocess``) looks in (harmonics 0.5, 1, 2).
    guided_period_window_frac: float = 0.2
    #: Number of phase bins used for ``period_estimate.phase_coverage`` (the
    #: fraction of them holding at least one point).
    phase_coverage_bins: int = 20
    #: Most alias / next-peak candidate periods stored per period estimate.
    max_alias_candidates: int = 5
    #: Zero point (mag) of a night whose kept frames mostly lack a Gaia calibration and whose
    #: telescope is not in ``telescope_zp``: robo43's ``instrumental_zp``. Stored as
    #: ``night.zp`` with ``zp_source = 'assumed'`` when the night is loaded.
    assumed_zp: float = 20.0
    #: Measured zero point (mag) per telescope, used instead of ``assumed_zp`` for a night with
    #: no Gaia calibration (``zp_source = 'measured'``). Keys are ``night.telescope`` exactly as
    #: stored (``'T80S'``, ``'ROBO43'``). T80S 27.85 is the median of Gaia DR3 G minus
    #: relphot magnitude over bright isolated stars, deliberately kept on the Gaia G scale for a
    #: uniform magnitude scale across telescopes. T80S filter is S-PLUS rSDSS (not Bessell R).
    #: The rSDSS AB zero point measured 2026-09-30 from Gaia XP synthetic photometry is
    #: 27.846 ± 0.004 (colour slope ~0); the G-based value carries a colour term of about
    #: -0.038 mag per mag of BP-RP (±0.04 mag between BP-RP 0 and 2). A ``telescope_zp`` in a
    #: settings file replaces this table as a whole.
    telescope_zp: dict[str, float] = field(default_factory=lambda: {"T80S": 27.85})
    #: Cross-candidate check of one night's transit events (``relphot db analyze``): time window
    #: of a pair of events, as a fraction of the shorter T14 (the window is the largest of this,
    #: ``coincidence_tc_nsigma`` combined centre-time errors and one cadence of the night).
    coincidence_tc_frac: float = 0.1
    #: Time window of a pair of events, in combined 1-sigma centre-time errors.
    coincidence_tc_nsigma: float = 2.0
    #: Two events have a similar T14 when their durations differ by less than this factor (a
    #: duration that is only a lower limit is similar to any duration up to this factor shorter).
    coincidence_t14_ratio: float = 1.5
    #: Two events have a similar depth when their depths differ by less than this factor.
    coincidence_depth_ratio: float = 2.0
    #: An event is auto-rejected only if at least this many other events of its night coincide
    #: with it (same centre time within the window, similar T14 and depth) ...
    coincidence_min_similar: int = 3
    #: ... and the binomial probability of that many coincidences by chance (events spread
    #: uniformly over the night) is below this.
    coincidence_max_p: float = 1e-3

    def __post_init__(self) -> None:
        if not isinstance(self.telescope_zp, dict) or not all(
            isinstance(k, str) for k in self.telescope_zp
        ):
            msg = "db.telescope_zp must be a table of telescope name -> zero point"
            raise ConfigError(msg)
        zps = {"assumed_zp": self.assumed_zp}
        zps.update({f"telescope_zp[{k!r}]": v for k, v in self.telescope_zp.items()})
        for name, value in zps.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"db.{name} must be a finite number, got {value!r}"
                raise ConfigError(msg)
            if not math.isfinite(value):
                msg = f"db.{name} must be a finite number, got {value!r}"
                raise ConfigError(msg)
        for name in ("coincidence_t14_ratio", "coincidence_depth_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, int | float)
                or not (math.isfinite(value) and value > 1)
            ):
                msg = f"db.{name} must be a number > 1, got {value!r}"
                raise ConfigError(msg)
        for name in (
            "coincidence_tc_frac", "coincidence_tc_nsigma", "coincidence_max_p",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, int | float)
                or not (math.isfinite(value) and value > 0)
            ):
                msg = f"db.{name} must be a positive number, got {value!r}"
                raise ConfigError(msg)
        if self.coincidence_max_p > 1:
            msg = f"db.coincidence_max_p must be in (0, 1], got {self.coincidence_max_p!r}"
            raise ConfigError(msg)
        if self.coincidence_tc_frac > 1:
            msg = f"db.coincidence_tc_frac must be in (0, 1], got {self.coincidence_tc_frac!r}"
            raise ConfigError(msg)
        value = self.coincidence_min_similar
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"db.coincidence_min_similar must be an integer >= 1, got {value!r}"
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class LightcurveSettings:
    """Stage-5 light curves, statistics, and best-aperture selection."""

    n_mag_bins: int = 20
    bad_flag_mask: int = 252
    keep_all_apertures_in_table: bool = False
    output_format: str = "auto"
    make_plot: bool = True
    #: Which stars get ``lc_err`` inflated by their point-to-point excess scatter
    #: (``err_scale = max(1, sigma_p2p / median(lc_err))``, see
    #: :func:`relphot.lightcurve.error_inflation`): ``"none"``; ``"blended"`` (stars whose
    #: SExtractor neighbour/blend flags are set in at least ``blend_min_frame_fraction`` of
    #: the kept frames); ``"excess"`` (blended stars, plus any star whose measured
    #: ``err_scale`` is at least ``err_scale_excess_min`` -- the excess is a bright-star
    #: noise floor the formal errors lack, whether or not the star is blended); ``"all"``.
    inflate_errors: str = "excess"
    #: Fraction of a star's kept frames with a neighbour/blend FLAGS bit (1 or 2) at which
    #: it counts as blended.
    blend_min_frame_fraction: float = 0.05
    #: ``inflate_errors = "excess"``: smallest measured ``err_scale`` that is inflated
    #: for a star that is not blended.
    err_scale_excess_min: float = 1.5
    #: Fewest consecutive-epoch differences a star/aperture needs for a point-to-point
    #: estimate; below it ``err_scale`` is 1.
    p2p_min_pairs: int = 10


@dataclass(frozen=True, slots=True)
class DecorrelationSettings:
    """Tile-level seeing/airmass decorrelation for light curves.

    Fits per-star and then per-tile surface models to remove frame-level
    and stellar-property-dependent scatter before output.
    """

    enabled: bool = True
    min_epochs_per_star: int = 15
    clip_sigma_star: float = 4.0
    max_iter_star: int = 3
    mag_degree: int = 2
    use_crowding: bool = True
    crowding_degree: int = 1
    clip_sigma_surface: float = 4.0
    max_iter_surface: int = 2
    scope: str = "pooled"
    min_comparison_stars_per_tile: int = 30
    min_comparison_stars_total: int = 200
    use_airmass: bool = True
    max_missing_airmass_frac: float = 0.2


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Stage 6: cotrending basis vectors, single-event transit search, and
    star-variability characterisation (see :mod:`relphot.cotrend`,
    :mod:`relphot.transit_search`, :mod:`relphot.variability`,
    :mod:`relphot.catalogs`).

    The transit search never fits a free systematics model to a star's own
    light curve before searching it -- only the population-level
    cotrending basis vectors (:mod:`relphot.cotrend`) and a joint
    poly+CBV+box regression are used, so an injected transit is never
    absorbed by star-specific detrending. Per-star free CBV regression is
    used only downstream, for variability characterisation, where that
    concern does not apply.
    """

    # --- cotrending basis vectors (relphot.cotrend) ---
    #: Maximum number of leading principal components kept per (tile, aperture).
    n_cbv: int = 3
    #: Cumulative explained-variance fraction beyond which further components
    #: are dropped, even if ``n_cbv`` allows more.
    cbv_explained_variance: float = 0.95
    cbv_clip_sigma: float = 5.0
    cbv_max_iter: int = 3
    #: A frame is "systematic" for a tile/aperture when more than this
    #: fraction of its comparison-star ensemble is a simultaneous >3-sigma
    #: outlier there (relphot.cotrend.detect_systematic_frames).
    systematic_frame_fraction: float = 0.3

    # --- single-event transit search (relphot.transit_search) ---
    #: Absolute floor on good epochs at a star's best aperture (see ``min_epoch_fraction``).
    min_epochs: int = 20
    #: Required good epochs as a fraction of the night's kept frames; the effective cut
    #: is ``max(min_epochs, ceil(min_epoch_fraction * n_kept))``, so the cut scales with
    #: night length (a 45-frame night and a 351-frame night).
    min_epoch_fraction: float = 0.5
    #: Rolling-median-clip threshold (robust sigma) applied to a star's own
    #: light curve before any nuisance or box fit, to remove isolated
    #: single-epoch outliers (cosmic rays, dropped frames) that a box
    #: search would otherwise happily fit as part of an event. Not a
    #: systematics model -- a fixed global threshold this far out cannot
    #: remove a real percent-level transit dip.
    lc_clip_sigma: float = 6.0
    lc_clip_window: int = 15
    duration_min_hours: float = 0.4
    duration_max_hours: float = 2.5
    n_durations: int = 15
    #: Minimum in-transit points for a (duration, mid-time) trial to be evaluated.
    min_in_transit_points: int = 5
    #: Minimum fraction of the trial box's time span that overlaps the data span.
    min_box_coverage: float = 0.5
    #: Degree of the polynomial-in-time nuisance term (in addition to the CBVs).
    poly_degree: int = 2
    #: Transit-candidate SNR cut. On ROBO43 20250911 (516 stars, 3.1 h) the
    #: best-event SNR of stars without a real signal peaked at 4.9 (p99 4.7).
    snr_threshold: float = 6.0
    #: SNR floor for a star's own best event to count toward another star's
    #: coincidence count.
    coincidence_snr_threshold: float = 5.0
    coincidence_fraction_threshold: float = 0.3
    #: Absolute floor on coincidence_count before SHARED_EPOCH can be set,
    #: so a tiny tile pool (where one match is already a large fraction)
    #: does not trip the flag on a single coincidence.
    coincidence_min_count: int = 2
    #: A candidate's cross-aperture depth chi2/dof above this value is
    #: flagged APERTURE_INCONSISTENT.
    aperture_inconsistent_sigma: float = 3.0
    too_deep_fraction: float = 0.30
    #: Fraction of the trial duration used as the edge-touching margin.
    edge_fraction: float = 0.25
    high_beta_threshold: float = 3.0
    #: Minimum occupied time-bins (at the trial duration) required before
    #: the Pont et al. (2006) beta estimate is trusted; below this it
    #: defaults to 1.0 (see relphot.transit_search._red_noise_beta's
    #: docstring for why too few bins makes the estimator itself noisy).
    beta_min_bins: int = 8
    #: STEP_LIKE when the step model's chi2 is within this of the box model's.
    step_delta_chi2_threshold: float = 3.0
    #: A candidate's in-transit point count below this is flagged FEW_POINTS.
    few_points_threshold: int = 8
    #: Minimum epochs required on both sides of a candidate step position
    #: for the STEP_LIKE test to be evaluated there; a step position too
    #: close to either data edge is statistically indistinguishable from
    #: the box itself and would otherwise report a spurious perfect match.
    step_min_side_points: int = 10
    #: Per-frame error-inflation factor bounds (relphot.cotrend.compute_frame_error_scale).
    #: Below 1 is allowed (a quieter-than-typical frame is not penalised);
    #: the ceiling keeps one catastrophic frame from zeroing out a trial
    #: instead of merely down-weighting it.
    frame_error_scale_min: float = 0.5
    frame_error_scale_max: float = 10.0

    # --- variability characterisation (relphot.variability) ---
    excess_rms_threshold: float = 3.0
    von_neumann_sigma_threshold: float = 3.0
    ls_fap_threshold: float = 1.0e-3
    ls_min_period_minutes: float = 10.0
    ls_max_period_factor: float = 2.0
    variability_n_mag_bins: int = 20
    variability_min_bin_stars: int = 5
    #: Eclipse-like classification: fraction of points below this many robust
    #: sigma from the median needed to call the variability "eclipse-like".
    eclipse_depth_sigma: float = 3.0
    eclipse_min_fraction: float = 0.01

    # --- catalogue cross-match (relphot.catalogs) ---
    catalogs_enabled: bool = True
    neighbour_radius_arcsec: float = 8.0
    gaia_match_radius_arcsec: float = 2.0
    known_variable_match_radius_arcsec: float = 2.0
    known_planet_match_radius_arcsec: float = 5.0
    gaia_catalog: str = "I/355/gaiadr3"
    cache_dir: str = "~/.cache/relphot/search"
    #: Catalogue variability-type substrings (case-insensitive, matched
    #: against each ``|``-separated token of the catalogue's own type
    #: string) that list a star as a variable. Its transit events stay
    #: candidates and only carry the informational ON_VARIABLE flag.
    #: A generic/weak automated classification (e.g. Gaia's own
    #: "SOLAR_LIKE" or "ROT") is deliberately absent -- low-amplitude
    #: rotational modulation does not preclude a real transiting planet,
    #: and a known transiting host is routinely also catalogued this way.
    disqualifying_variable_types: tuple[str, ...] = (
        "EA", "EB", "EW", "ECL", "EC", "ESD", "RR", "CEP", "SR", "LPV",
        "CV", "UG", "DQ", "AM", "DSCT", "GDOR", "SXPHE", "M",
    )

    def effective_min_epochs(self, n_kept: int) -> int:
        """Good-epoch cut for a night with ``n_kept`` kept frames (see ``min_epoch_fraction``)."""
        return max(int(self.min_epochs), math.ceil(self.min_epoch_fraction * n_kept), 1)


@dataclass(frozen=True, slots=True)
class BorderSettings:
    """Night-level border-eligibility cut for reference and comparison stars.

    Comparison and reference stars must never come from near the detector border.
    Margins are computed per night from the measured per-night drift and telescope-
    specific extra margins. The T80S 20 rows at top and bottom are per the user's
    statement, not reproduced from the reduced SCI planes.

    ``telescope_extra_px``: per-telescope extra margins [left, right, bottom, top] (px),
    keys are upper-cased TELESCOP header values. List order and JSON round trip.
    ``edge_buffer_px`` covers the largest aperture (8 px) + background mesh + rotation.
    """

    enabled: bool = True
    edge_buffer_px: float = 30.0
    drift_min_common_stars: int = 20
    telescope_extra_px: dict[str, list[float]] = field(default_factory=lambda: {
        "T80": [0.0, 0.0, 20.0, 20.0], "T80S": [0.0, 0.0, 20.0, 20.0]})


@dataclass(frozen=True, slots=True)
class TailSettings:
    """Night-level isolated-dip ("tail") cut for reference and comparison stars.

    A star with at least ``min_low`` epochs below ``-k_sigma`` (and ``asym_ratio`` times
    as many below as above ``+k_sigma``) in its residual against the local median of its
    ``n_neighbours`` nearest bright stars is withdrawn from the reference and the
    comparison pool for the whole night; it keeps its light curve and is still searched.
    Residuals are detrended with a running median over ``window`` epochs (an odd number;
    a transit longer than ``window // 2`` epochs is not touched), divided by the star's
    point-to-point sigma and by a per-frame noise factor (cloud).

    ``aperture``: aperture index the screen is computed at; -1 means the default
    aperture (1 if the night has at least two apertures, else 0). One aperture serves
    the reference and every comparison pool, so the same stars are removed from both.
    ``pool_min_snr``: median SNR a neighbour star needs (it must also have FLAGS == 0
    wherever present). ``min_snr``: median SNR a star needs to be evaluated.
    ``min_epochs``: valid epochs a star needs to be evaluated.
    """

    enabled: bool = True
    aperture: int = -1
    k_sigma: float = 5.0
    min_low: int = 3
    asym_ratio: float = 3.0
    window: int = 7
    n_neighbours: int = 25
    pool_min_snr: float = 15.0
    min_snr: float = 5.0
    min_epochs: int = 20


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level relphot settings."""

    catalog: CatalogSettings = field(default_factory=CatalogSettings)
    site: SiteSettings = field(default_factory=SiteSettings)
    tile: TileSettings = field(default_factory=TileSettings)
    variable: VariableSettings = field(default_factory=VariableSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)
    lightcurve: LightcurveSettings = field(default_factory=LightcurveSettings)
    decorrelation: DecorrelationSettings = field(default_factory=DecorrelationSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    multinight: MultiNightSettings = field(default_factory=MultiNightSettings)
    db: DbSettings = field(default_factory=DbSettings)
    border: BorderSettings = field(default_factory=BorderSettings)
    tails: TailSettings = field(default_factory=TailSettings)


def _build(cls: type, data: dict[str, Any], origin: str, path: str) -> Any:
    """Recursively build a dataclass ``cls`` from a plain nested dict.

    Every field not present in ``data`` keeps ``cls``'s own default (so a
    partial override dict works exactly like a complete one). Raises
    :class:`~relphot.exceptions.ConfigError` on any key in ``data`` that is
    not a field of ``cls``. Nested dataclass fields are resolved via
    :func:`typing.get_type_hints` because ``from __future__ import
    annotations`` turns every ``dataclasses.Field.type`` into a plain string.
    A tuple-typed field receives a TOML/JSON list as a tuple, since neither
    source format has a tuple literal. ``path`` is the dotted location for
    error messages, empty at the root.
    """
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        bad = ", ".join(sorted(f"{path}.{k}" if path else k for k in unknown))
        msg = f"{origin}: unknown key(s) {bad}"
        raise ConfigError(msg)

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        sub_path = f"{path}.{name}" if path else name
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _build(hint, value, origin, sub_path)
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """``settings`` as a plain, JSON-serialisable nested dict."""
    return dataclasses.asdict(settings)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Inverse of :func:`settings_to_dict`. Validates unknown keys like :func:`load_settings`."""
    return _build(Settings, data, "<dict>", "")


def load_settings(path: Path | str | None = None) -> Settings:
    """Load :class:`Settings` from an optional TOML file.

    ``path=None`` (the default) returns pure defaults. A relative path is
    resolved against the current working directory. Every field not present
    in the file keeps its dataclass default.
    """
    if path is None:
        return Settings()

    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        msg = f"cannot read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid TOML in {path}: {exc}"
        raise ConfigError(msg) from exc

    return _build(Settings, data, str(path), "")
