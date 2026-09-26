"""Command-line entry points: ``relphot ingest`` and ``relphot reference``.

``ingest`` reads a night's per-frame catalogues, cross-matches them, and
writes the result as a ``.npz`` next to a CSV match report. ``reference``
takes that ``.npz``, builds the adaptive tile grid and per-tile reference
fluxes, and writes them as a ``.npz`` next to a tile-map CSV and a
per-tile-per-frame reference CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from relphot.catalogs import (
    compute_neighbour_dilution,
    is_disqualifying_variable_type,
    match_known_planets,
    match_known_variables,
)
from relphot.comparison import select_comparison_stars
from relphot.config import Settings, load_settings
from relphot.cotrend import compute_cbvs, detect_systematic_frames, select_star_epochs
from relphot.exceptions import ComparisonError, RelphotError
from relphot.ingest import read_catalogs
from relphot.io import (
    load_lightcurves_npz,
    load_night,
    load_reference,
    save_decorrelation_report,
    save_lightcurve_table,
    save_lightcurves_npz,
    save_night,
    save_reference,
    save_starstats_table,
)
from relphot.lightcurve import compute_light_curves
from relphot.match import match_night
from relphot.numeric import nanmedian_quiet
from relphot.reference import build_references, select_candidates, select_reference_frames_and_stars
from relphot.stats import (
    best_aperture_per_star,
    build_diagnostics_table,
    compute_star_stats,
    plot_rms_vs_magnitude,
    select_best_aperture,
)
from relphot.tiles import build_tilemap
from relphot.transit_search import (
    FLAG_NEIGHBOUR_BLEND,
    FLAG_TOO_DEEP,
    fit_nuisance_model,
    flags_to_string,
    plot_transit_candidate,
    search_one_star,
    search_transits,
    tier_for_flags,
)
from relphot.variables import (
    compute_star_variability,
    flag_known_variables,
    plot_variable_star,
)

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _expand_inputs(args: list[str]) -> list[Path]:
    """Expand CLI file arguments, including one or more ``@list.txt`` files.

    An ``@``-prefixed argument names a text file with one catalogue path per
    line (blank lines and ``#``-prefixed comments ignored); every other
    argument is a catalogue path taken as given (shell globbing has already
    expanded any ``*``).
    """
    paths: list[Path] = []
    for arg in args:
        if arg.startswith("@"):
            list_file = Path(arg[1:])
            for line in list_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(Path(line))
        else:
            paths.append(Path(arg))
    return paths


def _write_report_csv(night, path: Path) -> None:
    """Write the per-frame match report as a CSV next to the .npz output."""
    with path.open("w", newline="") as handle:
        fieldnames = [
            "file", "n_sources", "n_matched_to_master", "match_fraction", "median_sep_arcsec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in night.reports:
            writer.writerow(asdict(report))


def _write_reference_csv(night, tilemap, result, aper: int, frame_selection, path: Path) -> None:
    """Write one row per (tile, frame): bjd_tdb, airmass, reference at ``aper``, and star counts."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tile",
            "frame",
            "bjd_tdb",
            "airmass",
            "R",
            "sigma_R",
            "n_used",
            "frame_kept",
            "n_reference_stars",
        ])
        for t in range(tilemap.n_tiles):
            n_reference_stars = len(frame_selection.tile_stars[t])
            for j, meta in enumerate(night.frame_meta):
                writer.writerow([
                    t,
                    j,
                    meta.bjd_tdb,
                    meta.airmass,
                    result.R[t, j, aper],
                    result.sigma_R[t, j, aper],
                    int(result.n_used[t, j, aper]),
                    int(result.frame_kept[j]),
                    n_reference_stars,
                ])


def _run_ingest(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        settings: Settings = load_settings(args.config)
    except RelphotError:
        logger.exception("failed to load config %s", args.config)
        return 1

    paths = _expand_inputs(args.files)
    if not paths:
        logger.error("no input files given")
        return 1

    out_path = Path(args.out)

    t0 = time.monotonic()
    try:
        catalogs = read_catalogs(paths, settings, fmt=args.format, max_workers=args.np)
    except RelphotError:
        logger.exception("failed to read catalogues")
        return 1
    t1 = time.monotonic()
    logger.info("ingest: %d frames in %.2f s", len(catalogs), t1 - t0)

    try:
        night = match_night(catalogs, settings)
    except RelphotError:
        logger.exception("failed to match frames")
        return 1
    t2 = time.monotonic()
    logger.info("match: %.2f s", t2 - t1)

    save_night(night, settings, out_path)
    t3 = time.monotonic()
    logger.info("save: %.2f s", t3 - t2)
    logger.info("total: %.2f s", t3 - t0)

    report_path = out_path.with_suffix(".report.csv")
    _write_report_csv(night, report_path)

    print(f"stars before presence cut: {night.n_stars_before_cut}")
    print(f"stars after presence cut:  {night.n_stars_after_cut}")
    print(f"master frame: {night.frame_meta[night.master_frame_index].file}")
    fractions = [r.match_fraction for r in night.reports]
    print(f"match fraction range: {min(fractions):.3f} - {max(fractions):.3f}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


def _run_reference(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        settings: Settings = load_settings(args.config)
    except RelphotError:
        logger.exception("failed to load config %s", args.config)
        return 1

    try:
        night, _ingest_settings = load_night(args.night)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.night)
        return 1

    if night.n_aper == 0:
        logger.error("%s has no apertures", args.night)
        return 1
    aper = (
        args.aper if args.aper is not None else (1 if night.n_aper >= 2 else 0)
    )
    if not (0 <= aper < night.n_aper):
        logger.error("--aper %d out of range [0, %d)", aper, night.n_aper)
        return 1

    t0 = time.monotonic()
    if args.no_variables:
        logger.info("known-variable cross-match skipped (--no-variables)")
        variable_mask = np.zeros(night.n_stars, dtype=bool)
    else:
        variable_mask = flag_known_variables(night, settings)
    t1 = time.monotonic()
    logger.info("variables: %.2f s (%d flagged)", t1 - t0, int(variable_mask.sum()))

    candidates = select_candidates(night, variable_mask, settings, aper)

    try:
        tilemap = build_tilemap(night, candidates, settings)
    except RelphotError:
        logger.exception("tiling failed")
        return 1
    t2 = time.monotonic()
    logger.info("tiling: %.2f s (%d tiles)", t2 - t1, tilemap.n_tiles)

    try:
        frame_selection = select_reference_frames_and_stars(
            night, tilemap, candidates, settings, aper
        )
    except RelphotError:
        logger.exception("frame/star selection failed")
        return 1
    t3 = time.monotonic()
    logger.info("frame/star selection: %.2f s", t3 - t2)
    n_kept = int(np.sum(frame_selection.frame_kept))
    n_total = night.n_frames
    logger.info("frames kept: %d/%d", n_kept, n_total)
    if frame_selection.dropped_frames:
        dropped_files = [night.frame_meta[i].file.name for i in frame_selection.dropped_frames]
        logger.info("dropped frames: %s", ", ".join(dropped_files))

    result = build_references(night, tilemap, frame_selection, settings)
    t4 = time.monotonic()
    logger.info("reference: %.2f s", t4 - t3)

    out_path = Path(args.out)
    save_reference(tilemap, result, settings, out_path)

    tiles_csv = out_path.with_name(f"{out_path.stem}_tiles.csv")
    tilemap.to_csv(tiles_csv)
    reference_csv = out_path.with_name(f"{out_path.stem}_reference.csv")
    _write_reference_csv(night, tilemap, result, aper, frame_selection, reference_csv)

    print(f"stars: {night.n_stars}, tiles: {tilemap.n_tiles}, aperture: {aper}")
    print(f"candidates: {int(candidates.sum())}")
    print(f"frames kept: {n_kept}/{n_total}")
    print(f"wrote {out_path}")
    print(f"wrote {tiles_csv}")
    print(f"wrote {reference_csv}")
    return 0


def _run_lightcurves(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Load settings, night, and reference
    try:
        settings: Settings = load_settings(args.config)
    except RelphotError:
        logger.exception("failed to load config %s", args.config)
        return 1

    try:
        night, _ingest_settings = load_night(args.night)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.night)
        return 1

    try:
        tilemap, reference_result, _ref_settings = load_reference(args.reference)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.reference)
        return 1

    if night.n_aper == 0:
        logger.error("%s has no apertures", args.night)
        return 1

    # Determine mag_aper: use the reference settings' aperture if stored
    mag_aper = (
        1 if night.n_aper >= 2 else 0
    )  # Default rule

    # Override with CLI flags if present
    if hasattr(args, "mag_aper") and args.mag_aper is not None:
        mag_aper = args.mag_aper

    if not (0 <= mag_aper < night.n_aper):
        logger.error("--mag-aper %d out of range [0, %d)", mag_aper, night.n_aper)
        return 1

    # Settings overrides
    from dataclasses import replace

    if hasattr(args, "keep_all_apertures") and args.keep_all_apertures:
        settings = replace(
            settings,
            lightcurve=replace(settings.lightcurve, keep_all_apertures_in_table=True),
        )

    if hasattr(args, "no_decorrelation") and args.no_decorrelation:
        settings = replace(
            settings,
            decorrelation=replace(settings.decorrelation, enabled=False),
        )

    if hasattr(args, "format") and args.format:
        settings = replace(
            settings,
            lightcurve=replace(settings.lightcurve, output_format=args.format),
        )

    if hasattr(args, "no_plot") and args.no_plot:
        settings = replace(
            settings,
            lightcurve=replace(settings.lightcurve, make_plot=False),
        )

    t0 = time.monotonic()

    # Variable mask
    if args.no_variables:
        logger.info("known-variable cross-match skipped (--no-variables)")
        variable_mask = np.zeros(night.n_stars, dtype=bool)
    else:
        variable_mask = flag_known_variables(night, settings)
    t1 = time.monotonic()
    logger.info("variables: %.2f s (%d flagged)", t1 - t0, int(variable_mask.sum()))

    # Comparison stars
    try:
        comparison_result = select_comparison_stars(
            night, tilemap, reference_result, variable_mask, settings
        )
    except ComparisonError:
        logger.exception("comparison star selection failed")
        return 1
    t2 = time.monotonic()
    logger.info("comparison: %.2f s", t2 - t1)

    # Light curves
    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    t3 = time.monotonic()
    logger.info("light curves: %.2f s", t3 - t2)

    # Star statistics
    star_stats = compute_star_stats(lc_result)
    t4 = time.monotonic()
    logger.info("star stats: %.2f s", t4 - t3)

    # Best aperture selection
    best_aper_per_tile, bin_edges = select_best_aperture(
        tilemap, comparison_result, star_stats, settings, mag_aper
    )
    star_best_aper = best_aperture_per_star(
        tilemap, comparison_result, best_aper_per_tile, bin_edges, mag_aper
    )
    t5 = time.monotonic()
    logger.info("best aperture: %.2f s", t5 - t4)

    # Output
    out_path = Path(args.out)
    out_stem = out_path.with_suffix("")

    # Save npz
    save_lightcurves_npz(
        lc_result,
        star_stats,
        best_aper_per_tile,
        bin_edges,
        star_best_aper,
        comparison_result,
        settings,
        out_path,
    )

    # Save tables
    actual_fmt = settings.lightcurve.output_format
    if actual_fmt == "auto":
        try:
            import importlib.util
            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False
        actual_fmt = "parquet" if has_pyarrow else "fits"

    lc_out_path = out_stem.with_name(f"{out_stem.name}_lightcurves")
    lc_table_path = save_lightcurve_table(
        night, tilemap, lc_result, star_best_aper, settings, lc_out_path, actual_fmt
    )

    stats_out_path = out_stem.with_name(f"{out_stem.name}_starstats")
    starstats_path = save_starstats_table(
        night, tilemap, comparison_result, star_stats, star_best_aper, settings,
        stats_out_path, actual_fmt
    )

    # Comparison diagnostics
    diag_table = build_diagnostics_table(tilemap, comparison_result, lc_result.decorrelation)
    diag_path = out_stem.with_name(f"{out_stem.name}_comparison.csv")
    with diag_path.open("w", newline="") as handle:
        if diag_table:
            fieldnames = list(diag_table[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diag_table)
    logger.info("wrote %s", diag_path)

    # Decorrelation report
    if lc_result.decorrelation is not None:
        decorr_path = out_stem.with_name(f"{out_stem.name}_decorrelation.csv")
        aper_radii = night.frame_meta[0].aperture_radii_px
        save_decorrelation_report(lc_result.decorrelation, aper_radii, decorr_path)

    # Plots
    if settings.lightcurve.make_plot:
        # Per aperture, plot RMS vs mag for comparison stars in each tile
        apertures_used = set()
        for t in range(tilemap.n_tiles):
            for b in range(best_aper_per_tile.shape[1]):
                if best_aper_per_tile[t, b] >= 0:
                    apertures_used.add(int(best_aper_per_tile[t, b]))

        for a in sorted(apertures_used):
            # All stars with a core tile and this aperture
            core = tilemap.core_tile >= 0
            all_mag = np.where(core, comparison_result.mag[:, a], np.nan)
            all_rms = np.where(core, star_stats.rms[:, a], np.nan)
            all_exp = np.where(core, star_stats.expected_noise[:, a], np.nan)
            all_is_comp = core & comparison_result.mask[:, a]

            plot_path = out_stem.with_name(f"{out_stem.name}_rms_vs_mag_aper{a}.png")
            try:
                plot_rms_vs_magnitude(
                    all_mag,
                    all_rms,
                    all_exp,
                    all_is_comp,
                    str(plot_path),
                    title=f"RMS vs Magnitude (aperture {a})",
                )
            except RelphotError as e:
                logger.warning("failed to plot: %s", e)

    t6 = time.monotonic()
    logger.info("total: %.2f s", t6 - t0)

    # Summary
    with_lc = int((tilemap.core_tile >= 0).sum())
    kept_frames = int(np.sum(reference_result.frame_kept))
    total_frames = night.n_frames

    # Comparison count range
    n_comp_min = comparison_result.n_comparison[comparison_result.n_comparison > 0].min()
    n_comp_max = comparison_result.n_comparison.max()

    # Best aperture histogram
    unique_best, counts = np.unique(star_best_aper[star_best_aper >= 0], return_counts=True)
    best_hist = ", ".join(
        [f"aper{int(a)}: {int(c)}" for a, c in zip(unique_best, counts, strict=False)]
    )

    # Bright star floor (median RMS of brightest 1-mag bin of comparison stars)
    bright_floors = []
    for a in range(night.n_aper):
        comp_mag_a = comparison_result.mag[comparison_result.mask[:, a], a]
        comp_rms_a = star_stats.rms[comparison_result.mask[:, a], a]
        if comp_mag_a.size > 0:
            bright_mag = np.min(comp_mag_a)
            bright_mask = comp_mag_a < (bright_mag + 1.0)
            if np.any(bright_mask):
                # rms is a fractional scatter; 1 mmag = 1.0857e-3 in flux ratio
                bright_floor_mmag = nanmedian_quiet(comp_rms_a[bright_mask]) * 1085.7
                bright_floors.append(f"aper{a}: {bright_floor_mmag:.1f} mmag")
    bright_floor_str = ", ".join(bright_floors) if bright_floors else "N/A"

    print(f"stars with light curves: {with_lc}")
    print(f"frames kept: {kept_frames}/{total_frames}")
    print(f"comparison count range: {int(n_comp_min)}-{int(n_comp_max)}")
    print(f"best aperture histogram: {best_hist}")
    print(f"bright-star floor: {bright_floor_str}")
    print(f"wrote {out_path}")
    print(f"wrote {lc_table_path}")
    print(f"wrote {starstats_path}")
    print(f"wrote {diag_path}")

    return 0


def _frame_error_scale_at_tc(tilemap, transit_result, night) -> np.ndarray:
    """Per-star :func:`relphot.cotrend.compute_frame_error_scale` value at each star's own tc.

    NaN for a star that was not searched or has no finite tc. Stored in the
    per-star metrics table as a compact summary of the full
    ``(n_tiles, n_aper, n_frames)`` array (also written in full as
    ``frame_error_scale.csv`` next to the transit candidates).
    """
    n_stars = night.n_stars
    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    out = np.full(n_stars, np.nan)
    for i in range(n_stars):
        if not transit_result.searched[i] or not np.isfinite(transit_result.tc[i]):
            continue
        t_tile = int(tilemap.core_tile[i])
        a = int(transit_result.aper[i])
        if t_tile < 0 or a < 0:
            continue
        frame_idx = int(np.clip(np.searchsorted(bjd, transit_result.tc[i]), 0, len(bjd) - 1))
        out[i] = transit_result.frame_error_scale[t_tile, a, frame_idx]
    return out


def _search_table_columns(
    night, tilemap, comparison_result, star_stats, star_best_aper,
    transit_result, variability_result, variable_match, planet_match, neighbour,
):
    """One row per star: every Stage-6 metric, for ``search_metrics.parquet``."""
    n_stars = night.n_stars
    n_aper = night.n_aper
    best_a = np.asarray(star_best_aper, dtype=np.int64)
    has_aper = best_a >= 0
    a_safe = np.where(has_aper, best_a, 0)

    def _at_best(arr, fill=np.nan):
        return np.where(has_aper, arr[np.arange(n_stars), a_safe], fill)

    columns = {
        "star_id": np.arange(n_stars, dtype=np.int64),
        "tile": tilemap.core_tile.astype(np.int64),
        "ra": night.ra,
        "dec": night.dec,
        "mag": _at_best(comparison_result.mag),
        "best_aperture": best_a,
        "rms": _at_best(star_stats.rms),
        "n_epochs": _at_best(star_stats.n_epochs, 0).astype(np.int64),
        "transit_searched": transit_result.searched,
        "transit_snr": transit_result.snr,
        "transit_depth": transit_result.depth,
        "transit_tc_bjd_tdb": transit_result.tc,
        "transit_duration_hours": transit_result.duration * 24.0,
        "transit_n_in": transit_result.n_in,
        "transit_beta": transit_result.beta,
        "transit_coverage": transit_result.coverage,
        "transit_partial": transit_result.partial,
        "transit_tier": np.array(
            [tier_for_flags(int(b)) for b in transit_result.flags], dtype=np.int64
        ),
        "transit_frame_error_scale_at_tc": _frame_error_scale_at_tc(
            tilemap, transit_result, night
        ),
        "transit_dchi2_box_vs_flat": transit_result.dchi2_box_vs_flat,
        "transit_dchi2_box_vs_step": transit_result.dchi2_box_vs_step,
        "transit_coincidence_count": transit_result.coincidence_count,
        "transit_flags": transit_result.flags,
        "transit_flags_str": np.array(
            [flags_to_string(int(b)) for b in transit_result.flags], dtype=object
        ),
        "transit_candidate": transit_result.candidate,
        "variability_searched": variability_result.searched,
        "variability_rms_robust": variability_result.rms_robust,
        "variability_rms_std": variability_result.rms_std,
        "variability_excess": variability_result.excess,
        "variability_von_neumann": variability_result.von_neumann,
        "variability_von_neumann_significance": variability_result.von_neumann_significance,
        "variability_ls_period_days": variability_result.ls_period_days,
        "variability_ls_power": variability_result.ls_power,
        "variability_ls_fap": variability_result.ls_fap,
        "variability_amplitude": variability_result.amplitude,
        "variability_trend_slope": variability_result.trend_slope,
        "variability_trend_significance": variability_result.trend_significance,
        "variability_systematic_excluded": variability_result.systematic_excluded,
        "variability_candidate": variability_result.variable_candidate,
        "variability_class": np.array(variability_result.variable_class, dtype=object),
        "known_variable": variable_match.matched,
        "known_variable_name": variable_match.name,
        "known_variable_type": variable_match.var_type,
        "known_variable_period_days": variable_match.period_days,
        "known_planet": planet_match.matched,
        "known_planet_name": planet_match.name,
        "known_planet_period_days": planet_match.period_days,
        "known_planet_depth": planet_match.depth,
        "known_planet_is_toi": planet_match.is_toi,
        "gaia_id": neighbour.gaia_id,
        "neighbour_sep_arcsec": neighbour.neighbour_sep_arcsec,
        "max_dilutable_depth": neighbour.max_dilutable_depth,
    }
    for a in range(n_aper):
        columns[f"transit_depth_aper{a}"] = transit_result.depth_per_aper[:, a]
        columns[f"transit_sigma_depth_aper{a}"] = transit_result.sigma_depth_per_aper[:, a]
    return columns


def _write_table_auto(columns: dict, path: Path, fmt: str) -> Path:
    """Write ``columns`` (a plain dict of equal-length 1-D arrays) as ``path``.

    An ``object``-dtype column (every string column built in this module,
    since values are assembled with ``dtype=object`` to hold arbitrary-length
    strings) is cast to a fixed-width unicode array first -- astropy's
    parquet writer cannot serialise a plain ``object`` column.
    """
    from astropy.table import Table

    if fmt == "auto":
        try:
            import importlib.util

            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False
        fmt = "parquet" if has_pyarrow else "fits"

    columns = {
        key: (np.asarray(value, dtype=str) if np.asarray(value).dtype == object else value)
        for key, value in columns.items()
    }
    table = Table(columns)
    if fmt == "parquet":
        out_path = path.with_suffix(".parquet")
        table.write(out_path, format="parquet", overwrite=True)
    else:
        out_path = path.with_suffix(".fits")
        table.write(out_path, format="fits", overwrite=True)
    logger.info("wrote %s (%d rows)", out_path, len(table))
    return out_path


def _run_search(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        settings: Settings = load_settings(args.config)
    except RelphotError:
        logger.exception("failed to load config %s", args.config)
        return 1
    if args.snr_threshold is not None:
        settings = replace(
            settings, search=replace(settings.search, snr_threshold=args.snr_threshold)
        )

    try:
        night, _ingest_settings = load_night(args.night)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.night)
        return 1
    try:
        tilemap, reference_result, _ref_settings = load_reference(args.reference)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.reference)
        return 1
    try:
        (
            lc_result, star_stats, _best_aper_per_tile, _bin_edges, star_best_aper,
            comparison_result, _lc_settings, _decorrelation,
        ) = load_lightcurves_npz(args.lc)
    except (OSError, RelphotError):
        logger.exception("failed to load %s", args.lc)
        return 1

    t0 = time.monotonic()
    frame_kept = reference_result.frame_kept
    lc = lc_result.lc
    lc_err = lc_result.lc_err
    epoch_ok = lc_result.epoch_ok

    cotrend_result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    systematic_frames = detect_systematic_frames(
        tilemap, comparison_result, lc, frame_kept, settings.search
    )
    t1 = time.monotonic()
    logger.info("cotrend: %.2f s", t1 - t0)

    transit_result = search_transits(
        night, tilemap, cotrend_result, comparison_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, settings,
    )
    t2 = time.monotonic()
    logger.info(
        "transit search: %.2f s (%d candidates)", t2 - t1, int(transit_result.candidate.sum())
    )

    variability_result = compute_star_variability(
        night, tilemap, comparison_result, cotrend_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, systematic_frames, settings,
    )
    t3 = time.monotonic()
    logger.info(
        "variability search: %.2f s (%d candidates)",
        t3 - t2,
        int(variability_result.variable_candidate.sum()),
    )

    n_stars = night.n_stars
    if args.no_catalogs:
        empty_str = np.full(n_stars, "", dtype=object)
        empty_bool = np.zeros(n_stars, dtype=bool)
        empty_nan = np.full(n_stars, np.nan)
        from relphot.catalogs import NeighbourDilutionResult, PlanetMatchResult, VariableMatchResult

        variable_match = VariableMatchResult(empty_bool, empty_str, empty_str, empty_nan)
        planet_match = PlanetMatchResult(
            empty_bool, empty_str, empty_nan, empty_nan, np.zeros(n_stars, dtype=bool)
        )
        neighbour = NeighbourDilutionResult(empty_str, empty_bool, empty_nan, empty_nan)
    else:
        variable_match = match_known_variables(night, settings)
        planet_match = match_known_planets(night, settings)
        neighbour = compute_neighbour_dilution(night, settings)
    t4 = time.monotonic()
    logger.info("catalogue cross-match: %.2f s", t4 - t3)

    blend = (
        neighbour.has_neighbour
        & np.isfinite(transit_result.depth)
        & np.isfinite(neighbour.max_dilutable_depth)
        & (transit_result.depth < neighbour.max_dilutable_depth)
    )
    transit_result.flags[blend] |= FLAG_NEIGHBOUR_BLEND

    # Route TOO_DEEP transit events to variables (as eclipsing), and exclude
    # any star the variability search independently calls variable -- or that
    # a catalogue already lists as a known variable, whether or not this
    # night's own statistics reach significance for it -- from the transit
    # candidate list. A real periodic/eclipsing variable is not a
    # single-event transit, whichever check happened to catch it.
    variable_candidate = variability_result.variable_candidate.copy()
    variable_class = list(variability_result.variable_class)
    too_deep = (transit_result.flags & FLAG_TOO_DEEP).astype(bool) & transit_result.searched
    newly_too_deep = too_deep & ~variable_candidate
    variable_candidate[too_deep] = True
    for i in np.nonzero(newly_too_deep)[0]:
        variable_class[i] = "eclipse-like"

    disqualifying_known = np.array(
        [
            bool(variable_match.matched[i])
            and is_disqualifying_variable_type(str(variable_match.var_type[i]), settings)
            for i in range(n_stars)
        ]
    )
    known_only = disqualifying_known & ~variable_candidate
    variable_candidate[known_only] = True
    for i in np.nonzero(known_only)[0]:
        variable_class[i] = "known (catalogue only)"

    final_transit_candidate = transit_result.candidate & ~variable_candidate

    min_ep = settings.search.effective_min_epochs(int(np.count_nonzero(frame_kept)))

    transits_dir = Path(args.transits_dir)
    variables_dir = Path(args.variables_dir)
    transits_dir.mkdir(parents=True, exist_ok=True)
    variables_dir.mkdir(parents=True, exist_ok=True)

    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)

    # --- transit candidates ---
    transit_rows = []
    cand_idx = np.nonzero(final_transit_candidate)[0]
    for i in cand_idx:
        t_tile = int(tilemap.core_tile[i])
        pool = np.nonzero(
            transit_result.searched
            & (tilemap.core_tile == t_tile)
            & np.isfinite(transit_result.snr)
            & (transit_result.snr >= settings.search.coincidence_snr_threshold)
        )[0]
        shared_bjd = transit_result.tc[pool[pool != i]]

        row = {
            "star_id": int(i),
            "tile": t_tile,
            "ra": float(night.ra[i]),
            "dec": float(night.dec[i]),
            "mag": float(comparison_result.mag[i, transit_result.aper[i]]),
            "snr": float(transit_result.snr[i]),
            "depth": float(transit_result.depth[i]),
            "tc_bjd_tdb": float(transit_result.tc[i]),
            "duration_hours": float(transit_result.duration[i] * 24.0),
            "n_in": int(transit_result.n_in[i]),
            "beta": float(transit_result.beta[i]),
            "dchi2_box_vs_flat": float(transit_result.dchi2_box_vs_flat[i]),
            "dchi2_box_vs_step": float(transit_result.dchi2_box_vs_step[i]),
            "coincidence_count": int(transit_result.coincidence_count[i]),
            "flags": flags_to_string(int(transit_result.flags[i])),
            "tier": tier_for_flags(int(transit_result.flags[i])),
            "coverage": float(transit_result.coverage[i]),
            "best_aperture": int(transit_result.aper[i]),
            "known_planet_name": str(planet_match.name[i]),
            "known_planet_period_days": float(planet_match.period_days[i]),
            "known_planet_is_toi": bool(planet_match.is_toi[i]),
            "known_variable_name": str(variable_match.name[i]),
            "known_variable_type": str(variable_match.var_type[i]),
            "gaia_id": str(neighbour.gaia_id[i]),
        }
        for a in range(night.n_aper):
            row[f"depth_aper{a}"] = float(transit_result.depth_per_aper[i, a])
            row[f"sigma_depth_aper{a}"] = float(transit_result.sigma_depth_per_aper[i, a])
        transit_rows.append(row)

        a_i = int(transit_result.aper[i])
        n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a_i, :, 0])))
        cbv_rows = cotrend_result.basis[t_tile, a_i, :n_cbv, :]
        y = lc[i, :, a_i]
        err = lc_err[i, :, a_i] * transit_result.frame_error_scale[t_tile, a_i, :]
        med = nanmedian_quiet(np.where(epoch_ok[i] & frame_kept, y, np.nan))
        good = (
            epoch_ok[i] & frame_kept & np.isfinite(y) & np.isfinite(err)
        )
        star_result = search_one_star(
            bjd, y / med, err / med, good, cbv_rows, settings.search,
            aper=a_i, n_aper=night.n_aper, keep_grid=True,
        )

        lc_csv = transits_dir / f"star{i:04d}_lc.csv"
        with lc_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bjd_tdb", "lc", "lc_err", "model"])
            n_pts = len(star_result.t_good) if star_result.t_good is not None else 0
            model = (
                star_result.model_good if star_result.model_good is not None else [np.nan] * n_pts
            )
            if star_result.t_good is not None:
                for tt, yy, ee, mm in zip(
                    star_result.t_good, star_result.y_good, star_result.err_good, model, strict=True
                ):
                    writer.writerow([tt, yy, ee, mm])

        if not args.no_plot and star_result.ok:
            png_path = transits_dir / f"star{i:04d}_transit.png"
            try:
                plot_transit_candidate(
                    i, star_result, transit_result.depth_per_aper[i],
                    transit_result.sigma_depth_per_aper[i], shared_bjd, str(png_path),
                )
            except RelphotError as exc:
                logger.warning("failed to plot star %d: %s", i, exc)

    transit_rows.sort(key=lambda row: (row["tier"], -row["snr"]))
    transit_csv = transits_dir / "candidates.csv"
    with transit_csv.open("w", newline="") as handle:
        if transit_rows:
            fieldnames = list(transit_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transit_rows)
        else:
            handle.write("")
    logger.info("wrote %s (%d candidates)", transit_csv, len(transit_rows))

    frame_scale_csv = transits_dir / "frame_error_scale.csv"
    with frame_scale_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "aperture", "frame", "bjd_tdb", "frame_error_scale"])
        for t_tile in range(tilemap.n_tiles):
            for a in range(night.n_aper):
                for j in range(night.n_frames):
                    writer.writerow([
                        t_tile, a, j, bjd[j], transit_result.frame_error_scale[t_tile, a, j]
                    ])
    logger.info("wrote %s", frame_scale_csv)

    # --- variability candidates ---
    variable_rows = []
    var_idx = np.nonzero(variable_candidate)[0]
    for i in var_idx:
        known = bool(variable_match.matched[i] or planet_match.matched[i])
        row = {
            "star_id": int(i),
            "tile": int(tilemap.core_tile[i]),
            "ra": float(night.ra[i]),
            "dec": float(night.dec[i]),
            "mag": float(comparison_result.mag[i, max(int(variability_result.aper[i]), 0)]),
            "rms_robust_mmag": float(variability_result.rms_robust[i] * 1000.0),
            "rms_std_mmag": float(variability_result.rms_std[i] * 1000.0),
            "excess": float(variability_result.excess[i]),
            "von_neumann": float(variability_result.von_neumann[i]),
            "von_neumann_significance": float(variability_result.von_neumann_significance[i]),
            "ls_period_days": float(variability_result.ls_period_days[i]),
            "ls_fap": float(variability_result.ls_fap[i]),
            "amplitude": float(variability_result.amplitude[i]),
            "trend_significance": float(variability_result.trend_significance[i]),
            "variable_class": variable_class[i],
            "known": known,
            "known_name": str(variable_match.name[i]) or str(planet_match.name[i]),
            "known_type": str(variable_match.var_type[i]),
            "known_period_days": (
                float(variable_match.period_days[i])
                if variable_match.matched[i]
                else float(planet_match.period_days[i])
            ),
            "gaia_id": str(neighbour.gaia_id[i]),
        }
        variable_rows.append(row)

        ok_ep, _a_v, t_g, y_norm, err_norm, cbv_rows, _idx = select_star_epochs(
            tilemap, cotrend_result, bjd, lc, lc_err, epoch_ok, frame_kept, star_best_aper,
            min_ep, i,
        )
        prefix = "KNOWN_" if known else "NEW_"
        if ok_ep:
            # No polynomial here, matching relphot.variables.fit_star_variability_metrics's
            # primary (constant + CBV) fit exactly, so the exported residual/plot agree
            # with the reported rms/von Neumann/Lomb-Scargle metrics.
            base = fit_nuisance_model(t_g, y_norm, 1.0 / err_norm**2, cbv_rows, 0)
            resid = base[2] if base is not None else np.full_like(t_g, np.nan)
            csv_path = variables_dir / f"{prefix}star{i:04d}.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["bjd_tdb", "lc_norm", "lc_err_norm", "resid"])
                for tt, yy, ee, rr in zip(t_g, y_norm, err_norm, resid, strict=True):
                    writer.writerow([tt, yy, ee, rr])

            if not args.no_plot:
                png_path = variables_dir / f"{prefix}star{i:04d}.png"
                metrics = {
                    "rms_robust": variability_result.rms_robust[i],
                    "ls_period_days": variability_result.ls_period_days[i],
                    "ls_power": variability_result.ls_power[i],
                    "ls_fap": variability_result.ls_fap[i],
                }
                try:
                    plot_variable_star(i, t_g, y_norm, resid, metrics, str(png_path))
                except RelphotError as exc:
                    logger.warning("failed to plot star %d: %s", i, exc)

    variables_csv = variables_dir / "candidates.csv"
    new_variables_csv = variables_dir / "new_variables.csv"
    with variables_csv.open("w", newline="") as handle:
        if variable_rows:
            fieldnames = list(variable_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variable_rows)
    new_rows = [row for row in variable_rows if not row["known"]]
    with new_variables_csv.open("w", newline="") as handle:
        # Header even when empty, so "no new variables" reads as a valid table.
        if variable_rows:
            writer = csv.DictWriter(handle, fieldnames=list(variable_rows[0].keys()))
            writer.writeheader()
            writer.writerows(new_rows)
    logger.info(
        "wrote %s (%d candidates, %d new)", variables_csv, len(variable_rows), len(new_rows)
    )

    # --- full per-star metrics table ---
    columns = _search_table_columns(
        night, tilemap, comparison_result, star_stats, star_best_aper,
        transit_result, variability_result, variable_match, planet_match, neighbour,
    )
    metrics_path = Path(args.lc).with_suffix("")
    metrics_out = metrics_path.with_name(f"{metrics_path.name}_search_metrics")
    metrics_written = _write_table_auto(columns, metrics_out, "auto")

    t5 = time.monotonic()
    logger.info("total: %.2f s", t5 - t0)

    print(f"stars searched (transits): {int(transit_result.searched.sum())}")
    print(f"transit candidates: {len(transit_rows)}")
    print(f"variability candidates: {len(variable_rows)} ({len(new_rows)} new)")
    print(f"wrote {transit_csv}")
    print(f"wrote {variables_csv}")
    print(f"wrote {new_variables_csv}")
    print(f"wrote {metrics_written}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relphot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="read and cross-match one night's catalogues")
    ingest.add_argument("--config", type=Path, default=None, help="TOML settings file")
    ingest.add_argument("--out", required=True, help="output .npz path")
    ingest.add_argument(
        "--format", choices=["fits", "csv"], default=None,
        help=(
            "catalogue format (default: auto — FITS catalogue HDU, using a "
            "CSV's companion *_proc.fits when it has one)"
        ),
    )
    ingest.add_argument(
        "--np", type=int, default=4, dest="np",
        help="parallel readers (default: 4; measured optimum for this workload)",
    )
    ingest.add_argument("files", nargs="+", help="catalogue files, or @list.txt")
    ingest.set_defaults(func=_run_ingest)

    reference = subparsers.add_parser(
        "reference", help="build the adaptive tile grid and per-tile reference fluxes"
    )
    reference.add_argument("night", help="input .npz written by `relphot ingest`")
    reference.add_argument("--config", type=Path, default=None, help="TOML settings file")
    reference.add_argument("--out", required=True, help="output .npz path")
    reference.add_argument(
        "--no-variables", action="store_true",
        help="skip the known-variable cross-match (flag no star as a known variable)",
    )
    reference.add_argument(
        "--aper", type=int, default=None,
        help=(
            "aperture index for candidate selection and the reference CSV "
            "(default: index 1 if n_aper >= 2, else 0)"
        ),
    )
    reference.set_defaults(func=_run_reference)

    lightcurves = subparsers.add_parser(
        "lightcurves", help="extract light curves and statistics"
    )
    lightcurves.add_argument("night", help="input .npz written by `relphot ingest`")
    lightcurves.add_argument("reference", help="input .npz written by `relphot reference`")
    lightcurves.add_argument("--config", type=Path, default=None, help="TOML settings file")
    lightcurves.add_argument("--out", required=True, help="output .npz stem (without extension)")
    lightcurves.add_argument(
        "--no-variables", action="store_true",
        help="skip the known-variable cross-match (flag no star as a known variable)",
    )
    lightcurves.add_argument(
        "--mag-aper", type=int, default=None,
        help=(
            "aperture index for magnitude reference in best-aperture selection "
            "(default: index 1 if n_aper >= 2, else 0)"
        ),
    )
    lightcurves.add_argument(
        "--keep-all-apertures", action="store_true",
        help="write all apertures to light-curve table with is_best_aperture column",
    )
    lightcurves.add_argument(
        "--format", choices=["auto", "parquet", "fits"], default="auto",
        help="output table format (default: auto — parquet if pyarrow available, else fits)",
    )
    lightcurves.add_argument(
        "--no-plot", action="store_true",
        help="skip RMS vs magnitude plots",
    )
    lightcurves.add_argument(
        "--no-decorrelation", action="store_true",
        help="disable decorrelation (output lc = lc_raw)",
    )
    lightcurves.set_defaults(func=_run_lightcurves)

    search = subparsers.add_parser(
        "search", help="single-event transit search and variability characterisation"
    )
    search.add_argument("night", help="input .npz written by `relphot ingest`")
    search.add_argument("reference", help="input .npz written by `relphot reference`")
    search.add_argument("lc", help="input .npz written by `relphot lightcurves`")
    search.add_argument("--config", type=Path, default=None, help="TOML settings file")
    search.add_argument(
        "--transits-dir", required=True, help="output directory for transit-search products"
    )
    search.add_argument(
        "--variables-dir", required=True, help="output directory for variability products"
    )
    search.add_argument(
        "--snr-threshold", type=float, default=None,
        help="override settings.search.snr_threshold for the transit candidate cut",
    )
    search.add_argument(
        "--no-catalogs", action="store_true",
        help="skip every catalogue cross-match (known variables, exoplanets, Gaia neighbours)",
    )
    search.add_argument(
        "--no-plot", action="store_true", help="skip per-candidate/per-star PNG diagnostics",
    )
    search.set_defaults(func=_run_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
