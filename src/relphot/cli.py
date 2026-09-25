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
from dataclasses import asdict
from pathlib import Path

import numpy as np

from relphot.comparison import select_comparison_stars
from relphot.config import Settings, load_settings
from relphot.exceptions import ComparisonError, RelphotError
from relphot.ingest import read_catalogs
from relphot.io import (
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
from relphot.variables import flag_known_variables

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
