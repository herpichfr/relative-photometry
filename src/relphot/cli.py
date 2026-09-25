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

from relphot.config import Settings, load_settings
from relphot.exceptions import RelphotError
from relphot.ingest import read_catalogs
from relphot.io import load_night, save_night, save_reference
from relphot.match import match_night
from relphot.reference import build_references, select_candidates, select_reference_frames_and_stars
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
