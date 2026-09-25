"""Command-line entry point: ``relphot ingest``.

Reads a night's per-frame catalogues, cross-matches them, and writes the
result as a ``.npz`` next to a CSV match report.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from relphot.config import Settings, load_settings
from relphot.exceptions import RelphotError
from relphot.ingest import read_catalogs
from relphot.io import save_night
from relphot.match import match_night

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
