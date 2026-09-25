"""CLI smoke test: `relphot ingest` on the real cropped fixture frames."""

from __future__ import annotations

from relphot.cli import main
from relphot.io import load_night


def test_ingest_smoke(fits_files, tmp_path) -> None:
    out = tmp_path / "night.npz"
    argv = ["ingest", "--out", str(out), *[str(p) for p in fits_files]]
    rc = main(argv)
    assert rc == 0
    assert out.is_file()
    report = out.with_suffix(".report.csv")
    assert report.is_file()

    night, _settings = load_night(out)
    assert night.n_stars_after_cut > 0
    assert night.n_frames == len(fits_files)


def test_ingest_csv_format_smoke(csv_files, tmp_path) -> None:
    out = tmp_path / "night_csv.npz"
    argv = ["ingest", "--format", "csv", "--out", str(out), *[str(p) for p in csv_files]]
    rc = main(argv)
    assert rc == 0
    night, _settings = load_night(out)
    assert night.n_stars_after_cut > 0
