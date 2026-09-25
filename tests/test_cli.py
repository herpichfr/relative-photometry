"""CLI smoke tests: `relphot ingest` and `relphot reference` on the real cropped fixture frames."""

from __future__ import annotations

import csv

from relphot.cli import main
from relphot.io import load_night, load_reference


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


def test_reference_smoke(fits_files, tmp_path) -> None:
    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0
    assert ref_out.is_file()

    tiles_csv = ref_out.with_name("ref_tiles.csv")
    reference_csv = ref_out.with_name("ref_reference.csv")
    assert tiles_csv.is_file()
    assert reference_csv.is_file()

    tilemap, result, _settings = load_reference(ref_out)
    assert tilemap.n_tiles >= 1
    assert result.R.shape[0] == tilemap.n_tiles
    night, _night_settings = load_night(night_out)
    assert result.R.shape[1] == night.n_frames

    # Check reference CSV has the new columns
    with reference_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        assert "frame_kept" in fieldnames
        assert "n_reference_stars" in fieldnames
        rows = list(reader)
        assert len(rows) > 0
        # Check that frame_kept is 0 or 1
        for row in rows:
            assert row["frame_kept"] in ("0", "1")
            assert int(row["n_reference_stars"]) >= 0
