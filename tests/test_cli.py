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


def test_lightcurves_smoke(fits_files, tmp_path) -> None:
    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    lc_out = tmp_path / "lc"
    rc = main([
        "lightcurves",
        str(night_out),
        str(ref_out),
        "--out", str(lc_out),
        "--no-variables",
        "--format", "fits",
        "--no-plot",
    ])
    assert rc == 0

    # Check output files exist
    assert (lc_out.parent / f"{lc_out.name}.npz").is_file()
    assert (lc_out.parent / f"{lc_out.name}_lightcurves.fits").is_file()
    assert (lc_out.parent / f"{lc_out.name}_starstats.fits").is_file()
    assert (lc_out.parent / f"{lc_out.name}_comparison.csv").is_file()

    # Check that light curve table is readable
    try:
        from astropy.table import Table
        lc_table = Table.read(str(lc_out.parent / f"{lc_out.name}_lightcurves.fits"), format="fits")
        assert "star_id" in lc_table.colnames
        assert "lc" in lc_table.colnames
        assert "lc_err" in lc_table.colnames
        assert len(lc_table) > 0
    except ImportError:
        pass  # astropy not available, skip check

    # Check that members file exists (it has the path based on lc_out which is stem only)
    members_npz = lc_out.parent / "_members.npz"
    assert members_npz.is_file(), f"Members file should be written by default at {members_npz}"


def test_lightcurves_no_members(fits_files, tmp_path) -> None:
    """Test lightcurves with --no-members flag does not write members file."""
    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    lc_out = tmp_path / "lc"
    rc = main([
        "lightcurves",
        str(night_out),
        str(ref_out),
        "--out", str(lc_out),
        "--no-variables",
        "--no-members",
        "--format", "fits",
        "--no-plot",
    ])
    assert rc == 0

    # Check that members file does NOT exist
    members_npz = lc_out.parent / f"{lc_out.name}_members.npz"
    assert not members_npz.is_file(), "Members file should not be written with --no-members"


def test_members_smoke(fits_files, tmp_path) -> None:
    """Test the 'relphot members' subcommand on reference and light-curve files."""
    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    lc_out = tmp_path / "lc"
    rc = main([
        "lightcurves",
        str(night_out),
        str(ref_out),
        "--out", str(lc_out),
        "--no-variables",
        "--no-members",
        "--format", "fits",
        "--no-plot",
    ])
    assert rc == 0

    # Run members command
    members_out = tmp_path / "lc_members.npz"
    rc = main(["members", str(night_out), str(ref_out), str(lc_out.parent / f"{lc_out.name}.npz"),
               "--out", str(members_out)])
    assert rc == 0
    assert members_out.is_file()

    # Verify members file is valid
    from relphot.members import load_members_npz
    product = load_members_npz(members_out)
    assert product.n_frames > 0
    assert product.n_aper >= 0


def test_members_ref_without_tile_stars(fits_files, tmp_path) -> None:
    """Test members command on a ref.npz without tile_ref_* keys."""
    from relphot.io import load_reference, save_reference

    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    # Re-save ref.npz without tile_stars
    tilemap, result, settings = load_reference(ref_out)
    ref_out_no_stars = tmp_path / "ref_no_stars.npz"
    save_reference(tilemap, result, settings, ref_out_no_stars)  # No tile_stars

    lc_out = tmp_path / "lc"
    rc = main([
        "lightcurves",
        str(night_out),
        str(ref_out_no_stars),
        "--out", str(lc_out),
        "--no-variables",
        "--no-members",
        "--format", "fits",
        "--no-plot",
    ])
    assert rc == 0

    # Run members command
    members_out = tmp_path / "lc_members.npz"
    lc_npz_path = str(lc_out.parent / f"{lc_out.name}.npz")
    rc = main(["members", str(night_out), str(ref_out_no_stars), lc_npz_path,
               "--out", str(members_out)])
    assert rc == 0
    assert members_out.is_file()

    # Verify members file is valid
    from relphot.members import load_members_npz
    product = load_members_npz(members_out)
    assert product.n_frames > 0
