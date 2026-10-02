"""relphot.ingest in ``catalog.photometry = "forced"`` mode (robo43 ``forced`` catalogues).

The forced catalogues are derived from the real cropped fixture frames by scaling the
standard fluxes, so the tests can tell which file was read from the flux alone.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from relphot.cli import main
from relphot.config import Settings
from relphot.exceptions import IngestError
from relphot.ingest import read_catalog, read_catalogs, resolve_catalog_source
from relphot.io import load_night

SCALE_CSV = 2.0
SCALE_EXT = 3.0


def _forced_settings(**catalog) -> Settings:
    base = Settings()
    return replace(base, catalog=replace(base.catalog, photometry="forced", **catalog))


@pytest.fixture
def night_dir(fits_files, csv_files, tmp_path) -> Path:
    """Copies of the fixture frames with a forced CSV (flux x2) beside each FITS."""
    for f in (*fits_files, *csv_files):
        shutil.copy(f, tmp_path / f.name)
    for c in csv_files:
        table = Table.read(tmp_path / c.name, format="ascii.csv")
        for col in table.colnames:
            if col.startswith("FLUX_APER_"):
                table[col] = table[col] * SCALE_CSV
        table.write(
            tmp_path / c.name.replace("_proc_catalog.csv", "_proc_forced_catalog.csv"),
            format="csv",
        )
    return tmp_path


def test_standard_mode_ignores_forced_files(night_dir, fits_files) -> None:
    p = night_dir / fits_files[0].name
    std = read_catalog(p, Settings())
    ref = read_catalog(fits_files[0], Settings())
    np.testing.assert_array_equal(std.flux, ref.flux)


def test_forced_mode_maps_every_kind_of_input_to_the_forced_csv(night_dir, fits_files) -> None:
    settings = _forced_settings()
    std = read_catalog(night_dir / fits_files[0].name, Settings())
    stem = fits_files[0].name.removesuffix("_proc.fits")
    inputs = [
        night_dir / f"{stem}_proc.fits",
        night_dir / f"{stem}_proc_catalog.csv",
        night_dir / f"{stem}_proc_forced_catalog.csv",
    ]
    for path in inputs:
        cat = read_catalog(path, settings)
        assert cat.n_sources == std.n_sources
        np.testing.assert_allclose(cat.flux, SCALE_CSV * std.flux, rtol=1e-5, equal_nan=True)
        assert cat.meta.file == night_dir / f"{stem}_proc.fits"  # same names as standard photometry
        assert cat.meta.aperture_radii_px == std.meta.aperture_radii_px
        assert cat.meta.exptime == std.meta.exptime


def test_forced_mode_reads_the_fits_extension_when_there_is_no_csv(night_dir, fits_files) -> None:
    f = night_dir / fits_files[1].name
    stem = f.name.removesuffix("_proc.fits")
    (night_dir / f"{stem}_proc_forced_catalog.csv").unlink()
    with fits.open(f) as hdul:
        ext = fits.BinTableHDU(data=hdul["CATALOG"].data.copy(), header=hdul["CATALOG"].header)
        ext.name = "CATALOG_FORCED"
        ext.data["FLUX_APER"] = ext.data["FLUX_APER"] * SCALE_EXT
        hdul.append(ext)
        hdul.writeto(night_dir / "tmp.fits")
    shutil.move(night_dir / "tmp.fits", f)
    std = read_catalog(fits_files[1], Settings())
    for path in (f, night_dir / f"{stem}_proc_catalog.csv"):
        cat = read_catalog(path, _forced_settings())
        np.testing.assert_allclose(cat.flux, SCALE_EXT * std.flux, rtol=1e-5, equal_nan=True)
        assert cat.meta.file == f
    assert resolve_catalog_source(f, _forced_settings(), None) == (f, "fits")
    with pytest.raises(IngestError):  # no forced CSV, so an explicit csv request cannot be met
        resolve_catalog_source(f, _forced_settings(), "csv")


def test_forced_mode_without_a_forced_catalogue_is_an_error(fits_files, tmp_path) -> None:
    shutil.copy(fits_files[0], tmp_path / fits_files[0].name)
    with pytest.raises(IngestError, match="robo43 forced"):
        read_catalog(tmp_path / fits_files[0].name, _forced_settings())
    with pytest.raises(IngestError, match="cannot map"):
        read_catalog(tmp_path / "notes.txt", _forced_settings())


def test_unknown_photometry_mode_is_an_error(fits_files) -> None:
    with pytest.raises(IngestError, match="photometry"):
        read_catalog(fits_files[0], replace(
            Settings(), catalog=replace(Settings().catalog, photometry="bogus")
        ))


def test_standard_auto_mode_does_not_redirect_a_forced_csv_to_the_production_fits(
    night_dir, fits_files
) -> None:
    stem = fits_files[0].name.removesuffix("_proc.fits")
    forced_csv = night_dir / f"{stem}_proc_forced_catalog.csv"
    assert resolve_catalog_source(forced_csv, Settings(), None) == (forced_csv, "csv")
    cat = read_catalog(forced_csv, Settings())
    std = read_catalog(night_dir / f"{stem}_proc.fits", Settings())
    np.testing.assert_allclose(cat.flux, SCALE_CSV * std.flux, rtol=1e-5, equal_nan=True)


def test_read_catalogs_forced_deduplicates_inputs_that_map_to_one_catalogue(
    night_dir, fits_files
) -> None:
    f = night_dir / fits_files[0].name
    csv = night_dir / f.name.replace("_proc.fits", "_proc_catalog.csv")
    cats = read_catalogs([f, csv], _forced_settings())
    assert len(cats) == 1


def test_cli_ingest_forced_records_the_photometry_in_the_night(night_dir, fits_files, tmp_path):
    files = [str(night_dir / p.name) for p in fits_files]
    out_std = tmp_path / "std.npz"
    out_forced = tmp_path / "forced.npz"
    assert main(["ingest", "--out", str(out_std), *files]) == 0
    assert main(["ingest", "--photometry", "forced", "--out", str(out_forced), *files]) == 0
    night_std, settings_std = load_night(out_std)
    night_forced, settings_forced = load_night(out_forced)
    assert settings_std.catalog.photometry == "standard"
    assert settings_forced.catalog.photometry == "forced"
    assert night_forced.n_frames == night_std.n_frames
    ok = np.isfinite(night_std.flux) & np.isfinite(night_forced.flux)
    assert ok.any()
    # same stars, the forced fluxes are the scaled standard ones
    assert np.nanmedian(night_forced.flux[ok] / night_std.flux[ok]) == pytest.approx(
        SCALE_CSV, rel=1e-3
    )
