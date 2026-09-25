"""Tests for relphot.ingest: both adapters, and metadata sanity, on real cropped frames."""

from __future__ import annotations

import numpy as np
import pytest

from relphot.config import Settings
from relphot.exceptions import IngestError
from relphot.ingest import read_catalog, read_catalogs, read_csv_catalog, read_fits_catalog


def test_fits_and_csv_adapters_agree(fits_files, csv_files) -> None:
    settings = Settings()
    fc = read_fits_catalog(fits_files[0], settings)
    cc = read_csv_catalog(csv_files[0], settings)

    assert fc.n_sources == cc.n_sources > 0
    assert fc.n_aper == cc.n_aper == 5
    np.testing.assert_allclose(fc.ra, cc.ra)
    np.testing.assert_allclose(fc.dec, cc.dec)
    np.testing.assert_allclose(fc.x, cc.x)
    np.testing.assert_allclose(fc.y, cc.y)
    np.testing.assert_allclose(fc.flux, cc.flux, equal_nan=True)
    np.testing.assert_allclose(fc.fluxerr, cc.fluxerr, equal_nan=True)
    np.testing.assert_array_equal(fc.flags, cc.flags)
    np.testing.assert_allclose(fc.snr, cc.snr, equal_nan=True)
    np.testing.assert_allclose(fc.fwhm, cc.fwhm, equal_nan=True)
    np.testing.assert_allclose(fc.background, cc.background, equal_nan=True)

    assert fc.meta.date_obs == cc.meta.date_obs
    assert fc.meta.exptime == cc.meta.exptime
    assert fc.meta.aperture_radii_px == cc.meta.aperture_radii_px == (2.0, 3.0, 4.0, 6.0, 8.0)


def test_dispatch_by_extension(fits_files, csv_files) -> None:
    settings = Settings()
    a = read_catalog(fits_files[0], settings)
    b = read_catalog(csv_files[0], settings)
    assert a.n_sources == b.n_sources


def test_metadata_is_sane(fits_files) -> None:
    settings = Settings()
    fc = read_fits_catalog(fits_files[0], settings)
    meta = fc.meta
    assert meta.exptime == 90.0
    assert meta.filter == "R"
    # JD_UTC for a 2025-11-05 UTC observation.
    assert 2460984.0 < meta.jd_utc < 2460985.0
    # BJD_TDB differs from JD_UTC only by light-travel time plus the small
    # UTC/TDB offset -- at most a few minutes for a geocentric-ish target.
    assert np.isfinite(meta.bjd_tdb)
    assert abs(meta.bjd_tdb - meta.jd_utc) < 0.01
    assert meta.n_sources == fc.n_sources


def test_read_catalogs_preserves_order(fits_files) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits", max_workers=2)
    assert [c.meta.file for c in cats] == list(fits_files)


def test_auto_mode_prefers_companion_fits(fits_files, csv_files) -> None:
    """In auto mode, a CSV with a companion FITS is read as FITS instead."""
    settings = Settings()
    auto = read_catalog(csv_files[0], settings)
    direct = read_fits_catalog(fits_files[0], settings)

    assert auto.meta.file == fits_files[0]
    assert auto.meta.file.suffix == ".fits"
    np.testing.assert_array_equal(auto.ra, direct.ra)
    np.testing.assert_array_equal(auto.dec, direct.dec)
    np.testing.assert_array_equal(auto.x, direct.x)
    np.testing.assert_array_equal(auto.y, direct.y)
    np.testing.assert_array_equal(auto.flux, direct.flux)
    np.testing.assert_array_equal(auto.fluxerr, direct.fluxerr)


def test_explicit_csv_format_still_reads_csv(csv_files) -> None:
    """An explicit fmt='csv' reads the CSV even when a companion FITS exists."""
    settings = Settings()
    cc = read_catalog(csv_files[0], settings, fmt="csv")
    assert cc.meta.file == csv_files[0]
    assert cc.meta.file.suffix == ".csv"


def test_csv_without_companion_in_auto_mode(csv_files, tmp_path) -> None:
    """In auto mode, a CSV with no companion FITS falls through to the CSV
    reader, which raises IngestError."""
    settings = Settings()
    # Copy a CSV to tmp_path without its companion FITS
    csv = csv_files[0]
    csv_copy = tmp_path / csv.name
    csv_copy.write_bytes(csv.read_bytes())

    with pytest.raises(IngestError, match=r"companion FITS header.*is required"):
        read_catalog(csv_copy, settings)


def test_duplicate_csv_and_fits_yields_one_catalog(fits_files, csv_files) -> None:
    """Passing both a CSV and its companion FITS in auto mode yields one
    catalogue -- the FITS version -- not two."""
    settings = Settings()
    cats = read_catalogs([csv_files[0], fits_files[0]], settings)
    assert len(cats) == 1
    assert cats[0].meta.file == fits_files[0]
