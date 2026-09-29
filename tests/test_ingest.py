"""Tests for relphot.ingest: both adapters, and metadata sanity, on real cropped frames."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from relphot import ingest
from relphot.config import Settings
from relphot.exceptions import IngestError
from relphot.ingest import (
    _compute_times,
    _gaia_zero_point,
    _site_from_header,
    read_catalog,
    read_catalogs,
    read_csv_catalog,
    read_fits_catalog,
)


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


def test_site_from_header_with_nina_sitelat_sitelong_siteelev() -> None:
    """A header with only SITELAT/SITELONG/SITEELEV floats gives correct
    EarthLocation matching those values to 1e-6 deg / 1e-3 m."""
    settings = Settings()
    header = fits.Header()
    header["SITELAT"] = -22.534556
    header["SITELONG"] = -45.5825
    header["SITEELEV"] = 1864.0

    site = _site_from_header(header, settings)
    assert site is not None
    np.testing.assert_allclose(site.lat.deg, -22.534556, atol=1e-6)
    np.testing.assert_allclose(site.lon.deg, -45.5825, atol=1e-6)
    np.testing.assert_allclose(site.height.to("m").value, 1864.0, atol=1e-3)


def test_compute_times_with_nina_site_returns_finite_bjd_tdb() -> None:
    """_compute_times on a header with SITELAT/SITELONG/SITEELEV floats and
    valid DATE-OBS/EXPTIME/RA/DEC returns a finite bjd_tdb with proper
    barycentric correction."""
    settings = Settings()
    header = fits.Header()
    header["DATE-OBS"] = "2025-09-12T01:52:44.447"
    header["EXPTIME"] = 30.0
    header["SITELAT"] = -22.534556
    header["SITELONG"] = -45.5825
    header["SITEELEV"] = 1864.0
    # Use sexagesimal strings to avoid issues with RA as float hours
    header["RA"] = "21:29:00.9"  # 21:29:00.9 in hourangle = 322.2537644 deg
    header["DEC"] = "-58:50:10.1"  # -58:50:10.1 in deg

    jd_utc, bjd_tdb = _compute_times(header, None, settings)

    # BJD_TDB should be finite
    assert np.isfinite(bjd_tdb)
    # Barycentric correction should be at most ~8.3 min + TT-UTC (0.08 d)
    assert np.isfinite(jd_utc)
    assert abs(bjd_tdb - jd_utc) < 0.01


def test_latitude_longitud_take_precedence_over_sitelat_sitelong() -> None:
    """LATITUDE/LONGITUD take precedence over SITELAT/SITELONG when both are
    present."""
    settings = Settings()
    header = fits.Header()
    # Primary keys (T80S)
    header["LATITUDE"] = "-22.5"
    header["LONGITUD"] = "-45.5"
    header["ALTITUDE"] = 1850.0
    # ASCOM/N.I.N.A. keys that should be ignored
    header["SITELAT"] = -22.534556
    header["SITELONG"] = -45.5825
    header["SITEELEV"] = 1864.0

    site = _site_from_header(header, settings)
    assert site is not None
    # Should match LATITUDE/LONGITUD, not SITELAT/SITELONG
    np.testing.assert_allclose(site.lat.deg, -22.5, atol=1e-6)
    np.testing.assert_allclose(site.lon.deg, -45.5, atol=1e-6)
    np.testing.assert_allclose(site.height.to("m").value, 1850.0, atol=1e-3)


def test_gaia_zero_point_needs_a_valid_calibrated_zpabs() -> None:
    header = fits.Header()
    assert _gaia_zero_point(header) is None  # a frame without any calibration
    header["ZPABS"] = 24.31
    assert _gaia_zero_point(header) is None  # no ZPABSCAL
    header["ZPABSCAL"] = False  # RMS above the limit: ZPABS is there but not trusted
    assert _gaia_zero_point(header) is None
    header["ZPABSCAL"] = True
    assert _gaia_zero_point(header) == pytest.approx(24.31)
    header["ZPABS"] = "NONE"  # what robo43 writes when no zero point was fitted
    assert _gaia_zero_point(header) is None
    header["ZPABS"] = "nan"  # FITS cannot hold a NaN number, but a string can spell one
    assert _gaia_zero_point(header) is None


def test_read_catalogs_carry_no_zero_point_from_uncalibrated_frames(fits_files, csv_files) -> None:
    settings = Settings()
    assert read_fits_catalog(fits_files[0], settings).meta.zp is None
    assert read_csv_catalog(csv_files[0], settings).meta.zp is None


def test_read_fits_catalog_sets_naxis_and_telescope(fits_files) -> None:
    """read_fits_catalog reads NAXIS1/NAXIS2/TELESCOP from header."""
    settings = Settings()
    fc = read_fits_catalog(fits_files[0], settings)

    # Fields should be set (test fixtures have empty primary headers, so naxis=0)
    assert isinstance(fc.meta.naxis1, int)
    assert isinstance(fc.meta.naxis2, int)
    assert isinstance(fc.meta.telescope, str)


def test_read_csv_catalog_sets_naxis_and_telescope(csv_files) -> None:
    """read_csv_catalog reads NAXIS1/NAXIS2/TELESCOP from companion FITS header."""
    settings = Settings()
    fc = read_csv_catalog(csv_files[0], settings)

    # Fields should be set (test fixtures have empty primary headers)
    assert isinstance(fc.meta.naxis1, int)
    assert isinstance(fc.meta.naxis2, int)
    assert isinstance(fc.meta.telescope, str)


def test_finite_position_rows_drops_nan_rows(caplog) -> None:
    ra = np.array([10.0, np.nan, 12.0])
    dec = np.array([-70.0, np.nan, np.inf])
    with caplog.at_level("WARNING"):
        keep = ingest._finite_position_rows(ra, dec, "cat.fits")
    assert keep.tolist() == [True, False, False]
    assert "dropping 2 catalogue row(s)" in caplog.text
    assert ingest._finite_position_rows(ra[:1], dec[:1], "cat.fits") is None
