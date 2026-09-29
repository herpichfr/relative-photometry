"""Catalogue adapters: read one per-frame photometric catalogue into a
:class:`FrameCatalog` of dense per-source arrays plus :class:`FrameMeta`.

Two concrete adapters are provided -- :func:`read_fits_catalog` for the
robo43 MEF ``*_proc.fits`` (a ``CATALOG`` BinTableHDU alongside SCI/ERR/DQ/
WEIGHT image HDUs) and :func:`read_csv_catalog` for the companion
``*_proc_catalog.csv`` -- plus :func:`read_catalog`, which dispatches on file
extension, and :func:`read_catalogs`, which reads many files in parallel.
Both adapters go through :func:`_scalar_columns`, so a third photometry
method's catalogue is read by pointing :class:`relphot.config.ColumnMap` at
its own column names; nothing here is robo43-specific beyond the defaults.

The FITS adapter opens with ``memmap=True`` and reads only the primary header
and the named catalogue BinTableHDU -- real ``*_proc.fits`` files are ~1.3 GB
MEFs (SCI/ERR/DQ/WEIGHT images, each the full detector, then CATALOG then
CATFLAGS); the image planes are never touched.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from relphot.exceptions import IngestError

if TYPE_CHECKING:
    from relphot.config import ColumnMap, Settings

logger = logging.getLogger(__name__)

__all__ = [
    "FrameCatalog",
    "FrameMeta",
    "read_catalog",
    "read_catalogs",
    "read_csv_catalog",
    "read_fits_catalog",
]

#: (RA, Dec) sentinel for "no usable frame centre"; propagates to a NaN BJD_TDB.
_NO_CENTRE = (float("nan"), float("nan"))


@dataclass(slots=True)
class FrameMeta:
    """Per-frame metadata, independent of any individual source."""

    file: Path
    date_obs: str
    exptime: float
    jd_utc: float
    bjd_tdb: float
    airmass: float | None
    filter: str
    object: str
    median_fwhm: float
    n_sources: int
    #: Aperture radii as stored in the catalogue header (``APERRAD``), in the
    #: same unit the catalogue itself uses (pixels, for robo43 SExtractor
    #: output) -- never converted here.
    aperture_radii_px: tuple[float, ...]
    #: The frame's own WCS (primary header, TAN-SIP), used by
    #: :mod:`relphot.match` to project matched RA/Dec onto the master
    #: frame's pixel grid. Not JSON-serialisable -- excluded by :meth:`to_dict`.
    wcs: WCS | None = None
    #: Detector X size (NAXIS1), in pixels; 0 if unknown.
    naxis1: int = 0
    #: Detector Y size (NAXIS2), in pixels; 0 if unknown.
    naxis2: int = 0
    #: Telescope name (TELESCOP header), empty string if unknown.
    telescope: str = ""

    def to_dict(self) -> dict[str, object]:
        """This metadata as a plain, JSON-serialisable dict (drops ``wcs``)."""
        return {
            "file": str(self.file),
            "date_obs": self.date_obs,
            "exptime": self.exptime,
            "jd_utc": self.jd_utc,
            "bjd_tdb": self.bjd_tdb,
            "airmass": self.airmass,
            "filter": self.filter,
            "object": self.object,
            "median_fwhm": self.median_fwhm,
            "n_sources": self.n_sources,
            "aperture_radii_px": list(self.aperture_radii_px),
            "naxis1": self.naxis1,
            "naxis2": self.naxis2,
            "telescope": self.telescope,
        }


@dataclass(slots=True)
class FrameCatalog:
    """One frame's source catalogue as dense per-source arrays.

    ``flux``/``fluxerr`` are ``(n_sources, n_aper)``; every other array is
    ``(n_sources,)``. Positions are float64 (arcsec-level astrometry needs
    the precision); photometry and shape measurements are float32.
    """

    meta: FrameMeta
    ra: np.ndarray
    dec: np.ndarray
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    fluxerr: np.ndarray
    flags: np.ndarray
    snr: np.ndarray
    fwhm: np.ndarray
    background: np.ndarray

    @property
    def n_sources(self) -> int:
        return int(self.ra.shape[0])

    @property
    def n_aper(self) -> int:
        return int(self.flux.shape[1]) if self.flux.ndim == 2 else 1


def _as_vector(raw: np.ndarray) -> np.ndarray:
    """Ensure a flux/fluxerr array is 2-D ``(n_sources, n_aper)``.

    A FITS vector column with a single aperture element, or a scalar column
    used as a stand-in for it, reads back as 1-D; every downstream consumer
    expects a 2-D array.
    """
    return raw if raw.ndim == 2 else raw.reshape(-1, 1)


def _parse_aperrad(raw: object) -> tuple[float, ...]:
    """Parse the ``APERRAD`` header card, e.g. ``'2.0,3.0,4.0,6.0,8.0'``."""
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(float(tok) for tok in text.split(","))


def _sexagesimal_deg(raw: object, *, is_ra: bool) -> float | None:
    """Parse a sexagesimal header string (``LATITUDE``/``LONGITUD``/``RA``/``DEC``) to degrees."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    unit = "hourangle" if is_ra else "deg"
    try:
        return float(Angle(text, unit=unit).to("deg").value)
    except (ValueError, TypeError):
        return None


def _site_from_header(header: fits.Header, settings: Settings) -> EarthLocation | None:
    """Build an :class:`~astropy.coordinates.EarthLocation` from header keys.

    Header keys are tried first: ``LATITUDE``/``LONGITUD``/``ALTITUDE`` (T80S),
    then ``SITELAT``/``SITELONG``/``SITEELEV`` (N.I.N.A./ASCOM, as written for ROBO43).
    Falls back to ``settings.site`` for any missing piece; returns ``None``
    only when latitude or longitude is unavailable from either source.
    """
    lat = _sexagesimal_deg(header.get("LATITUDE"), is_ra=False)
    lon = _sexagesimal_deg(header.get("LONGITUD"), is_ra=False)
    try:
        elev = float(header["ALTITUDE"]) if header.get("ALTITUDE") is not None else None
    except (TypeError, ValueError):
        elev = None

    if lat is None:
        lat = _sexagesimal_deg(header.get("SITELAT"), is_ra=False)
    if lon is None:
        lon = _sexagesimal_deg(header.get("SITELONG"), is_ra=False)
    if elev is None:
        try:
            elev = float(header["SITEELEV"]) if header.get("SITEELEV") is not None else None
        except (TypeError, ValueError):
            elev = None

    if lat is None:
        lat = settings.site.latitude_deg
    if lon is None:
        lon = settings.site.longitude_deg
    if elev is None:
        elev = settings.site.elevation_m

    if lat is None or lon is None:
        return None
    return EarthLocation.from_geodetic(lon=lon, lat=lat, height=elev)


def _naxis(header: fits.Header, wcs: WCS | None) -> tuple[float, float] | None:
    """Frame size (NAXIS1, NAXIS2) from the header, or from ``wcs.pixel_shape``."""
    n1 = header.get("NAXIS1")
    n2 = header.get("NAXIS2")
    if n1 and n2:
        return float(n1), float(n2)
    if wcs is not None and wcs.pixel_shape is not None:
        return float(wcs.pixel_shape[0]), float(wcs.pixel_shape[1])
    return None


def _frame_centre(header: fits.Header, wcs: WCS | None) -> tuple[float, float]:
    """Frame-centre (RA, Dec) in degrees, from the WCS if present, else the header."""
    if wcs is not None and wcs.has_celestial:
        size = _naxis(header, wcs)
        if size is not None:
            try:
                sky = wcs.pixel_to_world(size[0] / 2.0, size[1] / 2.0)
                return float(sky.ra.deg), float(sky.dec.deg)
            except Exception:
                pass

    ra = _sexagesimal_deg(header.get("RA"), is_ra=True)
    dec = _sexagesimal_deg(header.get("DEC"), is_ra=False)
    if ra is not None and dec is not None:
        return ra, dec
    return _NO_CENTRE


def _compute_times(header: fits.Header, wcs: WCS | None, settings: Settings) -> tuple[float, float]:
    """Mid-exposure (JD_UTC, BJD_TDB); BJD_TDB is NaN when the site or centre is unusable.

    The site (if any) is passed to :class:`~astropy.time.Time` at
    construction, not assigned to ``.location`` afterwards -- Astropy
    deprecates the latter.
    """
    date_obs = header.get("DATE-OBS")
    exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    if date_obs is None:
        return float("nan"), float("nan")

    try:
        t_start = Time(str(date_obs), format="isot", scale="utc")
    except ValueError:
        return float("nan"), float("nan")

    site = _site_from_header(header, settings)
    mid_jd = t_start.jd + (exptime / 2.0) / 86400.0
    t_mid = Time(mid_jd, format="jd", scale="utc", location=site)
    jd_utc = float(t_mid.jd)

    ra, dec = _frame_centre(header, wcs)
    if site is None or not np.isfinite(ra) or not np.isfinite(dec):
        return jd_utc, float("nan")

    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    try:
        ltt = t_mid.light_travel_time(target, kind="barycentric")
        bjd_tdb = float((t_mid.tdb + ltt).jd)
    except Exception:
        bjd_tdb = float("nan")
    return jd_utc, bjd_tdb


_ScalarColumns = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _scalar_columns(table: Table, columns: ColumnMap, source: str) -> _ScalarColumns:
    """Pull the scalar columns shared by both adapters out of an Astropy Table."""
    try:
        ra = np.asarray(table[columns.ra], dtype=np.float64)
        dec = np.asarray(table[columns.dec], dtype=np.float64)
        x = np.asarray(table[columns.x], dtype=np.float64)
        y = np.asarray(table[columns.y], dtype=np.float64)
        flags = np.asarray(table[columns.flags], dtype=np.int32)
        snr = np.asarray(table[columns.snr], dtype=np.float32)
        fwhm = np.asarray(table[columns.fwhm], dtype=np.float32)
        background = np.asarray(table[columns.background], dtype=np.float32)
    except KeyError as exc:
        msg = f"{source}: missing expected column {exc}"
        raise IngestError(msg) from exc
    return ra, dec, x, y, flags, snr, fwhm, background


def read_fits_catalog(path: Path | str, settings: Settings) -> FrameCatalog:
    """Read the robo43 MEF ``*_proc.fits`` catalogue.

    Opens with ``memmap=True`` and touches only the primary header and the
    ``settings.catalog.hdu_name`` BinTableHDU; SCI/ERR/DQ/WEIGHT are never
    read.
    """
    path = Path(path)
    columns = settings.catalog.columns
    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header.copy()
            try:
                cat_hdu = hdul[settings.catalog.hdu_name]
            except KeyError as exc:
                msg = f"{path}: no '{settings.catalog.hdu_name}' HDU"
                raise IngestError(msg) from exc
            table = Table(cat_hdu.data)
            aperrad = _parse_aperrad(cat_hdu.header.get("APERRAD"))
    except OSError as exc:
        msg = f"cannot open {path}: {exc}"
        raise IngestError(msg) from exc

    flux = _as_vector(np.asarray(table[columns.flux], dtype=np.float32))
    fluxerr = _as_vector(np.asarray(table[columns.fluxerr], dtype=np.float32))
    ra, dec, x, y, flags, snr, fwhm, background = _scalar_columns(table, columns, str(path))

    try:
        wcs = WCS(header, relax=True)
    except Exception:
        wcs = None

    jd_utc, bjd_tdb = _compute_times(header, wcs, settings)
    size = _naxis(header, wcs)
    naxis1, naxis2 = (int(size[0]), int(size[1])) if size else (0, 0)
    telescope = str(header.get("TELESCOP", "")).strip()
    meta = FrameMeta(
        file=path,
        date_obs=str(header.get("DATE-OBS", "")),
        exptime=float(header.get("EXPTIME", 0.0) or 0.0),
        jd_utc=jd_utc,
        bjd_tdb=bjd_tdb,
        airmass=(float(header["AIRMASS"]) if header.get("AIRMASS") is not None else None),
        filter=str(header.get("FILTER", "")),
        object=str(header.get("OBJECT", "")),
        median_fwhm=float(np.nanmedian(fwhm)) if fwhm.size else float("nan"),
        n_sources=int(ra.shape[0]),
        aperture_radii_px=aperrad,
        wcs=wcs,
        naxis1=naxis1,
        naxis2=naxis2,
        telescope=telescope,
    )
    return FrameCatalog(
        meta=meta, ra=ra, dec=dec, x=x, y=y, flux=flux, fluxerr=fluxerr,
        flags=flags, snr=snr, fwhm=fwhm, background=background,
    )


def _companion_fits(csv_path: Path) -> Path | None:
    """The ``*_proc.fits`` a ``*_proc_catalog.csv`` was derived from, if present."""
    if csv_path.name.endswith("_proc_catalog.csv"):
        candidate = csv_path.with_name(csv_path.name[: -len("_catalog.csv")] + ".fits")
        if candidate.is_file():
            return candidate
    return None


def read_csv_catalog(path: Path | str, settings: Settings) -> FrameCatalog:
    """Read a robo43 ``*_proc_catalog.csv`` (split ``FLUX_APER_1..N`` columns).

    Frame metadata (DATE-OBS, EXPTIME, WCS, ...) comes from the companion
    ``*_proc.fits`` primary header (same stem, dropping ``_catalog`` from the
    CSV name). The companion FITS file is required.
    """
    path = Path(path)
    columns = settings.catalog.columns
    try:
        table = Table.read(path, format="ascii.csv")
    except OSError as exc:
        msg = f"cannot open {path}: {exc}"
        raise IngestError(msg) from exc

    flux_cols: list[str] = []
    fluxerr_cols: list[str] = []
    i = 1
    while f"{columns.flux_prefix}{i}" in table.colnames:
        flux_cols.append(f"{columns.flux_prefix}{i}")
        fluxerr_cols.append(f"{columns.fluxerr_prefix}{i}")
        i += 1
    if not flux_cols:
        msg = f"{path}: no columns matching '{columns.flux_prefix}<N>'"
        raise IngestError(msg)

    flux = np.column_stack([np.asarray(table[c], dtype=np.float32) for c in flux_cols])
    fluxerr = np.column_stack([np.asarray(table[c], dtype=np.float32) for c in fluxerr_cols])
    ra, dec, x, y, flags, snr, fwhm, background = _scalar_columns(table, columns, str(path))

    companion = _companion_fits(path)
    if companion is None:
        msg = (
            f"{path}: companion FITS header (same stem, `.fits`) is required for observation times"
        )
        raise IngestError(msg)

    header = fits.getheader(companion, 0)
    with fits.open(companion, memmap=True) as hdul:
        try:
            aperrad = _parse_aperrad(hdul[settings.catalog.hdu_name].header.get("APERRAD"))
        except KeyError:
            aperrad = ()
    try:
        wcs = WCS(header, relax=True)
    except Exception:
        wcs = None
    jd_utc, bjd_tdb = _compute_times(header, wcs, settings)
    date_obs = str(header.get("DATE-OBS", ""))
    exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    airmass = float(header["AIRMASS"]) if header.get("AIRMASS") is not None else None
    filt = str(header.get("FILTER", ""))
    obj = str(header.get("OBJECT", ""))
    size = _naxis(header, wcs)
    naxis1, naxis2 = (int(size[0]), int(size[1])) if size else (0, 0)
    telescope = str(header.get("TELESCOP", "")).strip()

    meta = FrameMeta(
        file=path,
        date_obs=date_obs,
        exptime=exptime,
        jd_utc=jd_utc,
        bjd_tdb=bjd_tdb,
        airmass=airmass,
        filter=filt,
        object=obj,
        median_fwhm=float(np.nanmedian(fwhm)) if fwhm.size else float("nan"),
        n_sources=int(ra.shape[0]),
        aperture_radii_px=aperrad,
        wcs=wcs,
        naxis1=naxis1,
        naxis2=naxis2,
        telescope=telescope,
    )
    return FrameCatalog(
        meta=meta, ra=ra, dec=dec, x=x, y=y, flux=flux, fluxerr=fluxerr,
        flags=flags, snr=snr, fwhm=fwhm, background=background,
    )


def resolve_catalog_source(path: Path, settings: Settings, fmt: str | None) -> tuple[Path, str]:
    """Resolve the file and format actually read for ``path``.

    An explicit ``fmt`` (``"fits"`` or ``"csv"``, or anything else a caller
    passes) is returned unchanged. In auto mode (``fmt is None``), a ``.csv``
    path is redirected to its companion ``*_proc.fits`` -- see
    :func:`_companion_fits` -- when that companion exists and its
    ``settings.catalog.hdu_name`` BinTableHDU is present; the HDU check opens
    the companion with ``memmap=True`` and only inspects the HDU list, never
    its data. Any failure to open the companion (``OSError``) or a missing
    HDU falls back to reading the CSV itself. A non-``.csv`` path in auto
    mode is always read as FITS.
    """
    if fmt is not None:
        return path, fmt
    if path.suffix.lower() != ".csv":
        return path, "fits"
    companion = _companion_fits(path)
    if companion is not None:
        try:
            with fits.open(companion, memmap=True) as hdul:
                if settings.catalog.hdu_name in hdul:
                    logger.info("%s: using companion FITS catalogue %s", path, companion)
                    return companion, "fits"
        except OSError:
            pass
    return path, "csv"


def read_catalog(path: Path | str, settings: Settings, fmt: str | None = None) -> FrameCatalog:
    """Read one catalogue file, dispatching on ``fmt`` or the file extension.

    In auto mode (``fmt is None``) a CSV is redirected to its companion FITS
    when one is available -- see :func:`resolve_catalog_source`.
    """
    path = Path(path)
    path, fmt = resolve_catalog_source(path, settings, fmt)
    if fmt == "fits":
        return read_fits_catalog(path, settings)
    if fmt == "csv":
        return read_csv_catalog(path, settings)
    msg = f"unknown catalogue format {fmt!r}"
    raise IngestError(msg)


def read_catalogs(
    paths: list[Path | str],
    settings: Settings,
    fmt: str | None = None,
    max_workers: int | None = None,
) -> list[FrameCatalog]:
    """Read many catalogue files in parallel, preserving ``paths`` order.

    Uses a thread pool: cfitsio/ascii I/O and the numpy array construction
    that follows it release the GIL for most of the work, and threads avoid
    the per-call pickling cost a process pool would pay for every returned
    :class:`FrameCatalog` (including its embedded WCS).

    Each path is first resolved with :func:`resolve_catalog_source`. In auto
    mode this can send two different inputs (e.g. a CSV and its companion
    FITS) to the same underlying file and format; when that happens only the
    first occurrence is kept, a warning names the dropped duplicate, and the
    returned list preserves the order of the kept inputs.
    """
    if not paths:
        return []

    resolved = [resolve_catalog_source(Path(p), settings, fmt) for p in paths]

    seen: dict[tuple[Path, str], int] = {}
    keep_indices: list[int] = []
    for i, res in enumerate(resolved):
        if res in seen:
            first = paths[seen[res]]
            logger.warning(
                "%s: duplicate of %s (both resolve to %s); skipping",
                paths[i], first, res[0],
            )
            continue
        seen[res] = i
        keep_indices.append(i)

    kept = [resolved[i] for i in keep_indices]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda pf: read_catalog(pf[0], settings, pf[1]), kept))
    return results
