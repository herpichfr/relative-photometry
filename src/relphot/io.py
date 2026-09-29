"""Save and load a :class:`~relphot.match.MatchedNight`, and a
:class:`~relphot.tiles.TileMap`/:class:`~relphot.reference.ReferenceResult`
pair, as uncompressed ``.npz``.

The dense arrays are stored as ordinary ``.npz`` members; per-frame metadata,
the match reports, and a snapshot of the :class:`~relphot.config.Settings`
that produced the file are stored alongside them as JSON strings, so the
whole file round-trips through :func:`save_night`/:func:`load_night` (or
:func:`save_reference`/:func:`load_reference`) without ``allow_pickle``. A
:class:`~relphot.tiles.TileMap`'s per-tile index lists are ragged, so they
are flattened into one array plus an offsets array per list, the standard
pickle-free encoding for a jagged array in ``.npz``.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from relphot.config import Settings, settings_from_dict, settings_to_dict
from relphot.ingest import FrameMeta
from relphot.match import MatchedNight, MatchReport
from relphot.reference import ReferenceResult
from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "load_lightcurves_npz",
    "load_night",
    "load_reference",
    "save_decorrelation_report",
    "save_lightcurve_table",
    "save_lightcurves_npz",
    "save_night",
    "save_reference",
    "save_starstats_table",
]

#: Dense-array members stored verbatim in the .npz.
_ARRAY_KEYS = (
    "ra",
    "dec",
    "x",
    "y",
    "frame_x",
    "frame_y",
    "flux",
    "fluxerr",
    "fwhm",
    "snr",
    "background",
    "flags",
    "presence",
)


def save_night(night: MatchedNight, settings: Settings, path: Path | str) -> None:
    """Write ``night`` and the ``settings`` that produced it to ``path`` (uncompressed)."""
    path = Path(path)
    frame_meta_json = json.dumps([m.to_dict() for m in night.frame_meta])
    reports_json = json.dumps([asdict(r) for r in night.reports])
    config_json = json.dumps(settings_to_dict(settings))

    np.savez(
        path,
        ra=night.ra,
        dec=night.dec,
        x=night.x,
        y=night.y,
        frame_x=night.frame_x,
        frame_y=night.frame_y,
        flux=night.flux,
        fluxerr=night.fluxerr,
        fwhm=night.fwhm,
        snr=night.snr,
        background=night.background,
        flags=night.flags,
        presence=night.presence,
        master_frame_index=np.int64(night.master_frame_index),
        n_stars_before_cut=np.int64(night.n_stars_before_cut),
        n_stars_after_cut=np.int64(night.n_stars_after_cut),
        frame_meta_json=frame_meta_json,
        reports_json=reports_json,
        config_json=config_json,
    )
    logger.info("wrote %s (%d stars, %d frames)", path, night.n_stars, night.n_frames)


def load_night(path: Path | str) -> tuple[MatchedNight, Settings]:
    """Read back a night written by :func:`save_night`.

    The reconstructed :class:`~relphot.ingest.FrameMeta` entries have
    ``wcs=None`` -- the WCS is not serialised (see
    :meth:`~relphot.ingest.FrameMeta.to_dict`) since it is only needed
    transiently during :func:`relphot.match.match_night`.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in _ARRAY_KEYS}
        master_frame_index = int(data["master_frame_index"])
        n_stars_before_cut = int(data["n_stars_before_cut"])
        n_stars_after_cut = int(data["n_stars_after_cut"])
        frame_meta_raw = json.loads(str(data["frame_meta_json"]))
        reports_raw = json.loads(str(data["reports_json"]))
        config_raw = json.loads(str(data["config_json"]))

    frame_meta = [
        FrameMeta(
            file=Path(d["file"]),
            date_obs=d["date_obs"],
            exptime=d["exptime"],
            jd_utc=d["jd_utc"],
            bjd_tdb=d["bjd_tdb"],
            airmass=d["airmass"],
            filter=d["filter"],
            object=d["object"],
            median_fwhm=d["median_fwhm"],
            n_sources=d["n_sources"],
            aperture_radii_px=tuple(d["aperture_radii_px"]),
            wcs=None,
            naxis1=d.get("naxis1", 0),
            naxis2=d.get("naxis2", 0),
            telescope=d.get("telescope", ""),
            zp=d.get("zp"),
        )
        for d in frame_meta_raw
    ]
    reports = [MatchReport(**r) for r in reports_raw]
    settings = settings_from_dict(config_raw)

    night = MatchedNight(
        ra=arrays["ra"],
        dec=arrays["dec"],
        x=arrays["x"],
        y=arrays["y"],
        frame_x=arrays["frame_x"],
        frame_y=arrays["frame_y"],
        flux=arrays["flux"],
        fluxerr=arrays["fluxerr"],
        fwhm=arrays["fwhm"],
        snr=arrays["snr"],
        background=arrays["background"],
        flags=arrays["flags"],
        presence=arrays["presence"],
        frame_meta=frame_meta,
        reports=reports,
        master_frame_index=master_frame_index,
        n_stars_before_cut=n_stars_before_cut,
        n_stars_after_cut=n_stars_after_cut,
    )
    return night, settings


def _ragged_to_flat(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A list of 1-D int arrays as one concatenated array plus ``len(lists) + 1`` offsets."""
    offsets = np.zeros(len(lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([arr.shape[0] for arr in lists])
    flat = (
        np.concatenate(lists).astype(np.int64)
        if lists and offsets[-1] > 0
        else np.empty(0, dtype=np.int64)
    )
    return flat, offsets


def _flat_to_ragged(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i] : offsets[i + 1]] for i in range(offsets.shape[0] - 1)]


def save_reference(
    tilemap: TileMap, result: ReferenceResult, settings: Settings, path: Path | str
) -> None:
    """Write a :class:`~relphot.tiles.TileMap`/:class:`~relphot.reference.ReferenceResult`
    pair, and the ``settings`` that produced them, to ``path`` (uncompressed).
    """
    path = Path(path)
    core_flat, core_offsets = _ragged_to_flat(tilemap.core_indices)
    ext_flat, ext_offsets = _ragged_to_flat(tilemap.extended_indices)
    config_json = json.dumps(settings_to_dict(settings))
    dropped_frames = [int(i) for i in result.frame_kept == False]  # noqa: E712
    meta_json = json.dumps({"method": result.method, "dropped_frames": dropped_frames})

    np.savez(
        path,
        tile_xmin=tilemap.xmin,
        tile_xmax=tilemap.xmax,
        tile_ymin=tilemap.ymin,
        tile_ymax=tilemap.ymax,
        tile_n_candidates=tilemap.n_candidates,
        tile_core_tile=tilemap.core_tile,
        tile_core_flat=core_flat,
        tile_core_offsets=core_offsets,
        tile_extended_flat=ext_flat,
        tile_extended_offsets=ext_offsets,
        R=result.R,
        sigma_R=result.sigma_R,
        n_used=result.n_used,
        relative_flux=result.relative_flux,
        frame_kept=result.frame_kept,
        config_json=config_json,
        meta_json=meta_json,
    )
    logger.info("wrote %s (%d tiles, %s method)", path, tilemap.n_tiles, result.method)


def load_reference(path: Path | str) -> tuple[TileMap, ReferenceResult, Settings]:
    """Read back a tile map and reference result written by :func:`save_reference`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        tilemap = TileMap(
            xmin=data["tile_xmin"],
            xmax=data["tile_xmax"],
            ymin=data["tile_ymin"],
            ymax=data["tile_ymax"],
            core_indices=_flat_to_ragged(data["tile_core_flat"], data["tile_core_offsets"]),
            extended_indices=_flat_to_ragged(
                data["tile_extended_flat"], data["tile_extended_offsets"]
            ),
            n_candidates=data["tile_n_candidates"],
            core_tile=data["tile_core_tile"],
        )
        meta = json.loads(str(data["meta_json"]))
        result = ReferenceResult(
            R=data["R"],
            sigma_R=data["sigma_R"],
            n_used=data["n_used"],
            relative_flux=data["relative_flux"],
            method=meta["method"],
            frame_kept=data["frame_kept"],
        )
        settings = settings_from_dict(json.loads(str(data["config_json"])))
    return tilemap, result, settings


def save_lightcurves_npz(
    lc_result,
    star_stats,
    best_aper_per_tile,
    bin_edges,
    star_best_aper,
    comparison_result,
    settings,
    path: Path | str,
) -> None:
    """Save light curves and all auxiliary data as uncompressed .npz.

    Parameters
    ----------
    lc_result : LightCurveResult
        Light curve result.
    star_stats : StarStats
        Star statistics.
    best_aper_per_tile : np.ndarray
        Best aperture per (tile, bin).
    bin_edges : np.ndarray
        Bin edges per tile.
    star_best_aper : np.ndarray
        Best aperture per star.
    comparison_result : ComparisonResult
        Comparison results.
    settings : Settings
        Settings used to produce the output.
    path : Path or str
        Output .npz file path.
    """
    path = Path(path)
    config_json = json.dumps(settings_to_dict(settings))

    savez_dict = {
        "lc": lc_result.lc,
        "lc_err": lc_result.lc_err,
        "lc_raw": lc_result.lc_raw,
        "epoch_ok": lc_result.epoch_ok,
        "rms": star_stats.rms,
        "chi2_reduced": star_stats.chi2_reduced,
        "expected_noise": star_stats.expected_noise,
        "n_epochs": star_stats.n_epochs,
        "best_aper_per_tile": best_aper_per_tile,
        "bin_edges": bin_edges,
        "star_best_aper": star_best_aper,
        "comparison_mask": comparison_result.mask,
        "comparison_ensemble": comparison_result.ensemble,
        "comparison_sigma_ensemble": comparison_result.sigma_ensemble,
        "comparison_sigma_star": comparison_result.sigma_star,
        "comparison_mag": comparison_result.mag,
        "comparison_n_comparison": comparison_result.n_comparison,
        "comparison_n_rounds_used": comparison_result.n_rounds_used,
        "config_json": config_json,
    }

    if lc_result.lc_err_raw is not None:
        savez_dict["lc_err_raw"] = lc_result.lc_err_raw
    if lc_result.err_scale is not None:
        savez_dict["err_scale"] = lc_result.err_scale
    if lc_result.blended is not None:
        savez_dict["blended"] = lc_result.blended

    if lc_result.decorrelation is not None:
        savez_dict["decorr_beta_star"] = lc_result.decorrelation.beta_star
        savez_dict["decorr_lc_corrected"] = lc_result.decorrelation.lc_corrected
        savez_dict["decorr_surface_coef"] = lc_result.decorrelation.surface_coef
        if lc_result.decorrelation.tile_surface_coef is not None:
            savez_dict["decorr_tile_surface_coef"] = lc_result.decorrelation.tile_surface_coef
        decorr_json = json.dumps({
            "method": lc_result.decorrelation.method,
            "centring": {k: v.tolist() for k, v in lc_result.decorrelation.centring.items()},
            "diagnostics": {
                k: v.tolist() for k, v in lc_result.decorrelation.diagnostics.items()
            },
        })
        savez_dict["decorr_json"] = decorr_json

    np.savez(path, **savez_dict)
    logger.info("wrote %s", path)


def load_lightcurves_npz(path: Path | str):
    """Load light curves and auxiliary data written by save_lightcurves_npz.

    Returns
    -------
    tuple
        (lc_result, star_stats, best_aper_per_tile, bin_edges, star_best_aper,
         comparison_result, settings, decorrelation_result or None)
    """
    from relphot.comparison import ComparisonResult
    from relphot.decorrelate import DecorrelationResult
    from relphot.lightcurve import LightCurveResult
    from relphot.stats import StarStats

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        lc_result = LightCurveResult(
            lc=data["lc"],
            lc_err=data["lc_err"],
            lc_raw=data["lc_raw"],
            epoch_ok=data["epoch_ok"],
            decorrelation=None,  # Will be set below if present
            # products written before error inflation carry none of these
            lc_err_raw=data.get("lc_err_raw"),
            err_scale=data.get("err_scale"),
            blended=data.get("blended"),
        )

        star_stats = StarStats(
            rms=data["rms"],
            chi2_reduced=data["chi2_reduced"],
            expected_noise=data["expected_noise"],
            n_epochs=data["n_epochs"],
            err_scale=lc_result.err_scale,
            blended=lc_result.blended,
        )

        best_aper_per_tile = data["best_aper_per_tile"]
        bin_edges = data["bin_edges"]
        star_best_aper = data["star_best_aper"]

        comparison_result = ComparisonResult(
            mask=data["comparison_mask"],
            ensemble=data["comparison_ensemble"],
            sigma_ensemble=data["comparison_sigma_ensemble"],
            sigma_star=data["comparison_sigma_star"],
            mag=data["comparison_mag"],
            n_comparison=data["comparison_n_comparison"],
            n_rounds_used=data["comparison_n_rounds_used"],
            method="loaded",
        )

        settings = settings_from_dict(json.loads(str(data["config_json"])))

        decorrelation = None
        if "decorr_json" in data:
            decorr_meta = json.loads(str(data["decorr_json"]))
            tile_coef = None
            if "decorr_tile_surface_coef" in data:
                tile_coef = data["decorr_tile_surface_coef"]
            decorrelation = DecorrelationResult(
                beta_star=data["decorr_beta_star"],
                lc_corrected=data["decorr_lc_corrected"],
                surface_coef=data["decorr_surface_coef"],
                tile_surface_coef=tile_coef,
                centring={k: np.array(v) for k, v in decorr_meta["centring"].items()},
                method=decorr_meta["method"],
                diagnostics={k: np.array(v) for k, v in decorr_meta["diagnostics"].items()},
            )
            lc_result.decorrelation = decorrelation

    return (
        lc_result,
        star_stats,
        best_aper_per_tile,
        bin_edges,
        star_best_aper,
        comparison_result,
        settings,
        decorrelation,
    )


def save_lightcurve_table(
    night, tilemap, lc_result, star_best_aper, settings, path: Path | str, fmt: str
) -> Path:
    """Save light curve table in long format (one row per star, frame, best aperture).

    Parameters
    ----------
    night : MatchedNight
        Matched night data.
    tilemap : TileMap
        Tile mapping.
    lc_result : LightCurveResult
        Light curve result.
    star_best_aper : np.ndarray
        Best aperture per star.
    settings : Settings
        Settings.
    path : Path or str
        Output file path (without extension).
    fmt : str
        Output format: "auto", "parquet", or "fits".

    Returns
    -------
    Path
        The written file path (with appropriate extension).
    """
    path = Path(path)

    # Determine format
    if fmt == "auto":
        try:
            import importlib.util
            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False
        actual_fmt = "parquet" if has_pyarrow else "fits"
    else:
        actual_fmt = fmt

    from astropy.table import Table

    # One row per (star with a best aperture, kept epoch); vectorised.
    best_a = np.asarray(star_best_aper, dtype=np.int64)
    ok_star = (tilemap.core_tile >= 0) & (best_a >= 0)
    si, fj = np.nonzero(ok_star[:, None] & lc_result.epoch_ok)
    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    airmass = np.array(
        [np.nan if m.airmass is None else m.airmass for m in night.frame_meta], dtype=np.float64
    )
    if settings.lightcurve.keep_all_apertures_in_table:
        n_aper = night.n_aper
        si = np.repeat(si, n_aper)
        fj = np.repeat(fj, n_aper)
        ak = np.tile(np.arange(n_aper), si.size // n_aper if n_aper else 0)
        keep = np.isfinite(lc_result.lc[si, fj, ak])
        si, fj, ak = si[keep], fj[keep], ak[keep]
    else:
        ak = best_a[si]
    columns = {
        "star_id": si.astype(np.int64),
        "tile": tilemap.core_tile[si].astype(np.int64),
        "frame": fj.astype(np.int64),
        "bjd_tdb": bjd[fj],
        "airmass": airmass[fj],
        "aperture": ak.astype(np.int64),
        "lc": lc_result.lc[si, fj, ak],
        "lc_err": lc_result.lc_err[si, fj, ak],
        "lc_raw": lc_result.lc_raw[si, fj, ak],
    }
    if lc_result.lc_err_raw is not None:
        # lc_err above is the inflated error; the formal one stays available
        columns["lc_err_raw"] = lc_result.lc_err_raw[si, fj, ak]
    if settings.lightcurve.keep_all_apertures_in_table:
        columns["is_best_aperture"] = ak == best_a[si]
    table = Table(columns)

    # Write to file
    if actual_fmt == "parquet":
        try:
            import importlib.util

            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False

        if not has_pyarrow:
            from relphot.exceptions import RelphotError

            msg = "pyarrow is required for parquet output"
            raise RelphotError(msg) from None
        out_path = path.with_suffix(".parquet")
        table.write(out_path, format="parquet", overwrite=True)
    else:
        out_path = path.with_suffix(".fits")
        table.write(out_path, format="fits", overwrite=True)

    logger.info("wrote %s (%d rows)", out_path, len(table))
    return out_path


def save_starstats_table(
    night,
    tilemap,
    comparison_result,
    star_stats,
    star_best_aper,
    _settings,
    path: Path | str,
    fmt: str,
) -> Path:
    """Save star statistics table (one row per star with core tile).

    Parameters
    ----------
    night : MatchedNight
        Matched night data.
    tilemap : TileMap
        Tile mapping.
    comparison_result : ComparisonResult
        Comparison results.
    star_stats : StarStats
        Star statistics.
    star_best_aper : np.ndarray
        Best aperture per star.
    settings : Settings
        Settings.
    path : Path or str
        Output file path (without extension).
    fmt : str
        Output format: "auto", "parquet", or "fits".

    Returns
    -------
    Path
        The written file path (with appropriate extension).
    """
    path = Path(path)

    # Determine format
    if fmt == "auto":
        try:
            import importlib.util
            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False
        actual_fmt = "parquet" if has_pyarrow else "fits"
    else:
        actual_fmt = fmt

    from astropy.table import Table

    best_a = np.asarray(star_best_aper, dtype=np.int64)
    idx = np.nonzero(tilemap.core_tile >= 0)[0]
    ba = best_a[idx]
    has = ba >= 0
    a_safe = np.where(has, ba, 0)

    def _at_best(arr, fill):
        return np.where(has, arr[idx, a_safe], fill)

    table = Table(
        {
            "star_id": idx.astype(np.int64),
            "tile": tilemap.core_tile[idx].astype(np.int64),
            "ra": night.ra[idx],
            "dec": night.dec[idx],
            "mag": _at_best(comparison_result.mag, np.nan),
            "best_aperture": ba,
            "rms": _at_best(star_stats.rms, np.nan),
            "chi2_reduced": _at_best(star_stats.chi2_reduced, np.nan),
            "expected_noise": _at_best(star_stats.expected_noise, np.nan),
            "n_epochs": _at_best(star_stats.n_epochs, 0).astype(np.int64),
            "is_comparison": _at_best(comparison_result.mask, False).astype(bool),
            # factor applied to lc_err at the best aperture (1 = none) and the neighbour-flag
            # verdict; a stats object without them (older product) reads as 1 / False
            "err_scale": (
                _at_best(star_stats.err_scale, 1.0)
                if star_stats.err_scale is not None
                else np.ones(idx.size)
            ),
            "blended": (
                np.asarray(star_stats.blended, dtype=bool)[idx]
                if star_stats.blended is not None
                else np.zeros(idx.size, dtype=bool)
            ),
            "near_edge": (
                np.asarray(star_stats.near_edge, dtype=bool)[idx]
                if star_stats.near_edge is not None
                else np.zeros(idx.size, dtype=bool)
            ),
            "tailed": (
                np.asarray(star_stats.tailed, dtype=bool)[idx]
                if star_stats.tailed is not None
                else np.zeros(idx.size, dtype=bool)
            ),
        }
    )

    # Write to file
    if actual_fmt == "parquet":
        try:
            import importlib.util

            has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        except (ImportError, AttributeError):
            has_pyarrow = False

        if not has_pyarrow:
            from relphot.exceptions import RelphotError

            msg = "pyarrow is required for parquet output"
            raise RelphotError(msg) from None
        out_path = path.with_suffix(".parquet")
        table.write(out_path, format="parquet", overwrite=True)
    else:
        out_path = path.with_suffix(".fits")
        table.write(out_path, format="fits", overwrite=True)

    logger.info("wrote %s (%d rows)", out_path, len(table))
    return out_path


def save_decorrelation_report(decorrelation, aperture_radii, path: Path | str) -> None:
    """Save decorrelation diagnostics as CSV.

    Parameters
    ----------
    decorrelation : DecorrelationResult or None
        Decorrelation result.
    aperture_radii : tuple
        Aperture radii in pixels.
    path : Path or str
        Output CSV file path.
    """
    if decorrelation is None:
        logger.info("decorrelation disabled; skipping report")
        return

    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "aperture",
            "aperture_radius_px",
            "method",
            "n_comparison_used",
            "n_fallback_tiles",
            "median_mag",
            "median_crowd",
            "median_fwhm",
            "median_airmass",
        ])

        median_crowd = (
            decorrelation.centring.get("median_crowd", np.nan)
        )
        for a in range(len(decorrelation.method)):
            row = [
                a,
                aperture_radii[a] if a < len(aperture_radii) else np.nan,
                decorrelation.method[a],
                int(decorrelation.diagnostics["n_comparison_used"][a]),
                int(decorrelation.diagnostics["n_fallback_tiles"][a]),
                decorrelation.centring["median_mag"][a],
                median_crowd,
                decorrelation.centring["median_fwhm"],
                decorrelation.centring["median_airmass"],
            ]
            writer.writerow(row)

    logger.info("wrote %s", path)
