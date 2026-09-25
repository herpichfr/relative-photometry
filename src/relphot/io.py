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

__all__ = ["load_night", "load_reference", "save_night", "save_reference"]

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
    meta_json = json.dumps({"method": result.method})

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
        )
        settings = settings_from_dict(json.loads(str(data["config_json"])))
    return tilemap, result, settings
