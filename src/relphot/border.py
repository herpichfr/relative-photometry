"""Night-level border-eligibility cut for reference and comparison stars.

Comparison and reference stars must never come from near the detector border;
margins are computed per night from the measured per-night drift and telescope-
specific extra margins.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from relphot.config import BorderSettings
    from relphot.match import MatchedNight

logger = logging.getLogger(__name__)

__all__ = ["BorderInfo", "border_eligibility", "detector_info", "measure_frame_offsets"]


@dataclass(frozen=True, slots=True)
class BorderInfo:
    """Night-level border eligibility information.

    Attributes
    ----------
    eligible : np.ndarray
        (n_stars,) bool array, True if star is eligible (not too close to border)
    detector : tuple | None
        (nx, ny, telescope) if known, else None
    telescope : str
        Telescope name from detector info or empty string
    dx_range : tuple
        (dx_min, dx_max) measured frame drift range in pixels
    dy_range : tuple
        (dy_min, dy_max) measured frame drift range in pixels
    margins : tuple
        (left, right, bottom, top) total margins in pixels (buffer + drift + telescope extra)
    enabled : bool
        Whether the border cut was enabled; False if detector unknown or disabled
    """

    eligible: np.ndarray
    detector: tuple | None
    telescope: str
    dx_range: tuple
    dy_range: tuple
    margins: tuple
    enabled: bool


def detector_info(night: MatchedNight) -> tuple[float, float, str] | None:
    """Detector size and telescope name from the master frame.

    Returns (nx, ny, telescope) if available, else None. Tries the frame_meta
    first, falls back to reading the FITS header directly (for old night.npz
    without these fields), then tries to extract telescope from the file path
    (directory name ending in '_reduced').

    Parameters
    ----------
    night : MatchedNight
        The night to interrogate

    Returns
    -------
    tuple[float, float, str] | None
        (nx, ny, telescope) or None if detector size is unknown
    """
    from astropy.io import fits

    m = night.frame_meta[night.master_frame_index]
    nx, ny, tel = getattr(m, "naxis1", 0), getattr(m, "naxis2", 0), getattr(m, "telescope", "")
    if nx > 0 and ny > 0 and tel:
        return float(nx), float(ny), tel

    try:
        h = fits.getheader(m.file, 0)
        nx = float(h["NAXIS1"])
        ny = float(h["NAXIS2"])
        tel = str(h.get("TELESCOP", "")).strip()
        if nx > 0 and ny > 0:
            return nx, ny, tel
    except (OSError, KeyError, TypeError, ValueError):
        pass

    # Fallback: try to extract telescope from path
    try:
        path_str = str(m.file)
        if "_reduced" in path_str:
            parts = path_str.rsplit("_reduced", 1)[0].rsplit("/", 1)[-1]
            if parts:
                tel = parts
    except (OSError, ValueError, TypeError, IndexError, AttributeError):
        pass

    return None


def measure_frame_offsets(
    night: MatchedNight, min_common: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Measure per-frame offsets (dx, dy) from master frame positions.

    For each frame, computes the median offset (frame_x - x, frame_y - y) over
    stars that are finite in both coordinates and appear in at least min_common
    stars. Returns NaN for frames with insufficient common stars.

    Parameters
    ----------
    night : MatchedNight
        The night
    min_common : int, optional
        Minimum number of stars with finite positions in both frame and master
        (default 20)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dx, dy) arrays of shape (n_frames,), with NaN for sparse frames
    """
    n_f = night.n_frames
    dx = np.full(n_f, np.nan)
    dy = np.full(n_f, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f in range(n_f):
            ddx = night.frame_x[:, f] - night.x
            ddy = night.frame_y[:, f] - night.y
            ok = np.isfinite(ddx) & np.isfinite(ddy)
            if np.count_nonzero(ok) >= min_common:
                dx[f] = np.median(ddx[ok])
                dy[f] = np.median(ddy[ok])
    return dx, dy


def border_eligibility(night: MatchedNight, settings: BorderSettings) -> BorderInfo:
    """Compute night-level border eligibility for reference and comparison stars.

    Disabled or detector-unknown nights return all True with enabled=False.

    Parameters
    ----------
    night : MatchedNight
        The night
    settings : BorderSettings
        Border settings (enabled, edge_buffer_px, drift_min_common_stars,
        telescope_extra_px)

    Returns
    -------
    BorderInfo
        Border eligibility info, with enabled=False if disabled/unknown

    Raises
    ------
    ConfigError
        If a telescope_extra_px entry is not a 4-element list
    """
    from relphot.exceptions import ConfigError

    n = night.n_stars
    all_true = np.ones(n, dtype=bool)

    # Check if disabled
    if not settings.enabled:
        logger.info("border cut disabled")
        return BorderInfo(
            eligible=all_true,
            detector=None,
            telescope="",
            dx_range=(np.nan, np.nan),
            dy_range=(np.nan, np.nan),
            margins=(0, 0, 0, 0),
            enabled=False,
        )

    # Get detector info
    info = detector_info(night)
    if info is None:
        logger.warning("border cut skipped: detector size unknown")
        return BorderInfo(
            eligible=all_true,
            detector=None,
            telescope="",
            dx_range=(np.nan, np.nan),
            dy_range=(np.nan, np.nan),
            margins=(0, 0, 0, 0),
            enabled=False,
        )

    nx, ny, tel = info

    # Measure frame offsets
    dx, dy = measure_frame_offsets(night, min_common=settings.drift_min_common_stars)

    # If no finite offsets, disable the cut
    if not np.isfinite(dx).any():
        logger.warning("border cut skipped: no finite frame offsets")
        return BorderInfo(
            eligible=all_true,
            detector=(nx, ny, tel),
            telescope=tel,
            dx_range=(np.nan, np.nan),
            dy_range=(np.nan, np.nan),
            margins=(0, 0, 0, 0),
            enabled=False,
        )

    # Get telescope extra margins
    try:
        tel_upper = tel.upper()
        extra = settings.telescope_extra_px.get(tel_upper, [0.0, 0.0, 0.0, 0.0])
        if len(extra) != 4:
            raise ConfigError(
                f"border.telescope_extra_px['{tel_upper}'] must be a 4-element list"
            )
        extra = list(extra)
    except AttributeError:
        extra = [0.0, 0.0, 0.0, 0.0]

    # Compute total margins: buffer + telescope extra
    left = settings.edge_buffer_px + extra[0]
    right = settings.edge_buffer_px + extra[1]
    bottom = settings.edge_buffer_px + extra[2]
    top = settings.edge_buffer_px + extra[3]

    # Compute envelope of all star positions across all frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xlo = np.fmin(np.nanmin(night.frame_x, axis=1), night.x + np.nanmin(dx))
        xhi = np.fmax(np.nanmax(night.frame_x, axis=1), night.x + np.nanmax(dx))
        ylo = np.fmin(np.nanmin(night.frame_y, axis=1), night.y + np.nanmin(dy))
        yhi = np.fmax(np.nanmax(night.frame_y, axis=1), night.y + np.nanmax(dy))

    # Check eligibility
    elig = (
        np.isfinite(xlo)
        & np.isfinite(ylo)
        & (xlo >= 1 + left)
        & (xhi <= nx - right)
        & (ylo >= 1 + bottom)
        & (yhi <= ny - top)
    )

    # Log the result
    dx_min = np.nanmin(dx)
    dx_max = np.nanmax(dx)
    dy_min = np.nanmin(dy)
    dy_max = np.nanmax(dy)
    dx_range_val = dx_max - dx_min
    dy_range_val = dy_max - dy_min
    n_ineligible = np.count_nonzero(~elig)
    ineligible_pct = 100.0 * n_ineligible / n if n > 0 else 0.0
    logger.info(
        "border: %s %.0fx%.0f; measured drift dx [%.1f,+%.1f] (range %.1f) "
        "dy [%.1f,+%.1f] (%.1f) px; margins L/R/B/T = %.0f/%.0f/%.0f/%.0f + drift; "
        "ineligible %d/%d (%.1f%%)",
        tel,
        nx,
        ny,
        dx_min,
        dx_max,
        dx_range_val,
        dy_min,
        dy_max,
        dy_range_val,
        left,
        right,
        bottom,
        top,
        n_ineligible,
        n,
        ineligible_pct,
    )

    return BorderInfo(
        eligible=elig,
        detector=(nx, ny, tel),
        telescope=tel,
        dx_range=(dx_min, dx_max),
        dy_range=(dy_min, dy_max),
        margins=(left, right, bottom, top),
        enabled=True,
    )
