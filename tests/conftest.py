"""Shared pytest fixtures and synthetic-data helpers.

``fits_files``/``csv_files`` point at the small real T80S frames under
``tests/data`` (see ``generate_fixtures.py``). :func:`make_synthetic_night`
builds a fully synthetic :class:`~relphot.match.MatchedNight` -- a uniform
star field over a chosen pixel area with an injected airmass/cloud
transparency curve and photon-like noise -- used by ``test_tiles.py`` and
``test_reference.py`` where a real night is too small (232 stars, 4 frames)
to exercise tiling and reference recovery meaningfully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from relphot.ingest import FrameMeta
from relphot.match import MatchedNight, MatchReport

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def fits_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*_proc.fits"))


@pytest.fixture
def csv_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*_proc_catalog.csv"))


def make_synthetic_night(
    n_stars: int = 2000,
    n_frames: int = 40,
    size_px: float = 3000.0,
    k_extinction: float = 0.17,
    seed: int = 0,
    n_aper: int = 1,
    ra0_deg: float = 150.0,
    dec0_deg: float = -30.0,
    pixel_scale_arcsec: float = 0.55,
) -> tuple[MatchedNight, np.ndarray, np.ndarray]:
    """A synthetic night with a known injected transparency curve.

    Star positions are uniform over ``[0, size_px) x [0, size_px)``; RA/Dec
    come from a simple flat-sky tangent projection (accurate enough for the
    isolation cross-match, never used as real astrometry). Each star has a
    fixed intrinsic flux ``flux0_i`` (magnitudes uniform in ``[12, 20]``,
    about a 1600x dynamic range); the observed flux is
    ``f_ij = flux0_i * T_j + noise_ij``, with ``T_j = 10**(-0.4 *
    k_extinction * airmass_j) * cloud_j`` (``cloud_j`` a smooth few-percent
    multiplicative wiggle) and ``noise_ij`` Gaussian with the Poisson-like
    sigma ``sqrt(max(f_true, 1))``. Every star is present in every frame,
    with ``FLAGS = 0``.

    Returns ``(night, airmass, flux0)`` -- ``airmass``/``flux0`` are the
    injected per-frame airmass and per-star intrinsic flux, returned
    alongside ``night`` so a test can check the reference recovers them.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, size_px, n_stars)
    y = rng.uniform(0.0, size_px, n_stars)

    scale_deg = pixel_scale_arcsec / 3600.0
    dec = dec0_deg + (y - size_px / 2.0) * scale_deg
    ra = ra0_deg + (x - size_px / 2.0) * scale_deg / np.cos(np.radians(dec0_deg))

    mag = rng.uniform(12.0, 20.0, n_stars)
    flux0 = 10.0 ** (-0.4 * (mag - 25.0))

    phase = np.linspace(-1.0, 1.0, n_frames)
    airmass = np.clip(1.0 + 0.6 * phase**2, 1.0, None)
    bjd_tdb = 2460000.0 + np.linspace(0.0, 0.3, n_frames)
    cloud = 1.0 + 0.03 * np.sin(2.0 * np.pi * np.arange(n_frames) / 13.0)
    transparency = 10.0 ** (-0.4 * k_extinction * airmass) * cloud

    true_flux = flux0[:, None] * transparency[None, :]
    sigma = np.sqrt(np.maximum(true_flux, 1.0))
    flux = np.maximum(true_flux + rng.standard_normal((n_stars, n_frames)) * sigma, 1.0)
    snr = flux / sigma

    flux3 = np.repeat(flux[:, :, None], n_aper, axis=2).astype(np.float32)
    sigma3 = np.repeat(sigma[:, :, None], n_aper, axis=2).astype(np.float32)

    frame_meta = [
        FrameMeta(
            file=Path(f"synthetic_{j:03d}.fits"),
            date_obs="2026-01-01T00:00:00",
            exptime=90.0,
            jd_utc=bjd_tdb[j],
            bjd_tdb=float(bjd_tdb[j]),
            airmass=float(airmass[j]),
            filter="R",
            object="synthetic",
            median_fwhm=3.0,
            n_sources=n_stars,
            aperture_radii_px=tuple(float(i + 2) for i in range(n_aper)),
            wcs=None,
        )
        for j in range(n_frames)
    ]
    reports = [
        MatchReport(
            file=str(frame_meta[j].file),
            n_sources=n_stars,
            n_matched_to_master=n_stars,
            match_fraction=1.0,
            median_sep_arcsec=0.0,
        )
        for j in range(n_frames)
    ]

    night = MatchedNight(
        ra=ra,
        dec=dec,
        x=x,
        y=y,
        frame_x=np.repeat(x[:, None], n_frames, axis=1),
        frame_y=np.repeat(y[:, None], n_frames, axis=1),
        flux=flux3,
        fluxerr=sigma3,
        fwhm=np.full((n_stars, n_frames), 3.0, dtype=np.float32),
        snr=snr.astype(np.float32),
        background=np.zeros((n_stars, n_frames), dtype=np.float32),
        flags=np.zeros((n_stars, n_frames), dtype=np.int32),
        presence=np.ones(n_stars),
        frame_meta=frame_meta,
        reports=reports,
        master_frame_index=0,
        n_stars_before_cut=n_stars,
        n_stars_after_cut=n_stars,
    )
    return night, airmass, flux0
