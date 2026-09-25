"""Tests for relphot.io: exact .npz round trip of a matched night."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.ingest import read_catalogs
from relphot.io import load_night, save_night
from relphot.match import match_night


def test_round_trip_is_exact(fits_files, tmp_path) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    out = tmp_path / "night.npz"
    save_night(night, settings, out)
    night2, settings2 = load_night(out)

    assert settings2 == settings
    assert night2.master_frame_index == night.master_frame_index
    assert night2.n_stars_before_cut == night.n_stars_before_cut
    assert night2.n_stars_after_cut == night.n_stars_after_cut
    assert night2.reports == night.reports
    assert [m.to_dict() for m in night2.frame_meta] == [m.to_dict() for m in night.frame_meta]

    attrs = (
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
    for attr in attrs:
        a = getattr(night, attr)
        b = getattr(night2, attr)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b, strict=True)
