"""Tests for relphot.match: master selection, one-to-one matching, presence cut."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.ingest import read_catalogs
from relphot.match import match_night, select_master


def test_select_master_is_largest_good_seeing_frame(fits_files) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    idx = select_master(cats)
    fwhms = np.array([c.meta.median_fwhm for c in cats])
    threshold = 1.2 * np.median(fwhms)
    eligible = [i for i, f in enumerate(fwhms) if f <= threshold]
    assert idx == max(eligible, key=lambda i: cats[i].n_sources)


def test_match_is_one_to_one_per_frame(fits_files) -> None:
    """Every source lands in exactly one master-star row: a duplicate claim or
    a dropped source would break this global accounting identity.

    The master frame seeds the initial master list (its n_sources stars, all
    counted as "matched to master" by convention -- see match_night). Every
    other frame either matches an existing star or creates exactly one new
    one per unmatched source, so summing (n_sources - n_matched_to_master)
    over those frames must equal the total star count before the presence
    cut, minus the master frame's own contribution. This can only hold if no
    source was matched twice, matched and also counted as new, or lost.
    """
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    master_cat = cats[night.master_frame_index]
    n_new_total = sum(
        report.n_sources - report.n_matched_to_master
        for i, report in enumerate(night.reports)
        if i != night.master_frame_index
    )
    assert master_cat.n_sources + n_new_total == night.n_stars_before_cut


def test_presence_cut_reduces_or_keeps_star_count(fits_files) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)
    assert night.n_stars_after_cut <= night.n_stars_before_cut
    assert night.n_stars == night.n_stars_after_cut
    assert np.all(night.presence[night.presence >= 0] >= settings.catalog.min_presence)


def test_match_reports_are_plausible(fits_files) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)
    assert len(night.reports) == len(cats)
    for report in night.reports:
        assert 0.0 <= report.match_fraction <= 1.0
        assert report.n_matched_to_master <= report.n_sources


def test_frame_x_y_are_finite_where_present(fits_files) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)
    # frame_x/frame_y must be finite exactly where flags != -1, NaN elsewhere.
    present = night.flags != -1
    assert np.all(np.isfinite(night.frame_x[present]))
    assert np.all(np.isfinite(night.frame_y[present]))
    missing = night.flags == -1
    assert np.all(np.isnan(night.frame_x[missing]))
    assert np.all(np.isnan(night.frame_y[missing]))
