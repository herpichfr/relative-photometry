"""Tests for the per-frame quality cut of the reference stage (relphot.reference)."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
from conftest import make_synthetic_night

from relphot.cli import main
from relphot.config import Settings
from relphot.exceptions import ConfigError, ReferenceFrameError
from relphot.io import load_reference, save_night
from relphot.reference import (
    QUALITY_METRICS,
    assess_frame_quality,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.tiles import build_tilemap

N_FRAMES = 40
NOISY, CLOUDY, BLURRED, BRIGHT_SKY = 5, 12, 20, 30


def _night(seed: int = 11, n_stars: int = 800):
    night, _airmass, _flux0 = make_synthetic_night(n_stars=n_stars, n_frames=N_FRAMES, seed=seed)
    return night


def _with_bad_frames(night, seed: int = 3) -> None:
    """One frame each with extra noise, a flux deficit, a wide FWHM and a bright sky."""
    rng = np.random.default_rng(seed)
    night.flux[:, NOISY, :] *= 1.0 + 0.06 * rng.standard_normal((night.n_stars, 1))
    night.flux[:, CLOUDY, :] *= 0.7
    night.fwhm += rng.normal(0.0, 0.03, size=night.fwhm.shape).astype(np.float32)
    night.fwhm[:, BLURRED] *= 1.6
    night.background[:, BRIGHT_SKY] += 40.0
    night.background += rng.normal(0.0, 1.0, size=night.background.shape).astype(np.float32)


def _forced(settings: Settings | None = None) -> Settings:
    settings = settings if settings is not None else Settings()
    return replace(settings, catalog=replace(settings.catalog, photometry="forced"))


def _ref(settings: Settings, **changes) -> Settings:
    return replace(settings, reference=replace(settings.reference, **changes))


def _assess(night, settings: Settings, **kwargs):
    candidates = select_candidates(night, np.zeros(night.n_stars, dtype=bool), settings, aper=0)
    return candidates, assess_frame_quality(night, candidates, settings, aper=0, **kwargs)


def test_clean_night_flags_nothing() -> None:
    night = _night()
    _candidates, quality = _assess(night, _forced())
    assert quality is not None
    assert quality.n_ensemble > 100
    assert not quality.flagged.any()
    assert not quality.dropped.any()
    assert quality.z.shape == (N_FRAMES, len(QUALITY_METRICS))


def test_each_metric_flags_its_frame_with_a_reason() -> None:
    night = _night()
    _with_bad_frames(night)
    _candidates, quality = _assess(night, _forced())
    assert quality is not None
    assert set(np.nonzero(quality.flagged)[0]) == {NOISY, CLOUDY, BLURRED, BRIGHT_SKY}
    column = {name: k for k, name in enumerate(QUALITY_METRICS)}
    assert quality.z[NOISY, column["scatter"]] > 3.0
    assert quality.z[CLOUDY, column["transparency"]] > 3.0
    assert quality.z[BLURRED, column["fwhm"]] > 3.0
    assert quality.z[BRIGHT_SKY, column["sky"]] > 3.0
    assert quality.transparency[CLOUDY] == pytest.approx(0.7, abs=0.03)
    assert "transparency" in quality.reason[CLOUDY]
    assert "fwhm" in quality.reason[BLURRED]
    assert quality.reason[0] == ""
    assert set(np.nonzero(quality.dropped)[0]) == {NOISY, CLOUDY, BLURRED, BRIGHT_SKY}


def test_smooth_transparency_trend_is_not_a_deficit() -> None:
    night = _night()
    trend = 1.0 - 0.15 * np.linspace(-1.0, 1.0, N_FRAMES) ** 2  # a 15 % smooth airmass-like dip
    night.flux *= trend[None, :, None].astype(np.float32)
    _candidates, quality = _assess(night, _forced())
    assert quality is not None
    assert not quality.flagged.any()
    assert np.all(np.abs(quality.transparency - 1.0) < 0.04)  # the 3 % cloud wiggle stays


def test_min_excess_guard_on_a_very_uniform_night() -> None:
    night = _night()
    rng = np.random.default_rng(5)
    night.fwhm[:] = 3.0 + rng.normal(0.0, 0.002, size=night.fwhm.shape).astype(np.float32)
    night.fwhm[:, 7] += 0.09  # 3 % wider: a huge z (MAD ~ 0) but below frame_quality_min_excess
    _candidates, quality = _assess(night, _forced())
    assert quality is not None
    assert quality.z[7, 2] > 3.0
    assert not quality.flagged[7]
    strict = _ref(_forced(), frame_quality_min_excess=0.01)
    _candidates, quality = _assess(night, strict)
    assert quality is not None and quality.flagged[7]


def test_auto_applies_to_forced_photometry_only() -> None:
    night = _night()
    _with_bad_frames(night)
    standard = Settings()
    assert standard.reference.frame_quality == "auto"
    _c, q_standard = _assess(night, standard)
    _c, q_forced = _assess(night, standard, photometry="forced")
    assert q_standard is not None and q_forced is not None
    assert q_standard.flagged.sum() == 4 and not q_standard.applied and not q_standard.dropped.any()
    assert q_forced.applied and q_forced.dropped.sum() == 4
    # the metrics do not depend on whether the cut applies
    np.testing.assert_allclose(q_standard.z, q_forced.z)

    _c, q_on = _assess(night, _ref(standard, frame_quality="on"))
    _c, q_off = _assess(night, _ref(_forced(), frame_quality="off"))
    assert q_on is not None and q_off is not None
    assert q_on.dropped.sum() == 4
    assert q_off.flagged.sum() == 4 and not q_off.dropped.any()


def test_selection_drops_the_flagged_frames_from_frame_kept() -> None:
    night = _night()
    _with_bad_frames(night)
    settings = _forced()
    candidates = select_candidates(night, np.zeros(night.n_stars, dtype=bool), settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    selection = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    assert set(selection.dropped_frames) == {NOISY, CLOUDY, BLURRED, BRIGHT_SKY}
    assert selection.frame_kept.sum() == N_FRAMES - 4
    assert selection.quality is not None and selection.quality.applied
    # with the standard photometry the same night keeps every frame
    standard = Settings()
    selection = select_reference_frames_and_stars(night, tilemap, candidates, standard, aper=0)
    assert selection.frame_kept.all()
    # an explicit photometry argument overrides the settings
    selection = select_reference_frames_and_stars(
        night, tilemap, candidates, standard, aper=0, photometry="forced"
    )
    assert selection.frame_kept.sum() == N_FRAMES - 4


def test_cuts_are_capped_worst_first() -> None:
    night = _night()
    bad = [3, 9, 15, 21, 27, 33]
    for rank, j in enumerate(bad):  # increasingly bad flux deficits
        night.flux[:, j, :] *= 0.8 - 0.1 * rank
    settings = _ref(_forced(), frame_quality_max_fraction=0.1)
    _c, quality = _assess(night, settings)
    assert quality is not None and quality.flagged.sum() == len(bad)
    # frame_quality_max_fraction 0.1 of 40 frames -> at most 4, the worst four
    assert quality.dropped.sum() == 4
    assert set(np.nonzero(quality.dropped)[0]) == {33, 27, 21, 15}
    # max_dropped_frame_fraction bounds the cut as well
    tight = _ref(settings, max_dropped_frame_fraction=0.05)
    _c, quality = _assess(night, tight)
    assert quality is not None and quality.dropped.sum() == 2
    assert set(np.nonzero(quality.dropped)[0]) == {33, 27}


def test_quality_drops_count_against_the_star_set_drop_cap() -> None:
    night = _night()
    _with_bad_frames(night)
    settings = _ref(_forced(), max_dropped_frame_fraction=0.1)
    candidates = select_candidates(night, np.zeros(night.n_stars, dtype=bool), settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    # four quality drops use the whole budget (0.1 * 40); a frame that lacks reference stars cannot
    # be dropped on top of them
    night.flux[5:, 2, :] = np.nan
    with pytest.raises(ReferenceFrameError):
        select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)


def test_unknown_times_fall_back_to_the_median_transparency() -> None:
    night = _night()
    _with_bad_frames(night)
    for meta in night.frame_meta:
        meta.bjd_tdb = float("nan")  # as when the site keywords are missing
    _candidates, quality = _assess(night, _forced())
    assert quality is not None
    assert quality.flagged[CLOUDY] and quality.flagged[NOISY]
    assert quality.transparency[CLOUDY] == pytest.approx(0.7, abs=0.05)


def test_short_night_is_not_cut() -> None:
    night, _a, _f = make_synthetic_night(n_stars=800, n_frames=8, seed=2)
    night.flux[:, 3, :] *= 0.5
    _c, quality = _assess(night, _forced())
    assert quality is not None and not quality.dropped.any()


def test_no_ensemble_returns_none() -> None:
    night = _night(n_stars=60)
    night.flags[:] = 1  # no star has FLAGS == 0
    settings = _forced()
    candidates = np.ones(night.n_stars, dtype=bool)
    assert assess_frame_quality(night, candidates, settings, aper=0) is None


def test_settings_validation() -> None:
    with pytest.raises(ConfigError):
        replace(Settings().reference, frame_quality="sometimes")
    with pytest.raises(ConfigError):
        replace(Settings().reference, frame_quality_max_fraction=1.5)
    with pytest.raises(ConfigError):
        replace(Settings().reference, frame_quality_sigma=0.0)


def test_reference_cli_uses_the_ingest_photometry_and_writes_frames_csv(tmp_path) -> None:
    night = _night()
    _with_bad_frames(night)
    results = {}
    for mode in ("standard", "forced"):
        settings = Settings()
        settings = replace(settings, catalog=replace(settings.catalog, photometry=mode))
        night_npz = tmp_path / f"night_{mode}.npz"
        save_night(night, settings, night_npz)
        ref_npz = tmp_path / f"ref_{mode}.npz"
        assert main(["reference", str(night_npz), "--out", str(ref_npz), "--no-variables",
                     "--aper", "0"]) == 0
        _tilemap, result, _s = load_reference(ref_npz)
        results[mode] = result.frame_kept
        with (tmp_path / f"ref_{mode}_frames.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == N_FRAMES
        assert {"scatter", "transparency", "fwhm", "sky", "z_scatter", "flagged",
                "quality_dropped", "reason", "kept"} <= set(rows[0])
        assert sum(int(r["flagged"]) for r in rows) == 4
        assert [int(r["kept"]) for r in rows] == [int(k) for k in result.frame_kept]
    assert results["standard"].all()
    assert results["forced"].sum() == N_FRAMES - 4
    assert not results["forced"][[NOISY, CLOUDY, BLURRED, BRIGHT_SKY]].any()
