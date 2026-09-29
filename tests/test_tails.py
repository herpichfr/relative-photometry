"""Tests for the night-level tail cut (relphot.tails) and its combination with the border cut."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from relphot.comparison import select_comparison_pool, select_comparison_stars
from relphot.config import Settings, TailSettings
from relphot.eligibility import star_eligibility
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.tails import running_median, tail_eligibility
from relphot.tiles import build_tilemap
from tests.conftest import make_synthetic_night


def _night(n_stars: int = 2000, n_frames: int = 40, seed: int = 3, n_aper: int = 2):
    night, _, _ = make_synthetic_night(
        n_stars=n_stars, n_frames=n_frames, seed=seed, n_aper=n_aper
    )
    return night


def _bright(night, n: int = 6) -> np.ndarray:
    return np.argsort(-np.median(night.snr, axis=1))[:n]


def _scaled(night, star: int, epochs, factor: float):
    flux = night.flux.copy()
    flux[star, epochs, :] *= factor
    return replace(night, flux=flux)


def test_running_median_spike_removed_and_step_kept() -> None:
    a = np.zeros((2, 30))
    a[0, 10] = 5.0  # one-epoch spike
    a[1, 15:] = 1.0  # step
    out = running_median(a, 7)
    assert out.shape == a.shape
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(out[1], a[1])


def test_running_median_nan_handling() -> None:
    a = np.ones((1, 12))
    a[0, 3:9] = np.nan  # too few finite values inside the window
    out = running_median(a, 5)
    assert np.isnan(out[0, 5])
    assert out[0, 0] == 1.0


def test_clean_synthetic_night_has_no_tailed_stars() -> None:
    night = _night()
    info = tail_eligibility(night, TailSettings())
    assert info.enabled
    assert info.aperture == 1
    assert not info.tailed.any()
    assert info.eligible.all()
    assert info.tailed.shape == (night.n_stars,)
    assert info.n_low.max() <= 1 and info.n_high.max() <= 1


def test_isolated_dips_flag_star_but_spikes_and_single_dip_do_not() -> None:
    night = _night()
    a, b, c, *_ = _bright(night)
    epochs = [5, 12, 20, 27, 33]
    night = _scaled(night, a, epochs, 0.90)  # five 10 % dips
    night = _scaled(night, b, epochs, 1.10)  # five 10 % upward spikes
    night = _scaled(night, c, [15], 0.90)  # a single dip
    info = tail_eligibility(night, TailSettings())
    assert info.tailed[a] and info.n_low[a] == 5 and info.n_high[a] == 0
    assert not info.tailed[b] and info.n_high[b] == 5
    assert not info.tailed[c] and info.n_low[c] == 1
    assert info.tailed.sum() == 1
    np.testing.assert_array_equal(info.eligible, ~info.tailed)


def test_transit_is_not_a_tail() -> None:
    night = _night()
    d, e, *_ = _bright(night)
    clean = tail_eligibility(night, TailSettings())
    night = _scaled(night, d, slice(10, 25), 0.98)  # 15-epoch 2 % transit
    night = _scaled(night, e, slice(10, 25), 0.90)  # 15-epoch 10 % eclipse
    info = tail_eligibility(night, TailSettings())
    assert not info.tailed[d] and not info.tailed[e]
    assert info.n_low[d] == clean.n_low[d]
    assert info.n_low[e] == clean.n_low[e]


def test_cloudy_frames_are_not_counted() -> None:
    night = _night()
    rng = np.random.default_rng(0)
    flux = night.flux.astype(float)
    flux[:, 30:33, :] *= 0.4 * (1.0 + 0.05 * rng.standard_normal((night.n_stars, 3, 1)))
    info = tail_eligibility(replace(night, flux=flux.astype(np.float32)), TailSettings())
    assert not info.tailed.any()
    assert info.frame_noise[30:33].min() > 1.5
    assert info.frame_noise[:30].max() == 1.0


def test_flagged_epochs_are_ignored() -> None:
    night = _night()
    a = _bright(night)[0]
    night = _scaled(night, a, [5, 12, 20, 27, 33], 0.90)
    flags = night.flags.copy()
    flags[a, [5, 12, 20, 27, 33]] = 1
    info = tail_eligibility(replace(night, flags=flags), TailSettings())
    assert not info.tailed[a]
    assert info.n_valid[a] == night.n_frames - 5


def test_threshold_settings_are_honoured() -> None:
    night = _night()
    a = _bright(night)[0]
    night = _scaled(night, a, [5, 12, 20, 27, 33], 0.90)
    assert tail_eligibility(night, TailSettings(min_low=6)).tailed[a] == False  # noqa: E712
    assert tail_eligibility(night, TailSettings(k_sigma=1000.0)).tailed[a] == False  # noqa: E712
    assert tail_eligibility(night, TailSettings(min_low=5)).tailed[a] == True  # noqa: E712


def test_aperture_setting_selects_the_aperture() -> None:
    night = _night()
    a = _bright(night)[0]
    flux = night.flux.copy()
    flux[a, [5, 12, 20, 27, 33], 0] *= 0.90  # dips in aperture 0 only
    night = replace(night, flux=flux)
    assert not tail_eligibility(night, TailSettings()).tailed[a]  # default = aperture 1
    assert tail_eligibility(night, TailSettings(aperture=0)).tailed[a]
    one = _night(n_aper=1)
    assert tail_eligibility(one, TailSettings()).aperture == 0


def test_disabled_and_too_few_neighbours_return_all_eligible(caplog) -> None:
    night = _night()
    off = tail_eligibility(night, TailSettings(enabled=False))
    assert not off.enabled and off.eligible.all() and not off.tailed.any()
    small = _night(n_stars=20)
    with caplog.at_level("WARNING"):
        info = tail_eligibility(small, TailSettings())
    assert not info.enabled and info.eligible.all()
    assert any("tail cut skipped" in r.message for r in caplog.records)


def test_star_eligibility_combines_border_and_tails() -> None:
    night = _night()
    a = _bright(night)[0]
    night = _scaled(night, a, [5, 12, 20, 27, 33], 0.90)
    info = star_eligibility(night, Settings())
    assert not info.tails.eligible[a]
    assert not info.eligible[a]
    np.testing.assert_array_equal(info.eligible, info.border.eligible & info.tails.eligible)
    off = star_eligibility(night, replace(Settings(), tails=TailSettings(enabled=False)))
    assert off.eligible[a] == off.border.eligible[a]


def test_default_paths_exclude_tailed_star_from_reference_and_comparison() -> None:
    night = _night()
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    off = replace(settings, tails=TailSettings(enabled=False))
    cand0 = select_candidates(night, variable_mask, off, aper=1)
    pool0 = select_comparison_pool(night, variable_mask, off, aper=1)
    a = int(np.nonzero(cand0 & pool0)[0][np.argmax(np.median(night.snr[cand0 & pool0], axis=1))])
    night = _scaled(night, a, [5, 12, 20, 27, 33], 0.90)
    assert select_candidates(night, variable_mask, off, aper=1)[a]
    assert not select_candidates(night, variable_mask, settings, aper=1)[a]
    assert not select_comparison_pool(night, variable_mask, settings, aper=1)[a]
    # an explicit star_eligible replaces the default computation
    explicit = np.ones(night.n_stars, dtype=bool)
    assert select_candidates(night, variable_mask, settings, aper=1, star_eligible=explicit)[a]
    assert select_comparison_pool(night, variable_mask, settings, aper=1, star_eligible=explicit)[a]


def test_tailed_star_is_in_no_tile_reference_and_no_comparison_mask() -> None:
    night = _night()
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    off = replace(settings, tails=TailSettings(enabled=False))
    cand0 = select_candidates(night, variable_mask, off, aper=1)
    a = int(np.nonzero(cand0)[0][np.argmax(np.median(night.snr[cand0], axis=1))])
    night = _scaled(night, a, [5, 12, 20, 27, 33], 0.90)
    eligibility = star_eligibility(night, settings)
    assert not eligibility.eligible[a]
    candidates = select_candidates(
        night, variable_mask, settings, 1, star_eligible=eligibility.eligible
    )
    tilemap = build_tilemap(night, candidates, settings)
    selection = select_reference_frames_and_stars(night, tilemap, candidates, settings, 1)
    assert all(a not in stars for stars in selection.tile_stars)
    reference = build_references(night, tilemap, selection, settings)
    comparison = select_comparison_stars(
        night, tilemap, reference, variable_mask, settings, star_eligible=eligibility.eligible
    )
    assert not comparison.mask[a].any()
