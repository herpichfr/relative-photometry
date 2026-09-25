"""Tests for relphot.config: defaults, TOML overrides, unknown-key errors."""

from __future__ import annotations

import pytest

from relphot.config import Settings, load_settings, settings_from_dict, settings_to_dict
from relphot.exceptions import ConfigError


def test_defaults() -> None:
    s = Settings()
    assert s.catalog.hdu_name == "CATALOG"
    assert s.catalog.match_radius_arcsec == 1.0
    assert s.catalog.min_presence == 0.8
    assert s.catalog.saturation_flag_bit == 4
    assert s.catalog.nonlinear_flag_bit == 64
    assert s.catalog.columns.flux_prefix == "FLUX_APER_"


def test_load_settings_no_path_returns_defaults() -> None:
    assert load_settings(None) == Settings()


def test_load_settings_overrides_only_named_fields(tmp_path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('[catalog]\nmatch_radius_arcsec = 2.5\n[catalog.columns]\nra = "ALPHA_J2000"\n')
    s = load_settings(cfg)
    assert s.catalog.match_radius_arcsec == 2.5
    assert s.catalog.columns.ra == "ALPHA_J2000"
    # Everything else keeps its default.
    assert s.catalog.min_presence == 0.8
    assert s.catalog.columns.dec == "DEC"


def test_load_settings_unknown_top_level_key_raises(tmp_path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[bogus]\nx = 1\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_load_settings_unknown_nested_key_raises(tmp_path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[catalog]\nnot_a_real_field = 1\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_settings_dict_round_trip() -> None:
    s = Settings()
    assert settings_from_dict(settings_to_dict(s)) == s


def test_old_reference_setting_raises(tmp_path) -> None:
    """Old min_used_per_frame setting should raise ConfigError."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[reference]\nmin_used_per_frame = 5\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_reference_new_settings_round_trip() -> None:
    """New reference settings should round-trip through dict conversion."""
    s = Settings()
    d = settings_to_dict(s)
    # Check new settings are present
    assert "min_ref_stars" in d["reference"]
    assert "max_dropped_frame_fraction" in d["reference"]
    assert "star_outlier_sigma" in d["reference"]
    assert "star_reject_iter" in d["reference"]
    # Round-trip should succeed
    s2 = settings_from_dict(d)
    assert s2.reference.min_ref_stars == s.reference.min_ref_stars
    assert s2.reference.max_dropped_frame_fraction == s.reference.max_dropped_frame_fraction
    assert s2.reference.star_outlier_sigma == s.reference.star_outlier_sigma
    assert s2.reference.star_reject_iter == s.reference.star_reject_iter


def test_comparison_toml_override(tmp_path) -> None:
    """TOML override of [comparison] k_floor works."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[comparison]\nk_floor = 3.0\n")
    s = load_settings(cfg)
    assert s.comparison.k_floor == 3.0
    # Other fields should keep defaults
    assert s.comparison.min_snr == 5.0
    assert s.comparison.n_rounds == 3


def test_lightcurve_unknown_key_raises(tmp_path) -> None:
    """Unknown key under [lightcurve] raises ConfigError."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[lightcurve]\nunknown_field = true\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)
