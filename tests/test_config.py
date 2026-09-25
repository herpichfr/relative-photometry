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
