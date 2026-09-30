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


def test_ensemble_statistic_defaults_to_median_and_stays_selectable(tmp_path) -> None:
    """New runs use the median ensemble; the weighted clipped mean is one TOML line away."""
    assert Settings().comparison.ensemble_statistic == "median"
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('[comparison]\nensemble_statistic = "weighted_clipped_mean"\n')
    assert load_settings(cfg).comparison.ensemble_statistic == "weighted_clipped_mean"


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


def test_effective_min_epochs() -> None:
    """Test SearchSettings.effective_min_epochs computes correct floor."""
    s = Settings()
    # Defaults: min_epochs=20, min_epoch_fraction=0.5
    assert s.search.effective_min_epochs(10) == 20  # max(20, ceil(0.5*10)) = 20
    assert s.search.effective_min_epochs(43) == 22  # max(20, ceil(0.5*43)) = 22
    assert s.search.effective_min_epochs(351) == 176  # max(20, ceil(0.5*351)) = 176
    # Custom: min_epochs=2, min_epoch_fraction=0.0
    s2 = Settings(search=s.search.__class__(min_epochs=2, min_epoch_fraction=0.0, **{
        f.name: getattr(s.search, f.name) for f in s.search.__dataclass_fields__.values()
        if f.name not in ('min_epochs', 'min_epoch_fraction')
    }))
    assert s2.search.effective_min_epochs(43) == 2  # max(2, 0) = 2


def test_effective_min_epochs_toml_override(tmp_path) -> None:
    """Test SearchSettings.effective_min_epochs with TOML override."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[search]\nmin_epoch_fraction = 0.7\n")
    s = load_settings(cfg)
    assert s.search.min_epoch_fraction == 0.7
    assert s.search.effective_min_epochs(100) == max(20, 70)  # max(20, ceil(0.7*100))


def test_border_defaults() -> None:
    """BorderSettings has correct defaults."""
    s = Settings()
    assert s.border.enabled is True
    assert s.border.edge_buffer_px == 30.0
    assert s.border.drift_min_common_stars == 20
    assert s.border.telescope_extra_px == {
        "T80": [0.0, 0.0, 20.0, 20.0],
        "T80S": [0.0, 0.0, 20.0, 20.0],
    }


def test_border_toml_override(tmp_path) -> None:
    """TOML [border] override works."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[border]\n"
        "enabled = false\n"
        "edge_buffer_px = 50.0\n"
        "drift_min_common_stars = 10\n"
    )
    s = load_settings(cfg)
    assert s.border.enabled is False
    assert s.border.edge_buffer_px == 50.0
    assert s.border.drift_min_common_stars == 10


def test_border_telescope_extra_px_round_trip() -> None:
    """telescope_extra_px survives TOML round-trip as list (JSON-compatible)."""
    s = Settings()
    d = settings_to_dict(s)
    # Verify lists, not tuples
    for _tel, margins in d["border"]["telescope_extra_px"].items():
        assert isinstance(margins, list)
        assert len(margins) == 4
    # Round-trip
    s2 = settings_from_dict(d)
    assert s2.border.telescope_extra_px == s.border.telescope_extra_px


def test_border_unknown_key_raises(tmp_path) -> None:
    """Unknown key under [border] raises ConfigError."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[border]\nunknown_field = true\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_tails_defaults() -> None:
    """TailSettings has correct defaults."""
    t = Settings().tails
    assert t.enabled is True
    assert t.aperture == -1
    assert t.k_sigma == 5.0
    assert t.min_low == 3
    assert t.asym_ratio == 3.0
    assert t.window == 7
    assert t.n_neighbours == 25
    assert t.pool_min_snr == 15.0
    assert t.min_snr == 5.0
    assert t.min_epochs == 20


def test_tails_toml_override_and_round_trip(tmp_path) -> None:
    """TOML [tails] override works and survives a dict round trip."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[tails]\nenabled = false\nk_sigma = 6.0\nmin_low = 4\n")
    s = load_settings(cfg)
    assert s.tails.enabled is False
    assert s.tails.k_sigma == 6.0
    assert s.tails.min_low == 4
    assert s.tails.window == 7
    assert settings_from_dict(settings_to_dict(s)).tails == s.tails


def test_tails_unknown_key_raises(tmp_path) -> None:
    """Unknown key under [tails] raises ConfigError."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[tails]\nunknown_field = true\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_db_zero_point_defaults_and_toml_override(tmp_path) -> None:
    """assumed_zp 20 and the measured T80S zero point, both overridable and round-trippable."""
    d = Settings().db
    assert d.assumed_zp == 20.0
    assert d.telescope_zp == {"T80S": 27.85}
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[db]\nassumed_zp = 21\n[db.telescope_zp]\nT80S = 27.9\nROBO43 = 22.5\n")
    s = load_settings(cfg)
    assert s.db.assumed_zp == 21
    assert s.db.telescope_zp == {"T80S": 27.9, "ROBO43": 22.5}
    assert settings_from_dict(settings_to_dict(s)).db == s.db
    assert Settings().db.telescope_zp == {"T80S": 27.85}  # the default table is not shared


@pytest.mark.parametrize(
    "body",
    [
        "assumed_zp = nan",
        "assumed_zp = inf",
        "assumed_zp = true",
        'assumed_zp = "20"',
        "telescope_zp = {T80S = nan}",
        "telescope_zp = {T80S = -inf}",
        'telescope_zp = {T80S = "27.85"}',
        "telescope_zp = {T80S = true}",
        "telescope_zp = [27.85]",
    ],
)
def test_db_zero_point_must_be_a_finite_number(tmp_path, body) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f"[db]\n{body}\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_db_coincidence_defaults_and_toml_override(tmp_path) -> None:
    """The cross-candidate check's thresholds, overridable from [db] and round-trippable."""
    d = Settings().db
    assert (d.coincidence_tc_frac, d.coincidence_tc_nsigma) == (0.1, 2.0)
    assert (d.coincidence_t14_ratio, d.coincidence_depth_ratio) == (1.5, 2.0)
    assert (d.coincidence_min_similar, d.coincidence_max_p) == (3, 1e-3)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[db]\ncoincidence_tc_frac = 0.2\ncoincidence_tc_nsigma = 3\ncoincidence_t14_ratio = 2\n"
        "coincidence_depth_ratio = 1.5\ncoincidence_min_similar = 5\ncoincidence_max_p = 1e-4\n"
    )
    s = load_settings(cfg)
    assert (s.db.coincidence_tc_frac, s.db.coincidence_tc_nsigma) == (0.2, 3)
    assert (s.db.coincidence_t14_ratio, s.db.coincidence_depth_ratio) == (2, 1.5)
    assert (s.db.coincidence_min_similar, s.db.coincidence_max_p) == (5, 1e-4)
    assert settings_from_dict(settings_to_dict(s)).db == s.db
    assert s.db.max_expected_noise == Settings().db.max_expected_noise  # the rest is untouched


@pytest.mark.parametrize(
    "body",
    [
        "coincidence_t14_ratio = 1",
        "coincidence_t14_ratio = 0.5",
        "coincidence_depth_ratio = 1.0",
        "coincidence_depth_ratio = nan",
        "coincidence_depth_ratio = true",
        "coincidence_tc_frac = 0",
        "coincidence_tc_frac = 1.5",
        "coincidence_tc_nsigma = -1",
        "coincidence_tc_nsigma = inf",
        "coincidence_max_p = 0",
        "coincidence_max_p = 2",
        'coincidence_max_p = "1e-3"',
        "coincidence_min_similar = 0",
        "coincidence_min_similar = 2.5",
        "coincidence_min_similar = true",
    ],
)
def test_db_coincidence_settings_are_validated(tmp_path, body) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f"[db]\n{body}\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)
