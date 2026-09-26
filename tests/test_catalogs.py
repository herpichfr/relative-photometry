"""Tests for relphot.catalogs: cross-match against known variables, exoplanets, and Gaia neighbours.

Every test injects a fake fetcher; none touches the network or requires astroquery.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from relphot.catalogs import (
    compute_neighbour_dilution,
    is_disqualifying_variable_type,
    match_known_planets,
    match_known_variables,
)
from relphot.config import Settings
from relphot.match import MatchedNight


def _minimal_night(ra: np.ndarray, dec: np.ndarray) -> MatchedNight:
    n = ra.shape[0]
    return MatchedNight(
        ra=ra,
        dec=dec,
        x=np.zeros(n),
        y=np.zeros(n),
        frame_x=np.zeros((n, 1)),
        frame_y=np.zeros((n, 1)),
        flux=np.zeros((n, 1, 1)),
        fluxerr=np.zeros((n, 1, 1)),
        fwhm=np.zeros((n, 1), dtype=np.float32),
        snr=np.zeros((n, 1), dtype=np.float32),
        background=np.zeros((n, 1), dtype=np.float32),
        flags=np.zeros((n, 1), dtype=np.int32),
        presence=np.ones(n),
        frame_meta=[],
        reports=[],
        master_frame_index=0,
        n_stars_before_cut=n,
        n_stars_after_cut=n,
    )


def test_match_known_variables_keeps_name_type_period() -> None:
    ra = np.array([10.0, 10.00003, 10.5])
    dec = np.array([-30.0, -30.00001, -30.2])
    night = _minimal_night(ra, dec)
    settings = Settings()

    def fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        return Table({
            "ra_deg": [10.00003], "dec_deg": [-30.00001],
            "name": ["ASAS J1234"], "var_type": ["EA|ESD"], "period_days": [1.23],
        })

    result = match_known_variables(night, settings, fetcher=fetcher)
    assert result.matched.tolist() == [True, True, False]
    assert result.name[0] == "ASAS J1234"
    assert result.var_type[0] == "EA|ESD"
    assert result.period_days[0] == 1.23
    assert not result.matched[2]


def test_match_known_variables_merges_all_catalogue_entries() -> None:
    night = _minimal_night(np.array([10.0]), np.array([-30.0]))

    def fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        # Nearest entry is an unnamed Gaia class; a VSX entry sits slightly further.
        return Table({
            "ra_deg": [10.0, 10.0002], "dec_deg": [-30.0, -30.0],
            "name": ["0", "V1234 Sgr"], "var_type": ["SOLAR_LIKE", "EA"],
            "period_days": [np.nan, 2.5],
        })

    result = match_known_variables(night, Settings(), fetcher=fetcher)
    assert result.matched[0]
    assert result.name[0] == "V1234 Sgr"
    assert result.var_type[0] == "SOLAR_LIKE|EA"
    assert result.period_days[0] == 2.5


def test_match_known_variables_degrades_on_fetch_failure(caplog) -> None:
    night = _minimal_night(np.array([10.0]), np.array([-30.0]))
    settings = Settings()

    def bad_fetcher(*_a, **_k):
        raise RuntimeError("no network")

    result = match_known_variables(night, settings, fetcher=bad_fetcher)
    assert not result.matched.any()
    assert "failed" in caplog.text.lower()


def test_match_known_planets_keeps_period_depth_toi() -> None:
    ra = np.array([10.0, 10.00003, 10.5])
    dec = np.array([-30.0, -30.00001, -30.2])
    night = _minimal_night(ra, dec)
    settings = Settings()

    def fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        return Table({
            "ra_deg": [10.00003], "dec_deg": [-30.00001],
            "name": ["WASP-999 b"], "period_days": [3.4], "depth": [0.0116], "is_toi": [False],
        })

    result = match_known_planets(night, settings, fetcher=fetcher)
    assert result.matched.tolist() == [True, True, False]
    assert result.name[0] == "WASP-999 b"
    assert result.period_days[0] == 3.4
    assert result.depth[0] == 0.0116
    assert not result.is_toi[0]


def test_compute_neighbour_dilution_finds_nearest_other_source() -> None:
    ra = np.array([10.0, 10.5])
    dec = np.array([-30.0, -30.0])
    night = _minimal_night(ra, dec)
    settings = Settings()

    def gaia_fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        # Star 0 has a close, fainter neighbour 3 arcsec away; star 1 has none.
        return Table({
            "ra_deg": [10.0, 10.0 + 3.0 / 3600.0 / np.cos(np.radians(-30.0))],
            "dec_deg": [-30.0, -30.0],
            "gaia_id": ["1", "2"],
            "gmag": [15.0, 17.0],
        })

    result = compute_neighbour_dilution(night, settings, fetcher=gaia_fetcher)
    assert result.has_neighbour[0]
    assert not result.has_neighbour[1]
    assert 2.0 < result.neighbour_sep_arcsec[0] < 4.0
    ratio = 10.0 ** (-0.4 * (17.0 - 15.0))
    expected_depth = ratio / (1.0 + ratio)
    assert abs(result.max_dilutable_depth[0] - expected_depth) < 1e-6


def test_compute_neighbour_dilution_degrades_offline() -> None:
    night = _minimal_night(np.array([10.0]), np.array([-30.0]))
    settings = Settings()

    def bad_fetcher(*_a, **_k):
        raise RuntimeError("no network")

    result = compute_neighbour_dilution(night, settings, fetcher=bad_fetcher)
    assert not result.has_neighbour.any()


def test_is_disqualifying_variable_type() -> None:
    settings = Settings()
    assert is_disqualifying_variable_type("EA|ESD", settings)
    assert is_disqualifying_variable_type("ECL", settings)
    assert is_disqualifying_variable_type("DSCT|GDOR|SXPHE", settings)
    assert not is_disqualifying_variable_type("SOLAR_LIKE", settings)
    assert not is_disqualifying_variable_type("ROT", settings)
    assert not is_disqualifying_variable_type("", settings)
