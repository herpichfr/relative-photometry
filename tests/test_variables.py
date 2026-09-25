"""Tests for relphot.variables: known-variable cross-match, entirely offline.

Every test injects a fake fetcher (or a fake ``_query_vizier``); none of them
touch the network or require astroquery.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from astropy.table import Table

import relphot.variables as variables_mod
from relphot.config import Settings
from relphot.match import MatchedNight
from relphot.variables import fetch_known_variables, flag_known_variables


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


def test_disabled_flags_nothing(caplog) -> None:
    night = _minimal_night(np.array([10.0, 10.1]), np.array([-30.0, -30.1]))
    settings = replace(Settings(), variable=replace(Settings().variable, enabled=False))
    mask = flag_known_variables(night, settings)
    assert not mask.any()
    assert "disabled" in caplog.text.lower()


def test_fetcher_failure_flags_nothing_and_warns(caplog) -> None:
    night = _minimal_night(np.array([10.0, 10.1]), np.array([-30.0, -30.1]))
    settings = Settings()

    def bad_fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        raise RuntimeError("no network")

    mask = flag_known_variables(night, settings, fetcher=bad_fetcher)
    assert not mask.any()
    assert "failed" in caplog.text.lower()


def test_cross_match_flags_only_the_close_star() -> None:
    ra = np.array([10.0, 10.00003, 10.5])
    dec = np.array([-30.0, -30.00001, -30.2])
    night = _minimal_night(ra, dec)
    settings = Settings()

    def fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        return Table({"ra_deg": [10.00003], "dec_deg": [-30.00001]})

    mask = flag_known_variables(night, settings, fetcher=fetcher)
    assert mask.tolist() == [True, True, False]


def test_empty_fetch_result_flags_nothing() -> None:
    night = _minimal_night(np.array([10.0]), np.array([-30.0]))
    settings = Settings()

    def fetcher(_ra_c, _dec_c, _radius_deg, _settings):
        return Table(names=["ra_deg", "dec_deg"], dtype=[float, float])

    mask = flag_known_variables(night, settings, fetcher=fetcher)
    assert not mask.any()


def test_fetch_known_variables_caches_to_ecsv(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_query(_catalog, _ra_c, _dec_c, _radius_deg):
        calls["n"] += 1
        return Table({"ra_deg": [10.0], "dec_deg": [-30.0]})

    monkeypatch.setattr(variables_mod, "_query_vizier", fake_query)
    settings = replace(
        Settings(),
        variable=replace(
            Settings().variable, cache_dir=str(tmp_path), catalogs=("TEST/CAT",), extra_catalogs=()
        ),
    )
    table1 = fetch_known_variables(10.0, -30.0, 0.1, settings)
    table2 = fetch_known_variables(10.0, -30.0, 0.1, settings)
    assert calls["n"] == 1  # second call was served from the cache
    assert len(table1) == 1
    assert len(table2) == 1
    assert len(list(tmp_path.glob("*.ecsv"))) == 1
