"""CLI smoke test for `relphot search`, on the real cropped fixture frames.

The fixture is tiny (232 stars, 4 frames -- see conftest.py), far below any
realistic transit-search epoch count, so ``search.min_epochs`` and the
duration grid are overridden via a TOML config to actually exercise the
Stage-6 code paths on every star instead of skipping all of them. This is a
smoke test: it checks the command completes and writes the expected files,
not that it finds anything.
"""

from __future__ import annotations

from relphot.cli import main


def test_search_smoke(fits_files, tmp_path) -> None:
    night_out = tmp_path / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = tmp_path / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    lc_out = tmp_path / "lc"
    rc = main([
        "lightcurves", str(night_out), str(ref_out),
        "--out", str(lc_out), "--no-variables", "--format", "fits", "--no-plot",
    ])
    assert rc == 0
    lc_npz = tmp_path / "lc.npz"

    config = tmp_path / "search.toml"
    config.write_text(
        "[search]\n"
        "min_epochs = 2\n"
        "min_epoch_fraction = 0.0\n"
        "n_cbv = 1\n"
        "cbv_explained_variance = 0.999\n"
        "duration_min_hours = 0.01\n"
        "duration_max_hours = 0.03\n"
        "n_durations = 2\n"
        "min_in_transit_points = 1\n"
        "few_points_threshold = 1\n"
        "lc_clip_window = 3\n"
        "coincidence_snr_threshold = 1.0\n"
        "excess_rms_threshold = 1.0\n"
        "variability_min_bin_stars = 1\n"
    )

    transits_dir = tmp_path / "transits"
    variables_dir = tmp_path / "variables"
    rc = main([
        "search", str(night_out), str(ref_out), str(lc_npz),
        "--config", str(config),
        "--transits-dir", str(transits_dir),
        "--variables-dir", str(variables_dir),
        "--no-catalogs", "--no-plot",
    ])
    assert rc == 0

    assert (transits_dir / "candidates.csv").is_file()
    assert (variables_dir / "candidates.csv").is_file()
    assert (variables_dir / "new_variables.csv").is_file()
    metrics_candidates = [
        tmp_path / "lc_search_metrics.parquet", tmp_path / "lc_search_metrics.fits"
    ]
    metrics = next((p for p in metrics_candidates if p.is_file()), None)
    assert metrics is not None, metrics_candidates

    from astropy.table import Table

    table = Table.read(metrics)
    assert "transit_snr" in table.colnames
    assert "variability_excess" in table.colnames
    assert len(table) > 0
    for column in (
        "transit_r90_evaluated", "transit_r90_pass", "transit_top1_share",
        "transit_top3_share", "transit_reg_dchi2_ratio", "transit_reg_depth_ratio",
        "transit_clip3_dchi2", "transit_dbic_flat",
    ):
        assert column in table.colnames, column


def test_search_known_variable_with_transit_stays_a_candidate(
    fits_files, tmp_path, monkeypatch
) -> None:
    """A transit event on a known (disqualifying-type) variable is still a transit candidate,
    carrying ON_VARIABLE, in candidates.csv and in the search-metrics table."""
    import numpy as np
    from astropy.table import Table

    import relphot.cli as cli
    from relphot.catalogs import (
        NeighbourDilutionResult,
        PlanetMatchResult,
        VariableMatchResult,
    )

    night_out = tmp_path / "night.npz"
    assert main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]]) == 0
    ref_out = tmp_path / "ref.npz"
    assert main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"]) == 0
    assert main([
        "lightcurves", str(night_out), str(ref_out),
        "--out", str(tmp_path / "lc"), "--no-variables", "--format", "fits", "--no-plot",
    ]) == 0

    config = tmp_path / "search.toml"
    config.write_text(
        "[search]\nmin_epochs = 2\nmin_epoch_fraction = 0.0\nn_cbv = 1\n"
        "cbv_explained_variance = 0.999\nduration_min_hours = 0.01\n"
        "duration_max_hours = 0.03\nn_durations = 2\nmin_in_transit_points = 1\n"
        "few_points_threshold = 1\nlc_clip_window = 3\ncoincidence_snr_threshold = 1.0\n"
        "excess_rms_threshold = 1.0\nvariability_min_bin_stars = 1\n"
    )

    target = 5
    real_search_transits = cli.search_transits

    def search_with_forced_candidate(*args, **kwargs):
        result = real_search_transits(*args, **kwargs)
        result.searched[target] = True
        result.candidate[target] = True
        result.flags[target] &= ~cli.FLAG_TOO_DEEP
        return result

    def known_variable(night, _settings):
        n = night.n_stars
        matched = np.zeros(n, dtype=bool)
        matched[target] = True
        name = np.full(n, "", dtype=object)
        name[target] = "V* Test"
        var_type = np.full(n, "", dtype=object)
        var_type[target] = "EA"
        return VariableMatchResult(matched, name, var_type, np.full(n, np.nan))

    def no_planets(night, _settings):
        n = night.n_stars
        return PlanetMatchResult(
            np.zeros(n, dtype=bool), np.full(n, "", dtype=object), np.full(n, np.nan),
            np.full(n, np.nan), np.zeros(n, dtype=bool),
        )

    def no_neighbours(night, _settings):
        n = night.n_stars
        return NeighbourDilutionResult(
            np.full(n, "", dtype=object), np.zeros(n, dtype=bool), np.full(n, np.nan),
            np.full(n, np.nan),
        )

    monkeypatch.setattr(cli, "search_transits", search_with_forced_candidate)
    monkeypatch.setattr(cli, "match_known_variables", known_variable)
    monkeypatch.setattr(cli, "match_known_planets", no_planets)
    monkeypatch.setattr(cli, "compute_neighbour_dilution", no_neighbours)

    transits_dir = tmp_path / "transits"
    variables_dir = tmp_path / "variables"
    rc = main([
        "search", str(night_out), str(ref_out), str(tmp_path / "lc.npz"),
        "--config", str(config),
        "--transits-dir", str(transits_dir), "--variables-dir", str(variables_dir),
        "--no-plot",
    ])
    assert rc == 0

    import csv

    with (transits_dir / "candidates.csv").open() as handle:
        rows = {int(r["star_id"]): r for r in csv.DictReader(handle)}
    assert target in rows
    assert "ON_VARIABLE" in rows[target]["flags"].split("|")
    assert rows[target]["r90_pass"] == ""  # forced in, never screened: not evaluated
    assert {"top1_share", "top3_share", "reg_dchi2_ratio", "reg_depth_ratio", "clip3_dchi2",
            "dbic_flat"} <= set(rows[target])
    with (variables_dir / "candidates.csv").open() as handle:
        variable_ids = {int(r["star_id"]) for r in csv.DictReader(handle)}
    assert target in variable_ids  # still listed as a (known) variable as well

    metrics = next(
        p for p in (tmp_path / "lc_search_metrics.parquet", tmp_path / "lc_search_metrics.fits")
        if p.is_file()
    )
    table = Table.read(metrics)
    row = table[int(np.flatnonzero(np.asarray(table["star_id"]) == target)[0])]
    assert bool(row["transit_candidate"])
    assert "ON_VARIABLE" in str(row["transit_flags_str"]).split("|")
