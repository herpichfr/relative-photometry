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
