"""CLI smoke test for `relphot multinight`.

Builds two "nights" by running the real ingest/reference/lightcurves CLI
twice on the same tiny fixture frames (232 stars, 4 frames -- see
conftest.py), into two separate ``relphot/`` directories so
:func:`relphot.multinight.load_night_products` can read them back with its
default file layout. ``multinight.min_tie_stars`` is lowered via a TOML
config since the fixture is far below any realistic tie-star count. This is
a smoke test: it checks the command completes and writes the expected
files, not that the tie is scientifically meaningful (the two "nights" are
literally identical data).
"""

from __future__ import annotations

import numpy as np

from relphot.cli import main


def _build_night(fits_files, out_dir) -> None:
    out_dir.mkdir(parents=True)
    (out_dir / "lc").mkdir()

    night_out = out_dir / "night.npz"
    rc = main(["ingest", "--out", str(night_out), *[str(p) for p in fits_files]])
    assert rc == 0

    ref_out = out_dir / "ref.npz"
    rc = main(["reference", str(night_out), "--out", str(ref_out), "--no-variables"])
    assert rc == 0

    lc_out = out_dir / "lc" / "night_lc"
    rc = main([
        "lightcurves", str(night_out), str(ref_out),
        "--out", str(lc_out), "--no-variables", "--format", "fits", "--no-plot",
    ])
    assert rc == 0


def test_multinight_smoke(fits_files, tmp_path) -> None:
    night1_dir = tmp_path / "20250101" / "relphot"
    night2_dir = tmp_path / "20250102" / "relphot"
    _build_night(fits_files, night1_dir)
    _build_night(fits_files, night2_dir)

    config = tmp_path / "multinight.toml"
    config.write_text("[multinight]\nmin_tie_stars = 5\n")

    out_stem = tmp_path / "mn" / "mn_out"
    (tmp_path / "mn").mkdir()

    rc = main([
        "multinight", str(night1_dir), str(night2_dir),
        "--out", str(out_stem), "--config", str(config), "--format", "fits",
    ])
    assert rc == 0

    npz_path = out_stem.with_suffix(".npz")
    tie_csv = out_stem.with_name(f"{out_stem.name}_tie.csv")
    stars_path = out_stem.with_name(f"{out_stem.name}_stars.fits")
    lc_path = out_stem.with_name(f"{out_stem.name}_lightcurves.fits")
    assert npz_path.is_file()
    assert tie_csv.is_file()
    assert stars_path.is_file()
    assert lc_path.is_file()

    from relphot.multinight import load_multinight

    xmatch, tie, mlc, night_info, _settings = load_multinight(npz_path)
    assert xmatch.ra.shape[0] > 0
    assert len(night_info) == 2
    assert tie.coef.shape[0] == 2
    assert mlc.mag.shape[0] == xmatch.ra.shape[0]


def test_multinight_requires_two_nights(fits_files, tmp_path) -> None:
    night1_dir = tmp_path / "20250101" / "relphot"
    _build_night(fits_files, night1_dir)

    rc = main(["multinight", str(night1_dir), "--out", str(tmp_path / "mn_out")])
    assert rc == 1


def test_multisearch_smoke(fits_files, tmp_path) -> None:
    """`relphot multisearch` on the Unit A smoke output (B8): rc == 0, files written."""
    night1_dir = tmp_path / "20250101" / "relphot"
    night2_dir = tmp_path / "20250102" / "relphot"
    _build_night(fits_files, night1_dir)
    _build_night(fits_files, night2_dir)

    config = tmp_path / "multinight.toml"
    config.write_text("[multinight]\nmin_tie_stars = 5\n")

    out_stem = tmp_path / "mn" / "mn_out"
    (tmp_path / "mn").mkdir()
    rc = main([
        "multinight", str(night1_dir), str(night2_dir),
        "--out", str(out_stem), "--config", str(config), "--format", "fits",
    ])
    assert rc == 0

    search_dir = tmp_path / "search"
    rc = main([
        "multisearch", str(out_stem.with_suffix(".npz")), "--out-dir", str(search_dir), "--no-plot",
    ])
    assert rc == 0

    metrics_candidates = [
        search_dir / "multinight_search_metrics.parquet",
        search_dir / "multinight_search_metrics.fits",
    ]
    metrics = next((p for p in metrics_candidates if p.is_file()), None)
    assert metrics is not None, metrics_candidates
    assert (search_dir / "multinight_variables.csv").is_file()
    assert (search_dir / "multinight_transits.csv").is_file()
    assert (search_dir / "period_compat").is_dir()

    from astropy.table import Table

    table = Table.read(metrics)
    assert "internight_chi2" in table.colnames
    assert "bls_candidate" in table.colnames
    assert len(table) == 232


def test_multinight_loose_night_smoke_and_option_errors(fits_files, tmp_path) -> None:
    """`--loose` ties the last night loosely; multisearch reads the flag from the npz."""
    dirs = [tmp_path / f"2025010{i}" / "relphot" for i in (1, 2, 3)]
    for d in dirs:
        _build_night(fits_files, d)
    night_dirs = [str(d) for d in dirs]

    config = tmp_path / "multinight.toml"
    config.write_text("[multinight]\nmin_tie_stars = 5\nfloor_min_bin_stars = 5\n")
    (tmp_path / "mn").mkdir()
    out_stem = tmp_path / "mn" / "mn_loose"

    rc = main([
        "multinight", *night_dirs, "--out", str(out_stem), "--config", str(config),
        "--format", "fits", "--no-plot", "--loose", "20250103",
    ])
    assert rc == 0

    from relphot.multinight import load_multinight

    xmatch, tie, mlc, _info, _settings = load_multinight(out_stem.with_suffix(".npz"))
    assert xmatch.labels == ("20250101", "20250102", "20250103")
    assert list(tie.loose) == [False, False, True]
    assert tie.coef.shape[0] == 3 and mlc.night_mean_mag.shape[0] == 3
    assert np.isfinite(tie.zp[2]).any()

    rc = main([
        "multisearch", str(out_stem.with_suffix(".npz")), "--out-dir", str(tmp_path / "search"),
        "--no-plot",
    ])
    assert rc == 0

    # an unknown loose label, or the anchor given as loose, is an error, not a traceback
    for extra in (["--loose", "nope"], ["--loose", "20250102", "--anchor", "20250102"]):
        rc = main([
            "multinight", *night_dirs, "--out", str(tmp_path / "mn" / "bad"),
            "--config", str(config), "--format", "fits", "--no-plot", *extra,
        ])
        assert rc == 1
