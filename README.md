# relative-photometry (relphot)

High-precision relative photometry for exoplanet transit detection, built on
existing per-frame photometric catalogues (robo43 SExtractor `CATALOG`/`.csv`,
or any other method through a configurable column mapping).

See [`PLAN.md`](PLAN.md) for the full design: goals, algorithm stages,
decisions, and the phase breakdown. This package currently implements
**Phase 1a** (ingest, catalogue adapters, cross-match, star x frame arrays)
and **Phase 1b** (adaptive tiling, known-variable cross-match, per-tile
reference construction) — see `PLAN.md`'s "Phases" table for what comes next.

## Quick start

```
pip install -e .
relphot ingest --out night.npz /path/to/*_proc.fits
relphot reference night.npz --out ref.npz --no-variables
```

### Forced (fixed-centroid) photometry

After `robo43 forced` has written `*_proc_forced_catalog.csv` (or a `CATALOG_FORCED` extension)
next to the products, ingest them with the same file list:

```
relphot ingest --photometry forced --out night.npz /path/to/*_proc.fits
```

Each `*_proc.fits` (or `*_proc_catalog.csv`) is mapped to its frame's forced catalogue, which has
the same columns; nothing downstream changes. The choice is stored in the `night.npz` settings
(`catalog.photometry`, `"standard"` by default; `catalog.forced_hdu_name` names the extension).

### Frame-quality cut

The reference stage drops a frame globally when it is a robust outlier of the night in the
reference ensemble's scatter, transparency (against a quadratic trend in time), FWHM or sky
level (median/MAD z-score above `reference.frame_quality_sigma`, 3.0, and also off by at least
`frame_quality_min_excess`, 5 %). At most `frame_quality_max_fraction` (15 %) of the frames are
cut, worst first, and never more than `max_dropped_frame_fraction` (30 %) together with the
star-set rule. `reference.frame_quality` is `"auto"` by default: the cut applies to forced
photometry only (where every frame has the same sources and the star-set rule never fires),
`"on"` applies it to every night, `"off"` never. `relphot reference` always writes
`<stem>_frames.csv` (metrics, z-scores, flag, reason, kept) and logs every flagged frame.

## Results database

Loaded night-by-night outputs (light curves, transit/variability detections,
catalogue cross-matches) live in a queryable PostgreSQL database with a web
front end -- see [`deploy/README.md`](deploy/README.md) for the operator
guide (install, nightly workflow, CLASS/PERIOD rules, backups) and
[`docs/DB_PLAN.md`](docs/DB_PLAN.md) for the design.
