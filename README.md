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

## Results database

Loaded night-by-night outputs (light curves, transit/variability detections,
catalogue cross-matches) live in a queryable PostgreSQL database with a web
front end -- see [`deploy/README.md`](deploy/README.md) for the operator
guide (install, nightly workflow, CLASS/PERIOD rules, backups) and
[`docs/DB_PLAN.md`](docs/DB_PLAN.md) for the design.
