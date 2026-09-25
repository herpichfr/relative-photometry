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
