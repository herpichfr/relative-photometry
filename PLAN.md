# Relative photometry pipeline — plan

Status: plan agreed 2026-09-25. Phase 1a (ingest, cross-match) done; next is 1b.

## Goal

High-precision relative photometry for every star in a field across a night's
frames, aimed at exoplanet transit detection. Input is per-frame photometric
catalogues that already exist; this pipeline does no photometry of its own.

## Requirements

1. Large images are split into rectangles (tiles) so field inhomogeneities do
   not enter the relative photometry.
2. Each tile has its own reference star, which removes the sky/instrument
   variability common to all stars in the tile (clouds, airmass, transparency).
3. The reference is built from stars that are neither saturated/non-linear nor
   known variables (checked against variable-star databases), using a robust
   combination so outliers do not bias it.
4. Every star's flux in the tile is divided by the reference flux.
5. Comparison stars are the most flux-stable stars after reference division.
   Their number varies per tile; the more the better.
6. Every other star is compared with each comparison star; the combined
   residual flux is its light curve, and the comparison stars' own residuals
   give the scatter used for statistics.
7. Lives in `~/Dropbox/relative-photometry/`.
8. As fast as possible.
9. Supersedes the legacy `get_relative_photometry.py` of the robo43 reduction
   pipeline (one target / one comparison / one check star, fixed coordinates).

## Decisions

- **Flux source:** existing catalogues only — SExtractor (the robo43 `CATALOG`
  HDU of `*_proc.fits`, or its `_catalog.csv`) or any other photometric method,
  through a configurable column mapping (position, flux, flux error, flags,
  SNR, FWHM, per-aperture vectors).
- **Coverage:** a star need not appear in every catalogue of the night. It
  enters statistics (reference, comparison, light-curve metrics) only if it is
  present in >= 80% of the frames (configurable). Missing epochs are NaN and
  every statistic is NaN-aware.
- **Tiles:** default about 1000x1000 px (in the master frame's sky projection,
  not detector coordinates, so dithers do not move stars between tiles). Size
  is user-settable, and an adaptive loop grows or merges tiles until each holds
  at least `min_ref_candidates` (default 50, hard floor 20) clean candidates.
  The final tile map is written to the diagnostics.
- **Scope:** Phase 1 is single night. Multi-night comparison is Phase 2.
- **Reference construction:** inverse-variance weighted, MAD sigma-clipped mean
  of per-star normalised fluxes (see the investigation below). Median of
  normalised fluxes is kept as a fallback option. Median or sum of raw fluxes
  is not offered.
- **Language:** Python with vectorised numpy on dense star x frame arrays,
  numba only for hot loops that profiling identifies; a Rust (PyO3) core only
  if a benchmark shows the need.

## Reference-star investigation (2026-09-25)

Tested on T80S night 20251104 (45 R-band 90 s frames, LMC field, 59,704 stars
present in >= 80% of frames), 4 tiles of 1000x1000 px, candidates FLAGS==0,
SNR >= 15, several random seeds. Scripts lived in a session scratchpad and did
not persist.

Constructions: (A) median of raw fluxes; (B) median of per-star normalised
fluxes; (C) sum of raw fluxes; (D) inverse-variance weighted, sigma-clipped
mean of normalised fluxes.

Reference self-noise, measured as the robust RMS of R(subset 1)/R(subset 2)
from two disjoint candidate subsets, divided by sqrt(2), in mmag:

| N candidates | A | B | C | D |
|---|---|---|---|---|
| 5 | 109.1 | 54.2 | 47.4 | 44.3 |
| 10 | 67.3 | 34.4 | 32.5 | 25.5 |
| 20 | 57.6 | 24.1 | 25.2 | 16.7 |
| 50 | 42.6 | 16.3 | 17.5 | 11.0 |
| 100 | 30.7 | 11.6 | 12.8 | 8.2 |
| 200 | 20.5 | 6.9 | 8.4 | 5.8 |

Findings:

- (A) is 2-4x worse at every N. The median-ranked star changes identity in
  every frame, and candidates span about 40x in raw flux, so neighbouring order
  statistics are far apart. Normalising each star first removes this.
- (D) is best at every N and best on the bright (SNR >= 200) evaluation stars
  (1.87x photon noise vs 2.37 for B, 2.78 for C).
- Gains flatten past about 50 candidates, hence the tile minimum of 50.
- The common transparency/airmass signal across the night is 10-25% peak to
  peak per tile, so a reference is essential.
- A uniform dimming of whole frames (simulated cloud) cancels exactly for A and
  C. For B and D it shifts each star's normalising median, which by derivation
  rescales the whole light curve by a near-constant factor (removed by the final
  normalisation) rather than distorting the injected frames. Mitigation anyway:
  compute each star's normalising baseline iteratively, excluding frames whose
  preliminary reference is an outlier.
- Candidate selection (FLAGS==0, SNR cut) matters as much as the combining
  statistic. Never skip it.

Open point from the test: bright-star absolute RMS came out about 35 mmag,
far above the 2-3 mmag per 90 s reported earlier for the same night with a 4"
aperture. The test used the largest aperture (APERRAD index 4, 8 px) in a
crowded field; aperture choice and the error model must be checked in Phase 1.

## Algorithm

### Stage 1 — ingest and cross-match
- Read only catalogue tables (never image planes).
- Master star list from the frame with most sources and good seeing, plus stars
  not detected there; match every frame to it with a KD-tree on sky unit
  vectors (about 1").
- Dense float32 arrays `F[star, frame, aper]`, `sigmaF`, `flags`, `x`, `y`,
  `fwhm`, cached as `.npz`/HDF5.
- Frame metadata: BJD_TDB at mid-exposure, airmass, FWHM, filter, exposure.
  Frames grouped by field and filter.

### Stage 2 — tiling
- Grid in master-frame projection with an overlap margin: edge stars may use
  reference/comparison stars from the margin; each star's light curve comes
  from its core tile.
- Adaptive loop until every tile reaches `min_ref_candidates`.

### Stage 3 — reference star per tile
- Candidates: FLAGS==0 whenever present (saturated bit 4 and non-linear bit 64
  never set), SNR >= 15, present >= 80%, isolated, not a known variable.
- Known variables: cached VizieR cone searches — VSX, Gaia DR3 variability,
  ASAS-SN variables, OGLE (for Magellanic fields).
- R_j = inverse-variance weighted, MAD-clipped mean over candidates of
  f_ij / baseline_i, with baseline_i the star's median over frames computed
  iteratively without outlier frames.
- r_ij = f_ij / R_j for every star in the tile.

### Stage 4 — comparison selection
- Robust (MAD) scatter of r_ij over time per star; fit a scatter-vs-magnitude
  noise floor per tile; keep every star within k x the floor (no fixed count).
- Iterate (2-4 rounds): rebuild the comparison ensemble, re-measure, re-select.
- Score each comparison star against a leave-one-out ensemble.
- Ensemble statistic configurable: weighted clipped mean (default, per the
  investigation) or median; validate both in Phase 1 with the split-subset
  metric.

### Stage 5 — light curves and statistics
- LC_ij = r_ij / ensemble_j (for a median ensemble this equals the median over
  comparison stars of r_ij / r_kj, so no per-pair loop is needed).
- Per-epoch error: the star's own photon error combined with the comparison
  dispersion.
- Comparison stars' leave-one-out curves give the scatter-vs-magnitude
  relation.
- Per-star metrics: RMS, reduced chi-square, expected noise, epochs used.
- All apertures processed; best aperture chosen per tile and magnitude bin.

### Outputs
Light-curve table (Parquet, plus FITS), per-star statistics table, per-tile
diagnostics (tile map, reference curve, candidate and comparison counts),
RMS-vs-magnitude plots saved non-interactively.

## Performance
- Catalogue reading dominates; 45 frames x 70k stars expected well under a
  minute end to end.
- Tiles processed in parallel (ProcessPoolExecutor); tiles streamed for long
  runs; RAM released between stages.
- Error bars depend on consistent flux units (ADU vs electrons) upstream in the
  robo43 pipeline; ratios do not.

## Layout
```
relative-photometry/
  src/relphot/  ingest.py  match.py  tiles.py  variables.py  reference.py
                comparison.py  lightcurve.py  stats.py  io.py  cli.py
  tests/        small real-data sample copied from actual frames
```

## Phases

| Phase | Work | Done when |
|---|---|---|
| 1a | Ingest, catalogue adapters, cross-match, star x frame arrays | 45 T80S frames load in seconds; match rate reported |
| 1b | Tiling loop, variable-star lookup, reference (D) | Reference curves track airmass (k about 0.17); tile minimum met |
| 1c | Comparison selection, light curves, statistics, aperture choice | Bright-star RMS reconciled with the earlier 2-3 mmag result |
| 1d | Validation | Injected 5 mmag transits recovered; ROBO43 EtaCar agrees with the legacy script |
| 1e | Profiling, parallel tiles, CLI | Benchmark recorded |
| 2 | Multi-night: tie nights to a common per-star baseline via shared comparison stars | Night-to-night offsets consistent within errors |
