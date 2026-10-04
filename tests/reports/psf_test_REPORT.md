# PSF photometry test (PLAN_IMPROVE_DETECT.md, 2026-10-03)

Forced PSF photometry for ROBO43 20250911 (WASP-145 field, 351 frames) and T80S 20251104 (MC0087, 45 frames),
run through the unchanged relphot chain and loaded into a parallel DB schema `test_psf` in the production database.
No relphot or robo43 code was changed. The plotting script is `scripts/psf_vs_aper_lc.py`.

## Where things are

| What | Path |
|---|---|
| PSF catalogues (`*_proc_forced_catalog.csv`), flag mapping/counts, relphot products | `/ssdsto1/data/tmp/psf_photometry/<TEL>/<night>/` (`relphot/`, stems `wasp145_psf`, `night_psf`; pre-clip originals in `relphot/lc/preclip/`) |
| PSF code (final: `psfphot.py`, `run_night.py`, `finalize.py`, `clip_lc.py`, `relphot_psf.sh`, `run_all.sh`) | `/ssdsto1/data/tmp/psf_photometry/code/` |
| Candidate pairs (40, same star in both modes) | `/ssdsto1/data/tmp/psf_photometry/candidate_pairs.csv` |
| `test_psf` schema dump, load log, prod row counts before/after | `/ssdsto1/data/tmp/psf_photometry/db/` |
| Prod backup taken before the restore | `/ssdsto1/data/relphotDB/backups/relphot_20261003_2037.dump` |
| Plots (default output of the script; not generated) | `/ssdsto1/data/tmp/psf_photometry/plots/` |

Generate the plots:

```
python scripts/psf_vs_aper_lc.py --pairs /ssdsto1/data/tmp/psf_photometry/candidate_pairs.csv
python scripts/psf_vs_aper_lc.py --aper-obj 64813 --psf-obj 232      # WASP-145 A b
python scripts/psf_vs_aper_lc.py --aper-obj 168850 --psf-obj 230     # its 5.2" neighbour
```

One figure per (star, night): aperture (schema `relphot`) left, PSF (schema `test_psf`) right; detections, fitted
trapezoids and dropped frames overlaid. Options: `--night`, `--shared-y`, `--bin-min`, `--format pdf`, `--outdir`,
`--aper-schema`, `--psf-schema`, `--dsn`. Read-only connection; schema names whitelisted and quoted.

## Method

- **Positions:** the production robo43 forced master positions, so sources map 1:1 to the aperture catalogues.
  Per-frame affine offset (PSF model frame vs catalogue) from bright isolated stars; linearised centroid refinement
  only for flux/err > 20 and no fitted source within 1.5 FWHM, shift capped at 1.2 px (else catalogue position).
  The first pure fixed-position version gave floors of 8-9 mmag (ROBO43) and 26-37 mmag (T80S proxy) against
  4-6 mmag for apertures, which is why these refinements were added.
- **PSF model:** SExtractor 2.28 + PSFEx 3.24.2 per frame, spatial degree 2, SAMPLE_MINSN 250; no frame failed.
- **Fit:** weighted least squares per stamp (2h+1 px, h = ceil(2.2 FWHM) <= 17) of the star, its neighbours and a
  local sky plane; cubic B-spline interpolated PSF; ERR-plane then model-Poisson weights.
- **Companions:** the production master list merges sources within ~4"; a static list of extra sources (32023 T80S,
  3 ROBO43) from a pass-1 residual search on every 7th / 50th frame, kept if present in >= 50 % of pass-1 frames and
  > 1.4" from a master source, is fitted identically in every frame. A per-frame companion search made the source
  model vary between frames and gave 12585 TOO_DEEP stars on T80S (2717 with the static list, 207 in production).
- **Columns:** PSF flux in `FLUX_APER_1` only, so relphot sees one aperture: no best-aperture choice and the
  APERTURE_INCONSISTENT check cannot fire. FWHM and BACKGROUND copied from the production forced catalogue.
- **relphot settings:** as production. ROBO43: `reference --aper 0`, search `--snr-threshold 5.5`. T80S:
  `--no-variables`, default threshold.

## Flags (relphot bad mask 252 = bits 4-128)

| Bit | Meaning | ROBO43 (235593 source-epochs) | T80S (4661135) |
|---|---|---|---|
| 1 | another source within 4" (master or static extra) | 0.26 % | 21.1 % |
| 2 | blend: another source within 1.4" | 0 | 0.16 % |
| 4 | saturated or non-linear pixel within 4" | 0.61 % | 0.42 % |
| 8 | centre within 20 px of the edge | 1.20 % | 0.78 % |
| 16 | bad pixel within 1 FWHM, or > 20 % of the stamp masked | 0 | 0.14 % |
| 32 | fit failed | 0.41 % | 0.23 % |
| 64 | poor fit: core chi2 / star median / frame factor > 1 + 6 robust sigma (2.29 ROBO43, 2.49 T80S) | 0.23 % | 1.40 % |
| 128 | flux or error non-finite or <= 0 | 0.43 % | 1.33 % |

Production uses bit 64 for non-linear; here non-linear pixels are in bit 4 (the flag bits are read only in
`config.py`). Bits 1 and 2 are not masked.

## Sigma-clipping

Per star, MAD around a running median (window ~15 min: 29 epochs ROBO43, 7 T80S, epoch itself excluded), k = 4,
sigma floor 0.5 x median lc_err, <= 5 iterations; runs of >= 3 same-sign outliers never clipped (transit-safe).
Applied between `lightcurves` and `search` (`code/clip_lc.py`): clipped epochs are NaN in `lc/<stem>.npz`, removed
from `_lightcurves.parquet`, and starstats rms / chi2 / expected noise / n_epochs recomputed.
Clipped: ROBO43 334 of 201397 epochs (0.17 %, 102 stars); T80S 11981 of 4187802 (0.29 %, 7752 stars).
WASP-145 A lost 19 isolated epochs and is still detected.

## Database

`relphot.` is hard-coded in every query, so: scratch database `relphot_psfwork` (owner relphot_owner, q3c),
`relphot db init` (v14), `db load-night` for both nights (`--telescope`, `--label`, `--lc-stem`), `db analyze --all`;
then `ALTER SCHEMA relphot RENAME TO test_psf`, dropped the `reprocess_request_notify` trigger (it would wake the
production worker on channel `relphot_reprocess`), `pg_dump -n test_psf`, `pg_restore` into the production database
`relphot`, scratch database dropped. Grants for relphot_ro / relphot_web came along.
Production `relphot` row counts identical before and after (night 14, object 216586, detection 23224,
star_night 1524050, user_night_review 1365, statuses CONFIRMED 2 / REJECTED 1128 / UNCONFIRMED 22094).
The relphot CLI cannot target `test_psf` directly (re-loading means repeating the scratch-database route).

test_psf: night_id 1 = ROBO43 20250911 (314/351 frames, 405 stars stored), night_id 2 = T80S 20251104 (42/45
frames, 74316 stored); 74721 objects, 239 detections.

## Results

PSF vs production forced aperture photometry, same relphot code:

| | ROBO43 aperture | ROBO43 PSF | T80S aperture | T80S PSF |
|---|---|---|---|---|
| frames kept | 310/351 | 314/351 | 45/45 | 42/45 |
| bright-star floor | 6.47 mmag (all stars, N=26) | 7.08 (N=30) | 4.8 (ap0) | 4.6 |
| transit candidates (tiers 1/2/3) | 6 (2/3/1) | 6 (3/2/1) | 127 (12/83/32) | 152 (28/77/47) |
| DB transit / variable detections | 6 / 4 | 6 / 2 | 141 / 63 | 152 / 79 |
| TOO_DEEP stars | 2 | 0 | 207 | 2717 |

- RMS ratio PSF/aperture, T80S: 0.90 (-15..-14 instr. mag), 0.98, 1.02, 1.04, 1.10, 1.14 (-10..-9); comparison stars
  0.93-1.02. Faint non-comparison stars are ~14 % noisier in PSF. ROBO43: 0.98-1.07 bright, 0.94-0.97 faint.
- PSF/aperture (8") flux ratio, bright isolated stars: 1.0018 (ROBO43), 1.0038 (T80S); per-frame zero-point
  difference rms 1.5 / 2.2 mmag.
- T80S PSF variable candidates (5321 vs 2826 in candidates.csv) are inflated by faint-star TOO_DEEP events
  (median SNR 2.7): a faint-end artefact of the PSF fluxes, not new variables.
- Three T80S frames (18, 23, 24) fail the frame-quality scatter cut in PSF only (the PSF frame-scatter
  distribution is tighter, so they stand out).
- **WASP-145 A b** (aperture obj 64813, PSF obj 232): PSF transit SNR 8.9, fit depth 0.96 %, T14 0.95 h, LC rms
  5.7 ppt; aperture SNR 9.4, depth 1.92 %, T14 0.61 h, rms 21.8 ppt (outliers). Literature 1.16 %, 0.98 h.
  A different static-companion iteration gave SNR 6.3 and a 2.05 h box: detection robust, box parameters not.
- **Neighbour 5.2"** (aperture obj 168850, PSF obj 230, Gaia DR3 6458529931463278976): aperture shows a 5.2 %
  "transit" and a variable, rms 58.8 ppt (WASP-145 A's light in its aperture); PSF: no detection, rms 7.5 ppt.
  PSF removes the neighbour contamination.
- Candidate pairs: 59 stars have detections in both modes; 40 in the CSV (the two WASP-145 stars, then by the lower
  SNR of the two modes). cand23 and cand24 got a different (neighbouring) Gaia id in `test_psf` than in `relphot`;
  same fixed position.

## Not verified

- Clip sensitivity (search with and without the clip); injection recovery on PSF fluxes.
- `test_psf` in the web app (the app reads `relphot` only).
- The edge-outlier overlay of the script on rows with non-empty `transit_shape.edge_clip_bjd`.
