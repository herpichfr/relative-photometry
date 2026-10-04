# PSF re-processing of all reduced nights (2026-10-03/04)

All 14 reduced nights (ROBO43 20250911 and 13 T80S MC0087 nights, 20251104-20251209) were re-processed with the forced PSF
photometry of `tests/reports/psf_test_REPORT.md`, one night at a time. Then relphot ran per night, and the results database was
backed up, wiped and repopulated from the PSF products only. All earlier user vetting is gone with the old database. The aperture
photometry was moved to `/mnt/sto01/scratch/aperture/`.

## What was done

1. **Code.**
   - The test code went into the repo as `scripts/psf/` (commit 26e2237, merge e9baf26). It has env-configurable paths, relative
     `*_proc.fits` links, atomic pickles and loud frame errors, and frame counts are checked at every step.
   - Two failure-path bugs were fixed. The 3-value return of `_calibrate_shift` would have crashed frames with fewer than 8
     calibration stars. Non-finite rows in robo43 detection catalogues crashed pass 1 (commit e5bfb24, merge 969516c; see Issues).
   - **The port reproduces the test catalogues byte for byte**: ROBO43 20250911 (351 frames) and T80S 20251104 (45 frames),
     including `static_extras.csv` and the flag counts.
2. **Database (schema `relphot`).**
   - Web and worker were stopped first.
   - Backup `relphot_20261003_2322.dump`, pinned as `pre_psf_wipe_20261003_2322.dump` (aperture DB plus `test_psf`; 27 + 27 tables).
   - `TRUNCATE` of all 26 tables except `schema_version`, with `RESTART IDENTITY`. Schema v14, grants and the reprocess trigger
     are unchanged, and the `test_psf` schema is untouched.
   - Nights were loaded in date order, so night_id order is time order (`analyze` and the web use it as a time proxy).
3. **Per night** (`scripts/psf/rerun_all.sh` calling `psf_night.sh`; logs in
   `/mnt/sto01/T80S/reduced/psf_rerun_logs_20261004/`):
   - PSF fit and finalize on SSD staging.
   - Move the aperture products to scratch.
   - Sync `psf/` to `/mnt/sto01/<TEL>/reduced/<night>/psf/`.
   - relphot, with the ingest reading the permanent `psf/*_proc.fits` paths, so the DB `frame.file_path` values are permanent.
   - Sync `relphot/` to the permanent night dir.
   - `db load-night` from the permanent dir.
   - Remove staging.
   - relphot settings are the production ones:
     - T80S: `--no-variables` for reference and light curves, default search threshold, stem `night_lc`.
     - ROBO43: `reference --aper 0`, `--snr-threshold 5.5`, stem `wasp145_psf`.
     - Both: `clip_lc.py <dir> <stem> 4.0 15` between `lightcurves` and `search`.
   - The reference rule (identical star set in every kept frame; a frame where a tile has too few reference stars is dropped)
     and the comparison strategy are unchanged relphot code.
4. **Tie.**
   - `scripts/psf/tie_psf.sh` produced `mn_1104_1209_loose_psf` (mn_run 1).
   - Core nights 1104-1107; the other 9 nights are loose; anchor (auto) 20251105.
   - Then `multisearch`, `load-multinight` and `analyze --all --keep-vetted` (601 s).
5. **After the load.**
   - Web and worker restarted; the API was smoke-tested (`/api/nights`, `/api/search` and object detail all return 200).
   - Backup `relphot_20261004_0814.dump`, pinned as `post_psf_rerun_20261004_0814.dump`.
6. **Single-aperture tie smoke test.** It ran before the full tie: 1104 + 1105 in scratch, `multinight` and `multisearch`
   both rc 0, floor(bright) 5.94 mmag, holdout chi2 1.00.

## Per night

The aperture-era columns are from the last forced-aperture relphot run, now in scratch. "Best ap." is the lowest of the five
aperture floors.

- **floor**: relphot bright-star floor, in mmag.
- **cand.**: transit candidates.
- **open**: transit detections in the DB that are not auto-rejected after `analyze`.
- **corner**: frames with a 3x3 detector region where more than 15 % of the stars have PSF bit 64 (poor fit) set.

| night | id | frames kept (PSF / ap.) | floor PSF / best ap. | static extras | fit s | cand. PSF (ap.) | open | variability cand. (new) | stored | corner |
|---|---|---|---|---|---|---|---|---|---|---|
| ROBO43 20250911 | 1 | 314/351 / 310 | 5.6 / 6.5* | 3 | 220 | 6 (6) | 6 | 7 (0) | 405 | 0 |
| 20251104 | 2 | 42/45 / 45 | 4.6 / 4.8 | 32023 | 892 | 152 (139) | 124 | 5321 (2712) | 74316 | 2 |
| 20251105 | 3 | 65/73 / 69 | 3.9 / 4.2 | 27911 | 1205 | 451 (407) | 184 | 3796 (1318) | 68467 | 0 |
| 20251106 | 4 | 64/69 / 68 | 3.8 / 4.0 | 19487 | 1144 | 145 (87) | 104 | 2496 (168) | 63137 | 0 |
| 20251107 | 5 | 65/67 / 65 | 5.5 / 5.8 | 62907 | 2141 | **3622** (206) | 693 | 11671 (8400) | 155147 | **29** |
| 20251112 | 6 | 23/27 / 23 | 3.7 / 4.7 | 142079 | 1234 | 75 (93) | 73 | 9519 (6477) | 137623 | 0 |
| 20251118 | 7 | 23/27 / 23 | 3.6 / 4.5 | 140135 | 1243 | 30 (42) | 28 | 8360 (5350) | 131505 | 0 |
| 20251130 | 8 | 27/27 / 27 | 3.9 / 4.7 | 113977 | 1073 | 134 (47) | 72 | 9551 (6727) | 115514 | 2 |
| 20251201 | 9 | 37/41 / 37 | 4.0 / 4.4 | 72522 | 1065 | 40 (44) | 35 | 6576 (3850) | 96680 | 0 |
| 20251204 | 10 | 40/40 / 39 | 4.5 / 4.5 | 41751 | 793 | 76 (59) | 70 | 5214 (2720) | 70788 | 0 |
| 20251206 | 11 | 41/42 / 39 | 4.9 / 4.5 | 70310 | 1068 | 54 (23) | 26 | 7182 (4476) | 97643 | 1 |
| 20251207 | 12 | 67/67 / 66 | 6.0 / 6.0 | 73311 | 2246 | **3388** (404) | 635 | 11327 (8052) | 166024 | **23** |
| 20251208 | 13 | 36/42 / 42 | 4.0 / 4.7 | 109445 | 1332 | 43 (41) | 38 | 9142 (6290) | 118960 | 0 |
| 20251209 | 14 | 66/67 / 67 | 3.9 / 5.1 | 104869 | 1866 | 432 (272) | 232 | 7828 (4987) | 116882 | 2 |

\* ROBO43 aperture floor from the PSF test report (all-star bin); the other values are relphot's 0.25th-percentile anchor.

- **Per-night bright floor:** PSF is lower than the best aperture on 9 of the 13 T80S nights (by up to 1.2 mmag), equal on 1204
  and 1207, and worse on 1206 (4.9 vs 4.5).
- **Frames:** the frame-quality cut ("auto" applies to forced/PSF photometry) removes 1-8 frames on several nights. No night was
  lost to the reference rule.
- **Variability candidates** are inflated on every night by faint stars (second issue below). Only a small subset reaches the DB
  as `variable` detections: 2-1171 per night.
- **WASP-145 A b** (obj_id 232): SNR 8.2, fitted depth 0.96 %, T14 0.88 h (literature 1.16 %, 0.98 h), class EXOP+VAR.
  This is the same result as the PSF test.

## Multi-night tie (`mn_1104_1209_loose_psf`)

floor(bright) per night in mmag, PSF vs the aperture-era tie `mn_1104_1209_loose_forced`:

| night | 1104 | 1105 | 1106 | 1107 | 1112 | 1118 | 1130 | 1201 | 1204 | 1206 | 1207 | 1208 | 1209 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSF | 6.01 | 12.12 | 24.76 | 11.94 | 18.70 | 20.89 | 17.43 | 18.46 | 18.81 | 14.76 | 17.10 | 18.39 | 18.98 |
| aperture | 4.89 | 3.82 | 9.97 | 12.14 | 13.81 | 15.99 | 14.41 | 10.67 | 10.34 | 14.09 | 17.69 | 11.54 | 13.04 |

- **The night-to-night calibration is worse with PSF on 11 of 13 nights**, and 1106 (the broadest seeing) is worst. This fits
  the seeing-dependent faint-star bias below, which shifts whole nights.
- **Tie warnings:** not converged in 20 iterations (max |dZ| 1.4e-3). NNLS clipped the 1105 floor in bin 1, and the 1104 floor in
  bins 2-3.
- **Detections:** internight 33066, ls_periodic 36332, bls 9289, recurrent 398. The aperture era had 8203 / 9052 / 2253 / 108.
  As before, only `recurrent` sets CLASS.
- **analyze:**
  - 7965 transit events judged; 6328 auto-rejected (coincidence 6252, no baseline 84, no dip 32, edge outlier 6).
  - 16 repeated-event families on 14 objects.
- **Classes now:** EXOP 1817, EXOP+VAR 423, VAR 6041, UNC 194945; 203226 objects in total. The aperture DB had EXOP 435,
  EXOP+VAR 91, VAR 5627.

## Issues found

### 1. Corner registration failure on 20251107 and 20251207

This is the source of 94 % of the 1107 candidates and about 90 % of the 1207 candidates.

- **Where:** in 18-29 frames of 1107 and 23 frames of 1207, the top-left detector corner gets the PSF poor-fit flag (bit 64) for
  43-75 % of its stars. Elsewhere on the detector the rate is 0.2-2 %.
- **Cause (probable, not proven):** the per-frame fit corrects only an affine offset between the PSF frame and the catalogue
  positions (`psfphot._calibrate_shift`). Centroid refinement is capped at 1.2 px (`MAXSHIFT_REF`).
  - The non-affine part of the frame WCS (TAN-SIP) differs by 2.4-4.5 px in that corner between bad and good frames.
  - The bad frames have only 132-154 matched astrometry stars, against 220-330 in good frames.
  - The stars are therefore fitted 1-4 px off, which loses flux in every magnitude bin (corner/rest flux ratio down to 0.25).
  - The aperture runs were immune: their largest apertures absorb such offsets.
- **Effect in relphot:** the comparison pool requires FLAGS == 0 in every present frame. It empties in those tiles: relaxed to the
  5 lowest scores in 6 tiles on 1107, 6 on 1207, and 1 on 1209. The targets then lose real flux in the bad frames on top of a
  noisy ensemble.
- **Mitigation already in place:** the coincidence veto leaves 693 (1107) and 635 (1207) open events, against about 200 and 300
  in the aperture era. The photometry in those corners is still damaged.
- **Milder cases:** 1104 (2 frames, the frames that lacked corner detections in the aperture era), 1130 (2), 1206 (1) and
  1209 (2). The other nights are clean.
- **Figures:** `figures/psf_rerun/fig1_band_location.png` and `figures/psf_rerun/fig2_timeseries.png` (1107).

### 2. Seeing-dependent PSF flux of faint sources (all nights)

- **The effect:** below about 4000 counts (relphot mag fainter than about -9.5), the PSF flux tracks the frame PSF FWHM (r about
  -0.97 on 1104-1107). The flux rises by 25-40 % from the sharpest to the broadest frames at 1500-2500 counts, and is flat above
  about 5000 counts.
- **Per-frame scatter of the median relphot offset:**

  | mag | PSF (mmag) | aperture, same stars (mmag) |
  |---|---|---|
  | -9 to -8.5 | 34 | 1.5 |
  | -8.5 to -8 | 118 | 2.3 |
  | -8 to -7 | 148 | 2.6 |

- **Not the cause (tested by refits):** stamp size, reweighting, centroid refinement, and frame-level PSFEx failures.
- **Probably not the cause:** the static companion list (removing it changes the bias but does not remove it).
- **Best-supported reading (not proven):** faint master entries are often unresolved blends, and how a fixed-position PSF fit
  partitions their flux depends on the seeing.
- **Not corrected by relphot's decorrelation:** the regressor is the catalogue FWHM, not the PSF FWHM, and the magnitude surface
  is fitted on comparison stars brighter than about -9.5.
- **What it explains:** the faint-star variability candidates on every night, much of the larger tie floor, and the faint end
  ("faint stars ~14 % noisier", TOO_DEEP) already noted in the PSF test.
- **Figures:** `figures/psf_rerun/fig3_faint_vs_fwhm.png` and `figures/psf_rerun/fig4_relphot_lc_perframe.png`.

### 3. Non-finite rows in robo43 detection catalogues (fixed)

- The `*_proc_catalog.csv` of 20251118 onwards carry all-NaN rows, 10-51 per pass-1 sample.
- Pass 1 of the PSF fit passed them to a KD-tree query, which raises. 20251118 stopped with NIGHT_FAIL.
- Fix: drop non-finite detections before the query, and make pass-1 frame errors fatal (commit e5bfb24).
- Nights finished before the fix had no pass-1 errors (checked in each `run_fit.log`), so they are unaffected.

## Options (not implemented; decision needed)

1. **Registration fix in the PSF fit**, for issue 1:
   - Replace the affine shift with a smooth per-frame shift field (quadratic or cubic) fitted to refined bright stars; even the
     bad frames have hundreds of them.
   - Alternatively re-solve the astrometry with more matched stars, then redo forced and PSF.
   - Re-fit 1107 and 1207 (about 40 min each), and check the other nights with the corner scan.
   - Expected to remove most of the 1107/1207 floods.
   - The inputs for a re-fit (`forced/`, `*_proc_catalog.csv`) are now in scratch; `PSF_FORCED` and `PSF_CATDIR` point the fit
     there.
2. **relphot guard:** mask (frame, tile) cells whose bit-64 fraction exceeds about 0.15-0.2, and stop one transient bit-64 epoch
   from excluding a star from the comparison pool. Cheap, but it costs those corner frames.
3. **Per-frame, flux-dependent zero point for faint stars**, estimated from thousands of stars at 1000-6000 counts, for issue 2.
   It assumes the common mode also applies to real faint variables.
4. **For the faint end**, use aperture fluxes for faint stars (hybrid). This is a larger change.

## Where things are

| What | Path |
|---|---|
| PSF catalogues, PSFEx models, static companions, fit logs, PROVENANCE.txt | `/mnt/sto01/<TEL>/reduced/<night>/psf/` |
| relphot on PSF (stems `night_lc` T80S, `wasp145_psf` ROBO43; `load_night.log`) | `/mnt/sto01/<TEL>/reduced/<night>/relphot/` |
| Multi-night tie, search, pipeline/analyze logs | `/mnt/sto01/T80S/reduced/multinight/mn_1104_1209_loose_psf*` |
| Run logs, step markers, memory logs | `/mnt/sto01/T80S/reduced/psf_rerun_logs_20261004/` |
| Aperture photometry (robo43 `*_proc_catalog.csv`, `forced/`, old `relphot/`, `relphot_standard_20261002/`, `superseded/` of 1104, photometry/forced logs, old multinight products) | `/mnt/sto01/scratch/aperture/<TEL>/reduced/<night>/` and `.../T80S/reduced/multinight/` (192 GB T80S, 0.7 GB ROBO43) |
| DB backups (pinned) | `/ssdsto1/data/relphotDB/backups/pre_psf_wipe_20261003_2322.dump`, `post_psf_rerun_20261004_0814.dump` |

- The aperture catalogue embedded in each `*_proc.fits` (HDU `CATALOG`) stays; removing it would mean rewriting 969 GB.
- The tie scripts (`run_*.sh`, `analyze_retry.sh`, `drop_mn_run.py`) stay in `multinight/`.
- `psf_night.sh` refuses to run on a night whose `psf/` or `relphot/` exists. A re-fit (option 1) needs those moved aside first,
  and `PSF_FORCED`/`PSF_CATDIR` set to the scratch inputs.

## Not verified

- No web page was clicked in a browser; only API calls were made.
- The registration cause of issue 1 is probable, not proven. The blend mechanism of issue 2 is not proven.
- Injection-recovery on the PSF fluxes was not done.
- The coincidence veto was applied by `analyze`, but its effect on real transits in the damaged corners was not measured.
- The `test_psf` schema (PSF test, 2 nights) is still in the production database; dropping it is your decision.
