> Report dated 2026-10-02; source: scratch `<scratchpad>/final/` (forced runs in `/var/tmp/final_data/`; scratch paths and data may no longer exist).
> Numbers: relphot main 355b8e4 + forced-ingest + nanmedian speed + frame-quality cut, robo43 `forced-phot` + skip_unsolved; 2 threads per relphot job.

# Forced (fixed-centroid) photometry: build test, frame selection and the frame-quality cut

Forced photometry = `robo43 forced` (apertures at fixed sky positions from the best frames' master, projected through each frame's WCS) read by
`relphot ingest --photometry forced`. Merged together with `forced.skip_unsolved` and the relphot frame-quality cut. The FWHM-scaled aperture
modes were tested and **not** merged (`forced_aperture_modes_REPORT.md`: fixed radii are better on every aperture index).

## 1. Why
Per-frame centroids make blended sources flip between two extraction levels (the "two-level" false transits). Fixed positions remove them:
20251107, the 11 labelled two-step events of night 15: 9 of 11 are production transit candidates (mean |S| 26 at the epoch), 0 of 11 are forced candidates
(box depth 2.5-10 % -> |S| < 4). More stars survive (189.5k vs 104k) because a forced catalogue has every master source in every frame.

## 2. Centroid frames chosen (score = n_useful/max * (FWHM_min/FWHM)^2, transparency >= 0.9, astrometry solved, top 3)
| night | frames | centroid frames: FWHM [arcsec] / transparency / n_useful | master | note |
|---|---|---|---|---|
| T80S 20251107 | 67 | 055308 3.07 / 1.004 / 92.6k; 055735 3.07 / 0.999 / 90.0k; 060900 3.11 / 1.000 / 90.0k | 189,810 | clear night (t 0.96-1.00): the best-seeing set |
| T80S 20251112 | 27 | 060448 3.60 / 0.904 / 86.1k; 060026 3.57 / 0.928 / 86.1k; 054309 3.57 / 0.926 / 87.2k | 165,799 | transparency cut swaps out a t = 0.76 frame (seeing-only would take 060237) |
| T80S 20251207 | 69 | 032006 3.20 / 0.990 / 102.8k; 051252 3.12 / 0.991 / 96.9k; 055517 3.14 / 0.997 / 98.4k | 195,803 | star count swaps out 053250; 2 frames unsolved (below) |
| ROBO43 20250911 | 351 | 0148 2.47 / 0.931 / 339; 0161 2.53 / 0.933 / 336; 0175 2.55 / 0.957 / 339 | 686 | 273/351 eligible (78 below t 0.9: the cloud at the end); seeing-only would take 0148/0156/0167 |

## 3. Two defects found while testing, both fixed in this merge
1. **Unsolved astrometry.** 20251207 frames TOO-20251208-045719 and -051502 have `ASTRSOLV` false and `ASTRRMS` missing. Sky-fixed apertures through a missing WCS land off the
   stars (flux ratios 0.04-0.13 against the other frames) and relphot kept them. Fix: `forced.skip_unsolved` (default true) leaves such frames out (error row in `forced_status.csv`, no
   catalogue); also applies to `ASTRRMS` above `astrometry_rms_max_arcsec` and to missing `ASTRSOLV`.
2. **relphot rejects no frames with forced photometry.** Its only frame rejection is the fixed-star-set rule (a tile running short of reference stars); a forced catalogue has constant source
   counts, so the rule never fires. Production drops 7 frames on 20251107 and 12 on 20251207 (incl. the 2 unsolved); forced dropped none, and its floors on relphot's own frames
   were worse. Fix: the frame-quality cut below.

## 4. The frame-quality cut (`relphot reference`, `reference.frame_quality`)
Ensemble: the (<= 4000) brightest reference candidates with FLAGS == 0 and a finite positive flux at the reference aperture in >= 90 % of the frames. Per frame:
`scatter` (MAD sigma over the ensemble of flux / star median / frame transparency), `transparency` (ensemble median relative flux over its robust quadratic trend in time),
`fwhm` and `sky` (ensemble medians of the FWHM and BACKGROUND columns). A frame is flagged when the one-sided robust z (median/MAD over the frames; scatter, FWHM, sky high, transparency low)
exceeds `frame_quality_sigma` = 3.0 **and** the metric deviates by >= `frame_quality_min_excess` = 5 % (scatter/FWHM above the median, flux below the trend; none for sky).
Flagged frames are cut worst-z first, at most `frame_quality_max_fraction` = 15 % of the night and never beyond `max_dropped_frame_fraction` (30 %) together with the star-set rule
(whose own loop then runs on top; `ReferenceResult.frame_kept` stays authoritative). Nights under 10 frames are not cut. Every flagged frame is logged with its reasons and
`<stem>_frames.csv` has all metrics, z-scores, flag, cut and kept (written in every mode). `frame_quality = "auto"` (default) applies the cut to forced photometry only; `"on"`/`"off"` force it.

Choice of the sigma: floors of the bright comparison bin against the number of frames cut (frame-level cut thresholds scanned on the forced light curves):
20251107 gains 0.12 mmag at sigma 3.0 (2 frames), 0 at >= 3.5 (nothing flagged); ROBO43 plateaus at sigma <= 3.5 (9.89-9.91 mmag vs 10.68 uncut); a clear T80S night flags 1-3 of 67 frames.
Recall against the frames that are noisy in the light curves themselves (bright comparison stars, frame noise z > 3.5, scan at sigma 3.5): ROBO43 25 of 26 caught, 2 of 34 flagged quiet; T80S nights have
(almost) no such frames (0 on 20251107, 1 on 20251207). The metric correlates with the frame's light-curve noise at rho = 0.8-0.85 on 20251107/20251207 (4000-star ensemble; 0.5 with 300 stars), so the ensemble size matters.
Not separable: the 10 further frames production drops on 20251207 (z_scatter 0.2-2.6 there): they are mildly noisy, not outliers; restricting the forced 20251207 light curves to production's kept frames
lowers its floor by 0.3 mmag (5.75 -> 5.44), but that is trading data for scatter (neutral for transit S/N), not something an outlier rule should do.

## 5. Evaluation (same code everywhere: relphot main 355b8e4 + this patch; "production" = catalogue mode re-run with this code, which reproduces the stored frame sets exactly)
Same-star bright floor [mmag] (comparison stars in the bin of the production floor in every variant, matched within 0.7 arcsec; each variant on its own kept frames):
| night | variant | frames kept | ap0 | ap1 | ap2 | ap3 | ap4 | stars |
|---|---|---|---|---|---|---|---|---|
| T80S 20251107 | production (catalogue) | 60/67 | 4.69 | 5.52 | 6.59 | 9.21 | 13.67 | 202-213 |
| | forced, no cut | 67/67 | 5.22 | 5.64 | 7.00 | 9.72 | 12.19 | |
| | forced + frame-quality cut | 65/67 | 5.13 | 5.53 | 6.75 | 9.29 | 12.07 | |
| T80S 20251207 | production (catalogue) | 57/69 | 4.92 | 6.12 | 7.17 | 10.35 | 13.87 | 154-173 |
| | forced, no cut (2 unsolved skipped) | 67/67 | 5.52 | 6.35 | 8.03 | 11.52 | 13.59 | |
| | forced + frame-quality cut | 66/67 | 5.53 | 6.28 | 7.79 | 11.49 | 13.42 | |
| ROBO43 20250911 | production (catalogue) | 351/351 | 10.50 | 10.58 | 12.34 | 17.14 | 18.56 | 34-45 |
| | forced, no cut | 351/351 | 10.68 | 10.44 | 11.13 | 13.80 | 15.35 | |
| | forced + frame-quality cut | 310/351 | 9.85 | 9.86 | 10.79 | 12.96 | 14.34 | |

The cut closes the 20251107 gap to production by 0.1-0.35 mmag (ap0 0.53 -> 0.44, ap1 0.12 -> 0.01, ap2 0.41 -> 0.16, ap3 0.51 -> 0.08), by 0-0.24 on 20251207 (it flags one frame; the remaining
+0.6 mmag at ap0 and +1.1 at ap3 is the seeing-correlated noise of the photometry itself, `forced_aperture_modes_REPORT.md`) and by 0.8-1.7 mmag on ROBO43 (cloud; forced beats production there by 0.6-4 mmag).
A lower sigma trades frames for floor on the homogeneous T80S nights (forced light curves, no refit): sigma 2.0 cuts 5 / 8 of 67 frames on 20251107 / 20251207 and lowers the floor by 4 % / 5 %, sigma 2.5 cuts 4 / 3 for 2 % / 1.5 %;
at constant exposure time that is neutral for transit S/N, so it is left as a setting (`frame_quality_sigma`), not the default.

Transit search (relphot search, default threshold 6 on T80S):
| night | variant | candidates | tier 1 | R90 pass |
|---|---|---|---|---|
| 20251107 | production (catalogue) | 93 | 16 | 33 (35 %) |
| | forced, no cut | 201 | 26 | 118 (59 %) |
| | forced + cut | 206 | 20 | 111 (54 %) |
| 20251207 | production (catalogue) | 300 | 57 | 150 (50 %) |
| | forced, no cut | 417 | 73 | 284 (68 %) |
| | forced + cut | 404 | 62 | 273 (68 %) |
| ROBO43 (threshold 5.5) | production (catalogue) | 4 | 2 | 3 |
| | forced, no cut | 8 | 3 | 5 |
| | forced + cut | 6 | 2 | 5 |

20251207: 364 of the 404 forced+cut candidates are also in the no-cut list; the cut does not change the candidate census, it removes the marginal ones from the worst frames.

Frames cut by the quality rule (not by the star-set rule):
| night | cut by the quality rule | production's star-set rule dropped | overlap |
|---|---|---|---|
| 20251107 | 2: -051142 (FWHM z 3.1), -054813 (scatter z 3.2) | 7: idx 5, 12, 29, 30, 47, 50, 65 | 1 (054813 = idx 50) |
| 20251207 | 1: -054138 (scatter z 3.1) | 12: idx 4, 9, 30, 31, 40, 48, 50, 58, 60, 62, 63, 64 (40, 48 are the unsolved frames, skipped by `forced`) | 1 (054138 = idx 58) |
| ROBO43 20250911 | 41 (11.7 %): 322-350 (the cloud, transparency 0.93 -> 0.40) and 315, 320; mid-night 18, 30, 60, 62, 63, 123, 198-200, 213 (transparency 0.78-0.84 / FWHM / scatter) | 0 | - |

WASP-145 A b transit window (BJD 2460930.6480 +- 0.5 T14, T14 = 0.98 h, 111 frames): 4 frames cut (198, 199, 200, 213: transparency 0.78-0.80, scatter z up to 10), 3.6 %; none of the other 107.

### Catalogue (production) mode
`frame_quality = "auto"` leaves it byte-identical (frame_kept equal to `off` on 20251107, 20251207, ROBO43; equal to the stored production frame sets 7 / 12 / 0 dropped).
With `"on"` (tested for information): 20251107 13 frames dropped instead of 7 (+ idx 1, 34, 37, 46, 52, 59: some from the quality rule, the rest from the star-set loop, whose greedy path changes once a frame is gone), same-star floors
4.70/5.53/6.59/9.20/13.55 -> 4.61/5.50/6.34/9.05/13.20 (-2 to -4 %), candidates 93 -> 99 with only 65 in common, tier 1 16 -> 16, R90 pass 33 -> 30.
20251207 identical (the only flagged frames are the two unsolved ones, z ~ 1000, which production drops anyway). ROBO43 standard photometry: 40 frames cut, floors 10.50/10.58/12.34/17.14/18.56 -> 9.74/9.98/11.49/15.53/17.66,
WASP-145 A b SNR 8.1 -> 12.6 with its depth 0.98 -> 2.13 % and a new R90_SYSTEMATICS flag. So `"on"` changes catalogue-mode candidate lists materially; the default does not.

## 6. ROBO43 20250911 (351 frames, 30 s, R; forced run into scratch)
- `robo43 forced` needed no change for ROBO43 headers: CATALOG radii (2/3/4/6/8 arcsec) read from the frames' own catalogues, `ASTRSOLV` true and `ASTRRMS` 0.22-0.29 arcsec on all 351, gain handled
  (forced flux / standard flux of matched sources: median 0.9997, FLUXERR the same). 351/351 measured, 686 master sources (520 stars survive the standard-photometry presence cut; 652 forced).
- Runtime 1298 s for the whole command (serial: `runtime.nprocs` defaults to 1 and `--np`/`--instrument` placed before the subcommand are ignored; give options after `forced`), 3.6 s per frame (median),
  peak RSS 1.0 GB. T80S 9k x 9k frames: ~47-50 s and 5.3 GB per frame and worker.
- relphot (reference aperture 0, variables on, search threshold 5.5):
  | | production (catalogue, 351 frames) | forced, no cut | forced + cut (310 frames) |
  |---|---|---|---|
  | WASP-145 A b (star 369 forced / 516 standard) | SNR 8.1, depth 0.98 %, box 0.40 h, tier 2, APERTURE_INCONSISTENT + NEIGHBOUR_BLEND | SNR 8.97, 1.25 %, 0.40 h, tier 2, SHARED_EPOCH + NEIGHBOUR_BLEND, R90 pass | SNR 9.42, 1.37 %, 0.40 h, tier 2, SHARED_EPOCH + NEIGHBOUR_BLEND, R90 pass |
  | candidates / tier 1 / R90 pass | 4 / 2 / 3 | 8 / 3 / 5 | 6 / 2 / 5 |
  Forced + cut candidate list (star, SNR, depth, box T14, tier, flags, R90): 448 9.2 3.6 % 0.70 h t1 ON_VARIABLE (SSS J212942.9-584538) pass; 27 5.7 2.2 % 0.55 h t1 OK pass;
  272 14.0 1.9 % 1.00 h t2 SHARED_EPOCH + APERTURE_INCONSISTENT + R90_SYSTEMATICS **fail**; 369 = WASP-145 A b pass; 367 6.8 5.2 % 0.55 h t2 SHARED_EPOCH + APERTURE_INCONSISTENT + NEIGHBOUR_BLEND + ON_VARIABLE pass;
  229 6.8 1.1 % 2.2 h t3 SHARED_EPOCH + EDGE + PARTIAL + ON_VARIABLE pass. The two candidates the cut removes are 317 (5.8, 5.9 %, 0.40 h, NEIGHBOUR_BLEND + R90_SINGLE_POINT) and 135 (5.9, 1.4 %, SHARED_EPOCH + APERTURE_INCONSISTENT + NEIGHBOUR_BLEND).

## 7. Runtime and memory (T80S 20251107, 189.5k stars, 2 threads)
`forced` 47-50 s/frame/worker, 5.3 GB RSS per worker (`STAGE_BYTES_PER_PIXEL["forced"]` = 72); relphot ingest 75 s, reference 20-25 s, lightcurves ~9 min, search ~8 min. The quality cut is part of the 2 s frame/star selection step of `reference`.

## 8. Caveats
- The `sky` metric uses the catalogues' BACKGROUND column, which is the residual local sky after the 2-D sky model (about 0 +- 0.4 ADU): it never flagged a frame in these tests and is mostly a guard.
- Production's `ASTRRMS` is not read by relphot; unsolved frames are handled in `robo43 forced`. For standard photometry nothing changes.
- Candidate sets are sensitive to the frame set: dropping or adding a few frames reshuffles 30 % of the marginal candidates (20251107 catalogue mode, `"on"` vs `"off"`: 65 of 93 in common).
