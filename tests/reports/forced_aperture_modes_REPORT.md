> Report dated 2026-10-02; source: scratch `<scratchpad>/fwhmap` (scratch paths and the data/tables named below may no longer exist).
> Figure `floor_vs_mode.png` (floors by mode) is not referenced in the text and was left out; the mode code itself was not merged.

# FWHM-scaled apertures in the forced step (T80S 20251107, 20251207)

All numbers: same relphot code (main 355b8e4 + forced-ingest patch + nanmedian speed patch), same master star list per night,
same-star sets, matched to production within 0.7". Scratch: `.../scratchpad/fwhmap/` (data on /var/tmp/fwhmap_data, /tmp tmpfs was full).

## Modes (robo43 `forced.aperture_mode`)
- `fixed` (default, unchanged): radii 2/3/4/6/8" on every frame.
- `fwhm`: radius_j = k_j * FWHM_frame, k = 0.6/0.9/1.2/1.75/2.35. FWHM_frame = CATALOG `FWHMARC` of the frame
  (flux-weighted second-moment, Gaussian-equivalent FWHM, 2.3548*median(sigma) of the brightest stars, 17x17 px stamps).
  k_j chosen = r_j / median FWHM (3.44") so that radii equal the fixed set at median seeing (2.06/3.1/4.1/6.0/8.1"):
  the comparison then isolates seeing-compensation, not a different radius set. Frame FWHM: 20251107 3.44" (std 6.0 %, 3.07-3.85),
  20251207 3.49" (std 5.4 %, 3.09-3.78; the "8 %" is relphot's per-source median FWHM in px).
- `hybrid`: radius_j = r_j * FWHM_frame / FWHM_ref, FWHM_ref = median FWHM of the 3 centroid frames (3.07"/3.09"), i.e. radii 12 % larger than fixed at median seeing.
  (hybrid == fwhm with k_j = r_j / FWHM_ref; kept as its own mode as asked.)
- Sky annulus code unchanged (smoothed annulus, starts 1" beyond the frame's own largest aperture).
- Not run: the example set k = 0.6/0.8/1.0/1.3/1.8 (ap0 identical to the tested k = 0.6, which already loses 25-28 % at ap0 vs fixed).

## Two defects found on the way (not caused by the aperture modes)
1. 20251207 frames idx 40 and 48 (TOO-20251208-045719, -051502) have unsolved astrometry (ASTRRMS NaN). `forced` projected the
   master through the missing/wrong WCS and relphot kept them (forced catalogues have ~constant source counts, so relphot's frame
   rejection never fires): bright-star flux ratio 0.04-0.13 at those epochs, comparison ensemble 1.13 there. First-pass 20251207
   forced results were corrupted; final numbers below drop those two frames (`subset_night.py` on night.npz, then
   reference/lightcurves/search rerun). Fix delivered separately: `fwhmap_skip_unsolved_robo43.patch` (`forced.skip_unsolved`, default true).
   20251107: all 67 frames solved, unaffected.
2. Production relphot rejects frames (prod keeps 60/67 and 57/69); forced keeps all. All RMS below are therefore recomputed
   (MAD-sigma of lc/median, relphot definition) on the **common frames** = production-kept and astrometry-solved, for every mode.
   relphot's own-frame RMS (column `floor_same_star_ownframes_mmag` in floor_by_mode.csv) gives the same ordering, forced +0.1-0.9 mmag higher.

## 1. Same-star bright floor [mmag] (production's own bright bin, comparison stars in all four modes; N = 179 / 149 stars)
| night | mode | ap0 | ap1 | ap2 | ap3 | ap4 |
|---|---|---|---|---|---|---|
| 20251107 | production | 4.66 | 5.42 | 6.45 | 9.11 | 12.91 |
| | forced fixed | 5.04 | 5.54 | 6.75 | 9.56 | 11.64 |
| | forced FWHM-scaled | 6.29 | 6.71 | 10.66 | 26.22 | 36.47 |
| | forced hybrid | 6.34 | 7.58 | 12.65 | 32.04 | 51.97 |
| 20251207 | production | 4.91 | 6.04 | 7.09 | 10.20 | 13.67 |
| | forced fixed | 5.41 | 6.10 | 7.36 | 10.71 | 12.54 |
| | forced FWHM-scaled | 6.89 | 7.67 | 11.81 | 27.62 | 47.59 |
| | forced hybrid | 6.93 | 8.19 | 13.28 | 34.07 | 58.89 |

(The 5.2/5.6/7.0/9.7/12.2 of the earlier 20251107 test is the same fixed run on relphot's own 67 frames; on the common 60 frames the ap0 loss vs production is 8 %, not 11 %.)

## 2. RMS vs magnitude (ap0, median over same stars, common frames; bright = floor bin [lo,lo+1), mid [lo+1,lo+2.5), faint [lo+2.5,lo+4.5))
20251107 all/blended/isolated: bright prod 5.00/5.14/4.69, fixed 5.18/5.25/5.07, fwhm 6.68/6.80/6.40, hybrid 6.67/6.81/6.40;
mid prod 9.68, fixed 9.47, fwhm 11.00, hybrid 11.11; faint prod 36.1, fixed 33.5, fwhm 39.3, hybrid 40.2.
20251207 bright prod 5.42, fixed 5.69, fwhm 7.68, hybrid 7.71; mid 9.62/9.53/11.14/11.14; faint 34.4/31.9/37.3/37.9.
Fixed radii beat production for mid and faint stars (-2 % / -7 to -8 %), lose 3-5 % on bright; blended and isolated behave alike.
Scaled modes are worse than fixed in every bin (+16-35 % bright/mid, +10-17 % faint at ap0). At the per-star best aperture (min over the 5), faint: prod 35.6/33.6, fixed 31.0/29.1, fwhm 35.0/32.8.
Full tables: rms_bins_by_mode.csv.

## 3. Seeing term (bright comparison floor-bin stars; residual lc/median-1 vs fractional frame FWHM; common frames)
median R^2 of a per-star linear fit on FWHM (pure noise expectation 0.008), ap0 / ap2 / ap4:
- raw light curve before relphot's decorrelation, 20251107: prod 0.28/0.47/0.07, fixed 0.12/0.16/0.20, fwhm 0.38/0.53/0.95, hybrid 0.37/0.65/0.95.
- after decorrelation (`lc`), 20251107: prod 0.04/0.08/0.05, fixed 0.03/0.03/0.02, fwhm 0.06/0.27/0.23, hybrid 0.06/0.29/0.64; 20251207: prod 0.05/0.06/0.05, fixed 0.05/0.04/0.02, fwhm 0.05/0.37/0.31, hybrid 0.04/0.40/0.67.
- median slope of the bright stars relative to the ensemble (raw, mmag per +10 % FWHM), ap0/ap2/ap4: prod +5/+14/+7, fixed +3/+6/+13, fwhm -7/-14/-130, hybrid -7/-20/-136 (same on 20251207).
So scaling does not remove the seeing term, it creates one: a common-sign (all bright stars move against the faint-star-dominated ensemble)
term that grows with aperture area. Fixed radii already have the smallest seeing coupling (below production's even before decorrelation).
Probable mechanism (consistent with the evidence, not isolated): an additive per-pixel sky/neighbour bias times aperture area; at constant
area it is a constant offset, with seeing-scaled area it varies with seeing and differs between bright stars and the faint ensemble.
Per-star slope is not tied to neighbour count within 8" (|rho| <= 0.2; comparison stars have none within 4").

## 4. Best aperture (share of stars, same stars; relphot best_aperture / argmin RMS)
ap0 share (best_aperture): 20251107 prod 99.9 %, fixed 98.0, fwhm 98.7, hybrid 99.3; 20251207 99.6/98.5/98.1/99.4. Remaining weight: ap1 (1-4 %), fixed also ap4 (0.04-0.15 %).
argmin RMS: ap0 64-84 % in all modes (prod 84/82 %, fixed 67/64, fwhm 68/66, hybrid 69/67); fixed shifts 12 % of stars to ap4 (prod 0.05 %, fwhm 4.5-5 %).

## 5. Flip events (20251107, the 11 labelled night-15 events; 10 measurable, production |S| >= 9)
| | still a candidate | mean S at the epoch (10 measurable) | events with S >= 5 |
|---|---|---|---|
| production | 9 of 11 | 26.0 | 10 of 10 |
| forced fixed | 0 | 0.6 | 0 |
| forced FWHM-scaled | 0 | -1.3 | 0 |
| forced hybrid | 0 | -1.7 | 0 |
Flips are removed by any forced mode (it is the per-frame centroiding, not the aperture size). Scaled modes add a spurious brightening at det 8323 (S = -8.6/-9.1, not a transit candidate).
Production candidates (common stars) seen again in the forced light curve at the same epoch (S >= 5 / measurable): 20251107 fixed 36/70, fwhm 35/70, hybrid 34/70 (tier-1: 11-13 of 16 each, at most 1 vanishes); 20251207 156/230, 139/230, 141/230 (tier-1 54/57, 51/57, 52/57).

## 6. Candidate counts and R90 (all stars searched / common star set)
| night | mode | stars searched | candidates all | tier-1 | R90 pass all | common-star cand. | R90 pass common |
|---|---|---|---|---|---|---|---|
| 20251107 | production | 104035 | 93 | 16 | 33 (35 %) | 93 | 35 % |
| | fixed | 189500 | 201 | 26 | 118 (59 %) | 106 | 57 % |
| | fwhm | 189500 | 221 | 29 | 110 (50 %) | 123 | 42 % |
| | hybrid | 189500 | 223 | 33 | 102 (46 %) | 133 | 36 % |
| 20251207 | production | 112258 | 300 | 57 | 151 (50 %) | 289 | 51 % |
| | fixed | 194788 | 417 | 75 | 287 (69 %) | 241 | 72 % |
| | fwhm | 194788 | 520 | 80 | 300 (58 %) | 311 | 58 % |
| | hybrid | 194788 | 492 | 62 | 277 (56 %) | 302 | 55 % |
Fixed radii give the highest R90 pass fraction and fewest common-star candidates on 20251207 (241 vs 311/302); on 20251107 fixed also has the fewest of the forced modes.
Candidates flagged APERTURE_INCONSISTENT (common stars): 36/35/46 (1107 fixed/fwhm/hybrid), 80/108/110 (1207).

## 7. Runtime / memory
Forced, one frame, one worker, idle machine (20251107, --master given): fixed 42.5 s, fwhm 39.2 s, hybrid 38.7 s (no cost difference).
Peak RSS per worker: fixed 5.3 GB, fwhm 5.7, hybrid 6.0-6.1 GB (larger max aperture -> larger annulus/masks). Whole-night walls (np 4-5, machine shared with a 12-worker job, not comparable): 864-1503 s.
relphot (1.9e5 stars, threads capped at 2, shared machine): ingest 1.5-2.7 min, reference 15-26 s, lightcurves 7-17 min, search 8-15 min, independent of the aperture mode. NB: unthrottled BLAS threads on the shared machine made the first relphot search jobs ~10x slower (OMP_NUM_THREADS=2 fixed it).

## Recommendation
Keep `aperture_mode = "fixed"`. Neither FWHM-scaled nor hybrid radii improve any aperture index on either night: ap0 (98-99 % of stars) +25-28 % worse floor than fixed (+35-40 % than production),
ap2 1.6-1.9x, ap3-ap4 2.7-4.7x worse, seeing correlation of the residuals larger instead of smaller, R90 pass fraction lower, flips not removed any better than fixed.
With radii fixed in the sky the residual seeing term is already the smallest of all modes (median R^2 after decorrelation 0.02-0.05, noise 0.008). The remaining ap0 gap to production is 8-10 % (4.66->5.04 and 4.91->5.41 mmag), while forced fixed is 7-8 % better on mid/faint stars, 8-10 % better at ap4 and removes the 10 measurable flips.
The plan sentence about FWHM variations is satisfied by *not* changing the area: in a crowded field contamination scales with aperture area, not with enclosed flux fraction.
Untested: a constant-area aperture with a seeing-dependent aperture correction (PSF/curve-of-growth), PSF-matched convolution, smaller k sets, other fields (sparse), k exponent < 1.
Merge: the mode code (`fwhmap_robo43.patch`) is optional/diagnostic, default unchanged; the unsolved-frame skip (`fwhmap_skip_unsolved_robo43.patch`) is a bug fix worth merging.
