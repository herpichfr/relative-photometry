> Report dated 2026-10-02; source: scratch `<scratchpad>/lookalike` (scratch paths and the data/tables named below may no longer exist).
> Figures referenced below are copied to `figures/` as `lookalike_<name>.png`.

# Look-alike light-curve detrending vs production search (read-only test)

Scratch: this directory (scripts: lk.py core/similarity, meth.py methods M0-M5 + controls, run_night.py driver, analyze.py tables, groups.py, figs_*.py, timing*.py;
per-night raw results out/*.pkl; large tables cache/). Nothing was written to the repo, worktree or DB. Repo git status was clean before and after.

## 0. Setup
- Nights (T80S, ~100 k stars each, 57-73 kept frames): 20251107 (night_id 15; 102 740 searched stars, 60 frames), 20251207 (22; 111 109, 57), 20251105 (2; 59 451, 73); ROBO43 20250911 (6; 516 stars, 351 frames; WASP-145 A b).
- Events = the DB search events on those nights: REJECTED 636 (379 user-only + 257 also coincidence-vetoed), UNCONFIRMED 223, WASP-145 A b 1 (+3 UNC on night 6). All 859 production candidates of nights 15/22/2 (97/300/462) are reproduced exactly (snr, tc) by `search_one_star`.
- Injections: the SAME injection sets as detect_plan (inj_*.pkl: 10 000 per T80S night, 750 ROBO43), regenerated from the stored parameters (ld/trapezoid, depth 0.5-5 %, T14 0.5-3 h, tc uniform). M0 re-run recovers 924/924 on night 15 (100 % identical to the stored result). Pooled 30 000 injections, M0 recovers 2 813 (9.4 %).
- "Candidate" = snr >= 6 and none of STEP_LIKE/TOO_DEEP/FEW_POINTS (as production). "Recovered/kept" = candidate with |tc - tc_ref| <= max(T14, found dur)/2 + 0.01 d. R90 = `search_one_star(regressors=...)` R90 bits (production thresholds), computed on whatever nuisance model the method used. Population-level flags (SHARED_EPOCH, coincidence veto, ON_VARIABLE, cross-aperture, NEIGHBOUR_BLEND) are NOT applied anywhere, so candidate counts are the raw search candidates.
- Every method was run on: the labelled events (A view), all 30 000 injections, and ALL searched stars of each night (B view: candidate counts, new candidates). SNR threshold scanned 5-12; R90 on/off.
- Pairwise similarity: blocked float32 matrix products (512-row blocks x whole pool, BLAS) + argpartition top-48, no Python pair loops. Pool = all searched stars of the night (whole detector, not per tile). Spatial exclusion (<= 10") only in M2/M3, self always excluded.

## 1. Do the look-alikes exist? (lookalike_groups.csv, fig_group_stats.png)
- Normalised (noise-weighted, centred) correlation of the best neighbour among ~1e5 stars with 57-73 frames is dominated by chance: median best |corr| = 0.51-0.60 (null from frame-permuted queries 0.50-0.55); 99 % of stars have a neighbour with |corr| >= 0.5 on nights 15/22 (57 % on the 73-frame night 2).
  Stars with best |corr| >= 0.65: 12.0 / 14.2 / 10.5 % (nights 15/22/2) vs null 5.0 / 4.2 / 0.6 %; >= 0.8: 3.8 / 2.9 / 2.8 % (null 2.5 / 0.9 / 0.1 %). Real look-alikes exist for ~5-10 % of stars, mostly bright (mag < -11: 11-18 % have |corr| >= 0.7 vs 1-4 % at mag -9.5..-9).
- M1 (user's literal least-RMS rule): star j is a look-alike of i if rms(R_i - R_j) <= f * rms(R_i), searched in the 30 least-RMS stars. "No look-alike" threshold used: f = 0.7 (primary), also 0.5 / 0.85 / 1.0.
  Fraction of stars with >= 1 look-alike (null in brackets): f=0.5: 1.6 / 0.7 / 0.8 % (1.0 / 0.2 / 0.0); f=0.7: 5.1 / 4.7 / 5.0 % (2.8 / 1.2 / 0.1); f=0.85: 28.9 / 40.5 / 18.3 % (16.5 / 21.3 / 0.9); f=1.0: ~100 % (degenerate: the 30 quietest-difference stars always pass).
  Group size at f=0.7 (given >= 1): median 9 / 5 / 11, p90 = 30 (the cap of the top-30 list); >= 3 members for 3.8 / 3.0 / 3.7 % of stars. Groups containing a star within 10": <= 0.2 % (never for REJECTED events).
- REJECTED events are strongly enriched: 82 % have a f=0.7 group (70 % >= 3 members, median size 20) and 96 % have best |corr| >= 0.65, vs 5 % / ~10 % of all stars, 18 % / 40 % of the injected (bright) stars. BUT 85 % of the M0-recovered injections also have best |corr| >= 0.65: the recovered injections live in the same look-alike-rich stars (systematics help an injected dip reach SNR 6), so look-alike membership alone does not separate FPs from signal.
- ROBO43 (516 stars, 351 frames): essentially no look-alikes (|corr| >= 0.5 for 0.4 % of stars; no M1 group at f <= 0.85) -> the method is inapplicable/identical to M0 there.

## 2. Results (pooled nights 15+22+2; method_comparison.csv, roc_points_all.csv, matched_*.csv)
Reference M0 (production) + R90: removes 56.1 % of REJECTED (50.7 % user-only) and 58.3 % of UNCONFIRMED at 93.2 % retention of M0-recovered injections; 372 of 859 candidates pass R90; absolute injected recall 9.4 % (no R90) / 8.7 % (R90).
Pure SNR-threshold on M0 (the "do nothing smart" curve): thr 6.5: 28 % REJ removed @ 82 % retention; thr 7: 49 % @ 66 %; thr 8: 73 % @ 45 %.

| variant (thr 6, no R90) | REJ removed | UNC removed | retention of M0-recovered | abs recall | candidates (M0 859) | new cands (R90 pass) | median depth ratio (M0 0.81) |
|---|---|---|---|---|---|---|---|
| M1 f0.5 ex | 25 % | 27 % | 0.90 | 8.5 % | 648 | 10 (8) | 0.80 |
| M1 f0.7 ex (literal, primary) | 77 % | 74 % | 0.69 | 6.8 % | 243 | 35 (16) | 0.79 |
| M1 f0.7 ex + tile CBV | 79 % | 76 % | 0.66 | 6.3 % | 193 | 8 (4) | - |
| M1 f0.85 ex | 88 % | 84 % | 0.61 | 6.4 % | 189 | 79 (26) | 0.65 |
| M1 f1.0 ex (= mean of 30 nearest-RMS stars) | 87 % | 85 % | 0.67 | 7.2 % | 214 | 100 (32) | 0.65 |
| M1 "in" (self in own stack) | same as ex (+-0.01) | | 0.60-0.69 | | | | lower by 0.01-0.06 |
| M2 unmasked K=2/4/6, c0=0 | 89/92/93 % | 81/87/88 % | 0.41/0.31/0.27 | 4.2/3.2/2.8 % | 155/115/102 | 40/32/28 | 0.70/0.69/0.68 |
| M2 unmasked K=2, |corr|>=0.65 | 86 % | 76 % | 0.50 | 5.1 % | 175 | 29 | 0.75 |
| M2m masked 2-stage K=2/4/6 (tpl) | 49/59/64 % | 50/62/66 % | 0.80/0.72/0.66 | 9.5/8.5/7.8 % | 840/699/647 | 406/351/342 | 0.81/0.80/0.80 |
| M4 global wPCA k=3/5/8 | 68/84/88 % | 67/74/78 % | 0.63/0.52/0.45 | 6.3/5.3/4.5 % | 500/221/198 | 221/64/71 | 0.76/0.75/0.75 |
| M4c cluster-stack n=50 / 200 | 75/80 % | 66/67 % | 0.74/0.71 | 7.9/7.4 % | 305/258 | 71/55 | 0.77/0.75 |
| M5 CBV + FWHM/bkg/x/y | 80 % | 76 % | 0.70 | 7.3 % | 330 | 150 (104)* | 0.79 |
| M5 regressors only | 72 % | 67 % | 0.76 | 8.8 % | 750 | 495 (374)* | 0.80 |
| control: 6 / 10 per-tile CBVs | 33 / 48 % | 30 / 44 % | 0.88 / 0.79 | 8.5 / 7.7 % | 640 / 520 | 57 / 64 | 0.81 / 0.80 |
| control: M1 with RANDOM group of same size | 25 % | 24 % | 0.69 | 7.0 % | 1228 | 583 | 0.88 |
| control: M2 with K RANDOM templates | 11-17 % | 14-19 % | 0.90-0.94 | 11-12.5 % | 1591-1914 | 880-1156 | 0.85 |
(*) the R90 regressor test is not independent of M5 (those regressors are already in the nuisance model), so M5's R90-pass counts are inflated.

M3 veto (K most-correlated stars, window masked +-0.75 T14, self+10" excluded, stack fitted to the star out of window, box fitted to the stack at the same tc; flag if stack dip >= S1 x sigma_cand or >= q x candidate depth): at K=10, S1>=3: 62 % REJ removed @ 0.47 retention; q>=0.5: 55 % @ 0.55; q>=0.75: 39 % @ 0.68 -> on the SNR-threshold curve, not above it; beyond R90 (K=10, q>=0.5): 61 % of R90-surviving REJ @ 0.56 retention of R90-surviving injections vs 56 % @ 0.67 for plain SNR>=7. 54 % of M0-recovered injections have a stack dip >= 3 sigma_cand (recovered injections borrow SNR from shared systematics).

### Matched-recall ROC (retention = fraction of M0-recovered injections kept; roc_matched_retention.csv, fig_roc.png A)
- >= 95 % retention: nothing removes more than ~20 % of REJ (best: control cbv10 / random templates). >= 90 %: R90 alone 56 % (93 %); best look-alike variant M1 f0.5 25 % (control cbv10 31 %).
- >= 80 %: M1 f1.0 (thr 5) 71 % @ 0.83; M4c n50 (thr 5.5) 68 % @ 0.81; M2m K4 51 % @ 0.81; M5 regs 63 %; controls: 6 CBV + R90 68 % @ 0.82; M0 + R90 + thr 6.5: 70 % @ 0.77. The 90/95 % points belong to R90, not to any look-alike method.
- Removal BEYOND R90 (279 REJ and 93 UNC that pass R90) vs retention of R90-surviving injections (complementarity_beyond_R90.csv): @0.70: M4c n200 85 %, M1 f0.7 78 %, M5 64 %, M2m 63 %, M3 33 %, plain SNR threshold 32 % (@0.82); @0.80: M2m K2 49 %, controls 31-32 %, M0 SNR 32 %; @0.90: M1 f0.5 31 %.

### Matched candidate count (B view; fig_roc.png B, matched_N_per_night.csv, matched_fp_interpolated.csv)
Recall at the same number of candidates (best over R90 on/off and SNR thresholds), ratio to M0+R90, per night 15 / 22 / 2, at N = 0.25 x and 0.5 x the M0+R90 count of that night (M0+R90 N = 34 / 149 / 189; absolute recall of M0+R90 at 0.25 x: 5.8 / 3.1 / 5.4 %, at 0.5 x: 7.4 / 4.5 / 7.3 %):
- M1 f1.0: 1.08 / 1.22 / 1.68 and 0.97 / 0.91 / 1.34; M1 f0.7 (literal): 0.96 / 1.30 / 1.57 and 0.91 / 0.95 / 1.16; M4c n50: 0.93 / 1.24 / 1.64 and 0.99 / 1.05 / 1.27; M4c n200: 0.95 / 1.20 / 1.61 and 0.99 / 0.91 / 1.38.
- No gain (0.25 x and 0.5 x points, all nights): M2m 0.90-1.21; M4 global CBV 0.67-1.04; M5 0.48-1.05; controls 6/10 CBV 0.73-1.11; random templates 0.84-1.01; random groups 0.45-0.61.
- Pooled at N = 0.25 x (93 cands): M0+R90 4.4 % vs M1 f1.0 6.7 %, M4c n200 6.8 %, M4c n50 6.4 %, M1 f0.7 6.2 % (+40-55 %, lower bounds when the curve ends); at 0.5 x: M0+R90 6.1 % vs 6.4-7.4 % (+5-21 %); at N >= prod+R90 all within +-3 % of M0+R90. The gain is concentrated on night 2 (73 frames) and night 22; none on night 15 (and its 0.25 x point is only 8 candidates).

## 3. Verdict per method
- M1 literal: does work as a purity booster only in the sense that stars with a (genuine) group lose most of their events, but it trades recall ~1:1 (REJ 77 % removed for 31 % of M0-recovered injections lost), depth is biased low (0.79 vs 0.81 at f=0.7, 0.60-0.65 at f >= 0.85), f is arbitrary (f=1 degenerates into a global mean) and noise-matched "look-alikes" at f >= 0.85 are mostly chance (null 17-21 % vs observed 29-41 %). Self in/out is irrelevant when groups are large (identical removal), 'in' only adds depth attenuation. Vs SNR thresholding it is better; vs R90 it is complementary but costs more recall than it removes at the production operating point.
- M2 unmasked (TFA-like with selection on the full LC): fails - selection absorbs the transit (retention 0.27-0.50, in-sample rms drops 45 % for every magnitude, i.e. overfit). Masking the candidate window is essential (M2m) but then it is equivalent to adding more per-tile CBVs (control cbv10: 48 % @ 0.79 vs M2m K2 49 % @ 0.80) and creates 165-406 new candidates with the same R90 pass rate as M0 (42 % vs 43 %).
- M3 veto: not useful (see above); recovered injections are flagged as often as REJECTED events.
- M4 global CBVs: no better than per-tile; M4c cluster-mean stack is ~M1-level at 1/10 the cost (no pairwise matrix).
- M5: no matched-N gain; reduces WASP-145 b SNR 9.2 -> 7.1.
- Best overall: M4c (or M1 f>=0.7) used as an extra pass AND combined with R90 (nominal +20-55 % recall at 25-50 % of the production+R90 candidate count on 2 of 3 nights, nothing at the production operating point).

## 4. WASP-145 A b
Survives (snr >= 6, R90 pass) in all 37 base variants and the 7 controls. Depth: M0 1.17 %; 1.11-1.19 % for M2m/M4/M4c/M5/M1 f<=0.85; 0.99-1.01 % M2 unmasked c0=0; M1 f1.0 ex 1.14 %, in 0.76 %. SNR 9.18 (M0); 8.0-8.3 (M2 unmasked), 8.8-9.2 (M4), 7.1-7.2 (M5). M3 flags it only at S1 thresholds <= 0.25-1.0 (K-dependent; S1 = 0.31-1.29), q = 0.03-0.14 (never at q >= 0.2). Weak test: ROBO43 has no look-alikes, so most variants reduce to M0 there. No T80S real transit exists to test on.

## 5. Runtime / memory per night (run_night.py; measured on a shared machine, 12 workers)
- Similarity tables (all pairs, top-48): corr 18-28 s wall / 27-94 CPU-s, least-RMS 27-42 s wall / 32-138 CPU-s (nights 2/15/22); peak 2.0-2.8 GB main + 0.37-0.68 GB per worker block (512 rows); Z matrix = 17-25 MB (n_star x n_frame float32). Global weighted PCA 2-4 s, K-means cluster stacks 7-13 s.
- Searches: ~1-4 ms per star per variant (search_one_star), R90 ~25 ms per candidate only. All 37 variants over all ~100 k stars: 330-440 s wall; 10 000 injections x 37 variants: 115-330 s; events 13-25 s. M4c alone costs ~1 PCA + kmeans + 1 extra search per star (minutes per night); M1 adds the least-RMS pairwise table (~0.5-2.5 CPU-min).

## 6. Caveats / not tested
- Three T80S nights only, strongly night-dependent results; REJECTED/UNCONFIRMED are visual labels (65 % "Systematics/Noise"); 'removal' is relative to the production candidate list, new candidates are unlabelled (R90-pass rate ~ M0's, median depth 4-12 %, up to 79 % partial for M4 k3 -> not evidently real).
- Injection recall is for my injection design (stars mag <= -9.6, 3 % per bin etc.), recall of M0 itself is only 9.4 %; retention/removal numbers include a winner's-curse component (any re-detrending drops marginal events: see controls). Depth bias means any look-alike SNR/depth must not replace production values.
- Not tested: SysRem (weighted PCA used), bright-only template pools, K > 6, ridge/regularised TFA, K-mean cluster counts other than 50/200, multi-night/period search, nights other than 15/22/2/ROBO43, interplay with SHARED_EPOCH/coincidence flags, aperture choice per method, hours of human re-vetting of new candidates, WASP-145 b on a T80S-like field, M1 with sign-aware groups.
- Files outside the scratch: none. /dev/shm was used temporarily (emptied).
