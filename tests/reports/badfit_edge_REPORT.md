# Bad fit / bad phot edge — diagnosis and proposal (2026-10-03)

RESEARCHER study, read-only (no code/DB/data changed). DB state: after `relphot db analyze --all` rc=0 at 08:40
(mn_run 14). 1602 per-night transit events (detection.kind='transit', all with a transit_shape row): 2 CONFIRMED,
1128 REJECTED (user), 472 UNCONFIRMED (154 of them auto-REJECTED by the coincidence veto). Labelled cases =
relphot.user_night_review notes "Bad fit" / "Bad phot edge" / older "Wrong fit".

## Summary

1. The web draws the **analyze trapezoid** (relphot.transit_shape), not the search box (box only as fallback when no
   shape row exists; never the case now). Trapezoid drawn only when ingress_frac is non-NULL (548 events); the 1054
   events with unobserved ingress/egress show only a dotted tc line.
2. **"Bad phot edge" root cause:** the pre-fit clip `robust_clip_series` (numeric.py, `median_filter(mode="nearest")`)
   is structurally blind to the first/last epoch (its residual is exactly 0: the padded window holds 8 of 15 copies of
   the point itself). Verified independently: a +22 % last / +13 % first epoch are kept while the same outlier
   mid-series is clipped. Edge epochs of z = 8-39 survive, bend the poly baseline in `search_one_star`, and the box
   starts/ends right next to them (box edge 0.01-0.05 h from the night boundary).
3. The analyze shape fit (`_fit_transit_shape`) has **no clip at all** and a free baseline in [0.1, 10] x b0; with
   <= 2 out-of-transit epochs it extrapolates (baseline 1.43-1.73 x median flux, T14 longer than the night, depth
   0.30-0.42) -> the "open box".
4. In DB flux all 128 edge-driven events have the edge epoch HIGH (+8..+33 %); the rest of the night then looks like a
   dip. These are per-star epochs, not bad frames (ensemble median at first/last frames 1.000 +- 0.001 on every night),
   so the frame-quality cut cannot act on them.
5. Labelled cases: 6 of 8 testable (4085, 11114, 16876, 27931, 29139 + old 3500) are edge-driven (replay SNR 5-17 ->
   1.5-3.2 after clipping); 4536/2 = depth fit at lower bound (1e-6); 23395/35 = variable ramp, T14 at upper bound;
   8727/35 and 40563/20 = marginal noise at SNR 6.1/6.4 (not edge); 1669/15, 8492/15, 16999/17: events no longer in DB.
6. **P1 (root cause):** `edge_outlier_mask` — clip isolated first/last 1-2 epochs with |dev| > 6 s from the median of
   the next 6 (s = robust point-to-point sigma), next epoch consistent within 3 s; runs >= 3 never clipped. Apply in the
   search and before the analyze shape fit. Measured: removes 128/1602 events (REJ 100/1118, UNC 22/472, labelled
   6/10), CONFIRMED lost 0/2, proxy positives (50 matched-pair + 4 family events) lost 0. Injection (9454 real LCs,
   1853 recovered 1-5 % transits): 1 lost at z=6 (0/75 edge-partial).
7. **P2 (auto-reject stored events, no re-search):** H1 edge outlier adjacent to box (117), H2 fitted depth < 1e-3
   (24), H3b < 5 out-of-transit epochs and fit depth > 0.2 (67); union 153 events (9.6 %): REJ 109/1118, UNC 37/472,
   labelled 7/10, CONFIRMED 0/2, proxy 0/54. Baseline-extrapolation (> 1.2) is NOT safe as a reject (hits LMC V2587
   eclipses) -> informational only.
8. Leave-k-out of the lowest in-transit points is the wrong test (the culprit is a HIGH out-of-transit edge point) and
   unsafe (loses 8 % / 15 % of injected recovered transits at k=1/2). Not proposed.
9. Caveats: CONFIRMED set is only 2 events, so safety rests on injection + proxy sets; the search replay lacks CBVs /
   frame-error scale (reproduces the edge cases within ~15 %); no pipeline re-run, no web rendering check.

## a) Code paths

- Search candidate (`relphot search`): `transit_search.py` `search_one_star`: decorrelated LC ->
  `robust_clip_series(y, lc_clip_sigma=6, lc_clip_window=15)` -> `fit_nuisance_model` poly(2)+CBVs -> joint box grid
  (0.4-2.5 h, `min_in_transit_points=5`, `min_box_coverage=0.5`) -> flags (EDGE = box edge within 0.25 T of data edge;
  TOO_DEEP depth > 0.30 hard reject; STEP_LIKE; R90). Stored as detection tc/depth/duration/snr/flags.
- Analyze shape: `db/analyze.py` `_transit_shapes` -> `_fit_transit_shape`: DB LC via `_good_night_flux` (no
  clipping, normalised to night median); window |t-tc| <= max(1.5 d0, d0+1 h); constant baseline; `least_squares` on
  (tc, depth, T14, ingress_frac, baseline), bounds tc +-0.5 d0, depth >= 1e-6, T14 in [0.3,3] x d0, ingress [0,0.5],
  baseline [0.1,10] b0; `_trapezoid_shape` ingress floor 1e-3 T14 (box limit); ingress_frac NULL unless ingress and
  egress both observed.
- Web: `web/app.py` joins detection + transit_shape; `web/static/app.js` draws `trapezoidTrace` when ingress_frac is
  non-null, else a dotted tc line; the search box `rect` only when the object has no transit_event for that night.
- Auto status: `db/coincidence.py` `update_coincidence` sets auto_status/auto_reason (called from analyze).
- The same blind clip is also used in `depth_at_other_aperture` (transit_search.py) and `variables.py`.

## b) Per-case table

Hours relative to the night's first epoch; cadence 2.2 min; "det box" = search box; "shape" = stored trapezoid;
z in units of robust point-to-point sigma vs median of the next 6 epochs.

| case | note | n / span | edge epochs (norm. flux) | det box [h], n_in, top1 | shape: depth, T14, ingress, baseline, n_pre/in/post, conv | edge z | replay SNR no-clip -> clip | cause |
|---|---|---|---|---|---|---|---|---|
| 11114/35 | Bad fit, Bad phot edge | 42 / 1.51 | last 3: .997 .982 **1.216** | [-0.40,1.50], 41, 0.60 | 0.42, 1.79 h (> night), 0.063, **1.73**, 0/40/2, yes | end +19.6 | 17.0 -> 2.1 | last epoch +22 % 0.01 h after box end; only 2 out-of-transit epochs; baseline/depth degenerate |
| 4085/35 | Bad phot edge | 42 / 1.51 | last 3: 1.004 1.02 **1.236** | [-0.40,1.50], 41, 0.75 | 0.37, 1.74 h, 0.055, **1.58**, 0/40/2, yes | end +17.7 | 15.9 -> 2.3 | identical to 11114 (same tc) |
| 16876/22 | Bad fit, Bad phot edge | 66 / 2.70 | first: **1.127** .995 1.018 | [0.02,2.07], 49, 0.70 | 0.0055, 1.41 h, **0.001 (box)**, 1.005, 19/31/16, **no** | start +17.9 | 5.2 -> 2.4 | first epoch +12.7 % bends baseline; box 0.5 % deep, unconverged |
| 27931/22 | Bad phot edge | 66 / 2.70 | first: **1.330** 1.029 .989 | [0.02,0.87], 23, 0.77 | 0.011, 0.44 h, 0.065, 1.008, 13/11/42, no | start +19.7 | 6.2 -> 1.5 | same mechanism |
| 29139/16 | Bad phot edge | 23 / 0.97 | last: .974 .998 **1.103** | [0.55,0.95], 10, 0.40 | 0.0165, 0.29 h, 0.016 (box), 1.012, 12/8/3, no | end +7.7 | 9.1 -> 3.0 | same |
| 40563/20 | Bad phot edge | 39 / 1.52 | first 3: **1.076 1.040** .984 | [0.05,0.45], 11, 0.35 | 0.009, 0.38 h, 0.21, 1.001, 2/10/27, yes | start +2.8 (not clipped) | 4.7 -> 4.7 | two mildly high first epochs + marginal event (SNR 6.4) |
| 8727/35 | Bad fit | 42 / 1.51 | normal | [0.97,1.52], 15, 0.22 | 0.0083, **0.24 h**, 0.153, 1.002, 26/7/9, yes | none | 4.9 -> 4.9 | marginal noise event at SNR 6.1 |
| 23395/35 | Bad fit, Var | 42 / 1.51 | 1.345 ... 0.832 (40 % ramp) | [0.70,1.40], 19, 0.13 | 0.385, **2.10 h (upper bound)**, 0.43, 1.31, 17/25/0, yes | 3.6/5.3 (trend) | 8.0 -> 8.0 | variable-star ramp; ON_VARIABLE, R90_SYSTEMATICS |
| 3500/1 | old "Wrong fit" | 45 / 1.66 | first: **1.136** 1.015 1.004 | [0.02,2.07], 44, 0.70 | 0.30, 2.17 h, 0.065, **1.43**, 5/40/0, yes | start +10.8 | 7.5 -> 3.2 | edge mechanism |
| 4536/2 | old "Wrong fit" | 69 / 3.16 | .886 .882 .923 (ramp) | [1.12,2.58], 25, 0.09 | **depth 1e-6 (bound)**, 1.10 h, 0.135, 0.987, chi2_red 16.3 | none | 8.7 -> 8.7 | fit collapsed to zero depth on a ramp |
| 1669/15, 8492/15, 16999/17 | old "Wrong fit" | 65/65/23 | normal | event rows no longer in DB | - | - | replay 5.7/5.8/2.4 | not attributable |
| 5276/2 CONFIRMED | "Possible eclipse" | 69 / 3.16 | normal | [-0.43,1.32], 35, 0.16 | 0.052, 1.10 h, 0.20, 1.005, 9/28/32 | 0.5/0.8 | 14.7 -> 14.7 | unaffected |
| 64813/6 CONFIRMED (WASP-145 A b) | | 310 / 2.83 | normal | [1.25,1.65], 42, 0.08 | 0.019, 0.61 h, 0.267, 0.998, 129/67/114 | 1.6/0.1 | 9.7 -> 9.7 | unaffected |

## c) Diagnostics on all 1602 events

LAB = 10 labelled events with a DB row; CONF = 2; REJ by note category; UNC = 472.

| rule | LAB | CONF | REJ none | REJ noise/syst | REJ phot | REJ var/ecl | UNC | total |
|---|---|---|---|---|---|---|---|---|
| n | 10 | 2 | 1027 | 63 | 13 | 15 | 472 | 1602 |
| R1: edge clip (z=6) and replay SNR < 6 | 6 | 0 | 97 | 3 | 0 | 0 | 22 | 128 |
| H1: clipped epoch(s) are the only epoch(s) outside the box on that side | 6 | 0 | 89 | 3 | 0 | 0 | 19 | 117 |
| H2: fitted depth < 1e-3 | 1 | 0 | 2 | 2 | 1 | 0 | 18 | 24 |
| H3b: < 5 epochs outside trapezoid AND depth > 0.2 | 2 | 0 | 54 | 2 | 0 | 0 | 9 | 67 |
| H1 or H2 or H3b | 7 | 0 | 103 | 5 | 1 | 0 | 37 | 153 |
| (info) EDGE_ADJACENT: 1-2 epochs outside box on one side | 7 | 0 | 149 | 5 | 0 | 0 | 48 | 209 |
| (info) fit baseline > 1.2 x median | 4 | 0 | 88 | 3 | 1 | 3 | 30 | 129 |
| (info) unconverged AND ingress < 1 cadence | 3 | 0 | 3 | 12 | 2 | 0 | 39 | 59 |
| fit at a bound | 2 | 0 | 67 | 2 | 1 | 8 | 43 | 123 |
| R90_SINGLE_POINT (existing) | 7 | 0 | 259 | 11 | 5 | 0 | 86 | 368 |
| EDGE & R90_SINGLE_POINT | 7 | 0 | 232 | 9 | 4 | 0 | 68 | 320 |
| existing coincidence auto-REJECTED | 0 | 1 | 300 | 6 | 4 | 2 | 154 | 467 |

R1 at z=4/5/8: LAB 7/6/5, UNC 28/22/20, REJ 109/104/85.

- All 128 R1 events have a HIGH edge epoch (65 start, 59 end, 4 with 2 epochs); clipped-epoch z min 6.1, median 11.8,
  p90 20.8, max 39. 105/128 PARTIAL+SHARED_EPOCH, 127/128 R90_SINGLE_POINT, 34 NEIGHBOUR_BLEND, 20
  NEIGHBOUR_SHARED_EVENT, 12 ON_VARIABLE; 20 already auto-REJECTED. By night (R1/n): 1: 15/141, 15: 5/206, 16: 12/93,
  17: 18/42, 18: 12/47, 19: 12/44, 20: 16/59, 21: 8/23, 22: 17/404, 35: 13/41; nights 2, 3, 6: 0/502.
- Base rate (6000 random stars/night, z=6): first/last-epoch outlier in 0.02-1.7 % of stars, mostly low-going; the
  event list is enriched because a high edge epoch manufactures a >= 6 sigma box.
- Injection: 9454 real LCs (nights 35, 22, 2, 16, 20), trapezoids 1/2/5 %, T14 0.5/1/2 h, centres incl. partial;
  1853 recovered. Lost/gained by the clip: z=4 17/5 (2 of 75 edge-partial lost), z=5 6/3, z=6 1/2 (0/75), z=8 0/2.
- Proxy positives (50 transit_match pairs p >= 0.05 + 4 family members): H1/H2/H3b hit 0; R1 hits 2 (obj 148077 nights
  20 and 22, both ON_VARIABLE|R90_SINGLE_POINT, a pair made of edge artefacts). Baseline > 1.2 hits obj 56847 (LMC V2587
  eclipses, real) -> informational only.
- Leave-k-out: labelled edge cases unchanged; injected recovered transits lost 144/1839 (k=1), 274/1839 (k=2) -> unsafe.
- Dropped shape rules: ingress < 0.02 alone (hits real sharp dips), depth/err, chi2_red > 5 (hits a CONFIRMED event),
  not-converged alone (151).

## d) Proposal

**P1 `edge_outlier_mask`** (new, `numeric.py`; reference implementation in the study scratch `edge.py`):

```
def edge_outlier_mask(t, y, err, z=6.0, k_max=2, n_ref=6, z_ref=3.0, min_n=15):
    # keep-mask in input order; finite epochs only; no-op if < min_n epochs
    s = max(1.4826*MAD(diff(y_sorted))/sqrt(2), median(err), 1e-4)   # robust point-to-point sigma
    for end in (start, end):
        for j in (k_max, ..., 1):
            ref = median(y_end[j:j+n_ref])
            if |y_end[j] - ref| > z_ref*s: continue          # the run must end sharply
            dev = y_end[:j] - ref
            if all(|dev| > z*s) and all same sign: drop the j edge epochs; break
```

- Runs of >= 3 deviant epochs are never touched (real partial transits at the night edge). Two-sided. z = 6 reuses
  `lc_clip_sigma`; new settings `edge_clip_max_epochs=2`, `edge_clip_ref_epochs=6`.
- Apply in `search_one_star` right after `robust_clip_series`, in `depth_at_other_aperture`, in `variables.py` (same
  blindness), and in analyze before the trapezoid fit (`_fit_transit_shape` after `_good_night_flux`), so stored and
  RERUN shapes never use the epoch.
- New informational bit `FLAG_EDGE_OUTLIER = 1 << 16`; clipped indices kept (detection.extra) so the web can draw them
  as grey crosses.
- Expected: 128 events (8.0 %) lose candidacy (REJ 100, UNC 22, LAB 6/10), CONFIRMED 0/2, proxy 0, injected-transit
  loss 1/1853. Takes effect only after `relphot search` is re-run per night.

**P2 analyze auto-rejects** (stored events, no re-search; reason in detection.auto_reason, a person's verdict stands):

- H1 `EDGE_OUTLIER`: the mask removes k epochs at one end and these are the only epochs outside the detection box on
  that side: 117 events.
- H2 `NO_DIP`: fitted depth < 1e-3: 24 events.
- H3b `NO_BASELINE`: < 5 epochs outside the fitted trapezoid and depth > 0.2: 67 events.
- Union 153 (9.6 %): REJ 109, UNC 37, LAB 7/10, CONF 0/2, matched/family 0/54. Not caught: 8727/35, 40563/20 (marginal
  noise at the SNR 6 cut), 23395/35 (variable ramp; only the informational baseline rule).
- `update_coincidence` clears and rewrites auto_status/auto_reason per night: merge into one `update_auto_verdicts`
  that unions reasons, or the new rules get overwritten.
- Informational only (bits 17-19 / web badges): EDGE_ADJACENT, BASELINE_EXTRAPOLATED, BOX_UNCONSTRAINED, fit-at-bound.

Interactions: SHARED_EPOCH drops once the artefacts are not searched; coincidence-veto counts fall (re-run
update_coincidence); NEIGHBOUR_* pairs formed through an edge artefact vanish (transit_neighbour.py should use the same
mask); R90 features computed on the clipped series; frame-quality cut unchanged (per-star edge epochs are invisible to it).

## Test plan

1. `edge_outlier_mask` unit tests: 1-2 isolated high/low edge epochs masked; 3 consecutive deviant epochs not masked;
   real partial transit at the night start not masked; trends and 40 % ramps not masked; n < 15, NaNs, unsorted,
   constant series; symmetry y -> 2 - y; characterisation of the `robust_clip_series` blind spot.
2. Search: synthetic 45-epoch night with a 1.3x first (or last, or two) epoch is no longer a candidate; injected 2 %
   transit with 6 epochs at the night start stays recovered with the same tc/depth.
3. Analyze: `_fit_transit_shape` with last epoch +22 % never gives baseline > 1.1 or T14 > night span; injected
   trapezoid + edge outlier recovers depth within 10 %; NO_DIP and NO_BASELINE fixtures.
4. Auto verdicts: reasons from different rules unioned, idempotent, a person's CONFIRMED/REJECTED unchanged.
5. Golden regression from the labelled cases (synthetic): all flagged H1; 5276-like and 64813-like not.
6. Slow seeded injection test: recovery loss <= 0.2 % at z=6, clipped-star base rate < 0.5 %.
7. Web: clipped epochs as grey crosses; EDGE_OUTLIER in the legend; new auto reasons shown.
8. Re-run protocol: re-search nights 1, 15-22, 35; expect ~128 events leaving; 5276/2 and 64813/6 unchanged.

## Verified / not verified

Verified: code paths read; exact refit reproduces stored shapes (60/60); all labelled numbers from the live DB; the
`robust_clip_series` edge blind spot (also independently by the main session). Not verified: end-to-end `relphot
search` with the clip; web rendering of the views the user reviewed (shape rows rewritten 08:37-08:40); the 3 older
"Wrong fit" events (gone from DB); statistical power on CONFIRMED (n = 2).
