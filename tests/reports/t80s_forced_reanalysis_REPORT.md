> Report dated 2026-10-02. relphot main 2f3b709 (forced ingest + frame-quality cut + R90 + NEIGHBOUR_SHARED_EVENT) for the per-night runs; tie fix e629dd5, merged as 003d450 and used for the final multinight run. Scratch: `/ssdsto1/data/mnt/rejected_survival/`, `/ssdsto1/data/mnt/tie_fix/`, `/ssdsto1/data/mnt/neighbour_flag/` (may no longer exist).

# T80S re-analysis with forced photometry, neighbour shared-event flag, and survival of user-REJECTED events

## 1. What was run

- `robo43 forced` (fixed Gaia/master positions, fixed radii) on all 11 T80S nights 20251104-20251207: 527 frames, 0 errors except the 2 unsolved
  20251207 frames (045719, 051502) skipped by `forced.skip_unsolved`; 87 min at 5 workers; ~150 MB CSV per frame in `NIGHT/forced/`.
- Per night (`/ssdsto1/data/T80S_reduced/run_forced_step2_night.sh`): `relphot ingest --photometry forced`, `reference --no-variables`
  (frame-quality cut active), `lightcurves --no-variables`, `search` (default threshold) into a fresh `NIGHT/relphot/`; the catalogue-mode
  products are kept in `NIGHT/relphot_standard_20261002/`. 5-15 min per night.
- DB: backup `relphot_20261002_0959.dump` (pinned copy `pre_forced_t80s_20261002_0959.dump`), pre-reload export of all T80S detections,
  `relphot db load-night` for the 11 nights (night_ids kept; reviews re-attached by obj_id + |dtc| <= 0.5 old duration, otherwise
  `detection_review_orphan`), multinight `mn_1104_1207_loose_forced` (mn_run 12), old mn_run 11 dropped.
- `relphot db analyze --all` FAILED (OverflowError at analyze.py:794): see section 4.
- The NASA Exoplanet Archive cross-match fails on every night (`ORA-00904: 'HTM20': invalid identifier`, archive side); non-fatal, 0 known
  planets matched (none expected in this LMC field).

## 2. NEIGHBOUR_SHARED_EVENT (commit 827279a, merge 2f3b709)

- Informational bit 15 (`FLAG_NEIGHBOUR_SHARED_EVENT`), not a hard reject, tier unchanged; module `relphot.transit_neighbour`, called in
  `cli._run_search` after the NEIGHBOUR_BLEND step. Settings: `neighbour_event_enabled = true`, `neighbour_event_radius_arcsec = 12.0`
  (largest aperture 8" + ~2 FWHM), `neighbour_event_dip_sigma = 3.0`.
- For each candidate, every star within the radius is fitted at the candidate's own (tc, duration) with the cross-aperture joint
  poly + CBV + box fit; flagged when the best neighbour's depth/error >= 3 sigma (catches leakage dips below the search threshold).
  Probable source = larger absolute flux deficit (depth x median flux); a mutual pair is decided once at the higher-SNR member's window.
- Outputs: `shared_*` columns in candidates.csv, `transit_shared_*` in search metrics and `detection.extra`.
- Validation: ROBO43 20250911 WASP-145 A b and its Gaia neighbour 5.2" away both flagged, A as source (deficit ratio 0.78 / 1.28);
  2 of 6 candidates. T80S 20251107 catalogue products: 2 of 169. In this re-analysis: 20251104 6 of 139 candidates.
- 676 tests at the merge. Thresholds validated on two nights only.

## 3. Survival of user-REJECTED transit events

Question: of the 1171 transit events the user marked REJECTED (catalogue photometry, pre-reload snapshot), how many does the new
implementation still produce?

Definitions: same star = Gaia id (string), else nearest within 1", same night; same event = new candidate with
|tc_new - tc_old| <= 0.5 x max(old, new duration). Levels: (a) star not searched; (b) searched, no candidate at that epoch;
(c) still a candidate; (d) (c) and `r90_pass`; (e) (d) and no NEIGHBOUR_SHARED_EVENT; (f) (c) and the REJECTED status re-attached in the DB.

| level | REJECTED n (of 1171) | % | UNCONFIRMED n (of 447) | % |
|---|---|---|---|---|
| (a) star not searched | 10 | 0.9 | | |
| (b) no candidate at epoch | 810 | 69.2 | | |
| **(c) still a candidate** | **351** | **30.0** | 90 | 20.1 |
| (d) + R90 pass | 197 | 16.8 | 56 | 12.5 |
| **(e) + no NEIGHBOUR_SHARED_EVENT** | **194** | **16.6** | 56 | 12.5 |
| (f) re-attached in DB | 328 | 28.0 | | |

Tolerance 1.0 x max(duration): identical for REJECTED (351/197/194); UNCONFIRMED 91/57/57.
DB cross-check: 328 REJECTED transits after reload, 846 orphan rows; 23 survivors are orphaned by the DB's narrower tc rule (tc moved
0.4-1 h under forced photometry), 3 changed obj_id on reload.

**Headline: 30.0 % of user-REJECTED events are still search candidates; 16.6 % still pass with no R90 or neighbour flag.**

By night (REJECTED N, %c, %e; UNCONFIRMED N, %c, %e):

| night | REJ N | %c | %e | UNC N | %c | %e |
|---|---|---|---|---|---|---|
| 20251104 | 110 | 42.7 | 26.4 | 16 | 31.2 | 25.0 |
| 20251105 | 329 | 34.7 | 20.7 | 133 | 36.1 | 24.8 |
| 20251106 | 98 | 31.6 | 9.2 | 39 | 17.9 | 10.3 |
| 20251107 | 57 | 28.1 | 12.3 | 40 | 27.5 | 17.5 |
| 20251112 | 32 | 21.9 | 0 | 38 | 5.3 | 0 |
| 20251118 | 45 | 13.3 | 0 | 42 | 4.8 | 0 |
| 20251130 | 98 | 1.0 | 0 | 37 | 0 | 0 |
| 20251201 | 71 | 12.7 | 1.4 | 13 | 0 | 0 |
| 20251204 | 62 | 37.1 | 8.1 | 14 | 14.3 | 7.1 |
| 20251206 | 19 | 15.8 | 0 | 25 | 20.0 | 4.0 |
| 20251207 | 250 | 37.6 | 30.0 | 50 | 16.0 | 12.0 |

By old tier (REJECTED %c / %e): tier 1 (218) 39.4 / 23.4; tier 2 (388) 32.2 / 21.1; tier 3 (565) 24.8 / 10.8.

By the user's note (REJECTED N, %c, %e): "Systematics. Noise" 763, 30.8, 19.0; "Noise" 230, 20.9, 3.5; "Systematics" 131, 43.5, 23.7;
"Bad phot estimation" 26, 38.5, 34.6; "Two-step phot" (incl. "Teo-step") 12, **0**, 0; other 9, 11.1, 11.1.
Bulk "REJECT ALL ... look-alikes" notes: 182 events, %c 31.3, %e 19.8.

Findings:
- ~70 % of rejected events vanish: the star's new best event has SNR < 6 (276 below 3, 148 at 3-4, 163 at 4-5, 202 at 5-6); 21 with SNR >= 6
  are vetoed (TOO_DEEP, FEW_POINTS).
- All 12 "two-step phot" events vanish (new SNR 1.4-3.7): the blend extraction flips are removed by forced photometry, as intended.
- R90 removes 44 % of the REJECTED survivors and 38 % of the UNCONFIRMED survivors: not selective for user-rejected events.
  79.6 % of (c) survivors are REJECTED (77.6 % at (e)) against a prior of 72.4 %.
- NEIGHBOUR_SHARED_EVENT flags 7 REJECTED and 2 UNCONFIRMED survivors: nearly inert on this sample (it targets close pairs, not the
  dominant systematics/noise rejections).
- Survival is strongly night dependent: 20251130 keeps 1 of 98; 20251112/1118/1130/1206 keep none past R90; 20251104 and 20251207 keep 26-30 %.
- New candidates per night (old events / new candidates / new R90 pass): 1104 126/139/99, 1105 464/407/266, 1106 137/87/32,
  1107 97/206/111, 1112 70/93/9, 1118 87/42/5, 1130 135/47/2, 1201 84/44/10, 1204 76/59/17, 1206 44/23/6, 1207 300/404/273;
  total 1620 / 1551 / 830 (814 of the 830 without NEIGHBOUR_SHARED_EVENT). 1110 of the 1551 new candidates match no old event and are unreviewed.

Figures: `figures/t80s_forced_rejected_survival_levels.png`, `figures/t80s_forced_rejected_survival_by_night.png`.
Per-event table: `t80s_forced_rejected_survival_per_event.csv` (all 1620 old transit events, level, new SNR/flags, 1.0x columns).

Caveats: REJECTED is a user verdict, UNCONFIRMED is not a true-positive set, so survival is pipeline stability, not purity; old events
are catalogue photometry mostly without R90; search_metrics keeps only each star's best event, so the (b) SNR is an upper bound at the
old epoch.

## 4. Multinight tie divergence on forced photometry (fix on branch `tie-fix`, e629dd5)

- mn_run 12: 24378 of 911280 tie rows had |mag| ~ 1e20 (mag_err 1.0000x), only in loose nights 20251112 (7629), 20251118 (6495),
  20251207 (10254); `db analyze` overflowed at analyze.py:794.
- Root cause: (1) `tie_nights` — forced photometry gives some stars an uninformative night (3.5-6 mag off, error 1-11 mag); those rows are
  weighted by their own error only and have high leverage in the dm**2 term; the leave-one-out feedback diverges (aperture 4 core zp to
  -1.2e40, star 106195 mean_mag 7.6e35, "did not converge, max|dZ| = 10.7"). (2) `tie_loose_nights` admits such a core star (a loose
  comparison star exactly in 1112, 1118, 1207) into its first unclipped fit, giving coefficients ~1e19-1e22 and a saturated 1000 mmag floor.
- Fix: drop tie rows > 3 mag from the other nights beyond the night's median offset at iteration 1; reject stars with |zp| > 5 mag
  (NaN zp/mean_mag); loose pre-trim > 5 mag from the night's offset, non-finite fit raises, |zp| > 50 -> NaN; analyze skips tie mags
  beyond 100 and applies no offset beyond 20 mag. 5 regression tests, 681 passed.
- Re-run in scratch: tied mags of 1112/1118/1207 back to -15.0..-8.6; aperture-4 floors of those nights 1000 -> 69-74 mmag
  (catalogue run 69-78); aperture-0 floors unchanged; 17 core stars in aperture 4 excluded; aperture-4 core zps of 1104/1107 shift
  by a median 14-17 mmag for 1356/4355 stars.
- Remaining: the tie still weights rows by their own error only (the correct 1/(s^2 + var_loo + floor^2) would change every tie);
  thresholds are judgement calls; "did not converge in 20 iterations" is logged for all apertures as before.
- Forced per-night tie floors (aperture 0, mean of bins) are 1.1-1.9x the catalogue-mode values on the core nights (1104 7.6 vs 3.4,
  1105 5.8 vs 4.0, 1106 27.3 vs 18.3, 1107 20.8 vs 15.4 mmag).

## 5. Completion (tie fix merged 003d450)

- `run_forced_step3.sh`: mn_run 12 dropped (diverged products kept in `multinight/mn_1104_1207_loose_forced_diverged/`), `relphot multinight`
  re-run with the fix (97 s; 17 aperture-4 stars excluded as runaway), `multisearch`, `db load-multinight` -> mn_run 13,
  `relphot db analyze --all` (196 s): 911280 tie rows, 0 non-finite or |mag| > 30; 1559 transit shapes; coincidence veto 472 auto-rejected.
- With the coincidence veto (DB, after analyze), the 328 REJECTED events re-attached in the DB split into 214 not auto-rejected
  (119 of them without any R90 or NEIGHBOUR_SHARED_EVENT flag) and 114 auto-rejected. **Strictest survival: 119 of 1171 = 10.2 %**
  (DB-based, so it excludes the 23 survivors orphaned by the tc re-attach rule). UNCONFIRMED in the DB after reload: 1225 transit events,
  358 auto-rejected; 399 not auto-rejected and without R90/NEIGHBOUR_SHARED_EVENT flags.

## 6. Pending

- Re-attach rule: 23 REJECTED reviews orphaned by tc shifts of 0.4-1 h; consider a wider tc window for forced reloads.
- Visual review of the NEIGHBOUR_SHARED_EVENT flags and of the 1110 new unreviewed candidates (mostly 20251107, 20251207).
