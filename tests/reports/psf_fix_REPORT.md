# PSF fix: corner registration, faint-star zero point, bin cap, variability auto-reject (2026-10-04/05)

This follows up `tests/reports/psf_rerun_REPORT.md`, whose two open issues it fixes:

1. The corner registration failure on 20251107/20251207 (and mildly on four other nights).
2. The seeing-dependent PSF flux of faint stars.

What was done after the fixes:

- All 13 T80S nights were redone in place (night_id kept) and the multi-night tie was rebuilt.
- The `test_psf` schema was dropped from the production database.
- Two smaller defects found on the way were fixed: the corner tile 9 noise floor, and transit fits missing after a night reload.
- Then, at the user's request:
  - Three event-level "is this dip the star's own variability?" tests were measured.
  - The two that work became the VARIABILITY auto-reject rule.
  - A slope detrending for long-period variables was checked and found to be useless.

Branch `psf-fix`: commits 2e47f4e, 709ed99, da4b356, eff0be6, 7c85220, c9f6c5b, 65b8d39, 6783702, 3346409, f875b2e, db2607a.
User vetting was empty during the whole run: 0 vetted events, and the user cleared their own notes.

## What was done

1. **Registration** (da4b356, `scripts/psf/psfphot.py`): a smooth per-frame shift field replaces the affine shift. 1107, 1207,
   1104, 1130, 1206 and 1209 were re-fitted (pass 1, static companions and pass 2; the PSFEx models are reused).
2. **Faint-star zero point** (`scripts/psf/finalize.py`, final version 65b8d39), on all 13 T80S nights. It took three
   versions; two were rejected (see below).
3. **In-place redo** (`scripts/psf/psf_redo.sh TEL NIGHT refit|refinalize|relphot`, `redo_all.sh`; 2e47f4e, c9f6c5b, f875b2e).
   - Superseded products are moved, never deleted, to `$PSF_REDO_OLD/<TEL>/reduced/<night>/`.
   - Then the relphot chain and `db load-night` run again, on the same night_id. A line is appended to `psf/PROVENANCE.txt`.
4. **Bin cap** (3346409, `relphot.numeric.fit_noise_floor`): the magnitude bin count is capped at `n_valid // min_bin_stars`.
   1130 and 1207 were rerun in `relphot` mode.
5. **Web** (6783702): a night without a trapezoid fit draws the search box dashed and labelled "search box, no trapezoid fit".
6. **`test_psf` dropped** (2026-10-04 14:37).
   - First a pinned backup, `pre_psf_fix_20261004_1431.dump`, checked with `pg_restore -l`.
   - Then `DROP SCHEMA test_psf CASCADE` (28 objects). The `relphot` schema and the 14 nights are intact.
   - `scripts/psf_vs_aper_lc.py` still defaults to `test_psf` and is obsolete.
7. **Tie and analyze**:
   - `tie_psf.sh` again, then `analyze --all --keep-vetted`.
   - Pinned backup after: `post_psf_fix_20261005_0009.dump` (4.48 GB, 27 tables).
8. **Dip tests** measured, and the **VARIABILITY rule** added (db2607a). The reprocess worker was restarted on the new code.
9. **LPV slope**: checked, not implemented (see the last section).

## 1. Corner registration

**The defect.** The old fit corrected only an affine offset between the PSF frame and the catalogue positions. On frames with a
poor WCS (132-154 matched astrometry stars, against 220-330), the top-left corner was off by 2-4.5 px:

- 43-75 % of the corner stars got bit 64.
- The comparison pools emptied there.
- 94 % of the 1107 and about 90 % of the 1207 transit candidates came from this.

**The fix: a smooth shift field per frame.**

- **Calibration stars**: bright isolated stars (SNR > 100, unflagged, isolated), stratified over a 10 x 10 grid, at most 400.
- **Free-shift fits**: searched from the centre outwards, so the corner search starts from its neighbours' shifts:
  - the 40 centre-most stars on a coarse +-6 px grid, then Nelder-Mead;
  - the rest on a +-4.5 px grid around the median of their 6 nearest fitted neighbours;
  - shifts of 12 px or more are dropped.
- **The field**: the best half of the stars by chi2 are kept, and a polynomial of degree 1-3 is fitted with 3-sigma clipping.
  - The degree is chosen by 5-fold CV. A higher degree must cut the median CV residual by more than 2 %.
  - Degree 2-3 need 7 of 9 coarse cells occupied.
- **Clamps**:
  - The field is clamped to the box of the calibration stars, so it never extrapolates.
  - It falls back to affine if it exceeds 13 px anywhere.
  - Detections are matched to the shifted master positions.
- **Per-frame record**: `shift_deg`, `shift_nuse` and `shift_max` go into the raw pickles.

**Why cubic.** CV residual in the top-left cell, in px:

| frame | affine | cubic |
|---|---|---|
| bad frames (1107 034537, 041837; 1207 054614) | 1.47 / 1.39 / 1.97 | 0.19 / 0.21 / 0.26 |
| good frames | 0.39 / 0.23 | 0.17 / 0.23 |

- The per-frame noise level is 0.13-0.26 px.
- In a survey of 104 frames, every bad frame chose degree 3, with a shift_max of 1.4-9.9 px. Good frames stay at 0.13-1.63 px.
- shift_max > 2 px is a clean "bad WCS" diagnostic.

**Before and after.** Prototype, corner cell, bit-64 fraction / chi2 against a reference / flux against a reference:

| frame | before | after | shift_max (px) |
|---|---|---|---|
| 1107 034537 | 0.638 / 6.36 / 0.763 | 0.001 / 1.02 / 0.999 | 5.5 |
| 1107 050246 | 0.669 / 83.9 / 0.486 | 0.002 / 1.07 / 1.004 | 9.9 |
| 1207 032636 | 0.742 / 88.5 / 0.408 | 0.005 / 1.19 / 1.007 | — |

- **Good frames are unchanged**: bit 64 changes by at most 0.007, chi2 by at most 0.02, flux by at most 0.001. Bright-star flux
  median 0.00 mmag.

**Production re-fits** (2 h 59 min for the six nights). Frames whose corner chi2 exceeds 1.5 x the rest of the frame, before >
after, with the worst ratio:

| night | frames | worst ratio | largest field (px) |
|---|---|---|---|
| 1107 | 28 > 0 of 67 | 10.4 > 0.91 | — |
| 1207 | 19 > 0 | 12.3 > 0.92 | 10.1 |
| 1104 | 2 > 0 | 7.2 > 0.97 | 7.1 |
| 1130 | 2 > 0 | 3.8 > 0.94 | 3.8 |
| 1206 | 1 > 0 | 1.5 > 0.94 | 2.7 |
| 1209 | 0 > 0 | 1.1 > 0.99 | 2.5 |

- The calibration adds 2.1 s per frame, about 1 % of a pass-2 frame.
- Pass 1 had to be redone: 31 % (1107) and 10 % (1207) of the old corner static companions were ghosts of the misregistration.

**Caveats.**

- `MAXSHIFT_REF` (1.2 px cap of the centroid refinement) is unchanged.
  - Raising it to 2.0 would cut refine accept/reject flips on good frames (old-vs-new flux changes > 10 mmag for 6.0 % > 0.6 % of
    bright isolated stars).
  - It changes every star and needs the finalize `REL_THR` recalibrated: a separate validation, not done.
- **Sparse frames**: below about 100 usable stars the field degrades towards affine and under-corrects. WCS errors above 12 px
  are not recoverable.
- **T80S only**: ROBO43 needed no re-fit and is untested.
- **Out of scope**: `FLUX_APER_PROD_8ARCSEC` stays off-centre in such corners (the production aperture). The upstream cross-frame
  WCS mismatch (1.3-2.7") that leaves the robo43 master list sparse in the top-left corner (tile 9) is also out of scope.

## 2. Faint-star zero point

**The defect.** Below about 4000 counts, the PSF flux tracks the frame PSF FWHM:

- +25-40 % from the sharpest to the broadest frame.
- Per-frame scatter of the median offset of 34-148 mmag, against 1.5-2.6 for apertures.

**The model** (all versions): dm_ij = zp_j + a_j h(cell_i), a rank-1 alternating least squares.

- **Cell**: night-median SNR bin x CHI2_CORE class (3 classes, at least 30 stars per cell).
- **Why rank 1**: one component carries 90-96.5 % of the variance, and a_j follows the FWHM (r = -0.95 to -0.99 on 11 of 13
  nights).
- **Corrected columns**: `FLUX_APER_1` and `FLUXERR_APER_1`; the error is scaled with the flux, so the SNR is unchanged.
- **New columns**:
  - `FLUX_PSF_RAW`: the raw flux.
  - `FZP_MAG`: the applied correction, in mag (> 0 brightens).
  - `FZP_A`: a_j, 0 when the frame is uncorrected.
- **Per-night outputs**: `psf_faint_zp.csv`, and a `faint_zp` entry in `psf_flag_counts.json`.
- **When a night or frame is left uncorrected**:
  - Status `few_frames`, `few_stars`, `few_cells`, `no_signal` or `insignificant`: std(a_j) < 3 x the median sigma.
  - A frame with fewer than 40 stars, or |a_j| > 1 mag.
- **Switch**: `PSF_FAINTZP=0` turns it off; this was checked bit-identical on 1104 (45 catalogues).

| version | rule | result | verdict |
|---|---|---|---|
| 709ed99 ("v1"; scratch `psf_v2_rejected`) | every star; reference cell = brightest SNR bin, lowest chi2 | catalogue scatter 140-195 > 10-15 mmag, but **1104 floor 4.6 > 7.5 mmag** | rejected |
| 7c85220 (v3) | SNR ceiling 100: stars above it are never corrected and define zp_j; SNR bins <= 0.2 dex | 1104/1105/1209 fine (4.7/3.8/4.0), but **1112 3.7 > 4.0, 1118 3.6 > 3.9**, bright bins +10-17 % | rejected |
| 65b8d39 (v4) | ceiling SNR 30; a cell is corrected only if the model halves its per-frame scatter (`FZP_GAIN` 0.5) | acceptance passed (below) | in production |

- **Why v1 failed.** CHI2_CORE rises with flux in the top SNR bin, so stars above 1e5 counts landed in a high-chi2 cell with
  h = 0.0385. 55-97 % of them were shifted by up to 26 mmag, and bright comparison stars moved. The original neutrality check
  (median over > 10000 counts) hid this.
- **Why v3 failed.** On weak-bias nights the rank-1 h is normalised to the faintest cell. The comparison pool (SNR 10-100) then
  got median h +0.058 (1112) and +0.153 (1118), against about 0 on strong-bias nights. The non-seeing part of a_j became a
  coherent ensemble shift of 1.2-2.6 mmag, which matches the added noise.
- **v4 acceptance criteria**, set before the run:
  - bright floor not worse than v1 by more than 0.2 mmag;
  - identical frames kept;
  - faint bins improve;
  - similar candidate counts;
  - bright bins (-14.5 to -12) within +2 %;
  - injection recovery >= 0.95.
- **v4 results** on 7 nights:

  | night | floor v1 / v3 / v4 (mmag) |
  |---|---|
  | 1104 | 4.6 / 4.7 / 4.6 |
  | 1105 | 3.9 / 3.8 / 3.7 |
  | 1106 | 3.8 / – / 3.6 |
  | 1112 | 3.7 / 4.0 / 3.7 |
  | 1118 | 3.6 / 3.9 / 3.6 |
  | 1201 | 4.0 / – / 4.0 |
  | 1209 | 3.9 / 4.0 / 3.9 |

  - Bright bins: at most +1.7 %.
  - Faint bins (-9.5 to -8), rms v4 / v1: 0.76-0.85 on 1104, 0.90-0.95 on 1105 and 1106; 1112 and 1118 are nearly uncorrected
    by design (0.95-1.00).
  - Candidates: 1104 152 > 150, 1105 451 > 441, 1209 432 > 434.
  - Catalogue scatter on 1104 at 1500-2500 counts: 190 > 16 mmag.
- **Injection** (3 % box, catalogue level): pooled recovery >= 0.985 on every night. On 1104 the per-star 5th percentile is 0.933,
  below 0.95: a 1 h box covers 60 % of that 1.7 h night.
- **Neutrality**: epochs above 1e5 counts with `FZP_MAG` != 0 are 0 of 16935 / 28052 / 24886 on 1104 / 1105 / 1209 (every 6th
  frame).
- **Production**: status ok on all 13 T80S nights, with no uncorrected frames.
  - The `cells` count in the log (33-42) is the number of cells built, not the number kept by the gain test, which is not logged.
  - Kept cells, estimated afterwards from the distinct corrections (inferred): 22-35 per night, but 14 on 1112 and 9 on 1118
    (the weak-bias nights).
- **Runtime**: finalize 2-6 min per night.
- **ROBO43 20250911** stays on the 709ed99 finalize.
  - Its zero point was `insignificant` (669 stars, 12 cells), so all 351 frames are uncorrected and the products are numerically
    the v1 ones.
  - Floor 5.6 mmag, 314/351 frames, 6 transit candidates.
  - v4 would also leave it uncorrected (`no_signal`), so it was not re-finalized. A richer ROBO43 night could pass the
    significance test.

## Operations

- **Redo runs** (`/ssdsto1/data/psf_redo_all.log`; per-night logs in `/ssdsto1/data/<TEL>_reduced/psf_redo_logs/`):
  - First chain (709ed99): ROBO43, then 1104; stopped during 1105 when v1 was rejected. 1104 was rolled back.
  - v3 chain: 1105, 1106, 1112, 1118, 1201; paused when v3 was rejected.
  - Chain 2 (65b8d39, 19:59-22:12): re-fits of 1107, 1207, 1104, 1130, 1206 and 1209 (11-24 min each), and re-finalizes of
    1204 and 1208.
  - Second pass (`PSF_REDO_OLD=.../psf_v3`, 22:12-23:16): 1105, 1106, 1112, 1118, 1201.
  - Bin cap (`relphot` mode, `PSF_REDO_OLD=.../psf_v4_prebincap`): 1130 and 1207 (23:17-23:47).
- **Superseded products** (scratch, NFS):

  | path | size | content |
  |---|---|---|
  | `psf_v1` | 39 GB | pre-fix psf and relphot of all 14 nights, plus the old multinight products |
  | `psf_v2_rejected` | 4.0 GB | the 709ed99 products of 1104 and a partial 1105 |
  | `psf_v3` | 12 GB | 1105, 1106, 1112, 1118, 1201 |
  | `psf_v4_prebincap` | 2.4 GB | relphot of 1130 and 1207 |

  Nothing in the DB points there. The user decided to delete them after this report.
- **Tile 9** (1130, 1207):
  - The robo43 master list is sparse in the top-left corner, so tile 9 had only 45 comparison candidates.
  - The 20 noise-floor bins held about 2 stars each and were all dropped, which gave a NaN floor and the "relaxed, keeping lowest 5
    scores" fallback.
  - With the cap: 5 > 43 (1130) and 5 > 37 (1207) comparison stars, floors unchanged. The cap is bit-identical when
    n_valid >= n_bins x min_bin_stars.
- **Transit fits lost on reload**:
  - `load-night` recreates the search detections, and the foreign key cascade deletes their `transit_shape` rows. They come back
    only with `analyze`.
  - Between the night reloads and the tie, 1666 events on 10 nights had no fit, and the web drew the search box ("rectangles").
  - The final analyze restored all 2265. ROBO43 was repaired by hand first.
  - The new label makes such a gap visible. **A night reload must always be followed by `analyze`.**

## Per night (final state)

From `driver/cmp_nights.py`: PSF v1 (the run of `psf_rerun_REPORT.md`) against the final products. Columns:

- **floor**: relphot bright-star floor, in mmag.
- **kept**: frames kept / frames.
- **transit / variability**: candidates of `relphot search`.
- **bit 64**: share of epochs with the PSF poor-fit flag.
- **relaxed**: comparison tiles that fell back to "keeping lowest 5 scores".
- **fzp**: faint-star zero-point status and the number of cells built (before the gain test).

| night | id | floor | kept | transit | variability | bit 64 % | relaxed | fzp |
|---|---|---|---|---|---|---|---|---|
| 20251104 | 2 | 4.6 > 4.5 | 42 > 44 /45 | 152 > 153 | 5321 > 4963 | 1.40 > 0.85 | 0 > 0 | ok / 39 |
| 20251105 | 3 | 3.9 > 3.7 | 65 /73 | 451 > 441 | 3796 > 3779 | 0.78 | 0 | ok / 39 |
| 20251106 | 4 | 3.8 > 3.6 | 64 /69 | 145 > 140 | 2496 > 2508 | 0.94 | 0 | ok / 33 |
| 20251107 | 5 | 5.5 > 5.5 | 65 > 64 /67 | **3622 > 241** | 11671 > 6373 | **4.06 > 0.61** | **17 > 0** | ok / 42 |
| 20251112 | 6 | 3.7 | 23 /27 | 75 > 72 | 9519 > 9486 | 0.93 | 0 | ok / 39 |
| 20251118 | 7 | 3.6 | 23 /27 | 30 > 26 | 8360 > 8360 | 0.73 | 0 | ok / 42 |
| 20251130 | 8 | 3.9 | 27 /27 | 134 > 162 | 9551 > 8683 | 1.32 > 0.82 | 0 | ok / 42 |
| 20251201 | 9 | 4.0 | 37 /41 | 40 > 41 | 6576 > 6581 | 0.81 | 0 | ok / 42 |
| 20251204 | 10 | 4.5 | 40 /40 | 76 > 75 | 5214 > 5209 | 0.59 | 0 | ok / 42 |
| 20251206 | 11 | 4.9 > 4.8 | 41 /42 | 54 > 49 | 7182 > 6357 | 0.97 > 0.62 | 0 | ok / 39 |
| 20251207 | 12 | 6.0 > 5.9 | 67 /67 | **3388 > 390** | 11327 > 7001 | **3.31 > 0.75** | **18 > 0** | ok / 39 |
| 20251208 | 13 | 4.0 | 36 /42 | 43 > 39 | 9142 > 8894 | 1.01 | 0 | ok / 36 |
| 20251209 | 14 | 3.9 | 66 /67 | 432 > 430 | 7828 > 7453 | 1.07 > 0.67 | 2 > 0 | ok / 42 |

- **1107 and 1207**: the transit candidates drop by 93 % and 88 %. Bit 64 is back to the level of the clean nights.
- **Relaxed tiles**: none remain on any night. The registration fix removed the bit-64 floods, and the bin cap fixed tile 9 of
  1130 and 1207 (below).
- **Floors**: unchanged or lower on every night (1105 and 1106 by 0.2 mmag).
- **Variability candidates**: they drop on the re-fitted nights. They are still inflated by faint stars on every night: the zero
  point only removes the frame-to-frame common mode (see Not verified).

## Multi-night tie and analyze

`scripts/psf/tie_psf.sh` produced `mn_1104_1209_loose_psf` again (core 1104-1107, the other nine nights loose).

- **Steps**: multinight 140 s, multisearch 477 s, load-multinight 33 s, analyze 510 s.
- **Core floor per magnitude bin**, in mmag: v1 [13.71 18.05 39.37 52.95 66.91 99.55], now [14.0 16.79 36.45 49.79 61.07 89.67].
  5 of the 6 bins are lower.
- **Per-night bright-bin floor (bin 0)**, v1 > now, in mmag:
  - Core: 1104 6.01 > 5.41, 1105 12.12 > 13.23, 1106 24.76 > 26.71, 1107 11.94 > 10.64.
  - Loose: 14.76-20.89 > 15.44-20.88; 1201 -1.1, 1204 -1.6, 1206 +0.7.
- **Holdout chi2**: about 1.0 on every night. NNLS clip warnings went from 3 to 6.
- **The night-to-night calibration is still worse than in the aperture era** (`psf_rerun_REPORT.md`, tie table). The faint-star
  zero point did not change that.
- **analyze**:
  - 2265 transit events, each with its trapezoid fit.
  - 762 auto-rejected: coincidence 706, no baseline 77, no dip 11, edge outlier 5 (one event can have several reasons).
  - 17 repeated-event families on 15 objects.
- **Before the VARIABILITY rule**, 1503 transit events are open (not auto-rejected).
- **Classes**: EXOP 1183, EXOP+VAR 255, VAR 4964, UNC 197571. In v1 they were EXOP 1817, EXOP+VAR 423, VAR 6041.
- **WASP-145 A b** (obj 232): SNR 8.19, depth 0.96 %, T14 0.88 h. This is unchanged.

## Dip tests: is a transit event the star's own variability?

The question: many variable stars get their regular dips listed as transits. Can those be recognised before the EXOP fit? Three
event-level tests were measured on the final DB (all 2265 transit events), with a prototype in scratch
(`/ssdsto1/data/mnt/diptest/`). The tests now in the code are `src/relphot/dip_variability.py`.

- **Shape**: a box plus a 2nd-degree polynomial (6 parameters) against the best smooth curve (3rd- or 4th-degree polynomial, or
  a line plus a sinusoid with a period of at least 3 T14), by BIC. No period is needed.
- **Phase**: fold the star's other nights at a period, with a free offset per night. Predict the event night, and compare the
  predicted box depth with the observed one in the event window. "Explained" = coverage >= 0.8, predicted/observed >= 0.7, and a
  better chi2 than a flat baseline.
- **Repeat**: other transit events of the star at |dt - nP| (or odd multiples of P/2) within half a duration. The chance
  probability of that many matches is a Poisson-binomial tail.

**Event populations.** A: stars with a catalogue variable type, or with a variability detection on that night (360 events).
B: all the others (1905 events). Sub-classes of A: RR 79, ECL 24, CEP 10, LPV 76, other catalogue types 13, variable only on that
night 158.

**Injections.** 2700 trapezoid transits were injected into real light curves of 1545 stars, 0.5-5 % deep with T14 0.6-1.5 h, on
nights without an event. A scan with the search's grid recovered 899 at SNR >= 6 (2.2 / 6.1 / 34.8 / 56.9 / 66.5 % at 0.5 / 1 / 2 /
3 / 5 % depth). The same scan on the uninjected light curves found 15 events (the control).

Fraction flagged (flagged / testable):

| test | A (variables) | B (others) | ECL | RR | LPV | injected, recovered |
|---|---|---|---|---|---|---|
| shape, dBIC < 0 | 133/360 (37 %) | 441/1905 (23 %) | 20/24 | 66/79 | 11/76 | 213/899 (24 %) |
| phase >= 0.7 | 11/156 | 0/206 | 10/17 | 1/23 | 0/45 | 7/359 (2 %) |
| repeat p <= 0.01 | 11/58 | 0/70 | 11/17 | 0/10 | 0/15 | 0/88 |

- **The shape test cannot be a rule.** It also flags a quarter of the injected transits. At box SNR >= 12 it flags 3 % of B and
  7 % of the injected transits on variables, so it is useful as information only.
- **The phase and repeat tests separate the eclipsing binaries** from everything else:
  - They never fire on B.
  - On injected transits, phase fires on 7 of 207 testable ones on stars with a catalogue period. They are mostly shallow: 3/8 at
    0.5 %, 2/15 at 1 %, 1/71 at 2 %, 1/123 at 3 %, 0/142 at 5 %. Repeat never fires.
- **A fitted period is no substitute for a catalogue one.** With the LS period of the tied light curve (FAP <= 1e-3), the phase
  test flags 0 of 157 events. Many of those periods are about 1 d aliases.
- **Rule chosen**: phase or repeat, with a catalogue period only. It flags 15 events (ECL 14, RR 1) and nothing in B; 2 of the 15
  were already rejected by the coincidence rule.
- **Not covered**: most of A (variable only on that night: 158; LPV: 76) is not a periodic dip, and no test explains it.
  Subtracting a variability model (the plan in `PLAN_IMPROVE_DETECT.md`) would not help those either.
- **Small control sample**: only 15 control events (dips of the variability alone, without injection). This is a small sample.
- **Figures**:
  - `figures/psf_fix/phase_flagged_det95640.png`: contact binary, P 0.276 d; the folded other nights reproduce the dip.
  - `figures/psf_fix/repeat_flagged_det92649.png`: repeat test.
  - `figures/psf_fix/shape_flagged_det95462.png`: a flux jump that a smooth curve fits better.

## VARIABILITY auto-verdict rule (commit db2607a)

- **Where**: `relphot db analyze`, in the per-night verdict pass (`relphot.db.coincidence.update_auto_verdicts`). It is the fifth
  reason after COINCIDENCE, EDGE_OUTLIER, NO_DIP and NO_BASELINE.
  - Pure numerics: `src/relphot/dip_variability.py`, a port of the prototype.
  - DB gatherer: `coincidence.variability_reasons`, four batched queries per night.
- **Eligible event**: a search transit event of an object with a catalogue period (`catalog_match`, not a planet catalogue,
  `period > 0`, not VSX type EP; the row nearest on the sky).
  - An object with any planet-catalogue row (NASA Exoplanet Archive, TOI) is never judged.
  - The event's own tc and duration are used, so no trapezoid fit is needed.
- **Fires on** phase (`db.auto_var_phase_cov_min` 0.8, `db.auto_var_phase_ratio_min` 0.7, and beats a flat baseline) or repeat
  (`db.auto_var_repeat_p_max` 0.01).
  - Repeat counts the object's search events on other nights, whatever their status.
- **Same contract as the other rules**:
  - Vetted events are not judged.
  - A person's CONFIRMED overrides the rule.
  - Re-running is idempotent; a reason is cleared when the rule stops firing.
  - No schema change.
- **Reason text**, for example: `variability: the catalogue period P=0.3472 d (VSX|Gaia DR3|ASAS-SN (merged)) predicts this dip
  from 10 other nights folded (predicted/observed depth 0.88, phase coverage 100 %)`.
- **Validation**:
  - 32 new tests (12 pure, 20 DB, including an `analyze()` end-to-end test). The full suite passes: 811 tests, ruff clean.
  - Read-only on the production DB, the rule fires on exactly the 15 measured events:

    | object | events | type |
    |---|---|---|
    | 672 | 7 | ECL |
    | 23466 | 3 | ECL |
    | 23705 | 3 | ECL |
    | 22088 | 2 | ECL |
    | 101804 | 1 | RR |

  - It adds about 0.2 s to the verdict pass of all 14 nights.
- **Deployment**:
  - The reprocess worker was restarted on the new code. It had been running code from 2026-10-04 08:14, so an old worker would
    have cleared the new reasons on any night it re-analysed.
  - **`relphot db analyze --all --keep-vetted` was not run here** (the production run is left to the user). Expected:
    `variability=15`, `auto_rejected` 762 > 775.

## Long-period variables: a slope would not help (not implemented)

The proposal: remove a slope from LPV light curves before the EXOP search, to cut their false transits.

- **The per-night search already does this, jointly with the box**: `transit_search` fits poly(t, 2) + CBVs + box
  (`search.poly_degree` = 2, with no override in production). Within one night an LPV (P 40-700 d) is a straight line to high
  accuracy.
- **The 58 LPVs flagged EXOP all come from per-night transit events.** Multinight BLS detections do not set the flag
  (`class_multinight_kinds` = recurrent).
- **At the same magnitude, LPVs have the same transit-event rate as non-variables**:

  | relphot mag | non-variables with an event | LPVs with an event |
  |---|---|---|
  | -12.5 to -12.0 | 2.9 % of 2947 | 2.4 % of 873 |
  | -12.0 to -11.5 | 2.8 % of 6156 | 3.1 % of 516 |
  | -11.5 to -11.0 | 1.6 % of 9729 | 1.5 % of 265 |
  | -10.0 to -9.5 | 0.5 % of 45813 | 0 of 27 |

  LPVs are simply concentrated at bright magnitudes.
- **What the events look like**, for the 24 strongest (`figures/psf_fix/lpv_top24.png`):
  - Mostly abrupt flux steps of 2-30 %, and dips that follow the PSF FWHM.
  - Example: det 93793, a 29 % dip while the seeing is sharpest, correlation with FWHM 0.57.
  - This points to a bright-star systematic (saturation or non-linearity), not variability.
- **Candidate follow-up (not measured)**: a bright-star test applying to every star, measured with injections like the dip tests.
  Either a step model against a box, or the dip's correlation with the frame FWHM.

## Decisions recorded

- **The VARIABILITY rule is in; the shape test is information only.**
- **The variability-subtracted EXOP search** (plan in `PLAN_IMPROVE_DETECT.md`, decisions a-h): deferred by the user on
  2026-10-05.
- **Bright-star false events**: the step and seeing tests are being measured; the result will go in its own report.

## Where things are

| What | Path |
|---|---|
| PSF products (v4 zero point; re-fitted nights with the shift field), relphot, PROVENANCE.txt | `/mnt/sto01/<TEL>/reduced/<night>/{psf,relphot}/` |
| Multi-night tie | `/mnt/sto01/T80S/reduced/multinight/mn_1104_1209_loose_psf*` |
| Redo logs and markers | `/ssdsto1/data/psf_redo_all.log`, `/ssdsto1/data/<TEL>_reduced/psf_redo_logs/` |
| DB backups (pinned) | `/ssdsto1/data/relphotDB/backups/pre_psf_fix_20261004_1431.dump` (with test_psf), `post_psf_fix_20261005_0009.dump` |
| Figures | `tests/reports/figures/psf_fix/` |

The scratch of this work was deleted after this report: the local `/ssdsto1/data/mnt/psf_fix` (prototypes, faint-ZP test runs,
68 GB) and the dip-test prototype. The NFS `psf_v*` directories are left for the user to delete.

## Not verified

- **No relphot-level injection-recovery** on the corrected faint fluxes; catalogue level only.
  - Faint-star false positives could not be shown to drop: no transit candidate is fainter than mag -9, in v1 either.
- **Weak-bias nights keep their faint seeing bias by design** (1112, 1118, 1204): few or no cells are corrected there. The
  cross-night faint-flux scale was not checked.
- **v4 acceptance ran on 7 nights.** The other six were checked only through the production comparison table.
- **ROBO43** was not re-finalized with v4, and was not re-fitted.
- **The cause of the bad WCS frames** (few matched astrometry stars) is not proven. The refine-flip noise (`MAXSHIFT_REF`) is not
  addressed.
- **VARIABILITY rule**:
  - Measured on one snapshot of the DB.
  - Its reason text has not been seen in a browser.
  - It has not been run on production (`analyze` was left to the user).
- **The dip tests' no-injection control** is only 15 events.
- **No web page was clicked in a browser**; the API and the served `app.js` were checked.
