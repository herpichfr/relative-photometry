# Bright-star false transit events: event-level tests (2026-10-05)

**Question.** Bright stars carry far more transit events than faint ones. Measured as the share of stars with at least one live
(not auto-rejected) event:

| relphot mag | < -12 | ... | faintest bin |
|---|---|---|---|
| share of stars | 3.1 % | 2.8, 1.6, 1.2, 0.85, 0.48 % | 0.25 % |

Long-period variables follow the same curve as non-variables (`psf_fix_REPORT.md`). Is there an event-level test that removes
these events while keeping real transits, suitable as an auto-reject rule?

**Answer: no strong rule exists.**

- The best usable rule removes 17 % of the live bright events at about 2 % false flags on injected 2-5 % transits.
- It was **not implemented**; the user decides.
- The bright excess has no shape signature. Saturation is not supported.

Measured on the post-PSF-fix DB (read-only snapshot, before the VARIABILITY rule). Scratch prototype: `/ssdsto1/data/mnt/brighttest/`,
deleted after this report.

## Method

- **Events**: 1490 live transit events (auto_status NULL) of 2265, grouped by the star's relphot mean magnitude.

  | group | magnitude | live events |
  |---|---|---|
  | B1 | < -11.5 | 412 |
  | B2 | -11.5 to -11 | 176 |
  | M | -11 to -10 | 442 |
  | F | fainter | 460 |

- **Injections**: 28000 trapezoid transits into event-free star-nights of four pools of 1500 stars each (B1, B2, M, F).
  - 0.5-12 % deep, T14 0.6-1.5 h.
  - Every 4th one is placed 20-60 % outside the night (the "partial" class).
  - 8151 recovered at SNR >= 6 (poly(2) + box on the search grid). The headline false-flag set is the 1910 recovered 2-5 % deep
    injections on bright stars.
- **Statistics**:
  - **Step**: a free-position step against the box (dchi2).
  - **Offset**: the flank level after the box minus the level before it, divided by the box depth.
  - **Seeing**:
    - dchi2_fw: poly(2) + c * FWHM against the box;
    - the box significance kept after adding the FWHM term;
    - box/FWHM and residual/FWHM correlation;
    - the FWHM z-score inside the box.
  - **Edge and coverage**.
  - **Negative controls**: in the same night, the best brightening SNR and the best other dimming SNR, divided by the event SNR.
  - **Stored flags**: the R90 and SHARED_EPOCH flags, for real events only. Their inputs (per-frame background and centroids) are
    not in the DB, so they cannot be injected.
- **WASP-145 A b** (obj 232) must not be flagged.

## Results

% flagged. Live columns are real events, injected columns the recovered 2-5 % deep injections, W = WASP-145 A b flagged:

| test, threshold | live B1 | live B2 | live M | live F | inj B1 | inj B2 | inj M | W |
|---|---|---|---|---|---|---|---|---|
| step dchi2 < 0 | 9.5 | 10.2 | 14.9 | 15.4 | 0.4 | 1.5 | 1.4 | no |
| step dchi2 < 3 | 47 | 47 | 52 | 48 | 6.5 | 13.9 | 14.0 | no |
| offset \|ratio\| > 0.7 | 5.6 | 6.2 | 7.5 | 10.4 | 1.0 | 0.6 | 1.1 | no |
| dchi2_fw < 10 | 4.6 | 6.8 | 11.3 | 12.8 | 0.4 | 0.5 | 0.4 | no |
| dchi2_fw < 30 | 25.5 | 33.0 | 34.4 | 39.3 | 4.4 | 10.3 | 12.4 | **yes** |
| box significance kept < 0.5 | 1.9 | 1.1 | 3.4 | 0.7 | 0.4 | 0.8 | 0.0 | **yes** |
| r(box, FWHM) < -0.5 | 3.4 | 2.8 | 2.7 | 1.3 | 1.1 | 1.5 | 0.2 | **yes** |
| brightening / event SNR > 1.1 | 6.6 | 6.2 | 5.7 | 5.2 | 0.6 | 0.8 | 1.6 | no |
| edge (coverage < 0.9 or box within 0.1 h of the night edge) | 49 | 52 | 50 | 44 | 6.7 | 9.7 | 11.2 | no |
| R90 any (stored) | 37 | 42 | 47 | 44 | – | – | – | no |

Rules (combinations), with % flagged:

| rule | live bright | inj B1 | inj B2 | inj M | inj F | inj 2 % | inj 3 % | inj 5 % | W |
|---|---|---|---|---|---|---|---|---|---|
| offset | 5.8 | 1.0 | 0.6 | 1.1 | 4.2 | 1.9 | 1.4 | 0.1 | no |
| offset or negative control | 13.3 | 1.6 | 1.4 | 2.7 | 7.6 | 3.5 | 2.1 | 0.4 | no |
| **A** = offset or negative control or dchi2_fw < 10 | **17.2** | 2.0 | 1.9 | 3.0 | 9.3 | 4.8 | 2.6 | 0.6 | no |
| A with dchi2_fw < 15 | 21.3 | 2.3 | 2.3 | 3.6 | 13.6 | 6.1 | 2.9 | 0.6 | no |
| edge | 50.0 | 6.7 | 9.7 | 11.2 | 15.3 | 11.2 | 7.8 | 6.9 | no |
| A or (edge and SNR < 7) | 33.8 | 3.7 | 5.9 | 7.8 | 18.6 | 9.9 | 5.3 | 2.4 | no |

**Rule A** (bright stars only, mean mag < -11). Auto-reject if any of:

- |off_ratio| > 0.7;
- brightening/event SNR > 1.1;
- other-dimming/event SNR > 0.9;
- dchi2_fw < 10.

| quantity | value |
|---|---|
| live bright events removed | 17 % (B1 16.0 %, B2 19.9 %), about 100 events |
| false flags, 2-5 % injections | about 2 % on bright stars (pooled 95 % CI 1.3-2.7 %, cluster bootstrap) |
| false flags by depth (0.5 / 1 / 2 / 3 / 5 / 8 / 12 %) | 34.6 / 9.8 / 4.8 / 2.6 / 0.6 / 0.9 / 0.5 % |
| false flags on faint stars | 9.3 %, so it must not be used on faint stars |

- It meets the <= 3 % target only from 3 % depth up.
- WASP-145 A b is not flagged.

## Findings

- **The bright excess has no shape signature.** Edge or partial events are 44-52 % of live events at every magnitude. Only the
  rate changes with brightness, not the composition.
- **Every seeing correlation fails.** The box/FWHM correlation, the FWHM z-score and the significance kept after adding the FWHM
  term have the same distributions for live bright events as for injections (`figures/bright_events/stat_distributions.png`).
  - At <= 3 % false flags they remove 1-5 %, which is chance level.
  - Every cut that flags more than chance also flags WASP-145 A b: its box/FWHM correlation is -0.54, its FWHM z-score -1.14 and
    its significance kept 0.43 (a planet transit can coincide with sharp seeing).
  - dchi2_fw is the only informative seeing-type statistic, and it really tests "is the event smooth".
- **Saturation is not supported.** On uninjected star-nights, a flux-FWHM correlation |r| > 0.5 occurs on 4.3 % of bright and
  2.1 % of mid stars, with no preferred sign. The visual impression from the LPV events in `psf_fix_REPORT.md` was not confirmed.
- **The step test adds nothing** beyond the offset test, and no step threshold reaches <= 3 % false flags with useful removal.
- **The edge rule is a policy choice, not a test.** It removes half of all events, but it rejects 91 % of deliberately partial
  transits: a partial transit cannot be told from a step.
- **An SNR cut is not a lever.** SNR < 8 removes 71 % of the live B1 events and loses 38 % of the injections.
- **The stored flags already mark most bright events.** 86 % of live bright events carry an informational flag (EDGE or PARTIAL
  60 %, SHARED_EPOCH 63 %, R90 37 %).
- **After edge or offset, bright stars still keep their excess**: 1.5 % of stars at mag < -12 against 0.1 % faint. Clean tier-1
  events: 0.75 % against 0.06 %.

**Figures** in `figures/bright_events/`:

- `star_rate_vs_mag.png`: share of stars with an event, against magnitude.
- `removal_vs_falseflag.png`: events removed against false flags, for every test and threshold.
- `stat_distributions.png`: live against injected distributions.
- `example_wasp145.png`: WASP-145 A b.

## Options (not implemented; user decision)

1. **Do nothing; vet with the stored flags.** They already mark 86 % of the bright events.
2. **Rule A as an informational flag** (not an auto-reject), on stars brighter than mag -11. Before using it, re-check the
   thresholds with injections through the real search (CBVs and red-noise beta change the SNR scale).
3. **Rule A as an auto-reject**, after that re-check and a held-out injection set. It saves about 100 events, at about 2 % of real
   2-5 % transits on bright stars (4.8 % at 2 % depth).

## Not verified

- **Not the real search's model**: the fits have no CBVs and no red-noise beta, so the SNRs are not the search's.
- **Thresholds not held out**: they were chosen on the same injection set they were scored on.
- **Clean injection hosts**: event-free star-nights with p2p <= 0.015 (0.03 for F), cleaner than average. Injections within one
  star-night are not independent: the 1910 headline injections sit on 1325 stars and 1783 star-nights.
- **One known planet**: WASP-145 A b (dchi2_fw 28, against the threshold of 10).
- **R90 and SHARED_EPOCH false-flag rates** on injections cannot be measured from the DB.
