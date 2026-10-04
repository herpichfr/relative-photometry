# scripts/psf -- PSF photometry re-processing of the reduced nights

Forced PSF photometry (PSFEx model, joint stamp fits) at the fixed positions of the production forced catalogues, then the unchanged
relphot chain on it. The method, validation and flag design are described in `tests/reports/psf_test_REPORT.md`.

## Files

| File | What |
|---|---|
| `psfphot.py` | per-frame PSF fit (`fit_frame`), FLAGS bit definitions (`FLAG_DOC`) |
| `run_night.py` | per-night fit: SExtractor + PSFEx per frame, pass 1 (companion search on every `pass1_step`-th frame) -> `psf_work/static_extras.csv`, pass 2 (all frames) -> `psf_work/raw/*.pkl` |
| `finalize.py` | assigns bit 64, writes the PSF catalogues (`*_proc_forced_catalog.csv`) and the relative `*_proc.fits` links, flag statistics |
| `clip_lc.py` | sigma-clips outlier epochs of the light curves between `relphot lightcurves` and `relphot search` |
| `psf.sex`, `psfex.conf`, `psf.param`, `default.conv` | SExtractor / PSFEx configuration |
| `psf_night.sh TEL NIGHT` | production driver for one night (resumable, see below) |
| `rerun_all.sh [TEL:NIGHT ...]` | all nights one after the other, chronologically; stops at the first failure |
| `tie_psf.sh` | after all nights: T80S multi-night tie `mn_1104_1209_loose_psf`, multisearch, DB load, analyze |

## Layout of a re-processed night (`/mnt/sto01/<TEL>/reduced/<night>/`)

- `psf/` -- PSF catalogues `<stem>_proc_forced_catalog.csv` (production forced format; the PSF flux is in `FLUX_APER_1`, error in
  `FLUXERR_APER_1`; extra columns `CHI2_CORE`, `CHI2_STAMP`, `NSRC_FIT`, `SKY_FIT`, `NPEEL`, `DX_REFINE`, `DY_REFINE`, `FLAGS_APER`,
  `FLUX_APER_PROD_8ARCSEC`), relative links `<stem>_proc.fits -> ../<stem>_proc.fits`, `psf_flag_counts.json`, `psf_flag_mapping.txt`,
  `PROVENANCE.txt`, `psf_rerun.log`, and `psf_work/` (PSFEx models `psfex/`, `static_extras.csv`, `pass1/`, `raw/` pickles, `log/`).
- `relphot/` -- relphot run on the PSF catalogues (`relphot ingest --photometry forced`), light-curve stem per telescope (below).
- The old aperture products (`forced/`, old `relphot/`, `*_proc_catalog.csv`, photometry/forced logs, ...) are moved to
  `/mnt/sto01/scratch/aperture/<TEL>/reduced/<night>/`. `*_proc.fits`, `calib`, `raw_calib` are never touched.

## FLAGS (relphot masks `FLAGS & 252`, i.e. bits 4-128)

| Bit | Meaning |
|---|---|
| 1 | another source (master row or unmatched per-frame detection) within 4.0 arcsec (production flag_radius) |
| 2 | fitted as a blend: another source within 1.4 arcsec (master row or per-frame detection), or a companion added from the residual image of the stamp |
| 4 | saturated or non-linear DQ pixel within 4.0 arcsec of the source (pixels also excluded from the fit) |
| 8 | source centre within 20 px of the detector edge (production edge_margin_px; stamp truncated) |
| 16 | hot/dead/unstable/no-data/bad-column pixel within 1 PSF FWHM of the source, or >20% of the stamp masked in the fit (incl. repaired cosmic-ray pixels) |
| 32 | fit failed (frame without PSF model, <50% usable stamp pixels, singular normal matrix, non-finite result) |
| 64 | poor fit: core chi2 above threshold relative to the star's own median and the frame (assigned in finalize) |
| 128 | flux non-finite or <= 0 (or non-finite/<=0 error) |

## Per-telescope settings (in `psf_night.sh`)

| | T80S | ROBO43 |
|---|---|---|
| fit workers (`NP`) / pass-1 step | 13 / 7 | 12 / 50 |
| light-curve stem | `night_lc` (relphot multinight hard-codes `lc/night_lc.npz`) | `wasp145_psf` |
| `relphot reference` | `--no-variables` | `--aper 0` |
| `relphot lightcurves` | `--no-variables` | (none) |
| `relphot search` | (none) | `--snr-threshold 5.5` |

All other relphot steps: `ingest --photometry forced`, `lightcurves` with plots, `clip_lc.py <dir> <stem> 4.0 15`, `search --no-plot`.

## Running

```
scripts/psf/psf_night.sh T80S 20251104        # one night
scripts/psf/rerun_all.sh                       # all 14 nights (ROBO43 20250911, 13 T80S nights), one at a time
scripts/psf/rerun_all.sh T80S:20251112         # only the listed nights
scripts/psf/tie_psf.sh                         # after all T80S nights
```

`psf_night.sh` steps: `pre FIT FINALIZE MOVE_APERTURE SYNC_PSF RELPHOT SYNC_RELPHOT DB_LOAD CLEAN`. Fit, finalize and relphot run in the SSD
staging dir `/ssdsto1/data/<TEL>_reduced/<night>` and are synced to the permanent dir with `deploy/relocate_reduced.sh --sync`; `db load-night`
runs on the permanent copy. Logs and one marker word per finished step: `/ssdsto1/data/<TEL>_reduced/psf_rerun_logs/<night>.{log,state,mem.log}`;
a rerun skips the steps with a marker. Failure line: `NIGHT_FAIL <TEL> <night> <step>`; success: `NIGHT_DONE`. The start waits for
MemAvailable > 30 GB; a guard kills the fit/relphot processes (`GUARD_KILL`) below 4 GB.

The fit scripts alone (environment: `PSF_OUT` required; `PSF_PROD`, `PSF_FORCED`, `PSF_CATDIR` optional):

```
PSF_OUT=<dir> python run_night.py <TEL> <night> <nproc> [pass1_step=7] [max_frames]   # ends with RUN_NIGHT_OK
PSF_OUT=<dir> python finalize.py <TEL> <night> [rel_thr=auto]                          # ends with FINALIZE_OK
```
