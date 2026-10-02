# relphot results database -- operator guide

This is the deployment and day-to-day operator guide for the results database
introduced in `docs/DB_PLAN.md` (design/decisions/schema) and built by
`relphot db ...` (loader, runs on the host) plus `relphot.web` (query API +
front end, runs in a container). Read `docs/DB_PLAN.md` first for *why*
things are shaped this way; this file is *how to run it*.

## What runs where

| Component | Where | Image | Port | Notes |
|---|---|---|---|---|
| `relphotdb-db` | podman container (rootless) | `localhost/relphotdb-db:latest` (PostgreSQL 17 + q3c) | `127.0.0.1:5433` -> 5432 | data in `/ssdsto1/data/relphotDB/pgdata` |
| `relphotdb-web` | podman container (rootless) | `localhost/relphotdb-web:latest` (FastAPI + Jinja2 + Plotly.js, vendored) | `127.0.0.1:8080` -> 8050 (host port configurable, see "Web usage") | stateless; talks to `relphotdb-db` over the `relphotdb.network` podman network |
| loader (`relphot db ...`) | host, plain CLI | -- | -- | connects to the DB over `127.0.0.1:5433`, same as any other client |
| `relphotdb-worker` (`relphot db reprocess --watch`) | host, systemd user unit | -- | -- | works off the web's RERUN queue (owner role, like the loader); see "User-guided reprocessing" |

Both containers are managed as podman quadlet user units (`~/.config/containers/systemd/relphotdb-db.container`, `relphotdb-web.container`, `relphotdb.network`) and run under `systemctl --user`, i.e. they start at login (`WantedBy=default.target`) and restart on failure (`Restart=always`).

Data directory: `/ssdsto1/data/relphotDB/` --
- `pgdata/` -- the PostgreSQL data directory (bind-mounted into `relphotdb-db`).
- `backups/` -- `pg_dump` output (see "Backup and restore" below).

Env file: `~/.config/relphot/relphotdb.env` -- written once by `deploy/install.sh db` (mode 600), holds the generated superuser/role passwords, `RELPHOT_DB_DSN` (the loader's default DSN, role `relphot_owner`), and `RELPHOT_TEST_DSN` (a second database, `relphot_test`, for the test suite). It is sourced as a systemd `EnvironmentFile` by both quadlet containers and read directly by `relphot.db.connect.resolve_dsn` when `RELPHOT_DB_DSN` is not already in the environment.

## Install

Everything is idempotent -- re-running any step is safe.

```
bash deploy/install.sh db       # env file + relphotdb-db container + roles/databases
bash deploy/install.sh web      # relphotdb-web container (needs `db` first)
bash deploy/install.sh backup   # relphotdb-backup.service/.timer (needs `db` first)
bash deploy/install.sh worker   # relphotdb-worker.service, the reprocess worker (needs `db` first
                                # and `relphot` on PATH: pip install -e '.[db]')
relphot db init                 # create/upgrade the schema (idempotent, versioned)
```

`install.sh db` generates and writes the env file only if it does not already exist, builds the `relphotdb-db` image, installs its quadlet unit, starts it, and waits for `pg_isready`. The image's first-boot init script (`deploy/db/10-relphot-roles.sh`) creates the `relphot_owner`/`relphot_web`/`relphot_ro` roles, the `relphot`/`relphot_test` databases (owned by `relphot_owner`), and the `q3c` extension in each.

`install.sh web` builds `deploy/web/Containerfile` (which `pip install`s `relphot[web]` from the repo as it stands **at build time** -- see "Upgrading" below), installs the quadlet unit, restarts the container, and waits for `/api/search?limit=1` to answer.

`install.sh worker` writes `relphotdb-worker.service` (a plain systemd user unit running the host's `relphot db reprocess --watch`, `Restart=on-failure`, `EnvironmentFile` = the env file above, so `RELPHOT_DB_DSN` is the owner DSN) to `~/.config/systemd/user/`, enables and restarts it. Run `relphot db init` first: the worker needs schema v6.

`install.sh backup` installs and enables `relphotdb-backup.timer`/`.service` (plain systemd user units, not quadlet -- see "Backup and restore") and runs one backup immediately.

`relphot db init` applies every schema migration in `src/relphot/db/sql/NNN_*.sql` newer than the database's recorded `schema_version`; a fresh database gets all of them, an existing one only the new ones. Run it once after `install.sh db`, and again after pulling in a relphot version that adds a migration. **After `relphot db init` applies version 006, run `relphot db analyze --all` to backfill the new per-night review columns** (`n_review_pending`, `n_nights_reviewed`). Version 007 adds `night.zp` / `night.zp_source` (the photometric zero point of a night, see "Web usage"): the nights already loaded get the assumed 20 mag (none of their frame headers carries a Gaia calibration), so no backfill is needed; a night picks up a calibration by being re-ingested (`relphot ingest` reads `ZPABS` into `night.npz`) and re-loaded. Version 008 adds the `'measured'` value to `night.zp_source` (the CHECK constraint only; no row is changed). A night loaded from now on gets, without a Gaia calibration, the telescope's measured zero point from `[db] telescope_zp` (default `T80S = 27.85`) instead of the assumed 20 mag; **the nights already loaded keep `'assumed'` until re-loaded, so after `relphot db init` applies version 008 run this one-off backfill as the owner role** (it uses the default `telescope_zp`; adapt the value and telescope if you changed it; one-time backfill of 2026-09-29):

```sql
UPDATE relphot.night SET zp = 27.85, zp_source = 'measured'
WHERE telescope = 'T80S' AND zp_source = 'assumed';
```

The web reads `night.zp` directly, so no `relphot db analyze` is needed.

Version 009 adds the automatic cross-candidate check of the per-night transit events (see "Coincident events" below): `detection.auto_status` / `auto_reason` and the table `relphot.transit_coincidence`. **After `relphot db init` applies version 009, run `relphot db analyze --all` once** to judge the nights already loaded (it re-judges every night that has a transit shape of a processed object; the `[db] coincidence_*` settings tune it). A night reloaded with `relphot db load-night` loses its verdicts with its search detections until `relphot db analyze` runs again.

Version 011 adds traceability of reference and comparison star sets used to build each tile's light curves (`relphot.night_tile`, `relphot.tile_lc`, `relphot.reference_member`, `relphot.comparison_member` tables). **After `relphot db init` applies version 011, backfill members** with `relphot members <night>/night.npz <night>/ref.npz <night>/lc/<stem>.npz` (recovers reference stars from the npz products) and then `relphot db load-members <night>/relphot [--lc-stem STEM]` per night (or use the `--no-members` flag to skip the automatic backfill in `relphot lightcurves` after that).

## Nightly workflow

Per night, per telescope, after the usual relphot photometry stages:

```
relphot ingest --out night.npz /path/to/*_proc.fits
relphot reference night.npz --out ref.npz --no-variables
relphot lightcurves night.npz ref.npz --out lc/<stem>
relphot search night.npz ref.npz lc/<stem>.npz \
    --transits-dir lc/transits --variables-dir lc/variables

relphot db load-night /path/to/<telescope>_reduced/<label>/relphot \
    [--lc-stem <stem>] [--telescope T] [--label YYYYMMDD]
```

`load-night` reads `night.npz`/`ref.npz`/`lc/*_starstats.parquet`/`lc/*_lightcurves.parquet`/`lc/*_search_metrics.parquet` from a night's `relphot/` directory, applies the noise cut (below), cross-matches stars to existing `object` rows (q3c, `db.match_radius_arcsec`), and stores light curves, detections, and catalogue matches. `--lc-stem` is only needed when more than one `*_starstats.parquet` is present in `lc/` (e.g. a `wasp145` and a `wasp145_all` stem side by side -- pick the one that has `*_search_metrics.parquet`, i.e. the one that actually ran `relphot search`). It is one transaction; reloading a night first deletes that night's `star_night`/`lightcurve`/`detection` rows, but a person's `status`/`notes` on those detections are saved and re-attached to the new detection of the same object and kind (a transit only if its `tc` is within half its duration of the old one). A verdict that cannot be matched is logged as a warning and kept in `relphot.detection_review_orphan`. Objects with a manual flag or period, a non-default status, or notes are never removed by a reload even when they no longer have any `star_night` row.

Optionally, once two or more nights of the same field are loaded, tie them together:

```
relphot multinight night1/relphot night2/relphot night3/relphot --out multinight/mn_<label>
relphot multisearch multinight/mn_<label>.npz --out-dir multinight/mn_<label>_search

relphot db load-multinight multinight/mn_<label> \
    --search-dir multinight/mn_<label>_search
```

`load-multinight` stores the `mn_run`/`tie` rows (per-night tied magnitudes, used by the web's "All nights" combined light curve) and, when `--search-dir` is given, the cross-night `internight`/`ls_periodic`/`bls`/`recurrent` detections from `multinight_search_metrics.parquet`.

Finally, after any load:

```
relphot db analyze [--all]
```

recomputes periodograms (per-night LS always; combined LS; combined BLS for planet-host-eligible objects), the transit-event analysis and period verification described in "Transit events, matching and period verification" below, and the derived summary fields (`n_nights`, `mean_mag`, `best_snr`/`depth`/`duration_h`/`amplitude`, `first_night`/`last_night`, the `is_exop`/`is_var` flags whose source is not manual, `class`, and -- for non-manual rows -- `period`). Without `--all` it only touches objects with no periodogram yet, or touched since their oldest one (a newly loaded night touches its objects, so each re-observation gets its period-verification row); `--all` forces every planet-host/variable/detected object (needed once after changing a `[db]` threshold, e.g. `class_multinight_kinds`, or after `relphot db init` applies a new migration, so every affected object is recomputed).

## Error inflation (blended and noisier-than-formal stars)

`relphot lightcurves` multiplies each star's `lc_err` by `err_scale = max(1, sigma_p2p / median(lc_err))` (per aperture), where `sigma_p2p` is the robust point-to-point scatter (1.4826 x MAD of the differences of consecutive kept epochs, / sqrt 2). It is insensitive to transits and slow variability (no trend is fitted to the star's own light curve, so it is transit-safe). Because it is applied at the light-curve stage, the search, the multi-night tie and the database all use the inflated error. `--inflate-errors` / `[lightcurve] inflate_errors` chooses who gets it: `none`; `blended` (SExtractor neighbour/blend FLAGS bits 1 and 2 set in at least `blend_min_frame_fraction` = 5 % of the star's kept frames -- no catalogue is available at that stage); `excess` (the default: blended stars plus any star whose measured factor is at least `err_scale_excess_min` = 1.5); `all`. The reason for the default: on ROBO43 20250911 the excess is mostly a bright-star noise floor that the formal (photon + ensemble) error lacks -- WASP-145 A (star 516) scatters by 6.6 ppt point to point against a formal 1.8 ppt (factor 3.7), and the unblended comparison stars of the same brightness show 3.4-3.8 -- so `blended` alone would leave it uncorrected. `*_lightcurves.parquet` keeps the formal error as `lc_err_raw`; `*_starstats.parquet` and `relphot.star_night` carry `err_scale` and `blended` (NULL for a night loaded from an older product, read as 1). `expected_noise` stays the formal prediction, so the noise cut below is unchanged; `chi2_reduced` is against the inflated error. Heavy tails (about 7 % of the bright stars' epochs sit more than 4 sigma low, single-frame dips) are not modelled by a MAD-based factor, so their `chi2_reduced` stays above 1.

## Noise cut

A star of a night is stored in the results database (`relphot.star_night`/`lightcurve`) iff, at its best aperture (from `*_starstats.parquet`):

1. `n_epochs >= SearchSettings.effective_min_epochs(n_kept_frames)` (the same good-epoch cut `relphot search` uses: `max(min_epochs, ceil(min_epoch_fraction * n_kept), 1)`), and
2. `expected_noise <= db.max_expected_noise` (default `0.05`, i.e. 50 mmag predicted per-point noise -- a photon/background prediction, so it does not reject real variables on the strength of their own scatter), and
3. `rms` is finite.

Exception: a star that is a `transit_candidate` or `variability_candidate` of that night's search (`*_search_metrics.parquet`) is always stored (`db.keep_candidates`, default `true`), even if it fails 1-3. Stars that fail the cut and are not a candidate are dropped entirely -- never written to the database.

On T80S 20251104-06 this keeps ~97% of stars; on ROBO43 20250911 (351 frames, `effective_min_epochs` = 176) it kept 382 of 520.

## CLASS and PERIOD rules

An object has two independent flags, `is_exop` (planet host) and `is_var` (variable); neither excludes the other, and `class` is only their derived label (`EXOP+VAR` if both, else `EXOP`/`VAR`/`UNC`). Each flag is set from the data unless its `exop_source`/`var_source` is `'manual'` (set by hand in the web), in which case `relphot db analyze` never touches it. `class_source` is `'manual'` iff either flag source is. `object.status` and `detection.status` are never set automatically. `is_exop`: known planet match, or a transit detection, or a multi-night `'bls'` detection in `db.class_multinight_kinds`. `is_var`: known variable match, or a variability detection, or a multi-night `'internight'`/`'ls_periodic'`/`'recurrent'` detection in `db.class_multinight_kinds`. A star that is a variability candidate or a known variable of a disqualifying type keeps its transit events as candidates: they carry the informational `ON_VARIABLE` flag in `flags` (no effect on the tier).

Transit candidates also carry the informational **R90** screen (`relphot.transit_r90`, `[search] r90_*` settings; `r90_enabled = false` skips it): one `flags` token per failed criterion group -- `R90_SINGLE_POINT` (one epoch carries more than 40 % of the event's chi2 improvement, or three more than 60 %), `R90_SYSTEMATICS` (a refit with the star's own per-frame FWHM, local background and centroid x/y keeps less than 30 % of the chi2 improvement or 70 % of the depth), `R90_CLIP` (chi2 improvement below 16 after a 3-sigma residual clip), `R90_FLAT` (the trapezoid beats the flat model by less than 6 in BIC), `R90_FIT_FAIL` (the trapezoid refit did not converge). They never reject, change the tier or the candidate set; a candidate with none of them passes (`transit_r90_pass`). On 1622 reviewed events of 12 nights the full set removed 67 % of the rejected events at 91 % recall of injected transits. The raw numbers (`top1_share`, `top3_share`, `reg_dchi2_ratio`, `reg_depth_ratio`, `clip3_dchi2`, `dbic_flat`) are columns of the search-metrics table, of `candidates.csv` and, as `transit_*`, of `detection.extra`, so thresholds can be re-tuned without re-running. `detection.flags` is text, so no migration is needed; nights searched before this feature simply have no R90 tokens (re-run `relphot search` and reload to add them).

Transit candidates also carry the informational **`NEIGHBOUR_SHARED_EVENT`** flag (bit 15, `relphot.transit_neighbour`, `[search] neighbour_event_*` settings; `neighbour_event_enabled = false` skips it): one star's eclipse leaking into the aperture of a close neighbour makes both look like candidates, which the coincident-transit veto (`SHARED_EPOCH`, many stars at one epoch) cannot see for a pair of two. Every candidate has each star within `neighbour_event_radius_arcsec` (12", the largest 8" aperture radius plus about 2 FWHM) measured at the candidate's own (tc, duration) window with the same joint nuisance + box fit as the cross-aperture depth check, so a neighbour dip too shallow to be a candidate itself is still found and a dip displaced by about one duration is not; the flag is set when the best neighbour dims there at `neighbour_event_dip_sigma` (3) or more. It never rejects, changes the tier or the candidate set. Both members get it when both are candidates. The raw partner numbers -- `partner` (`star_id` of this night's search), `sep_arcsec`, `depth`, `dip_sigma`, `dtc_hours` (partner's own best-event tc minus this one's), `deficit_ratio`, `is_source`, `gaia_id` -- are the `transit_shared_*` columns of the search-metrics table, the `shared_*` columns of `candidates.csv` and `transit_shared_*` in `detection.extra`. `is_source` marks the probable source of the pair: the star with the larger absolute flux deficit (depth x median flux, in its own best aperture), decided once per pair at the window of the higher-SNR member; the other is the star the light leaked into. Re-run `relphot search` and reload a night to add it.

A per-night `'transit'`/`'variable'` detection always counts. A multi-night detection (`'bls'`/`'ls_periodic'`/`'internight'`/`'recurrent'`, always `mn_run_id`-scoped) counts only if its kind is listed in `db.class_multinight_kinds` -- **by default only `'recurrent'`**. `'bls'`, `'ls_periodic'`, and `'internight'` are still stored and queryable (see "Web usage" below), but excluded from CLASS/PERIOD by default until their thresholds are calibrated against real data: on the T80S 20251104-06 3-night run, letting them count made 246 objects `EXOP` and 263 `VAR` solely off uncalibrated multi-night thresholds, which are currently dominated by night-step artefacts rather than real signal.

To change this, add to your `[db]` TOML settings section, then re-run `relphot db analyze --all` so every affected object is recomputed:

```toml
[db]
class_multinight_kinds = ["recurrent", "bls"]   # example: also trust multi-night BLS
```

`n_detections`, `best_snr`, `depth`, `duration_h`, and `amplitude` are unaffected by this gate -- they always summarise every detection, per-night or multi-night.

PERIOD priority (auto rows only, independent manual guard on `period_source`): manual > literature `known_period` (`'catalog'`, `period_err` = the catalogue's error, NULL if it gives none) > the object's best `period_estimate` (most nights, then latest) with `fap <= db.ls_fap_threshold` (`'LS'`, with `period_err` and `period_n_nights`; the FAP gate tests significance, not precision -- periods are provisional and an imprecise one is still stored) > combined BLS peak for planet hosts with `depth_snr >= db.bls_min_snr` (`'BLS'`) > combined LS peak for variables with `fap <= db.ls_fap_threshold` (`'LS'`) > median per-night LS period (`'night-LS'`) > `NULL`. Because the BLS branch requires `is_exop`, gating `'bls'` out of `class_multinight_kinds` also removes it from the `'BLS'` PERIOD source until it is added back.

## Transit events, matching and period verification

`relphot db analyze` also computes, for the objects it processes:

- **Transit shapes** (`relphot.transit_shape`): each per-night `'transit'` detection is fitted with a trapezoid (tc, depth, T14, ingress fraction T12/T14) plus one constant baseline, on tied flux when a multi-night tie covers the night, else per-night flux. Errors come from `J^T J` scaled by `max(1, chi2_red)`; a failed fit is stored with `converged = false` and NULL errors. No trend is fitted to the star's own light curve.
- **Incomplete transits (duration lower limits):** a transit whose search flags include `EDGE` or `PARTIAL`, whose predicted ingress/egress falls outside the night's observed span, or that has a gap longer than twice the median cadence covering the predicted ingress or egress (a gap in the middle of the transit does not shorten it), has only a *minimum* duration. It is stored as such (`detection.duration_lower_limit`, set from the flags at load and reset by every analyze run to the search flags OR the fresh shape verdict, so an earlier over-eager true is cleared; `transit_shape.t14_lower_limit` with the `incomplete_reason`; `object.duration_lower_limit` for the displayed `duration_h`), its T14 error is NULL, and its ingress fraction is NULL unless both ingress and egress were observed. The web and CSV show it as "≥ x.xx h" and never as a measured value (CSV columns `duration_lower_limit` and `duration_display`).
- **Matching transits** (`relphot.transit_match`): for every pair of converged fits of one object, z-scores of the depth, T14 and ingress-fraction differences (with a systematic floor: `db.match_depth_sys_frac` of the mean depth, plus `db.match_depth_sys_frac_cross_telescope` when the telescopes differ; `db.match_t14_sys_frac`/`db.match_ingress_sys` default 0), `chi2`, `dof` and `p_match`, with the commensurate periods `|dt|/k` (at most 50, none below `db.match_period_min_days`). A lower-limit T14 enters the comparison one-sidedly (`z = max(0, L - T)/sigma_T`: no penalty when the measured duration is at least the limit; two lower limits have no duration term), and the ingress term is dropped when either event has none. This is informational: **events are never merged automatically**, a lower `p_match` does not reject anything (a multi-planet host is expected to show different depths), and no status is changed.
- **Coincident events** (`relphot.transit_coincidence`, schema v9): one planet cannot transit two stars at once, so a per-night transit event with many look-alikes on the same night (other objects' converged events with the same centre time within `max(coincidence_tc_frac x T14, coincidence_tc_nsigma x tc error, one cadence)` and a similar T14 and depth, `coincidence_t14_ratio` / `coincidence_depth_ratio`) is a systematic. It is kept as a detection but marked `detection.auto_status = 'REJECTED'` with `auto_reason` when it has at least `coincidence_min_similar` look-alikes and the chance probability (binomial, centre times uniform over the night) is below `coincidence_max_p`. `n_similar`, `n_expected`, `p_chance` and the look-alike detections are stored per event. The person's `detection.status` is never set automatically: an auto-rejected event with status UNCONFIRMED counts like a REJECTED one (no automatic EXOP evidence, not awaiting review, not the best transit), a CONFIRMED one is evidence again. The whole night is re-judged whenever `analyze` processes any object with a transit shape on it; the web shows "REJECTED (auto)" with the reason and links to the look-alikes.
- **Period verification** (`relphot.period_estimate`): for every variable and every object with a literature variable period, a combined Lomb-Scargle period refined by a 2-harmonic Fourier fit with one offset per night, with its error (for a period longer than a night see "Long periods" below). With a literature period the peak is taken in the windows `P_lit x h x (1 +/- db.lit_period_window_frac)` for `h` in 0.5, 1, 2, so an eclipsing binary seen at half its catalogued period reports `harmonic = 0.5`; `delta = P_obs / harmonic - P_lit` with its error and z. One row per set of nights: re-running on the same nights updates the row, a new night inserts a new one, so the history of re-observations is kept (a single night is stored too). An object with a literature period gets a row for every re-observation even when it cannot be verified; `verify_status` (schema v5) says which case -- `verified`, `no_literature` (a variable without a literature period), `lit_period_outside_grid` (e.g. `verify_note` "P_lit 217 d > LS max period 1.96 d": the nights are too short for that period), `no_peak_in_window` (no LS peak within the window), or `insufficient_data` -- and `verify_note` why; without a verified peak `harmonic`/`delta` are NULL and the period is the data's own LS peak. The web's Period verification table, the search results (`period_verify_status`, `period_verify_note`) and the CSV show them.

**Long periods (P > `db.long_period_days`, default 1 d, on >= 2 nights).** Per-night-normalised flux removes night-to-night changes, and a free offset per night absorbs them, so neither can measure a period longer than a night. For such a period the data are tie-calibrated magnitudes (a multi-night tie covering >= 2 of the object's nights defines the series -- only those nights -- with the tie's own zero-point errors in the error budget) and the Fourier refinement has a single offset. When no tie covers >= 2 nights (e.g. nights from different telescopes) the estimate is stored with `verify_status = 'long_period_needs_tie'` ("long period needs a multi-night tie"), `period_err` NULL, no verification, and it never becomes the object's PERIOD. The period error comes from the Fourier fit covariance (scaled by `max(1, sqrt(chi2_red))`); `verify_note` says when the baseline covers fewer than two cycles ("period unconstrained beyond the lower bound"); a literature-window peak that sits on the edge of the LS grid is not a resolved peak and is never `verified`. Every row also stores `phase_coverage` (the fraction of 20 phase bins holding a point) and `n_cycles` (baseline / period), and, for two or more nights, `alias_periods` / `alias_powers`: the one-day aliases `|f +/- 1 d^-1|`, the two-day aliases and the next-highest LS peaks (at most 5, decreasing power). The web shows coverage and cycles (red below 50 % coverage) and lists the aliases as buttons for the phase diagram.

After pulling a version with these features, re-run `relphot search` on the loaded nights (so their transit events get the `ON_VARIABLE` flag), reload them, and run `relphot db analyze --all`.

## User-guided reprocessing

On an object's detail view, RERUN queues a re-run with your own guess (`relphot.reprocess_request`, filled by `POST /api/object/{id}/reprocess` through the `relphot_web` role, which may only INSERT the guess columns of a request -- never set its status or result). An insert sends `NOTIFY relphot_reprocess`; the worker (`relphot db reprocess --watch`, unit `relphotdb-worker.service`; or `relphot db reprocess --once` by hand) works the queue off first-in first-out, one transaction per request, and marks a request that raises `failed` with the error text. It assumes it is the only worker: on start it queues again any request a crashed worker left `running`. `--poll-seconds` (default 60) is the fallback when a NOTIFY is missed; SIGTERM stops it cleanly.

- **variable** (period guess in days > 0): Lomb-Scargle in the windows `guess x h x (1 +/- db.guided_period_window_frac)` (default 0.2) for h in 0.5, 1, 2, refined by the Fourier fit (long periods as above), stored as a `period_estimate` with `method = 'LS-guided'` and the guess in `guess`, verified against the literature period as usual. A later guided request on the same nights replaces the earlier guided row (the request history keeps every result). It never changes PERIOD: the result offers **Adopt as period** (`POST /api/object/{id}/adopt_period`), which sets a manual PERIOD with the estimate's error (`period_source = 'manual'`; "reset period to auto" hands it back). Estimates flagged `long_period_needs_tie` cannot be adopted. 'All nights rerun' unticked restricts the search to the chosen night (no multi-night tie; only periods shorter than the night's span can be found).
- **transit** (centre in BJD_TDB inside an observed night of the object -- or click the light curve to fill it -- and a width of 0.1-12 hours): the trapezoid fit of "Transit events" (window `tc +/- max(1.5 w, w + 1 h)`, incomplete events are lower limits) started there. The result is a new `transit_shape` on a NEW detection with `origin = 'user'`; a search detection of that night within half the width is linked in the new detection's `extra` and never overwritten. User detections appear in the Transit events and Matching transits tables (events are never merged), are **not** deleted when the night is reloaded (a reload replaces the `origin = 'search'` detections only, and never drops an object that holds a user detection or a request), and never set `is_exop` or the best-transit summary; set the flag by hand if you want it.

Both kinds then rebuild the object's transit matches and refresh its summary fields. The request history (status, result) is polled by the page, which also shows the queue depth.

Filling the form from the light curve: while RERUN is ticked, a plain click fills the transit centre of the entry you last edited, and a horizontal **drag** selects an x span instead of zooming (unticking RERUN, or the mode bar's zoom button, brings zoom-by-drag back). The drag fills that entry from its kind -- EXOP (also when both are ticked): the transit centre (the midpoint of the span) and the width of the suspected eclipse in hours; VAR only: the period guess in days (drag from one peak to the next) -- and switches the entry to the night with data in the span. Tick VAR (and not EXOP) in the entry before dragging for a period.

Pending RERUNs are visible without opening the object: the results table marks each object that has a queued/running request (`RERUN n`; else the outcome of its newest request, `rerun done`/`rerun failed`), the columns `n_rerun_pending`, `last_rerun_status` and `last_rerun_finished_at` can be sorted on and exported, and the query window's **RERUN pending** filter (`rerun_pending=true|false` on the API) finds them. The top bar shows "N RERUNs pending" while any request of any object is queued or running (click it for the list of objects, click one to load it); the page polls `GET /api/reprocess` every 10 s while that is above 0, and shows a dismissible notice "RERUN for obj N finished: done/failed" with an Open button when a request it had seen pending ends. The object detail header shows the obj_id and a badge (queued/running, done -- see results below, failed with the error) that links to the request history. `GET /api/reprocess?status=queued,running&watch=<req_id>&limit=200` (read-only, `relphot_ro`) returns those requests of all objects, the `watch`-ed ones whatever their status, and the queue depth; nothing but the RERUN submit ever queues a request.

## Web usage

Open <http://127.0.0.1:8080> on the host running the containers. From elsewhere, tunnel it first:

```
ssh -L 8080:127.0.0.1:8080 <host>
```

then open <http://127.0.0.1:8080> locally as usual.

The host port defaults to 8080 and is always bound to 127.0.0.1 only (the container listens on 8050 inside). To use another one, put `RELPHOT_WEB_PORT=<port>` in `~/.config/relphot/relphotdb.env` (or export it for one run) and re-run `bash deploy/install.sh web`, which writes the port into the installed quadlet unit and waits on it.

The query window filters on CLASS, KNOWN, SOURCE_DB, status, cone search (RA/Dec/radius), magnitude, period, SNR, depth, tier, n_nights, night date range, telescope, and name/Gaia ID substring, plus a **detection kind** filter (`transit`/`variable`/`internight`/`ls_periodic`/`bls`/`recurrent`, multi-select) with a **scope** of `any`/`night`/`multinight` -- this is how to find the multi-night `bls`/`ls_periodic`/`internight` candidates that (by default) no longer set CLASS, e.g. to review them by hand. The same filters are exposed on the API directly:

```
curl 'http://127.0.0.1:8080/api/search?detection_kind=bls&detection_scope=multinight&limit=25'
```

The query window also filters on the two independent flags (exoplanet host / variable, each yes/no/any), the combined class `EXOP+VAR`, and a minimum transit-match probability (objects with any pair of transit events matching with `p >= x`). Two review filters (`needs_review`, `user_reviewed`, each yes/no/any) find the objects with a night or multi-night event still awaiting your verdict (`n_review_pending > 0`) and the objects on which you have set a verdict, note or event status on at least one night (`n_nights_reviewed > 0`). Result columns add `is_exop`, `is_var`, `period_err`, `n_transit_events`, `max_p_match` and, for literature variables, the latest `period_delta` +/- `period_delta_err`; the CSV export includes them.

**Apparent magnitude.** `mean_mag` (and `star_night.mag`) are relphot's *instrumental* magnitudes, `-2.5 log10(F / R)` with `F` the flux in counts per exposure (`R` is the night-normalised reference, about 1) -- robo43's convention `MAG = zp - 2.5 log10(flux)` -- so the apparent magnitude is `m + zp`. `night.zp` is the median Gaia zero point (`ZPABS` of the frame headers, only frames with `ZPABSCAL` true) of the night's kept frames when at least half of them carry one (`zp_source = 'gaia'`), else the telescope's measured zero point from `[db] telescope_zp` (`'measured'`; default `T80S = 27.85`, the median of Gaia DR3 G minus relphot magnitude over 200 bright isolated T80S stars, MAD-sigma 0.12, at the best aperture and without aperture correction, relphot magnitudes being per 90 s exposure), else `[db] assumed_zp`, default 20 mag, robo43's `instrumental_zp` (`'assumed'`). Both are set in the `[db]` TOML section (`assumed_zp = 20.0`, `[db.telescope_zp]` with keys as stored in `night.telescope`, e.g. `T80S`, `ROBO43`; a `telescope_zp` table replaces the default one as a whole); a night is re-loaded to pick up a change. The results table and CSV gain `mean_mag_app` (the mean over nights of `mag + zp`; `mean_mag` is unchanged) and `mag_zp_source` (the source all the object's nights share -- `gaia`, `measured` or `assumed` -- else `mixed`); the detail view shows "16.32 (Gaia ZP)", "≈ 17.19 (ZP 27.85, measured)" or "≈ 17.10 (ZP=20 assumed)" beside the instrumental value. It is an approximation: the aperture of the star's best photometry is not corrected to the aperture the Gaia zero point was fitted on (`ZPAPER`, 4 arcsec by default), and the assumed 20 mag ignores the exposure time (per-exposure counts, 30 s for ROBO43 against 90 s for T80S; the measured T80S value is per 90 s exposure and T80S rSDSS instrumental magnitudes
  against Gaia DR3 G (Gaia G scale, kept for all telescopes), spread 0.12 mag).

The light curves are drawn in apparent magnitude by default (brighter up): a night's mean apparent magnitude plus `-2.5 log10(flux / median)`, hovering a point shows its relative flux and delta magnitude; tied "All nights" curves add the anchor night's zero point to the tie-calibrated magnitudes, untied ones put every night at its own mean apparent magnitude (the nights are still not tied), and the phase diagram likewise (tied: anchor zero point; untied: each night at its own mean magnitude). The **Y axis** selector switches back to relative flux.

Results are paged and sortable (click a column header), with a "Load" button per row and a CSV export. The detail view shows metadata, catalogue matches, detections, a **Transit events** table (night, tc, depth, T14 and ingress fraction with errors, tier, flags, and a status selector per event), a **Matching transits** table (pair, dt, z-scores, `p(match)`, commensurate periods; events are never merged automatically), a **Period verification** table with a plot of delta against the number of nights, one button per observed night (loads that night's light curve with the trapezoid fit drawn over it; hovering a point shows its image file name), an "All nights" button (tie-calibrated magnitudes when a multi-night run covers the object's nights, otherwise per-night-normalised flux, labelled `night-normalised`/untied), a **phase diagram** directly below the light curve (for variables and any object with a PERIOD or a guided period: all nights, tie-calibrated magnitudes when a tie covers them, otherwise per-night flux labelled *untied*; points coloured by night with error bars, a period selector -- PERIOD, latest estimate, latest guided, literature, a typed value, x2 and /2, the stored alias candidates -- the epoch (first epoch or the fitted Fourier phase zero, the deepest minimum), the 2-harmonic model curve, an optional -0.25 to 1.25 phase range, and the phase coverage with a warning below 50 %), the RERUN section and request history, a periodogram panel (scope x method), a **Night reviews** table (one row per night: the automatic EXOP/VAR evidence, your EXOP and VAR verdicts -- blank = automatic, CONFIRMED or REJECTED -- a note, the effective result; nights with an unconfirmed automatic event are highlighted), and manual editing of the two flags (Exoplanet host, Variable checkboxes: a shorthand that writes verdicts on the nights loaded now), status (informational), period and notes.

**Advanced SQL** (`relphot_ro` role, read-only, 30 s `statement_timeout`) lets you run arbitrary `SELECT`s against: `night`, `frame`, `object`, `star_night`, `lightcurve`, `detection`, `catalog_match`, `mn_run`, `tie`, `periodogram`, `transit_shape`, `transit_match`, `transit_coincidence`, `period_estimate`, `reprocess_request`, `night_tile`, `tile_lc`, `reference_member`, `comparison_member`, `schema_version` (all under the `relphot` schema).

## Manual edits: per-night reviews and reset-to-auto

Per-night user verdicts (`relphot.user_night_review`, schema v6) are CONFIRMED / REJECTED / auto per object per night:

- **Per-night endpoint:** `PUT /api/object/{obj_id}/night/{night_id}/review` (role `relphot_web`, INSERT/UPDATE/DELETE on `user_night_review`) accepts `{exop: 'CONFIRMED'|'REJECTED'|null, var: '...'|...|null, note: '...' or null}`. All-null (and note null/empty) deletes the row; otherwise upserts it. Verdicts override the automatic evidence of that night only; per-night new nights are evaluated automatically. Literature (known planets/variables) always counts and cannot be removed by a verdict. Returns the updated night's state (auto evidence, effective flags, pending status) and the object's derived flags.

- **Object-level shorthand (old API):** `PATCH /api/object/{obj_id}` with `is_exop`/`is_var` (role `relphot_web`, UPDATE on those columns) is a shorthand that upserts CONFIRMED/REJECTED verdicts on all nights the object *currently* has data for; `exop_source`/`var_source = 'auto'` resets those verdicts to NULL on all nights and deletes empty rows. The two flags are independent: setting one never changes the other. The older `class` field is still accepted and only sets the flags it names (`'UNC'` clearing both). The web's "reset to auto" buttons call this endpoint. `relphot db analyze` then re-evaluates the verdicts via `refresh_flags` (in the same transaction before the main `refresh_objects` call), and `class`/`class_source` are re-derived.

- **Detection verdict:** `PATCH /api/detection/{det_id}` (UPDATE only on `detection.status`/`notes`) records a person's verdict (`UNCONFIRMED`/`CONFIRMED`/`REJECTED`) on one transit event; this changes the automatic evidence of that night and triggers a `refresh_flags` call for the object in the same transaction.

- **Period and notes:** `PATCH /api/object/{obj_id}` with `period` stamps `period_source = 'manual'` and `relphot db analyze` never touches it again; `period_source: "auto"` hands it back to the pipeline. `notes` is always editable. `status` (object-level informational field) is also editable but never affects CLASS.

## Backup and restore

`relphotdb-backup.service`/`.timer` are **plain** systemd user units (not podman quadlet, since they run a host-side script, not a container) installed to `~/.config/systemd/user/` by `deploy/install.sh backup`. The timer fires daily at 03:30 (`Persistent=true`, so a missed run -- e.g. the host was off -- fires as soon as it's back). The service runs `deploy/backup.sh`, which does:

```
podman exec relphotdb-db pg_dump -U postgres -Fc relphot > <backups>/relphot_<YYYYmmdd_HHMM>.dump.tmp
mv ... relphot_<YYYYmmdd_HHMM>.dump.tmp relphot_<YYYYmmdd_HHMM>.dump   # atomic rename
```

then deletes all but the newest 14 dumps in `/ssdsto1/data/relphotDB/backups/`. Output is captured by the systemd journal (`journalctl --user -u relphotdb-backup.service`).

Check it's running:

```
systemctl --user list-timers | grep relphotdb-backup
ls -la /ssdsto1/data/relphotDB/backups/
```

To restore (or just inspect) a dump, note that `pg_restore` runs *inside* the `relphotdb-db` container, but the backups directory is not bind-mounted there -- copy the dump in first:

```
podman cp /ssdsto1/data/relphotDB/backups/relphot_<stamp>.dump relphotdb-db:/tmp/restore.dump
podman exec relphotdb-db pg_restore --list /tmp/restore.dump          # inspect
podman exec relphotdb-db pg_restore -U postgres -d relphot --clean --if-exists /tmp/restore.dump
podman exec relphotdb-db rm -f /tmp/restore.dump
```

(`--clean --if-exists` drops and recreates the `relphot` schema's objects before restoring; omit it to restore into an empty/different database instead.)

## Upgrading

1. Pull/merge the new relphot source.
2. `relphot db init` -- applies any new schema migration (schema v4/v5 add the classification tables and columns the web UI queries, v6 the reprocess queue, error-inflation columns and guided estimates, so run it **before** restarting the web container); idempotent, safe to run even when there is nothing new.
3. `bash deploy/install.sh web` -- rebuilds the `relphotdb-web` image from the repo as it now stands (the web container has no other way to pick up a code change) and restarts it. The `relphotdb-db` image only needs rebuilding if `deploy/db/Containerfile` itself changed (e.g. a PostgreSQL/q3c version bump).
4. If the change affects derived fields (a new `[db]` threshold, a CLASS/PERIOD rule change, ...), `relphot db analyze --all` to recompute every affected object.
5. `bash deploy/install.sh worker` (once; a restart picks up new worker code: `systemctl --user restart relphotdb-worker`).

After schema v6 the live data needs, in this order: `relphot lightcurves` per night (inflated `lc_err`, `lc_err_raw`, `err_scale`, `blended`); `relphot search` per night (it reads the inflated errors, so bright stars' SNRs fall by their factor) and `relphot multinight` / `multisearch` (they read the light-curve product); `relphot db load-night` per night (a reload keeps user detections) and `relphot db load-multinight`; `relphot db analyze --all` (long-period series, aliases, coverage; timings scale as before).

## Known limitations

- **Catalogue source is merged for ~7% of variable matches.** `relphot.catalogs.match_known_variables` merges VSX, Gaia DR3 variability, and ASAS-SN hits into one `known_name` (the first non-empty one, in query order) with no catalogue column kept per star, so `object.source_db` can list more than one catalogue for a single match without saying which one actually supplied the name/type/period.
- **BLS/multi-night detection thresholds are uncalibrated.** See "CLASS and PERIOD rules" above -- `'bls'`, `'ls_periodic'`, and `'internight'` are stored and queryable but excluded from CLASS/PERIOD by default until validated against confirmed detections.
- **Object positions are first-seen.** `object.ra`/`dec` are set when the object row is first created and never updated by a later night's astrometry, even if that night's solution is more precise.
- **Global multi-night ids are not stable across runs.** `mn_run`/`tie` map a multi-night run's own per-night `star_id`s to `object` rows at load time; a different multi-night run over the same nights (different stem, different global-id assignment upstream) is not guaranteed to reuse the same mapping.

## No sudo required

Every command above runs as the invoking user: rootless podman, user-scoped systemd units (`systemctl --user`, `~/.config/systemd/user/`, `~/.config/containers/systemd/`), and a user-writable data directory. Nothing in this guide needs `sudo`.

## Data storage: SSD staging and permanent NFS storage

Since 2026-10-02 telescope data lives permanently on the NFS share `/mnt/sto01/<TEL>/raw/` and `/mnt/sto01/<TEL>/reduced/<NIGHT>/`
(slow random IO). `/ssdsto1/data/<TEL>_reduced/<NIGHT>/` is only a staging area for the IO-heavy steps (robo43 reduction and
`forced`, `relphot ingest/reference/lightcurves/search`). The results DB (`/ssdsto1/data/relphotDB`) stays on the SSD.

Rule: the DB holds the PERMANENT path. `relphot db load-night` and `load-multinight` key a night on `night.source_dir`, so
1. process the night in staging;
2. `deploy/relocate_reduced.sh --sync /ssdsto1/data/<TEL>_reduced/<NIGHT> /mnt/sto01/<TEL>/reduced/<NIGHT>`;
3. `relphot db load-night /mnt/sto01/<TEL>/reduced/<NIGHT>/relphot` (the telescope is inferred from `<TEL>/reduced`);
4. run `relphot multinight` on the permanent night directories (its npz records them) and load it from there;
5. remove the staging copy.
A night already in the DB under its staging path is moved with `--db` (rewrites `night.source_dir`, `frame.file_path`,
`mn_run.stem` after a backup). The script never deletes the source.
