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
| `relphotdb-web` | podman container (rootless) | `localhost/relphotdb-web:latest` (FastAPI + Jinja2 + Plotly.js, vendored) | `127.0.0.1:8050` | stateless; talks to `relphotdb-db` over the `relphotdb.network` podman network |
| loader (`relphot db ...`) | host, plain CLI | -- | -- | connects to the DB over `127.0.0.1:5433`, same as any other client |

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
relphot db init                 # create/upgrade the schema (idempotent, versioned)
```

`install.sh db` generates and writes the env file only if it does not already exist, builds the `relphotdb-db` image, installs its quadlet unit, starts it, and waits for `pg_isready`. The image's first-boot init script (`deploy/db/10-relphot-roles.sh`) creates the `relphot_owner`/`relphot_web`/`relphot_ro` roles, the `relphot`/`relphot_test` databases (owned by `relphot_owner`), and the `q3c` extension in each.

`install.sh web` builds `deploy/web/Containerfile` (which `pip install`s `relphot[web]` from the repo as it stands **at build time** -- see "Upgrading" below), installs the quadlet unit, restarts the container, and waits for `/api/search?limit=1` to answer.

`install.sh backup` installs and enables `relphotdb-backup.timer`/`.service` (plain systemd user units, not quadlet -- see "Backup and restore") and runs one backup immediately.

`relphot db init` applies every schema migration in `src/relphot/db/sql/NNN_*.sql` newer than the database's recorded `schema_version`; a fresh database gets all of them, an existing one only the new ones. Run it once after `install.sh db`, and again after pulling in a relphot version that adds a migration.

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

## Noise cut

A star of a night is stored in the results database (`relphot.star_night`/`lightcurve`) iff, at its best aperture (from `*_starstats.parquet`):

1. `n_epochs >= SearchSettings.effective_min_epochs(n_kept_frames)` (the same good-epoch cut `relphot search` uses: `max(min_epochs, ceil(min_epoch_fraction * n_kept), 1)`), and
2. `expected_noise <= db.max_expected_noise` (default `0.05`, i.e. 50 mmag predicted per-point noise -- a photon/background prediction, so it does not reject real variables on the strength of their own scatter), and
3. `rms` is finite.

Exception: a star that is a `transit_candidate` or `variability_candidate` of that night's search (`*_search_metrics.parquet`) is always stored (`db.keep_candidates`, default `true`), even if it fails 1-3. Stars that fail the cut and are not a candidate are dropped entirely -- never written to the database.

On T80S 20251104-06 this keeps ~97% of stars; on ROBO43 20250911 (351 frames, `effective_min_epochs` = 176) it kept 382 of 520.

## CLASS and PERIOD rules

An object has two independent flags, `is_exop` (planet host) and `is_var` (variable); neither excludes the other, and `class` is only their derived label (`EXOP+VAR` if both, else `EXOP`/`VAR`/`UNC`). Each flag is set from the data unless its `exop_source`/`var_source` is `'manual'` (set by hand in the web), in which case `relphot db analyze` never touches it. `class_source` is `'manual'` iff either flag source is. `object.status` and `detection.status` are never set automatically. `is_exop`: known planet match, or a transit detection, or a multi-night `'bls'` detection in `db.class_multinight_kinds`. `is_var`: known variable match, or a variability detection, or a multi-night `'internight'`/`'ls_periodic'`/`'recurrent'` detection in `db.class_multinight_kinds`. A star that is a variability candidate or a known variable of a disqualifying type keeps its transit events as candidates: they carry the informational `ON_VARIABLE` flag in `flags` (no effect on the tier).

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
- **Period verification** (`relphot.period_estimate`): for every variable and every object with a literature variable period, a combined Lomb-Scargle period refined by a 2-harmonic Fourier fit with one offset per night, with its error. With a literature period the peak is taken in the windows `P_lit x h x (1 +/- db.lit_period_window_frac)` for `h` in 0.5, 1, 2, so an eclipsing binary seen at half its catalogued period reports `harmonic = 0.5`; `delta = P_obs / harmonic - P_lit` with its error and z. One row per set of nights: re-running on the same nights updates the row, a new night inserts a new one, so the history of re-observations is kept (a single night is stored too). An object with a literature period gets a row for every re-observation even when it cannot be verified; `verify_status` (schema v5) says which case -- `verified`, `no_literature` (a variable without a literature period), `lit_period_outside_grid` (e.g. `verify_note` "P_lit 217 d > LS max period 1.96 d": the nights are too short for that period), `no_peak_in_window` (no LS peak within the window), or `insufficient_data` -- and `verify_note` why; without a verified peak `harmonic`/`delta` are NULL and the period is the data's own LS peak. The web's Period verification table, the search results (`period_verify_status`, `period_verify_note`) and the CSV show them.

After pulling a version with these features, re-run `relphot search` on the loaded nights (so their transit events get the `ON_VARIABLE` flag), reload them, and run `relphot db analyze --all`.

## Web usage

Open <http://127.0.0.1:8050> on the host running the containers. From elsewhere, tunnel it first:

```
ssh -L 8050:127.0.0.1:8050 <host>
```

then open <http://127.0.0.1:8050> locally as usual.

The query window filters on CLASS, KNOWN, SOURCE_DB, status, cone search (RA/Dec/radius), magnitude, period, SNR, depth, tier, n_nights, night date range, telescope, and name/Gaia ID substring, plus a **detection kind** filter (`transit`/`variable`/`internight`/`ls_periodic`/`bls`/`recurrent`, multi-select) with a **scope** of `any`/`night`/`multinight` -- this is how to find the multi-night `bls`/`ls_periodic`/`internight` candidates that (by default) no longer set CLASS, e.g. to review them by hand. The same filters are exposed on the API directly:

```
curl 'http://127.0.0.1:8050/api/search?detection_kind=bls&detection_scope=multinight&limit=25'
```

The query window also filters on the two independent flags (exoplanet host / variable, each yes/no/any), the combined class `EXOP+VAR`, and a minimum transit-match probability (objects with any pair of transit events matching with `p >= x`). Result columns add `is_exop`, `is_var`, `period_err`, `n_transit_events`, `max_p_match` and, for literature variables, the latest `period_delta` +/- `period_delta_err`; the CSV export includes them.

Results are paged and sortable (click a column header), with a "Load" button per row and a CSV export. The detail view shows metadata, catalogue matches, detections, a **Transit events** table (night, tc, depth, T14 and ingress fraction with errors, tier, flags, and a status selector per event), a **Matching transits** table (pair, dt, z-scores, `p(match)`, commensurate periods; events are never merged automatically), a **Period verification** table with a plot of delta against the number of nights, one button per observed night (loads that night's light curve with the trapezoid fit drawn over it; hovering a point shows its image file name), an "All nights" button (tie-calibrated magnitudes when a multi-night run covers the object's nights, otherwise per-night-normalised flux, labelled `night-normalised`/untied), a phase diagram at PERIOD (editable, with x2/`/2` and a note on where the period came from), a periodogram panel (scope x method), and manual editing of the two flags (Exoplanet host, Variable checkboxes), status, period and notes.

**Advanced SQL** (`relphot_ro` role, read-only, 30 s `statement_timeout`) lets you run arbitrary `SELECT`s against: `night`, `frame`, `object`, `star_night`, `lightcurve`, `detection`, `catalog_match`, `mn_run`, `tie`, `periodogram`, `transit_shape`, `transit_match`, `period_estimate`, `schema_version` (all under the `relphot` schema).

## Manual edits and reset-to-auto

`PATCH /api/object/{obj_id}` (role `relphot_web`, UPDATE only on `class`/`class_source`/`is_exop`/`is_var`/`exop_source`/`var_source`/`status`/`notes`/`period`/`period_source`/`period_err`/`period_n_nights`) sets `is_exop`/`is_var` and/or `period` and stamps `exop_source`/`var_source`/`period_source = 'manual'` in the same call (the two flags are independent: setting one never changes the other; `class`/`class_source` are re-derived from the flags; the older `class` field is still accepted and only sets the flags it names, `UNC` clearing both); `relphot db analyze` never touches a manual flag or PERIOD again. The web's "reset to auto" buttons call the same endpoint with `exop_source`/`var_source`/`period_source: "auto"` instead, which hands the field back to the next `analyze` run. `PATCH /api/detection/{det_id}` (UPDATE only on `detection.status`/`notes`) records a person's verdict (`UNCONFIRMED`/`CONFIRMED`/`REJECTED`) on one transit event; nothing sets it automatically.

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
2. `relphot db init` -- applies any new schema migration (schema v4/v5 add the classification tables and columns the current web UI queries, so run it **before** restarting the web container); idempotent, safe to run even when there is nothing new.
3. `bash deploy/install.sh web` -- rebuilds the `relphotdb-web` image from the repo as it now stands (the web container has no other way to pick up a code change) and restarts it. The `relphotdb-db` image only needs rebuilding if `deploy/db/Containerfile` itself changed (e.g. a PostgreSQL/q3c version bump).
4. If the change affects derived fields (a new `[db]` threshold, a CLASS/PERIOD rule change, ...), `relphot db analyze --all` to recompute every affected object.

## Known limitations

- **Catalogue source is merged for ~7% of variable matches.** `relphot.catalogs.match_known_variables` merges VSX, Gaia DR3 variability, and ASAS-SN hits into one `known_name` (the first non-empty one, in query order) with no catalogue column kept per star, so `object.source_db` can list more than one catalogue for a single match without saying which one actually supplied the name/type/period.
- **BLS/multi-night detection thresholds are uncalibrated.** See "CLASS and PERIOD rules" above -- `'bls'`, `'ls_periodic'`, and `'internight'` are stored and queryable but excluded from CLASS/PERIOD by default until validated against confirmed detections.
- **Object positions are first-seen.** `object.ra`/`dec` are set when the object row is first created and never updated by a later night's astrometry, even if that night's solution is more precise.
- **Global multi-night ids are not stable across runs.** `mn_run`/`tie` map a multi-night run's own per-night `star_id`s to `object` rows at load time; a different multi-night run over the same nights (different stem, different global-id assignment upstream) is not guaranteed to reuse the same mapping.

## No sudo required

Every command above runs as the invoking user: rootless podman, user-scoped systemd units (`systemctl --user`, `~/.config/systemd/user/`, `~/.config/containers/systemd/`), and a user-writable data directory. Nothing in this guide needs `sudo`.
