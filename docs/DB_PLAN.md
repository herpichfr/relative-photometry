# relphot results database — design (DB_PLAN)

Status: agreed with the user 2026-09-28. This file is the spec for `relphot db` (loader) and
`relphot.web` (query server). Build order at the end.

## Decisions

- PostgreSQL 17 + q3c, rootless podman, data in `/ssdsto1/data/relphotDB/pgdata`, published on
  `127.0.0.1:5433` only (the host already runs PostgreSQL on 5432 and MariaDB on 3306).
- Web server (FastAPI + Jinja2 + Plotly.js, vendored) in its own container on `127.0.0.1:8050`.
- Stars that fail the noise cut are NOT stored at all.
- CLASS precedence: manual > known planet (EXOP) > known variable (VAR) > transit detection (EXOP)
  > variability detection (VAR) > UNC. A per-night 'transit'/'variable' detection always counts;
  a multi-night detection ('bls'/'ls_periodic'/'internight'/'recurrent') counts only if its kind is
  in `db.class_multinight_kinds` (default: only 'recurrent' -- 'bls'/'ls_periodic'/'internight'
  thresholds are uncalibrated and dominated by night-step artefacts). Since the BLS PERIOD branch
  requires computed class EXOP, gating 'bls' out of CLASS also removes it from the PERIOD 'BLS'
  source until `class_multinight_kinds` includes it.
- Plots are drawn in the browser from database data; no PNGs are stored.

## Noise cut (one definition, configurable in `[db]` settings)

A star of a night is stored iff, at its best aperture (values from `*_starstats.parquet`):

1. `n_epochs >= SearchSettings.effective_min_epochs(n_kept_frames)` (the same cut the search uses), and
2. `expected_noise <= db.max_expected_noise` (default 0.05, i.e. 50 mmag predicted per-point noise;
   a photon/background prediction, so it does not reject real variables), and
3. finite `rms`.

Exception: any star that is a candidate of that night's search (`transit_candidate` or
`variability_candidate` in `*_search_metrics.parquet`) is always stored (`db.keep_candidates`,
default true). On T80S 20251104-06 the cut keeps ~97 % of stars.

## Schema (schema `relphot`)

- `night(night_id serial pk, telescope text, night_date date, label text, site_lat, site_lon, site_elev,
  object text, filter text, n_frames int, n_kept int, source_dir text unique, relphot_commit text,
  settings jsonb, noise_cut jsonb, loaded_at timestamptz)`; unique (telescope, label).
- `frame(frame_id bigserial pk, night_id fk, frame_index int, file_name text, file_path text,
  date_obs timestamptz, jd_utc float8, bjd_tdb float8, exptime real, airmass real, fwhm real,
  n_sources int, kept bool)`; unique (night_id, frame_index); index on file_name.
- `object(obj_id bigserial pk, name text unique /* 'RP Jhhmmss.ss+ddmmss.s' */, ra float8, dec float8,
  gaia_id text, mean_mag real, n_nights int, class text check in ('UNC','EXOP','VAR'),
  class_source text check in ('auto','manual'), period float8 /* days */, period_source text,
  known bool, source_db text, known_name text, known_type text, known_period float8,
  status text check in ('UNCONFIRMED','CONFIRMED','REJECTED') default 'UNCONFIRMED',
  best_snr real, depth real, duration_h real, amplitude real, n_detections int,
  first_night date, last_night date, neighbour_sep_arcsec real, notes text, updated_at timestamptz)`;
  q3c index `q3c_ang2ipix(ra, dec)`; indexes on class, known, period, mean_mag.
- `star_night(obj_id fk, night_id fk, star_id int, tile int, mag real, best_aperture smallint, rms real,
  expected_noise real, chi2_reduced real, n_epochs int, is_comparison bool, pk (obj_id, night_id))`;
  unique (night_id, star_id).
- `lightcurve(obj_id, night_id, pk (obj_id, night_id) fk star_night, frame_index smallint[],
  bjd_tdb float8[], flux real[], flux_err real[], flux_raw real[])`  — kept epochs only, best aperture.
- `detection(det_id bigserial pk, obj_id fk, night_id fk null, mn_run_id fk null,
  kind text check in ('transit','variable','internight','ls_periodic','bls','recurrent'),
  snr real, depth real, tc_bjd_tdb float8, duration_h real, tier smallint, flags text,
  amplitude real, excess real, period float8, fap real, extra jsonb)`; exactly one of night_id / mn_run_id set.
- `catalog_match(obj_id fk, catalog text, name text, type text, period float8, sep_arcsec real,
  reference text, pk (obj_id, catalog, name))`.
- `mn_run(mn_run_id serial pk, stem text unique, labels text[], anchor text, settings jsonb, loaded_at)`;
  `tie(mn_run_id, obj_id, night_id, mag real, mag_err real, pk (mn_run_id, obj_id, night_id))` — the
  night's mean calibrated magnitude from the multi-night tie (`mlc_night_mean_mag`).
- `periodogram(obj_id fk, scope text /* 'night:<night_id>' or 'combined' */, method text check in ('LS','BLS'),
  fmin float8, df float8, n int, power real[], peak_period float8, peak_power real, fap real,
  computed_at, pk (obj_id, scope, method))` — frequency grid f_k = fmin + k*df, cycles/day.
- `schema_version(version int)`.
- Roles: `relphot_owner` (loader), `relphot_ro` (web queries, SELECT only, statement_timeout 30 s);
  the web's manual edits (class/status/notes) use a third role `relphot_web` with UPDATE on those
  object columns only.

## Loader (`relphot db ...`, runs on the host, DSN from `RELPHOT_DB_DSN` or `~/.config/relphot/db.env`)

- `relphot db init` — create/upgrade the schema (idempotent, versioned SQL files).
- `relphot db load-night NIGHT_RELPHOT_DIR [--telescope T] [--label YYYYMMDD]` — one transaction; a reload
  deletes that night's star_night/lightcurve/detection rows first. Frames from `night.npz`
  `frame_meta_json` + `ref.npz` `frame_kept`. Objects matched by q3c within `db.match_radius_arcsec`
  (default 1.0"), nearest match, else new object. Light curves from `*_lightcurves.parquet`,
  detections and catalogue matches from `*_search_metrics.parquet`.
- `relphot db load-multinight STEM` — `mn_run`, `tie` (via `_stars.parquet` `star_id_{label}` ->
  star_night -> obj_id), multi-night detections from `multinight_search_metrics.parquet`.
- `relphot db analyze [--all]` — for objects whose data changed (or all): recompute periodograms
  (LS per night and combined; BLS combined for EXOP when the baseline allows), then CLASS
  (auto rows only), PERIOD, n_nights, first/last night, best_* summaries.
- PERIOD priority: manual > literature period > combined BLS (EXOP, depth_snr >= bls_min_snr) / combined LS
  with FAP <= 0.01 (VAR) > best per-night LS period > NULL; `period_source` records which.

## Web (requirements 10-15)

- Query window: CLASS, KNOWN, SOURCE_DB, status, cone (RA, Dec, radius"), mag, period, SNR, depth, tier,
  n_nights, night date range, telescope, name / Gaia id; advanced read-only SQL box (relphot_ro, timeout).
- Paged sortable results with a Load button per row; CSV export.
- Detail: metadata, catalogue matches, detections; one button per observed night (loads that night's LC;
  hover shows image file name); "All nights" button (tie-calibrated magnitudes when a multi-night run
  covers the nights, otherwise per-night-normalised flux, labelled as untied); phase diagram at PERIOD
  with editable period and x2 / /2; periodogram panel; transit box overlay from tc/depth/duration;
  manual CLASS / status / notes.

## Build order

1. podman + quadlet units, db image with q3c, schema + roles.
2. Loader + noise cut, tests on synthetic fixtures.
3. `analyze` + periodogram storage.
4. Multi-night loader.
5. Web API.
6. Web front end.
7. Backfill T80S 20251104/05/06 and ROBO43 20250911; verify counts; pg_dump backup timer; README.
