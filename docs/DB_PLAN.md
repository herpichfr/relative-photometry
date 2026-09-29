# relphot results database — design (DB_PLAN)

Status: agreed with the user 2026-09-28. This file is the spec for `relphot db` (loader) and
`relphot.web` (query server). Build order at the end.

## Decisions

- PostgreSQL 17 + q3c, rootless podman, data in `/ssdsto1/data/relphotDB/pgdata`, published on
  `127.0.0.1:5433` only (the host already runs PostgreSQL on 5432 and MariaDB on 3306).
- Web server (FastAPI + Jinja2 + Plotly.js, vendored) in its own container on `127.0.0.1:8050`.
- Stars that fail the noise cut are NOT stored at all.
- Classification is two independent flags (schema v4), not one exclusive class. Binding rules:
  R1 a planet host is never rejected or downgraded because a new transit differs in depth or
  shape (multi-planet systems); R2 a star can be a variable AND a planet host, and neither the
  pipeline nor a manual edit makes the two exclusive; R3 periods are provisional -- never withheld
  for being imprecise; store uncertainty, provenance and history; R4 transit events are never
  merged automatically -- every pair of transit events of one object gets a "matching transits"
  probability from depth and eclipse shape; R5 known (literature) variables are re-analysed on the
  multi-night data and every re-observation (each new distinct set of nights) yields a
  period-verification record (observed period, literature period, difference, errors), history kept.
- `is_exop` (auto) = known planet match OR per-night transit detection OR a multi-night 'bls'
  detection in `db.class_multinight_kinds`. `is_var` (auto) = known variable match OR per-night
  variability detection OR a multi-night 'internight'/'ls_periodic'/'recurrent' detection in
  `db.class_multinight_kinds` (default: only 'recurrent' -- 'bls'/'ls_periodic'/'internight'
  thresholds are uncalibrated and dominated by night-step artefacts). A flag whose `*_source` is
  'manual' is never touched by the pipeline; there is no precedence between the two flags.
  `class` is only a derived label (EXOP+VAR if both, else EXOP / VAR / UNC), `class_source` is
  'manual' iff either flag source is. Since the BLS PERIOD branch requires `is_exop`, gating 'bls'
  out of CLASS also removes it from the PERIOD 'BLS' source until `class_multinight_kinds` includes it.
- The per-night search no longer drops a transit event because the star is also variable (or a known
  variable of a disqualifying type): the event stays a candidate and carries the informational
  `ON_VARIABLE` flag (it never changes the tier). TOO_DEEP events still route to variables.
- `object.status` and `detection.status` are only ever set by a person. A reload of a night keeps
  them: saved `status`/`notes` of the night's detections are re-attached to the new detection of the
  same object and kind (a transit only if its tc is within half its duration; any other kind by object
  and kind); what cannot be matched is logged and kept in `detection_review_orphan`. Objects a person
  touched (manual flag or period source, non-default status, notes) are never dropped as orphans.
- R6: a transit duration is reported for every event, but for an incomplete transit (flags EDGE or
  PARTIAL, a fit window truncated by the night's first/last epoch, or a gap longer than 2x the median
  cadence that contains the predicted ingress or egress -- a gap in the middle of the transit does not
  shorten it) it is only a minimum: it is stored with a lower-limit flag
  (`detection.duration_lower_limit`, `transit_shape.t14_lower_limit` + `incomplete_reason`,
  `object.duration_lower_limit`), compared one-sidedly, and shown as "≥ x.xx h", never as a measurement.
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
  gaia_id text, mean_mag real, n_nights int, class text check in ('UNC','EXOP','VAR','EXOP+VAR') /* derived */,
  class_source text check in ('auto','manual') /* 'manual' iff exop_source or var_source is */,
  is_exop bool not null default false, is_var bool not null default false,
  exop_source text check in ('auto','manual'), var_source text check in ('auto','manual'),
  period float8 /* days */, period_source text, period_err float8, period_n_nights int,
  duration_lower_limit bool /* duration_h is a minimum: the best transit is incomplete */,
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
  amplitude real, excess real, period float8, fap real, extra jsonb,
  status text check in ('UNCONFIRMED','CONFIRMED','REJECTED') default 'UNCONFIRMED', notes text,
  duration_lower_limit bool not null default false /* EDGE/PARTIAL at load; analyze resets it to (flags OR shape verdict) */)`;
  exactly one of night_id / mn_run_id set.
- `catalog_match(obj_id fk, catalog text, name text, type text, period float8, period_err float8 /* NULL if the
  catalogue gives none */, sep_arcsec real, reference text, pk (obj_id, catalog, name))`.
- `transit_shape(det_id pk fk detection on delete cascade, obj_id, tc float8, tc_err, depth real, depth_err,
  t14_h real, t14_err, t14_lower_limit bool not null default false, incomplete_reason text,
  ingress_frac real /* T12/T14 in [0, 0.5] */, ingress_err, chi2_red real, n_points int,
  input text /* 'tied' | 'night' */, converged bool, computed_at)` -- trapezoid fit to one per-night transit.
  For an incomplete event `t14_h` is the observed in-transit span (a lower limit), `t14_err` is NULL and
  `ingress_frac`/`ingress_err` are NULL unless both ingress and egress were observed.
- `detection_review_orphan(obj_id, night_id, kind, tc_bjd_tdb, status, notes, saved_at)` -- a person's
  detection verdict a night reload could not re-attach to a new detection.
- `transit_match(det_a, det_b, check det_a < det_b, pk (det_a, det_b), both fk detection on delete cascade,
  obj_id, dt_days float8, depth_z real, t14_z real, ingress_z real, chi2 real, dof smallint, p_match real,
  same_telescope bool, commensurate_periods float8[], computed_at)` -- pairwise "matching transits"
  probability; informational only, never merges events or changes a status.
- `period_estimate(est_id bigserial pk, obj_id fk, computed_at, method text /* 'LS' */, input text /* 'tied'|'night' */,
  night_ids int[] /* sorted */, n_nights int, last_night date, baseline_days float8, period float8, period_err float8,
  power real, fap real, lit_period float8, lit_period_err float8, lit_catalog text,
  harmonic real /* P_obs = harmonic * P_lit, one of 0.5, 1, 2 */, delta float8 /* period/harmonic - lit_period */,
  delta_err float8, delta_z real, verify_status text /* schema v5: 'verified', 'no_literature',
  'lit_period_outside_grid', 'no_peak_in_window', 'insufficient_data' */, verify_note text /* e.g. "P_lit 217 d >
  LS max period 1.96 d" */, unique (obj_id, method, night_ids))` -- one row per set of nights (history); an
  object with a literature period gets a row for every re-observation even when it cannot be verified
  (then harmonic/delta are NULL and the period is the data's own global LS peak).
- `mn_run(mn_run_id serial pk, stem text unique, labels text[], anchor text, settings jsonb, loaded_at)`;
  `tie(mn_run_id, obj_id, night_id, mag real, mag_err real, pk (mn_run_id, obj_id, night_id))` — the
  night's mean calibrated magnitude from the multi-night tie (`mlc_night_mean_mag`).
- `periodogram(obj_id fk, scope text /* 'night:<night_id>' or 'combined' */, method text check in ('LS','BLS'),
  fmin float8, df float8, n int, power real[], peak_period float8, peak_power real, fap real,
  computed_at, pk (obj_id, scope, method))` — frequency grid f_k = fmin + k*df, cycles/day.
- `schema_version(version int)`.
- Roles: `relphot_owner` (loader), `relphot_ro` (web queries, SELECT only, statement_timeout 30 s);
  the web's manual edits (class flags, status, notes, period) use a third role `relphot_web` with UPDATE
  on those object columns and on `detection.status` / `detection.notes` only.

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
  (LS per night and combined; BLS combined for planet hosts when the baseline allows), then
  - transit shapes: for every per-night transit detection a trapezoid fit (tc, depth, T14, ingress fraction,
    one constant baseline; tied flux when a tie covers the night, else per-night flux; window
    tc +/- max(1.5 T14, T14 + 1 h); `scipy.optimize.least_squares`; errors from J^T J scaled by
    max(1, chi2_red)) -> `transit_shape`;
  - matching transits: for each pair of converged fits, z = (x_a - x_b) / sqrt(err_a^2 + err_b^2 + sys^2)
    for depth, T14 and ingress fraction (depth sys = `db.match_depth_sys_frac` x mean depth, plus
    `db.match_depth_sys_frac_cross_telescope` when the telescopes differ; T14/ingress sys default 0),
    chi2 = sum z^2, p_match = chi2.sf(chi2, dof) -> `transit_match`, plus the commensurate periods
    |dt|/k >= `db.match_period_min_days` (at most 50). The T14 term is one-sided when a duration is a
    lower limit L: z = max(0, L - T)/sigma_T against a measured T (no penalty if T >= L); with two
    lower limits there is no duration term (dropped from dof); the ingress term is dropped when either
    event has no ingress fraction;
  - period estimate / verification, for every variable or object with a literature variable period: combined
    LS (tied magnitudes when a tie covers the nights, else per-night-normalised flux), global peak;
    with a literature period the LS peak is instead taken in the windows lit_period x h x (1 +/-
    `db.lit_period_window_frac`) for h in (0.5, 1, 2) and verified there (an eclipsing binary found at half
    the catalogued period reports harmonic 0.5); the peak is refined by a 2-harmonic Fourier fit with one
    offset per night (period_err = sigma_f / f^2, scaled by max(1, sqrt(chi2_red))) -> `period_estimate`
    (upserted on (obj_id, method, night_ids): the same nights update, a new set of nights inserts).
    No free trend is fitted to a star's own light curve;
  then CLASS flags (each independently; manual flags untouched), PERIOD, n_nights, first/last night,
  best_* summaries.
- PERIOD priority: manual > literature period (period_err = catalogue error) > the object's latest
  `period_estimate` (most nights, then latest) with FAP <= `db.ls_fap_threshold` (a significance gate,
  not a precision gate; period_err and period_n_nights from it) > combined BLS (planet host, depth_snr >=
  bls_min_snr) / combined LS with FAP <= 0.01 (variable) > best per-night LS period > NULL;
  `period_source` records which.

## Web (requirements 10-15)

- Query window: CLASS, KNOWN, SOURCE_DB, status, cone (RA, Dec, radius"), mag, period, SNR, depth, tier,
  n_nights, night date range, telescope, name / Gaia id; advanced read-only SQL box (relphot_ro, timeout).
- Paged sortable results with a Load button per row; CSV export.
- Detail: metadata, catalogue matches, detections; one button per observed night (loads that night's LC;
  hover shows image file name); "All nights" button (tie-calibrated magnitudes when a multi-night run
  covers the nights, otherwise per-night-normalised flux, labelled as untied); phase diagram at PERIOD
  with editable period and x2 / /2; periodogram panel; manual status / notes.
- Search adds the filters `is_exop`, `is_var` (tri-state), class `EXOP+VAR`, and `min_p_match` (objects with
  any transit pair matching with p >= x), and the result columns `is_exop`, `is_var`, `period_err`,
  `n_transit_events`, `max_p_match` and, for literature variables, the latest `period_delta` +/-
  `period_delta_err` (CSV export included).
- Detail adds a "Transit events" table (night, tc, depth, T14, ingress fraction with errors, tier, flags, and a
  status selector per event, `PATCH /api/detection/{det_id}`), a "Matching transits" table (pair, dt, z-scores,
  p(match), first commensurate periods) with a note that events are never merged automatically, the trapezoid
  fit drawn over the night's light curve, and a "Period verification" table plus a plot of delta against the
  number of nights.
- Manual classification is two checkboxes (Exoplanet host, Variable), each with its own reset-to-auto;
  `PATCH /api/object/{id}` takes `is_exop` / `is_var` (sets that flag's source to 'manual') and
  `exop_source` / `var_source` = 'auto'; `class` and `class_source` are re-derived. The advanced SQL box
  reads the new tables (`relphot_ro`).

## Build order

1. podman + quadlet units, db image with q3c, schema + roles.
2. Loader + noise cut, tests on synthetic fixtures.
3. `analyze` + periodogram storage.
4. Multi-night loader.
5. Web API.
6. Web front end.
7. Backfill T80S 20251104/05/06 and ROBO43 20250911; verify counts; pg_dump backup timer; README.
8. Schema v4 (`004_classification.sql`): independent EXOP/VAR flags, transit shapes and pairwise matching,
   period verification history; then `relphot search` (for the ON_VARIABLE flag) and `relphot db analyze --all`.
