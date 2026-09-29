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
- CLASS rules (no manual pin; `relphot.objflags.refresh_flags`, run first by every refresh). User
  decisions are per night, never absolute: a person's verdict (CONFIRMED / REJECTED / NULL = auto, one for
  EXOP and one for VAR, in `user_night_review`) overrides only the automatic evidence of the night it is
  set on. `is_exop` = known planet match (catalog 'NASA Exoplanet Archive' or 'TOI') OR a multi-night
  'bls' detection in `db.class_multinight_kinds` OR at least one night that is "on" for EXOP. A night is
  on for EXOP when its verdict is CONFIRMED, or the verdict is NULL and the night has a search
  'transit'/'bls' detection whose status is not REJECTED (a REJECTED detection removes only that event) and
  that is not auto-rejected (see "Coincident events" below; a person's CONFIRMED overrides that).
  `is_var` likewise: known variable match (any other catalog) OR a multi-night
  'internight'/'ls_periodic'/'recurrent' detection in `db.class_multinight_kinds` (default: only
  'recurrent' -- 'bls'/'ls_periodic'/'internight' thresholds are uncalibrated and dominated by night-step
  artefacts) OR a night on for VAR ('variable'/'internight'/'ls_periodic'/'recurrent'). Literature and
  gated multi-night evidence always count and no night verdict removes them. A new night is always
  evaluated automatically, even for an object with verdicts on older nights (those apply only to their
  own nights). Detections a person created by reprocessing (`origin = 'user'`) never count. There is no
  precedence between the two flags. `exop_source` / `var_source` are 'manual' iff the object has a verdict
  of that kind on some night; `class` is only a derived label (EXOP+VAR if both, else EXOP / VAR / UNC),
  `class_source` is 'manual' iff either source is. `n_review_pending` counts the nights with an
  UNCONFIRMED search transit/variable event and no verdict for that flag, plus the UNCONFIRMED gated
  multi-night events (filter `needs_review`); `n_nights_reviewed` counts the nights with a verdict or note
  of the person's or a CONFIRMED/REJECTED event (filter `user_reviewed`). `object.status` / `notes` are
  informational and never change the class. Since the BLS PERIOD branch requires `is_exop`, gating 'bls'
  out of CLASS also removes it from the PERIOD 'BLS' source until `class_multinight_kinds` includes it.
- The per-night search no longer drops a transit event because the star is also variable (or a known
  variable of a disqualifying type): the event stays a candidate and carries the informational
  `ON_VARIABLE` flag (it never changes the tier). TOO_DEEP events still route to variables.
- `object.status` and `detection.status` are only ever set by a person (the automatic cross-candidate verdict
  has its own column, `detection.auto_status`). A reload of a night keeps
  them: saved `status`/`notes` of the night's detections are re-attached to the new detection of the
  same object and kind (a transit only if its tc is within half its duration; any other kind by object
  and kind); what cannot be matched is logged and kept in `detection_review_orphan`. Objects a person
  touched (per-night review verdicts, period source, non-default status, notes) are never dropped as orphans
  (a reload of the same night reuses its night_id and keeps the verdicts intact).
- R6: a transit duration is reported for every event, but for an incomplete transit (flags EDGE or
  PARTIAL, a fit window truncated by the night's first/last epoch, or a gap longer than 2x the median
  cadence that contains the predicted ingress or egress -- a gap in the middle of the transit does not
  shorten it) it is only a minimum: it is stored with a lower-limit flag
  (`detection.duration_lower_limit`, `transit_shape.t14_lower_limit` + `incomplete_reason`,
  `object.duration_lower_limit`), compared one-sidedly, and shown as "≥ x.xx h", never as a measurement.
- Errors of blended / noisier-than-formal stars are inflated at the light-curve stage (schema v6), so the
  search, the multi-night tie and the database all use them. Per star and aperture
  `err_scale = max(1, sigma_p2p / median(lc_err))`, `sigma_p2p` = 1.4826 x MAD of the first differences of
  consecutive kept epochs / sqrt 2 (a transit's edges and any slow variability add a handful of differences
  at most, so it is transit-safe: no trend is fitted). `lightcurve.inflate_errors` chooses who gets it:
  `none`, `blended`, `excess` (default), `all`. "Blended" is data-defined (no catalogue exists at that stage):
  the SExtractor neighbour/blend FLAGS bits (1, 2) are set in >= `lightcurve.blend_min_frame_fraction`
  (0.05) of the star's kept frames. `excess` = blended stars plus any star whose measured factor is
  >= `lightcurve.err_scale_excess_min` (1.5): on ROBO43 20250911 the excess is a bright-star noise floor
  (WASP-145 A, star 516: point-to-point 6.6 ppt against a formal 1.8 ppt, factor 3.7; the unblended
  comparison stars of the same brightness show 3.4-3.8), which "blended" alone would not cover.
  `lc_err` in `*_lightcurves.parquet` is the inflated error (`lc_err_raw` keeps the formal one);
  `*_starstats.parquet` and `star_night` carry `err_scale` and `blended`; `expected_noise` stays the
  formal prediction (so the noise cut is unchanged) while `chi2_reduced` uses the inflated error.
- User-guided reprocessing (schema v6). Reprocessing is strictly on demand: only the web's RERUN submit inserts
  a request; reloads, analyze, edits and adopt never do. A person can queue a re-run of one object with their
  own guesses (`reprocess_request`, INSERT-only for the web role, `NOTIFY relphot_reprocess` by trigger); a
  host worker (`relphot db reprocess --watch`, owner role) works the queue off FIFO, one transaction per
  request, failures marked with their error text. A variable request is a period search in the windows
  `guess x h x (1 +/- db.guided_period_window_frac)`, h in (0.5, 1, 2) (default 0.2), refined by the Fourier
  fit and stored as a `period_estimate` with `method = 'LS-guided'` (the guess in `guess`; a later request on
  the same nights replaces it, the request history keeps every result); it never changes PERIOD -- the web's
  "Adopt as period" sets a manual period with the estimate's error. A variable request with night_id runs on
  that night alone (no tie); night_id NULL means all nights. A transit request is the D1 trapezoid
  fit started at the guessed centre/width; the result is a new `transit_shape` on a NEW detection with
  `origin = 'user'` (a search detection within half the width is linked in `extra`, never overwritten).
  User detections take part in the matching-transit table (never merged), are not deleted by a night reload
  (a reload replaces `origin = 'search'` only, and spares objects holding a user detection or a request), and
  never set `is_exop` or the best-transit summary.
- Long periods (schema v6): per-night-normalised flux and a free offset per night both absorb variability
  slower than a night. A period estimate whose chosen peak is longer than `db.long_period_days` (1 d) uses
  tie-calibrated magnitudes (a tie covering >= 2 of the object's nights defines the data, only those nights)
  and a Fourier fit with a single offset; without such a tie it is stored with
  `verify_status = 'long_period_needs_tie'`, no `period_err`, no verification and is never the object's
  PERIOD. The note says when the baseline covers < 2 cycles ("unconstrained beyond the lower bound"), and a
  literature-window peak on the LS grid's edge is not a resolved peak, so never `verified`. Every estimate stores
  `phase_coverage` (fraction of 20 phase bins with a point), `n_cycles` (baseline / period) and, for >= 2
  nights, `alias_periods` / `alias_powers` (the one-day aliases |f +/- 1 d^-1|, the two-day aliases and the
  next-highest LS peaks, at most `db.max_alias_candidates` = 5).
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
  n_review_pending int not null default 0, n_nights_reviewed int not null default 0 /* see CLASS rules */,
  period float8 /* days */, period_source text, period_err float8, period_n_nights int,
  duration_lower_limit bool /* duration_h is a minimum: the best transit is incomplete */,
  known bool, source_db text, known_name text, known_type text, known_period float8,
  status text check in ('UNCONFIRMED','CONFIRMED','REJECTED') default 'UNCONFIRMED',
  best_snr real, depth real, duration_h real, amplitude real, n_detections int,
  first_night date, last_night date, neighbour_sep_arcsec real, notes text, updated_at timestamptz)`;
  q3c index `q3c_ang2ipix(ra, dec)`; indexes on class, known, period, mean_mag.
- `star_night(obj_id fk, night_id fk, star_id int, tile int, mag real, best_aperture smallint, rms real,
  expected_noise real, chi2_reduced real, n_epochs int, is_comparison bool,
  err_scale real /* factor lc_err was inflated by; NULL = product predates it, read as 1 */,
  blended bool /* neighbour/blend flags set in enough kept frames */, pk (obj_id, night_id))`;
  unique (night_id, star_id).
- `lightcurve(obj_id, night_id, pk (obj_id, night_id) fk star_night, frame_index smallint[],
  bjd_tdb float8[], flux real[], flux_err real[], flux_raw real[])`  — kept epochs only, best aperture.
- `detection(det_id bigserial pk, obj_id fk, night_id fk null, mn_run_id fk null,
  kind text check in ('transit','variable','internight','ls_periodic','bls','recurrent'),
  snr real, depth real, tc_bjd_tdb float8, duration_h real, tier smallint, flags text,
  amplitude real, excess real, period float8, fap real, extra jsonb,
  status text check in ('UNCONFIRMED','CONFIRMED','REJECTED') default 'UNCONFIRMED', notes text,
  duration_lower_limit bool not null default false /* EDGE/PARTIAL at load; analyze resets it to (flags OR shape verdict) */,
  origin text not null default 'search' check in ('search','user') /* 'user': a reprocess request's event,
  never deleted by a night reload, never sets is_exop */,
  auto_status text check in ('REJECTED') /* schema v9: NULL = no automatic verdict; set only by
  `analyze`'s cross-candidate check, never by a person */, auto_reason text)`;
  exactly one of night_id / mn_run_id set.
- `catalog_match(obj_id fk, catalog text, name text, type text, period float8, period_err float8 /* NULL if the
  catalogue gives none */, sep_arcsec real, reference text, pk (obj_id, catalog, name))`.
- `transit_shape(det_id pk fk detection on delete cascade, obj_id, tc float8, tc_err, depth real, depth_err,
  t14_h real, t14_err, t14_lower_limit bool not null default false, incomplete_reason text,
  ingress_frac real /* T12/T14 in [0, 0.5] */, ingress_err, chi2_red real, n_points int,
  input text /* 'tied' | 'night' */, converged bool, computed_at)` -- trapezoid fit to one per-night transit.
  For an incomplete event `t14_h` is the observed in-transit span (a lower limit), `t14_err` is NULL and
  `ingress_frac`/`ingress_err` are NULL unless both ingress and egress were observed.
- `transit_coincidence(det_id pk fk detection on delete cascade, night_id fk night on delete cascade, n_similar int,
  n_expected real, p_chance float8, similar_det_ids bigint[] /* nearest in time first */, rejected bool not null,
  computed_at)` (schema v9) -- the numbers behind `detection.auto_status`; one row per per-night search transit
  event with a converged shape; rewritten for a whole night by every `analyze` that touches it.
- `user_night_review(obj_id fk, night_id fk, exop_verdict text check in ('CONFIRMED','REJECTED'),
  var_verdict text check in ('CONFIRMED','REJECTED'), note text, updated_at, pk (obj_id, night_id))`
  -- per-night user verdicts on exoplanet/variable classification; NULL verdict = auto (overridable per night).
  Literature (known planets/variables) always counts and verdicts cannot remove evidence. Verdicts survive
  a reload of the same night; new nights are evaluated automatically.
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
  'lit_period_outside_grid', 'no_peak_in_window', 'insufficient_data'; v6 adds 'long_period_needs_tie' */,
  verify_note text /* e.g. "P_lit 217 d > LS max period 1.96 d" */,
  guess float8 /* v6: the user's period guess of an 'LS-guided' row */,
  phase_coverage real /* v6: fraction of 20 phase bins with a point */, n_cycles real /* baseline / period */,
  alias_periods float8[], alias_powers real[] /* v6: alias / next-peak candidates, decreasing power */,
  method check in ('LS','LS-guided'), unique (obj_id, method, night_ids))` -- one row per set of nights (history); an
  object with a literature period gets a row for every re-observation even when it cannot be verified
  (then harmonic/delta are NULL and the period is the data's own global LS peak).
- `reprocess_request(req_id bigserial pk, obj_id fk on delete cascade, kind text check in ('variable','transit'),
  period_guess float8, tc_guess float8 /* BJD_TDB */, width_guess_h real, night_id int null
  /* variable: the only night used, NULL = all nights; transit: the night of tc; ON DELETE CASCADE */,
  note text, status text check in ('queued','running','done','failed') default 'queued', requested_at, started_at,
  finished_at, error text, result jsonb)` -- a variable request needs period_guess > 0, a transit one tc_guess
  and width_guess_h > 0 (table checks); an insert sends `NOTIFY relphot_reprocess`.
- `mn_run(mn_run_id serial pk, stem text unique, labels text[], anchor text, settings jsonb, loaded_at,
  loose_night_ids integer[] not null default '{}' /* schema v10: the run's loose nights */)`;
  `tie(mn_run_id, obj_id, night_id, mag real, mag_err real, pk (mn_run_id, obj_id, night_id))` — the
  night's mean calibrated magnitude from the multi-night tie (`mlc_night_mean_mag`).
- `periodogram(obj_id fk, scope text /* 'night:<night_id>' or 'combined' */, method text check in ('LS','BLS'),
  fmin float8, df float8, n int, power real[], peak_period float8, peak_power real, fap real,
  computed_at, pk (obj_id, scope, method))` — frequency grid f_k = fmin + k*df, cycles/day.
- `schema_version(version int)`.
- Roles: `relphot_owner` (loader), `relphot_ro` (web queries, SELECT only, statement_timeout 30 s);
  the web's manual edits (per-night verdicts, status, notes, period) use a third role `relphot_web` with
  INSERT/UPDATE/DELETE on `user_night_review`, UPDATE on the derived flag columns of `object` (it re-derives
  them itself after an edit) and on `detection.status` / `detection.notes` only, plus INSERT (obj_id, kind, guesses,
  night_id, note only -- never the status or result) on `reprocess_request`.

## Loader (`relphot db ...`, runs on the host, DSN from `RELPHOT_DB_DSN` or `~/.config/relphot/db.env`)

- `relphot db init` — create/upgrade the schema (idempotent, versioned SQL files).
- `relphot db load-night NIGHT_RELPHOT_DIR [--telescope T] [--label YYYYMMDD]` — one transaction; a reload
  deletes that night's star_night/lightcurve/`origin = 'search'` detection rows first. Frames from `night.npz`
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
  - coincident events (after every chunk is committed): the cross-candidate check below, on every night
    that has a transit shape of a target object -> `transit_coincidence`, `detection.auto_status`;
  - period estimate / verification, for every variable or object with a literature variable period: combined
    LS (tied magnitudes when a tie covers the nights, else per-night-normalised flux), global peak;
    with a literature period the LS peak is instead taken in the windows lit_period x h x (1 +/-
    `db.lit_period_window_frac`) for h in (0.5, 1, 2) and verified there (an eclipsing binary found at half
    the catalogued period reports harmonic 0.5); the peak is refined by a 2-harmonic Fourier fit with one
    offset per night (period_err = sigma_f / f^2, scaled by max(1, sqrt(chi2_red))) -> `period_estimate`
    (upserted on (obj_id, method, night_ids): the same nights update, a new set of nights inserts).
    No free trend is fitted to a star's own light curve;
  then CLASS flags (each independently, from the per-night verdicts, literature and detections),
  PERIOD, n_nights, first/last night,
  best_* summaries.
- `relphot db reprocess [--once | --watch] [--poll-seconds 60]` — the worker for the user-guided requests
  (owner role, on the host): `--watch` LISTENs on `relphot_reprocess` with a poll fallback until SIGTERM;
  deployed as the user unit `deploy/systemd/relphotdb-worker.service` (`deploy/install.sh worker`).
- PERIOD priority: manual > literature period (period_err = catalogue error) > the object's latest
  `method = 'LS'` `period_estimate` (most nights, then latest; not a user-guided nor a `long_period_needs_tie`
  one) with FAP <= `db.ls_fap_threshold` (a significance gate,
  not a precision gate; period_err and period_n_nights from it) > combined BLS (planet host, depth_snr >=
  bls_min_snr) / combined LS with FAP <= 0.01 (variable) > best per-night LS period > NULL;
  `period_source` records which.

## Web (requirements 10-15)

- Query window: CLASS, KNOWN, SOURCE_DB, status, needs_review (objects awaiting per-night review verdicts),
  user_reviewed (objects with at least one verdict set), cone (RA, Dec, radius"), mag, period, SNR, depth,
  tier, n_nights, night date range, telescope, name / Gaia id; advanced read-only SQL box (relphot_ro, timeout).
- Paged sortable results with a Load button per row; CSV export.
- Detail: metadata, catalogue matches, detections; one button per observed night (loads that night's LC;
  hover shows image file name); per-night review table (verdicts, note); "All nights" button (tie-calibrated
  magnitudes when a multi-night run covers the nights, otherwise per-night-normalised flux, labelled as
  untied); phase diagram at PERIOD with editable period and x2 / /2; periodogram panel; manual status / notes.
- Search adds the filters `is_exop`, `is_var` (tri-state), class `EXOP+VAR`, and `min_p_match` (objects with
  any transit pair matching with p >= x), and the result columns `is_exop`, `is_var`, `period_err`,
  `n_transit_events`, `max_p_match`, `n_review_pending`, `n_nights_reviewed` and, for literature variables,
  the latest `period_delta` +/- `period_delta_err` (CSV export included).
- Detail adds a "Transit events" table (night, tc, depth, T14, ingress fraction with errors, tier, flags, and a
  status selector per event, `PATCH /api/detection/{det_id}`; an auto-rejected event shows "REJECTED (auto)" with
  its reason and a collapsible list of its first 20 look-alikes as links to their objects), a "Matching transits" table (pair, dt, z-scores,
  p(match), first commensurate periods) with a note that events are never merged automatically, the trapezoid
  fit drawn over the night's light curve, and a "Period verification" table plus a plot of delta against the
  number of nights.
- Detail adds, below the light-curve plot, a phase diagram of ALL nights (tie-calibrated magnitudes when a
  multi-night run ties >= 2 of them, else per-night flux labelled untied) for variables and any object with a
  PERIOD or guided period: points coloured by night with error bars, a period selector (PERIOD, latest
  estimate, latest guided, literature, a typed value, x2 and /2 buttons, the stored alias candidates as
  buttons), epoch = first epoch or the fitted Fourier phase zero (deepest minimum), the 2-harmonic model curve,
  an optional -0.25..1.25 range, and the phase coverage (a warning below 50 %). `GET /api/object/{id}/phase`
  serves the payload.
- Detail has a RERUN checkbox. Checked, it opens entries (night, EXOP and/or VAR; EXOP: transit centre, or a
  light-curve click, and width 0.1–12 h; VAR: period guess and 'All nights rerun'); 'Add another rerun' adds
  another night's entry; the run button POSTs `{entries:[...], note}` to `/api/object/{id}/reprocess`, which
  validates every entry and queues them atomically (one request per EXOP or VAR). The request history and queue
  depth are polled (`GET /api/object/{id}/reprocess`) and a finished request shows its result; a guided period
  offers "Adopt as period" (`POST /api/object/{id}/adopt_period`: a manual PERIOD with the estimate's error).
- Manual classification is per-night user verdicts (CONFIRMED / REJECTED / auto), stored in
  `user_night_review`. The object-level edit endpoint `PATCH /api/object/{id}` with `is_exop` / `is_var`
  is a shorthand that upserts CONFIRMED/REJECTED verdicts on all nights the object currently has data for;
  `exop_source` / `var_source` = 'auto' resets those verdicts to NULL on all nights. Per-night edits
  use `PUT /api/object/{id}/night/{night_id}/review` to set verdicts directly; literature (known
  planets/variables) always counts and cannot be removed by a verdict. Verdicts are re-evaluated by
  `refresh_flags`, and `class` / `class_source` are re-derived. The advanced SQL box reads the new
  tables (`relphot_ro`).

## Loose nights of a multi-night run (schema v10)

A night that would distort the zero-point tie of the others (a cloudy one) is tied *loosely*:
`relphot multinight ... --loose LABEL` (setting `multinight.loose_nights`) ties the other, core, nights exactly as
without it and fits the loose night only to their fixed frame with a low-order surface and its own measured
calibration floor. The run is loaded as one `mn_run` holding all the nights (the core nights' `tie` rows are
identical to a run without the loose night, so it can replace that run: drop the old one with
`drop_mn_run.py`); `mn_run.loose_night_ids` lists the loose `night_id`s. A loose night takes part in the variability
work (inter-night chi2, whose candidates need support from the core nights alone: setting
`multinight.loose_internight_support_p`, Lomb-Scargle, per-night verdict recurrence) but never in the transit work:
no `bls` detection, per-night transit event in the period-compatibility search, or `transit_match` pair uses it.
`db analyze` reads the flag from the run that ties the object (most nights covered, then latest loaded); such a run
counts as tying an object even when a loose night of the object has no `tie` row (the tied series omits it).

## Coincident events (schema v9)

One planet cannot transit two stars at once. A per-night transit event whose trapezoid fit has many look-alikes on
the SAME night -- other objects' events with the same centre time and a similar T14 and depth -- is a systematic
(on the live data these coincide with seeing excursions and spread uniformly over the detector). It stays a
detection but `relphot db analyze` marks it `auto_status = 'REJECTED'` with `auto_reason` ("too many similar events:
57 other events on this night within +-6 min with similar depth and T14 (expected 21.3 by chance, p=3e-9)").
`detection.status` remains the person's verdict and is never set automatically.

- Judged and counted: the converged `transit_shape` rows of `kind = 'transit'`, `origin = 'search'` detections of
  the night (user-origin events are neither). For events i != j: window `max(coincidence_tc_frac x min(T14),
  coincidence_tc_nsigma x hypot(tc_err), cadence)` (cadence = median spacing of the night's frames; NaN tc_err = 0);
  T14 similar when `|ln(T14_i/T14_j)| < ln(coincidence_t14_ratio)` (a lower-limit T14 is also similar to any
  duration up to that ratio shorter); depth similar when `|ln(|d_i|/|d_j|)| < ln(coincidence_depth_ratio)`.
  n_i counts the shape-similar j within the window.
- Chance baseline: for a centre time uniform over `[lo, hi]` (first/last frame, widened to hold every event) the
  probability of landing within the window of tc_i is pw_ij (truncated at the span's edges); with M_i shape-similar
  events and pbar_i the mean pw_ij, `n_expected = M_i x pbar_i` and `p_chance = P(Binomial(M_i, pbar_i) >= n_i)`.
- Rejected iff `n_i >= coincidence_min_similar` and `p_chance < coincidence_max_p`. Defaults (`[db]`): tc_frac 0.1,
  tc_nsigma 2, t14_ratio 1.5, depth_ratio 2, min_similar 3, max_p 1e-3. On the four T80S nights loaded so far it
  rejects 26/119, 258/423, 66/126 and 6/79 events; with tc randomised over the night it rejects < 0.1 events per night.
- Effect on classification (`objflags`, `refresh`): a search detection with `auto_status = 'REJECTED'` whose status
  is UNCONFIRMED (or NULL) is treated like a REJECTED one -- not automatic EXOP evidence, not open for review, not the
  object's best transit. A person's CONFIRMED overrides it; a person's REJECTED stays rejected; per-night verdicts in
  `user_night_review` override as before. `analyze` re-judges the WHOLE night from the stored shapes (not only the
  target objects), so the verdict of an event can change when another object of its night is re-analysed, and
  refreshes the objects whose verdict changed. A night reload deletes the search detections, and with them their
  verdict: run `relphot db analyze` again afterwards.
- After migration 009 run `relphot db analyze --all` once to fill the verdicts of the nights already loaded.

## Rerunning after schema v6 (order)

1. `relphot db init` (migration 006), then rebuild/restart the web (`deploy/install.sh web`).
2. `relphot lightcurves` per night (new `lc_err`, `lc_err_raw`, `err_scale`, `blended`).
3. `relphot search` per night (the search reads the inflated `lc_err`: bright-star SNRs fall by their factor),
   `relphot multinight` / `multisearch` (they read the light-curve product too).
4. `relphot db load-night` per night (reload: user detections are kept), `relphot db load-multinight`.
5. `relphot db analyze --all` (long-period series, aliases, coverage), then `deploy/install.sh worker`.

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
9. Schema v6 (`006_guided.sql`): error inflation, user-guided reprocessing (queue, worker, user detections,
   guided estimates), phase diagram, long-period analysis; see "Rerunning after schema v6".
