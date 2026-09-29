-- relphot results database, schema version 6: error inflation of blended stars, user-guided
-- reprocessing (request queue, user-origin detections, guided period estimates) and
-- long-period / phase-coverage / alias information on period estimates, and per-night user
-- reviews of the exoplanet / variable classification (see docs/DB_PLAN.md).

-- star_night: the factor the light curve's errors were inflated by (point-to-point excess
-- scatter over the formal error; NULL for a night loaded from a product written before error
-- inflation existed, read as 1) and whether the star's neighbour/blend flags are set.
ALTER TABLE relphot.star_night ADD COLUMN err_scale real;
ALTER TABLE relphot.star_night ADD COLUMN blended boolean;

-- detection.origin: 'search' detections come from a night's search and are replaced when the
-- night is reloaded; 'user' detections come from a person's reprocess request and are never
-- deleted by a reload.
ALTER TABLE relphot.detection ADD COLUMN origin text NOT NULL DEFAULT 'search'
    CHECK (origin IN ('search', 'user'));
CREATE INDEX detection_user_idx ON relphot.detection (obj_id) WHERE origin = 'user';

-- reprocess_request: a person's ON-DEMAND request (the web inserts a row only when the user
-- ticks RERUN and presses the submit button; nothing else ever inserts), worked off by
-- `relphot db reprocess` (owner role). 'variable': a period search around period_guess; night_id
-- names the only night used, NULL = all of the object's nights. 'transit': a trapezoid fit
-- around tc_guess (BJD_TDB) with width_guess_h on night_id (the web always sets it).
CREATE TABLE relphot.reprocess_request (
    req_id        bigserial PRIMARY KEY,
    obj_id        bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    kind          text NOT NULL CHECK (kind IN ('variable', 'transit')),
    period_guess  double precision,
    tc_guess      double precision,
    width_guess_h real,
    night_id      integer REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    note          text,
    status        text NOT NULL DEFAULT 'queued'
                      CHECK (status IN ('queued', 'running', 'done', 'failed')),
    requested_at  timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz,
    error         text,
    result        jsonb,
    CHECK (kind <> 'variable' OR COALESCE(period_guess > 0, false)),
    CHECK (kind <> 'transit' OR COALESCE(tc_guess IS NOT NULL AND width_guess_h > 0, false))
);

CREATE INDEX reprocess_request_queue_idx ON relphot.reprocess_request (requested_at, req_id)
    WHERE status = 'queued';
CREATE INDEX reprocess_request_obj_idx ON relphot.reprocess_request (obj_id, requested_at);

-- A new request wakes the worker (`relphot db reprocess --watch` LISTENs on this channel).
CREATE FUNCTION relphot.notify_reprocess_request() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('relphot_reprocess', NEW.req_id::text);
    RETURN NEW;
END
$$;

CREATE TRIGGER reprocess_request_notify AFTER INSERT ON relphot.reprocess_request
    FOR EACH ROW EXECUTE FUNCTION relphot.notify_reprocess_request();

-- period_estimate: user-guided estimates (method 'LS-guided', the guess kept in `guess`),
-- phase coverage, cycles spanned, and alias / next-peak candidates with their LS powers.
ALTER TABLE relphot.period_estimate ADD CONSTRAINT period_estimate_method_check
    CHECK (method IN ('LS', 'LS-guided'));
ALTER TABLE relphot.period_estimate ADD COLUMN guess double precision;
ALTER TABLE relphot.period_estimate ADD COLUMN phase_coverage real;
ALTER TABLE relphot.period_estimate ADD COLUMN n_cycles real;
ALTER TABLE relphot.period_estimate ADD COLUMN alias_periods double precision[];
ALTER TABLE relphot.period_estimate ADD COLUMN alias_powers real[];

ALTER TABLE relphot.period_estimate DROP CONSTRAINT IF EXISTS period_estimate_verify_status_check;
ALTER TABLE relphot.period_estimate ADD CONSTRAINT period_estimate_verify_status_check
    CHECK (verify_status IN (
        'verified', 'no_literature', 'lit_period_outside_grid', 'no_peak_in_window',
        'insufficient_data', 'long_period_needs_tie'
    ));

GRANT SELECT ON relphot.reprocess_request TO relphot_ro, relphot_web;
GRANT INSERT (obj_id, kind, period_guess, tc_guess, width_guess_h, night_id, note)
    ON relphot.reprocess_request TO relphot_web;
GRANT USAGE ON SEQUENCE relphot.reprocess_request_req_id_seq TO relphot_web;

-- user_night_review: per-night user verdicts on exoplanet/variable classification.
-- Null verdict = auto (overridable per night); CONFIRMED/REJECTED = user override.
-- Literature (known planet/variable) always counts and cannot be removed by verdict.
-- Verdicts survive a reload of the same night; new nights are evaluated automatically.
CREATE TABLE relphot.user_night_review (
    obj_id       bigint  NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_id     integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    exop_verdict text CHECK (exop_verdict IN ('CONFIRMED', 'REJECTED')),
    var_verdict  text CHECK (var_verdict IN ('CONFIRMED', 'REJECTED')),
    note         text,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (obj_id, night_id)
);
CREATE INDEX user_night_review_night_idx ON relphot.user_night_review (night_id);

-- Counters for review status
ALTER TABLE relphot.object ADD COLUMN n_review_pending integer NOT NULL DEFAULT 0;
ALTER TABLE relphot.object ADD COLUMN n_nights_reviewed integer NOT NULL DEFAULT 0;
CREATE INDEX object_review_pending_idx ON relphot.object (n_review_pending) WHERE n_review_pending > 0;

-- Backfill: object-level manual flags become verdicts on every night the object has now.
-- No-op on live DB (no manual flags yet) but correct for tests.
INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict, var_verdict)
SELECT o.obj_id, sn.night_id,
       CASE WHEN o.exop_source = 'manual' THEN CASE WHEN o.is_exop THEN 'CONFIRMED' ELSE 'REJECTED' END END,
       CASE WHEN o.var_source  = 'manual' THEN CASE WHEN o.is_var  THEN 'CONFIRMED' ELSE 'REJECTED' END END
FROM relphot.object o JOIN relphot.star_night sn ON sn.obj_id = o.obj_id
WHERE o.exop_source = 'manual' OR o.var_source = 'manual';

GRANT SELECT ON relphot.user_night_review TO relphot_ro, relphot_web;
GRANT INSERT (obj_id, night_id, exop_verdict, var_verdict, note) ON relphot.user_night_review TO relphot_web;
GRANT UPDATE (exop_verdict, var_verdict, note, updated_at) ON relphot.user_night_review TO relphot_web;
GRANT DELETE ON relphot.user_night_review TO relphot_web;
GRANT UPDATE (n_review_pending, n_nights_reviewed) ON relphot.object TO relphot_web;
