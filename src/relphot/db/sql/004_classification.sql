-- relphot results database, schema version 4: independent planet-host / variable
-- flags, per-event transit shapes and pairwise "matching transits" probabilities,
-- and period-verification history (see docs/DB_PLAN.md).

-- object: is_exop / is_var replace the exclusive class as the source of truth;
-- class stays as a derived label (EXOP+VAR if both) so existing queries keep working.
ALTER TABLE relphot.object ADD COLUMN is_exop boolean NOT NULL DEFAULT false;
ALTER TABLE relphot.object ADD COLUMN is_var boolean NOT NULL DEFAULT false;
ALTER TABLE relphot.object ADD COLUMN exop_source text
    CHECK (exop_source IN ('auto', 'manual'));
ALTER TABLE relphot.object ADD COLUMN var_source text
    CHECK (var_source IN ('auto', 'manual'));
ALTER TABLE relphot.object ADD COLUMN period_err double precision;
ALTER TABLE relphot.object ADD COLUMN period_n_nights integer;
-- duration_h is a lower limit when the best transit is incomplete (EDGE / PARTIAL / truncated).
ALTER TABLE relphot.object ADD COLUMN duration_lower_limit boolean;

ALTER TABLE relphot.object DROP CONSTRAINT IF EXISTS object_class_check;
ALTER TABLE relphot.object ADD CONSTRAINT object_class_check
    CHECK (class IN ('UNC', 'EXOP', 'VAR', 'EXOP+VAR'));

-- Existing rows: an automatic class becomes the matching automatic flag; a manual
-- class keeps its decision as a manual flag. A manual EXOP (VAR) no longer says
-- anything about the other flag, which goes back to automatic; a manual UNC pins
-- both flags to false.
UPDATE relphot.object SET
    is_exop = (class = 'EXOP'),
    is_var = (class = 'VAR'),
    exop_source = 'auto',
    var_source = 'auto'
WHERE class_source IS DISTINCT FROM 'manual';

UPDATE relphot.object SET
    is_exop = (class = 'EXOP'),
    is_var = (class = 'VAR'),
    exop_source = CASE WHEN class IN ('EXOP', 'UNC') THEN 'manual' ELSE 'auto' END,
    var_source = CASE WHEN class IN ('VAR', 'UNC') THEN 'manual' ELSE 'auto' END
WHERE class_source = 'manual';

CREATE INDEX object_is_exop_idx ON relphot.object (is_exop) WHERE is_exop;
CREATE INDEX object_is_var_idx ON relphot.object (is_var) WHERE is_var;

-- detection: a person's verdict and notes on one event (never set automatically).
ALTER TABLE relphot.detection ADD COLUMN status text
    CHECK (status IN ('UNCONFIRMED', 'CONFIRMED', 'REJECTED')) DEFAULT 'UNCONFIRMED';
ALTER TABLE relphot.detection ADD COLUMN notes text;

-- detection.duration_h of an incomplete transit (flags EDGE or PARTIAL, or found truncated by
-- analyze) is only a minimum and is stored, compared and shown as a lower limit.
ALTER TABLE relphot.detection ADD COLUMN duration_lower_limit boolean NOT NULL DEFAULT false;
UPDATE relphot.detection SET duration_lower_limit = true
WHERE kind = 'transit' AND flags ~ '(^|[|])(EDGE|PARTIAL)([|]|$)';

-- A person's verdict on a detection that a night reload could not re-attach to a new detection.
CREATE TABLE relphot.detection_review_orphan (
    obj_id     bigint NOT NULL,
    night_id   integer,
    kind       text NOT NULL,
    tc_bjd_tdb double precision,
    status     text,
    notes      text,
    saved_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX detection_review_orphan_obj_id_idx ON relphot.detection_review_orphan (obj_id);

-- catalog_match: the catalogue's own period uncertainty, NULL when it gives none.
ALTER TABLE relphot.catalog_match ADD COLUMN period_err double precision;

-- transit_shape: trapezoid fit to one per-night transit detection.
CREATE TABLE relphot.transit_shape (
    det_id       bigint PRIMARY KEY REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    obj_id       bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    tc           double precision,
    tc_err       double precision,
    depth        real,
    depth_err    real,
    t14_h        real,
    t14_err      real,
    t14_lower_limit boolean NOT NULL DEFAULT false,
    incomplete_reason text,
    ingress_frac real,
    ingress_err  real,
    chi2_red     real,
    n_points     integer,
    input        text CHECK (input IN ('tied', 'night')),
    converged    boolean,
    computed_at  timestamptz
);

CREATE INDEX transit_shape_obj_id_idx ON relphot.transit_shape (obj_id);

-- transit_match: probability that two transit events of one object are the same
-- kind of event, from depth and shape. Rows are never used to merge or reject.
CREATE TABLE relphot.transit_match (
    det_a               bigint NOT NULL REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    det_b               bigint NOT NULL REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    obj_id              bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    dt_days             double precision,
    depth_z             real,
    t14_z               real,
    ingress_z           real,
    chi2                real,
    dof                 smallint,
    p_match             real,
    same_telescope      boolean,
    commensurate_periods double precision[],
    computed_at         timestamptz,
    PRIMARY KEY (det_a, det_b),
    CHECK (det_a < det_b)
);

CREATE INDEX transit_match_obj_id_idx ON relphot.transit_match (obj_id);

-- period_estimate: one row per (object, method, set of nights); a new set of
-- nights inserts a new row, so the history of re-observations is kept.
CREATE TABLE relphot.period_estimate (
    est_id          bigserial PRIMARY KEY,
    obj_id          bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    computed_at     timestamptz,
    method          text NOT NULL,
    input           text CHECK (input IN ('tied', 'night')),
    night_ids       integer[] NOT NULL,
    n_nights        integer,
    last_night      date,
    baseline_days   double precision,
    period          double precision,
    period_err      double precision,
    power           real,
    fap             real,
    lit_period      double precision,
    lit_period_err  double precision,
    lit_catalog     text,
    harmonic        real,
    delta           double precision,
    delta_err       double precision,
    delta_z         real,
    UNIQUE (obj_id, method, night_ids)
);

GRANT SELECT ON relphot.transit_shape, relphot.transit_match, relphot.period_estimate,
    relphot.detection_review_orphan TO relphot_ro, relphot_web;
GRANT UPDATE (is_exop, is_var, exop_source, var_source, period_err, period_n_nights)
    ON relphot.object TO relphot_web;
GRANT UPDATE (status, notes) ON relphot.detection TO relphot_web;
