-- relphot results database, schema version 12: repeated transit events of one object (see
-- relphot.db.families).
--
-- Two eligible transit events of an object (not rejected by the person, not auto-rejected unless
-- confirmed) that look alike -- same depth / T14 / ingress from the trapezoid fits and from a
-- joint refit of their light curves -- are a scored LINK; a maximal set of mutually linked events
-- whose centre times share an integer-epoch period is a FAMILY. Events are never merged and no
-- detection.status / object.class is touched: a family is a candidate the person accepts or
-- rejects (repeat_decision), and its allowed periods give the windows of future transits.
-- Everything but repeat_decision is recomputed by `relphot db analyze` / `relphot db families`
-- and goes with its detections when a night is reloaded.

-- repeat_link: every pair of eligible events of one object. linked = the pair is an edge of the
-- family graph (p_match and p_joint >= repeat_p_min, or the person said SAME, and not DIFFERENT);
-- p_match is the existing depth/T14/ingress z-score probability (T14 systematic floor and blend
-- depth floor added), p_joint the likelihood-ratio test of a common trapezoid (chi2_joint is the
-- delta chi2, dof_joint its degrees of freedom; NULL without enough light-curve points);
-- phys_ok: some period dt/k is long enough for the stellar density bound (P_min); n_alias: how
-- many of them; diurnal: dt is an integer number of days within half a T14 (a nightly systematic
-- at a fixed time looks the same); decision: the person's SAME / DIFFERENT when one is stored.
CREATE TABLE relphot.repeat_link (
    det_a          bigint NOT NULL REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    det_b          bigint NOT NULL REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    obj_id         bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_a        integer NOT NULL,
    night_b        integer NOT NULL,
    dt_days        double precision,
    p_match        real,
    p_joint        real,
    chi2_joint     real,
    dof_joint      smallint,
    phys_ok        boolean NOT NULL,
    n_alias        integer NOT NULL DEFAULT 0,
    diurnal        boolean NOT NULL DEFAULT false,
    involves_loose boolean NOT NULL DEFAULT false,
    linked         boolean NOT NULL,
    decision       text CHECK (decision IN ('SAME', 'DIFFERENT')),
    computed_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (det_a, det_b),
    CHECK (det_a < det_b)
);
CREATE INDEX repeat_link_obj_idx ON relphot.repeat_link (obj_id);

-- repeat_family: one row per family (overlapping families are allowed: an event may be in
-- several). depth / t14_h / ingress_frac summarise the members (t14_h is the longest member T14,
-- a lower limit when the member holding it is incomplete); score is the smallest link p_joint
-- (p_match where none); accepted: the person said SAME on every pair of members.
CREATE TABLE relphot.repeat_family (
    fam_id          bigserial PRIMARY KEY,
    obj_id          bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    family_key      text NOT NULL,
    n_members       integer NOT NULL,
    member_night_ids integer[] NOT NULL,
    involves_loose  boolean NOT NULL DEFAULT false,
    depth           real,
    depth_err       real,
    t14_h           real,
    t14_lower_limit boolean NOT NULL DEFAULT false,
    ingress_frac    real,
    score           real,
    n_alias         integer NOT NULL DEFAULT 0,
    n_allowed       integer NOT NULL DEFAULT 0,
    accepted        boolean NOT NULL DEFAULT false,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (obj_id, family_key)
);
CREATE INDEX repeat_family_obj_idx ON relphot.repeat_family (obj_id);

CREATE TABLE relphot.repeat_family_member (
    fam_id bigint NOT NULL REFERENCES relphot.repeat_family (fam_id) ON DELETE CASCADE,
    det_id bigint NOT NULL REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    PRIMARY KEY (fam_id, det_id)
);
CREATE INDEX repeat_family_member_det_idx ON relphot.repeat_family_member (det_id);

-- repeat_ephemeris: the candidate periods of a family, one row per integer alias k (P = dt/k of
-- the two earliest members; a fit over all members when there are more), tc0 the reference epoch
-- (BJD_TDB) and its error. status: 'allowed'; 'vetoed_density' (P below P_min, the period at
-- which a star of density repeat_rho_max_cgs has a transit as long as the longest member);
-- 'vetoed_nondetection' (on a night of the object the light curve excludes a transit of the
-- family's depth and T14 at every allowed timing: veto_night_id, veto_dchi2). Periods are
-- provisional (R3): rows are keyed by (obj_id, family_key, alias_k) and kept as history when the
-- family is recomputed or its detections are reloaded (fam_id becomes NULL).
CREATE TABLE relphot.repeat_ephemeris (
    eph_id          bigserial PRIMARY KEY,
    fam_id          bigint REFERENCES relphot.repeat_family (fam_id) ON DELETE SET NULL,
    obj_id          bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    family_key      text NOT NULL,
    alias_k         integer NOT NULL,
    period          double precision NOT NULL,
    period_err      double precision,
    tc0             double precision NOT NULL,
    tc0_err         double precision,
    status          text NOT NULL CHECK (
                        status IN ('allowed', 'vetoed_density', 'vetoed_nondetection')
                    ),
    veto_night_id   integer,
    veto_dchi2      real,
    n_nights_tested integer,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (obj_id, family_key, alias_k)
);
CREATE INDEX repeat_ephemeris_fam_idx ON relphot.repeat_ephemeris (fam_id);

-- repeat_decision: the person's verdict on a pair of events, SAME (same planet) or DIFFERENT.
-- Keyed by night and centre time, not by detection, so it survives a night reload: the next
-- recompute attaches it to the pair whose centre times are within half a T14 of (tc_a, tc_b).
-- night_a <= night_b, and tc_a <= tc_b when the nights are equal.
CREATE TABLE relphot.repeat_decision (
    obj_id     bigint  NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_a    integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    night_b    integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    tc_a       double precision NOT NULL,
    tc_b       double precision NOT NULL,
    decision   text NOT NULL CHECK (decision IN ('SAME', 'DIFFERENT')),
    note       text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (obj_id, night_a, night_b, tc_a, tc_b),
    CHECK (night_a <= night_b)
);

GRANT SELECT ON relphot.repeat_link, relphot.repeat_family, relphot.repeat_family_member,
    relphot.repeat_ephemeris, relphot.repeat_decision TO relphot_ro, relphot_web;
GRANT INSERT (obj_id, night_a, night_b, tc_a, tc_b, decision, note)
    ON relphot.repeat_decision TO relphot_web;
GRANT UPDATE (decision, note, updated_at) ON relphot.repeat_decision TO relphot_web;
GRANT DELETE ON relphot.repeat_decision TO relphot_web;
