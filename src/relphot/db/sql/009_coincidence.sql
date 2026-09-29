-- relphot results database, schema version 9: automatic cross-candidate check of the per-night
-- transit events (see docs/DB_PLAN.md, "Coincident events").
--
-- One planet cannot transit two stars at once: an event with many look-alikes on the SAME night
-- (other objects' events with the same centre time and similar depth and T14) is a systematic,
-- and `relphot db analyze` marks it auto-rejected. detection.status stays the PERSON's verdict
-- and is never set automatically; the automatic verdict has its own columns.

-- detection.auto_status: NULL = no automatic verdict, 'REJECTED' = too many similar events on
-- the night; auto_reason says why (numbers included). A search detection that is auto-REJECTED
-- and whose status is UNCONFIRMED (or NULL) is not automatic evidence for a flag, is not open for
-- review and is not the object's best transit; a person's CONFIRMED overrides it.
ALTER TABLE relphot.detection ADD COLUMN auto_status text
    CHECK (auto_status IN ('REJECTED'));
ALTER TABLE relphot.detection ADD COLUMN auto_reason text;

-- transit_coincidence: the numbers behind the verdict, one row per per-night search transit event
-- with a converged trapezoid fit. n_similar: events of other detections within the time window
-- whose depth and T14 are similar; n_expected: how many a uniformly random centre time would
-- give; p_chance: binomial tail probability of n_similar or more; similar_det_ids: those events,
-- nearest in time first.
CREATE TABLE relphot.transit_coincidence (
    det_id          bigint PRIMARY KEY REFERENCES relphot.detection (det_id) ON DELETE CASCADE,
    night_id        integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    n_similar       integer,
    n_expected      real,
    p_chance        double precision,
    similar_det_ids bigint[],
    rejected        boolean NOT NULL,
    computed_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX transit_coincidence_night_idx ON relphot.transit_coincidence (night_id);

GRANT SELECT ON relphot.transit_coincidence TO relphot_ro, relphot_web;
