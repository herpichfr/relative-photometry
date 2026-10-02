-- relphot results database, schema version 13: a RERUN's transit event supersedes an older event of
-- the same light curve.
--
-- A person's RERUN of one object on one night (relphot.db.reprocess) may fit the very transit the
-- search (or an earlier RERUN) already fitted, e.g. one whose fit did not converge. Both events stay
-- stored, never merged; superseded_by says which one stands: an event with superseded_by set was
-- replaced by the event it points to and no longer counts as an open candidate, nor in the night's
-- EXOP verdict, the repeated-event families or the object's automatic evidence. superseded_by
-- NULL = active. The person can swap them back and forth with the "keep this" button of the
-- Transit events table (relphot.objflags.keep_transit_event).
--
-- Scope: ONE light curve = one object on one night (relphot.lightcurve is keyed by (obj_id,
-- night_id)). The composite foreign key below makes the database refuse a link to an event of
-- another object or another night -- two stars of one night (look-alike events, the Similar
-- events window) are never linked. Only per-night transit events (night_id set, kind 'transit')
-- take part. Links are flat: every superseded event points at the one active event.
--
-- A night reload (relphot db load-night) re-creates the origin = 'search' events with new det_ids;
-- load_night saves the links and re-attaches them (like the person's status / notes). Deleting
-- the event a link points at (ON DELETE SET NULL) just makes the other event active again.

ALTER TABLE relphot.detection
    ADD CONSTRAINT detection_det_obj_night_key UNIQUE (det_id, obj_id, night_id);

ALTER TABLE relphot.detection ADD COLUMN superseded_by bigint;

ALTER TABLE relphot.detection
    ADD CONSTRAINT detection_superseded_by_chk CHECK (
        superseded_by IS NULL
        OR (kind = 'transit' AND night_id IS NOT NULL AND superseded_by <> det_id)
    );

ALTER TABLE relphot.detection
    ADD CONSTRAINT detection_superseded_by_fk
    FOREIGN KEY (superseded_by, obj_id, night_id)
    REFERENCES relphot.detection (det_id, obj_id, night_id)
    ON DELETE SET NULL (superseded_by);

CREATE INDEX detection_superseded_by_idx ON relphot.detection (superseded_by)
    WHERE superseded_by IS NOT NULL;

-- the web's "keep this" button swaps the links; nothing else of detection becomes writable
GRANT UPDATE (superseded_by) ON relphot.detection TO relphot_web;
