-- relphot results database, schema version 2: indexes not already covered by
-- a primary key or unique constraint from 001_init.sql (see docs/DB_PLAN.md).

CREATE INDEX detection_obj_id_idx ON relphot.detection (obj_id);
CREATE INDEX detection_night_id_idx ON relphot.detection (night_id);
CREATE INDEX detection_mn_run_id_idx ON relphot.detection (mn_run_id);
CREATE INDEX detection_kind_idx ON relphot.detection (kind);
CREATE INDEX tie_obj_id_idx ON relphot.tie (obj_id);
CREATE INDEX tie_night_id_idx ON relphot.tie (night_id);
CREATE INDEX catalog_match_catalog_idx ON relphot.catalog_match (catalog);
