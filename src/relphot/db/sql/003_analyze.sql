-- relphot results database, schema version 3: object.data_updated_at and
-- periodogram.coarsened/input/extra columns for `relphot db analyze`
-- (periodograms + PERIOD; see docs/DB_PLAN.md).

ALTER TABLE relphot.object ADD COLUMN data_updated_at timestamptz;
UPDATE relphot.object SET data_updated_at = now();

ALTER TABLE relphot.periodogram ADD COLUMN coarsened boolean NOT NULL DEFAULT false;
ALTER TABLE relphot.periodogram ADD COLUMN input text;
ALTER TABLE relphot.periodogram ADD COLUMN extra jsonb;
