-- relphot results database, schema version 8: a third source of a night's zero point,
-- 'measured' -- the telescope's zero point from the [db] `telescope_zp` setting (T80S 27.85
-- mag, measured by matching bright isolated stars to Gaia DR3 G), used by `relphot db
-- load-night` when the night's frames carry no Gaia calibration. 'assumed' stays the
-- `assumed_zp` setting (20 mag) for a telescope without a measurement.
--
-- Only the CHECK on night.zp_source is widened; existing rows are untouched, so a night
-- loaded before this version keeps 'assumed' until it is re-loaded or backfilled by hand
-- (deploy/README.md, "Web usage").
ALTER TABLE relphot.night DROP CONSTRAINT IF EXISTS night_zp_source_check;
ALTER TABLE relphot.night ADD CONSTRAINT night_zp_source_check
    CHECK (zp_source IN ('gaia', 'measured', 'assumed'));
