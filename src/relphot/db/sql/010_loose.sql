-- relphot results database, schema version 10: loose nights of a multi-night run.
--
-- `relphot multinight --loose LABEL` ties a night (e.g. a cloudy one) only loosely to the fixed
-- frame of the other nights: it takes part in the variability work but never in the transit
-- work (BLS, cross-night transit matching). `mn_run.loose_night_ids` lists the run's loose
-- nights (night_id); `relphot db analyze` reads it from the run that ties an object. Runs loaded
-- before this version have none.
ALTER TABLE relphot.mn_run ADD COLUMN loose_night_ids integer[] NOT NULL DEFAULT '{}';
