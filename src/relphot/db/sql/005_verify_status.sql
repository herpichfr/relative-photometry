-- relphot results database, schema version 5: why a period verification could or could not be
-- made (see docs/DB_PLAN.md, period_estimate).

ALTER TABLE relphot.period_estimate ADD COLUMN verify_status text
    CHECK (verify_status IN (
        'verified', 'no_literature', 'lit_period_outside_grid', 'no_peak_in_window',
        'insufficient_data'
    ));
ALTER TABLE relphot.period_estimate ADD COLUMN verify_note text;

-- Existing rows: the two cases that can be told apart from what is stored; a row with a
-- literature period but no harmonic is left NULL until `relphot db analyze --all` reruns it.
UPDATE relphot.period_estimate SET verify_status = 'no_literature'
WHERE lit_period IS NULL;
UPDATE relphot.period_estimate SET verify_status = 'verified'
WHERE lit_period IS NOT NULL AND harmonic IS NOT NULL;
