-- relphot results database, schema version 14: edge outliers excluded from a transit shape fit.
--
-- `relphot db analyze` (and a RERUN) now drops an isolated outlier run of one or two epochs at
-- the start or end of the night (relphot.numeric.edge_outlier_mask) before the trapezoid fit, since
-- a high edge epoch makes the rest of the night look like a dip. The three columns below record
-- what the fit saw; the automatic verdicts of relphot.db.coincidence (EDGE_OUTLIER, NO_DIP,
-- NO_BASELINE in detection.auto_reason) are computed from them. All NULL until the object is
-- re-analysed (`relphot db analyze --all`); detection.extra['transit_edge_clip_bjd'] holds the
-- same epochs for the search's own view (loaded with the night).

-- edge_clip_bjd: BJD_TDB of the epochs excluded from the fit; NULL = none (or not yet analysed).
ALTER TABLE relphot.transit_shape ADD COLUMN edge_clip_bjd double precision[];

-- edge_adjacent: the excluded epoch(s) of one end were the only epoch(s) outside the detection's
-- own search box on that side (the box ends right next to the artefact). NULL = not evaluated.
ALTER TABLE relphot.transit_shape ADD COLUMN edge_adjacent boolean;

-- n_outside: epochs of the (edge-clipped) night outside the fitted trapezoid |t - tc| > T14 / 2;
-- fewer than a handful means the baseline is extrapolated, not measured. NULL = no fit.
ALTER TABLE relphot.transit_shape ADD COLUMN n_outside integer;
