-- relphot results database, schema version 7: the photometric zero point of a night, so that
-- relphot's instrumental magnitudes can be shown as apparent ones.
--
-- A relphot magnitude is m = -2.5 log10(F / R) with F the star's flux in counts per exposure
-- (R is the night-normalised reference, ~1), the same convention as robo43's
-- MAG = zp - 2.5 log10(flux); the apparent magnitude is therefore m + zp.
--
-- zp is the median of the kept frames' Gaia-calibrated ZPABS (robo43 SCI header; only frames
-- with ZPABSCAL true) when at least half of the kept frames carry one (zp_source 'gaia'),
-- else the assumed 20.0 mag, robo43's instrumental_zp (zp_source 'assumed'). Nights loaded
-- before this version take the assumed value: none of their frames' headers carries a
-- calibration. `relphot db load-night` sets both columns from the frame metadata of
-- night.npz; a night is re-loaded to pick up a calibration.
ALTER TABLE relphot.night ADD COLUMN zp real NOT NULL DEFAULT 20.0;
ALTER TABLE relphot.night ADD COLUMN zp_source text NOT NULL DEFAULT 'assumed'
    CHECK (zp_source IN ('gaia', 'assumed'));
