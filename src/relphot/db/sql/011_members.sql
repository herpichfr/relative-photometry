-- relphot results database, schema version 11: which reference and comparison stars built each
-- tile's light curves on each night (trace-back) and the tile-level reference / ensemble curves.
--
-- Arrays are full length (= night.n_frames) and indexed by frame.frame_index; NaN = frame dropped
-- from the night (frame.kept false) or no value. Comparison curves are only stored for the
-- (tile, aperture) pairs some star of the tile uses as its best aperture (`best_apertures`).
-- norm_flux = flux / R / baseline (median over frames), i.e. the quantity the ensemble averages;
-- it is BEFORE decorrelation. Instrumental mags: apparent = mag + night.zp.
-- Stars that failed the DB noise cut have obj_id NULL (or a position match to an existing object).
-- reference_member.weight: normalised inverse-variance weight, sums to 1 per tile (1/n for median_fixed),
--   weight and mag at night_tile.ref_aperture.
-- comparison_member.weight: share of the total ensemble weight, sum_j(w*m) / sum_ij(w*m); sums to 1 per
--   (tile, aperture); 1/n for the median ensemble statistic.
-- clipped_frames: frame_index values where the member was sigma-clipped out of the ensemble mean.
-- tile_lc.n_ensemble = comparison_n_comparison[t,a]; n_comp = comparison_member rows stored (0 if unused).
-- Deleting a night's night_tile rows cascades to everything below (replace-on-reload).

CREATE TABLE relphot.night_tile (
    night_id       integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    tile           integer NOT NULL,
    x_min          double precision,
    x_max          double precision,
    y_min          double precision,
    y_max          double precision,
    n_core         integer,
    n_extended     integer,
    n_ref_stars    integer,
    ref_aperture   smallint,
    best_apertures smallint[] NOT NULL DEFAULT '{}',
    PRIMARY KEY (night_id, tile)
);

CREATE TABLE relphot.tile_lc (
    night_id     integer NOT NULL,
    tile         integer NOT NULL,
    aperture     smallint NOT NULL,
    ref_flux     real[] NOT NULL,
    ref_flux_err real[] NOT NULL,
    ens_flux     real[],
    ens_flux_err real[],
    n_ensemble   integer,
    n_comp       integer NOT NULL DEFAULT 0,
    n_rounds     smallint,
    PRIMARY KEY (night_id, tile, aperture),
    FOREIGN KEY (night_id, tile) REFERENCES relphot.night_tile (night_id, tile) ON DELETE CASCADE
);

CREATE TABLE relphot.reference_member (
    night_id integer NOT NULL,
    tile     integer NOT NULL,
    star_id  integer NOT NULL,
    obj_id   bigint REFERENCES relphot.object (obj_id) ON DELETE SET NULL,
    ra       double precision,
    dec      double precision,
    mag      real,
    weight   real,
    in_core  boolean,
    PRIMARY KEY (night_id, tile, star_id),
    FOREIGN KEY (night_id, tile) REFERENCES relphot.night_tile (night_id, tile) ON DELETE CASCADE
);
CREATE INDEX reference_member_obj_idx ON relphot.reference_member (obj_id) WHERE obj_id IS NOT NULL;

CREATE TABLE relphot.comparison_member (
    night_id       integer NOT NULL,
    tile           integer NOT NULL,
    aperture       smallint NOT NULL,
    star_id        integer NOT NULL,
    obj_id         bigint REFERENCES relphot.object (obj_id) ON DELETE SET NULL,
    ra             double precision,
    dec            double precision,
    mag            real,
    weight         real,
    n_clipped      smallint NOT NULL DEFAULT 0,
    clipped_frames smallint[] NOT NULL DEFAULT '{}',
    norm_flux      real[] NOT NULL,
    PRIMARY KEY (night_id, tile, aperture, star_id),
    FOREIGN KEY (night_id, tile, aperture) REFERENCES relphot.tile_lc (night_id, tile, aperture) ON DELETE CASCADE
);
CREATE INDEX comparison_member_obj_idx ON relphot.comparison_member (obj_id) WHERE obj_id IS NOT NULL;

GRANT SELECT ON relphot.night_tile, relphot.tile_lc, relphot.reference_member,
    relphot.comparison_member TO relphot_ro, relphot_web;
