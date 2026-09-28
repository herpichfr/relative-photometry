-- relphot results database, schema version 1 (see docs/DB_PLAN.md).
-- Applied by relphot.db.schema.init_schema as the relphot_owner role.

CREATE SCHEMA IF NOT EXISTS relphot AUTHORIZATION relphot_owner;

CREATE TABLE relphot.night (
    night_id       serial PRIMARY KEY,
    telescope      text NOT NULL,
    night_date     date NOT NULL,
    label          text NOT NULL,
    site_lat       double precision,
    site_lon       double precision,
    site_elev      double precision,
    object         text,
    filter         text,
    n_frames       integer,
    n_kept         integer,
    source_dir     text UNIQUE,
    relphot_commit text,
    settings       jsonb,
    noise_cut      jsonb,
    loaded_at      timestamptz,
    UNIQUE (telescope, label)
);

CREATE TABLE relphot.frame (
    frame_id    bigserial PRIMARY KEY,
    night_id    integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    file_name   text,
    file_path   text,
    date_obs    timestamptz,
    jd_utc      double precision,
    bjd_tdb     double precision,
    exptime     real,
    airmass     real,
    fwhm        real,
    n_sources   integer,
    kept        boolean,
    UNIQUE (night_id, frame_index)
);

CREATE INDEX frame_file_name_idx ON relphot.frame (file_name);

CREATE TABLE relphot.mn_run (
    mn_run_id serial PRIMARY KEY,
    stem      text UNIQUE NOT NULL,
    labels    text[],
    anchor    text,
    settings  jsonb,
    loaded_at timestamptz
);

CREATE TABLE relphot.object (
    obj_id               bigserial PRIMARY KEY,
    name                 text UNIQUE NOT NULL,
    ra                   double precision NOT NULL,
    dec                  double precision NOT NULL,
    gaia_id              text,
    mean_mag             real,
    n_nights             integer,
    class                text CHECK (class IN ('UNC', 'EXOP', 'VAR')),
    class_source         text CHECK (class_source IN ('auto', 'manual')),
    period               double precision,
    period_source        text,
    known                boolean,
    source_db            text,
    known_name           text,
    known_type           text,
    known_period         double precision,
    status               text CHECK (status IN ('UNCONFIRMED', 'CONFIRMED', 'REJECTED'))
                             DEFAULT 'UNCONFIRMED',
    best_snr             real,
    depth                real,
    duration_h           real,
    amplitude            real,
    n_detections         integer,
    first_night          date,
    last_night           date,
    neighbour_sep_arcsec real,
    notes                text,
    updated_at           timestamptz
);

CREATE INDEX object_q3c_idx ON relphot.object (q3c_ang2ipix(ra, dec));
CREATE INDEX object_class_idx ON relphot.object (class);
CREATE INDEX object_known_idx ON relphot.object (known);
CREATE INDEX object_period_idx ON relphot.object (period);
CREATE INDEX object_mean_mag_idx ON relphot.object (mean_mag);

CREATE TABLE relphot.star_night (
    obj_id         bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_id       integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    star_id        integer NOT NULL,
    tile           integer,
    mag            real,
    best_aperture  smallint,
    rms            real,
    expected_noise real,
    chi2_reduced   real,
    n_epochs       integer,
    is_comparison  boolean,
    PRIMARY KEY (obj_id, night_id),
    UNIQUE (night_id, star_id)
);

CREATE TABLE relphot.lightcurve (
    obj_id      bigint NOT NULL,
    night_id    integer NOT NULL,
    frame_index smallint[],
    bjd_tdb     double precision[],
    flux        real[],
    flux_err    real[],
    flux_raw    real[],
    PRIMARY KEY (obj_id, night_id),
    FOREIGN KEY (obj_id, night_id) REFERENCES relphot.star_night (obj_id, night_id)
        ON DELETE CASCADE
);

CREATE TABLE relphot.detection (
    det_id     bigserial PRIMARY KEY,
    obj_id     bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_id   integer REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    mn_run_id  integer REFERENCES relphot.mn_run (mn_run_id) ON DELETE CASCADE,
    kind       text NOT NULL CHECK (
                   kind IN ('transit', 'variable', 'internight', 'ls_periodic', 'bls', 'recurrent')
               ),
    snr        real,
    depth      real,
    tc_bjd_tdb double precision,
    duration_h real,
    tier       smallint,
    flags      text,
    amplitude  real,
    excess     real,
    period     double precision,
    fap        real,
    extra      jsonb,
    CHECK ((night_id IS NOT NULL) <> (mn_run_id IS NOT NULL))
);

CREATE TABLE relphot.catalog_match (
    obj_id     bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    catalog    text NOT NULL,
    name       text NOT NULL,
    type       text,
    period     double precision,
    sep_arcsec real,
    reference  text,
    PRIMARY KEY (obj_id, catalog, name)
);

CREATE TABLE relphot.tie (
    mn_run_id integer NOT NULL REFERENCES relphot.mn_run (mn_run_id) ON DELETE CASCADE,
    obj_id    bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    night_id  integer NOT NULL REFERENCES relphot.night (night_id) ON DELETE CASCADE,
    mag       real,
    mag_err   real,
    PRIMARY KEY (mn_run_id, obj_id, night_id)
);

CREATE TABLE relphot.periodogram (
    obj_id      bigint NOT NULL REFERENCES relphot.object (obj_id) ON DELETE CASCADE,
    scope       text NOT NULL,
    method      text NOT NULL CHECK (method IN ('LS', 'BLS')),
    fmin        double precision,
    df          double precision,
    n           integer,
    power       real[],
    peak_period double precision,
    peak_power  real,
    fap         real,
    computed_at timestamptz,
    PRIMARY KEY (obj_id, scope, method)
);

CREATE TABLE relphot.schema_version (
    version integer PRIMARY KEY
);

-- Roles: relphot_ro (web queries, SELECT only), relphot_web (SELECT all,
-- UPDATE on the manual-edit columns of object only). Both roles and
-- relphot_ro's statement_timeout are created by deploy/db's container init
-- script (as the postgres superuser); this migration only grants on the
-- objects relphot_owner itself owns.
GRANT USAGE ON SCHEMA relphot TO relphot_ro, relphot_web;

GRANT SELECT ON ALL TABLES IN SCHEMA relphot TO relphot_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA relphot GRANT SELECT ON TABLES TO relphot_ro;

GRANT SELECT ON ALL TABLES IN SCHEMA relphot TO relphot_web;
ALTER DEFAULT PRIVILEGES IN SCHEMA relphot GRANT SELECT ON TABLES TO relphot_web;
GRANT UPDATE (class, class_source, status, notes, period, period_source, updated_at)
    ON relphot.object TO relphot_web;
