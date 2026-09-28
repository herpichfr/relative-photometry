#!/usr/bin/env bash
# Runs once, as the postgres superuser, when the data directory is first
# initialised (see the postgres image's /docker-entrypoint-initdb.d/
# convention). Creates the relphot login roles, the relphot/relphot_test
# databases owned by relphot_owner, and the q3c extension in each.
set -euo pipefail

: "${RELPHOT_OWNER_PASSWORD:?RELPHOT_OWNER_PASSWORD must be set}"
: "${RELPHOT_WEB_PASSWORD:?RELPHOT_WEB_PASSWORD must be set}"
: "${RELPHOT_RO_PASSWORD:?RELPHOT_RO_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'relphot_owner') THEN
            CREATE ROLE relphot_owner LOGIN PASSWORD '${RELPHOT_OWNER_PASSWORD}';
        END IF;
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'relphot_web') THEN
            CREATE ROLE relphot_web LOGIN PASSWORD '${RELPHOT_WEB_PASSWORD}';
        END IF;
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'relphot_ro') THEN
            CREATE ROLE relphot_ro LOGIN PASSWORD '${RELPHOT_RO_PASSWORD}';
        END IF;
    END
    \$\$;

    ALTER ROLE relphot_ro SET statement_timeout = '30s';
EOSQL

for dbname in relphot relphot_test; do
    exists="$(psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname = '${dbname}'")"
    if [[ "${exists}" != "1" ]]; then
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
            -c "CREATE DATABASE ${dbname} OWNER relphot_owner;"
    fi
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "${dbname}" \
        -c "CREATE EXTENSION IF NOT EXISTS q3c;"
done
