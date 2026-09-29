#!/usr/bin/env bash
# Nightly relphot results-database backup: pg_dump inside the relphotdb-db
# container (podman exec), streamed to the host and written to a .tmp file
# then renamed, keeping only the newest KEEP dumps. Installed as the
# relphotdb-backup.service/.timer user units (see deploy/install.sh backup).
set -euo pipefail

BACKUP_DIR=/ssdsto1/data/relphotDB/backups
CONTAINER=relphotdb-db
DBNAME=relphot
KEEP=14

mkdir -p "${BACKUP_DIR}"

stamp="$(date +%Y%m%d_%H%M)"
out="${BACKUP_DIR}/relphot_${stamp}.dump"
tmp="${out}.tmp"

echo "relphotdb-backup: dumping ${DBNAME} (container ${CONTAINER}) to ${out}"
podman exec "${CONTAINER}" pg_dump -U postgres -Fc "${DBNAME}" > "${tmp}"
mv "${tmp}" "${out}"
echo "relphotdb-backup: wrote ${out} ($(du -h "${out}" | cut -f1))"

mapfile -t dumps < <(ls -1t "${BACKUP_DIR}"/relphot_*.dump 2>/dev/null)
if (( ${#dumps[@]} > KEEP )); then
    for old in "${dumps[@]:KEEP}"; do
        echo "relphotdb-backup: removing old backup ${old}"
        rm -f "${old}"
    done
fi

echo "relphotdb-backup: done, $(ls -1 "${BACKUP_DIR}"/relphot_*.dump | wc -l) dump(s) kept"
