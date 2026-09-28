#!/usr/bin/env bash
# Idempotent installer for the relphot results database (steps: `db`, `web`, `backup`).
# Usage: deploy/install.sh [db|web|backup]
set -euo pipefail

STEP="${1:-db}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/ssdsto1/data/relphotDB
CONFIG_DIR="${HOME}/.config/relphot"
ENV_FILE="${CONFIG_DIR}/relphotdb.env"
QUADLET_DIR="${HOME}/.config/containers/systemd"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

gen_secret() {
    python3 -c 'import secrets; print(secrets.token_hex(24))'
}

install_db() {
    mkdir -p "${DATA_DIR}/pgdata" "${DATA_DIR}/backups"
    mkdir -p "${CONFIG_DIR}"

    if [[ ! -f "${ENV_FILE}" ]]; then
        local owner_pw web_pw ro_pw superuser_pw
        owner_pw="$(gen_secret)"
        web_pw="$(gen_secret)"
        ro_pw="$(gen_secret)"
        superuser_pw="$(gen_secret)"

        (
            umask 177
            cat > "${ENV_FILE}" <<-ENVEOF
POSTGRES_USER=postgres
POSTGRES_PASSWORD=${superuser_pw}
RELPHOT_OWNER_PASSWORD=${owner_pw}
RELPHOT_WEB_PASSWORD=${web_pw}
RELPHOT_RO_PASSWORD=${ro_pw}
RELPHOT_DB_DSN=postgresql://relphot_owner:${owner_pw}@127.0.0.1:5433/relphot
RELPHOT_TEST_DSN=postgresql://relphot_owner:${owner_pw}@127.0.0.1:5433/relphot_test
ENVEOF
        )
        chmod 600 "${ENV_FILE}"
        echo "wrote ${ENV_FILE}"
    else
        echo "${ENV_FILE} already exists, leaving it unchanged"
    fi

    podman build -t localhost/relphotdb-db:latest "${REPO_DIR}/deploy/db"

    mkdir -p "${QUADLET_DIR}"
    cp "${REPO_DIR}/deploy/quadlet/relphotdb.network" "${QUADLET_DIR}/relphotdb.network"
    cp "${REPO_DIR}/deploy/quadlet/relphotdb-db.container" "${QUADLET_DIR}/relphotdb-db.container"

    systemctl --user daemon-reload
    systemctl --user start relphotdb-db.service

    echo "waiting for relphotdb-db to accept connections..."
    for _ in $(seq 1 60); do
        if podman exec relphotdb-db pg_isready -U postgres >/dev/null 2>&1; then
            echo "relphotdb-db is ready"
            return 0
        fi
        sleep 1
    done
    echo "relphotdb-db did not become ready in time" >&2
    return 1
}

install_web() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "${ENV_FILE} not found -- run 'deploy/install.sh db' first" >&2
        return 1
    fi

    podman build -t localhost/relphotdb-web:latest -f "${REPO_DIR}/deploy/web/Containerfile" "${REPO_DIR}"

    mkdir -p "${QUADLET_DIR}"
    cp "${REPO_DIR}/deploy/quadlet/relphotdb-web.container" "${QUADLET_DIR}/relphotdb-web.container"

    systemctl --user daemon-reload
    systemctl --user restart relphotdb-web.service

    echo "waiting for relphotdb-web to answer requests..."
    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:8050/api/search?limit=1" >/dev/null 2>&1; then
            echo "relphotdb-web is ready"
            return 0
        fi
        sleep 1
    done
    echo "relphotdb-web did not become ready in time" >&2
    return 1
}

install_backup() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "${ENV_FILE} not found -- run 'deploy/install.sh db' first" >&2
        return 1
    fi

    chmod +x "${REPO_DIR}/deploy/backup.sh"

    mkdir -p "${SYSTEMD_USER_DIR}"
    sed "s#__REPO_DIR__#${REPO_DIR}#g" "${REPO_DIR}/deploy/systemd/relphotdb-backup.service" \
        > "${SYSTEMD_USER_DIR}/relphotdb-backup.service"
    cp "${REPO_DIR}/deploy/systemd/relphotdb-backup.timer" \
        "${SYSTEMD_USER_DIR}/relphotdb-backup.timer"

    systemctl --user daemon-reload
    systemctl --user enable --now relphotdb-backup.timer

    echo "running one backup now..."
    systemctl --user start relphotdb-backup.service
}

case "${STEP}" in
    db)
        install_db
        ;;
    web)
        install_web
        ;;
    backup)
        install_backup
        ;;
    *)
        echo "unknown step: ${STEP}" >&2
        exit 1
        ;;
esac
