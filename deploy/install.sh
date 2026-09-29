#!/usr/bin/env bash
# Idempotent installer for the relphot results database (steps: `db`, `web`, `backup`, `worker`).
# Usage: deploy/install.sh [db|web|backup|worker]
set -euo pipefail

STEP="${1:-db}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=/ssdsto1/data/relphotDB
CONFIG_DIR="${HOME}/.config/relphot"
ENV_FILE="${CONFIG_DIR}/relphotdb.env"
QUADLET_DIR="${HOME}/.config/containers/systemd"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

# Host port of the web front end (always bound to 127.0.0.1; the container listens on 8050 inside).
# Precedence: RELPHOT_WEB_PORT in the environment > RELPHOT_WEB_PORT= line in ${ENV_FILE} > 8080.
web_port() {
    local port="${RELPHOT_WEB_PORT:-}"
    if [[ -z "${port}" && -f "${ENV_FILE}" ]]; then
        port="$(sed -n 's/^RELPHOT_WEB_PORT=//p' "${ENV_FILE}" | tail -n 1)"
    fi
    port="${port:-8080}"
    if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        echo "invalid RELPHOT_WEB_PORT: ${port}" >&2
        return 1
    fi
    echo "${port}"
}

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

    local port
    port="$(web_port)"

    podman build -t localhost/relphotdb-web:latest -f "${REPO_DIR}/deploy/web/Containerfile" "${REPO_DIR}"

    mkdir -p "${QUADLET_DIR}"
    sed "s#__WEB_PORT__#${port}#g" "${REPO_DIR}/deploy/quadlet/relphotdb-web.container" \
        > "${QUADLET_DIR}/relphotdb-web.container"

    systemctl --user daemon-reload
    systemctl --user restart relphotdb-web.service

    echo "waiting for relphotdb-web to answer requests..."
    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:${port}/api/search?limit=1" >/dev/null 2>&1; then
            echo "relphotdb-web is ready on http://127.0.0.1:${port}"
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

install_worker() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "${ENV_FILE} not found -- run 'deploy/install.sh db' first" >&2
        return 1
    fi

    # the unit runs the host's `relphot` (the same install that runs `relphot db load-night`)
    local relphot_bin
    relphot_bin="$(command -v relphot || true)"
    if [[ -z "${relphot_bin}" ]]; then
        echo "relphot not found on PATH -- install it first: pip install -e '${REPO_DIR}[db]'" >&2
        return 1
    fi

    mkdir -p "${SYSTEMD_USER_DIR}"
    sed "s#__RELPHOT_BIN__#${relphot_bin}#g" "${REPO_DIR}/deploy/systemd/relphotdb-worker.service" \
        > "${SYSTEMD_USER_DIR}/relphotdb-worker.service"

    systemctl --user daemon-reload
    systemctl --user enable relphotdb-worker.service
    systemctl --user restart relphotdb-worker.service

    echo "relphotdb-worker is running: $(systemctl --user is-active relphotdb-worker.service)"
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
    worker)
        install_worker
        ;;
    *)
        echo "unknown step: ${STEP}" >&2
        exit 1
        ;;
esac
