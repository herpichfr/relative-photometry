"""Tests for the relphot web API (:mod:`relphot.web.app`) against a live
PostgreSQL test database.

Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for
how it is resolved), plus RELPHOT_RO_PASSWORD/RELPHOT_WEB_PASSWORD to build
``relphot_ro``/``relphot_web`` DSNs on the same test database -- both
skipped with an explicit reason when unavailable. Otherwise these tests run
for real: the ``test_conn`` fixture drops and rebuilds a throwaway
``relphot`` schema on RELPHOT_TEST_DSN, and ``client`` points the FastAPI
app's own DSN resolution at role-scoped DSNs on that same database.
"""

from __future__ import annotations

import os
import re

import psycopg
import pytest
from fastapi.testclient import TestClient

from relphot.db.connect import default_env_path, read_env_value, resolve_dsn
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError


def _test_dsn() -> str:
    try:
        return resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")


def _role_dsn(owner_dsn: str, role: str, password_env_var: str) -> str:
    password = os.environ.get(password_env_var) or read_env_value(
        default_env_path(), password_env_var
    )
    if not password:
        pytest.skip(f"{password_env_var} not available to build a {role} DSN")
    return re.sub(r"//[^:]+:[^@]+@", f"//{role}:{password}@", owner_dsn)


@pytest.fixture
def ids(test_conn) -> dict:
    """Insert a small, fully-linked fixture and return the ids referenced by the tests.

    Objects: UNC01 (plain, gaia_id set), EXOP01 (known exoplanet, transit
    detection, 2 nights), VAR01 (known variable, tied across both nights via
    an mn_run, has an LS periodogram), VAR02 (2 nights, no covering tie ->
    the combined light curve falls back to night-normalised mode).
    """
    conn = test_conn
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO relphot.night (telescope, night_date, label, site_lat, site_lon, site_elev, "
        "n_frames, n_kept, source_dir, loaded_at) VALUES "
        "('T80S', '2025-01-01', '20250101', -30.2, -70.8, 2200, 3, 3, '/data/n1', now()) "
        "RETURNING night_id"
    )
    (night1,) = cur.fetchone()
    cur.execute(
        "INSERT INTO relphot.night (telescope, night_date, label, site_lat, site_lon, site_elev, "
        "n_frames, n_kept, source_dir, loaded_at) VALUES "
        "('T80S', '2025-01-02', '20250102', -30.2, -70.8, 2200, 3, 3, '/data/n2', now()) "
        "RETURNING night_id"
    )
    (night2,) = cur.fetchone()

    for night_id, prefix in ((night1, "n1"), (night2, "n2")):
        for i in range(3):
            cur.execute(
                "INSERT INTO relphot.frame (night_id, frame_index, file_name, file_path, "
                "date_obs, jd_utc, bjd_tdb, exptime, airmass, kept) VALUES "
                "(%s, %s, %s, %s, now(), %s, %s, 90.0, %s, true)",
                (
                    night_id, i, f"{prefix}_frame{i:03d}.fits", f"/data/{prefix}_frame{i:03d}.fits",
                    2460310.5 + i * 0.01, 2460310.5006 + i * 0.01, 1.2 + 0.01 * i,
                ),
            )

    def insert_object(**kwargs) -> int:
        cols = ", ".join(kwargs)
        placeholders = ", ".join(f"%({k})s" for k in kwargs)
        cur.execute(
            f"INSERT INTO relphot.object ({cols}) VALUES ({placeholders}) RETURNING obj_id",
            kwargs,
        )
        return cur.fetchone()[0]

    obj_unc = insert_object(
        name="RP UNC01", ra=50.0, dec=-10.0, gaia_id="1234567890123456789",
        class_source="auto", mean_mag=17.0, n_nights=1, known=False,
    )
    # "class" is a Python keyword and cannot be passed as a kwarg to insert_object();
    # set it with a separate UPDATE, as for every other object below.
    cur.execute("UPDATE relphot.object SET class = 'UNC' WHERE obj_id = %s", (obj_unc,))

    obj_exop = insert_object(
        name="RP EXOP01", ra=10.001, dec=-20.001, class_source="auto", known=True,
        source_db="NASA Exoplanet Archive", known_name="Test b", known_period=3.5,
        mean_mag=13.5, period=3.5, period_source="catalog", best_snr=12.0, depth=0.01,
        duration_h=2.0, status="CONFIRMED", n_nights=2,
    )
    cur.execute("UPDATE relphot.object SET class = 'EXOP' WHERE obj_id = %s", (obj_exop,))

    obj_var = insert_object(
        name="RP VAR01", ra=200.0, dec=40.0, class_source="auto", known=True,
        source_db="VSX", known_name="V* Test", known_type="EA", known_period=1.2,
        mean_mag=14.0, period=1.2, period_source="catalog", amplitude=0.3, n_nights=2,
    )
    cur.execute("UPDATE relphot.object SET class = 'VAR' WHERE obj_id = %s", (obj_var,))

    obj_var2 = insert_object(
        name="RP VAR02", ra=201.0, dec=41.0, class_source="auto", known=False,
        mean_mag=16.0, n_nights=2,
    )
    cur.execute("UPDATE relphot.object SET class = 'VAR' WHERE obj_id = %s", (obj_var2,))

    def insert_star_night(obj_id: int, night_id: int, star_id: int) -> None:
        cur.execute(
            "INSERT INTO relphot.star_night (obj_id, night_id, star_id, tile, mag, "
            "best_aperture, rms, expected_noise, chi2_reduced, n_epochs, is_comparison) "
            "VALUES (%s, %s, %s, 0, 14.0, 1, 0.01, 0.01, 1.0, 3, false)",
            (obj_id, night_id, star_id),
        )

    insert_star_night(obj_unc, night1, 0)
    insert_star_night(obj_exop, night1, 1)
    insert_star_night(obj_exop, night2, 1)
    insert_star_night(obj_var, night1, 2)
    insert_star_night(obj_var, night2, 2)
    insert_star_night(obj_var2, night1, 3)
    insert_star_night(obj_var2, night2, 3)

    def insert_lightcurve(obj_id: int, night_id: int, flux) -> None:
        cur.execute(
            "INSERT INTO relphot.lightcurve (obj_id, night_id, frame_index, bjd_tdb, flux, "
            "flux_err, flux_raw) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                obj_id, night_id, [0, 1, 2],
                [2460310.5006, 2460310.5106, 2460310.5206],
                flux, [0.01, 0.01, 0.01], flux,
            ),
        )

    insert_lightcurve(obj_var, night1, [1.0, 1.01, 0.99])
    insert_lightcurve(obj_var, night2, [1.02, 1.0, 0.98])
    insert_lightcurve(obj_var2, night1, [1.0, 1.05, 0.95])
    insert_lightcurve(obj_var2, night2, [0.9, 1.0, 1.1])

    cur.execute(
        "INSERT INTO relphot.mn_run (stem, labels, anchor, loaded_at) VALUES "
        "('test_mn', %s, '20250101', now()) RETURNING mn_run_id",
        (["20250101", "20250102"],),
    )
    (mn_run_id,) = cur.fetchone()
    cur.execute(
        "INSERT INTO relphot.tie (mn_run_id, obj_id, night_id, mag, mag_err) VALUES "
        "(%s, %s, %s, 14.0, 0.01), (%s, %s, %s, 14.05, 0.01)",
        (mn_run_id, obj_var, night1, mn_run_id, obj_var, night2),
    )

    cur.execute(
        "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
        "duration_h, tier) VALUES (%s, %s, 'transit', 12.0, 0.01, 2460310.55, 2.0, 1)",
        (obj_exop, night1),
    )
    cur.execute(
        "INSERT INTO relphot.detection (obj_id, night_id, kind, amplitude, period) "
        "VALUES (%s, %s, 'variable', 0.3, 1.2)",
        (obj_var, night1),
    )
    cur.execute(
        "INSERT INTO relphot.detection (obj_id, mn_run_id, kind, snr) "
        "VALUES (%s, %s, 'bls', 8.0)",
        (obj_var2, mn_run_id),
    )

    cur.execute(
        "INSERT INTO relphot.catalog_match (obj_id, catalog, name, period) "
        "VALUES (%s, 'NASA Exoplanet Archive', 'Test b', 3.5)",
        (obj_exop,),
    )
    cur.execute(
        "INSERT INTO relphot.catalog_match (obj_id, catalog, name, type, period) "
        "VALUES (%s, 'VSX', 'V* Test', 'EA', 1.2)",
        (obj_var,),
    )

    power = [0.1] * 100
    power[42] = 0.9
    cur.execute(
        "INSERT INTO relphot.periodogram (obj_id, scope, method, fmin, df, n, power, "
        "peak_period, peak_power, fap, computed_at) VALUES "
        "(%s, 'combined', 'LS', 0.1, 0.01, 100, %s, 1.2, 0.9, 0.0001, now())",
        (obj_var, power),
    )

    conn.commit()
    return {
        "night1": night1, "night2": night2,
        "obj_unc": obj_unc, "obj_exop": obj_exop, "obj_var": obj_var, "obj_var2": obj_var2,
    }


@pytest.fixture
def test_conn():
    dsn = _test_dsn()
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def client(ids, monkeypatch):
    owner_dsn = _test_dsn()
    ro_dsn = _role_dsn(owner_dsn, "relphot_ro", "RELPHOT_RO_PASSWORD")
    rw_dsn = _role_dsn(owner_dsn, "relphot_web", "RELPHOT_WEB_PASSWORD")
    monkeypatch.setenv("RELPHOT_WEB_RO_DSN", ro_dsn)
    monkeypatch.setenv("RELPHOT_WEB_RW_DSN", rw_dsn)

    from relphot.web.app import app

    with TestClient(app) as test_client:
        yield test_client, ids


# --------------------------------------------------------------------------
# /api/search
# --------------------------------------------------------------------------


def test_search_filters_class(client) -> None:
    test_client, ids = client
    resp = test_client.get("/api/search", params={"class": "EXOP"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["rows"][0]["obj_id"] == ids["obj_exop"]


def test_search_cone(client) -> None:
    test_client, ids = client
    resp = test_client.get("/api/search", params={"ra": 10.001, "dec": -20.001, "radius": 10})
    assert resp.status_code == 200
    data = resp.json()
    found = {row["obj_id"] for row in data["rows"]}
    assert ids["obj_exop"] in found
    assert ids["obj_var"] not in found


def test_search_period_range(client) -> None:
    test_client, ids = client
    resp = test_client.get("/api/search", params={"period_min": 1.0, "period_max": 2.0})
    assert resp.status_code == 200
    data = resp.json()
    found = {row["obj_id"] for row in data["rows"]}
    assert found == {ids["obj_var"]}


def test_search_sort_whitelist_rejects_injection(client) -> None:
    test_client, _ids = client
    resp = test_client.get("/api/search", params={"sort": "obj_id; DROP TABLE relphot.object; --"})
    assert resp.status_code == 400


def test_search_csv(client) -> None:
    test_client, _ids = client
    resp = test_client.get("/api/search.csv", params={"class": "UNC"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    assert "obj_id" in body.splitlines()[0]
    assert "RP UNC01" in body


def test_search_detection_kind_multinight_scope(client) -> None:
    test_client, ids = client
    resp = test_client.get(
        "/api/search", params={"detection_kind": "bls", "detection_scope": "multinight"}
    )
    assert resp.status_code == 200
    data = resp.json()
    found = {row["obj_id"] for row in data["rows"]}
    assert found == {ids["obj_var2"]}


def test_search_detection_kind_night_scope_excludes_multinight(client) -> None:
    test_client, _ids = client
    resp = test_client.get(
        "/api/search", params={"detection_kind": "bls", "detection_scope": "night"}
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_search_detection_kind_invalid_rejected(client) -> None:
    test_client, _ids = client
    resp = test_client.get("/api/search", params={"detection_kind": "bogus"})
    assert resp.status_code == 400


def test_search_detection_scope_invalid_rejected(client) -> None:
    test_client, _ids = client
    resp = test_client.get(
        "/api/search", params={"detection_kind": "bls", "detection_scope": "bogus"}
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# /api/sql
# --------------------------------------------------------------------------


def test_sql_rejects_write(client) -> None:
    test_client, _ids = client
    resp = test_client.post("/api/sql", json={"sql": "DELETE FROM relphot.object"})
    assert resp.status_code == 400
    check = test_client.get("/api/search", params={"limit": 1})
    assert check.json()["total"] >= 4


def test_sql_truncated_flag(client) -> None:
    test_client, _ids = client
    resp = test_client.post("/api/sql", json={"sql": "SELECT generate_series(1, 6000) AS n"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert len(data["rows"]) == 5000


# --------------------------------------------------------------------------
# /api/object/{obj_id}
# --------------------------------------------------------------------------


def test_object_detail(client) -> None:
    test_client, ids = client
    resp = test_client.get(f"/api/object/{ids['obj_exop']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"]["name"] == "RP EXOP01"
    assert len(data["catalog_matches"]) == 1
    assert len(data["detections"]) == 1
    assert data["detections"][0]["kind"] == "transit"
    assert data["detections"][0]["tier"] == 1
    assert len(data["nights"]) == 2


def test_object_not_found(client) -> None:
    test_client, _ids = client
    resp = test_client.get("/api/object/999999")
    assert resp.status_code == 404


def test_lc_with_file_names(client) -> None:
    test_client, ids = client
    resp = test_client.get(
        f"/api/object/{ids['obj_var']}/lc", params={"night_id": ids["night1"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == ["n1_frame000.fits", "n1_frame001.fits", "n1_frame002.fits"]
    assert len(data["flux"]) == 3


def test_lc_combined_tied_mode(client) -> None:
    test_client, ids = client
    resp = test_client.get(f"/api/object/{ids['obj_var']}/lc/combined")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "tied-mag"
    assert len(data["value"]) == 6
    assert all(13.0 < v < 15.0 for v in data["value"])


def test_lc_combined_night_normalised_mode(client) -> None:
    test_client, ids = client
    resp = test_client.get(f"/api/object/{ids['obj_var2']}/lc/combined")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "night-normalised"
    assert len(data["value"]) == 6


def test_periodogram(client) -> None:
    test_client, ids = client
    resp = test_client.get(
        f"/api/object/{ids['obj_var']}/periodogram", params={"scope": "combined", "method": "LS"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["power"]) == 100
    assert data["peak_period"] == pytest.approx(1.2)


def test_periodogram_not_found(client) -> None:
    test_client, ids = client
    resp = test_client.get(
        f"/api/object/{ids['obj_unc']}/periodogram", params={"scope": "combined", "method": "LS"}
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# PATCH /api/object/{obj_id}
# --------------------------------------------------------------------------


def test_patch_sets_class_source_manual(client) -> None:
    test_client, ids = client
    resp = test_client.patch(f"/api/object/{ids['obj_unc']}", json={"class": "VAR"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["class"] == "VAR"
    assert data["class_source"] == "manual"

    reset = test_client.patch(f"/api/object/{ids['obj_unc']}", json={"class_source": "auto"})
    assert reset.status_code == 200
    assert reset.json()["class_source"] == "auto"


def test_patch_invalid_class_rejected(client) -> None:
    test_client, ids = client
    resp = test_client.patch(f"/api/object/{ids['obj_unc']}", json={"class": "BOGUS"})
    assert resp.status_code == 400


def test_ro_dsn_cannot_write(client) -> None:
    _test_client, _ids = client
    ro_dsn = _role_dsn(_test_dsn(), "relphot_ro", "RELPHOT_RO_PASSWORD")
    with (
        psycopg.connect(ro_dsn, autocommit=True) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        cur.execute("UPDATE relphot.object SET notes = 'x' WHERE obj_id = 1")
