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
from importlib import resources

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
    cur.execute(
        "UPDATE relphot.object SET class = 'EXOP', is_exop = true, exop_source = 'auto', "
        "var_source = 'auto' WHERE obj_id = %s",
        (obj_exop,),
    )

    obj_var = insert_object(
        name="RP VAR01", ra=200.0, dec=40.0, class_source="auto", known=True,
        source_db="VSX", known_name="V* Test", known_type="EA", known_period=1.2,
        mean_mag=14.0, period=1.2, period_source="catalog", amplitude=0.3, n_nights=2,
    )
    cur.execute(
        "UPDATE relphot.object SET class = 'VAR', is_var = true, exop_source = 'auto', "
        "var_source = 'auto' WHERE obj_id = %s",
        (obj_var,),
    )

    obj_var2 = insert_object(
        name="RP VAR02", ra=201.0, dec=41.0, class_source="auto", known=False,
        mean_mag=16.0, n_nights=2,
    )
    cur.execute(
        "UPDATE relphot.object SET class = 'VAR', is_var = true, exop_source = 'auto', "
        "var_source = 'auto' WHERE obj_id = %s",
        (obj_var2,),
    )

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

    # a planet host that is also variable: two transit events of different depth (a
    # multi-planet-like case), each with a shape fit, one pairwise match, two period checks
    obj_both = insert_object(
        name="RP BOTH01", ra=120.0, dec=5.0, class_source="auto", known=True,
        source_db="VSX", known_name="V* Both", known_period=4.0, mean_mag=12.0, period=4.0,
        period_source="catalog", period_err=0.001, n_nights=2, duration_h=2.0,
        duration_lower_limit=True,
    )
    cur.execute(
        "UPDATE relphot.object SET class = 'EXOP+VAR', is_exop = true, is_var = true, "
        "exop_source = 'auto', var_source = 'auto' WHERE obj_id = %s",
        (obj_both,),
    )
    insert_star_night(obj_both, night1, 4)
    insert_star_night(obj_both, night2, 4)
    det_ids = []
    for night_id, depth in ((night1, 0.010), (night2, 0.020)):
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, flags) VALUES (%s, %s, 'transit', 9.0, %s, 2460310.52, 2.0, 1, "
            "'ON_VARIABLE') RETURNING det_id",
            (obj_both, night_id, depth),
        )
        det_ids.append(cur.fetchone()[0])
    det_a, det_b = det_ids
    for det_id, tc, depth in ((det_a, 2460310.52, 0.010), (det_b, 2460311.52, 0.020)):
        cur.execute(
            "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, tc_err, depth, depth_err, "
            "t14_h, t14_err, ingress_frac, ingress_err, chi2_red, n_points, input, converged, "
            "computed_at) VALUES (%s, %s, %s, 0.0005, %s, 0.001, 2.0, 0.1, 0.2, 0.05, 1.0, "
            "80, 'night', true, now())",
            (det_id, obj_both, tc, depth),
        )
    # the first event is incomplete (EDGE): its duration is only a lower limit
    cur.execute(
        "UPDATE relphot.detection SET duration_lower_limit = true, flags = 'EDGE|ON_VARIABLE' "
        "WHERE det_id = %s",
        (det_a,),
    )
    cur.execute(
        "UPDATE relphot.transit_shape SET t14_lower_limit = true, t14_err = NULL, "
        "ingress_frac = NULL, ingress_err = NULL, incomplete_reason = 'flag EDGE' "
        "WHERE det_id = %s",
        (det_a,),
    )
    cur.execute(
        "INSERT INTO relphot.transit_match (det_a, det_b, obj_id, dt_days, depth_z, t14_z, "
        "ingress_z, chi2, dof, p_match, same_telescope, commensurate_periods, computed_at) "
        "VALUES (%s, %s, %s, 1.0, -5.0, 0.0, 0.0, 25.0, 3, 0.6, true, %s, now())",
        (det_a, det_b, obj_both, [1.0, 0.5]),
    )
    estimates = ((1, "2025-01-01", 4.02, 0.02), (2, "2025-01-02", 4.001, 0.001))
    for n_nights, last_night, period, delta in estimates:
        cur.execute(
            "INSERT INTO relphot.period_estimate (obj_id, computed_at, method, input, night_ids, "
            "n_nights, last_night, baseline_days, period, period_err, power, fap, lit_period, "
            "lit_period_err, lit_catalog, harmonic, delta, delta_err, delta_z, verify_status) "
            "VALUES (%s, now(), 'LS', 'night', %s, %s, %s, 1.0, %s, 0.01, 0.5, 0.001, 4.0, "
            "0.001, 'VSX', 1.0, %s, 0.01, %s, 'verified')",
            (obj_both, list(range(n_nights)), n_nights, last_night, period, delta, delta / 0.01),
        )

    # a long-period literature variable that the nights' LS grid cannot verify
    cur.execute(
        "INSERT INTO relphot.period_estimate (obj_id, computed_at, method, input, night_ids, "
        "n_nights, last_night, baseline_days, period, lit_period, lit_catalog, verify_status, "
        "verify_note) VALUES (%s, now(), 'LS', 'night', %s, 2, '2025-01-02', 1.0, 0.7, 217.0, "
        "'VSX', 'lit_period_outside_grid', 'P_lit 217 d > LS max period 1.96 d')",
        (obj_var, [1, 2]),
    )

    conn.commit()
    return {
        "night1": night1, "night2": night2,
        "obj_unc": obj_unc, "obj_exop": obj_exop, "obj_var": obj_var, "obj_var2": obj_var2,
        "obj_both": obj_both, "det_a": det_a, "det_b": det_b,
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
    assert check.json()["total"] >= 5


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


# --------------------------------------------------------------------------
# classification: independent flags, transit events / matches, period verification
# --------------------------------------------------------------------------


def _ids_of(resp) -> set[int]:
    assert resp.status_code == 200, resp.text
    return {row["obj_id"] for row in resp.json()["rows"]}


def test_search_is_exop_is_var_and_combined_class_filters(client) -> None:
    test_client, ids = client
    assert _ids_of(test_client.get("/api/search", params={"is_exop": "true"})) == {
        ids["obj_exop"], ids["obj_both"]
    }
    assert _ids_of(test_client.get("/api/search", params={"is_var": "true"})) == {
        ids["obj_var"], ids["obj_var2"], ids["obj_both"]
    }
    assert _ids_of(
        test_client.get("/api/search", params={"is_exop": "true", "is_var": "true"})
    ) == {ids["obj_both"]}
    neither = test_client.get("/api/search", params={"is_exop": "false", "is_var": "false"})
    assert _ids_of(neither) == {ids["obj_unc"]}
    assert _ids_of(test_client.get("/api/search", params={"class": "EXOP+VAR"})) == {
        ids["obj_both"]
    }
    assert _ids_of(test_client.get("/api/search", params={"class": ["EXOP", "EXOP+VAR"]})) == {
        ids["obj_exop"], ids["obj_both"]
    }
    assert test_client.get("/api/search", params={"class": "BOGUS"}).status_code == 400


def test_search_min_p_match_filter(client) -> None:
    test_client, ids = client
    assert _ids_of(test_client.get("/api/search", params={"min_p_match": 0.5})) == {
        ids["obj_both"]
    }
    assert _ids_of(test_client.get("/api/search", params={"min_p_match": 0.9})) == set()


def test_search_result_columns_for_events_matches_and_period_verification(client) -> None:
    test_client, ids = client
    resp = test_client.get("/api/search", params={"name": "BOTH01"})
    (row,) = resp.json()["rows"]
    assert row["is_exop"] is True
    assert row["is_var"] is True
    assert row["class"] == "EXOP+VAR"
    assert row["n_transit_events"] == 2
    assert row["max_p_match"] == pytest.approx(0.6)
    assert row["period_err"] == pytest.approx(0.001)
    # the latest re-observation (last_night 2025-01-02, two nights)
    assert row["period_delta"] == pytest.approx(0.001)
    assert row["period_delta_err"] == pytest.approx(0.01)

    other = test_client.get("/api/search", params={"name": "EXOP01"}).json()["rows"][0]
    assert other["n_transit_events"] == 1
    assert other["max_p_match"] is None
    assert other["period_delta"] is None

    ordered = test_client.get(
        "/api/search", params={"sort": "n_transit_events", "order": "desc", "limit": 1}
    ).json()["rows"]
    assert ordered[0]["obj_id"] == ids["obj_both"]
    by_match = test_client.get(
        "/api/search", params={"sort": "max_p_match", "order": "desc", "limit": 1}
    ).json()["rows"]
    assert by_match[0]["obj_id"] == ids["obj_both"]


def test_search_csv_includes_new_columns(client) -> None:
    test_client, _ids = client
    resp = test_client.get("/api/search.csv", params={"is_exop": "true", "is_var": "true"})
    assert resp.status_code == 200
    lines = resp.text.splitlines()
    header = lines[0].split(",")
    for col in (
        "is_exop", "is_var", "period_err", "n_transit_events", "max_p_match", "period_delta",
        "period_delta_err",
    ):
        assert col in header
    assert len(lines) == 2
    assert "RP BOTH01" in lines[1]


def test_object_detail_has_transit_events_matches_and_period_estimates(client) -> None:
    test_client, ids = client
    data = test_client.get(f"/api/object/{ids['obj_both']}").json()

    assert data["object"]["is_exop"] is True
    assert data["object"]["is_var"] is True
    assert data["object"]["class"] == "EXOP+VAR"
    assert data["object"]["period_err"] == pytest.approx(0.001)

    events = data["transit_events"]
    assert [e["det_id"] for e in events] == [ids["det_a"], ids["det_b"]]
    assert events[0]["night_label"] == "20250101"
    assert events[0]["depth"] == pytest.approx(0.010)
    assert events[1]["depth"] == pytest.approx(0.020)
    assert events[0]["ingress_frac"] is None  # the incomplete event stores no ingress fraction
    assert events[1]["ingress_frac"] == pytest.approx(0.2)
    assert events[0]["status"] == "UNCONFIRMED"
    assert events[0]["flags"] == "EDGE|ON_VARIABLE"
    assert events[0]["converged"] is True

    (match,) = data["transit_matches"]
    assert (match["det_a"], match["det_b"]) == (ids["det_a"], ids["det_b"])
    assert (match["night_a"], match["night_b"]) == ("20250101", "20250102")
    assert match["p_match"] == pytest.approx(0.6)
    assert match["depth_z"] == pytest.approx(-5.0)
    assert match["commensurate_periods"] == [1.0, 0.5]

    estimates = data["period_estimates"]
    assert [e["n_nights"] for e in estimates] == [1, 2]
    assert estimates[1]["delta"] == pytest.approx(0.001)
    assert estimates[1]["harmonic"] == 1.0
    assert estimates[1]["lit_catalog"] == "VSX"

    plain = test_client.get(f"/api/object/{ids['obj_unc']}").json()
    assert plain["transit_events"] == []
    assert plain["transit_matches"] == []
    assert plain["period_estimates"] == []


def _review_rows(test_conn, obj_id: int) -> list[tuple]:
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT night_id, exop_verdict, var_verdict, note FROM relphot.user_night_review "
            "WHERE obj_id = %s ORDER BY night_id",
            (obj_id,),
        )
        return cur.fetchall()


def test_patch_flags_are_independent_and_class_is_derived(client, test_conn) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_unc']}"

    resp = test_client.patch(url, json={"is_var": True})
    assert resp.status_code == 200
    data = resp.json()
    assert (data["is_var"], data["var_source"]) == (True, "manual")
    assert data["is_exop"] is False
    assert data["exop_source"] == "auto"
    assert (data["class"], data["class_source"]) == ("VAR", "manual")
    # the shorthand is a verdict on the (one) night the object has data on
    assert _review_rows(test_conn, ids["obj_unc"]) == [(ids["night1"], None, "CONFIRMED", None)]

    data = test_client.patch(url, json={"is_exop": True}).json()
    assert (data["is_exop"], data["exop_source"]) == (True, "manual")
    assert data["is_var"] is True  # setting one flag never clears the other
    assert data["class"] == "EXOP+VAR"

    # resetting to auto drops the verdict: with no evidence the flag is off again
    data = test_client.patch(url, json={"var_source": "auto"}).json()
    assert (data["is_var"], data["var_source"]) == (False, "auto")
    assert (data["class"], data["class_source"]) == ("EXOP", "manual")  # exop verdict remains
    assert _review_rows(test_conn, ids["obj_unc"]) == [(ids["night1"], "CONFIRMED", None, None)]

    data = test_client.patch(url, json={"exop_source": "auto"}).json()
    assert (data["class"], data["class_source"]) == ("UNC", "auto")
    assert _review_rows(test_conn, ids["obj_unc"]) == []

    data = test_client.patch(url, json={"is_exop": False, "is_var": False}).json()
    assert (data["class"], data["is_exop"], data["is_var"]) == ("UNC", False, False)
    assert (data["exop_source"], data["var_source"]) == ("manual", "manual")


def test_patch_flag_shorthand_survives_refresh_but_never_beats_the_literature(
    client, test_conn
) -> None:
    from relphot.db.refresh import refresh_objects

    test_client, ids = client
    obj_unc, obj_exop = ids["obj_unc"], ids["obj_exop"]
    assert test_client.patch(f"/api/object/{obj_unc}", json={"is_var": True}).json()["is_var"]

    # a known planet stays a host whatever the person says about its nights
    data = test_client.patch(f"/api/object/{obj_exop}", json={"is_exop": False}).json()
    assert (data["is_exop"], data["exop_source"], data["class"]) == (True, "manual", "EXOP")
    assert [r[1] for r in _review_rows(test_conn, obj_exop)] == ["REJECTED", "REJECTED"]

    refresh_objects(test_conn, [obj_unc, obj_exop])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, is_var, is_exop, var_source, exop_source, class_source "
            "FROM relphot.object WHERE obj_id = ANY(%s) ORDER BY obj_id",
            ([obj_unc, obj_exop],),
        )
        rows = {r[0]: r[1:] for r in cur.fetchall()}
    assert rows[obj_unc] == (True, False, "manual", "auto", "manual")
    assert rows[obj_exop] == (False, True, "auto", "manual", "manual")

    data = test_client.patch(f"/api/object/{obj_exop}", json={"exop_source": "auto"}).json()
    assert (data["is_exop"], data["exop_source"], data["class_source"]) == (True, "auto", "auto")
    assert _review_rows(test_conn, obj_exop) == []


def test_patch_legacy_class_field_sets_only_the_named_flags(client) -> None:
    test_client, ids = client
    data = test_client.patch(f"/api/object/{ids['obj_var']}", json={"class": "EXOP"}).json()
    assert (data["is_exop"], data["is_var"]) == (True, True)  # 'EXOP' does not clear VAR
    assert data["class"] == "EXOP+VAR"
    # 'UNC' rejects both flags on the nights loaded now; the known variable still counts
    data = test_client.patch(f"/api/object/{ids['obj_var']}", json={"class": "UNC"}).json()
    assert (data["is_exop"], data["is_var"], data["class"]) == (False, True, "VAR")
    data = test_client.patch(f"/api/object/{ids['obj_var2']}", json={"class": "UNC"}).json()
    assert (data["is_exop"], data["is_var"], data["class"]) == (False, False, "UNC")


def test_patch_flag_validation(client) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_unc']}"
    assert test_client.patch(url, json={"exop_source": "bogus"}).status_code == 400
    assert test_client.patch(url, json={"is_var": True, "var_source": "auto"}).status_code == 400
    assert test_client.patch(url, json={"class": "EXOP+BOGUS"}).status_code == 400
    assert test_client.patch(url, json={"is_exop": None}).status_code == 400
    assert test_client.patch("/api/object/999999", json={"is_var": True}).status_code == 404


def test_patch_shorthand_writes_verdicts_only_for_the_nights_loaded_now(
    client, test_conn
) -> None:
    test_client, ids = client
    obj_unc, night1, night2 = ids["obj_unc"], ids["night1"], ids["night2"]
    url = f"/api/object/{obj_unc}"

    test_client.patch(url, json={"is_exop": True})
    assert _review_rows(test_conn, obj_unc) == [(night1, "CONFIRMED", None, None)]

    # the star is observed on a second night afterwards: the old verdict does not cover it
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night (obj_id, night_id, star_id) VALUES (%s, %s, 9)",
            (obj_unc, night2),
        )
    test_conn.commit()
    test_client.patch(url, json={"is_var": True})
    assert _review_rows(test_conn, obj_unc) == [
        (night1, "CONFIRMED", "CONFIRMED", None), (night2, None, "CONFIRMED", None),
    ]

    # a note keeps a row alive when its verdicts are reset to auto
    put = test_client.put(
        f"{url}/night/{night1}/review",
        json={"exop": "CONFIRMED", "var": "CONFIRMED", "note": "keep me"},
    )
    assert put.status_code == 200
    test_client.patch(url, json={"exop_source": "auto", "var_source": "auto"})
    assert _review_rows(test_conn, obj_unc) == [(night1, None, None, "keep me")]


def test_put_review_upserts_and_returns_the_new_flags(client, test_conn) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_unc']}/night/{ids['night1']}/review"

    resp = test_client.put(url, json={"exop": "CONFIRMED", "var": None, "note": "dip visible"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["review"]["exop"] == "CONFIRMED"
    assert data["review"]["var"] is None
    assert data["review"]["note"] == "dip visible"
    assert data["review"]["updated_at"]
    assert data["night"]["exop_effective"] is True
    assert (data["night"]["auto_exop"], data["night"]["pending"]) == (False, False)
    assert data["object"] == {
        "is_exop": True, "is_var": False, "class": "EXOP", "class_source": "manual",
        "exop_source": "manual", "var_source": "auto", "n_review_pending": 0,
        "n_nights_reviewed": 1,
    }
    assert _review_rows(test_conn, ids["obj_unc"]) == [
        (ids["night1"], "CONFIRMED", None, "dip visible")
    ]

    # a true PUT: the omitted note is cleared, the verdicts are replaced
    data = test_client.put(url, json={"exop": "REJECTED", "var": "CONFIRMED"}).json()
    assert data["review"]["note"] is None
    assert (data["object"]["is_exop"], data["object"]["is_var"]) == (False, True)
    assert data["object"]["class"] == "VAR"
    assert _review_rows(test_conn, ids["obj_unc"]) == [
        (ids["night1"], "REJECTED", "CONFIRMED", None)
    ]


def test_put_review_rejecting_a_night_never_removes_the_literature(client) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_exop']}/night/{ids['night1']}/review"

    data = test_client.put(url, json={"exop": "REJECTED"}).json()

    assert data["night"]["auto_exop"] is True
    assert data["night"]["exop_effective"] is False
    assert data["night"]["pending"] is False  # the person has decided this night
    assert (data["object"]["is_exop"], data["object"]["class"]) == (True, "EXOP")  # known planet
    assert data["object"]["exop_source"] == "manual"
    assert data["object"]["n_nights_reviewed"] == 1


def test_put_all_null_and_delete_remove_the_row(client, test_conn) -> None:
    test_client, ids = client
    obj_id, night1 = ids["obj_unc"], ids["night1"]
    url = f"/api/object/{obj_id}/night/{night1}/review"

    test_client.put(url, json={"exop": "CONFIRMED"})
    assert len(_review_rows(test_conn, obj_id)) == 1
    resp = test_client.put(url, json={"exop": None, "var": None, "note": ""})
    assert resp.status_code == 200
    assert resp.json()["review"] is None
    assert _review_rows(test_conn, obj_id) == []
    assert resp.json()["object"]["class"] == "UNC"
    assert resp.json()["object"]["exop_source"] == "auto"
    assert resp.json()["object"]["n_nights_reviewed"] == 0

    test_client.put(url, json={"var": "REJECTED", "note": "x"})
    assert len(_review_rows(test_conn, obj_id)) == 1
    resp = test_client.delete(url)
    assert resp.status_code == 200
    assert resp.json()["review"] is None
    assert _review_rows(test_conn, obj_id) == []
    assert test_client.delete(url).status_code == 200  # deleting nothing is fine


def test_put_review_validation_and_missing_targets(client, test_conn) -> None:
    test_client, ids = client
    obj_id, night1, night2 = ids["obj_unc"], ids["night1"], ids["night2"]
    url = f"/api/object/{obj_id}/night/{night1}/review"

    assert test_client.put(url, json={"exop": "MAYBE"}).status_code == 400
    assert test_client.put(url, json={"var": "confirmed"}).status_code == 400
    assert test_client.put(url, json={"note": "x" * 2001}).status_code == 400
    assert test_client.put(url, json={"note": "x" * 2000}).status_code == 200
    test_client.delete(url)
    assert _review_rows(test_conn, obj_id) == []

    missing = test_client.put(
        f"/api/object/999999/night/{night1}/review", json={"exop": "CONFIRMED"}
    )
    assert missing.status_code == 404
    # obj_unc has no data on night 2, and night 999999 does not exist
    no_data = test_client.put(
        f"/api/object/{obj_id}/night/{night2}/review", json={"exop": "CONFIRMED"}
    )
    assert no_data.status_code == 404
    assert "no data on this night" in no_data.json()["detail"]
    no_night = test_client.put(
        f"/api/object/{obj_id}/night/999999/review", json={"exop": "CONFIRMED"}
    )
    assert no_night.status_code == 404
    assert test_client.delete(f"/api/object/{obj_id}/night/{night2}/review").status_code == 404
    assert _review_rows(test_conn, obj_id) == []


def test_object_detail_nights_carry_the_review_and_night_state(client) -> None:
    test_client, ids = client
    data = test_client.get(f"/api/object/{ids['obj_exop']}").json()
    assert (data["object"]["lit_exop"], data["object"]["lit_var"]) == (True, False)
    first, second = data["nights"]  # ordered by night date
    for night in (first, second):
        for key in (
            "review_exop", "review_var", "review_note", "review_updated_at", "auto_exop",
            "auto_var", "exop_open", "var_open", "exop_effective", "var_effective", "pending",
        ):
            assert key in night
    assert first["night_id"] == ids["night1"]
    assert (first["auto_exop"], first["exop_open"], first["pending"]) == (True, True, True)
    assert (second["auto_exop"], second["pending"]) == (False, False)
    assert first["review_exop"] is None

    lit_var = test_client.get(f"/api/object/{ids['obj_var']}").json()["object"]
    assert (lit_var["lit_exop"], lit_var["lit_var"]) == (False, True)
    plain = test_client.get(f"/api/object/{ids['obj_both']}").json()["object"]
    assert (plain["lit_exop"], plain["lit_var"]) == (False, False)

    test_client.put(
        f"/api/object/{ids['obj_exop']}/night/{ids['night1']}/review",
        json={"exop": "REJECTED", "note": "systematic"},
    )
    first = test_client.get(f"/api/object/{ids['obj_exop']}").json()["nights"][0]
    assert (first["review_exop"], first["review_var"]) == ("REJECTED", None)
    assert first["review_note"] == "systematic"
    assert first["review_updated_at"]
    assert (first["auto_exop"], first["exop_effective"], first["pending"]) == (True, False, False)


def test_search_needs_review_and_user_reviewed_filters(client, test_conn) -> None:
    from relphot.objflags import refresh_flags

    test_client, ids = client
    # the raw-SQL fixture never ran the flags step: derive every object's state first
    refresh_flags(test_conn, class_multinight_kinds=("recurrent",))
    test_conn.commit()
    everything = {
        ids["obj_unc"], ids["obj_exop"], ids["obj_var"], ids["obj_var2"], ids["obj_both"]
    }
    awaiting = {ids["obj_exop"], ids["obj_var"], ids["obj_both"]}

    def search(**params) -> set[int]:
        return _ids_of(test_client.get("/api/search", params=params))

    assert search(needs_review="true") == awaiting
    assert search(needs_review="false") == everything - awaiting
    assert search(user_reviewed="true") == set()
    assert search(user_reviewed="false") == everything

    resp = test_client.put(
        f"/api/object/{ids['obj_exop']}/night/{ids['night1']}/review", json={"exop": "REJECTED"}
    )
    assert resp.status_code == 200
    assert search(needs_review="true") == awaiting - {ids["obj_exop"]}
    assert search(user_reviewed="true") == {ids["obj_exop"]}
    assert search(user_reviewed="false") == everything - {ids["obj_exop"]}

    rows = test_client.get("/api/search", params={"user_reviewed": "true"}).json()["rows"]
    assert (rows[0]["n_review_pending"], rows[0]["n_nights_reviewed"]) == (0, 1)
    csv_header = test_client.get("/api/search.csv").text.splitlines()[0].split(",")
    assert "n_review_pending" in csv_header
    assert "n_nights_reviewed" in csv_header


def test_patch_detection_rejected_refreshes_the_object_flags(client) -> None:
    test_client, ids = client
    obj = f"/api/object/{ids['obj_both']}"  # two transit events, no literature entry

    def state() -> tuple:
        o = test_client.get(obj).json()["object"]
        return o["is_exop"], o["class"], o["n_review_pending"], o["n_nights_reviewed"]

    resp = test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "REJECTED"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "REJECTED"
    assert state() == (True, "EXOP", 1, 1)  # the other event still stands

    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "REJECTED"})
    assert state() == (False, "UNC", 0, 2)

    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "UNCONFIRMED"})
    assert state() == (True, "EXOP", 1, 1)

    # a notes-only edit changes no evidence
    assert test_client.patch(
        f"/api/detection/{ids['det_b']}", json={"notes": "check"}
    ).status_code == 200
    assert state() == (True, "EXOP", 1, 1)


def test_web_role_can_write_reviews_but_not_move_them_and_ro_cannot_write(
    client, test_conn
) -> None:
    test_client, ids = client
    obj_id, night1 = ids["obj_unc"], ids["night1"]
    test_client.put(f"/api/object/{obj_id}/night/{night1}/review", json={"exop": "CONFIRMED"})

    rw_dsn = _role_dsn(_test_dsn(), "relphot_web", "RELPHOT_WEB_PASSWORD")
    with psycopg.connect(rw_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.user_night_review SET var_verdict = 'REJECTED', note = 'n' "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "UPDATE relphot.user_night_review SET night_id = %s WHERE obj_id = %s",
                (ids["night2"], obj_id),
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "UPDATE relphot.user_night_review SET obj_id = %s WHERE obj_id = %s",
                (ids["obj_var"], obj_id),
            )
    assert _review_rows(test_conn, obj_id) == [(night1, "CONFIRMED", "REJECTED", "n")]

    ro_dsn = _role_dsn(_test_dsn(), "relphot_ro", "RELPHOT_RO_PASSWORD")
    for sql in (
        "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
        "VALUES (%(o)s, %(n)s, 'CONFIRMED')",
        "UPDATE relphot.user_night_review SET note = 'x'",
        "DELETE FROM relphot.user_night_review",
    ):
        with (
            psycopg.connect(ro_dsn, autocommit=True) as conn,
            conn.cursor() as cur,
            pytest.raises(psycopg.errors.InsufficientPrivilege),
        ):
            cur.execute(sql, {"o": ids["obj_var"], "n": night1})


def test_patch_manual_period_clears_period_error(client) -> None:
    test_client, ids = client
    data = test_client.patch(f"/api/object/{ids['obj_both']}", json={"period": 5.0}).json()
    assert (data["period"], data["period_source"]) == (5.0, "manual")
    assert data["period_err"] is None
    assert data["period_n_nights"] is None


def test_patch_detection_status_and_notes(client) -> None:
    test_client, ids = client
    url = f"/api/detection/{ids['det_b']}"
    resp = test_client.patch(url, json={"status": "CONFIRMED", "notes": "second planet?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "CONFIRMED"
    assert resp.json()["notes"] == "second planet?"

    events = test_client.get(f"/api/object/{ids['obj_both']}").json()["transit_events"]
    assert {e["det_id"]: e["status"] for e in events} == {
        ids["det_a"]: "UNCONFIRMED", ids["det_b"]: "CONFIRMED"
    }
    # a person's verdict on one event does not touch the other event or the object status
    obj = test_client.get(f"/api/object/{ids['obj_both']}").json()["object"]
    assert obj["status"] == "UNCONFIRMED"

    assert test_client.patch(url, json={"status": "MERGED"}).status_code == 400
    assert test_client.patch(url, json={}).status_code == 400
    missing = test_client.patch("/api/detection/999999", json={"status": "REJECTED"})
    assert missing.status_code == 404


def test_web_role_cannot_edit_other_detection_columns(client) -> None:
    _test_client, ids = client
    rw_dsn = _role_dsn(_test_dsn(), "relphot_web", "RELPHOT_WEB_PASSWORD")
    with psycopg.connect(rw_dsn, autocommit=True) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("UPDATE relphot.detection SET snr = 1 WHERE det_id = %s", (ids["det_a"],))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("DELETE FROM relphot.transit_match")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("UPDATE relphot.period_estimate SET period = 1")
    ro_dsn = _role_dsn(_test_dsn(), "relphot_ro", "RELPHOT_RO_PASSWORD")
    with (
        psycopg.connect(ro_dsn, autocommit=True) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        cur.execute("UPDATE relphot.detection SET status = 'CONFIRMED'")


def test_sql_box_reads_new_tables(client) -> None:
    test_client, _ids = client
    resp = test_client.post(
        "/api/sql",
        json={
            "sql": "SELECT m.p_match, s.depth, e.delta FROM relphot.transit_match m "
            "JOIN relphot.transit_shape s ON s.det_id = m.det_a "
            "JOIN relphot.period_estimate e ON e.obj_id = m.obj_id "
            "WHERE e.n_nights = 2"
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["columns"] == ["p_match", "depth", "delta"]
    assert len(data["rows"]) == 1
    assert data["rows"][0][0] == pytest.approx(0.6)
    write = test_client.post("/api/sql", json={"sql": "DELETE FROM relphot.transit_match"})
    assert write.status_code == 400


def test_duration_lower_limit_is_rendered_in_search_csv_and_detail(client) -> None:
    test_client, ids = client

    (row,) = test_client.get("/api/search", params={"name": "BOTH01"}).json()["rows"]
    assert row["duration_h"] == pytest.approx(2.0)
    assert row["duration_lower_limit"] is True
    assert row["duration_display"] == "\u2265 2.00 h"
    (measured,) = test_client.get("/api/search", params={"name": "EXOP01"}).json()["rows"]
    assert measured["duration_display"] == "2.00 h"

    resp = test_client.get("/api/search.csv", params={"name": "BOTH01"})
    header, line = resp.text.splitlines()[:2]
    assert "duration_lower_limit" in header.split(",")
    assert "duration_display" in header.split(",")
    assert "\u2265 2.00 h" in line

    data = test_client.get(f"/api/object/{ids['obj_both']}").json()
    assert data["object"]["duration_lower_limit"] is True
    assert data["object"]["duration_display"] == "\u2265 2.00 h"
    events = data["transit_events"]
    assert events[0]["duration_lower_limit"] is True
    assert events[0]["duration_display"] == "\u2265 2.00 h"
    assert events[0]["incomplete_reason"] == "flag EDGE"
    assert events[1]["duration_lower_limit"] is False
    assert events[1]["duration_display"] == "2.00 h"
    (match,) = data["transit_matches"]
    assert match["t14_a_display"] == "\u2265 2.00 h"
    assert match["t14_b_display"] == "2.00 h"
    displays = {d["det_id"]: d["duration_display"] for d in data["detections"]}
    assert displays[ids["det_a"]] == "\u2265 2.00 h"
    assert displays[ids["det_b"]] == "2.00 h"


def test_period_verification_status_and_note_in_detail_search_and_csv(client) -> None:
    test_client, ids = client
    data = test_client.get(f"/api/object/{ids['obj_var']}").json()
    (est,) = data["period_estimates"]
    assert est["verify_status"] == "lit_period_outside_grid"
    assert est["verify_note"] == "P_lit 217 d > LS max period 1.96 d"
    assert est["delta"] is None

    both = test_client.get(f"/api/object/{ids['obj_both']}").json()["period_estimates"]
    assert [e["verify_status"] for e in both] == ["verified", "verified"]

    (row,) = test_client.get("/api/search", params={"name": "VAR01"}).json()["rows"]
    assert row["period_verify_status"] == "lit_period_outside_grid"
    assert row["period_verify_note"].startswith("P_lit 217 d")
    (row_both,) = test_client.get("/api/search", params={"name": "BOTH01"}).json()["rows"]
    assert row_both["period_verify_status"] == "verified"
    assert row_both["period_verify_note"] is None
    (unc,) = test_client.get("/api/search", params={"name": "UNC01"}).json()["rows"]
    assert unc["period_verify_status"] is None

    resp = test_client.get("/api/search.csv", params={"name": "VAR01"})
    header, line = resp.text.splitlines()[:2]
    assert "period_verify_status" in header.split(",")
    assert "period_verify_note" in header.split(",")
    assert "lit_period_outside_grid" in line


# --------------------------------------------------------------------------
# user-guided reprocessing: queue insert (relphot_web), history, adopt-as-period
# --------------------------------------------------------------------------


def _requests(test_conn) -> list[tuple]:
    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, kind, period_guess, tc_guess, width_guess_h, night_id, note, status "
            "FROM relphot.reprocess_request ORDER BY req_id"
        )
        return cur.fetchall()


def _var_entry(night=None, period=1.25, all_nights=False):
    """Helper to create a VAR entry dict."""
    return {
        "night_id": night,
        "exop": None,
        "var": {"period_guess": period, "all_nights": all_nights},
    }


def _exop_entry(night, tc, width):
    """Helper to create an EXOP entry dict."""
    return {
        "night_id": night,
        "exop": {"tc_guess": tc, "width_guess_h": width},
        "var": None,
    }


def test_post_reprocess_queues_a_variable_request_through_the_web_role(client, test_conn) -> None:
    test_client, ids = client
    # night-only
    entry = _var_entry(night=ids["night1"], period=1.25, all_nights=False)
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [entry], "note": "looks like an EB"},
    )
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["requests"]) == 1
    req = resp.json()["requests"][0]
    assert req["kind"] == "variable" and req["night_id"] == ids["night1"]
    rows = _requests(test_conn)
    expected = [
        (ids["obj_var"], "variable", 1.25, None, None, ids["night1"],
         "looks like an EB", "queued"),
    ]
    assert rows == expected

    # all_nights true -> night_id None
    test_conn.execute("DELETE FROM relphot.reprocess_request")
    test_conn.commit()
    entry_all = _var_entry(night=ids["night1"], period=1.25, all_nights=True)
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [entry_all], "note": "all"},
    )
    assert resp.status_code == 201, resp.text
    rows = _requests(test_conn)
    assert rows == [(ids["obj_var"], "variable", 1.25, None, None, None, "all", "queued")]


def test_post_reprocess_transit_uses_the_chosen_night(client, test_conn) -> None:
    test_client, ids = client
    tc = 2460310.5106  # inside both nights
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [_exop_entry(ids["night1"], tc, 2.0)]},
    )
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["requests"]) == 1
    rows = _requests(test_conn)
    assert len(rows) == 1
    assert rows[0][:6] == (ids["obj_var"], "transit", None, tc, 2.0, ids["night1"])


def test_post_reprocess_both_and_several_entries_queue_atomically(client, test_conn) -> None:
    test_client, ids = client
    tc = 2460310.5106
    # One entry EXOP+VAR -> 2 rows
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={
            "entries": [{
                "night_id": ids["night1"],
                "exop": {"tc_guess": tc, "width_guess_h": 2.0},
                "var": {"period_guess": 1.25, "all_nights": False},
            }],
        },
    )
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["requests"]) == 2
    rows = _requests(test_conn)
    assert len(rows) == 2
    kinds = {r[1] for r in rows}
    assert kinds == {"transit", "variable"}

    # Two entries -> 2 rows (not 4; each entry generates 1 request per kind combo)
    test_conn.execute("DELETE FROM relphot.reprocess_request")
    test_conn.commit()
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={
            "entries": [
                _exop_entry(ids["night1"], tc, 2.0),
                _exop_entry(ids["night2"], tc, 3.0),
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["requests"]) == 2
    rows = _requests(test_conn)
    assert len(rows) == 2


@pytest.mark.parametrize(
    "body_fn,expected_status,expected_substring",
    [
        # entry with neither exop nor var
        (
            lambda _: {"entries": [{"night_id": None, "exop": None, "var": None}]},
            400,
            "tick EXOP and/or VAR",
        ),
        # var without period
        (
            lambda ids: {"entries": [_var_entry(night=ids["night1"], period=None)]},
            400,
            "period_guess",
        ),
        # var period 0
        (
            lambda ids: {"entries": [_var_entry(night=ids["night1"], period=0)]},
            400,
            "period_guess",
        ),
        # var period -2
        (
            lambda ids: {"entries": [_var_entry(night=ids["night1"], period=-2)]},
            400,
            "period_guess",
        ),
        # var period 1e9
        (
            lambda ids: {"entries": [_var_entry(night=ids["night1"], period=1e9)]},
            400,
            "period_guess",
        ),
        # var night None, all_nights False
        (
            lambda _: {"entries": [_var_entry(night=None, period=1.25, all_nights=False)]},
            400,
            "needs a night",
        ),
        # var night 99999
        (
            lambda _: {
                "entries": [_var_entry(night=99999, period=1.25, all_nights=False)]
            },
            400,
            "night 99999 has no light curve",
        ),
        # exop with tc but width None
        (
            lambda _: {
                "entries": [
                    {
                        "night_id": None,
                        "exop": {"tc_guess": 2460310.51, "width_guess_h": None},
                        "var": None,
                    }
                ]
            },
            400,
            "tc_guess and width_guess_h",
        ),
        # exop width 0.05 (too small)
        (
            lambda ids: {"entries": [_exop_entry(ids["night1"], 2460310.51, 0.05)]},
            400,
            "between 0.1 and 12",
        ),
        # exop width 13 (too large)
        (
            lambda ids: {"entries": [_exop_entry(ids["night1"], 2460310.51, 13.0)]},
            400,
            "between 0.1 and 12",
        ),
        # exop valid tc/width but night None
        (
            lambda _: {"entries": [_exop_entry(None, 2460310.51, 2.0)]},
            400,
            "EXOP needs a night",
        ),
        # exop night1 tc 2450000 (outside night)
        (
            lambda ids: {"entries": [_exop_entry(ids["night1"], 2450000.0, 2.0)]},
            400,
            "not inside night",
        ),
        # exop night 99999
        (
            lambda _: {"entries": [_exop_entry(99999, 2460310.51, 2.0)]},
            400,
            "night 99999 has no light curve",
        ),
        # two var entries on night1
        (
            lambda ids: {
                "entries": [
                    _var_entry(ids["night1"], 1.25, False),
                    _var_entry(ids["night1"], 1.3, False),
                ]
            },
            400,
            "repeats",
        ),
        # two all-nights var entries
        (
            lambda _: {
                "entries": [
                    _var_entry(None, 1.25, True),
                    _var_entry(None, 1.3, True),
                ]
            },
            400,
            "repeats",
        ),
        # valid entry 1 + invalid entry 2
        (
            lambda ids: {
                "entries": [
                    _var_entry(ids["night1"], 1.25, False),
                    _exop_entry(99999, 2460310.51, 2.0),
                ]
            },
            400,
            "entry 2:",
        ),
    ],
    ids=[
        "neither_exop_nor_var",
        "var_period_none",
        "var_period_0",
        "var_period_negative",
        "var_period_toolarge",
        "var_night_none",
        "var_night_missing",
        "exop_width_none",
        "exop_width_too_small",
        "exop_width_too_large",
        "exop_night_none",
        "exop_tc_outside_night",
        "exop_night_missing",
        "two_var_same_night",
        "two_var_allnights",
        "valid_and_invalid",
    ],
)
def test_post_reprocess_validates_its_input(
    client, test_conn, body_fn, expected_status, expected_substring
) -> None:
    test_client, ids = client
    body = body_fn(ids)
    test_conn.execute("DELETE FROM relphot.reprocess_request")
    test_conn.commit()
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess", json=body
    )
    assert resp.status_code == expected_status, (
        f"Body {body}: expected {expected_status}, "
        f"got {resp.status_code} - {resp.text}"
    )
    if expected_status == 400:
        assert expected_substring in resp.json()["detail"]
    assert _requests(test_conn) == []


def test_post_reprocess_rejects_empty_list_too_many_and_unknown_object(
    client,
) -> None:
    test_client, ids = client
    n1, n2 = ids["night1"], ids["night2"]
    # Empty list
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess", json={"entries": []}
    )
    assert resp.status_code == 422
    # Too many (>20)
    entries = [
        _var_entry(n1 if i % 2 == 0 else n2, 1.25 + i * 0.01, False)
        for i in range(21)
    ]
    resp = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess", json={"entries": entries}
    )
    assert resp.status_code == 422
    # Unknown object
    resp = test_client.post(
        "/api/object/999999/reprocess", json={"entries": [_var_entry(n1)]}
    )
    assert resp.status_code == 404


def test_web_role_can_insert_requests_but_not_set_their_status(client) -> None:
    _test_client, ids = client
    rw_dsn = _role_dsn(_test_dsn(), "relphot_web", "RELPHOT_WEB_PASSWORD")
    with psycopg.connect(rw_dsn, autocommit=True) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess, status) "
                "VALUES (%s, 'variable', 1.0, 'done')",
                (ids["obj_var"],),
            )
        cur.execute(
            "INSERT INTO relphot.reprocess_request (obj_id, kind, period_guess) "
            "VALUES (%s, 'variable', 1.0) RETURNING req_id",
            (ids["obj_var"],),
        )
        (req_id,) = cur.fetchone()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "UPDATE relphot.reprocess_request SET status = 'done' WHERE req_id = %s", (req_id,)
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("DELETE FROM relphot.reprocess_request")


def test_get_reprocess_lists_the_history_and_queue_depth(client, test_conn) -> None:
    test_client, ids = client
    resp1 = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 1.1)]},
    )
    first = resp1.json()["requests"][0]["req_id"]

    test_client.post(
        f"/api/object/{ids['obj_var2']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 0.5)]},
    )

    resp2 = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 1.3)]},
    )
    second = resp2.json()["requests"][0]["req_id"]

    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'done', finished_at = now(), "
            "result = '{\"found\": true}' WHERE req_id = %s",
            (first,),
        )
    test_conn.commit()

    data = test_client.get(f"/api/object/{ids['obj_var']}/reprocess").json()
    assert [r["req_id"] for r in data["requests"]] == [second, first]
    by_id = {r["req_id"]: r for r in data["requests"]}
    assert by_id[first]["status"] == "done" and by_id[first]["result"] == {"found": True}
    assert by_id[first]["requests_ahead"] is None
    assert by_id[first]["night_label"] is not None  # should have night_label
    assert by_id[second]["status"] == "queued" and by_id[second]["requests_ahead"] == 1
    assert data["queue"] == {"queued": 2, "running": 0}
    assert test_client.get("/api/object/999999/reprocess").status_code == 404


def test_nothing_is_queued_by_edits_or_reads(client, test_conn) -> None:
    test_client, ids = client
    # GETs should not queue
    test_client.get(f"/api/object/{ids['obj_var']}")
    test_client.get(f"/api/object/{ids['obj_var']}/reprocess")
    # PATCH object should not queue
    test_client.patch(f"/api/object/{ids['obj_var']}", json={"is_exop": True})
    # PATCH detection should not queue
    with test_conn.cursor() as cur:
        cur.execute("SELECT det_id FROM relphot.detection LIMIT 1")
        det_id_row = cur.fetchone()
    if det_id_row:
        test_client.patch(f"/api/detection/{det_id_row[0]}", json={"status": "CONFIRMED"})
    # adopt_period should not queue
    with test_conn.cursor() as cur:
        cur.execute("SELECT est_id FROM relphot.period_estimate LIMIT 1")
        est_row = cur.fetchone()
    if est_row:
        test_client.post(f"/api/object/{ids['obj_var']}/adopt_period", json={"est_id": est_row[0]})

    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM relphot.reprocess_request")
        count = cur.fetchone()[0]
    assert count == 0


def _queue_three_requests(test_client, test_conn, ids) -> tuple[int, int, int]:
    """obj_var: one done (older) and one running; obj_var2: one queued. Returns their req_ids."""
    done = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 1.1)]},
    ).json()["requests"][0]["req_id"]
    running = test_client.post(
        f"/api/object/{ids['obj_var']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 1.3)]},
    ).json()["requests"][0]["req_id"]
    queued = test_client.post(
        f"/api/object/{ids['obj_var2']}/reprocess",
        json={"entries": [_var_entry(ids["night1"], 0.5)]},
    ).json()["requests"][0]["req_id"]
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'done', finished_at = now(), "
            "result = '{\"found\": true}' WHERE req_id = %s",
            (done,),
        )
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'running', started_at = now() "
            "WHERE req_id = %s",
            (running,),
        )
    test_conn.commit()
    return done, running, queued


def test_search_reports_pending_and_last_rerun(client, test_conn) -> None:
    test_client, ids = client
    _queue_three_requests(test_client, test_conn, ids)

    rows = test_client.get("/api/search", params={"limit": 100}).json()["rows"]
    by_id = {r["obj_id"]: r for r in rows}
    var = by_id[ids["obj_var"]]
    assert var["n_rerun_pending"] == 1  # the done one does not count
    assert var["last_rerun_status"] == "running"  # newest request of the object
    assert var["last_rerun_finished_at"] is None
    assert by_id[ids["obj_var2"]]["n_rerun_pending"] == 1
    assert by_id[ids["obj_var2"]]["last_rerun_status"] == "queued"
    for key in ("obj_unc", "obj_exop"):  # never re-run
        row = by_id[ids[key]]
        assert row["n_rerun_pending"] == 0
        assert row["last_rerun_status"] is None and row["last_rerun_finished_at"] is None

    # the newest request of obj_var finishes: nothing pending, the outcome is reported
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.reprocess_request SET status = 'failed', finished_at = now(), "
            "error = 'boom' WHERE obj_id = %s AND status = 'running'",
            (ids["obj_var"],),
        )
    test_conn.commit()
    rows = test_client.get("/api/search", params={"limit": 100}).json()["rows"]
    var = {r["obj_id"]: r for r in rows}[ids["obj_var"]]
    assert var["n_rerun_pending"] == 0
    assert var["last_rerun_status"] == "failed" and var["last_rerun_finished_at"] is not None


def test_search_rerun_pending_filter_sort_and_csv(client, test_conn) -> None:
    test_client, ids = client
    _queue_three_requests(test_client, test_conn, ids)

    pending = test_client.get("/api/search", params={"rerun_pending": "true"}).json()
    assert {r["obj_id"] for r in pending["rows"]} == {ids["obj_var"], ids["obj_var2"]}
    assert pending["total"] == 2
    idle = test_client.get("/api/search", params={"rerun_pending": "false"}).json()
    assert ids["obj_var"] not in {r["obj_id"] for r in idle["rows"]}
    assert idle["total"] + pending["total"] == test_client.get("/api/search").json()["total"]

    ordered = test_client.get(
        "/api/search", params={"sort": "n_rerun_pending", "order": "desc", "limit": 2}
    ).json()["rows"]
    assert {r["obj_id"] for r in ordered} == {ids["obj_var"], ids["obj_var2"]}

    csv_resp = test_client.get("/api/search.csv", params={"rerun_pending": "true"})
    assert csv_resp.status_code == 200
    header = csv_resp.text.splitlines()[0].split(",")
    for col in ("n_rerun_pending", "last_rerun_status", "last_rerun_finished_at"):
        assert col in header
    assert len(csv_resp.text.splitlines()) == 3


def test_list_reprocess_lists_pending_and_reports_watched_ones(client, test_conn) -> None:
    test_client, ids = client
    assert test_client.get("/api/reprocess").json() == {
        "requests": [], "watched": [], "queue": {"queued": 0, "running": 0},
    }
    done, running, queued = _queue_three_requests(test_client, test_conn, ids)

    data = test_client.get("/api/reprocess").json()
    assert [r["req_id"] for r in data["requests"]] == [queued, running]  # newest first
    assert data["watched"] == []
    assert data["queue"] == {"queued": 1, "running": 1}
    by_id = {r["req_id"]: r for r in data["requests"]}
    assert by_id[queued]["obj_id"] == ids["obj_var2"] and by_id[queued]["obj_name"] == "RP VAR02"
    assert by_id[queued]["status"] == "queued" and by_id[queued]["requests_ahead"] == 1
    assert by_id[running]["obj_id"] == ids["obj_var"] and by_id[running]["status"] == "running"
    assert by_id[running]["night_label"] == "20250101"

    # a request the page had seen pending comes back whatever its status
    data = test_client.get("/api/reprocess", params={"watch": [done, running, 999999]}).json()
    assert [r["req_id"] for r in data["watched"]] == [running, done]  # the unknown id is absent
    watched = {r["req_id"]: r for r in data["watched"]}
    assert watched[done]["status"] == "done" and watched[done]["finished_at"] is not None
    assert watched[done]["obj_name"] == "RP VAR01"
    assert [r["req_id"] for r in data["requests"]] == [queued, running]

    # other statuses can be listed too
    data = test_client.get("/api/reprocess", params={"status": "done,failed"}).json()
    assert [r["req_id"] for r in data["requests"]] == [done]
    assert data["queue"] == {"queued": 1, "running": 1}
    data = test_client.get("/api/reprocess", params={"limit": 1}).json()
    assert [r["req_id"] for r in data["requests"]] == [queued]


def test_list_reprocess_validates_its_input(client) -> None:
    test_client, _ids = client
    for status in ("bogus", "queued,bogus", ""):
        assert test_client.get("/api/reprocess", params={"status": status}).status_code == 400
    assert test_client.get("/api/reprocess", params={"limit": 0}).status_code == 422
    assert test_client.get("/api/reprocess", params={"watch": "x"}).status_code == 422
    assert test_client.get("/api/reprocess", params={"watch": list(range(101))}).status_code == 422
    assert test_client.get("/api/reprocess", params={"watch": list(range(100))}).status_code == 200


def test_rerun_status_endpoints_never_queue(client, test_conn) -> None:
    test_client, ids = client
    for params in ({}, {"rerun_pending": "true"}, {"rerun_pending": "false"},
                   {"sort": "last_rerun_status"}):
        assert test_client.get("/api/search", params=params).status_code == 200
    assert test_client.get("/api/search.csv", params={"rerun_pending": "true"}).status_code == 200
    assert test_client.get("/api/reprocess").status_code == 200
    assert test_client.get("/api/reprocess", params={"watch": [1, 2, 3]}).status_code == 200
    assert test_client.get(f"/api/object/{ids['obj_var']}/reprocess").status_code == 200
    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.reprocess_request")
        assert cur.fetchone()[0] == 0


def test_front_end_element_ids_exist() -> None:
    """Every id app.js looks up (``$("id")``) is in index.html (or its rerun template)."""
    static = resources.files("relphot.web") / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    html_ids = set(re.findall(r'\bid="([\w-]+)"', html))
    used = set(re.findall(r'\$\("([\w-]+)"\)', js))
    assert used - html_ids == set()
    for new_id in ("detail-obj-id", "rerun-badge", "rerun-global", "btn-rerun-list",
                   "rerun-list", "rerun-notices", "filter-rerun-pending", "rerun-heading"):
        assert new_id in html_ids


def _insert_guided_estimate(test_conn, obj_id: int, **kw) -> int:
    values = {
        "method": "LS-guided", "input": "tied", "night_ids": [1, 2], "n_nights": 2,
        "period": 1.2007, "period_err": 0.0015, "guess": 1.2, "verify_status": "no_literature",
    }
    values.update(kw)
    cols = ["obj_id", *values]
    with test_conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO relphot.period_estimate ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING est_id",
            [obj_id, *values.values()],
        )
        est_id = cur.fetchone()[0]
    test_conn.commit()
    return est_id


def _set_night_zp(test_conn, night_id: int, zp: float, source: str) -> None:
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.night SET zp = %s, zp_source = %s WHERE night_id = %s",
            (zp, source, night_id),
        )
    test_conn.commit()


def _mean_star_mag(test_conn, obj_id: int) -> tuple[float, float]:
    """(mean of star_night.mag over all nights, mag of the first night) of one object."""
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT avg(mag), (array_agg(mag ORDER BY night_id))[1] FROM relphot.star_night "
            "WHERE obj_id = %s",
            (obj_id,),
        )
        return cur.fetchone()


def test_apparent_mean_mag_assumes_a_20_mag_zero_point_by_default(client, test_conn) -> None:
    test_client, ids = client
    mean_mag, _first = _mean_star_mag(test_conn, ids["obj_var"])
    assert mean_mag is not None
    row = {r["obj_id"]: r for r in test_client.get("/api/search").json()["rows"]}[ids["obj_var"]]
    assert row["mean_mag_app"] == pytest.approx(mean_mag + 20.0, abs=1e-4)
    assert row["mag_zp_source"] == "assumed"
    assert row["mean_mag"] == 14.0  # the instrumental column is untouched
    detail = test_client.get(f"/api/object/{ids['obj_var']}").json()
    assert detail["object"]["mean_mag_app"] == pytest.approx(mean_mag + 20.0, abs=1e-4)
    assert detail["object"]["mag_zp_source"] == "assumed"
    assert detail["object"]["mean_mag"] == 14.0
    assert {(n["zp"], n["zp_source"]) for n in detail["nights"]} == {(20.0, "assumed")}
    assert all(n["mag_app"] == pytest.approx(n["mag"] + 20.0) for n in detail["nights"])
    assert "mean_mag_app" in test_client.get("/api/search.csv").text.splitlines()[0].split(",")


def test_apparent_mean_mag_uses_the_gaia_zero_point_of_calibrated_nights(client, test_conn) -> None:
    test_client, ids = client
    _set_night_zp(test_conn, ids["night1"], 24.5, "gaia")

    def row_of(key: str) -> dict:
        rows = test_client.get("/api/search").json()["rows"]
        return {r["obj_id"]: r for r in rows}[ids[key]]

    # obj_var has both nights: one calibrated, one assumed -> mixed, mean of both
    mean_mag, _ = _mean_star_mag(test_conn, ids["obj_var"])
    var = row_of("obj_var")
    assert var["mag_zp_source"] == "mixed"
    assert var["mean_mag_app"] == pytest.approx(mean_mag + 0.5 * (24.5 + 20.0), abs=1e-4)
    # obj_unc has night 1 only: purely Gaia
    unc_mag, _ = _mean_star_mag(test_conn, ids["obj_unc"])
    unc = row_of("obj_unc")
    assert unc["mag_zp_source"] == "gaia"
    assert unc["mean_mag_app"] == pytest.approx(unc_mag + 24.5, abs=1e-4)
    # sorting on the derived column works
    ordered = test_client.get("/api/search", params={"sort": "mean_mag_app"}).json()["rows"]
    values = [r["mean_mag_app"] for r in ordered if r["mean_mag_app"] is not None]
    assert values == sorted(values)


def test_lc_payloads_carry_the_zero_point_for_magnitudes(client, test_conn) -> None:
    test_client, ids = client
    _set_night_zp(test_conn, ids["night1"], 24.5, "gaia")
    _mean, first_mag = _mean_star_mag(test_conn, ids["obj_var"])

    lc = test_client.get(
        f"/api/object/{ids['obj_var']}/lc", params={"night_id": ids["night1"]}
    ).json()
    assert lc["zp"] == 24.5 and lc["zp_source"] == "gaia"
    assert lc["app_mag"] == pytest.approx(first_mag + 24.5, abs=1e-3)
    lc2 = test_client.get(
        f"/api/object/{ids['obj_var']}/lc", params={"night_id": ids["night2"]}
    ).json()
    assert lc2["zp"] == 20.0 and lc2["zp_source"] == "assumed"

    # tied: the anchor night (20250101, calibrated) sets the zero point of the tied magnitudes
    tied = test_client.get(f"/api/object/{ids['obj_var']}/lc/combined").json()
    assert tied["mode"] == "tied-mag"
    assert tied["zp"] == 24.5 and tied["zp_source"] == "gaia"
    assert len(tied["app_mag"]) == len(tied["value"])
    phase = test_client.get(f"/api/object/{ids['obj_var']}/phase").json()
    assert phase["zp"] == 24.5 and phase["zp_source"] == "gaia"

    # untied: every point carries its night's apparent mean magnitude, sources summed up
    untied = test_client.get(f"/api/object/{ids['obj_var2']}/lc/combined").json()
    assert untied["mode"] == "night-normalised"
    assert untied["zp"] is None and untied["zp_source"] == "mixed"
    by_night = {}
    for nid, app in zip(untied["night_id"], untied["app_mag"], strict=True):
        by_night.setdefault(nid, set()).add(app)
    assert {n: len(v) for n, v in by_night.items()} == {ids["night1"]: 1, ids["night2"]: 1}
    assert all(v is not None for s in by_night.values() for v in s)
    phase = test_client.get(f"/api/object/{ids['obj_var2']}/phase").json()
    assert phase["zp"] is None and phase["zp_source"] == "mixed"
    assert phase["app_mag"] == untied["app_mag"]  # the untied phase diagram can draw magnitudes


def test_front_end_names_the_apparent_magnitude_and_its_zero_point() -> None:
    static = resources.files("relphot.web") / "static"
    js = (static / "app.js").read_text()
    html = (static / "index.html").read_text()
    for name in ("mean_mag_app", "mag_zp_source", "zpText", "lc-unit-select"):
        assert name in js or name in html


def test_adopt_period_sets_a_manual_period_with_its_error(client, test_conn) -> None:
    test_client, ids = client
    est_id = _insert_guided_estimate(test_conn, ids["obj_var2"])
    resp = test_client.post(
        f"/api/object/{ids['obj_var2']}/adopt_period", json={"est_id": est_id}
    )
    assert resp.status_code == 200, resp.text
    obj = resp.json()
    assert obj["period"] == 1.2007 and obj["period_err"] == 0.0015
    assert obj["period_source"] == "manual" and obj["period_n_nights"] == 2

    # a manual period survives the pipeline's refresh ...
    from relphot.db.refresh import refresh_objects

    refresh_objects(test_conn, [ids["obj_var2"]])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period, period_source FROM relphot.object WHERE obj_id = %s", (ids["obj_var2"],)
        )
        assert cur.fetchone() == (1.2007, "manual")
    # ... and "reset to auto" hands it back
    resp = test_client.patch(
        f"/api/object/{ids['obj_var2']}", json={"period_source": "auto"}
    )
    assert resp.status_code == 200 and resp.json()["period_source"] == "auto"


def test_adopt_period_refuses_unusable_or_foreign_estimates(client, test_conn) -> None:
    test_client, ids = client
    flagged = _insert_guided_estimate(
        test_conn, ids["obj_var2"], night_ids=[7], verify_status="long_period_needs_tie"
    )
    empty = _insert_guided_estimate(test_conn, ids["obj_var2"], night_ids=[8], period=None)
    foreign = _insert_guided_estimate(test_conn, ids["obj_var"], night_ids=[9])
    url = f"/api/object/{ids['obj_var2']}/adopt_period"
    resp = test_client.post(url, json={"est_id": flagged})
    assert resp.status_code == 400
    assert "long period needs a multi-night tie" in resp.json()["detail"]
    assert test_client.post(url, json={"est_id": empty}).status_code == 400
    assert test_client.post(url, json={"est_id": foreign}).status_code == 404
    assert test_client.post(url, json={"est_id": 424242}).status_code == 404
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT period_source FROM relphot.object WHERE obj_id = %s", (ids["obj_var2"],)
        )
        assert cur.fetchone() != ("manual",)


# --------------------------------------------------------------------------
# phase diagram payload
# --------------------------------------------------------------------------


def test_phase_payload_is_tied_for_a_tied_object_and_labelled_untied_otherwise(client) -> None:
    test_client, ids = client
    tied = test_client.get(f"/api/object/{ids['obj_var']}/phase").json()
    assert tied["tied"] is True and tied["mode"] == "tied-mag"
    assert "tie-calibrated" in tied["label"]
    assert len(tied["bjd_tdb"]) == len(tied["value"]) == len(tied["value_err"]) == 6
    assert all(13.0 < v < 15.0 for v in tied["value"])
    # tied error bars include the night's 0.01 mag tie error in quadrature
    assert all(e > 0.0107 for e in tied["value_err"])
    assert sorted(set(tied["night_label"])) == ["20250101", "20250102"]
    keys = [c["key"] for c in tied["period_candidates"]]
    # PERIOD 1.2 (catalog), the fixture's latest LS estimate (0.7 d) and the literature period
    assert keys == ["period", "estimate", "literature"]
    assert tied["period"] == 1.2 and tied["t_first"] == min(tied["bjd_tdb"])
    assert 0.0 < tied["phase_coverage"] <= 1.0 and tied["n_cycles"] > 0

    untied = test_client.get(f"/api/object/{ids['obj_var2']}/phase").json()
    assert untied["tied"] is False and untied["mode"] == "night-normalised"
    assert "untied" in untied["label"]
    assert all(0.8 < v < 1.2 for v in untied["value"])
    assert untied["period_candidates"] == [] and untied["period"] is None
    assert untied["model"] is None and untied["phase_coverage"] is None


def test_phase_payload_period_model_candidates_and_aliases(client, test_conn) -> None:
    test_client, ids = client
    obj = ids["obj_var2"]
    _insert_guided_estimate(test_conn, obj, night_ids=[1, 2], period=1.5, guess=1.4)
    _insert_guided_estimate(
        test_conn, obj, method="LS", night_ids=[3, 4], n_nights=2, period=1.6, period_err=0.02,
        alias_periods=[0.61, 2.7], alias_powers=[0.4, 0.2],
    )
    data = test_client.get(f"/api/object/{obj}/phase").json()
    assert [c["key"] for c in data["period_candidates"]] == ["estimate", "guided"]
    assert data["period"] == 1.6  # the first candidate
    assert data["aliases"] == [{"period": 0.61, "power": 0.4}, {"period": 2.7, "power": 0.2}]

    # an explicit period: the model and the coverage are computed there
    # (6 points: a period beyond a night is fitted with one offset, so the fit has 1 dof)
    tied = test_client.get(f"/api/object/{ids['obj_var']}/phase", params={"period": 2.0}).json()
    assert tied["period"] == 2.0
    assert tied["model"] is not None and len(tied["model"]["coef"]) == 4
    assert tied["model"]["period"] == 2.0 and tied["model"]["t_zero"] is not None
    assert test_client.get(f"/api/object/{obj}/phase", params={"period": 0}).status_code == 422
    assert test_client.get("/api/object/999999/phase").status_code == 404


def test_object_detail_carries_the_guided_and_inflation_fields(client, test_conn) -> None:
    test_client, ids = client
    est_id = _insert_guided_estimate(
        test_conn, ids["obj_var"], phase_coverage=0.4, n_cycles=1.3,
        alias_periods=[0.7], alias_powers=[0.3],
    )
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.star_night SET err_scale = 3.5, blended = true "
            "WHERE obj_id = %s AND night_id = %s",
            (ids["obj_var"], ids["night1"]),
        )
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, tc_bjd_tdb, duration_h, "
            "flags, origin) VALUES (%s, %s, 'transit', 2460310.51, 2.0, 'USER', 'user')",
            (ids["obj_var"], ids["night1"]),
        )
    test_conn.commit()
    data = test_client.get(f"/api/object/{ids['obj_var']}").json()
    est = next(e for e in data["period_estimates"] if e["est_id"] == est_id)
    assert (est["method"], est["guess"], est["phase_coverage"], est["n_cycles"]) == (
        "LS-guided", 1.2, pytest.approx(0.4), pytest.approx(1.3)
    )
    assert est["alias_periods"] == [0.7]
    night1 = next(n for n in data["nights"] if n["night_id"] == ids["night1"])
    assert (night1["err_scale"], night1["blended"]) == (pytest.approx(3.5), True)
    assert {d["origin"] for d in data["detections"] if d["kind"] == "transit"} == {"user"}
    assert [e["origin"] for e in data["transit_events"]] == ["user"]


def test_search_period_verification_ignores_guided_rows(client, test_conn) -> None:
    test_client, ids = client
    _insert_guided_estimate(
        test_conn, ids["obj_both"], night_ids=[5, 6], lit_period=4.0, period=3.0,
        delta=-1.0, delta_err=0.1, last_night="2030-01-01",
    )
    rows = test_client.get("/api/search", params={"name": "BOTH01"}).json()["rows"]
    # the fixture's latest re-observation has delta 0.001; the later guided row (-1.0) is not one
    assert rows[0]["period_delta"] == pytest.approx(0.001)
