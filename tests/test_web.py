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


def test_patch_flags_are_independent_and_class_is_derived(client) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_unc']}"

    resp = test_client.patch(url, json={"is_var": True})
    assert resp.status_code == 200
    data = resp.json()
    assert (data["is_var"], data["var_source"]) == (True, "manual")
    assert data["is_exop"] is False
    assert data["exop_source"] != "manual"
    assert (data["class"], data["class_source"]) == ("VAR", "manual")

    data = test_client.patch(url, json={"is_exop": True}).json()
    assert (data["is_exop"], data["exop_source"]) == (True, "manual")
    assert data["is_var"] is True  # setting one flag never clears the other
    assert data["class"] == "EXOP+VAR"

    data = test_client.patch(url, json={"var_source": "auto"}).json()
    assert data["var_source"] == "auto"
    assert (data["class"], data["class_source"]) == ("EXOP+VAR", "manual")  # exop still manual

    data = test_client.patch(url, json={"exop_source": "auto"}).json()
    assert data["class_source"] == "auto"

    data = test_client.patch(url, json={"is_exop": False, "is_var": False}).json()
    assert (data["class"], data["is_exop"], data["is_var"]) == ("UNC", False, False)
    assert (data["exop_source"], data["var_source"]) == ("manual", "manual")


def test_patch_manual_flag_survives_refresh(client, test_conn) -> None:
    from relphot.db.refresh import refresh_objects

    test_client, ids = client
    obj_exop = ids["obj_exop"]
    resp = test_client.patch(f"/api/object/{obj_exop}", json={"is_exop": False})
    assert resp.json()["class"] == "UNC"

    # the known planet would make it a host again, but the manual flag is never touched
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, exop_source, class FROM relphot.object WHERE obj_id = %s",
            (obj_exop,),
        )
        assert cur.fetchone() == (False, "manual", "UNC")
    refresh_objects(test_conn, [obj_exop])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, exop_source, class, class_source FROM relphot.object "
            "WHERE obj_id = %s",
            (obj_exop,),
        )
        assert cur.fetchone() == (False, "manual", "UNC", "manual")

    resp = test_client.patch(f"/api/object/{obj_exop}", json={"exop_source": "auto"})
    refresh_objects(test_conn, [obj_exop])
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute("SELECT is_exop, class FROM relphot.object WHERE obj_id = %s", (obj_exop,))
        assert cur.fetchone() == (True, "EXOP")


def test_patch_legacy_class_field_sets_only_the_named_flags(client) -> None:
    test_client, ids = client
    data = test_client.patch(f"/api/object/{ids['obj_var']}", json={"class": "EXOP"}).json()
    assert (data["is_exop"], data["is_var"]) == (True, True)  # 'EXOP' does not clear VAR
    assert data["class"] == "EXOP+VAR"
    data = test_client.patch(f"/api/object/{ids['obj_var']}", json={"class": "UNC"}).json()
    assert (data["is_exop"], data["is_var"], data["class"]) == (False, False, "UNC")


def test_patch_flag_validation(client) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_unc']}"
    assert test_client.patch(url, json={"exop_source": "bogus"}).status_code == 400
    assert test_client.patch(url, json={"is_var": True, "var_source": "auto"}).status_code == 400
    assert test_client.patch(url, json={"class": "EXOP+BOGUS"}).status_code == 400
    assert test_client.patch(url, json={"is_exop": None}).status_code == 400
    assert test_client.patch("/api/object/999999", json={"is_var": True}).status_code == 404


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
