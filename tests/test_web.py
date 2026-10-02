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
from datetime import date
from importlib import resources

import numpy as np
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


def _frame_window(test_conn, night_id: int) -> tuple[float, float]:
    """``(first, last)`` frame BJD_TDB of a night straight from the table (its window)."""
    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT min(bjd_tdb), max(bjd_tdb) FROM relphot.frame WHERE night_id = %s",
            (night_id,),
        )
        return cur.fetchone()


def test_lc_carries_the_nights_frame_window(client, test_conn) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_var']}/lc"
    window = _frame_window(test_conn, ids["night1"])
    assert window == pytest.approx((2460310.5006, 2460310.5206))

    data = test_client.get(url, params={"night_id": ids["night1"]}).json()
    assert (data["t_first"], data["t_last"]) == pytest.approx(window)
    own = (min(data["bjd_tdb"]), max(data["bjd_tdb"]))  # a complete star spans the same window
    assert (data["t_first"], data["t_last"]) == pytest.approx(own)

    # a star that lacks the night's first and last epochs still gets the whole night's window
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.lightcurve SET frame_index = '{1}', bjd_tdb = '{2460310.5106}', "
            "flux = '{1.0}', flux_err = '{0.01}', flux_raw = '{1.0}' "
            "WHERE obj_id = %s AND night_id = %s",
            (ids["obj_var"], ids["night1"]),
        )
    test_conn.commit()
    sparse = test_client.get(url, params={"night_id": ids["night1"]}).json()
    assert sparse["bjd_tdb"] == pytest.approx([2460310.5106])
    assert (sparse["t_first"], sparse["t_last"]) == pytest.approx(window)

    # the window is that of ALL the night's frames: dropping the first (and the last) one, or all
    # of them, does not move it
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.frame SET kept = false WHERE night_id = %s AND frame_index = 0",
            (ids["night1"],),
        )
    test_conn.commit()
    dropped = test_client.get(url, params={"night_id": ids["night1"]}).json()
    assert (dropped["t_first"], dropped["t_last"]) == pytest.approx(window)
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.frame SET kept = false WHERE night_id = %s", (ids["night1"],))
    test_conn.commit()
    none_kept = test_client.get(url, params={"night_id": ids["night1"]}).json()
    assert (none_kept["t_first"], none_kept["t_last"]) == pytest.approx(window)

    # the other night is not affected
    other = test_client.get(url, params={"night_id": ids["night2"]}).json()
    assert (other["t_first"], other["t_last"]) == pytest.approx(
        _frame_window(test_conn, ids["night2"])
    )


def test_lc_window_is_null_for_a_night_without_frame_times(client, test_conn) -> None:
    test_client, ids = client
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.frame SET bjd_tdb = 'NaN' WHERE night_id = %s AND frame_index = 0",
            (ids["night1"],),
        )
        cur.execute(
            "UPDATE relphot.frame SET bjd_tdb = NULL WHERE night_id = %s AND frame_index = 2",
            (ids["night1"],),
        )
    test_conn.commit()
    url = f"/api/object/{ids['obj_var']}/lc"
    one = test_client.get(url, params={"night_id": ids["night1"]}).json()
    # the NaN and the NULL time are skipped
    assert (one["t_first"], one["t_last"]) == pytest.approx((2460310.5106, 2460310.5106))
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.frame SET bjd_tdb = NULL WHERE night_id = %s", (ids["night1"],)
        )
    test_conn.commit()
    none = test_client.get(url, params={"night_id": ids["night1"]}).json()
    assert (none["t_first"], none["t_last"]) == (None, None)


def _add_members(test_conn, ids: dict, *, target_is_member: bool, median: bool) -> np.ndarray:
    """Tile 0 / aperture 1 of night 1: ``tile_lc`` + comparison members around ``obj_var`` (star 2).

    ``ens_flux`` is the per-frame median of the members' normalised fluxes (also with the target
    among them). Returns the ``(n_members, 3)`` normalised fluxes in ``star_id`` order.
    """
    star_ids = [10, 11, 12, 13, 14] + ([2] if target_is_member else [])
    c = np.array(
        [[1.00, 1.02, 0.98], [1.01, 0.99, 1.00], [0.99, 1.00, 1.02],
         [1.02, 1.01, 0.99], [0.98, 0.98, 1.01], [1.00, 1.00, 1.00]][: len(star_ids)]
    )
    ens = np.median(c, axis=0)
    n = len(star_ids)
    unequal = np.linspace(1.0, 2.0, n)
    weights = [1.0 / n] * n if median else list(unequal / unequal.sum())
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile, best_apertures) VALUES (%s, 0, '{1}')",
            (ids["night1"],),
        )
        cur.execute(
            "INSERT INTO relphot.tile_lc (night_id, tile, aperture, ref_flux, ref_flux_err, "
            "ens_flux, ens_flux_err, n_ensemble, n_comp) VALUES (%s, 0, 1, %s, %s, %s, %s, %s, %s)",
            (ids["night1"], [1.0] * 3, [0.01] * 3, ens.tolist(), [0.001] * 3, n, n),
        )
        for k, (sid, w) in enumerate(zip(star_ids, weights, strict=True)):
            cur.execute(
                "INSERT INTO relphot.comparison_member (night_id, tile, aperture, star_id, mag, "
                "weight, norm_flux) VALUES (%s, 0, 1, %s, %s, %s, %s)",
                (ids["night1"], sid, 14.0 + 0.1 * k, w, c[k].tolist()),
            )
    test_conn.commit()
    return c


def test_lc_ratios_median_night_reproduces_the_plotted_curve(client, test_conn) -> None:
    test_client, ids = client
    _add_members(test_conn, ids, target_is_member=False, median=True)
    resp = test_client.get(f"/api/object/{ids['obj_var']}/night/{ids['night1']}/lc_ratios")
    assert resp.status_code == 200, resp.text
    p = resp.json()
    assert p["ensemble"] == "median" and p["aperture"] == 1 and p["tile"] == 0
    assert (p["n_members"], p["n_shown"], p["target_is_member"]) == (5, 5, False)
    assert p["frame_index"] == [0, 1, 2]
    ratios = np.array([[np.nan if v is None else v for v in m["ratio"]] for m in p["members"]])
    lc = np.array([1.0, 1.01, 0.99])  # the fixture's obj_var / night 1 flux
    np.testing.assert_allclose(np.median(ratios, axis=0), lc, atol=1e-4)


def test_lc_ratios_leave_the_target_out_and_flag_a_weighted_night(client, test_conn) -> None:
    test_client, ids = client
    _add_members(test_conn, ids, target_is_member=True, median=False)
    p = test_client.get(f"/api/object/{ids['obj_var']}/night/{ids['night1']}/lc_ratios").json()
    assert p["target_is_member"] is True
    assert 2 not in [m["star_id"] for m in p["members"]]
    assert p["n_members"] == 5
    assert p["ensemble"] == "weighted"


def test_lc_ratios_limit_and_missing_members(client, test_conn) -> None:
    test_client, ids = client
    url = f"/api/object/{ids['obj_var']}/night/{ids['night1']}/lc_ratios"
    assert test_client.get(url).status_code == 404  # night without members
    _add_members(test_conn, ids, target_is_member=False, median=True)
    p = test_client.get(url, params={"limit": 2}).json()
    assert (p["n_members"], p["n_shown"]) == (5, 2)
    assert test_client.get(url, params={"limit": 0}).status_code == 422


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


def _add_lookalikes(test_conn, ids: dict, n: int = 25) -> list[int]:
    """``n`` other objects' transit events on night 1 and the automatic verdict on ``det_a``."""
    det_ids = []
    with test_conn.cursor() as cur:
        for k in range(n):
            cur.execute(
                "INSERT INTO relphot.object (name, ra, dec) VALUES (%s, 1.0, 1.0) "
                "RETURNING obj_id",
                (f"RP SIM{k:02d}",),
            )
            (obj_id,) = cur.fetchone()
            cur.execute(
                "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
                "duration_h, tier) VALUES (%s, %s, 'transit', 9.0, 0.011, %s, 2.1, 1) "
                "RETURNING det_id",
                (obj_id, ids["night1"], 2460310.52 + 0.0001 * k),
            )
            (det_id,) = cur.fetchone()
            cur.execute(
                "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, tc_err, depth, "
                "depth_err, t14_h, t14_err, converged, computed_at) "
                "VALUES (%s, %s, %s, 0.0005, 0.011, 0.001, 2.1, 0.1, true, now())",
                (det_id, obj_id, 2460310.52 + 0.0001 * k),
            )
            det_ids.append(det_id)
        cur.execute(
            "UPDATE relphot.detection SET auto_status = 'REJECTED', auto_reason = %s "
            "WHERE det_id = %s",
            (f"too many similar events: {n} other events on this night", ids["det_a"]),
        )
        cur.execute(
            "INSERT INTO relphot.transit_coincidence (det_id, night_id, n_similar, n_expected, "
            "p_chance, similar_det_ids, rejected) VALUES (%s, %s, %s, 3.5, 1e-9, %s, true)",
            (ids["det_a"], ids["night1"], n, det_ids),
        )
    test_conn.commit()
    return det_ids


def test_object_detail_carries_the_automatic_rejection_and_the_similar_events(
    client, test_conn
) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids)
    data = test_client.get(f"/api/object/{ids['obj_both']}").json()

    ev_a, ev_b = data["transit_events"]
    assert ev_a["det_id"] == ids["det_a"]
    assert ev_a["status"] == "UNCONFIRMED"  # the person's verdict is untouched
    assert ev_a["auto_status"] == "REJECTED"
    assert ev_a["auto_reason"].startswith("too many similar events: 25 other events")
    assert ev_a["effective_status"] == "REJECTED (auto)"
    assert (ev_a["n_similar"], ev_a["n_expected"]) == (25, pytest.approx(3.5))
    assert ev_a["p_chance"] == pytest.approx(1e-9)
    assert "similar_det_ids" not in ev_a
    # the list is capped (20 of 25), nearest first, each with the object to link to
    listed = ev_a["similar_events"]
    assert [e["det_id"] for e in listed] == similar[:20]
    first = listed[0]
    assert first["obj_name"] == "RP SIM00"
    assert first["obj_id"] > 0
    assert first["tc"] == pytest.approx(2460310.52)
    assert first["depth"] == pytest.approx(0.011)
    assert first["t14_h"] == pytest.approx(2.1)
    assert first["duration_display"] == "2.10 h"
    # an event with no verdict of the check reads as before
    assert ev_b["auto_status"] is None
    assert ev_b["effective_status"] == "UNCONFIRMED"
    assert (ev_b["n_similar"], ev_b["n_expected"], ev_b["p_chance"]) == (None, None, None)
    assert ev_b["similar_events"] == []
    # the detections table carries it too
    det = next(d for d in data["detections"] if d["det_id"] == ids["det_a"])
    assert (det["auto_status"], det["effective_status"]) == ("REJECTED", "REJECTED (auto)")
    assert det["auto_reason"] == ev_a["auto_reason"]
    # night 1 has no automatic exoplanet evidence left and awaits nothing
    night1 = next(n for n in data["nights"] if n["night_id"] == ids["night1"])
    assert (night1["auto_exop"], night1["exop_open"], night1["pending"]) == (False, False, False)

    # the per-night review endpoint reads the same state
    resp = test_client.put(
        f"/api/object/{ids['obj_both']}/night/{ids['night1']}/review", json={"note": "look"}
    )
    assert resp.json()["night"]["auto_exop"] is False

    # a person's CONFIRMED overrides the automatic rejection; a REJECTED is theirs
    def event_a() -> dict:
        events = test_client.get(f"/api/object/{ids['obj_both']}").json()["transit_events"]
        return events[0]

    test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "CONFIRMED"})
    ev = event_a()
    assert (ev["status"], ev["auto_status"], ev["effective_status"]) == (
        "CONFIRMED", "REJECTED", "CONFIRMED"
    )
    obj = test_client.get(f"/api/object/{ids['obj_both']}").json()
    assert obj["object"]["is_exop"] is True
    assert next(n for n in obj["nights"] if n["night_id"] == ids["night1"])["auto_exop"] is True
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "REJECTED"})
    assert event_a()["effective_status"] == "REJECTED"
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "UNCONFIRMED"})
    assert event_a()["effective_status"] == "REJECTED (auto)"


# --------------------------------------------------------------------------
# similar events: GET /api/detection/{det_id}/similar, POST /api/detections/review
# --------------------------------------------------------------------------


def _add_stack_lcs(test_conn, night_id: int, det_ids: list[int]) -> None:
    """A night light curve (flux 2.0 / 2.2 / 1.8, error 0.01) for the object of each event."""
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id FROM relphot.detection WHERE det_id = ANY(%s) ORDER BY det_id",
            (det_ids,),
        )
        for k, (obj_id,) in enumerate(cur.fetchall()):
            cur.execute(
                "INSERT INTO relphot.star_night (obj_id, night_id, star_id, tile, mag, "
                "best_aperture, rms, expected_noise, chi2_reduced, n_epochs, is_comparison) "
                "VALUES (%s, %s, %s, 0, 14.0, 1, 0.01, 0.01, 1.0, 3, false) "
                "ON CONFLICT (obj_id, night_id) DO NOTHING",
                (obj_id, night_id, 100 + k),
            )
            cur.execute(
                "INSERT INTO relphot.lightcurve (obj_id, night_id, frame_index, bjd_tdb, flux, "
                "flux_err, flux_raw) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    obj_id, night_id, [0, 1, 2], [2460310.5006, 2460310.5106, 2460310.5206],
                    [2.0, 2.2, 1.8], [0.01, 0.01, 0.01], [2.0, 2.2, 1.8],
                ),
            )
    test_conn.commit()


def _det_state(test_conn, det_id: int) -> tuple[str, str | None, str | None]:
    """``(status, notes, auto_status)`` of one detection, straight from the table."""
    test_conn.rollback()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT status, notes, auto_status FROM relphot.detection WHERE det_id = %s",
            (det_id,),
        )
        return cur.fetchone()


def _review(test_client, anchor: int, det_ids: list[int], status: str, **extra):
    body = {"anchor_det_id": anchor, "det_ids": det_ids, "status": status, **extra}
    return test_client.post("/api/detections/review", json=body)


def test_similar_endpoint_shape_cap_order_and_anchor_first(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids)
    _add_stack_lcs(test_conn, ids["night1"], [ids["det_a"], similar[0], similar[1]])
    url = f"/api/detection/{ids['det_a']}/similar"

    resp = test_client.get(url)
    assert resp.status_code == 200
    data = resp.json()
    assert (data["n_total"], data["n_returned"], data["n_rejected_hidden"]) == (25, 25, 0)
    # the viewed event first, with its own state; the look-alikes follow, nearest in tc first
    anchor = data["anchor"]
    assert (anchor["det_id"], anchor["obj_name"]) == (ids["det_a"], "RP BOTH01")
    assert (anchor["status"], anchor["auto_status"]) == ("UNCONFIRMED", "REJECTED")
    assert anchor["effective_status"] == "REJECTED (auto)"
    assert anchor["night_label"] == "20250101"
    assert anchor["ingress_frac"] is None  # the incomplete event has no full trapezoid
    assert anchor["duration_display"] == "\u2265 2.00 h"
    assert [e["det_id"] for e in data["events"]] == similar
    first = data["events"][0]
    assert (first["obj_name"], first["night_id"], first["night_label"]) == (
        "RP SIM00", ids["night1"], "20250101"
    )
    assert first["tc"] == pytest.approx(2460310.52)
    assert (first["depth"], first["t14_h"]) == (pytest.approx(0.011), pytest.approx(2.1))
    assert (first["converged"], first["status"], first["notes"]) == (True, "UNCONFIRMED", None)
    assert first["effective_status"] == "UNCONFIRMED"
    # the light curve of the event's own night, flux over its median (2.0), one for each event
    assert first["lc"]["flux"] == pytest.approx([1.0, 1.1, 0.9])
    assert first["lc"]["flux_err"] == pytest.approx([0.005, 0.005, 0.005])
    assert first["lc"]["bjd_tdb"][0] == pytest.approx(2460310.5006)
    assert anchor["lc"]["flux"] == pytest.approx([1.0, 1.1, 0.9])
    assert data["events"][2]["lc"] is None  # no light curve stored

    # every row is on the anchor's night: its observation window, not the rows' own points
    window = _frame_window(test_conn, ids["night1"])
    assert (data["t_first"], data["t_last"]) == pytest.approx(window)

    capped = test_client.get(url, params={"limit": 10}).json()
    assert (capped["n_total"], capped["n_returned"]) == (25, 10)
    assert [e["det_id"] for e in capped["events"]] == similar[:10]
    assert test_client.get(url, params={"limit": 0}).status_code == 422
    assert test_client.get(url, params={"limit": 101}).status_code == 422

    # an event with no coincidence row has no look-alikes; an unknown or non-transit id is a 404
    lone = test_client.get(f"/api/detection/{ids['det_b']}/similar").json()
    assert (lone["anchor"]["det_id"], lone["events"], lone["n_total"]) == (ids["det_b"], [], 0)
    assert (lone["t_first"], lone["t_last"]) == pytest.approx(window)
    assert test_client.get("/api/detection/999999/similar").status_code == 404
    with test_conn.cursor() as cur:
        cur.execute("SELECT det_id FROM relphot.detection WHERE kind = 'variable' LIMIT 1")
        (variable_det,) = cur.fetchone()
    assert test_client.get(f"/api/detection/{variable_det}/similar").status_code == 404


def test_similar_hides_events_a_person_rejected_not_the_auto_veto(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids)
    url = f"/api/detection/{ids['det_a']}/similar"
    # one look-alike a person rejected, one the check rejected automatically (stays listed)
    test_client.patch(f"/api/detection/{similar[1]}", json={"status": "REJECTED"})
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET auto_status = 'REJECTED' WHERE det_id = %s",
            (similar[3],),
        )
    test_conn.commit()

    data = test_client.get(url).json()
    assert [e["det_id"] for e in data["events"]] == similar[:1] + similar[2:]
    assert (data["n_total"], data["n_returned"]) == (25, 24)
    assert (data["n_rejected"], data["n_rejected_hidden"]) == (1, 1)
    assert next(e for e in data["events"] if e["det_id"] == similar[3])["effective_status"] == (
        "REJECTED (auto)"
    )
    shown = test_client.get(url, params={"include_rejected": 1}).json()
    assert [e["det_id"] for e in shown["events"]] == similar
    assert (shown["n_rejected"], shown["n_rejected_hidden"]) == (1, 0)
    assert shown["events"][1]["status"] == "REJECTED"
    # the cap counts what is listed, not what is hidden
    assert test_client.get(url, params={"limit": 5}).json()["events"][1]["det_id"] == similar[2]

    # the viewed event stays the first row even when it is the rejected one
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "REJECTED"})
    again = test_client.get(url).json()
    assert (again["anchor"]["det_id"], again["anchor"]["status"]) == (ids["det_a"], "REJECTED")
    assert [e["det_id"] for e in again["events"]] == similar[:1] + similar[2:]

    # the Transit events table drops the rejected look-alike too (the next one moves up under
    # the cap of 20) and counts it; the veto's own numbers stand
    ev_a = test_client.get(f"/api/object/{ids['obj_both']}").json()["transit_events"][0]
    assert [e["det_id"] for e in ev_a["similar_events"]] == similar[:1] + similar[2:21]
    assert ev_a["n_similar_rejected"] == 1
    assert (ev_a["n_similar"], ev_a["auto_status"]) == (25, "REJECTED")
    assert ev_a["status"] == "REJECTED"


def test_bulk_review_sets_status_notes_and_a_summary_on_the_viewed_event(
    client, test_conn
) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=3)
    today = date.today().isoformat()

    resp = _review(test_client, ids["det_a"], similar, "REJECTED")
    assert resp.status_code == 200
    data = resp.json()
    assert [r["det_id"] for r in data["updated"]] == sorted(similar)
    assert {r["status"] for r in data["updated"]} == {"REJECTED"}
    assert {r["effective_status"] for r in data["updated"]} == {"REJECTED"}
    for det_id in similar:
        assert _det_state(test_conn, det_id) == ("REJECTED", None, None)  # no note given
    # the viewed event: its own verdict untouched, a line about the action appended
    status, notes, _auto = _det_state(test_conn, ids["det_a"])
    assert status == "UNCONFIRMED"
    names = ", ".join(f"RP SIM{k:02d} (det {d})" for k, d in enumerate(similar))
    line = f"{today} REJECT ALL 3 look-alikes: {names}"
    assert notes == line
    assert data["anchor"] == {
        "det_id": ids["det_a"], "obj_id": ids["obj_both"], "status": "UNCONFIRMED", "notes": line,
    }
    # the summary is appended, also to existing notes and with the user's note after a dash
    resp = _review(test_client, ids["det_a"], similar[:1], "UNCONFIRMED", note="looked again")
    assert resp.status_code == 200
    assert _det_state(test_conn, ids["det_a"])[1] == (
        f"{line}\n{today} UNCONFIRM ALL 1 look-alike: RP SIM00 (det {similar[0]}) "
        "\u2014 looked again"
    )
    assert _det_state(test_conn, similar[0])[:2] == ("UNCONFIRMED", "looked again")
    # a note on the viewed event shows in the Transit events table
    ev_a = test_client.get(f"/api/object/{ids['obj_both']}").json()["transit_events"][0]
    assert ev_a["notes"].startswith(line)


def test_bulk_review_note_modes(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=3)
    test_client.patch(f"/api/detection/{similar[0]}", json={"notes": "old"})

    def notes() -> list[str | None]:
        return [_det_state(test_conn, d)[1] for d in similar]

    # append: after the old notes on a new line, alone where there are none
    resp = _review(test_client, ids["det_a"], similar, "CONFIRMED", note="by eye")
    assert resp.status_code == 200
    assert notes() == ["old\nby eye", "by eye", "by eye"]
    assert {_det_state(test_conn, d)[0] for d in similar} == {"CONFIRMED"}
    # an empty note leaves the notes alone, in either mode
    for extra in ({}, {"note": ""}, {"note": "  "}, {"note": None, "note_mode": "replace"},
                  {"note": "", "note_mode": "replace"}):
        resp = _review(test_client, ids["det_a"], similar, "UNCONFIRMED", **extra)
        assert resp.status_code == 200
        assert notes() == ["old\nby eye", "by eye", "by eye"]
    # replace overwrites
    resp = _review(test_client, ids["det_a"], similar, "REJECTED", note="fresh",
                   note_mode="replace")
    assert resp.status_code == 200
    assert notes() == ["fresh", "fresh", "fresh"]
    assert {_det_state(test_conn, d)[0] for d in similar} == {"REJECTED"}
    # every action left its line on the viewed event, the replace did not touch it
    anchor_notes = _det_state(test_conn, ids["det_a"])[1].splitlines()
    assert len(anchor_notes) == 7
    assert anchor_notes[-1].endswith("\u2014 fresh")
    assert [line.split(" ")[1] for line in anchor_notes] == [
        "CONFIRM", "UNCONFIRM", "UNCONFIRM", "UNCONFIRM", "UNCONFIRM", "UNCONFIRM", "REJECT",
    ]


def test_bulk_review_confirm_overrides_the_auto_rejection_and_refreshes_flags(
    client, test_conn
) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=3)
    obj = f"/api/object/{ids['obj_both']}"

    def state() -> tuple:
        o = test_client.get(obj).json()["object"]
        return o["is_exop"], o["class"], o["n_review_pending"], o["n_nights_reviewed"]

    # the viewed event (auto-rejected by the check) is among the ids: a CONFIRMED stands
    resp = _review(test_client, ids["det_a"], [ids["det_a"], *similar], "CONFIRMED", note="real")
    assert resp.status_code == 200
    updated = {r["det_id"]: r for r in resp.json()["updated"]}
    assert updated[ids["det_a"]]["effective_status"] == "CONFIRMED"
    assert updated[ids["det_a"]]["auto_status"] == "REJECTED"
    assert _det_state(test_conn, ids["det_a"])[2] == "REJECTED"  # the check's verdict stays
    assert state()[:2] == (True, "EXOP")
    night1 = next(
        n for n in test_client.get(obj).json()["nights"] if n["night_id"] == ids["night1"]
    )
    assert night1["auto_exop"] is True  # evidence again, as for a single PATCH

    # the viewed event in the ids: its status changes, and its notes are the user's note and
    # then the one summary line (no second append), the others carry the user's note alone
    today = date.today().isoformat()
    resp = _review(test_client, ids["det_a"], [ids["det_a"], *similar[:2]], "REJECTED",
                   note="dup", note_mode="replace")
    assert resp.status_code == 200
    status, notes, _auto = _det_state(test_conn, ids["det_a"])
    names = f"RP SIM00 (det {similar[0]}), RP SIM01 (det {similar[1]})"
    assert status == "REJECTED"
    assert notes == f"dup\n{today} REJECT ALL viewed event + 2 look-alikes: {names} \u2014 dup"
    assert resp.json()["anchor"]["notes"] == notes
    assert next(r for r in resp.json()["updated"] if r["det_id"] == ids["det_a"])["notes"] == notes
    assert _det_state(test_conn, similar[0])[:2] == ("REJECTED", "dup")
    assert _det_state(test_conn, similar[2])[0] == "CONFIRMED"  # not in this request
    assert state() == (True, "EXOP", 1, 1)  # the other event of the object still stands
    # the viewed event is still the first row, now rejected; the rejected look-alikes are hidden
    stack = test_client.get(f"/api/detection/{ids['det_a']}/similar").json()
    assert (stack["anchor"]["det_id"], stack["anchor"]["status"]) == (ids["det_a"], "REJECTED")
    assert [e["det_id"] for e in stack["events"]] == similar[2:]
    assert stack["n_rejected_hidden"] == 2

    # alone, the viewed event leaves no look-alike list; both events of the object rejected
    resp = _review(test_client, ids["det_a"], [ids["det_b"], ids["det_a"]], "REJECTED")
    assert resp.status_code == 200
    assert _det_state(test_conn, ids["det_a"])[1].endswith(
        f"{today} REJECT ALL viewed event + 1 look-alike: RP BOTH01 (det {ids['det_b']})"
    )
    assert state() == (False, "UNC", 0, 2)
    resp = _review(test_client, ids["det_a"], [ids["det_a"]], "UNCONFIRMED")
    assert resp.json()["anchor"]["notes"].endswith(f"{today} UNCONFIRM ALL viewed event")
    assert state() == (False, "UNC", 0, 1)  # unconfirmed, the automatic rejection counts again
    assert _review(test_client, ids["det_a"], [ids["det_a"]], "CONFIRMED").status_code == 200
    assert state()[:2] == (True, "EXOP")


def test_bulk_review_validation_and_atomicity(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=3)
    with test_conn.cursor() as cur:
        cur.execute("SELECT det_id FROM relphot.detection WHERE kind = 'variable' LIMIT 1")
        (variable_det,) = cur.fetchone()

    def untouched() -> bool:
        return all(_det_state(test_conn, d) == ("UNCONFIRMED", None, a) for d, a in (
            [(ids["det_a"], "REJECTED")] + [(d, None) for d in similar]
        ))

    anchor = ids["det_a"]
    assert _review(test_client, anchor, similar, "MAYBE").status_code == 400
    assert _review(test_client, anchor, similar, "REJECTED", note="x" * 2001).status_code == 400
    # an unknown id, a non-transit id or an unknown viewed event: 404 naming it, nothing changes
    resp = _review(test_client, anchor, [*similar, 999999], "REJECTED", note="n")
    assert resp.status_code == 404
    assert "999999" in resp.json()["detail"]
    assert untouched()
    resp = _review(test_client, anchor, [similar[0], variable_det], "REJECTED")
    assert resp.status_code == 404
    assert str(variable_det) in resp.json()["detail"]
    assert untouched()
    resp = _review(test_client, 999998, similar, "REJECTED")
    assert resp.status_code == 404
    assert "999998" in resp.json()["detail"]
    assert untouched()
    # malformed bodies
    assert _review(test_client, anchor, [], "REJECTED").status_code == 422
    assert _review(test_client, anchor, list(range(1, 102)), "REJECTED").status_code == 422
    assert _review(test_client, anchor, similar, "REJECTED", note_mode="merge").status_code == 422
    assert test_client.post("/api/detections/review", json={"det_ids": similar}).status_code == 422
    assert untouched()
    # 100 ids is the most; repeats count once
    assert _review(test_client, anchor, similar + similar, "REJECTED").status_code == 200
    assert _det_state(test_conn, anchor)[1].count("REJECT ALL 3 look-alikes") == 1


def test_bulk_review_runs_as_the_web_role(client, test_conn, monkeypatch) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=2)
    ro_dsn = _role_dsn(_test_dsn(), "relphot_ro", "RELPHOT_RO_PASSWORD")
    with (
        psycopg.connect(ro_dsn, autocommit=True) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        cur.execute("UPDATE relphot.detection SET notes = 'x'")
    # the same request over the read-only role cannot write
    monkeypatch.setenv("RELPHOT_WEB_RW_DSN", ro_dsn)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _review(test_client, ids["det_a"], similar, "REJECTED")
    assert {_det_state(test_conn, d)[0] for d in similar} == {"UNCONFIRMED"}


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

    # the night's EXOP verdict is also the status of its transit event: that event is rejected
    # with it, so it is no automatic evidence any more
    assert data["night"]["auto_exop"] is False
    assert data["night"]["exop_effective"] is False
    assert data["night"]["pending"] is False  # the person has decided this night
    assert data["events_updated"] != []
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
    # the REJECTED verdict rejects the night's transit event too: no automatic evidence left
    assert (first["auto_exop"], first["exop_effective"], first["pending"]) == (False, False, False)


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


def test_front_end_has_the_similar_events_window_and_scroll_boxes() -> None:
    """The similar-events window, the night-review scroll box and the older-nights select."""
    static = resources.files("relphot.web") / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    html_ids = set(re.findall(r'\bid="([\w-]+)"', html))
    for new_id in ("similar-window", "plot-similar", "similar-side", "similar-action",
                   "similar-check-all", "similar-notes", "similar-notes-replace",
                   "similar-apply", "similar-msg", "similar-count", "similar-show-rejected",
                   "similar-show-rejected-label", "similar-show-rejected-text",
                   "night-reviews-scroll"):
        assert new_id in html_ids, new_id
    # the window sits below the periodogram plot
    assert html.index('id="plot-periodogram"') < html.index('id="similar-window"')
    # the older-nights select is built by app.js (it exists only when there are older nights)
    assert "lc-older-nights" in js
    assert "/api/detection/" in js and "/api/detections/review" in js


def test_front_end_pins_every_per_night_time_axis_to_the_nights_window() -> None:
    """The per-night plots set an explicit x range (first to last frame) that a double
    click returns to, not the data extent; the multi-night and phase plots do not."""
    js = (resources.files("relphot.web") / "static" / "app.js").read_text()

    def body(name: str) -> str:
        start = js.index(f"function {name}(")
        return js[start : js.index("\nfunction ", start + 1)]

    for name in ("plotNightLc", "plotReferenceLc", "plotComparisonLc", "renderSimilar"):
        assert "nightXRange(" in body(name), name
        assert "nightPlotConfig(xRange)" in body(name), name
    assert "nightXAxis(xRange)" in body("plotNightLc")
    assert 'doubleClick: "reset"' in body("nightPlotConfig")
    assert "nightXRange" not in body("plotCombinedLc") + body("plotPhase")


def _js_function_body(js: str, name: str) -> str:
    start = js.index(f"function {name}(")
    return js[start : js.index("\nfunction ", start + 1)]


def test_front_end_draws_a_residual_panel_under_fitted_light_curves() -> None:
    """Both the night's light curve and the similar-events stack draw the trapezoid's residuals
    (obs - model) in a panel whose x axis is matched to the curve's, with dashed mean/P16/P84."""
    js = (resources.files("relphot.web") / "static" / "app.js").read_text()

    def body(name: str) -> str:
        return _js_function_body(js, name)

    # one residual helper, one drawn model: the drawn curve and the residuals share the shape
    assert "trapezoidShape(ev, t)" in body("trapezoidTrace")
    assert "trapezoidShape(ev, t)" in body("trapezoidModelFlux")
    assert "trapezoidModelFlux(events, baseline, bjd[i])" in body("lcResiduals")
    assert "yOf(flux[i]) - yOf(trapezoidModelFlux" in body("lcResiduals")
    # the night curve: the drawn events, the plot's own transform and errors, a matched x axis
    night = body("plotNightLc")
    assert "lcResiduals(lc.bjd_tdb, lc.flux, yErr, fitted, baseline, yOf" in night
    assert "residualTraces(res" in night and "residualXAxis(xRange" in night
    assert "residualYAxis(" in night and "useMag" in night.split("residualYAxis(")[1]
    assert "error_y: { type: \"data\", visible: true, array: yErr }" in night
    # the stack: only rows with a fit and a light curve get a panel, from the same layout numbers
    sim = body("renderSimilar")
    assert "similarResidual" in sim and "residualTraces(res" in sim and "residualXAxis(" in sim
    assert "similarRowLayout(s.resid)" in sim
    assert "similarRowLayout(s.resid)" in body("renderSimilarSide")
    side = body("renderSimilarSide")
    assert "rl.tops[i]" in side and "rl.heights[i]" in side
    assert "!ev.lc || !similarHasFit(ev)" in body("similarResidual")
    assert "similarHasFit(ev)" in sim
    # the x axis is matched to the light curve's and pinned like it; ticks only on the bottom panel
    assert 'matches: "x"' in body("residualXAxis")
    assert "range: xRange" in body("residualXAxis")
    assert "showticklabels: bottom" in body("residualXAxis")
    assert "zeroline: true" in body("residualYAxis")
    # percentiles by linear interpolation; the three dashed lines, the mean darker
    assert "q * (sorted.length - 1)" in body("quantileSorted")
    assert "(pos - lo)" in body("quantileSorted")
    stats = body("residualStats")
    assert "quantileSorted(sorted, 0.16)" in stats and "quantileSorted(sorted, 0.84)" in stats
    lines = body("residualTraces")
    assert 'dash: "dash"' in lines
    for label in ('"mean"', '"P16"', '"P84"'):
        assert label in lines
    assert "rgb(20,20,20)" in lines and "rgb(130,130,130)" in lines
    # the time marker spans the whole figure (paper), so it also crosses the residual panel
    assert 'yref: "paper"' in body("applyTimeMarker")


def test_front_end_residual_statistics_are_numerically_right() -> None:
    """Run the percentile / residual helpers of app.js under node (skipped without node)."""
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = (resources.files("relphot.web") / "static" / "app.js").read_text()
    names = ("trapezoidShape", "trapezoidModelFlux", "quantileSorted", "residualStats",
             "lcResiduals")
    src = "\n".join(_js_function_body(js, name) for name in names)
    script = src + """
const ev = { tc: 10.0, t14_h: 2.4, ingress_frac: 0.25, depth: 0.01 };
const ev2 = { tc: 10.0, t14_h: 2.4, ingress_frac: 0.25, depth: 0.02 };
const out = {
  q: [0, 0.16, 0.5, 0.84, 1].map((q) => quantileSorted([1, 2, 3, 4], q)),
  stats: residualStats([4, 1, 3, 2]),
  few: residualStats([1]),
  centre: trapezoidModelFlux([ev], 1, 10.0),
  edge: trapezoidModelFlux([ev], 2, 10.0 + 0.0501),
  both: trapezoidModelFlux([ev, ev2], 1, 10.0),
  res: lcResiduals([10.0, 11.0, NaN, 12.0], [0.99, 1.002, 1.0, null], [0.001, 0.001, 0.001, 0.001],
                   [ev], 1, (f) => f, ["a", "b", "c", "d"]),
  mag: lcResiduals([11.0], [1.0], [0.001], [ev], 1, (f) => 20 - 2.5 * Math.log10(f), null),
};
console.log(JSON.stringify(out));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    out = json.loads(done.stdout)
    assert out["q"] == pytest.approx([1.0, 1.48, 2.5, 3.52, 4.0])
    assert out["stats"]["mean"] == pytest.approx(2.5)
    assert out["stats"]["p16"] == pytest.approx(1.48)
    assert out["stats"]["p84"] == pytest.approx(3.52)
    assert out["few"] is None
    assert out["centre"] == pytest.approx(0.99)  # baseline * (1 - depth) on the flat bottom
    assert out["edge"] == pytest.approx(2.0, rel=0.01)  # outside / on the ramp: near the baseline
    assert out["both"] == pytest.approx(0.99 * 0.98)  # overlapping events multiply
    res = out["res"]  # the NaN epoch and the null flux are skipped
    assert res["x"] == pytest.approx([-2459990.0, -2459989.0])
    assert res["y"] == pytest.approx([0.99 - 0.99, 1.002 - 1.0])
    assert res["text"] == ["a", "b"]
    assert res["stats"]["mean"] == pytest.approx(0.001)
    assert out["mag"]["y"] == pytest.approx([0.0], abs=1e-12)  # same transform on obs and model


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


def test_apparent_mean_mag_source_is_the_common_source_else_mixed(client, test_conn) -> None:
    test_client, ids = client
    mean_mag, _ = _mean_star_mag(test_conn, ids["obj_var"])

    def detail_of(key: str) -> dict:
        return test_client.get(f"/api/object/{ids[key]}").json()["object"]

    def row_of(key: str) -> dict:
        rows = test_client.get("/api/search").json()["rows"]
        return {r["obj_id"]: r for r in rows}[ids[key]]

    # every night measured: the measured source, with its shared zero point in the detail
    _set_night_zp(test_conn, ids["night1"], 27.85, "measured")
    _set_night_zp(test_conn, ids["night2"], 27.85, "measured")
    var = row_of("obj_var")
    assert var["mag_zp_source"] == "measured"
    assert var["mean_mag_app"] == pytest.approx(mean_mag + 27.85, abs=1e-3)
    detail = detail_of("obj_var")
    assert detail["mag_zp_source"] == "measured" and detail["mag_zp"] == pytest.approx(27.85)

    # measured + assumed, and measured + Gaia: mixed, no single zero point
    _set_night_zp(test_conn, ids["night2"], 20.0, "assumed")
    assert row_of("obj_var")["mag_zp_source"] == "mixed"
    detail = detail_of("obj_var")
    assert detail["mag_zp_source"] == "mixed" and detail["mag_zp"] is None
    _set_night_zp(test_conn, ids["night2"], 24.5, "gaia")
    assert row_of("obj_var")["mag_zp_source"] == "mixed"

    # an object seen on night 1 only rests on that night's source alone
    assert row_of("obj_unc")["mag_zp_source"] == "measured"
    assert detail_of("obj_unc")["mag_zp"] == pytest.approx(27.85)

    # every night Gaia-calibrated, with different zero points: still Gaia, no single value
    _set_night_zp(test_conn, ids["night1"], 24.0, "gaia")
    assert row_of("obj_var")["mag_zp_source"] == "gaia"
    assert detail_of("obj_var")["mag_zp"] is None

    # the light-curve payloads carry the source of the night(s) they use
    _set_night_zp(test_conn, ids["night1"], 27.85, "measured")
    lc = test_client.get(
        f"/api/object/{ids['obj_var']}/lc", params={"night_id": ids["night1"]}
    ).json()
    assert lc["zp"] == pytest.approx(27.85) and lc["zp_source"] == "measured"
    untied = test_client.get(f"/api/object/{ids['obj_var2']}/lc/combined").json()
    assert untied["zp_source"] == "mixed"
    _set_night_zp(test_conn, ids["night2"], 27.85, "measured")
    untied = test_client.get(f"/api/object/{ids['obj_var2']}/lc/combined").json()
    assert untied["zp_source"] == "measured"


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
    # one label per zero-point source
    for text in ("Gaia ZP", "measured (Gaia-matched)", "assumed", "mixed ZP"):
        assert text in js


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


# --------------------------------------------------------------------------
# Schema v11: tile members (reference and comparison stars)
# --------------------------------------------------------------------------


def _insert_tile_members(test_conn, ids: dict) -> None:
    """Insert tile member data for schema v11 testing.

    For the fixture's night1 and the fixture star (obj_unc) at tile 0, best_aperture 1:
    - Insert night_tile with metadata
    - Insert tile_lc rows for apertures 0-2 (only aperture 1 has n_comp > 0)
    - Mark frame 1 as kept=false and put NaN at index 1 in all arrays
    - Insert 12 comparison_member rows (one is the fixture star with weight 0.05,
      one with obj_id NULL, one with clipped_frames [1])
    - Insert 3 reference_member rows with distinct weights
    - Ensure fixture star's lightcurve is sparse
    """
    # Prime the column cache to avoid AttributeError when endpoints call column_exists(None, ...)
    from relphot.web.db import _column_cache
    with test_conn.cursor() as cache_cur:
        for table in ["night_tile", "tile_lc", "reference_member", "comparison_member"]:
            cache_cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'relphot' AND table_name = %s",
                (table,),
            )
            _column_cache[table] = frozenset(row[0] for row in cache_cur.fetchall())

    night1 = ids["night1"]
    obj_unc = ids["obj_unc"]
    tile = 0
    best_aperture = 1

    with test_conn.cursor() as cur:
        # Get fixture star's star_id and update best_aperture
        cur.execute(
            "UPDATE relphot.star_night SET tile = %s, best_aperture = %s "
            "WHERE obj_id = %s AND night_id = %s "
            "RETURNING star_id",
            (tile, best_aperture, obj_unc, night1),
        )
        (star_id,) = cur.fetchone()

        # Insert night_tile
        cur.execute(
            "INSERT INTO relphot.night_tile (night_id, tile, x_min, x_max, y_min, y_max, "
            "n_core, n_extended, n_ref_stars, ref_aperture, best_apertures) "
            "VALUES (%s, %s, 100.0, 200.0, 50.0, 150.0, 10, 5, 3, %s, %s)",
            (night1, tile, best_aperture, [1]),
        )

        # Mark frame 1 as not kept
        cur.execute(
            "UPDATE relphot.frame SET kept = false WHERE night_id = %s AND frame_index = 1",
            (night1,),
        )

        # Insert tile_lc for apertures 0, 1, 2; only aperture 1 gets comparison members
        # Each tile_lc has 3 frames worth of data, with NaN at index 1
        ref_flux_template = [1.0, np.nan, 1.05]
        ref_flux_err_template = [0.01, np.nan, 0.01]
        ens_flux_template = [0.95, np.nan, 1.02]
        ens_flux_err_template = [0.015, np.nan, 0.015]

        for aperture in [0, 1, 2]:
            n_comp = 12 if aperture == best_aperture else 0
            cur.execute(
                "INSERT INTO relphot.tile_lc (night_id, tile, aperture, ref_flux, ref_flux_err, "
                "ens_flux, ens_flux_err, n_ensemble, n_comp, n_rounds) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (night1, tile, aperture, ref_flux_template, ref_flux_err_template,
                 ens_flux_template, ens_flux_err_template, 12, n_comp, 3),
            )

        # Insert 3 reference_member rows with distinct weights
        for i, weight in enumerate([0.4, 0.35, 0.25]):
            cur.execute(
                "INSERT INTO relphot.reference_member (night_id, tile, star_id, obj_id, "
                "ra, dec, mag, weight, in_core) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (night1, tile, 10 + i, None, 100.0 + i, 50.0 + i, 14.0 + 0.1*i, weight, True),
            )

        # Insert 12 comparison_member rows for aperture 1
        # Include: fixture star itself (weight 0.05), one with obj_id NULL (1/12),
        # one with clipped_frames [1], and 9 others
        comp_members = [
            # (star_id, obj_id, ra, dec, mag, weight, n_clipped, clipped_frames)
            (star_id, obj_unc, 50.0, 75.0, 14.0, 0.05, 0, []),  # the fixture star
            (100, None, 51.0, 76.0, 14.5, 1/12, 0, []),  # obj_id NULL
            (101, None, 52.0, 77.0, 14.8, 1/12, 1, [1]),  # clipped_frames [1]
        ]
        # Add 9 more comparison members
        for i in range(9):
            star_idx = 102 + i
            comp_members.append((
                star_idx, None, 53.0 + i*0.1, 78.0 + i*0.1, 15.0 + i*0.05, 1/12, 0, []
            ))

        norm_flux_template = [0.98, np.nan, 1.02]

        for (star_id_c, obj_id_c, ra_c, dec_c, mag_c, weight_c, n_clipped_c,
             clipped_c) in comp_members:
            cur.execute(
                "INSERT INTO relphot.comparison_member "
                "(night_id, tile, aperture, star_id, obj_id, ra, dec, mag, weight, "
                "n_clipped, clipped_frames, norm_flux) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (night1, tile, best_aperture, star_id_c, obj_id_c, ra_c, dec_c, mag_c,
                 weight_c, n_clipped_c, clipped_c, norm_flux_template),
            )

        # Make fixture star's lightcurve sparse (only frames 0 and 2)
        cur.execute(
            "DELETE FROM relphot.lightcurve WHERE obj_id = %s AND night_id = %s",
            (obj_unc, night1),
        )
        cur.execute(
            "INSERT INTO relphot.lightcurve (obj_id, night_id, frame_index, bjd_tdb, flux, "
            "flux_err, flux_raw) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (obj_unc, night1, [0, 2], [2460310.5006, 2460310.5206], [1.0, 0.99],
             [0.01, 0.01], [1.0, 0.99]),
        )

    test_conn.commit()


def test_nights_list_shape_and_has_members(client, test_conn) -> None:
    """GET /api/nights returns shape with has_members flag."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get("/api/nights")
    assert resp.status_code == 200
    data = resp.json()
    assert "nights" in data
    night1_data = next((n for n in data["nights"] if n["night_id"] == ids["night1"]), None)
    assert night1_data is not None
    assert "has_members" in night1_data
    assert night1_data["has_members"] is True
    assert "n_tiles" in night1_data


def test_night_tiles_shape_and_n_comp(client, test_conn) -> None:
    """GET /api/night/{id}/tiles returns tiles with n_comp (comparison member count)."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(f"/api/night/{ids['night1']}/tiles")
    assert resp.status_code == 200
    data = resp.json()
    assert "night" in data and "tiles" in data
    assert data["night"]["night_id"] == ids["night1"]
    assert len(data["tiles"]) == 1
    tile = data["tiles"][0]
    assert tile["tile"] == 0
    assert tile["n_comp"] == 3  # 3 apertures (0, 1, 2) in tile_lc
    assert "best_apertures" in tile


def test_reference_null_at_dropped_frame(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/reference has null at dropped frame."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/reference",
        params={"aperture": 1}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ref_flux" in data
    assert data["ref_flux"][1] is None  # frame 1 is kept=false, so NaN
    assert data["ref_flux"][0] is not None
    assert data["ref_flux"][2] is not None


def test_reference_frame_kept_false_at_dropped(client, test_conn) -> None:
    """GET /api/night/{id}/tiles/references frame has kept=false for dropped frame."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(f"/api/night/{ids['night1']}/tiles/references", params={"aperture": 1})
    assert resp.status_code == 200
    data = resp.json()
    frames = data["frames"]
    assert len(frames) == 3
    assert frames[1]["kept"] is False
    assert frames[0]["kept"] is True
    assert frames[2]["kept"] is True


def test_reference_n_ref_count(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/reference has correct n_ref count."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/reference",
        params={"aperture": 1}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_ref"] == 3  # 3 reference_member rows


def test_reference_members_ordered_by_weight_desc(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/reference/members ordered by weight desc."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(f"/api/night/{ids['night1']}/tile/0/reference/members")
    assert resp.status_code == 200
    data = resp.json()
    assert "members" in data
    members = data["members"]
    assert len(members) == 3
    # Check ordered by weight descending
    weights = [m["weight"] for m in members]
    assert weights == sorted(weights, reverse=True)
    assert weights == pytest.approx([0.4, 0.35, 0.25])


def test_reference_members_include_name_from_object(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/reference/members includes name joined from object."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(f"/api/night/{ids['night1']}/tile/0/reference/members")
    assert resp.status_code == 200
    data = resp.json()
    members = data["members"]
    # All reference members have obj_id NULL, so name should be None
    for m in members:
        assert m["name"] is None
        assert "mag_app" in m


def test_comparison_with_limit_and_n_shown(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/comparison limit param controls n_shown."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/comparison",
        params={"aperture": 1, "limit": 5}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "n_shown" in data
    assert "n_members" in data
    assert data["n_shown"] == 5
    assert data["n_members"] == 12
    # All members are returned, but only n_shown are marked as shown
    assert len(data["members"]) == 12
    assert sum(1 for m in data["members"] if m.get("shown", False)) == 5


def test_comparison_envelope_length_equals_n_frames(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/comparison envelope arrays have length n_frames."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/comparison",
        params={"aperture": 1, "limit": 10}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "envelope" in data
    if data["envelope"] is not None:
        assert "median" in data["envelope"]
        assert "lo" in data["envelope"]
        assert "hi" in data["envelope"]
        assert len(data["envelope"]["median"]) == 3  # 3 frames
        assert len(data["envelope"]["lo"]) == 3
        assert len(data["envelope"]["hi"]) == 3


def test_comparison_order_parameter_accepted(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/comparison accepts order parameter."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    for order in ["mag", "weight", "rms"]:
        resp = test_client.get(
            f"/api/night/{ids['night1']}/tile/0/comparison",
            params={"aperture": 1, "order": order}
        )
        assert resp.status_code == 200
        assert resp.json()["order"] == order


def test_comparison_limit_validation(client, test_conn) -> None:
    """GET /api/night/{id}/tile/{t}/comparison rejects invalid limit."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    # limit < 1 or > 1000 should be rejected
    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/comparison",
        params={"aperture": 1, "limit": 0}
    )
    assert resp.status_code == 422

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/comparison",
        params={"aperture": 1, "limit": 1001}
    )
    assert resp.status_code == 422


def test_comparison_member_norm_flux_is_full_length(client, test_conn) -> None:
    """Comparison member norm_flux is full length (n_frames) with NaN for dropped frames."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/night/{ids['night1']}/tile/0/comparison",
        params={"aperture": 1, "limit": 12}
    )
    assert resp.status_code == 200
    data = resp.json()
    for member in data["members"]:
        assert len(member["norm_flux"]) == 3  # 3 frames total
        # Frame 1 should be None (kept=false)
        assert member["norm_flux"][1] is None
        # Frames 0 and 2 should have values
        assert member["norm_flux"][0] is not None
        assert member["norm_flux"][2] is not None


def test_object_night_reference_endpoint(client, test_conn) -> None:
    """GET /api/object/{obj}/night/{id}/reference works for member."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/object/{ids['obj_unc']}/night/{ids['night1']}/reference"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ref_flux" in data
    assert data["night_id"] == ids["night1"]
    assert data["tile"] == 0


def test_object_night_comparison_endpoint(client, test_conn) -> None:
    """GET /api/object/{obj}/night/{id}/comparison works for member."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/object/{ids['obj_unc']}/night/{ids['night1']}/comparison",
        params={"limit": 12}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "members" in data
    assert "n_members" in data


def test_reference_and_comparison_carry_the_nights_frame_window(client, test_conn) -> None:
    """The fixture star's own light curve lacks frame 1, which is also the one dropped frame; the
    window is that of all the frames, on the object's and on the night's endpoints."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)
    window = _frame_window(test_conn, ids["night1"])
    assert window == pytest.approx((2460310.5006, 2460310.5206))
    night, obj = ids["night1"], ids["obj_unc"]

    for url, params in (
        (f"/api/object/{obj}/night/{night}/reference", {}),
        (f"/api/object/{obj}/night/{night}/comparison", {"limit": 12}),
        (f"/api/night/{night}/tile/0/reference", {"aperture": 1}),
        (f"/api/night/{night}/tile/0/comparison", {"limit": 12}),
    ):
        resp = test_client.get(url, params=params)
        assert resp.status_code == 200, (url, resp.text)
        data = resp.json()
        assert (data["t_first"], data["t_last"]) == pytest.approx(window), url

    # dropped first and last frames stay inside the window (their lines are drawn on the plot)
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.frame SET kept = false WHERE night_id = %s AND frame_index IN (0, 2)",
            (night,),
        )
    test_conn.commit()
    data = test_client.get(f"/api/object/{obj}/night/{night}/reference").json()
    assert (data["t_first"], data["t_last"]) == pytest.approx(window)


def test_object_wrapper_target_is_member_true_with_weight(client, test_conn) -> None:
    """Object wrapper in comparison payload: target.is_member True and weight 0.05."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/object/{ids['obj_unc']}/night/{ids['night1']}/comparison",
        params={"limit": 12}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "target" in data
    target = data["target"]
    assert target["is_member"] is True
    assert target["weight"] == pytest.approx(0.05)


def test_object_wrapper_target_norm_flux_sparse_nulls(client, test_conn) -> None:
    """Target norm_flux: full length with nulls at sparse LC gaps."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(
        f"/api/object/{ids['obj_unc']}/night/{ids['night1']}/comparison",
        params={"limit": 12}
    )
    assert resp.status_code == 200
    data = resp.json()
    target = data["target"]
    # Fixture star's LC is sparse: only frames 0 and 2
    # So norm_flux should have length 3 with None at indices 1
    assert "norm_flux" in target
    assert len(target["norm_flux"]) == 3
    assert target["norm_flux"][0] is not None
    assert target["norm_flux"][1] is None  # sparse LC doesn't have frame 1
    assert target["norm_flux"][2] is not None




def test_object_404_without_star_night_on_night(client, test_conn) -> None:
    """GET /api/object/{obj}/night/{id}/comparison 404 when object not on night."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    # obj_var is not part of our tile setup, so it won't have star_night at night1/tile0
    # But obj_var does have star_night on night1, just at a different tile
    # Use an object that truly has no star_night on night1
    resp = test_client.get(
        f"/api/object/{ids['obj_var2']}/night/{ids['night1']}/reference"
    )
    # obj_var2 does have star_night on night1, so let's use a higher obj_id that doesn't
    resp = test_client.get(f"/api/object/99999999/night/{ids['night1']}/reference")
    assert resp.status_code == 404


def test_unknown_night_returns_404(client, test_conn) -> None:
    """GET /api/night/{id}/tiles returns 404 for unknown night."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get("/api/night/99999/tiles")
    assert resp.status_code == 404


def test_object_detail_has_tile_and_reference_comparison_flags(client, test_conn) -> None:
    """Object detail nights entries include tile, has_reference, has_comparison."""
    test_client, ids = client
    _insert_tile_members(test_conn, ids)

    resp = test_client.get(f"/api/object/{ids['obj_unc']}")
    assert resp.status_code == 200
    data = resp.json()
    nights = data["nights"]
    night1_entry = next((n for n in nights if n["night_id"] == ids["night1"]), None)
    assert night1_entry is not None
    assert "tile" in night1_entry
    assert night1_entry["tile"] == 0
    assert "has_reference" in night1_entry
    assert night1_entry["has_reference"] is True
    assert "has_comparison" in night1_entry
    assert night1_entry["has_comparison"] is True


def test_object_detail_has_reference_false_without_members(client) -> None:
    """Night without tile_lc has has_reference False."""
    test_client, ids = client
    # Don't call _insert_tile_members; check objects on nights without members

    resp = test_client.get(f"/api/object/{ids['obj_var']}")
    assert resp.status_code == 200
    data = resp.json()
    # obj_var has star_night on night1 and night2, but no tile_lc entries
    for night in data["nights"]:
        assert night["has_reference"] is False
        assert night["has_comparison"] is False


def test_relphot_ro_cannot_insert_into_new_tables(client) -> None:
    """relphot_ro role cannot INSERT into night_tile, tile_lc, etc."""
    _, ids = client
    owner_dsn = _test_dsn()
    ro_dsn = _role_dsn(owner_dsn, "relphot_ro", "RELPHOT_RO_PASSWORD")

    with psycopg.connect(ro_dsn) as ro_conn, ro_conn.cursor() as cur:
        # Try INSERT into night_tile
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "INSERT INTO relphot.night_tile (night_id, tile) VALUES (%s, %s)",
                (ids["night1"], 0),
            )
        ro_conn.rollback()

        # Try INSERT into tile_lc
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "INSERT INTO relphot.tile_lc (night_id, tile, aperture, ref_flux, "
                "ref_flux_err) VALUES (%s, %s, %s, %s, %s)",
                (ids["night1"], 0, 1, [], []),
            )
        ro_conn.rollback()

        # Try INSERT into reference_member
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "INSERT INTO relphot.reference_member (night_id, tile, star_id) "
                "VALUES (%s, %s, %s)",
                (ids["night1"], 0, 0),
            )
        ro_conn.rollback()

        # Try INSERT into comparison_member
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(
                "INSERT INTO relphot.comparison_member (night_id, tile, aperture, "
                "star_id, norm_flux) VALUES (%s, %s, %s, %s, %s)",
                (ids["night1"], 0, 1, 0, []),
            )
        ro_conn.rollback()


def test_sql_box_can_select_from_new_tables(client) -> None:
    """SQL box can SELECT from night_tile, tile_lc, reference_member, comparison_member."""
    test_client, _ = client

    # Test SELECT from night_tile
    resp = test_client.post("/api/sql", json={"sql": "SELECT COUNT(*) FROM relphot.night_tile"})
    assert resp.status_code == 200
    assert resp.json()["rows"][0][0] == 0  # No members inserted yet

    # Test SELECT from tile_lc
    resp = test_client.post("/api/sql", json={"sql": "SELECT COUNT(*) FROM relphot.tile_lc"})
    assert resp.status_code == 200
    assert resp.json()["rows"][0][0] == 0

    # Test SELECT from reference_member
    resp = test_client.post(
        "/api/sql", json={"sql": "SELECT COUNT(*) FROM relphot.reference_member"}
    )
    assert resp.status_code == 200
    assert resp.json()["rows"][0][0] == 0

    # Test SELECT from comparison_member
    resp = test_client.post(
        "/api/sql", json={"sql": "SELECT COUNT(*) FROM relphot.comparison_member"}
    )
    assert resp.status_code == 200
    assert resp.json()["rows"][0][0] == 0


def test_front_end_element_ids_include_tile_member_controls() -> None:
    """Front-end HTML includes all required ids for tile member UI elements."""
    static = resources.files("relphot.web") / "static"
    html = (static / "index.html").read_text()
    html_ids = set(re.findall(r'\bid="([\w-]+)"', html))

    required_ids = [
        "btn-reference-lc", "btn-comparison-lc", "plot-reference", "plot-comparison",
        "comparison-order-select", "comparison-limit-select", "comparison-view-select",
        "tile-lc-note", "reference-members-details", "night-panel",
        "night-reviews-scroll", "similar-window", "plot-similar",
    ]
    for req_id in required_ids:
        assert req_id in html_ids, f"Required id '{req_id}' not found in index.html"


# --------------------------------------------------------------------------
# repeated transit events: /api/object repeat_families, repeat_link, /api/repeat/predict
# --------------------------------------------------------------------------

_TC_A, _TC_B = 2460310.52, 2460311.52  # the fixture's two events of obj_both (dt = 1 d)


def _add_family(test_conn, ids: dict) -> int:
    """A stored family of obj_both's two events with four aliases of every kind of status."""
    obj_id, det_a, det_b = ids["obj_both"], ids["det_a"], ids["det_b"]
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.repeat_link (det_a, det_b, obj_id, night_a, night_b, dt_days, "
            "p_match, p_joint, chi2_joint, dof_joint, phys_ok, n_alias, diurnal, linked) "
            "VALUES (%s, %s, %s, %s, %s, 1.0, 0.6, 0.4, 2.0, 1, true, 2, true, true)",
            (det_a, det_b, obj_id, ids["night1"], ids["night2"]),
        )
        cur.execute(
            "INSERT INTO relphot.repeat_family (obj_id, family_key, n_members, member_night_ids, "
            "depth, depth_err, t14_h, t14_lower_limit, ingress_frac, score, n_alias, n_allowed) "
            "VALUES (%s, 'k', 2, %s, 0.01, 0.001, 2.0, false, 0.2, 0.4, 4, 2) RETURNING fam_id",
            (obj_id, [ids["night1"], ids["night2"]]),
        )
        (fam_id,) = cur.fetchone()
        cur.executemany(
            "INSERT INTO relphot.repeat_family_member (fam_id, det_id) VALUES (%s, %s)",
            [(fam_id, det_a), (fam_id, det_b)],
        )
        for k, period, status, veto_night, dchi2 in (
            (1, 1.0, "allowed", None, None),
            (2, 0.5, "allowed", None, None),
            (3, 1.0 / 3.0, "vetoed_nondetection", ids["night2"], 25.0),
            (4, 0.25, "vetoed_density", None, None),
        ):
            cur.execute(
                "INSERT INTO relphot.repeat_ephemeris (fam_id, obj_id, family_key, alias_k, "
                "period, period_err, tc0, tc0_err, status, veto_night_id, veto_dchi2, "
                "n_nights_tested) VALUES (%s, %s, 'k', %s, %s, 0.0007, %s, 0.0005, %s, %s, %s, 1)",
                (fam_id, obj_id, k, period, _TC_A, status, veto_night, dchi2),
            )
    test_conn.commit()
    return fam_id


def test_object_detail_repeat_families_shape(client, test_conn) -> None:
    test_client, ids = client
    fam_id = _add_family(test_conn, ids)

    data = test_client.get(f"/api/object/{ids['obj_both']}").json()

    (fam,) = data["repeat_families"]
    assert fam["fam_id"] == fam_id
    assert (fam["n_members"], fam["n_members_stored"], fam["accepted"]) == (2, 2, False)
    assert (fam["stale"], fam["stale_reason"]) == (False, None)
    assert (fam["n_alias"], fam["n_allowed"]) == (4, 2)
    a, b = fam["members"]
    assert (a["det_id"], b["det_id"]) == (ids["det_a"], ids["det_b"])
    assert (a["night_label"], a["telescope"], a["loose"]) == ("20250101", "T80S", False)
    assert a["tc"] == pytest.approx(_TC_A) and a["depth"] == pytest.approx(0.010)
    assert a["t14_lower_limit"] is True and a["duration_display"] == "≥ 2.00 h"
    assert a["ingress_frac"] is None and b["ingress_frac"] == pytest.approx(0.2)
    assert b["duration_display"] == "2.00 h"
    (link,) = fam["links"]
    assert (link["det_a"], link["det_b"]) == (ids["det_a"], ids["det_b"])
    assert (link["p_match"], link["p_joint"], link["phys_ok"], link["diurnal"]) == (
        pytest.approx(0.6), pytest.approx(0.4), True, True
    )
    assert (link["decision"], link["decision_applied"], link["linked"]) == (None, None, True)
    assert (link["night_label_a"], link["night_label_b"]) == ("20250101", "20250102")
    # every alias is listed, with its status and the night that vetoed it
    assert [(x["alias_k"], x["status"]) for x in fam["aliases"]] == [
        (1, "allowed"), (2, "allowed"), (3, "vetoed_nondetection"), (4, "vetoed_density")
    ]
    veto = fam["aliases"][2]
    assert (veto["veto_night_label"], veto["veto_dchi2"], veto["n_nights_tested"]) == (
        "20250102", pytest.approx(25.0), 1
    )
    assert fam["aliases"][0]["period"] == pytest.approx(1.0)
    assert fam["aliases"][0]["period_err"] == pytest.approx(0.0007)
    assert data["repeat_decisions"] == []

    plain = test_client.get(f"/api/object/{ids['obj_unc']}").json()
    assert plain["repeat_families"] == [] and plain["repeat_decisions"] == []


def test_object_detail_marks_a_loose_night_member(client, test_conn) -> None:
    test_client, ids = client
    _add_family(test_conn, ids)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.mn_run (stem, labels, anchor, loaded_at, loose_night_ids) "
            "VALUES ('loose_run', %s, 'a', now(), %s)", (["a", "b"], [ids["night2"]]),
        )
    test_conn.commit()
    (fam,) = test_client.get(f"/api/object/{ids['obj_both']}").json()["repeat_families"]
    assert [m["loose"] for m in fam["members"]] == [False, True]


def test_a_rejected_member_leaves_the_family_marked_stale(client, test_conn) -> None:
    test_client, ids = client
    _add_family(test_conn, ids)
    url = f"/api/object/{ids['obj_both']}"

    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "REJECTED"})
    (fam,) = test_client.get(url).json()["repeat_families"]
    assert [m["det_id"] for m in fam["members"]] == [ids["det_a"]]
    assert fam["links"] == [] and fam["n_members_stored"] == 2
    assert fam["stale"] is True
    assert fam["stale_reason"].startswith("stale — recomputed at next analyze")
    assert len(fam["aliases"]) == 4  # nothing is recomputed before the next analyze

    # an auto-rejected member goes the same way unless the person CONFIRMED it
    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "UNCONFIRMED"})
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET auto_status = 'REJECTED', auto_reason = 'x' "
            "WHERE det_id = %s", (ids["det_b"],),
        )
    test_conn.commit()
    (fam,) = test_client.get(url).json()["repeat_families"]
    assert fam["stale"] is True and len(fam["members"]) == 1
    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "CONFIRMED"})
    (fam,) = test_client.get(url).json()["repeat_families"]
    assert fam["stale"] is False and len(fam["members"]) == 2


def _decisions(test_conn) -> list[tuple]:
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT obj_id, night_a, night_b, tc_a, tc_b, decision, note "
            "FROM relphot.repeat_decision ORDER BY night_a, night_b, tc_a"
        )
        return cur.fetchall()


def test_repeat_link_put_and_delete_round_trip(client, test_conn) -> None:
    test_client, ids = client
    _add_family(test_conn, ids)
    obj = ids["obj_both"]
    link = f"/api/object/{obj}/repeat_link"
    detail = f"/api/object/{obj}"

    resp = test_client.put(
        link, json={"det_a": ids["det_b"], "det_b": ids["det_a"], "decision": "DIFFERENT",
                    "note": "another planet"},  # the pair in either order
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "next `relphot db analyze`" in body["note"]
    assert (body["det_a"], body["det_b"]) == (ids["det_a"], ids["det_b"])
    assert body["decision"]["decision"] == "DIFFERENT"
    assert _decisions(test_conn) == [
        (obj, ids["night1"], ids["night2"], pytest.approx(_TC_A), pytest.approx(_TC_B),
         "DIFFERENT", "another planet")
    ]

    data = test_client.get(detail).json()
    (fam,) = data["repeat_families"]
    (lk,) = fam["links"]
    assert (lk["decision"], lk["decision_applied"]) == ("DIFFERENT", None)
    assert fam["stale"] is True and "decision" in fam["stale_reason"]
    assert len(fam["members"]) == 2  # DIFFERENT keeps both events; nothing is merged or rejected
    (dec,) = data["repeat_decisions"]
    assert (dec["label_a"], dec["label_b"], dec["decision"], dec["note"]) == (
        "20250101", "20250102", "DIFFERENT", "another planet"
    )
    with test_conn.cursor() as cur:
        cur.execute("SELECT status FROM relphot.detection WHERE det_id = ANY(%s)",
                    ([ids["det_a"], ids["det_b"]],))
        assert {r[0] for r in cur.fetchall()} == {"UNCONFIRMED"}

    # a second verdict on the pair replaces the first
    resp = test_client.put(link, json={"det_a": ids["det_a"], "det_b": ids["det_b"],
                                       "decision": "SAME"})
    assert resp.status_code == 200
    assert [d[5:] for d in _decisions(test_conn)] == [("SAME", None)]

    resp = test_client.delete(link, params={"det_a": ids["det_b"], "det_b": ids["det_a"]})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1 and "analyze" in resp.json()["note"]
    assert _decisions(test_conn) == []
    (fam,) = test_client.get(detail).json()["repeat_families"]
    assert fam["stale"] is False and fam["links"][0]["decision"] is None
    assert test_client.delete(link, params={"det_a": ids["det_a"], "det_b": ids["det_b"]}
                              ).json()["deleted"] == 0


def test_repeat_link_validation(client, test_conn) -> None:
    test_client, ids = client
    obj = ids["obj_both"]
    link = f"/api/object/{obj}/repeat_link"
    ok = {"det_a": ids["det_a"], "det_b": ids["det_b"], "decision": "SAME"}

    assert test_client.put(link, json={**ok, "decision": "MAYBE"}).status_code == 400
    assert test_client.put(link, json={**ok, "decision": None}).status_code == 422
    assert test_client.put(link, json={**ok, "note": "x" * 2001}).status_code == 400
    assert test_client.put(link, json={**ok, "det_b": ids["det_a"]}).status_code == 400
    # an event of another object, or one with no fit, is not a pair of this object
    assert test_client.put(
        f"/api/object/{ids['obj_exop']}/repeat_link", json=ok
    ).status_code == 404
    assert test_client.put(link, json={**ok, "det_b": 10**9}).status_code == 404
    assert test_client.delete(link, params={"det_a": ids["det_a"]}).status_code == 422
    assert _decisions(test_conn) == []


def test_web_role_writes_repeat_decisions_only(client, test_conn) -> None:
    test_client, ids = client
    fam_id = _add_family(test_conn, ids)
    obj = ids["obj_both"]
    test_client.put(f"/api/object/{obj}/repeat_link", json={
        "det_a": ids["det_a"], "det_b": ids["det_b"], "decision": "SAME"})

    rw_dsn = _role_dsn(_test_dsn(), "relphot_web", "RELPHOT_WEB_PASSWORD")
    with psycopg.connect(rw_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("UPDATE relphot.repeat_decision SET decision = 'DIFFERENT', note = 'n'")
        for sql in (
            "UPDATE relphot.repeat_decision SET tc_a = 1.0",
            "UPDATE relphot.repeat_link SET linked = false",
            "DELETE FROM relphot.repeat_link",
            "UPDATE relphot.repeat_family SET accepted = true",
            "DELETE FROM relphot.repeat_family",
            "UPDATE relphot.repeat_ephemeris SET status = 'allowed'",
            "INSERT INTO relphot.repeat_family_member (fam_id, det_id) VALUES (%(f)s, 1)",
            "INSERT INTO relphot.repeat_link (det_a, det_b, obj_id, night_a, night_b, phys_ok, "
            "linked) VALUES (1, 2, %(o)s, 1, 1, true, true)",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute(sql, {"f": fam_id, "o": obj})
    assert [d[5:] for d in _decisions(test_conn)] == [("DIFFERENT", "n")]

    ro_dsn = _role_dsn(_test_dsn(), "relphot_ro", "RELPHOT_RO_PASSWORD")
    for sql in (
        "INSERT INTO relphot.repeat_decision (obj_id, night_a, night_b, tc_a, tc_b, decision) "
        "VALUES (%(o)s, %(n)s, %(n)s, 1.0, 2.0, 'SAME')",
        "UPDATE relphot.repeat_decision SET note = 'x'",
        "DELETE FROM relphot.repeat_decision",
    ):
        with (
            psycopg.connect(ro_dsn, autocommit=True) as conn,
            conn.cursor() as cur,
            pytest.raises(psycopg.errors.InsufficientPrivilege),
        ):
            cur.execute(sql, {"o": obj, "n": ids["night1"]})


def test_repeat_predict_windows_and_filters(client, test_conn) -> None:
    test_client, ids = client
    _add_family(test_conn, ids)

    # +5.52 d holds the transit of both allowed aliases (P = 1 and 0.5 d from tc0 = _TC_A)
    resp = test_client.get("/api/repeat/predict", params={"start": 2460315.3, "end": 2460315.7})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True and data["n_families"] == 1
    (w,) = data["windows"]
    assert (w["obj_id"], w["obj_name"]) == (ids["obj_both"], "RP BOTH01")
    assert (w["n_aliases"], w["n_aliases_total"], w["stale"]) == (2, 2, False)
    assert w["start_bjd"] < _TC_A + 5.0 < w["end_bjd"]
    assert w["depth"] == pytest.approx(0.01) and w["duration_display"] == "2.00 h"
    assert len(w["start_utc"]) == 16 and w["fam_id"] is not None
    assert data["start_bjd"] == pytest.approx(2460315.3)

    # +5.02 d is predicted by the 0.5 d alias only: the window that tells the aliases apart
    params = {"start": 2460314.9, "end": 2460315.1}
    (w,) = test_client.get("/api/repeat/predict", params=params).json()["windows"]
    assert (w["n_aliases"], w["n_aliases_total"]) == (1, 2)
    assert test_client.get(
        "/api/repeat/predict", params={**params, "min_alias_frac": 1.0}
    ).json()["windows"] == []

    # filters: object, telescope, accepted_only
    for extra, n in (
        ({"obj_id": ids["obj_both"]}, 1), ({"obj_id": ids["obj_unc"]}, 0),
        ({"telescope": "T80S"}, 1), ({"telescope": "ROBO43"}, 0), ({"accepted_only": "true"}, 0),
    ):
        got = test_client.get("/api/repeat/predict", params={**params, **extra}).json()
        assert len(got["windows"]) == n, extra

    # UTC dates: the end date is included whole
    utc = test_client.get(
        "/api/repeat/predict", params={"start": "2025-01-06", "end": "2025-01-06"}
    ).json()
    assert utc["end_bjd"] - utc["start_bjd"] == pytest.approx(1.0)

    # the stored family is stale once a member is rejected
    test_client.patch(f"/api/detection/{ids['det_b']}", json={"status": "REJECTED"})
    (w,) = test_client.get("/api/repeat/predict", params=params).json()["windows"]
    assert w["stale"] is True


def test_repeat_predict_defaults_and_bad_input(client) -> None:
    test_client, _ids = client
    data = test_client.get("/api/repeat/predict").json()  # now .. now + 10 d
    assert data["end_bjd"] - data["start_bjd"] == pytest.approx(10.0)
    assert data["windows"] == [] and data["n_families"] == 0
    for params in (
        {"start": "garbage"}, {"start": "2025-01-05", "end": "2025-01-01"},
        {"start": "2025-01-01", "end": "2027-01-01"}, {"min_alias_frac": 2.0},
    ):
        assert test_client.get("/api/repeat/predict", params=params).status_code in (400, 422)


def test_search_repeat_family_filter_and_column(client, test_conn) -> None:
    test_client, ids = client
    _add_family(test_conn, ids)
    rows = test_client.get("/api/search", params={"has_repeat_family": "true"}).json()["rows"]
    assert [r["obj_id"] for r in rows] == [ids["obj_both"]]
    assert rows[0]["n_repeat_families"] == 1
    without = test_client.get("/api/search", params={"has_repeat_family": "false"}).json()
    assert ids["obj_both"] not in {r["obj_id"] for r in without["rows"]}
    assert all(r["n_repeat_families"] == 0 for r in without["rows"])
    sort = test_client.get("/api/search", params={"sort": "n_repeat_families", "order": "desc"})
    assert sort.json()["rows"][0]["obj_id"] == ids["obj_both"]
    assert "n_repeat_families" in test_client.get("/api/search.csv").text.splitlines()[0]


def test_the_web_app_does_not_import_relphot_db_or_pandas() -> None:
    # the web extra has no pandas: what the repeated-event endpoints use lives in relphot.repeat
    import subprocess
    import sys

    code = (
        "import sys, relphot.web.app, relphot.repeat; "
        "bad = [m for m in ('relphot.db', 'pandas') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------
# one verdict, two views: Night reviews (EXOP) <-> the status of the night's transit events
# --------------------------------------------------------------------------


def _exop_rows(test_conn, obj_id: int) -> list[tuple]:
    test_conn.rollback()
    return _review_rows(test_conn, obj_id)


def _night_review(test_client, obj_id: int, night_id: int) -> dict | None:
    """The review the Night reviews table shows for one night (``None``: no row)."""
    night = next(
        n for n in test_client.get(f"/api/object/{obj_id}").json()["nights"]
        if n["night_id"] == night_id
    )
    if night["review_exop"] is None and night["review_var"] is None and not night["review_note"]:
        return None
    return {"exop": night["review_exop"], "var": night["review_var"], "note": night["review_note"]}


def _event_statuses(test_client, obj_id: int) -> dict[int, str]:
    events = test_client.get(f"/api/object/{obj_id}").json()["transit_events"]
    return {e["det_id"]: e["status"] for e in events}


def test_bulk_reject_all_shows_in_night_reviews_and_unconfirm_clears_it(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=3)
    night1, night2 = ids["night1"], ids["night2"]

    resp = _review(test_client, ids["det_a"], [ids["det_a"], ids["det_b"], *similar], "REJECTED")
    assert resp.status_code == 200
    reviewed = {(r["obj_id"], r["night_id"]): r["exop"] for r in resp.json()["night_reviews"]}
    assert len(reviewed) == 5  # obj_both on two nights, three look-alike objects on night 1
    assert set(reviewed.values()) == {"REJECTED"}
    # Night reviews of the viewed object ...
    assert _exop_rows(test_conn, ids["obj_both"]) == [
        (night1, "REJECTED", None, None), (night2, "REJECTED", None, None),
    ]
    assert _night_review(test_client, ids["obj_both"], night1)["exop"] == "REJECTED"
    # ... and of each look-alike's object
    with test_conn.cursor() as cur:
        cur.execute("SELECT obj_id FROM relphot.detection WHERE det_id = ANY(%s)", (similar,))
        sim_objs = [o for (o,) in cur.fetchall()]
    for obj_id in sim_objs:
        assert _exop_rows(test_conn, obj_id) == [(night1, "REJECTED", None, None)]

    # one event of the object back to UNCONFIRMED: that night has no verdict any more
    resp = _review(test_client, ids["det_b"], [ids["det_b"]], "UNCONFIRMED")
    assert resp.status_code == 200
    assert resp.json()["night_reviews"] == [
        {"obj_id": ids["obj_both"], "night_id": night2, "exop": None}
    ]
    assert _exop_rows(test_conn, ids["obj_both"]) == [(night1, "REJECTED", None, None)]
    assert _night_review(test_client, ids["obj_both"], night2) is None
    # the effective state follows: night 2 awaits review again
    night = next(
        n for n in test_client.get(f"/api/object/{ids['obj_both']}").json()["nights"]
        if n["night_id"] == night2
    )
    assert night["pending"] is True


def test_bulk_confirm_all_is_a_confirmed_exop_verdict_and_keeps_var_and_note(
    client, test_conn
) -> None:
    test_client, ids = client
    obj, night1 = ids["obj_both"], ids["night1"]
    put = test_client.put(
        f"/api/object/{obj}/night/{night1}/review", json={"var": "CONFIRMED", "note": "keep"}
    )
    assert put.status_code == 200

    assert _review(test_client, ids["det_a"], [ids["det_a"]], "CONFIRMED").status_code == 200
    assert _exop_rows(test_conn, obj) == [(night1, "CONFIRMED", "CONFIRMED", "keep")]
    assert _review(test_client, ids["det_a"], [ids["det_a"]], "UNCONFIRMED").status_code == 200
    # the verdict goes, the VAR verdict and the note (their own row content) stay
    assert _exop_rows(test_conn, obj) == [(night1, None, "CONFIRMED", "keep")]


def test_bulk_review_failure_leaves_the_night_reviews_alone(client, test_conn) -> None:
    test_client, ids = client
    similar = _add_lookalikes(test_conn, ids, n=2)
    resp = _review(test_client, ids["det_a"], [*similar, 999999], "REJECTED")
    assert resp.status_code == 404
    with test_conn.cursor() as cur:
        test_conn.rollback()
        cur.execute("SELECT count(*) FROM relphot.user_night_review")
        assert cur.fetchone()[0] == 0


def test_night_exop_verdict_sets_the_transit_events_of_that_night(client, test_conn) -> None:
    test_client, ids = client
    obj, night1, night2 = ids["obj_both"], ids["night1"], ids["night2"]
    url = f"/api/object/{obj}/night/{night1}/review"

    data = test_client.put(url, json={"exop": "REJECTED", "note": "systematics"}).json()
    assert data["events_updated"] == [ids["det_a"]]
    assert _event_statuses(test_client, obj) == {
        ids["det_a"]: "REJECTED", ids["det_b"]: "UNCONFIRMED"  # the other night is untouched
    }
    assert data["night"]["exop_effective"] is False and data["night"]["pending"] is False
    assert (data["object"]["n_nights_reviewed"], data["object"]["n_review_pending"]) == (1, 1)

    # saving the same EXOP verdict again (e.g. only the note changed) touches no event
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"notes": "by hand"})
    data = test_client.put(url, json={"exop": "REJECTED", "note": "systematics 2"}).json()
    assert data["events_updated"] == []
    assert _event_statuses(test_client, obj)[ids["det_a"]] == "REJECTED"

    # all of the night's events were REJECTED: CONFIRMED confirms them
    data = test_client.put(url, json={"exop": "CONFIRMED"}).json()
    assert data["events_updated"] == [ids["det_a"]]
    assert _event_statuses(test_client, obj)[ids["det_a"]] == "CONFIRMED"

    # back to automatic: the events that carried the verdict are open again
    data = test_client.put(url, json={"exop": None, "note": "x"}).json()
    assert data["events_updated"] == [ids["det_a"]]
    assert _event_statuses(test_client, obj) == {
        ids["det_a"]: "UNCONFIRMED", ids["det_b"]: "UNCONFIRMED"
    }
    assert _exop_rows(test_conn, obj) == [(night1, None, None, "x")]
    # deleting the review (all null, no note) is the same as clearing it
    test_client.put(url, json={"exop": "REJECTED"})
    resp = test_client.delete(url)
    assert resp.json()["events_updated"] == [ids["det_a"]]
    assert _event_statuses(test_client, obj)[ids["det_a"]] == "UNCONFIRMED"
    assert _exop_rows(test_conn, obj) == []
    assert night2  # the second night never had a verdict


def test_night_with_several_transit_events_and_mixed_verdicts(client, test_conn) -> None:
    test_client, ids = client
    obj, night1 = ids["obj_both"], ids["night1"]
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier) VALUES (%s, %s, 'transit', 8.0, 0.02, 2460310.60, 1.5, 2) "
            "RETURNING det_id",
            (obj, night1),
        )
        (det_c,) = cur.fetchone()
    test_conn.commit()
    det_a = ids["det_a"]
    url = f"/api/object/{obj}/night/{night1}/review"

    # night REJECTED rejects both events of the night
    data = test_client.put(url, json={"exop": "REJECTED"}).json()
    assert data["events_updated"] == [det_a, det_c]
    # one event reopened: the night has no verdict while some event is open
    test_client.patch(f"/api/detection/{det_c}", json={"status": "UNCONFIRMED"})
    assert _exop_rows(test_conn, obj) == []
    assert _event_statuses(test_client, obj)[det_a] == "REJECTED"
    # night CONFIRMED confirms the open event and leaves the person's rejection of the other
    data = test_client.put(url, json={"exop": "CONFIRMED"}).json()
    assert data["events_updated"] == [det_c]
    assert _event_statuses(test_client, obj) == {
        det_a: "REJECTED", det_c: "CONFIRMED", ids["det_b"]: "UNCONFIRMED"
    }
    # a CONFIRMED event keeps the night CONFIRMED however the others stand ...
    test_client.patch(f"/api/detection/{det_a}", json={"status": "UNCONFIRMED"})
    assert _exop_rows(test_conn, obj) == [(night1, "CONFIRMED", None, None)]
    # ... and clearing the night resets only the CONFIRMED one
    test_client.patch(f"/api/detection/{det_a}", json={"status": "REJECTED"})
    assert test_client.put(url, json={"exop": None}).json()["events_updated"] == [det_c]
    assert _event_statuses(test_client, obj)[det_a] == "REJECTED"
    assert _event_statuses(test_client, obj)[det_c] == "UNCONFIRMED"


def test_patch_detection_on_a_transit_event_writes_the_night_verdict_other_kinds_do_not(
    client, test_conn
) -> None:
    test_client, ids = client
    obj, night1 = ids["obj_both"], ids["night1"]
    resp = test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "REJECTED"})
    assert resp.status_code == 200
    assert _exop_rows(test_conn, obj) == [(night1, "REJECTED", None, None)]
    # a notes-only edit changes no verdict
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"notes": "n"})
    assert _exop_rows(test_conn, obj) == [(night1, "REJECTED", None, None)]
    test_client.patch(f"/api/detection/{ids['det_a']}", json={"status": "UNCONFIRMED"})
    assert _exop_rows(test_conn, obj) == []

    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT det_id, obj_id, night_id FROM relphot.detection WHERE kind = 'variable'"
        )
        var_det, var_obj, var_night = cur.fetchone()
    resp = test_client.patch(f"/api/detection/{var_det}", json={"status": "REJECTED"})
    assert resp.status_code == 200
    assert _exop_rows(test_conn, var_obj) == []  # a VAR event is not an EXOP verdict
    assert var_night


def test_object_flag_shorthand_follows_to_the_transit_events(client, test_conn) -> None:
    test_client, ids = client
    obj, night1 = ids["obj_exop"], ids["night1"]  # one transit event, on night 1
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT det_id FROM relphot.detection WHERE obj_id = %s AND kind = 'transit'", (obj,)
        )
        (det,) = cur.fetchone()

    assert test_client.patch(f"/api/object/{obj}", json={"is_exop": False}).status_code == 200
    assert _event_statuses(test_client, obj) == {det: "REJECTED"}
    assert [r[1] for r in _exop_rows(test_conn, obj)] == ["REJECTED", "REJECTED"]
    assert test_client.patch(f"/api/object/{obj}", json={"exop_source": "auto"}).status_code == 200
    assert _event_statuses(test_client, obj) == {det: "UNCONFIRMED"}
    assert _exop_rows(test_conn, obj) == []
    assert night1


def test_front_end_shows_the_transit_event_status_read_only() -> None:
    js = (resources.files("relphot.web") / "static" / "app.js").read_text()
    assert "function detectionStatusText" in js
    # the Transit events table has no selector; the Detections table keeps one for other kinds
    events = js[js.index("function renderTransitEvents"):js.index("async function patchDetection")]
    assert "detectionStatusText(ev)" in events and "detectionStatusSelect" not in events
    assert 'r.kind === "transit" ? detectionStatusText(r) : detectionStatusSelect(r)' in js
