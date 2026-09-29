"""Tests for relphot.db.coincidence: the cross-candidate check of one night's transit events.

The pure function :func:`relphot.db.coincidence.coincidence` needs no database. The rest
(``update_coincidence``, the object flags, ``analyze``) needs RELPHOT_TEST_DSN (see
tests/test_db_schema.py's module docstring) and is skipped with an explicit reason without it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import psycopg
import pytest
from scipy.stats import binom

from relphot.config import DbSettings, Settings
from relphot.db import coincidence as coincidence_module
from relphot.db.analyze import analyze
from relphot.db.coincidence import coincidence, update_coincidence
from relphot.db.connect import resolve_dsn
from relphot.db.refresh import refresh_objects
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError
from relphot.objflags import night_state

T0 = 2460000.0

# --------------------------------------------------------------------------
# the pure function
# --------------------------------------------------------------------------


def _night_with_cluster(seed: int = 1, n_background: int = 100, n_cluster: int = 10):
    """Background events uniform in tc with random T14 / depth, plus a cluster of identical
    events at one tc, plus a lone event at the cluster time with a very different depth."""
    rng = np.random.default_rng(seed)
    tc = T0 + rng.uniform(0.0, 0.4, n_background)
    t14 = np.exp(rng.uniform(np.log(0.5), np.log(6.0), n_background))
    depth = np.exp(rng.uniform(np.log(0.002), np.log(0.05), n_background))
    t_cluster = T0 + 0.2
    tc = np.concatenate([tc, np.full(n_cluster, t_cluster), [t_cluster]])
    t14 = np.concatenate([t14, np.full(n_cluster, 2.0), [2.0]])
    depth = np.concatenate([depth, np.full(n_cluster, 0.01), [0.08]])
    err = np.full(tc.size, 1e-4)
    cluster = np.arange(n_background, n_background + n_cluster)
    lone = n_background + n_cluster
    return tc, t14, depth, err, cluster, lone


def test_a_cluster_of_identical_events_is_rejected_a_dissimilar_one_at_its_time_is_not() -> None:
    tc, t14, depth, err, cluster, lone = _night_with_cluster()
    res = coincidence(
        tc, t14, depth, err, t_first=T0, t_last=T0 + 0.4, cadence=0.002,
    )
    assert res.rejected[cluster].all()
    assert (res.n_similar[cluster] >= 9).all()
    assert (res.p_chance[cluster] < 1e-3).all()
    # the cluster's members are each other's look-alikes, nearest in time first
    for i in cluster:
        assert set(np.setdiff1d(cluster, [i])) <= set(res.similar[i])
        assert len(res.similar[i]) == res.n_similar[i]
        assert i not in res.similar[i]
    # the lone event has the cluster's time but not its depth: no look-alike at all
    assert res.n_similar[lone] == 0
    assert res.p_chance[lone] == 1.0
    assert not res.rejected[lone]
    assert res.similar[lone].size == 0
    # the uniformly placed background away from the cluster is not touched (a background event
    # of the cluster's time, depth and T14 would itself be one of its look-alikes)
    background = np.arange(cluster[0])
    far = np.abs(tc[background] - (T0 + 0.2)) > 0.05
    assert far.sum() > 50
    assert not res.rejected[background[far]].any()
    assert res.evaluated.all()


def test_similar_events_are_listed_nearest_in_time_first() -> None:
    tc = T0 + np.array([0.100, 0.1006, 0.0998, 0.1015])
    res = coincidence(
        tc, np.full(4, 2.0), np.full(4, 0.01), np.zeros(4), cadence=0.0,
        settings=replace(DbSettings(), coincidence_min_similar=1, coincidence_max_p=1.0),
    )
    # window = 0.1 * 2 h = 0.00833 d: all four are within it of the first
    assert list(res.similar[0]) == [2, 1, 3]
    assert res.n_similar[0] == 3


def test_the_time_window_is_the_largest_of_the_t14_fraction_the_tc_errors_and_a_cadence() -> None:
    loose = replace(DbSettings(), coincidence_min_similar=1, coincidence_max_p=1.0)
    t14 = np.full(2, 2.0)  # 0.1 * 2 h = 0.008333 d
    depth = np.full(2, 0.01)

    def n_similar(dt_days: float, err: float = 0.0, cadence: float = 0.0) -> int:
        res = coincidence(
            T0 + np.array([0.0, dt_days]), t14, depth, np.full(2, err), cadence=cadence,
            settings=loose,
        )
        assert res.n_similar[0] == res.n_similar[1]
        return int(res.n_similar[0])

    assert n_similar(0.008) == 1
    assert n_similar(0.009) == 0
    # 2 sigma of the combined tc error: 2 * hypot(0.004, 0.004) = 0.01131 d
    assert n_similar(0.011, err=0.004) == 1
    assert n_similar(0.012, err=0.004) == 0
    # one cadence of the night
    assert n_similar(0.019, cadence=0.02) == 1
    assert n_similar(0.021, cadence=0.02) == 0
    # a NaN tc_err counts as 0
    res = coincidence(
        T0 + np.array([0.0, 0.008]), t14, depth, np.array([np.nan, np.nan]), settings=loose
    )
    assert res.n_similar.tolist() == [1, 1]


def test_a_lower_limit_t14_is_similar_to_any_duration_up_to_the_ratio_shorter() -> None:
    loose = replace(DbSettings(), coincidence_min_similar=1, coincidence_max_p=1.0)
    tc = np.full(2, T0 + 0.1)
    depth = np.full(2, 0.01)

    def counts(t14: list[float], limit: list[bool]) -> list[int]:
        res = coincidence(tc, t14, depth, np.zeros(2), t14_lower_limit=limit, settings=loose)
        return res.n_similar.tolist()

    # 1.6 vs 1.0 h differ by more than the ratio 1.5 ...
    assert counts([1.6, 1.0], [False, False]) == [0, 0]
    # ... a limited 1.6 h (true duration >= 1.6) is not more similar to a 1.0 h one ...
    assert counts([1.6, 1.0], [True, False]) == [0, 0]
    assert counts([1.0, 1.6], [False, True]) == [0, 0]
    # ... but a limited 1.0 h may be really 1.6 h or longer, so it is similar to any longer one
    assert counts([1.0, 1.6], [True, False]) == [1, 1]
    assert counts([1.6, 1.0], [False, True]) == [1, 1]
    assert counts([1.0, 6.0], [True, False]) == [1, 1]
    # a limited duration still cannot be similar to one more than the ratio SHORTER than it
    assert counts([3.0, 1.0], [True, False]) == [0, 0]
    # the depth criterion applies as ever
    res = coincidence(
        tc, [1.0, 6.0], [0.01, 0.05], np.zeros(2), t14_lower_limit=[True, False], settings=loose
    )
    assert res.n_similar.tolist() == [0, 0]


def test_an_event_without_lookalikes_has_p_one_and_is_never_rejected() -> None:
    res = coincidence(
        T0 + np.array([0.0, 0.3]), [2.0, 2.0], [0.01, 0.01], [1e-4, 1e-4],
        t_first=T0, t_last=T0 + 0.4,
    )
    assert res.n_similar.tolist() == [0, 0]
    assert res.p_chance.tolist() == [1.0, 1.0]
    assert (res.n_expected < 0.2).all()  # M = 1 look-alike-shaped event, tiny pbar
    assert not res.rejected.any()


def test_events_without_a_usable_tc_t14_or_depth_are_not_evaluated_and_warn_nothing() -> None:
    import warnings

    tc = T0 + np.array([0.1, 0.1, 0.1, 0.1, np.nan, 0.1])
    t14 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0])
    depth = np.array([0.01, 0.01, 0.0, np.nan, 0.01, 0.01])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = coincidence(tc, t14, depth)
    assert res.evaluated.tolist() == [True, True, False, False, False, False]
    assert res.n_similar.tolist() == [1, 1, 0, 0, 0, 0]
    assert res.p_chance[2:].tolist() == [1.0] * 4
    assert not res.rejected[2:].any()
    # one event, none, or all unusable
    assert not coincidence([T0], [2.0], [0.01]).rejected.any()
    assert coincidence([], [], []).n_similar.size == 0
    assert not coincidence([np.nan, np.nan], [2.0, 2.0], [0.01, 0.01]).evaluated.any()


def test_events_all_at_one_time_with_no_frame_times_are_counted_but_never_rejected() -> None:
    res = coincidence(np.full(10, T0), np.full(10, 2.0), np.full(10, 0.01))
    assert res.n_similar.tolist() == [9] * 10  # no span to spread over: p_chance is 1
    assert res.p_chance.tolist() == [1.0] * 10
    assert not res.rejected.any()


def test_a_negative_depth_counts_as_its_magnitude() -> None:
    loose = replace(DbSettings(), coincidence_min_similar=1, coincidence_max_p=1.0)
    res = coincidence(np.full(2, T0), [2.0, 2.0], [0.01, -0.012], settings=loose,
                      t_first=T0 - 0.1, t_last=T0 + 0.1)
    assert res.n_similar.tolist() == [1, 1]


def _brute_force(tc, t14_h, depth, tc_err, limited, lo, hi, cadence, s):
    """The definition written out event by event (no vectorisation, no blocks)."""
    n = len(tc)
    out = []
    for i in range(n):
        shape_similar, near, pws = [], [], []
        for j in range(n):
            if i == j:
                continue
            ratio = t14_h[i] / t14_h[j]
            t14_ok = abs(math.log(ratio)) < math.log(s.coincidence_t14_ratio)
            if limited[i] and ratio <= s.coincidence_t14_ratio:
                t14_ok = True
            if limited[j] and 1.0 / ratio <= s.coincidence_t14_ratio:
                t14_ok = True
            depth_ok = abs(math.log(abs(depth[i]) / abs(depth[j]))) < math.log(
                s.coincidence_depth_ratio
            )
            if not (t14_ok and depth_ok):
                continue
            win = max(
                s.coincidence_tc_frac * min(t14_h[i], t14_h[j]) / 24.0,
                s.coincidence_tc_nsigma * math.hypot(tc_err[i], tc_err[j]),
                cadence,
            )
            shape_similar.append(j)
            near.append(abs(tc[i] - tc[j]) <= win)
            pws.append((min(tc[i] + win, hi) - max(tc[i] - win, lo)) / (hi - lo))
        m = len(shape_similar)
        n_i = sum(near)
        pbar = sum(pws) / m if m else 0.0
        p = float(binom.sf(n_i - 1, m, pbar)) if n_i else 1.0
        out.append((n_i, m * pbar, p))
    return out


@pytest.mark.parametrize("block", [512, 7])
def test_the_vectorised_computation_matches_the_definition_event_by_event(
    block, monkeypatch
) -> None:
    monkeypatch.setattr(coincidence_module, "_BLOCK", block)
    rng = np.random.default_rng(5)
    n = 60
    tc = T0 + rng.uniform(0.02, 0.38, n)
    tc[:12] = T0 + 0.2 + rng.normal(0.0, 0.002, 12)  # a loose cluster
    t14 = np.exp(rng.uniform(np.log(1.0), np.log(3.0), n))
    depth = np.exp(rng.uniform(np.log(0.004), np.log(0.02), n)) * rng.choice([-1.0, 1.0], n)
    err = rng.uniform(0.0, 0.003, n)
    err[3] = np.nan
    limited = rng.random(n) < 0.2
    s = DbSettings()
    res = coincidence(
        tc, t14, depth, err, limited, t_first=T0, t_last=T0 + 0.4, cadence=0.001, settings=s
    )
    expected = _brute_force(
        tc, t14, depth, np.nan_to_num(err), limited, T0 - 0.0, T0 + 0.4, 0.001, s
    )
    # the chance span widens to hold every event; here they all lie inside [T0, T0 + 0.4]
    assert res.n_similar.tolist() == [e[0] for e in expected]
    assert res.n_expected == pytest.approx([e[1] for e in expected])
    assert res.p_chance == pytest.approx([e[2] for e in expected], rel=1e-9, abs=1e-300)
    assert res.n_similar.max() >= 3  # the comparison is not vacuous
    for i, e in enumerate(expected):
        assert res.rejected[i] == (e[0] >= s.coincidence_min_similar and e[2] < s.coincidence_max_p)


def test_the_chance_span_is_widened_to_hold_every_event() -> None:
    # an event outside the stated first/last frame times still gets a probability in [0, 1]
    res = coincidence(
        T0 + np.array([-0.1, 0.0, 0.0, 0.5]), np.full(4, 2.0), np.full(4, 0.01),
        t_first=T0, t_last=T0 + 0.4,
    )
    assert np.isfinite(res.p_chance).all()
    assert ((res.p_chance >= 0) & (res.p_chance <= 1)).all()


def test_events_placed_uniformly_at_random_are_almost_never_rejected() -> None:
    # a live-like population: T14 and depth spread over a factor of a few, centre times good to
    # ~1e-3 d with a tail of poorly determined ones (5 % at 0.1 d)
    n_night, n_events = 60, 400
    rejected = 0
    for seed in range(n_night):
        rng = np.random.default_rng(1000 + seed)
        tc = T0 + rng.uniform(0.0, 0.4, n_events)
        t14 = np.exp(rng.normal(-0.17, 0.56, n_events))
        depth = np.exp(rng.normal(-2.59, 1.05, n_events))
        err = np.exp(rng.normal(np.log(1e-3), 1.0, n_events))
        err[rng.random(n_events) < 0.05] = 0.1
        res = coincidence(tc, t14, depth, err, t_first=T0, t_last=T0 + 0.4, cadence=0.002)
        rejected += int(res.rejected.sum())
    assert rejected / n_night < 0.1


# --------------------------------------------------------------------------
# the database
# --------------------------------------------------------------------------


@pytest.fixture
def test_conn():
    try:
        dsn = resolve_dsn(env_var="RELPHOT_TEST_DSN")
    except ConfigError as exc:
        pytest.skip(f"no RELPHOT_TEST_DSN available: {exc}")
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS relphot CASCADE")
    conn.commit()
    init_schema(conn)
    yield conn
    conn.close()


def _night(conn: psycopg.Connection, label: str = "20250101", frames: bool = True) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-01', %s, %s, now()) RETURNING night_id",
            (label, f"/tmp/{label}"),
        )
        (night_id,) = cur.fetchone()
        if frames:
            for k, t in enumerate(T0 + np.linspace(0.0, 0.4, 201)):
                cur.execute(
                    "INSERT INTO relphot.frame (night_id, frame_index, bjd_tdb, kept) "
                    "VALUES (%s, %s, %s, true)",
                    (night_id, k, float(t)),
                )
    return night_id


def _object(conn: psycopg.Connection, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, data_updated_at) "
            "VALUES (%s, 10.0, -20.0, now()) RETURNING obj_id",
            (name,),
        )
        (obj_id,) = cur.fetchone()
    return obj_id


def _event(
    conn: psycopg.Connection, obj_id: int, night_id: int, tc: float, t14_h: float = 2.0,
    depth: float = 0.01, *, origin: str = "search", converged: bool = True,
    status: str | None = None, tc_err: float = 1e-4, lower_limit: bool = False,
) -> int:
    """A per-night transit detection of ``obj_id`` with its trapezoid-fit row."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, origin, status) VALUES (%s, %s, 'transit', 10.0, %s, %s, %s, 1, "
            "%s, %s) RETURNING det_id",
            (obj_id, night_id, depth, tc, t14_h, origin, status),
        )
        (det_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO relphot.transit_shape (det_id, obj_id, tc, tc_err, depth, depth_err, "
            "t14_h, t14_err, t14_lower_limit, ingress_frac, chi2_red, n_points, input, converged,"
            " computed_at) VALUES (%s, %s, %s, %s, %s, 0.001, %s, 0.1, %s, 0.2, 1.0, 80, "
            "'night', %s, now())",
            (det_id, obj_id, tc, tc_err, depth, t14_h, lower_limit, converged),
        )
    return det_id


def _background(conn: psycopg.Connection, night_id: int, n: int = 60, seed: int = 2) -> list[int]:
    rng = np.random.default_rng(seed)
    dets = []
    for k in range(n):
        obj_id = _object(conn, f"bg{seed}_{k}")
        tc = float(rng.uniform(0.02, 0.26))  # nothing within 0.05 d of the clusters at 0.2
        tc = tc if tc < 0.15 else tc + 0.1
        dets.append(_event(
            conn, obj_id, night_id, T0 + tc,
            float(np.exp(rng.uniform(np.log(0.5), np.log(6.0)))),
            float(np.exp(rng.uniform(np.log(0.002), np.log(0.05)))),
        ))
    return dets


def _cluster(
    conn: psycopg.Connection, night_id: int, n: int = 8, **kwargs
) -> tuple[list[int], list[int]]:
    """``n`` objects with an event each at the same time, T14 and depth."""
    objs = [_object(conn, f"cl{k}_{night_id}") for k in range(n)]
    dets = [_event(conn, o, night_id, T0 + 0.2, 2.0, 0.01, **kwargs) for o in objs]
    return objs, dets


def _detection(conn: psycopg.Connection, det_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, auto_status, auto_reason FROM relphot.detection WHERE det_id = %s",
            (det_id,),
        )
        return cur.fetchone()


def _auto_rejected(conn: psycopg.Connection) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT det_id FROM relphot.detection WHERE auto_status = 'REJECTED'")
        return {row[0] for row in cur.fetchall()}


def test_update_coincidence_writes_rows_and_the_automatic_verdict(test_conn) -> None:
    night = _night(test_conn)
    background = _background(test_conn, night)
    objs, cluster = _cluster(test_conn, night)
    lone_obj = _object(test_conn, "lone")
    lone = _event(test_conn, lone_obj, night, T0 + 0.2, 2.0, 0.08)
    test_conn.commit()

    report = update_coincidence(test_conn, [night])
    test_conn.commit()

    assert report.n_nights == 1
    assert report.n_events == len(background) + len(cluster) + 1
    assert report.n_rejected == len(cluster)
    assert report.changed_obj_ids == sorted(objs)
    assert _auto_rejected(test_conn) == set(cluster)

    status, auto_status, reason = _detection(test_conn, cluster[0])
    assert (status, auto_status) == (None, "REJECTED")  # the person's status is never touched
    assert reason.startswith("too many similar events: 7 other events on this night within ±")
    assert " min with similar depth and T14 (expected " in reason
    assert " by chance, p=" in reason
    assert _detection(test_conn, lone) == (None, None, None)

    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.transit_coincidence")
        assert cur.fetchone() == (report.n_events,)
        cur.execute(
            "SELECT night_id, n_similar, n_expected, p_chance, similar_det_ids, rejected "
            "FROM relphot.transit_coincidence WHERE det_id = %s",
            (cluster[0],),
        )
        night_id, n_similar, n_expected, p_chance, similar, rejected = cur.fetchone()
        assert (night_id, n_similar, rejected) == (night, 7, True)
        assert set(similar) == set(cluster) - {cluster[0]}
        assert 0.0 < n_expected < 7.0
        assert 0.0 <= p_chance < 1e-3
        cur.execute(
            "SELECT n_similar, p_chance, similar_det_ids, rejected "
            "FROM relphot.transit_coincidence WHERE det_id = %s",
            (lone,),
        )
        assert cur.fetchone() == (0, 1.0, [], False)


def test_update_coincidence_is_idempotent_and_clears_a_stale_verdict(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night)
    objs, cluster = _cluster(test_conn, night)
    test_conn.commit()

    first = update_coincidence(test_conn, [night])
    test_conn.commit()
    assert first.n_rejected == len(cluster)
    snapshot = _auto_rejected(test_conn)
    second = update_coincidence(test_conn, [night, night])
    test_conn.commit()
    assert second.changed_obj_ids == []
    assert second.n_rejected == first.n_rejected
    assert _auto_rejected(test_conn) == snapshot
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.transit_coincidence")
        assert cur.fetchone() == (first.n_events,)

    # the events stop coinciding: spread them over the night; the verdict goes away
    with test_conn.cursor() as cur:
        for k, det_id in enumerate(cluster):
            cur.execute(
                "UPDATE relphot.transit_shape SET tc = %s WHERE det_id = %s",
                (T0 + 0.03 + 0.045 * k, det_id),
            )
        cur.execute(
            "UPDATE relphot.detection SET status = 'CONFIRMED' WHERE det_id = %s", (cluster[0],)
        )
    test_conn.commit()
    third = update_coincidence(test_conn, [night])
    test_conn.commit()
    assert third.n_rejected == 0
    assert third.changed_obj_ids == sorted(objs)
    assert _auto_rejected(test_conn) == set()
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.detection WHERE auto_reason IS NOT NULL")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM relphot.transit_coincidence WHERE rejected")
        assert cur.fetchone() == (0,)
    assert _detection(test_conn, cluster[0])[0] == "CONFIRMED"  # the person's verdict stays


def test_only_converged_search_transits_are_judged_or_counted(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night)
    # three look-alikes are not enough without a fourth ... which is user-origin / unconverged
    _objs, cluster = _cluster(test_conn, night, n=3)
    user_dets = [
        _event(test_conn, _object(test_conn, f"user{k}"), night, T0 + 0.2, 2.0, 0.01,
               origin="user")
        for k in range(6)
    ]
    unconverged = [
        _event(test_conn, _object(test_conn, f"unconv{k}"), night, T0 + 0.2, 2.0, 0.01,
               converged=False)
        for k in range(6)
    ]
    test_conn.commit()

    report = update_coincidence(test_conn, [night])
    test_conn.commit()
    # 3 events -> each has n = 2 < min_similar 3: nothing rejected, nothing counted from the rest
    assert report.n_rejected == 0
    assert _auto_rejected(test_conn) == set()
    with test_conn.cursor() as cur:
        cur.execute(
            "SELECT n_similar FROM relphot.transit_coincidence WHERE det_id = %s", (cluster[0],)
        )
        assert cur.fetchone() == (2,)
        cur.execute(
            "SELECT count(*) FROM relphot.transit_coincidence WHERE det_id = ANY(%s)",
            (user_dets + unconverged,),
        )
        assert cur.fetchone() == (0,)

    # more real look-alikes tip it over: the user / unconverged ones still do not count
    for k in range(5):
        _event(test_conn, _object(test_conn, f"more{k}"), night, T0 + 0.2, 2.0, 0.01)
    test_conn.commit()
    report = update_coincidence(test_conn, [night])
    assert report.n_rejected == 8
    assert not set(user_dets + unconverged) & _auto_rejected(test_conn)
    for det_id in user_dets + unconverged:
        assert _detection(test_conn, det_id)[1:] == (None, None)


def test_nights_are_judged_separately(test_conn) -> None:
    night1, night2 = _night(test_conn, "20250101"), _night(test_conn, "20250102")
    _background(test_conn, night1, seed=3)
    _background(test_conn, night2, seed=4)
    _objs1, cluster1 = _cluster(test_conn, night1, n=8)
    # the same look-alikes spread over two nights are not simultaneous
    objs2 = [_object(test_conn, f"split{k}") for k in range(4)]
    split = [_event(test_conn, o, night2, T0 + 0.05 + 0.08 * k, 2.0, 0.01)
             for k, o in enumerate(objs2)]
    test_conn.commit()
    report = update_coincidence(test_conn, [night1, night2])
    assert report.n_nights == 2
    assert _auto_rejected(test_conn) == set(cluster1)
    assert not set(split) & _auto_rejected(test_conn)


def _flags(conn: psycopg.Connection, obj_id: int) -> tuple:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, class, n_review_pending, n_nights_reviewed, depth, best_snr "
            "FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        return cur.fetchone()


def test_an_auto_rejected_event_is_no_evidence_unless_a_person_confirms_it(test_conn) -> None:
    night = _night(test_conn)
    _background(test_conn, night)
    objs, cluster = _cluster(test_conn, night)
    obj, det = objs[0], cluster[0]
    test_conn.commit()

    refresh_objects(test_conn)
    test_conn.commit()
    assert _flags(test_conn, obj) == (True, "EXOP", 1, 0, pytest.approx(0.01), 10.0)

    report = update_coincidence(test_conn, [night])
    refresh_objects(test_conn, report.changed_obj_ids)
    test_conn.commit()
    # not exoplanet evidence, not open for review, not the best transit
    assert _flags(test_conn, obj) == (False, "UNC", 0, 0, None, None)

    # a person's CONFIRMED overrides the automatic rejection ...
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'CONFIRMED' WHERE det_id = %s", (det,))
    refresh_objects(test_conn, [obj])
    test_conn.commit()
    assert _flags(test_conn, obj) == (True, "EXOP", 0, 1, pytest.approx(0.01), 10.0)
    assert _detection(test_conn, det)[1] == "REJECTED"  # ... which stays recorded

    # ... a person's REJECTED stays rejected, and UNCONFIRMED / NULL are back to the auto verdict
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED' WHERE det_id = %s", (det,))
    refresh_objects(test_conn, [obj])
    test_conn.commit()
    assert _flags(test_conn, obj)[:2] == (False, "UNC")
    for status in ("UNCONFIRMED", None):
        with test_conn.cursor() as cur:
            cur.execute("UPDATE relphot.detection SET status = %s WHERE det_id = %s", (status, det))
        refresh_objects(test_conn, [obj])
        test_conn.commit()
        assert _flags(test_conn, obj) == (False, "UNC", 0, 0, None, None)

    # a per-night verdict of the person keeps overriding, as ever
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
            "VALUES (%s, %s, 'CONFIRMED')", (obj, night),
        )
    refresh_objects(test_conn, [obj])
    test_conn.commit()
    assert _flags(test_conn, obj)[:2] == (True, "EXOP")

    # the other cluster members are unaffected by that person's decision on this one
    assert _flags(test_conn, objs[1])[:2] == (False, "UNC")

    # the event stops being auto-rejected: the evidence is back
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.user_night_review")
        cur.execute("UPDATE relphot.transit_shape SET tc = tc + 0.1 WHERE det_id = %s", (det,))
    report = update_coincidence(test_conn, [night])
    refresh_objects(test_conn, report.changed_obj_ids)
    test_conn.commit()
    assert obj in report.changed_obj_ids
    assert _flags(test_conn, obj)[:2] == (True, "EXOP")


def test_night_state_mirrors_the_auto_rejection() -> None:
    def state(status, auto_status, origin="search", kind="transit") -> dict:
        return night_state(
            [{"kind": kind, "status": status, "origin": origin, "auto_status": auto_status}],
            None, None,
        )

    plain = state("UNCONFIRMED", None)
    assert (plain["auto_exop"], plain["exop_open"], plain["pending"]) == (True, True, True)
    for status in ("UNCONFIRMED", None):
        auto = state(status, "REJECTED")
        assert (auto["auto_exop"], auto["exop_open"], auto["exop_effective"]) == (
            False, False, False
        )
        assert auto["pending"] is False
    confirmed = state("CONFIRMED", "REJECTED")
    assert (confirmed["auto_exop"], confirmed["exop_open"], confirmed["exop_effective"]) == (
        True, False, True
    )
    rejected = state("REJECTED", "REJECTED")
    assert (rejected["auto_exop"], rejected["exop_effective"]) == (False, False)
    # detections that carry no auto_status key at all read as before
    legacy = night_state([{"kind": "transit", "status": None, "origin": "search"}], None, None)
    assert legacy["auto_exop"] is True
    # another event of the night still counts
    two = night_state(
        [
            {"kind": "transit", "status": None, "origin": "search", "auto_status": "REJECTED"},
            {"kind": "transit", "status": None, "origin": "search", "auto_status": None},
        ],
        None, None,
    )
    assert (two["auto_exop"], two["exop_open"]) == (True, True)


# --------------------------------------------------------------------------
# analyze() runs the check on whole nights
# --------------------------------------------------------------------------


def _trapezoid(t, tc, t14, q, depth):
    half = 0.5 * t14
    tau = max(q * t14, 1e-4)
    return 1.0 - depth * np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _observed_event(conn: psycopg.Connection, obj_id: int, night_id: int, star_id: int, seed: int):
    """A light curve with an injected 2.4 h, 2 % transit at T0 + 0.2, plus its detection row."""
    rng = np.random.default_rng(seed)
    t = T0 + np.linspace(0.0, 0.4, 320)
    flux = _trapezoid(t, T0 + 0.2, 2.4 / 24.0, 0.2, 0.02) + rng.normal(0.0, 0.0006, t.size)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night "
            "(obj_id, night_id, star_id, tile, mag, best_aperture, rms, "
            "expected_noise, chi2_reduced, n_epochs, is_comparison) "
            "VALUES (%s, %s, %s, 0, 15.0, 1, 0.01, 0.01, 1.0, %s, false)",
            (obj_id, night_id, star_id, t.size),
        )
        cur.execute(
            "INSERT INTO relphot.lightcurve "
            "(obj_id, night_id, frame_index, bjd_tdb, flux, flux_err, flux_raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                obj_id, night_id, list(range(t.size)), [float(v) for v in t],
                [float(v) for v in flux], [0.0006] * t.size, [float(v) for v in flux],
            ),
        )
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier) VALUES (%s, %s, 'transit', 12.0, 0.02, %s, 2.4, 1) "
            "RETURNING det_id",
            (obj_id, night_id, T0 + 0.2 + 0.002),
        )
        (det_id,) = cur.fetchone()
    return det_id


def test_analyze_judges_the_whole_night_and_reports_the_count(test_conn) -> None:
    settings = replace(Settings(), db=replace(DbSettings(), max_expected_noise=0.05))
    night = _night(test_conn)
    _background(test_conn, night)
    # two look-alikes are analysed now (real light curves) ...
    targets = [_object(test_conn, f"target{k}") for k in range(2)]
    target_dets = [
        _observed_event(test_conn, obj, night, k, seed=30 + k) for k, obj in enumerate(targets)
    ]
    # ... four more sit in the database with shapes stored earlier, not analysed now
    others, other_dets = _cluster(test_conn, night, n=4)
    for det_id in other_dets:
        with test_conn.cursor() as cur:
            cur.execute("UPDATE relphot.transit_shape SET depth = 0.02, t14_h = 2.4 "
                        "WHERE det_id = %s", (det_id,))
    test_conn.commit()
    refresh_objects(test_conn)
    test_conn.commit()
    assert _flags(test_conn, others[0])[0] is True

    report = analyze(test_conn, obj_ids=targets, settings=settings, workers=1)
    test_conn.commit()

    assert report.n_transit_shapes == 2
    assert report.n_coincidence_nights == 1
    assert report.n_coincidence_rejected == 6
    assert _auto_rejected(test_conn) == set(target_dets) | set(other_dets)
    for obj in [*targets, *others]:
        assert _flags(test_conn, obj)[:2] == (False, "UNC")
    with test_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relphot.transit_coincidence WHERE rejected")
        assert cur.fetchone() == (6,)

    # analysing again is stable
    again = analyze(test_conn, obj_ids=targets, settings=settings, workers=1)
    test_conn.commit()
    assert again.n_coincidence_rejected == 6
    assert _auto_rejected(test_conn) == set(target_dets) | set(other_dets)


def test_analyze_without_transit_shapes_runs_no_check(test_conn) -> None:
    obj = _object(test_conn, "plain")
    night = _night(test_conn)
    t = T0 + np.linspace(0.0, 0.4, 60)
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.star_night "
            "(obj_id, night_id, star_id, tile, mag, best_aperture, rms, "
            "expected_noise, chi2_reduced, n_epochs, is_comparison) "
            "VALUES (%s, %s, 0, 0, 15.0, 1, 0.01, 0.01, 1.0, %s, false)",
            (obj, night, t.size),
        )
    test_conn.commit()
    report = analyze(test_conn, obj_ids=[obj], settings=Settings(), workers=1)
    assert (report.n_coincidence_nights, report.n_coincidence_rejected) == (0, 0)
