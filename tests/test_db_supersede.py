"""Tests for ``detection.superseded_by`` (schema v13) against a live PostgreSQL test database.

A RERUN's transit event supersedes an older event of the SAME light curve (one object on one
night). Needs RELPHOT_TEST_DSN (see tests/test_db_schema.py's module docstring for how it is
resolved); skipped with an explicit reason when there is none. The worker's own behaviour is in
tests/test_db_reprocess.py, the reload's in tests/test_db_load_night.py, the families' in
tests/test_db_families.py and the web's in tests/test_web.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
import pytest

from relphot.db.connect import resolve_dsn
from relphot.db.schema import init_schema
from relphot.exceptions import ConfigError
from relphot.objflags import keep_transit_event, refresh_flags, sync_night_exop_from_events

T0 = 2460000.0


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


def _night(conn, label: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.night (telescope, night_date, label, source_dir, loaded_at) "
            "VALUES ('T80S', '2025-01-01', %s, %s, now()) RETURNING night_id",
            (label, f"/tmp/{label}"),
        )
        return cur.fetchone()[0]


def _star(conn, name: str, night_ids: list[int]) -> int:
    """An object with a star_night row on each of ``night_ids``."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.object (name, ra, dec, data_updated_at) "
            "VALUES (%s, 10.0, -20.0, now()) RETURNING obj_id",
            (name,),
        )
        (obj_id,) = cur.fetchone()
        for i, night_id in enumerate(night_ids):
            cur.execute(
                "INSERT INTO relphot.star_night (obj_id, night_id, star_id, tile, mag, "
                "best_aperture, n_epochs) VALUES (%s, %s, %s, 0, 15.0, 1, 100)",
                (obj_id, night_id, obj_id * 10 + i),
            )
    return obj_id


def _event(
    conn, obj_id: int, night_id: int, tc: float, *, dur: float = 2.0, origin: str = "search",
    status: str | None = None, auto_status: str | None = None, kind: str = "transit",
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.detection (obj_id, night_id, kind, snr, depth, tc_bjd_tdb, "
            "duration_h, tier, origin, status, auto_status) VALUES (%s, %s, %s, 9.0, 0.02, %s, "
            "%s, 1, %s, COALESCE(%s, 'UNCONFIRMED'), %s) RETURNING det_id",
            (obj_id, night_id, kind, tc, dur, origin, status, auto_status),
        )
        return cur.fetchone()[0]


def _links(conn) -> dict[int, int | None]:
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT det_id, superseded_by FROM relphot.detection ORDER BY det_id")
        return dict(cur.fetchall())


def _flags(conn, obj_id: int) -> tuple:
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT is_exop, exop_source, n_review_pending FROM relphot.object WHERE obj_id = %s",
            (obj_id,),
        )
        return cur.fetchone()


# --------------------------------------------------------------------------
# the database keeps a link inside one light curve
# --------------------------------------------------------------------------


def test_a_link_is_refused_across_stars_nights_kinds_and_itself(test_conn) -> None:
    n1, n2 = _night(test_conn, "n1"), _night(test_conn, "n2")
    star_a = _star(test_conn, "A", [n1, n2])
    star_b = _star(test_conn, "B", [n1, n2])
    a1 = _event(test_conn, star_a, n1, T0 + 0.10)
    a2 = _event(test_conn, star_a, n1, T0 + 0.11, origin="user")
    b1 = _event(test_conn, star_b, n1, T0 + 0.10)  # another star, same night, same tc
    a_other_night = _event(test_conn, star_a, n2, T0 + 0.10)
    var = _event(test_conn, star_a, n1, T0 + 0.10, kind="variable")
    test_conn.commit()

    def link(child: int, parent: int) -> None:
        with test_conn.cursor() as cur:
            cur.execute(
                "UPDATE relphot.detection SET superseded_by = %s WHERE det_id = %s",
                (parent, child),
            )

    for child, parent, error in (
        (a1, b1, psycopg.errors.ForeignKeyViolation),  # another star of the same night
        (b1, a1, psycopg.errors.ForeignKeyViolation),
        (a1, a_other_night, psycopg.errors.ForeignKeyViolation),  # another night of the same star
        (a1, a1, psycopg.errors.CheckViolation),
        (var, a1, psycopg.errors.CheckViolation),  # only transit events are linked
    ):
        with pytest.raises(error):
            link(child, parent)
        test_conn.rollback()
    assert set(_links(test_conn).values()) == {None}

    link(a1, a2)  # the same light curve is fine
    test_conn.commit()
    assert _links(test_conn)[a1] == a2


def test_deleting_the_superseding_event_makes_the_other_active_again(test_conn) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    old = _event(test_conn, star, n1, T0 + 0.10)
    new = _event(test_conn, star, n1, T0 + 0.11, origin="user")
    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET superseded_by = %s WHERE det_id = %s", (new, old))
    test_conn.commit()
    with test_conn.cursor() as cur:
        cur.execute("DELETE FROM relphot.detection WHERE det_id = %s", (new,))
    test_conn.commit()
    assert _links(test_conn) == {old: None}  # and obj_id / night_id were left alone


# --------------------------------------------------------------------------
# keep_transit_event
# --------------------------------------------------------------------------


def test_keep_supersedes_the_overlapping_event_swaps_back_and_spares_other_transits(
    test_conn,
) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    search = _event(test_conn, star, n1, T0 + 0.10, dur=2.0)
    rerun = _event(test_conn, star, n1, T0 + 0.11, dur=1.0, origin="user")
    other = _event(test_conn, star, n1, T0 + 0.35, dur=1.0)  # another transit of the night
    test_conn.commit()

    with test_conn.cursor() as cur:
        assert keep_transit_event(cur, rerun) == (star, n1, {search: rerun})
    test_conn.commit()
    assert _links(test_conn) == {search: rerun, rerun: None, other: None}

    with test_conn.cursor() as cur:
        assert keep_transit_event(cur, rerun) == (star, n1, {})  # nothing to change
        assert keep_transit_event(cur, search) == (star, n1, {search: None, rerun: search})
    test_conn.commit()
    assert _links(test_conn) == {search: None, rerun: search, other: None}

    with test_conn.cursor() as cur:
        with pytest.raises(LookupError):
            keep_transit_event(cur, 999999)
        assert keep_transit_event(cur, other) == (star, n1, {})  # a lone transit: no-op
    test_conn.rollback()


def test_a_second_rerun_takes_the_whole_group_over(test_conn) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    search = _event(test_conn, star, n1, T0 + 0.10)
    first = _event(test_conn, star, n1, T0 + 0.11, origin="user")
    test_conn.commit()
    with test_conn.cursor() as cur:
        keep_transit_event(cur, first)
    second = _event(test_conn, star, n1, T0 + 0.105, origin="user")
    with test_conn.cursor() as cur:
        _, _, changes = keep_transit_event(cur, second)
    test_conn.commit()
    assert changes == {search: second, first: second}
    assert _links(test_conn) == {search: second, first: second, second: None}


def test_two_stars_of_one_night_with_overlapping_events_are_never_superseded(test_conn) -> None:
    n1 = _night(test_conn, "n1")
    star_a, star_b = _star(test_conn, "A", [n1]), _star(test_conn, "B", [n1])
    a = _event(test_conn, star_a, n1, T0 + 0.10)
    b = _event(test_conn, star_b, n1, T0 + 0.10, origin="user")  # same night, same tc
    test_conn.commit()
    with test_conn.cursor() as cur:
        assert keep_transit_event(cur, b) == (star_b, n1, {})
        assert keep_transit_event(cur, a) == (star_a, n1, {})
    test_conn.commit()
    assert _links(test_conn) == {a: None, b: None}


# --------------------------------------------------------------------------
# the night's EXOP verdict follows the active events
# --------------------------------------------------------------------------


def _verdict(conn, obj_id: int, night_id: int) -> str | None:
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT exop_verdict FROM relphot.user_night_review "
            "WHERE obj_id = %s AND night_id = %s",
            (obj_id, night_id),
        )
        row = cur.fetchone()
    return None if row is None else row[0]


def test_the_night_verdict_is_derived_from_the_active_events_only(test_conn) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    search = _event(test_conn, star, n1, T0 + 0.10, status="REJECTED")
    rerun = _event(test_conn, star, n1, T0 + 0.11, origin="user")
    test_conn.commit()

    with test_conn.cursor() as cur:
        assert sync_night_exop_from_events(cur, {(star, n1)}) == [
            {"obj_id": star, "night_id": n1, "exop": None}  # both active: one is open
        ]
        keep_transit_event(cur, rerun)  # the REJECTED search event is superseded
        sync_night_exop_from_events(cur, {(star, n1)})
    test_conn.commit()
    assert _verdict(test_conn, star, n1) is None  # the RERUN event is open

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'CONFIRMED' WHERE det_id = %s", (rerun,))
        sync_night_exop_from_events(cur, {(star, n1)})
    test_conn.commit()
    assert _verdict(test_conn, star, n1) == "CONFIRMED"

    with test_conn.cursor() as cur:  # swap back: the person's old rejection stands again
        keep_transit_event(cur, search)
        sync_night_exop_from_events(cur, {(star, n1)})
    test_conn.commit()
    assert _verdict(test_conn, star, n1) == "REJECTED"
    assert _links(test_conn) == {search: None, rerun: search}


# --------------------------------------------------------------------------
# automatic evidence and the review queue
# --------------------------------------------------------------------------


def test_refresh_flags_a_rerun_stands_for_the_search_event_it_superseded(test_conn) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    search = _event(test_conn, star, n1, T0 + 0.10)
    rerun = _event(test_conn, star, n1, T0 + 0.11, origin="user")
    test_conn.commit()

    refresh_flags(test_conn, [star])
    assert _flags(test_conn, star) == (True, "auto", 1)  # the search event alone

    with test_conn.cursor() as cur:
        keep_transit_event(cur, rerun)
    refresh_flags(test_conn, [star])
    test_conn.commit()
    # the open candidate is now the RERUN event; the night keeps its evidence and its place
    assert _flags(test_conn, star) == (True, "auto", 1)

    with test_conn.cursor() as cur:
        cur.execute("UPDATE relphot.detection SET status = 'REJECTED' WHERE det_id = %s", (rerun,))
    refresh_flags(test_conn, [star])
    test_conn.commit()
    assert _flags(test_conn, star) == (False, "auto", 0)  # the superseded event is no evidence

    with test_conn.cursor() as cur:  # swapped back: the search event counts again
        keep_transit_event(cur, search)
    refresh_flags(test_conn, [star])
    test_conn.commit()
    assert _flags(test_conn, star) == (True, "auto", 1)
    assert _links(test_conn) == {search: None, rerun: search}


def test_refresh_flags_a_rerun_of_something_the_search_did_not_find_is_no_evidence(
    test_conn,
) -> None:
    n1 = _night(test_conn, "n1")
    star = _star(test_conn, "A", [n1])
    auto_rejected = _event(test_conn, star, n1, T0 + 0.10, auto_status="REJECTED")
    rerun = _event(test_conn, star, n1, T0 + 0.11, origin="user")
    test_conn.commit()
    with test_conn.cursor() as cur:
        keep_transit_event(cur, rerun)
    refresh_flags(test_conn, [star])
    test_conn.commit()
    assert _flags(test_conn, star) == (False, "auto", 0)
    # a person's CONFIRMED on the auto-rejected search event overrides, as ever
    with test_conn.cursor() as cur:
        cur.execute(
            "UPDATE relphot.detection SET status = 'CONFIRMED' WHERE det_id = %s", (auto_rejected,)
        )
    refresh_flags(test_conn, [star])
    test_conn.commit()
    assert _flags(test_conn, star)[0] is True


# --------------------------------------------------------------------------
# the one-off backfill of the README
# --------------------------------------------------------------------------


def _backfill_sql() -> str:
    readme = (Path(__file__).resolve().parents[1] / "deploy" / "README.md").read_text()
    after = readme.split("One-off supersede backfill", 1)[1]
    return re.search(r"```sql\n(.*?)```", after, flags=re.S).group(1)


def test_readme_backfill_links_the_existing_rerun_events(test_conn) -> None:
    n1, n2 = _night(test_conn, "n1"), _night(test_conn, "n2")
    star = _star(test_conn, "A", [n1, n2])
    other_star = _star(test_conn, "B", [n1, n2])
    # a night with a search event (not converged, long) and two RERUNs of one transit, all
    # REJECTED, the night REJECTED; the newest RERUN is open
    search = _event(test_conn, star, n1, T0 + 0.200, dur=2.05, status="REJECTED")
    first = _event(test_conn, star, n1, T0 + 0.162, dur=0.668, origin="user", status="REJECTED")
    second = _event(test_conn, star, n1, T0 + 0.162, dur=0.668, origin="user")
    # a night with a RERUN of a different transit than the search event, and another star's
    # RERUN with the same centre time on the first night
    far = _event(test_conn, star, n2, T0 + 0.10, dur=1.0)
    far_rerun = _event(test_conn, star, n2, T0 + 0.50, dur=1.0, origin="user")
    star_b_rerun = _event(test_conn, other_star, n1, T0 + 0.162, dur=0.668, origin="user")
    with test_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict, note) VALUES "
            "(%s, %s, 'REJECTED', NULL), (%s, %s, 'REJECTED', 'keep me')",
            (star, n1, star, n2),
        )
    test_conn.commit()

    with test_conn.cursor() as cur:
        cur.execute(_backfill_sql())
    test_conn.commit()

    assert _links(test_conn) == {
        search: second, first: second, second: None, far: None, far_rerun: None,
        star_b_rerun: None,
    }
    # the night verdict now follows the active (open) RERUN event: cleared; the other night's
    # (and its note) untouched
    assert _verdict(test_conn, star, n1) is None
    assert _verdict(test_conn, star, n2) == "REJECTED"
