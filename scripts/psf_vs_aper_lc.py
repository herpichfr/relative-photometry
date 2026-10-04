#!/usr/bin/env python
"""Side-by-side light curves of one star from two photometry modes (read-only).

Plots, per (star pair, night), the production aperture photometry (DB schema ``relphot`` by
default) next to PSF photometry (schema ``test_psf`` by default). The two schemas have the same
tables but different ``obj_id`` values for the same star, so the ids are given as arguments::

    python scripts/psf_vs_aper_lc.py --aper-obj 64813 --psf-obj 64813 --night 2025-09-11
    python scripts/psf_vs_aper_lc.py --pairs /ssdsto1/data/tmp/psf_photometry/candidate_pairs.csv

``--pairs`` takes a CSV with at least ``aper_obj_id`` and ``psf_obj_id`` and optionally ``label``,
``night_date`` and ``telescope`` (the other columns are ignored). Nights of the two objects are
matched by ``(night_date, telescope)``, never by ``night_id`` or label. Every query is a SELECT on
a read-only connection; schema names are checked against an identifier whitelist and always
composed with :class:`psycopg.sql.Identifier`.

Light curves are stored per star and night as arrays in ``<schema>.lightcurve`` (BJD_TDB, flux,
flux_err; the page's own queries in ``relphot.web.app`` are reused). Fluxes are plotted over the
night's median, as the web page does. Drawn on each panel: the epochs (error bars), a binned curve,
the epochs the transit-shape fit excluded (``transit_shape.edge_clip_bjd``, grey crosses), the
night's frames that have no epoch for this star (ticks on the lower axis), and every per-night
``transit`` detection (``tc`` and its duration window) with its fitted trapezoid. Per-night
detections of any kind are annotated below each panel.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from relphot.db.connect import resolve_dsn
from relphot.exceptions import ConfigError

matplotlib.use("Agg")

DEFAULT_OUTDIR = "/ssdsto1/data/tmp/psf_photometry/plots"

#: A schema name is a plain SQL identifier; anything else is refused before it reaches a query.
_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: Epoch times of the edge-clip list match a light-curve epoch within this many days (as the web).
_CLIP_MATCH_DAYS = 2e-5

#: Per-mode colours (colour-blind safe pair) and the shared transit / binned-curve inks.
_MODE_COLOUR = {"aperture": "#1f5fa8", "PSF": "#c4501b"}
_INK = "#222222"
_MODEL = "#7a1fa2"
_MUTED = "#888888"

#: Most annotation lines drawn under one panel.
_MAX_ANNOTATIONS = 6


@dataclass(frozen=True)
class Pair:
    """One star as seen by the two modes, with the optional restrictions of a CSV row."""

    aper_obj: int
    psf_obj: int
    label: str | None = None
    night: str | None = None
    telescope: str | None = None
    gaia_id: str | None = None


@dataclass
class ModeData:
    """What one mode (one schema) holds for the star on one night."""

    kind: str
    schema: str
    obj: dict
    night: dict | None
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    flux: np.ndarray = field(default_factory=lambda: np.array([]))
    err: np.ndarray = field(default_factory=lambda: np.array([]))
    t_dropped: np.ndarray = field(default_factory=lambda: np.array([]))
    window: tuple[float, float] | None = None
    dets: list[dict] = field(default_factory=list)
    clip: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n(self) -> int:
        return int(self.t.size)


# ---------------------------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------------------------


def _check_schema_name(name: str) -> str:
    """``name`` if it is a plain identifier, else an :class:`argparse.ArgumentTypeError`."""
    if not _SCHEMA_RE.fullmatch(name):
        msg = f"invalid schema name {name!r}: letters, digits and '_' only, no leading digit"
        raise argparse.ArgumentTypeError(msg)
    return name


def _query(schema: str, template: str, **parts: sql.Composable) -> sql.Composed:
    """``template`` with ``{s}`` the quoted ``schema`` (and any further composed ``parts``)."""
    return sql.SQL(template).format(s=sql.Identifier(schema), **parts)


def _schema_exists(cur: psycopg.Cursor, schema: str) -> bool:
    cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,))
    return cur.fetchone() is not None


def _column_exists(cur: psycopg.Cursor, schema: str, table: str, column: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
        (schema, table, column),
    )
    return cur.fetchone() is not None


def fetch_object(cur: psycopg.Cursor, schema: str, obj_id: int) -> dict | None:
    """The object row (name, Gaia id) or ``None`` when ``obj_id`` is not in ``schema``."""
    cur.execute(
        _query(schema, "SELECT obj_id, name, gaia_id FROM {s}.object WHERE obj_id = %s"),
        (obj_id,),
    )
    return cur.fetchone()


def fetch_nights(cur: psycopg.Cursor, schema: str, obj_id: int) -> list[dict]:
    """The nights the object has a ``star_night`` row for (apparent mag = ``mag + zp``)."""
    cur.execute(
        _query(
            schema,
            "SELECT sn.night_id, n.label, n.telescope, n.night_date, sn.n_epochs, sn.rms, "
            "sn.mag, n.zp, n.zp_source, sn.mag + n.zp AS mag_app, sn.best_aperture "
            "FROM {s}.star_night sn JOIN {s}.night n ON n.night_id = sn.night_id "
            "WHERE sn.obj_id = %s ORDER BY n.night_date, n.night_id",
        ),
        (obj_id,),
    )
    return cur.fetchall()


def fetch_lightcurve(cur: psycopg.Cursor, schema: str, obj_id: int, night_id: int) -> dict | None:
    """The night's light-curve arrays of the object (as ``/api/object/{id}/lc``), or ``None``."""
    cur.execute(
        _query(
            schema,
            "SELECT frame_index, bjd_tdb, flux, flux_err FROM {s}.lightcurve "
            "WHERE obj_id = %s AND night_id = %s",
        ),
        (obj_id, night_id),
    )
    return cur.fetchone()


def fetch_frames(cur: psycopg.Cursor, schema: str, night_id: int) -> list[dict]:
    """The night's frames with a usable time (kept or dropped)."""
    cur.execute(
        _query(
            schema,
            "SELECT frame_index, bjd_tdb FROM {s}.frame "
            "WHERE night_id = %s AND bjd_tdb IS NOT NULL AND bjd_tdb <> 'NaN' "
            "ORDER BY frame_index",
        ),
        (night_id,),
    )
    return cur.fetchall()


def fetch_detections(cur: psycopg.Cursor, schema: str, obj_id: int, night_id: int) -> list[dict]:
    """The object's per-night detections with their transit-shape fit, if any."""
    clip = (
        sql.SQL(", ts.edge_clip_bjd")
        if _column_exists(cur, schema, "transit_shape", "edge_clip_bjd")
        else sql.SQL(", NULL::double precision[] AS edge_clip_bjd")
    )
    cur.execute(
        _query(
            schema,
            "SELECT d.det_id, d.kind, d.snr, d.depth, d.tc_bjd_tdb, d.duration_h, d.tier, "
            "d.amplitude, d.status, d.auto_status, d.superseded_by, d.duration_lower_limit, "
            "ts.tc AS ts_tc, ts.depth AS ts_depth, ts.t14_h, ts.ingress_frac, ts.converged, "
            "ts.t14_lower_limit{clip} "
            "FROM {s}.detection d LEFT JOIN {s}.transit_shape ts ON ts.det_id = d.det_id "
            "WHERE d.obj_id = %s AND d.night_id = %s "
            "ORDER BY d.tier NULLS LAST, d.snr DESC NULLS LAST, d.det_id",
            clip=clip,
        ),
        (obj_id, night_id),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------------------------


def effective_status(status: str | None, auto_status: str | None) -> str:
    """A detection's status as it counts (mirrors ``relphot.web.app._effective_status``)."""
    status = status or "UNCONFIRMED"
    if status == "UNCONFIRMED" and auto_status == "REJECTED":
        return "REJECTED (auto)"
    return status


def load_mode(
    cur: psycopg.Cursor, kind: str, schema: str, obj: dict, night: dict | None
) -> ModeData:
    """Everything one mode holds for the star on one night; empty arrays when it has no curve."""
    data = ModeData(kind=kind, schema=schema, obj=obj, night=night)
    if night is None:
        return data
    night_id = night["night_id"]
    frames = fetch_frames(cur, schema, night_id)
    if frames:
        times = np.array([f["bjd_tdb"] for f in frames], dtype=float)
        data.window = (float(times.min()), float(times.max()))
    lc = fetch_lightcurve(cur, schema, obj["obj_id"], night_id)
    if lc is not None:
        t = np.asarray(lc["bjd_tdb"], dtype=float)
        flux = np.asarray(lc["flux"], dtype=float)
        err = np.asarray(lc["flux_err"], dtype=float)
        good = np.isfinite(t) & np.isfinite(flux)
        median = float(np.median(flux[good])) if good.any() else float("nan")
        if median > 0:
            data.t, data.flux, data.err = t[good], flux[good] / median, err[good] / median
        have = {int(i) for i in lc["frame_index"]}
        missing = [f["bjd_tdb"] for f in frames if int(f["frame_index"]) not in have]
        data.t_dropped = np.sort(np.concatenate([np.array(missing, dtype=float), t[~good]]))
    data.dets = fetch_detections(cur, schema, obj["obj_id"], night_id)
    clip = [b for d in data.dets for b in (d["edge_clip_bjd"] or [])]
    data.clip = np.array(sorted(set(clip)), dtype=float)
    return data


def transit_windows(data: ModeData) -> list[tuple[float, float]]:
    """``(start, end)`` BJD_TDB of each per-night transit detection's duration window."""
    out = []
    for det in data.dets:
        if det["kind"] != "transit":
            continue
        tc = det["ts_tc"] if det["ts_tc"] is not None else det["tc_bjd_tdb"]
        hours = det["t14_h"] if det["t14_h"] is not None else det["duration_h"]
        if tc is not None and hours is not None:
            out.append((tc - hours / 48.0, tc + hours / 48.0))
    return out


def rms_stats(data: ModeData) -> tuple[float, float]:
    """``(rms, rms_out_of_transit)`` of the relative flux in ppt; NaN without epochs.

    The second value leaves out the epochs inside any transit window and equals the first when
    there is none.
    """
    if data.n < 2:
        return float("nan"), float("nan")
    rms = float(np.std(data.flux, ddof=1)) * 1e3
    inside = np.zeros(data.n, dtype=bool)
    for lo, hi in transit_windows(data):
        inside |= (data.t >= lo) & (data.t <= hi)
    if not inside.any() or data.n - int(inside.sum()) < 2:
        return rms, rms
    return rms, float(np.std(data.flux[~inside], ddof=1)) * 1e3


def bin_curve(
    t: np.ndarray, flux: np.ndarray, width_days: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean time, mean flux and standard error of the bins holding at least two epochs."""
    if t.size < 2 or width_days <= 0:
        return np.array([]), np.array([]), np.array([])
    index = np.floor((t - t.min()) / width_days).astype(int)
    xs, ys, es = [], [], []
    for k in np.unique(index):
        sel = index == k
        if sel.sum() < 2:
            continue
        xs.append(t[sel].mean())
        ys.append(flux[sel].mean())
        es.append(flux[sel].std(ddof=1) / np.sqrt(sel.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def trapezoid(t: np.ndarray, tc: float, t14_days: float, ingress_frac: float) -> np.ndarray:
    """Unit-depth trapezoid, as ``relphot.db.analyze._trapezoid_shape`` (and the web page)."""
    tau = max(ingress_frac * t14_days, 1e-3 * t14_days)
    return np.clip((0.5 * t14_days - np.abs(t - tc)) / tau, 0.0, 1.0)


def y_limits(data: ModeData) -> tuple[float, float] | None:
    """Flux range of one mode, padded: every epoch for short curves, the 0.5 - 99.5 percentile
    for long ones (>= 200 epochs); fitted dips and the edge-clipped epochs always included."""
    if data.n < 2:
        return None
    q = 0.5 if data.n >= 200 else 0.0
    lo, hi = np.percentile(data.flux, [q, 100.0 - q])
    if data.clip.size:
        hit = np.any(np.abs(data.t[:, None] - data.clip[None, :]) < _CLIP_MATCH_DAYS, axis=1)
        if hit.any():
            lo, hi = min(lo, data.flux[hit].min()), max(hi, data.flux[hit].max())
    for det in data.dets:
        depth = det["ts_depth"]
        if det["kind"] == "transit" and depth is not None and det["ts_tc"] is not None:
            lo = min(lo, 1.0 - float(depth))
    pad = 0.08 * (hi - lo) or 0.01
    return float(lo - pad), float(hi + pad)


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------


def annotation_lines(data: ModeData) -> list[str]:
    """One line per detection of the night: kind, SNR, depth, status."""
    lines = []
    for det in data.dets:
        status = effective_status(det["status"], det["auto_status"])
        if det["superseded_by"] is not None:
            status += ", superseded"
        snr = "-" if det["snr"] is None else f"{det['snr']:.1f}"
        depth = det["ts_depth"] if det["ts_depth"] is not None else det["depth"]
        if depth is not None:
            size = f"depth {100 * depth:.2f}%"
        elif det["amplitude"] is not None:
            size = f"amp {det['amplitude']:.4f}"
        else:
            size = "depth -"
        hours = det["t14_h"] if det["t14_h"] is not None else det["duration_h"]
        dur = "" if hours is None else f"  T14 {hours:.2f} h"
        lines.append(f"{det['kind']:<10} SNR {snr:>5}  {size}{dur}  {status}")
    if not lines:
        return ["no detections on this night"]
    if len(lines) > _MAX_ANNOTATIONS:
        extra = len(lines) - _MAX_ANNOTATIONS
        lines = [*lines[:_MAX_ANNOTATIONS], f"... and {extra} more"]
    return lines


def draw_panel(
    ax: plt.Axes, data: ModeData, t0: float, bin_min: float, ylim: tuple[float, float] | None
) -> None:
    """Epochs, binned curve, clipped epochs, dropped-frame ticks, transit overlays."""
    colour = _MODE_COLOUR[data.kind]
    hours = (data.t - t0) * 24.0
    if data.n:
        ax.errorbar(
            hours, data.flux, yerr=data.err, fmt="o", ms=2.5, color=colour, alpha=0.55,
            elinewidth=0.6, capsize=0, label=f"epochs (n={data.n})", zorder=2,
        )
        bx, by, be = bin_curve(data.t, data.flux, bin_min / 1440.0)
        if bx.size:
            ax.errorbar(
                (bx - t0) * 24.0, by, yerr=be, fmt="-o", ms=4, lw=1.4, color=_INK,
                mfc="white", elinewidth=1.0, capsize=0, label=f"binned ({bin_min:g} min)",
                zorder=4,
            )
        if data.clip.size:
            hit = np.any(np.abs(data.t[:, None] - data.clip[None, :]) < _CLIP_MATCH_DAYS, axis=1)
            if hit.any():
                ax.plot(
                    hours[hit], data.flux[hit], "x", ms=9, mew=2, color=_MUTED,
                    label="edge outlier (excluded from fit)", zorder=5,
                )
    else:
        ax.text(
            0.5, 0.5, "no light curve for this night", transform=ax.transAxes,
            ha="center", va="center", color=_MUTED, fontsize=11,
        )
    if data.t_dropped.size:
        ax.plot(
            (data.t_dropped - t0) * 24.0, np.zeros(data.t_dropped.size), "|", ms=7, color=_MUTED,
            transform=ax.get_xaxis_transform(), label="frames without epoch", zorder=1,
        )
    seen: set[str] = set()
    for det in data.dets:
        if det["kind"] != "transit":
            continue
        tc = det["ts_tc"] if det["ts_tc"] is not None else det["tc_bjd_tdb"]
        dur_h = det["t14_h"] if det["t14_h"] is not None else det["duration_h"]
        if tc is None:
            continue
        sup = det["superseded_by"] is not None
        col = _MUTED if sup else _MODEL
        x_tc = (tc - t0) * 24.0
        ax.axvline(
            x_tc, color=col, ls="--", lw=1.0, zorder=3,
            label=None if "tc" in seen else "transit tc",
        )
        seen.add("tc")
        if dur_h is not None:
            ax.axvspan(
                x_tc - dur_h / 2.0, x_tc + dur_h / 2.0, color=col, alpha=0.10, lw=0,
                label=None if "win" in seen else "T14 window", zorder=0,
            )
            seen.add("win")
        if det["ts_tc"] is not None and None not in (
            det["t14_h"], det["ingress_frac"], det["ts_depth"],
        ):
            t14 = det["t14_h"] / 24.0
            grid = det["ts_tc"] + np.linspace(-1.5, 1.5, 241) * t14
            model = 1.0 - det["ts_depth"] * trapezoid(
                grid, det["ts_tc"], t14, det["ingress_frac"]
            )
            ax.plot(
                (grid - t0) * 24.0, model, color=col, lw=1.8, zorder=6,
                label=None if "fit" in seen else "trapezoid fit",
            )
            seen.add("fit")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"hours since first frame of night (BJD_TDB {t0:.5f})")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)


def mag_text(night: dict | None) -> str:
    """Apparent magnitude (``mag + zp``) and the zero-point source, e.g. ``12.34 (zp measured)``."""
    if night is None or night["mag_app"] is None:
        return "-"
    return f"{night['mag_app']:.2f} (zp {night['zp_source']})"


def panel_title(data: ModeData, rms: float, rms_oot: float) -> str:
    mag = mag_text(data.night)
    rms_txt = "rms -"
    if np.isfinite(rms):
        rms_txt = f"rms {rms:.2f} ppt"
        if np.isfinite(rms_oot) and abs(rms_oot - rms) > 0.005:
            rms_txt += f" (out of transit {rms_oot:.2f})"
    return (
        f"{data.kind} [{data.schema}]  obj {data.obj['obj_id']}  {data.obj['name']}\n"
        f"mag {mag}  n = {data.n}  {rms_txt}"
    )


def make_figure(
    aper: ModeData,
    psf: ModeData,
    title: str,
    *,
    bin_min: float,
    shared_y: bool,
) -> tuple[plt.Figure, float, float]:
    """The two-column figure; returns it with the rms (ppt) of the aperture and PSF curves."""
    starts = [d.window[0] for d in (aper, psf) if d.window]
    starts += [float(d.t.min()) for d in (aper, psf) if d.n]
    t0 = min(starts) if starts else 0.0
    ends = [d.window[1] for d in (aper, psf) if d.window]
    ends += [float(d.t.max()) for d in (aper, psf) if d.n]

    fig = plt.figure(figsize=(15, 6.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[5.0, 1.0], hspace=0.40, wspace=0.16)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_p = fig.add_subplot(grid[0, 1], sharex=ax_a)

    limits = [y_limits(aper), y_limits(psf)]
    if shared_y:
        found = [lim for lim in limits if lim is not None]
        if found:
            limits = [(min(lo for lo, _ in found), max(hi for _, hi in found))] * 2
    stats = []
    for ax, data, lim in zip((ax_a, ax_p), (aper, psf), limits, strict=True):
        draw_panel(ax, data, t0, bin_min, lim)
        rms, rms_oot = rms_stats(data)
        stats.append(rms)
        ax.set_title(panel_title(data, rms, rms_oot), fontsize=9, loc="left", color=_INK)
    ax_a.set_ylabel("relative flux (flux / night median)")
    if starts and ends:
        pad = 0.02 * (max(ends) - min(starts)) * 24.0
        ax_a.set_xlim((min(starts) - t0) * 24.0 - pad, (max(ends) - t0) * 24.0 + pad)
    for col, data in enumerate((aper, psf)):
        text_ax = fig.add_subplot(grid[1, col])
        text_ax.axis("off")
        text_ax.text(
            0.0, 1.0, "\n".join(annotation_lines(data)), va="top", ha="left", fontsize=7.5,
            family="monospace", color=_INK, transform=text_ax.transAxes,
        )
    fig.suptitle(title, fontsize=11, color=_INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.07)
    return fig, stats[0], stats[1]


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def load_pairs(path: Path) -> list[Pair]:
    """The pairs of a CSV with ``aper_obj_id``, ``psf_obj_id`` and optional extra columns."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"aper_obj_id", "psf_obj_id"} - set(reader.fieldnames or [])
        if missing:
            msg = f"{path}: missing column(s) {', '.join(sorted(missing))}"
            raise SystemExit(msg)
        pairs = []
        for lineno, row in enumerate(reader, start=2):
            try:
                aper, psf = _int_or_none(row["aper_obj_id"]), _int_or_none(row["psf_obj_id"])
            except ValueError as exc:
                msg = f"{path}:{lineno}: obj ids must be integers ({exc})"
                raise SystemExit(msg) from exc
            if aper is None or psf is None:
                print(f"{path}:{lineno}: skipped, empty aper_obj_id / psf_obj_id", file=sys.stderr)
                continue
            pairs.append(
                Pair(
                    aper_obj=aper, psf_obj=psf, label=_clean(row.get("label")),
                    night=_clean(row.get("night_date")), telescope=_clean(row.get("telescope")),
                    gaia_id=_clean(row.get("gaia_source_id")),
                )
            )
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot aperture and PSF light curves of the same star side by side.",
    )
    ap.add_argument("--aper-obj", type=int, help="obj_id of the star in the aperture schema")
    ap.add_argument("--psf-obj", type=int, help="obj_id of the star in the PSF schema")
    ap.add_argument(
        "--pairs", type=Path,
        help="CSV with columns aper_obj_id, psf_obj_id and optionally label, night_date, telescope",
    )
    ap.add_argument(
        "--night", help="restrict to one night: YYYY-MM-DD, YYYYMMDD or a night label "
        "(default: every night of the object)",
    )
    ap.add_argument("--aper-schema", type=_check_schema_name, default="relphot")
    ap.add_argument("--psf-schema", type=_check_schema_name, default="test_psf")
    ap.add_argument("--dsn", help="PostgreSQL DSN (default: relphot.db.connect.resolve_dsn)")
    ap.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR))
    ap.add_argument("--format", choices=("png", "pdf"), default="png")
    ap.add_argument("--shared-y", action="store_true", help="one y-range for both panels")
    ap.add_argument("--bin-min", type=float, default=10.0, help="bin width in minutes")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args(argv)
    if args.pairs is None and (args.aper_obj is None or args.psf_obj is None):
        ap.error("give --aper-obj and --psf-obj, or --pairs CSV")
    if args.pairs is not None and (args.aper_obj is not None or args.psf_obj is not None):
        ap.error("--pairs cannot be combined with --aper-obj / --psf-obj")
    return args


def night_matches(spec: str, night: dict) -> bool:
    """Whether a ``--night`` / CSV ``night_date`` value names this night (date or label)."""
    spec = spec.strip()
    iso = night["night_date"].isoformat()
    return spec in {night["label"], iso, iso.replace("-", "")}


def _by_key(nights: list[dict], who: str) -> dict[tuple[date, str], dict]:
    """Nights keyed by ``(night_date, telescope)``; the first of a duplicate key is kept."""
    keyed: dict[tuple[date, str], dict] = {}
    for night in nights:
        key = (night["night_date"], night["telescope"])
        if key in keyed:
            print(
                f"warning: {who}: several nights for {key[1]} {key[0]}; "
                f"using {keyed[key]['label']}",
                file=sys.stderr,
            )
            continue
        keyed[key] = night
    return keyed


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", text)


def process_pair(
    cur: psycopg.Cursor, pair: Pair, args: argparse.Namespace, objects: dict[tuple[str, int], dict]
) -> tuple[int, int]:
    """Draw every matching night of one pair; returns ``(n_figures, n_problems)``."""
    aper_obj = objects[(args.aper_schema, pair.aper_obj)]
    psf_obj = objects[(args.psf_schema, pair.psf_obj)]
    spec = pair.night or args.night
    nights_a = fetch_nights(cur, args.aper_schema, pair.aper_obj)
    nights_p = fetch_nights(cur, args.psf_schema, pair.psf_obj)
    if spec:
        nights_a = [n for n in nights_a if night_matches(spec, n)]
        nights_p = [n for n in nights_p if night_matches(spec, n)]
    if pair.telescope:
        nights_a = [n for n in nights_a if n["telescope"] == pair.telescope]
        nights_p = [n for n in nights_p if n["telescope"] == pair.telescope]
    by_a = _by_key(nights_a, f"{args.aper_schema} obj {pair.aper_obj}")
    by_p = _by_key(nights_p, f"{args.psf_schema} obj {pair.psf_obj}")
    keys = sorted(set(by_a) | set(by_p), key=lambda k: (k[0], k[1]))
    name = pair.label or f"aper{pair.aper_obj}_psf{pair.psf_obj}"
    if not keys:
        where = f" on night {spec}" if spec else ""
        print(
            f"error: {name}: no night found{where} (aperture obj "
            f"{pair.aper_obj} in {args.aper_schema}, PSF obj {pair.psf_obj} in {args.psf_schema})",
            file=sys.stderr,
        )
        return 0, 1
    gaia_a, gaia_p = aper_obj["gaia_id"], psf_obj["gaia_id"]
    gaia = gaia_a or pair.gaia_id or gaia_p or "-"
    if gaia_a and gaia_p and gaia_a != gaia_p:
        gaia = f"{gaia_a} (PSF object: {gaia_p})"
    n_fig = 0
    for key in keys:
        night_date, telescope = key
        aper = load_mode(cur, "aperture", args.aper_schema, aper_obj, by_a.get(key))
        psf = load_mode(cur, "PSF", args.psf_schema, psf_obj, by_p.get(key))
        mag = mag_text(aper.night if aper.night is not None else psf.night)
        title = (
            f"{name}   obj {pair.aper_obj} (aperture) / {pair.psf_obj} (PSF)   Gaia {gaia}   "
            f"mag {mag}\nnight {night_date.isoformat()}  {telescope}"
        )
        fig, rms_a, rms_p = make_figure(
            aper, psf, title, bin_min=args.bin_min, shared_y=args.shared_y
        )
        stem = f"{_safe(name)}_{night_date.isoformat()}_{_safe(telescope)}"
        path = args.outdir / f"{stem}.{args.format}"
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(
            f"{path}  n_epochs aper={aper.n} psf={psf.n}  "
            f"rms_ppt aper={rms_a:.2f} psf={rms_p:.2f}"
        )
        n_fig += 1
    return n_fig, 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = (
        load_pairs(args.pairs)
        if args.pairs is not None
        else [Pair(aper_obj=args.aper_obj, psf_obj=args.psf_obj)]
    )
    if not pairs:
        print("error: no pairs to plot", file=sys.stderr)
        return 2
    try:
        dsn = resolve_dsn(args.dsn)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        print(f"error: cannot connect to the database: {exc}", file=sys.stderr)
        return 2
    n_fig = n_problem = 0
    with conn:
        conn.read_only = True
        with conn.cursor(row_factory=dict_row) as cur:
            for schema in (args.aper_schema, args.psf_schema):
                if not _schema_exists(cur, schema):
                    print(f"error: schema {schema!r} does not exist (yet)", file=sys.stderr)
                    return 2
            objects: dict[tuple[str, int], dict] = {}
            missing = []
            for pair in pairs:
                for schema, obj_id in (
                    (args.aper_schema, pair.aper_obj), (args.psf_schema, pair.psf_obj),
                ):
                    if (schema, obj_id) in objects or (schema, obj_id) in missing:
                        continue
                    obj = fetch_object(cur, schema, obj_id)
                    if obj is None:
                        missing.append((schema, obj_id))
                    else:
                        objects[(schema, obj_id)] = obj
            if missing:
                for schema, obj_id in missing:
                    print(
                        f"error: obj_id {obj_id} does not exist in schema {schema}",
                        file=sys.stderr,
                    )
                return 2
            for pair in pairs:
                made, problems = process_pair(cur, pair, args, objects)
                n_fig += made
                n_problem += problems
    print(f"{n_fig} figure(s) written to {args.outdir}", file=sys.stderr)
    return 1 if n_problem else 0


if __name__ == "__main__":
    sys.exit(main())
