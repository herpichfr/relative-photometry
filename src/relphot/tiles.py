"""Adaptive rectangular tiling of the master frame's pixel projection.

:func:`build_tilemap` grids the bounding box of the master ``(x, y)``
positions into ``~tile_size_px`` cells, then adaptively merges cells short of
``min_ref_candidates`` clean reference candidates (see
:func:`relphot.reference.select_candidates`) into their sparsest neighbour,
keeping every tile an axis-aligned rectangle. A tile also carries an
*extended* member set -- its core rectangle grown by ``overlap_px`` on every
side -- that reference candidates may be drawn from; every star's own light
curve, however, always comes from its *core* tile, so overlap never lets a
star contribute to two light curves.

Merge order: repeatedly pick the tile with the fewest candidates and its
sparsest neighbour (any tile touching its boundary), then merge the two. When
their direct union is not itself a rectangle -- the shared edge is shorter
than either tile, so simply combining the two would leave a notch -- the pair
is grown by repeatedly absorbing every tile overlapping the current bounding
box until the union stabilises, which is always a clean rectangle because the
tiles being absorbed already exactly partition the plane with no gaps. This
is the rectangle-merge-or-else-grow-until-one-is-possible the design calls
for, without ever fragmenting a tile. The loop stops once every tile meets
``min_ref_candidates`` or only one tile remains. A final tile below
``hard_min_ref_candidates`` raises :class:`~relphot.exceptions.TilingError`;
one between the hard and soft minimum only logs a warning.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from relphot.exceptions import TilingError

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.match import MatchedNight

logger = logging.getLogger(__name__)

__all__ = ["TileMap", "build_tilemap"]

#: A tile's rectangle in grid-cell index space: (row0, row1, col0, col1), half-open.
_Rect = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TileMap:
    """The final adaptive grid of rectangular tiles over the master (x, y) plane.

    Tile ``t`` spans ``[xmin[t], xmax[t])`` x ``[ymin[t], ymax[t])`` in master
    pixel coordinates (the outermost tile in each direction is effectively
    closed on the far edge: the grid's own max value is included).
    ``core_indices[t]``/``extended_indices[t]`` are master-star indices (into
    the arrays of the :class:`~relphot.match.MatchedNight` the tile map was
    built from); ``n_candidates[t]`` is the count of clean reference
    candidates among ``extended_indices[t]``.
    """

    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray
    core_indices: list[np.ndarray]
    extended_indices: list[np.ndarray]
    n_candidates: np.ndarray
    #: (n_stars,) tile index of each star's core tile.
    core_tile: np.ndarray

    @property
    def n_tiles(self) -> int:
        return int(self.xmin.shape[0])

    def to_csv(self, path: Path | str) -> None:
        """Write one row per tile: bounds, core/extended/candidate counts."""
        path = Path(path)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["tile", "xmin", "xmax", "ymin", "ymax", "n_core", "n_extended", "n_candidates"]
            )
            for t in range(self.n_tiles):
                writer.writerow([
                    t,
                    self.xmin[t],
                    self.xmax[t],
                    self.ymin[t],
                    self.ymax[t],
                    len(self.core_indices[t]),
                    len(self.extended_indices[t]),
                    int(self.n_candidates[t]),
                ])


def _grid_edges(lo: float, hi: float, size: float) -> np.ndarray:
    """``n + 1`` evenly spaced edges spanning ``[lo, hi]`` in steps of about ``size``."""
    span = max(hi - lo, 0.0)
    n = max(1, math.ceil(span / size)) if span > 0 else 1
    return np.linspace(lo, hi, n + 1)


def _cell_of(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Grid-cell index of each value along one axis, clipped to the valid range."""
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, edges.shape[0] - 2)


def _rect_union(a: _Rect, b: _Rect) -> _Rect:
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def _overlaps(a: _Rect, b: _Rect) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def build_tilemap(night: MatchedNight, candidate_mask: np.ndarray, settings: Settings) -> TileMap:
    """Build the adaptive tile grid for ``night``, using ``candidate_mask`` for tile sizing."""
    tile_settings = settings.tile
    x = night.x
    y = night.y
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        msg = "no stars with finite master (x, y) positions to tile"
        raise TilingError(msg)

    col_edges = _grid_edges(
        float(np.min(x[finite])), float(np.max(x[finite])), tile_settings.tile_size_px
    )
    row_edges = _grid_edges(
        float(np.min(y[finite])), float(np.max(y[finite])), tile_settings.tile_size_px
    )
    n_cols = col_edges.shape[0] - 1
    n_rows = row_edges.shape[0] - 1

    star_col = np.where(finite, _cell_of(x, col_edges), 0)
    star_row = np.where(finite, _cell_of(y, row_edges), 0)

    cell_owner = np.arange(n_rows * n_cols, dtype=np.int64).reshape(n_rows, n_cols)
    rects: dict[int, _Rect] = {
        int(cell_owner[r, c]): (r, r + 1, c, c + 1) for r in range(n_rows) for c in range(n_cols)
    }
    next_id = n_rows * n_cols

    def px_bounds(rect: _Rect) -> tuple[float, float, float, float]:
        r0, r1, c0, c1 = rect
        return (
            float(col_edges[c0]), float(col_edges[c1]), float(row_edges[r0]), float(row_edges[r1])
        )

    def extended_mask(rect: _Rect) -> np.ndarray:
        xmin, xmax, ymin, ymax = px_bounds(rect)
        pad = tile_settings.overlap_px
        return (
            finite
            & (x >= xmin - pad) & (x <= xmax + pad)
            & (y >= ymin - pad) & (y <= ymax + pad)
        )

    def n_candidates_of(rect: _Rect) -> int:
        return int(np.count_nonzero(extended_mask(rect) & candidate_mask))

    def boundary_neighbors(tid: int) -> set[int]:
        r0, r1, c0, c1 = rects[tid]
        found: set[int] = set()
        if r0 > 0:
            found.update(int(t) for t in np.unique(cell_owner[r0 - 1, c0:c1]))
        if r1 < n_rows:
            found.update(int(t) for t in np.unique(cell_owner[r1, c0:c1]))
        if c0 > 0:
            found.update(int(t) for t in np.unique(cell_owner[r0:r1, c0 - 1]))
        if c1 < n_cols:
            found.update(int(t) for t in np.unique(cell_owner[r0:r1, c1]))
        found.discard(tid)
        return found

    def rectangle_closure(seed: set[int]) -> tuple[set[int], _Rect]:
        """Grow ``seed`` by absorbing every tile overlapping its bounding box until stable."""
        members = set(seed)
        while True:
            bbox = None
            for tid in members:
                bbox = rects[tid] if bbox is None else _rect_union(bbox, rects[tid])
            grown = {t for t, r in rects.items() if _overlaps(r, bbox)}
            if grown == members:
                return members, bbox
            members = grown

    def merge_worst(tid: int) -> None:
        nonlocal next_id
        neighbors = boundary_neighbors(tid)
        if not neighbors:
            return
        chosen = min(neighbors, key=lambda n: n_candidates_of(rects[n]))
        members, bbox = rectangle_closure({tid, chosen})
        new_id = next_id
        next_id += 1
        r0, r1, c0, c1 = bbox
        cell_owner[r0:r1, c0:c1] = new_id
        for t in members:
            del rects[t]
        rects[new_id] = bbox

    min_ref = tile_settings.min_ref_candidates
    while len(rects) > 1:
        worst_id = min(rects, key=lambda tid: n_candidates_of(rects[tid]))
        if n_candidates_of(rects[worst_id]) >= min_ref:
            break
        before = len(rects)
        merge_worst(worst_id)
        if len(rects) == before:
            # No neighbour at all (should not happen for n_tiles > 1); avoid spinning.
            break

    final_ids = sorted(rects, key=lambda tid: (rects[tid][2], rects[tid][0]))
    n_tiles = len(final_ids)
    id_to_index = {tid: i for i, tid in enumerate(final_ids)}

    xmin_arr = np.empty(n_tiles)
    xmax_arr = np.empty(n_tiles)
    ymin_arr = np.empty(n_tiles)
    ymax_arr = np.empty(n_tiles)
    core_indices: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_tiles
    extended_indices: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_tiles
    n_candidates_arr = np.empty(n_tiles, dtype=np.int64)

    core_tile_owner = cell_owner[star_row, star_col]
    core_tile = np.full(night.n_stars, -1, dtype=np.int64)

    hard_min = tile_settings.hard_min_ref_candidates
    failing: list[str] = []
    for tid in final_ids:
        i = id_to_index[tid]
        rect = rects[tid]
        xmin, xmax, ymin, ymax = px_bounds(rect)
        xmin_arr[i], xmax_arr[i], ymin_arr[i], ymax_arr[i] = xmin, xmax, ymin, ymax

        core_mask = finite & (core_tile_owner == tid)
        core_tile[core_mask] = i
        core_indices[i] = np.nonzero(core_mask)[0]

        ext_mask = extended_mask(rect)
        extended_indices[i] = np.nonzero(ext_mask)[0]
        n_cand = int(np.count_nonzero(ext_mask & candidate_mask))
        n_candidates_arr[i] = n_cand

        if n_cand < hard_min:
            failing.append(f"tile {i} [{xmin:.0f}:{xmax:.0f}, {ymin:.0f}:{ymax:.0f}]: {n_cand}")
        elif n_cand < min_ref:
            logger.warning(
                "tile %d [%.0f:%.0f, %.0f:%.0f]: %d candidates, below the soft minimum %d",
                i, xmin, xmax, ymin, ymax, n_cand, min_ref,
            )

    if failing:
        msg = f"{len(failing)} tile(s) below hard_min_ref_candidates={hard_min}: " + "; ".join(
            failing
        )
        raise TilingError(msg)

    logger.info("tiling: %d tile(s) after adaptive merging", n_tiles)
    return TileMap(
        xmin=xmin_arr, xmax=xmax_arr, ymin=ymin_arr, ymax=ymax_arr,
        core_indices=core_indices, extended_indices=extended_indices,
        n_candidates=n_candidates_arr, core_tile=core_tile,
    )
