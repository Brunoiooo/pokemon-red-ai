"""Static map-to-map world-coordinate offsets, derived directly from
reference/pokered's `connection north/south/west/east` map-header data —
the exact alignment the game engine itself uses to keep wXCoord/wYCoord
continuous when walking across an outdoor map's edge into a directly
adjacent one. See reference/pokered/home/overworld.asm's
CheckMapConnections (~line 548) and each direction's
wXConnectedMap{X,Y}Alignment fields: north/south deltas the X coordinate
and resets Y to an absolute row (the far edge of the target map); west/east
is the mirror (deltas Y, resets X). The `connection` macro
(reference/pokered/macros/scripts/maps.asm) computes those alignment bytes
from the header's raw `param` (block units) as:

  north: target_x0 = current_x0 + 2*param   target_y0 = current_y0 - target_h
  south: target_x0 = current_x0 + 2*param   target_y0 = current_y0 + current_h
  west:  target_y0 = current_y0 + 2*param   target_x0 = current_x0 - target_w
  east:  target_y0 = current_y0 + 2*param   target_x0 = current_x0 + current_w

(widths/heights in cells; derivation cross-checked both directions of
several real connection pairs, e.g. Route1<->ViridianCity and
Route22<->ViridianCity, for exact +/- consistency.)

Unlike PositionHeatmap.RollingHeatmapAggregator.global_offsets (which
infers map-to-map offsets empirically from observed step deltas, because it
has no access to source data), this is exact and fully static: the whole
reachable graph is available immediately, with zero live training/eval
data. The `connection` macro is only ever used for outdoor Kanto edges —
interiors/caves/etc. exclusively warp, which isn't a geometric relationship
at all (see tools/render_route.py's module docstring) — so this naturally
leaves everything else disconnected, same as the empirical system's
side-panel fallback for maps it couldn't stitch in.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pokemon.map_constants import MAPS, MAPS_BY_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POKERED_DIR = _REPO_ROOT / "reference" / "pokered"

_MAP_HEADER_RE = re.compile(r"^\tmap_header\s+\w+,\s*(\w+),")
_CONNECTION_RE = re.compile(
    r"^\tconnection\s+(north|south|west|east),\s*\w+,\s*(\w+),\s*(-?\d+)", re.M
)


@lru_cache(maxsize=1)
def _edges() -> dict[str, list[tuple[str, str, int]]]:
    """MAP_CONST -> [(direction, target_MAP_CONST, param), ...], one entry
    per `connection` line in that map's own header (the reverse edge, if
    any, lives in the target's header, not duplicated here)."""
    edges: dict[str, list[tuple[str, str, int]]] = {}
    header_dir = _POKERED_DIR / "data" / "maps" / "headers"
    for path in sorted(header_dir.glob("*.asm")):
        text = path.read_text(encoding="utf-8")
        m = _MAP_HEADER_RE.match(text)
        if not m:
            continue
        source = m.group(1)
        for cm in _CONNECTION_RE.finditer(text):
            direction, target, param = cm.groups()
            edges.setdefault(source, []).append((direction, target, int(param)))
    return edges


def _dims_cells(map_const: str) -> tuple[int, int] | None:
    entry = MAPS.get(map_const)
    if entry is None:
        return None
    _map_id, width_blocks, height_blocks = entry
    return width_blocks * 2, height_blocks * 2


def _apply_edge(
    x0: int, y0: int, direction: str, cur_const: str, tgt_const: str, param: int
) -> tuple[int, int] | None:
    """World (x0, y0) for ``tgt_const`` given ``cur_const``'s own (x0, y0)
    — see module docstring for the formula's derivation. None if either
    map's cell dimensions can't be resolved."""
    cur_dims = _dims_cells(cur_const)
    tgt_dims = _dims_cells(tgt_const)
    if cur_dims is None or tgt_dims is None:
        return None
    cur_w, cur_h = cur_dims
    tgt_w, tgt_h = tgt_dims
    if direction == "north":
        return x0 + 2 * param, y0 - tgt_h
    if direction == "south":
        return x0 + 2 * param, y0 + cur_h
    if direction == "west":
        return x0 - tgt_w, y0 + 2 * param
    if direction == "east":
        return x0 + cur_w, y0 + 2 * param
    return None


@lru_cache(maxsize=None)
def direct_neighbors(map_id: int) -> frozenset[int]:
    """map_ids reachable from map_id by exactly one `connection` edge, in
    either direction (map_id's own header's edges, or another map's header
    that names map_id as its target) -- deliberately narrower than "any
    map sharing map_id's overworld_offsets cluster": that cluster is the
    full transitive flood-fill, so two maps can share it without a
    connection directly between them. Used by
    pokemon.world_graph._cross_border to only ever accept a border-cross
    between maps with a *declared* edge, never a coincidentally-aligned
    pair (see module docstring's no-signal-beats-wrong-signal rationale)."""
    entry = MAPS_BY_ID.get(map_id)
    if entry is None:
        return frozenset()
    map_const = entry[0]
    edges = _edges()
    out_consts = {target for _direction, target, _param in edges.get(map_const, ())}
    for src_const, src_edges in edges.items():
        if any(target == map_const for _d, target, _p in src_edges):
            out_consts.add(src_const)
    out_ids = set()
    for const in out_consts:
        tgt_entry = MAPS.get(const)
        if tgt_entry is not None:
            out_ids.add(tgt_entry[0])
    return frozenset(out_ids)


@lru_cache(maxsize=None)
def overworld_offsets(anchor_map_id: int) -> dict[int, tuple[int, int]]:
    """map_id -> (x0, y0) world cell offset for every map reachable from
    ``anchor_map_id`` by following `connection` edges (BFS/DFS,
    order-independent since every edge is a pure translation), anchor
    itself at (0, 0).

    Static and exact (see module docstring), so the full result is
    available immediately — no live episode data needed, unlike
    PositionHeatmap.RollingHeatmapAggregator.global_offsets. Empty if
    anchor_map_id has no MAPS_BY_ID entry.
    """
    anchor_entry = MAPS_BY_ID.get(anchor_map_id)
    if anchor_entry is None:
        return {}
    anchor_const = anchor_entry[0]

    offsets_by_const: dict[str, tuple[int, int]] = {anchor_const: (0, 0)}
    edges = _edges()
    frontier = [anchor_const]
    while frontier:
        cur_const = frontier.pop()
        x0, y0 = offsets_by_const[cur_const]
        for direction, tgt_const, param in edges.get(cur_const, ()):
            if tgt_const in offsets_by_const:
                continue
            result = _apply_edge(x0, y0, direction, cur_const, tgt_const, param)
            if result is None:
                continue
            offsets_by_const[tgt_const] = result
            frontier.append(tgt_const)

    out: dict[int, tuple[int, int]] = {}
    for const, offset in offsets_by_const.items():
        entry = MAPS.get(const)
        if entry is not None:
            out[entry[0]] = offset
    return out
