"""Whole-world walk graph: per-map walkable adjacency (pokemon.map_collision)
plus the static warp graph (pokemon.map_warps) -- lets a path cross from one
map onto another through a door/staircase/cave mouth, not just move around
within a single map. Also crosses outdoor Route<->City borders (walking off
a map's edge with no warp_event involved) via pokemon.map_connections'
exact static connection offsets -- see _cross_border.
"""
from __future__ import annotations

from collections import deque

from pokemon import map_collision as _map_collision
from pokemon import map_connections as _map_connections
from pokemon import map_warps as _map_warps

# (x, y, map_id) -- same field order Data.get_position() returns.
Pos = tuple[int, int, int]


def _cross_border(map_id: int, x: int, y: int) -> Pos | None:
    """(x, y) is out-of-bounds for map_id; translate it through
    pokemon.map_connections.overworld_offsets(map_id) into whichever
    direct-connection neighbor's local coords it lands in, or None if it
    lands nowhere (an edge with no declared `connection`, or map_id isn't
    an outdoor map at all). Only
    pokemon.map_connections.direct_neighbors(map_id) is tried -- a
    *declared* connection edge, never a coincidentally-
    aligned pair from the wider transitively-connected cluster (see
    map_connections.direct_neighbors' own docstring)."""
    if _map_collision.in_bounds(map_id, x, y):
        return None
    offsets = _map_connections.overworld_offsets(map_id)
    if map_id not in offsets:
        return None
    wx, wy = offsets[map_id][0] + x, offsets[map_id][1] + y
    for nb in _map_connections.direct_neighbors(map_id):
        nb_off = offsets.get(nb)
        if nb_off is None:
            continue
        lx, ly = wx - nb_off[0], wy - nb_off[1]
        if _map_collision.in_bounds(nb, lx, ly):
            return (lx, ly, nb)
    return None


def neighbors(pos: Pos, blocked: frozenset[Pos] = frozenset()) -> list[Pos]:
    """4-connected walkable moves within pos's map (crossing onto a
    directly-connected outdoor map when a move steps off pos's map's own
    edge -- see _cross_border), plus any warp leaving exactly this tile
    (see pokemon.map_warps.warps_from).

    `blocked` (empty by default -- no behavior change for existing
    callers) excludes specific tiles regardless of their static
    map_collision walkability: pokemon.event_triggers.BLOCKED_TILES
    models an "invisible wall" idiom (e.g. Viridian City's Old Man
    shoving the player back until EVENT_GOT_POKEDEX) that's conditional on
    live event-flag state, not on anything map_collision's static tileset
    data could ever encode -- see pokemon.event_compass, the only caller
    that passes this non-empty today."""
    x, y, map_id = pos
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if _map_collision.in_bounds(map_id, nx, ny):
            if _map_collision.is_walkable(map_id, nx, ny):
                out.append((nx, ny, map_id))
            continue
        crossed = _cross_border(map_id, nx, ny)
        if crossed is not None and _map_collision.is_walkable(
            crossed[2], crossed[0], crossed[1]
        ):
            out.append(crossed)
    for wx, wy, dest_map, dx, dy in _map_warps.warps_from(map_id):
        if (wx, wy) == (x, y):
            out.append((dx, dy, dest_map))
    if blocked:
        out = [p for p in out if p not in blocked]
    return out


def shortest_path_to_nearest(
    start: Pos,
    goals: list[Pos],
    max_hops: int | None = None,
    blocked: frozenset[Pos] = frozenset(),
) -> tuple[list[Pos], Pos] | None:
    """BFS from start over the whole warp- and border-connected walk graph.

    Returns (path including start and the reached goal, which goal was
    reached), or None if no goal is reachable at all (including "not within
    max_hops", if given). Since BFS explores in non-decreasing step order,
    the goal returned is always one at the minimum hop count; if several
    goals tie, the earliest one in `goals` wins the tie -- pass goals
    pre-ordered by whatever priority criteria the caller cares about (e.g.
    curriculum priority) when hop-distance ties are possible and matter.

    `max_hops` bounds the search (unbounded by default, preserving prior
    behavior): once outdoor maps are transitively connected via
    _cross_border, an unreachable-goal (or empty-goals) query would
    otherwise walk the entire connected graph before returning None --
    callers doing frequent live lookups (e.g. a per-step navigation compass)
    should pass a bound instead of paying that cost every call.

    `blocked` is forwarded to every neighbors() call unchanged -- see its
    own docstring (empty by default, no behavior change for existing
    callers).
    """
    if not goals:
        return None
    goal_set = set(goals)
    prev: dict[Pos, Pos] = {}
    dist: dict[Pos, int] = {start: 0}
    reached: dict[Pos, int] = {}
    if start in goal_set:
        reached[start] = 0

    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if reached and dist[cur] > min(reached.values()):
            break  # BFS frontier has passed every goal already found
        if max_hops is not None and dist[cur] >= max_hops:
            continue
        for nxt in neighbors(cur, blocked):
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            prev[nxt] = cur
            if nxt in goal_set:
                reached[nxt] = dist[nxt]
            queue.append(nxt)

    if not reached:
        return None
    min_d = min(reached.values())
    best = next(g for g in goals if reached.get(g) == min_d)

    path = [best]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, best
