"""Whole-world walk graph: per-map walkable adjacency (pokemon.map_collision)
plus the static warp graph (pokemon.map_warps) -- lets a path cross from one
map onto another through a door/staircase/cave mouth, not just move around
within a single map.

Deliberately does NOT include outdoor Route<->City edge-walks (walking off a
map's border onto an adjoining map with no warp_event involved) -- see
tools/gen_map_warps.py's module docstring for why. A route that needs to
leave a Route/City by walking off its border rather than through a warp will
not be found here today; everything reachable purely through warp_events
(buildings, floors, caves, gates) is fully covered.
"""
from __future__ import annotations

from collections import deque

from pokemon import map_collision as _map_collision
from pokemon import map_warps as _map_warps

# (x, y, map_id) -- same field order Data.get_position() returns.
Pos = tuple[int, int, int]


def neighbors(pos: Pos) -> list[Pos]:
    """4-connected walkable moves within pos's map, plus any warp leaving
    exactly this tile (see pokemon.map_warps.warps_from)."""
    x, y, map_id = pos
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if _map_collision.is_walkable(map_id, nx, ny):
            out.append((nx, ny, map_id))
    for wx, wy, dest_map, dx, dy in _map_warps.warps_from(map_id):
        if (wx, wy) == (x, y):
            out.append((dx, dy, dest_map))
    return out


def shortest_path_to_nearest(
    start: Pos, goals: list[Pos]
) -> tuple[list[Pos], Pos] | None:
    """BFS from start over the whole warp-connected walk graph.

    Returns (path including start and the reached goal, which goal was
    reached), or None if no goal is reachable at all. Since BFS explores in
    non-decreasing step order, the goal returned is always one at the
    minimum hop count; if several goals tie, the earliest one in `goals`
    wins the tie -- pass goals pre-ordered by whatever priority criteria the
    caller cares about (e.g. curriculum priority) when hop-distance ties are
    possible and matter.
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
        for nxt in neighbors(cur):
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
