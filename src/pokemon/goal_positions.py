"""Maps each curriculum goal (pokemon.Data.GOAL_CANDIDATES) to a concrete,
routable (x, y, map_id) tile pokemon.world_graph can path to, plus a
dependency-respecting visit order -- built for tools/render_route.py's
auto/--all mode (and any future goal-distance reward shaping that wants
"how far to the next unlockable goal").

Deliberately does NOT know the exact NPC/trigger tile inside a goal's map --
event_graph.EVENT_GRAPH only records which map_id an event is set on, not a
position (see its own module docstring for why), and guessing one here would
silently misroute. Instead each goal resolves to the tile you'd actually be
standing on right after walking through whatever warp_event leads into that
map (pokemon.map_warps) -- a real, static, unambiguous position, at the cost
of "routes to the right building, not necessarily next to the right NPC
inside it". A goal whose map has no resolvable warp entrance at all (only
reachable by walking off an outdoor map's edge -- see
tools/gen_map_warps.py's documented gap) has no tile here and is skipped by
callers, not guessed.
"""
from __future__ import annotations

import heapq

from pokemon import event_graph as _event_graph
from pokemon import map_warps as _map_warps

import curriculum_config as _cc

Pos = tuple[int, int, int]


def _entrance_tile(map_id: int) -> Pos | None:
    """The tile landed on when warping into map_id via the first inbound
    warp_event found in pokemon.map_warps.MAP_WARPS. Picks *a* real entrance,
    not necessarily "the" front door -- a map with several doors (e.g. a
    mansion) has no single canonical one, and any of them is an equally
    correct, actually-reachable tile to route to."""
    for _src_map_id, edges in _map_warps.MAP_WARPS.items():
        for _sx, _sy, dest_map, dx, dy in edges:
            if dest_map == map_id:
                return (dx, dy, map_id)
    return None


def describe(goal: str) -> str:
    """Human-readable label for `goal` (curriculum_config._description)."""
    return _cc._description(goal)


def goal_tile(goal: str) -> Pos | None:
    """Concrete (x, y, map_id) for `goal`, or None if its map_id is unknown
    (no EVENT_GRAPH entry) or unreachable through any warp (see module
    docstring)."""
    map_id = _cc._goal_map_id(goal)
    if map_id is None:
        return None
    return _entrance_tile(map_id)


def unlock_order(goals: list[str] | None = None) -> list[str]:
    """Topological order over `goals` (default: curriculum_config.GOAL_ORDER,
    i.e. every pokemon.Data.GOAL_CANDIDATES) respecting EVENT_GRAPH parent
    dependencies -- the same relation curriculum_config._parents_satisfied
    checks live at runtime (a goal's EVENT_GRAPH parents that are themselves
    in `goals` come first), computed once here ahead of time instead. Ties
    (goals with no ordering constraint between them) are broken by
    curriculum_config.STAGE_ORDER's stable event-bit-index order, for a
    deterministic result run to run.
    """
    pool = list(goals) if goals is not None else list(_cc.GOAL_ORDER)
    pool_set = set(pool)
    stage_index = {g: i for i, g in enumerate(_cc.STAGE_ORDER)}

    parents_of: dict[str, set[str]] = {}
    children_of: dict[str, set[str]] = {g: set() for g in pool}
    for g in pool:
        info = _event_graph.EVENT_GRAPH.get(_cc._graph_key(g))
        parents = (set(info["parents"]) & pool_set) if info else set()
        parents_of[g] = parents
        for p in parents:
            children_of.setdefault(p, set()).add(g)

    remaining = {g: len(parents_of[g]) for g in pool}
    ready = [(stage_index.get(g, 10**9), g) for g in pool if not parents_of[g]]
    heapq.heapify(ready)

    out: list[str] = []
    while ready:
        _key, g = heapq.heappop(ready)
        out.append(g)
        for child in children_of.get(g, ()):
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, (stage_index.get(child, 10**9), child))

    # Safety net, not an expected path: event_graph.AUTO_BLACKLIST_EVENTS
    # already strips cyclic events out of GOAL_CANDIDATES, so Kahn's
    # algorithm should place every goal. Anything left over (a genuine
    # dependency cycle slipping through) is appended in stable order rather
    # than silently dropped.
    if len(out) != len(pool):
        leftover = sorted(pool_set - set(out), key=lambda g: stage_index.get(g, 10**9))
        out.extend(leftover)
    return out
