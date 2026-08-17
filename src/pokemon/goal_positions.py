"""Maps each curriculum goal (pokemon.Data.GOAL_CANDIDATES) to a concrete,
routable (x, y, map_id) tile pokemon.world_graph can path to, plus a
dependency-respecting visit order -- built for tools/render_route.py's
auto/--all mode (and any future goal-distance reward shaping that wants
"how far to the next unlockable goal").

Also resolves coordinate-triggered scripts (pokemon.map_scripts.COORD_TRIGGERS
-- derived from every map's dispatch-table script graph, see that module --
a per-map Script callback that gates a SetEvent on wXCoord/wYCoord alone, no
warp_event involved at all, e.g. PalletTown.asm's wYCoord==1 check gating
EVENT_OAK_APPEARED_IN_PALLET) for a goal *and its EVENT_GRAPH ancestors* --
confirmed in practice this matters: EVENT_FOLLOWED_OAK_INTO_LAB's own map
(Oak's Lab) only has a door-warp entrance, routing all the way around to it
even though standing on PalletTown's wYCoord==1 line (its parent
EVENT_OAK_APPEARED_IN_PALLET's trigger) reaches the same outcome in far
fewer steps via the automatic cutscene that follows. See
pokemon.map_scripts' module docstring for the detected idiom and its
caveats (same "hint, not proof" status as everything else derived from
static analysis here).

Deliberately does NOT know the exact NPC/trigger tile inside a goal's map --
event_graph.EVENT_GRAPH only records which map_id an event is set on, not a
position (see its own module docstring for why), and guessing one here would
silently misroute. Instead each goal resolves to every tile you'd actually be
standing on right after walking through any warp_event leading into that map
(pokemon.map_warps) -- real, static, unambiguous positions, at the cost of
"routes to the right building, not necessarily next to the right NPC inside
it". A map with several doors (a mansion, a gym with two entrances, ...) has
no single canonical one, so callers (pokemon.navigation) get every entrance
as a candidate and let a live BFS pick whichever is actually nearest from
the current position, rather than this module guessing one -- picking a
fixed "first found" entrance regardless of where the player currently is
routes through a needlessly distant door when a closer one exists, or
"arrives" at a door far from where the event's real trigger sits inside,
which is exactly the kind of silently-wrong routing this module's docstring
already warned against. A goal whose map has no resolvable warp entrance at
all (only reachable by walking off an outdoor map's edge -- see
tools/gen_map_warps.py's documented gap) has no tiles here and is skipped by
callers, not guessed.
"""
from __future__ import annotations

import heapq

from pokemon import event_graph as _event_graph
from pokemon import map_scripts as _map_scripts
from pokemon import map_collision as _map_collision
from pokemon import map_constants as _map_constants
from pokemon import map_warps as _map_warps

import curriculum_config as _cc

Pos = tuple[int, int, int]

# Bound on how many EVENT_GRAPH ancestor hops goal_tiles() climbs looking
# for a coordinate trigger -- generous (real chains here run 2-4 deep, see
# EVENT_GOT_STARTER's chain back to EVENT_OAK_APPEARED_IN_PALLET) but finite
# in case a stray cycle slips through AUTO_BLACKLIST_EVENTS (belt-and-
# suspenders; _coord_trigger_ancestor_tiles also tracks visited nodes).
_MAX_ANCESTOR_HOPS = 10


def _entrance_tiles(map_id: int) -> list[Pos]:
    """Every tile landed on when warping into map_id, via every inbound
    warp_event found in pokemon.map_warps.MAP_WARPS -- not just the first
    one encountered while scanning the dict (see module docstring for why
    that was wrong). Order matches MAP_WARPS' own iteration order (stable,
    but not meaningful -- callers should treat this as an unordered set of
    equally-valid entrances, e.g. by BFSing to all of them at once via
    world_graph.shortest_path_to_nearest)."""
    out: list[Pos] = []
    for _src_map_id, edges in _map_warps.MAP_WARPS.items():
        for _sx, _sy, dest_map, dx, dy in edges:
            if dest_map == map_id:
                out.append((dx, dy, map_id))
    return out


def _coord_trigger_tiles_for(event: str) -> list[Pos]:
    """This event's own pokemon.map_scripts.COORD_TRIGGERS entries,
    expanded to concrete tiles -- a None axis (only the other axis was
    checked in the source, see map_scripts' docstring) expands to every
    walkable cell along that whole row/column, since any of them satisfies
    the gate and a live BFS should pick whichever is actually nearest."""
    out: list[Pos] = []
    for map_id, x, y in _map_scripts.COORD_TRIGGERS.get(event, []):
        if x is not None and y is not None:
            out.append((x, y, map_id))
            continue
        entry = _map_constants.MAPS_BY_ID.get(map_id)
        if entry is None:
            continue
        width_cells, height_cells = entry[1] * 2, entry[2] * 2
        if y is not None:
            out.extend(
                (cx, y, map_id) for cx in range(width_cells)
                if _map_collision.is_walkable(map_id, cx, y)
            )
        elif x is not None:
            out.extend(
                (x, cy, map_id) for cy in range(height_cells)
                if _map_collision.is_walkable(map_id, x, cy)
            )
    return out


def _coord_trigger_ancestor_tiles(goal: str) -> list[Pos]:
    """Coordinate-trigger tiles for `goal`'s EVENT_GRAPH ancestors,
    transitively (NOT `goal` itself -- see goal_tiles, which handles goal's
    own coordinate trigger separately since it needs different treatment).
    Reaching an ancestor's trigger is just as valid a route to `goal` as
    reaching goal's own map (see module docstring's EVENT_FOLLOWED_OAK_INTO_LAB
    example: its own map has no coordinate trigger, but its parent
    EVENT_OAK_APPEARED_IN_PALLET's does, and triggering that is what
    actually gets you there fastest)."""
    out: list[Pos] = []
    seen: set[str] = {goal}
    info = _event_graph.EVENT_GRAPH.get(goal)
    frontier = list(info["parents"]) if info else []
    hops = 0
    while frontier and hops < _MAX_ANCESTOR_HOPS:
        next_frontier: list[str] = []
        for name in frontier:
            if name in seen:
                continue
            seen.add(name)
            out.extend(_coord_trigger_tiles_for(name))
            parent_info = _event_graph.EVENT_GRAPH.get(name)
            if parent_info:
                next_frontier.extend(parent_info["parents"])
        frontier = next_frontier
        hops += 1
    return out


def describe(goal: str) -> str:
    """Human-readable label for `goal` (curriculum_config._description)."""
    return _cc._description(goal)


def goal_tiles(goal: str) -> list[Pos]:
    """Every concrete (x, y, map_id) entrance for `goal`.

    If `goal` has its own pokemon.map_scripts.COORD_TRIGGERS entry, that precise,
    directly-verified condition is used ALONE for goal's own map -- NOT
    unioned with the coarse warp_event entrance guess (_entrance_tiles).
    Reason, confirmed in practice: EVENT_OAK_APPEARED_IN_PALLET's home map
    is Pallet Town itself, so _entrance_tiles(PALLET_TOWN) would offer
    every house's front door as an "equally valid" entrance -- but walking
    in through any of them does nothing at all for this goal, only
    standing at wYCoord==1 does (see map_scripts/this module's
    docstring). Offering the door tiles as compass targets made the agent
    repeatedly "arrive" (BFS distance 0-1) at a real, walkable location
    that could never actually satisfy the goal, silently wasting the whole
    episode with the goal parked in the candidate pool.

    Ancestor coordinate triggers (_coord_trigger_ancestor_tiles) are always
    unioned in regardless -- they're an additional, independently-reachable
    way to (transitively) trigger `goal`, not a substitute guess for it.

    [] if goal's map_id is unknown AND it has no coordinate trigger of its
    own AND none of its ancestors do either. Callers wanting live-nearest
    routing among several entrances (pokemon.navigation) should use this,
    not goal_tile.
    """
    own_coord = _coord_trigger_tiles_for(goal)
    if own_coord:
        out = list(own_coord)
    else:
        map_id = _cc._goal_map_id(goal)
        out = _entrance_tiles(map_id) if map_id is not None else []
    out.extend(_coord_trigger_ancestor_tiles(goal))
    return out


def goal_tile(goal: str) -> Pos | None:
    """Single concrete (x, y, map_id) for `goal` -- the first of goal_tiles(),
    for callers that only need *a* routable tile (e.g. tools/render_route.py's
    single-path rendering) and don't care which door. Prefer goal_tiles() for
    anything that does a live BFS and can let it pick the nearest door
    itself."""
    tiles = goal_tiles(goal)
    return tiles[0] if tiles else None


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
