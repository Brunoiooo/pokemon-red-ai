"""Live per-step "nearest reachable objective" resolver.

Glues live game state (current position, which GOAL_CANDIDATES are already
satisfied this episode) to the static data pokemon.world_graph/
pokemon.goal_positions already provide. Reachability is decided by a live
BFS (pokemon.world_graph.shortest_path_to_nearest) actually finding a
walkable path right now -- never by pokemon.event_graph's static parent
inference, which only sees wEventFlags gates and misses badge/HM/item gates
and "trigger on map A, SetEvent fires on map B" mismatches entirely (see
pokemon.goal_positions' and curriculum_config.pick_new_goal's docstrings for
the same lesson learned the hard way). A goal that isn't currently,
physically walkable to produces no result here -- never a guessed one.

Deliberately does NOT import pokemon.Data (for GOAL_CANDIDATES) at module
level: pokemon.goal_positions -> curriculum_config -> pokemon.Data is
already a real import chain, so pokemon.Data importing this module back
would be circular. Callers pass their own already-filtered candidate-goal
set instead (Data.compass_progress passes GOAL_CANDIDATES minus whatever
this episode has already satisfied), and Data.py itself only ever imports
this module lazily, inside the method, never at top level.
"""
from __future__ import annotations

from functools import lru_cache

from pokemon import goal_positions as _goal_positions
from pokemon import world_graph as _world_graph

Pos = _world_graph.Pos

DEFAULT_MAX_HOPS = 300


@lru_cache(maxsize=None)
def _tiles_for_goal(goal: str) -> tuple[Pos, ...]:
    """goal_positions.goal_tiles(goal), memoized -- static over pret/pokered
    -derived data, so a goal's resolved entrances (or () -- see goal_tiles'
    own docstring for why some goals have none) never change within a
    process. Every entrance is included, not just one -- a map with several
    doors has no single canonical one (see goal_positions' module
    docstring), so every entrance becomes its own BFS goal tile below and a
    live search picks whichever is actually nearest right now."""
    return tuple(_goal_positions.goal_tiles(goal))


def nearest_objective(
    pos: Pos, candidate_goals: frozenset[str], max_hops: int = DEFAULT_MAX_HOPS
) -> tuple[int, int, int, str] | None:
    """(dx, dy, dist, target_goal) for the nearest tile among
    ``candidate_goals`` (caller-filtered -- e.g. GOAL_CANDIDATES minus
    whatever's already satisfied this episode) that a live BFS from ``pos``
    can actually walk to within ``max_hops``, or None if none is reachable
    (no signal, never a guessed direction).

    dx/dy are the direction of the path's first hop (path[1] - path[0]), NOT
    the raw endpoint offset -- deliberately, since pos and the target tile
    don't share a coordinate frame once a path crosses onto another map
    (world_graph.neighbors' warp edges especially can jump to an arbitrary
    (x, y) on a different map_id with no geometric relationship to pos at
    all). "Which way to move right now" is always well-defined from the
    graph itself; "straight-line direction to a possibly-distant, possibly
    differently-oriented tile" is not. (0, 0) when pos already *is* the
    target tile (dist == 0) -- distinguished from "no target" by the caller
    checking this function's return isn't None (see Data.compass_progress's
    has_target flag). ``dist`` is the true hop count to the target, for
    magnitude, since dx/dy no longer carry that information.
    """
    tile_to_goal: dict[Pos, str] = {}
    for g in candidate_goals:
        for tile in _tiles_for_goal(g):
            tile_to_goal[tile] = g
    if not tile_to_goal:
        return None

    result = _world_graph.shortest_path_to_nearest(
        pos, list(tile_to_goal), max_hops=max_hops
    )
    if result is None:
        return None
    path, best_tile = result
    dist = len(path) - 1
    if dist == 0:
        dx, dy = 0, 0
    else:
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
    return dx, dy, dist, tile_to_goal[best_tile]
