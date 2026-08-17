"""Live "nearest currently-unlockable event" resolver -- the compass engine
Data.compass_progress() feeds the model with (imported lazily from inside
that method; see its own docstring for why). Replaced the older
pokemon.navigation/pokemon.goal_positions BFS over a curated
Data.GOAL_CANDIDATES whitelist, which is retired (pokemon.navigation.py
deleted as dead code once this took over its only caller;
pokemon.goal_positions.py stays -- tools/render_route.py's AUTO mode still
uses it for route rendering, an offline visualization, not training).

Where the old pokemon.navigation.nearest_objective only routed to that
small, curated whitelist, this routes to *any* of the ~386 events
pokemon.event_triggers.EVENT_TRIGGERS (tools/gen_event_triggers.py's
exhaustive interaction scan) knows a real, non-no-op trigger for -- filtered
live to just the ones whose own flag isn't set yet and whose `requires`
guards currently hold (e.g. Pallet Town's EVENT_OAK_APPEARED_IN_PALLET only
qualifies while EVENT_FOLLOWED_OAK_INTO_LAB is NOT set -- see event_triggers'
module docstring).

Deliberately does not import pokemon.Data or pokemon.curriculum_config
(pokemon.event_triggers is pure generated data with no imports of its own,
so unlike the old goal_positions-based compass there's no real circular-
import risk here -- Data.compass_progress still imports this module lazily
anyway, for consistency with that established pattern and to avoid paying
the import cost anywhere compass_progress is never called).
"""
from __future__ import annotations

from functools import lru_cache

from pokemon import event_constants as _event_constants
from pokemon import event_triggers as _event_triggers
from pokemon import map_collision as _map_collision
from pokemon import map_constants as _map_constants
from pokemon import map_warps as _map_warps
from pokemon import world_graph as _world_graph

Pos = _world_graph.Pos

DEFAULT_MAX_HOPS = 300

WOBTAINEDBADGES_ADDR = 0xD356
# constants/ram_constants.asm's own bit order -- fixed, stable Gen 1 data
# (matches Data.badges()' bits_extractor over the same byte), reproduced
# here rather than imported to keep this module standalone (see its own
# docstring).
BADGE_BITS: dict[str, int] = {
    "BIT_BOULDERBADGE": 0,
    "BIT_CASCADEBADGE": 1,
    "BIT_THUNDERBADGE": 2,
    "BIT_RAINBOWBADGE": 3,
    "BIT_SOULBADGE": 4,
    "BIT_MARSHBADGE": 5,
    "BIT_VOLCANOBADGE": 6,
    "BIT_EARTHBADGE": 7,
}


def event_flag_set(event_name: str, memory) -> bool:
    """Direct wEventFlags bit read -- same formula as Data.
    is_event_satisfied, reimplemented locally (see module docstring)."""
    if event_name not in _event_constants.EVENTS:
        return False
    addr = _event_constants.event_address(event_name)
    bit = _event_constants.event_bit(event_name)
    return bool(memory[addr] & (1 << bit))


def has_all_badges_except(bit_name: str, memory) -> bool:
    """True once every badge EXCEPT bit_name is obtained -- the "closed
    until every OTHER badge" idiom (see event_triggers.badge_blocking_
    tiles' docstring: today, only Viridian Gym's own front door)."""
    bit = BADGE_BITS.get(bit_name)
    if bit is None:
        return False
    mask = 0xFF & ~(1 << bit)
    return (memory[WOBTAINEDBADGES_ADDR] & mask) == mask


def requires_satisfied(requires: tuple[tuple[str, str, bool], ...], memory) -> bool:
    """requires is a tuple of ("event", name, required_bool) guards -- the
    same shape pokered_asm.analyze_function_event_guards/pokemon.map_scripts
    already use, kept as-is rather than stripped to a bare (name, bool) pair
    so a future coord-guard addition here wouldn't need a shape change."""
    return all(
        event_flag_set(name, memory) == required for _kind, name, required in requires
    )


def _axis_tiles(map_id: int, x: int | None, y: int | None) -> list[Pos]:
    """A "coord"-kind TriggerSite with one axis None (only the other axis
    was checked in the source) expands to every walkable cell along that
    whole row/column -- any of them satisfies the gate. Mirrors
    pokemon.goal_positions._coord_trigger_tiles_for's identical logic,
    duplicated here rather than imported to keep this module standalone
    (see its own docstring)."""
    if x is not None and y is not None:
        return [(x, y, map_id)]
    entry = _map_constants.MAPS_BY_ID.get(map_id)
    if entry is None:
        return []
    width_cells, height_cells = entry[1] * 2, entry[2] * 2
    if y is not None:
        return [
            (cx, y, map_id) for cx in range(width_cells)
            if _map_collision.is_walkable(map_id, cx, y)
        ]
    if x is not None:
        return [
            (x, cy, map_id) for cy in range(height_cells)
            if _map_collision.is_walkable(map_id, x, cy)
        ]
    return []


def _adjacent_walkable_tiles(map_id: int, x: int, y: int) -> list[Pos]:
    """An "object"-kind site's own tile is solid (an NPC/item/trainer
    sprite blocks it) -- resolve to whichever orthogonal neighbor(s) are
    actually walkable, same approach as gen_map_scripts.py's identically-
    named helper."""
    return [
        (nx, ny, map_id)
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        if _map_collision.is_walkable(map_id, nx, ny)
    ]


def _entrance_tiles(map_id: int) -> list[Pos]:
    """Every tile landed on when warping into map_id, via every inbound
    warp_event in pokemon.map_warps.MAP_WARPS -- for an "entrance"-kind
    site (tools/gen_event_triggers.py's state_machine_triggers: a map's
    dispatch-table state 0 with no coordinate guard of its own, so its
    whole automatic forward chain fires the instant the player walks onto
    the map, no specific tile required). Same approach as
    pokemon.goal_positions._entrance_tiles, duplicated here rather than
    imported to keep this module standalone (see its own docstring)."""
    return [
        (dx, dy, map_id)
        for _src_map_id, edges in _map_warps.MAP_WARPS.items()
        for _sx, _sy, dest_map, dx, dy in edges
        if dest_map == map_id
    ]


@lru_cache(maxsize=None)
def _site_tiles(map_id: int, x: int | None, y: int | None, kind: str) -> tuple[Pos, ...]:
    """Every routable tile for one TriggerSite -- memoized (keyed by its own
    static fields, hashable unlike the source dict), static over
    pret/pokered-derived data (map geometry/collision never change within a
    process)."""
    if kind == "coord":
        return tuple(_axis_tiles(map_id, x, y))
    if kind == "object":
        return tuple(_adjacent_walkable_tiles(map_id, x, y))
    if kind == "entrance":
        return tuple(_entrance_tiles(map_id))
    # "walkable" (bg_event/hidden_event): the player stands directly on it
    # by convention; fall back to an adjacent tile if collision data
    # disagrees, same safety net as gen_map_scripts.py's _bg_event_tiles.
    if _map_collision.is_walkable(map_id, x, y):
        return ((x, y, map_id),)
    return tuple(_adjacent_walkable_tiles(map_id, x, y))


def unlockable_events(memory) -> list[str]:
    """Every event with at least one currently-satisfiable TriggerSite
    whose own flag isn't set yet -- the live candidate pool
    nearest_unlockable_event searches over."""
    out = []
    for event_name, sites in _event_triggers.EVENT_TRIGGERS.items():
        if event_flag_set(event_name, memory):
            continue
        if any(requires_satisfied(site["requires"], memory) for site in sites):
            out.append(event_name)
    return out


def tiles_for_site(site: dict) -> tuple[Pos, ...]:
    return _site_tiles(site["map_id"], site["x"], site["y"], site["kind"])


_STATIC_OBSTACLES: frozenset[Pos] = frozenset(
    (x, y, map_id) for map_id, x, y in _event_triggers.STATIC_OBSTACLES
)


def currently_blocked_tiles(memory) -> frozenset[Pos]:
    """Every currently-impassable tile the live BFS shouldn't route through:
    pokemon.event_triggers.BLOCKED_TILES entries still gated (the
    "invisible wall" idiom, e.g. Viridian City's Old Man shoving the player
    back until EVENT_GOT_POKEDEX), unioned with STATIC_OBSTACLES (every
    STAY-movement NPC/trainer/item-ball sprite's own tile -- always
    blocked, not RAM-dependent; pokemon.map_collision has no notion of
    sprite occupancy at all, only tileset background data, so these would
    otherwise read as walkable). Confirmed live bug this fixes: once the
    Old Man's own tile was correctly avoided, world_graph rerouted straight
    through Viridian City's sleeping gambler NPC one tile over -- a route
    just as physically impossible, for a different reason (a solid sprite,
    not a scripted push-back), and again with BADGE_BLOCKED_TILES (e.g.
    Viridian Gym's own front door, closed until every other badge is
    obtained -- has_all_badges_except)."""
    conditionally_blocked = frozenset(
        (x, y, map_id)
        for map_id, x, y, event_name in _event_triggers.BLOCKED_TILES
        if not event_flag_set(event_name, memory)
    )
    badge_blocked = frozenset(
        (x, y, map_id)
        for map_id, x, y, bit_name in _event_triggers.BADGE_BLOCKED_TILES
        if not has_all_badges_except(bit_name, memory)
    )
    return _STATIC_OBSTACLES | conditionally_blocked | badge_blocked


def nearest_unlockable_event(
    pos: Pos,
    memory,
    max_hops: int = DEFAULT_MAX_HOPS,
    excluded_events: frozenset[str] = frozenset(),
) -> tuple[int, int, int, str] | None:
    """(dx, dy, dist, event_name) for the nearest tile among every
    currently-unset, currently-satisfiable event trigger that a live BFS
    from `pos` can actually walk to within `max_hops` (without crossing any
    currently-active BLOCKED_TILES invisible wall -- see
    currently_blocked_tiles), or None if none is reachable. Same
    dx/dy-from-first-hop / dist-from-path-length contract as
    pokemon.navigation.nearest_objective (see its own docstring for why:
    a path can cross onto another map via a warp with no geometric
    relationship to `pos` at all, so only "which way to move right now" is
    well-defined for a possibly-distant target).

    `excluded_events` (empty by default) drops specific event names from
    candidacy regardless of live state -- for a caller-driven stall
    backstop (Data.compass_progress's own COMPASS_STALL_STEPS mechanism):
    this module's own guard/requires checks are still just a static-
    analysis hint, not proof (see event_triggers' module docstring), so a
    target that turns out to be gated on something this module structurally
    can't see (a badge/item check this repo's manual audit hasn't found
    yet) can still eat the compass slot indefinitely without one."""
    tile_to_event: dict[Pos, str] = {}
    for event_name, sites in _event_triggers.EVENT_TRIGGERS.items():
        if event_name in excluded_events or event_flag_set(event_name, memory):
            continue
        satisfiable_sites = [s for s in sites if requires_satisfied(s["requires"], memory)]
        if not satisfiable_sites:
            continue
        for site in satisfiable_sites:
            for tile in tiles_for_site(site):
                tile_to_event.setdefault(tile, event_name)
    if not tile_to_event:
        return None

    blocked = currently_blocked_tiles(memory)
    result = _world_graph.shortest_path_to_nearest(
        pos, list(tile_to_event), max_hops=max_hops, blocked=blocked
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
    return dx, dy, dist, tile_to_event[best_tile]
