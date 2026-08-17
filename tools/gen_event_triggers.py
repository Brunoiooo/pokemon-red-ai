#!/usr/bin/env python
"""Exhaustive "does interacting with this actually set a wEventFlags bit"
compass data, generated from pret/pokered static analysis.

This is a *different* question than pokemon.goal_positions/pokemon.
curriculum_config answer: those resolve a small, hand-curated whitelist of
story-progress goals (Data.GOAL_CANDIDATES) to routable tiles. This module
instead scans *every* interaction the game defines -- every
data/events/hidden_events.asm entry, every data/maps/objects/*.asm
bg_event/object_event, and every scripts/*.asm dispatch-table state -- and
classifies each one as either a real SetEvent*/CheckAndSetEvent* (a
"trigger") or a pure dialog/menu no-op (excluded). Two confirmed examples
that must come out empty: Red's House F2's hidden events OpenRedsPC (a PC
menu) and PrintRedSNESText (plain dialog) -- read live, neither touches an
EVENT_* macro, only text/menu code; Red's House F1's Mom/TV bg/object
events -- read live, Mom only tests wStatusFlags4 (a status bit, not an
event), the TV only prints flavor text.

Reuses, rather than reimplements, three already-proven pieces:
- tools/pokered_asm.py's analyze_function_event_guards: a real per-function
  branch-CFG guard analyzer (handles the "CheckEvent then jump to a
  *different* branch that still SetEvents" shape a linear bail-chain scan
  would miss -- see its own docstring's BluesHouse example).
- tools/gen_map_scripts.py's find_object_events/find_bg_events/
  find_text_handlers: locate each object/bg interaction's handler label.
- pokemon.map_scripts.MAP_SCRIPTS (already generated): per-map dispatch-
  table state guards+sets, for coordinate-triggered idioms like
  PalletTown's wYCoord==1 -> SetEvent EVENT_OAK_APPEARED_IN_PALLET, which
  only fires while EVENT_FOLLOWED_OAK_INTO_LAB is NOT set (a one-way,
  negatively-gated trigger -- exactly the shape TriggerSite.requires exists
  to represent, via the same ("event", name, required_bool) guard tuples
  gen_map_scripts.py's own MAP_SCRIPTS states already use).

New here: data/events/hidden_events.asm coverage (gen_map_scripts.py never
parsed it at all), `call`/`predef` target following in addition to `jr`/`jp`
(gen_map_scripts.find_reachable_set_events deliberately does not, "too
broad/noisy" -- but a *complete* trace is exactly what this module is for),
and per-interaction `requires` capture (gen_map_scripts.py only tracks
guards for its own per-map dispatch-table states, never for object/bg/
hidden interactions).

`call`/`predef` targets are only followed into scripts/*.asm and a
deliberately narrow engine/ subset (CALL_FOLLOW_DIRS/CALL_FOLLOW_FILES
below) -- confirmed by reading actual predef targets reachable from hidden
events (HiddenItems/HiddenCoins/slot machine) that the excluded subsystems
(battle/link/gfx/math/menus/movie/pokemon/debug/slots) touch their own
dedicated RAM (wObtainedHiddenItemsFlags, wObtainedHiddenCoinsFlags, slot
state), never wEventFlags -- so excluding them only saves traversal budget,
it never hides a real trigger. Depth/visited budgets (MAX_DEPTH/
MAX_VISITED) bound the search the same "prefer missing over wrong" way as
the rest of this project's static analysis: a budget exhausted before
reaching a real SetEvent is a silently missed trigger, never a wrong one.
A function whose own local-label CFG has a loop (analyze_function_event_
guards returns {} for it) falls back to a flat, unguarded scan of the same
range -- the interaction is still reported, its `requires` just comes back
empty (the same "hint, not proof" tradeoff gen_map_scripts.py's own
find_reachable_set_events already makes everywhere, accepted here too for
this one narrow case).

Manual overrides: tools/event_graph_manual_edges.MANUAL_PARENT_EDGES is
merged in verbatim as extra `requires` (as required=True) for any event it
names as a child -- the handful of idioms (bag-content checks via a helper
this module's guard analyzer doesn't model) even a `call`-following walk
can't see.

Outputs: src/pokemon/event_triggers.py (importable at runtime):
  EVENT_TRIGGERS: event_name -> list of TriggerSite dicts:
    {"map_id": int, "x": int|None, "y": int|None, "kind": str,
     "requires": tuple[("event", name, required_bool), ...]}
  kind is "object" (interacting NPC/item/trainer sprite -- own tile is
  solid, resolve to an adjacent walkable tile), "walkable" (bg_event/
  hidden_event -- player stands directly on it), "coord" (a per-map
  dispatch-table state's own coordinate guard -- x or y may be None,
  meaning the whole row/column satisfies it), or "entrance" (a map's
  dispatch-table state 0 with NO coordinate guard of its own -- the whole
  event fires automatically the instant the player walks onto the map at
  all, x/y always None, resolved at runtime to every warp-in tile).

Usage: python tools/gen_event_triggers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import pokered_asm as pa  # noqa: E402
import gen_map_scripts as gms  # noqa: E402
import event_graph_manual_edges as manual_edges  # noqa: E402
from pokemon import event_graph as _event_graph  # noqa: E402  (pa already put src/ on sys.path)
from pokemon import map_constants as mc  # noqa: E402
from pokemon import map_scripts as existing_map_scripts  # noqa: E402

REPO_ROOT = pa.REPO_ROOT
POKERED_DIR = pa.POKERED_DIR
SCRIPTS_DIR = pa.SCRIPTS_DIR
ENGINE_DIR = pa.ENGINE_DIR
HIDDEN_EVENTS_ASM = POKERED_DIR / "data" / "events" / "hidden_events.asm"
OUTPUT_PY = REPO_ROOT / "src" / "pokemon" / "event_triggers.py"

# See module docstring: confirmed-safe-to-skip engine/ subsystems excluded
# on purpose (battle/gfx/link/math/menus/movie/pokemon/debug/slots).
CALL_FOLLOW_DIRS = [
    ENGINE_DIR / "events",
    ENGINE_DIR / "items",
    ENGINE_DIR / "overworld",
]
CALL_FOLLOW_FILES = [
    ENGINE_DIR / "flag_action.asm",
]

MAX_DEPTH = 6
MAX_VISITED = 40

CALL_OR_PREDEF_RE = re.compile(r"^\t(?:call|predef|predef_jump)\s+([A-Za-z_]\w*)\s*(?:;.*)?$")

HIDDEN_EVENTS_FOR_RE = re.compile(r"^\thidden_events_for\s+(\w+)\s*$")
HIDDEN_EVENT_RE = re.compile(r"^\thidden_event\s+(\d+),\s*(\d+),\s*(\w+),")
HIDDEN_TEXT_PREDEF_RE = re.compile(r"^\thidden_text_predef\s+(\d+),\s*(\d+),\s*(\w+),")


def build_label_index() -> dict[str, tuple[list[str], int, int]]:
    """label -> (lines, start, end) across scripts/*.asm plus the allowed
    engine/ subset -- the single source `call`/`predef`/`jr`/`jp` targets
    are resolved against (top-level labels are unique project-wide, no
    per-file scoping needed)."""
    index: dict[str, tuple[list[str], int, int]] = {}
    files = list(SCRIPTS_DIR.glob("*.asm"))
    for d in CALL_FOLLOW_DIRS:
        files.extend(d.rglob("*.asm"))
    files.extend(CALL_FOLLOW_FILES)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for name, (start, end) in pa.find_label_ranges(lines).items():
            index.setdefault(name, (lines, start, end))
    return index


def trace_interaction(
    start_label: str,
    label_index: dict[str, tuple[list[str], int, int]],
    states: dict[int, dict] | None = None,
    script_const_to_index: dict[str, int] | None = None,
) -> list[tuple[str, tuple]]:
    """[(event_name, requires), ...] reachable from start_label's body --
    see module docstring for the guard-analysis/fallback/budget rules.

    `states`/`script_const_to_index` (this interaction's own map's
    MAP_SCRIPTS states + SCRIPT_CONST->index table, optional) enable a
    third mechanism beyond jr/jp/call-following: a `ld a, SCRIPT_X` / `ld
    [w*CurScript], a` state transition found anywhere in a visited label's
    body hands control back to the map's own per-frame dispatch at a new
    state, so whatever that state (or anything reachable from it via
    next_state) sets is just as reachable as a direct SetEvent right here
    -- the same mechanism gen_map_scripts.find_reachable_set_events already
    uses for object/bg interactions (see its own docstring's Oak's Lab
    example), which this module's own trace_interaction had NOT been doing
    at all until now. Confirmed live bug this fixes: talking to Oak with
    the parcel in hand (OaksLabOak1Text) transitions into
    SCRIPT_OAKSLAB_RIVAL_ARRIVES_AT_OAKS_REQUEST, which automatically
    chains into OaksLabOakGivesPokedexScript two states later -- the
    actual `SetEvent EVENT_GOT_POKEDEX` / `SetEvent EVENT_OAK_GOT_PARCEL`
    site -- a completely different top-level label jr/jp/call/predef-
    following alone would never reach. Both events already have a
    tools/event_graph_manual_edges.py override (EVENT_GOT_OAKS_PARCEL as
    parent, since the `IsItemInBag` bag-content check gating this
    transition isn't itself guard-representable) -- that merge step
    (main()) picks it up automatically once a real site exists here."""
    results: list[tuple[str, tuple]] = []
    seen: set[str] = set()
    frontier: list[tuple[str, int]] = [(start_label, 0)]
    while frontier:
        label, depth = frontier.pop()
        if label in seen or len(seen) >= MAX_VISITED or label not in label_index:
            continue
        seen.add(label)
        lines, start, end = label_index[label]

        guards_by_event = pa.analyze_function_event_guards(lines, start, end)
        if guards_by_event:
            for ev, guard_set in guards_by_event.items():
                requires = tuple(sorted(g for g in guard_set if g[0] == "event"))
                results.append((ev, requires))
        else:
            # No SetEvent found, or this function's local-label CFG has a
            # loop (see module docstring) -- flat unguarded fallback scan.
            for lineno in range(start, end):
                m = pa.CALL_RE.match(lines[lineno].strip())
                if m and m.group(1) in (
                    pa.SET_MACROS | pa.IMPLICIT_SET_MACROS | pa.CHECK_AND_SET_MACROS
                ):
                    for nm in m.group(2).split(","):
                        nm = nm.strip()
                        if nm in pa.EVENTS:
                            results.append((nm, ()))

        # `ld hl, <TrainerHeaderLabel>` right before handing off to the
        # shared TalkToTrainer engine routine -- a trainer's own
        # `trainer EVENT_X, level, ...` data line is IMPLICIT_SET_MACROS,
        # but it's data a battle win sets as a side effect, not code this
        # walk would otherwise reach via jr/jp/call. Same narrow, one-level
        # (not recursively BFS-followed) scan gen_map_scripts.
        # find_reachable_set_events already uses, for the same reason: a
        # generic "follow every ld hl" would risk false edges from
        # unrelated data-pointer loads.
        for lineno in range(start, end):
            m = gms.LD_HL_LABEL_RE.match(lines[lineno])
            if not m or m.group(1) not in label_index:
                continue
            th_lines, th_start, th_end = label_index[m.group(1)]
            for th_lineno in range(th_start, th_end):
                tm = pa.CALL_RE.match(th_lines[th_lineno].strip())
                if tm and tm.group(1) in pa.IMPLICIT_SET_MACROS:
                    for nm in tm.group(2).split(","):
                        nm = nm.strip()
                        if nm in pa.EVENTS:
                            results.append((nm, ()))

        # `ld a, SCRIPT_X` / `ld [w*CurScript], a` state transition -- see
        # this function's own docstring for why (Oak's Lab parcel/Pokedex
        # example).
        if states and script_const_to_index:
            for lineno in range(start, end - 1):
                tm = gms.TRANSITION_LD_A_RE.match(lines[lineno])
                if tm and gms.TRANSITION_STORE_RE.match(lines[lineno + 1]):
                    target_idx = script_const_to_index.get(tm.group(1))
                    if target_idx is not None:
                        for ev in gms._events_from_state_onward(states, target_idx):
                            results.append((ev, ()))

        if depth >= MAX_DEPTH:
            continue
        for lineno in range(start, end):
            line = lines[lineno]
            m = pa.COND_JUMP_RE.match(line)
            if m and not m.group(2).startswith("."):
                frontier.append((m.group(2), depth + 1))
                continue
            m = pa.UNCOND_JUMP_RE.match(line)
            if m and not m.group(1).startswith("."):
                frontier.append((m.group(1), depth + 1))
                continue
            m = CALL_OR_PREDEF_RE.match(line.strip())
            if m:
                frontier.append((m.group(1), depth + 1))
    return results


def parse_hidden_events(text: str) -> list[tuple[str, int, int, str]]:
    """[(map_name, x, y, function_label), ...] for every hidden_event/
    hidden_text_predef entry -- function_label is param 3 for both macros
    (see hidden_events.asm's own MACRO definitions; the emitted byte order
    is y-then-x, but the call-site argument order, like every other
    x/y-taking macro in this project's data files, is x-then-y)."""
    out: list[tuple[str, int, int, str]] = []
    current_map: str | None = None
    for line in text.splitlines():
        m = HIDDEN_EVENTS_FOR_RE.match(line)
        if m:
            current_map = m.group(1)
            continue
        if current_map is None:
            continue
        m = HIDDEN_EVENT_RE.match(line) or HIDDEN_TEXT_PREDEF_RE.match(line)
        if m:
            out.append((current_map, int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def hidden_event_triggers(
    label_index: dict[str, tuple[list[str], int, int]]
) -> list[tuple[str, int, int, int, str, tuple]]:
    """[(event_name, map_id, x, y, "walkable", requires), ...]"""
    out = []
    text = HIDDEN_EVENTS_ASM.read_text(encoding="utf-8", errors="replace")
    unresolved_maps = 0
    for map_name, x, y, func_label in parse_hidden_events(text):
        entry = mc.MAPS.get(map_name)
        if entry is None:
            unresolved_maps += 1
            continue
        map_id = entry[0]
        # states/script_const_to_index deliberately not passed here --
        # trace_interaction's state-transition following needs both (see
        # its own docstring), and there's no cheap way to recover this
        # map's own scripts/*.asm path from map_id alone here. Hidden-event
        # handlers rarely hand off into the per-map dispatch table the way
        # object/bg interactions do, so this is a low-value gap to leave
        # (same "prefer missing over wrong" tradeoff as elsewhere).
        for ev, requires in trace_interaction(func_label, label_index):
            out.append((ev, map_id, x, y, "walkable", requires))
    if unresolved_maps:
        print(f"warning: {unresolved_maps} hidden_events.asm entries referenced an unknown map name")
    return out


def object_bg_triggers(
    label_index: dict[str, tuple[list[str], int, int]]
) -> list[tuple[str, int, int, int, str, tuple]]:
    """[(event_name, map_id, x, y, kind, requires), ...] for every
    object_event/bg_event interaction across data/maps/objects/*.asm."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        objects_path = gms.OBJECTS_DIR / path.name
        if not objects_path.is_file():
            continue
        scripts_text = path.read_text(encoding="utf-8", errors="replace")
        objects_text = objects_path.read_text(encoding="utf-8", errors="replace")
        text_handlers = gms.find_text_handlers(scripts_text)
        # This map's own SCRIPT_CONST -> state-index table, for
        # trace_interaction's state-transition following (see its own
        # docstring) -- built the same way gen_map_scripts.build_map_script
        # does, from the same def_script_pointers table.
        tables = gms.find_state_tables(scripts_text)
        script_const_to_index = (
            {const: idx for idx, (_h, const) in enumerate(tables[0])} if tables else None
        )
        states = existing_map_scripts.MAP_SCRIPTS.get(map_id, {}).get("states")

        for x, y, text_const in gms.find_object_events(objects_text):
            handler = text_handlers.get(text_const)
            if handler is None:
                continue
            for ev, requires in trace_interaction(handler, label_index, states, script_const_to_index):
                out.append((ev, map_id, x, y, "object", requires))

        for x, y, text_const in gms.find_bg_events(objects_text):
            handler = text_handlers.get(text_const)
            if handler is None:
                continue
            for ev, requires in trace_interaction(handler, label_index, states, script_const_to_index):
                out.append((ev, map_id, x, y, "walkable", requires))
    return out


def state_machine_triggers() -> list[tuple[str, int, int | None, int | None, str, tuple]]:
    """[(event_name, map_id, x, y, kind, requires), ...] -- reuses the
    already-generated pokemon.map_scripts.MAP_SCRIPTS directly, no
    reparsing (see module docstring's PalletTown example). Two sources:

    1. "coord": a state with its own coordinate guard. Not just that
    state's OWN `sets` -- everything reachable via its next_state chain too
    (gms._events_from_state_onward). Confirmed live bug this fixes:
    OaksLab's OaksLabRivalChallengesPlayerScript gates on wYCoord==6 but
    sets nothing itself -- `SetEvent EVENT_BATTLED_RIVAL_IN_OAKS_LAB` only
    happens two states later, automatically, once the scripted rival battle
    resolves.

    2. "entrance": state 0 (a map's dispatch table always starts here on
    entry) when it has NO coordinate guard of its own -- its guards (if
    any) are pure event prerequisites, so its whole forward chain fires
    automatically the instant the player walks onto the map at all, no
    specific tile required. Confirmed live bug this fixes: ViridianMart's
    state 0 unconditionally hands over Oak's Parcel (`SetEvent
    EVENT_GOT_OAKS_PARCEL`) the moment the map loads (a scripted "clerk
    greets you" cutscene, not an object_event interaction at all) -- with
    only the "coord" source, this event had zero sites anywhere and could
    never become a compass candidate, no matter how close the player
    actually was. gms._events_from_state_onward's own cycle detection is
    what keeps this from over-reaching into events that DO need real
    interaction first: OaksLab's state 0 also qualifies (no coord guard,
    only `CheckEvent EVENT_OAK_APPEARED_IN_PALLET`), and correctly yields
    EVENT_FOLLOWED_OAK_INTO_LAB/_2 and EVENT_OAK_ASKED_TO_CHOOSE_MON (all
    genuinely automatic once you walk in with Oak having already appeared)
    but stops at state 6's own wYCoord==6 gate before ever reaching
    EVENT_GOT_STARTER, which needs a real Poke Ball interaction."""
    out = []
    for map_id, ms in existing_map_scripts.MAP_SCRIPTS.items():
        for idx, state in ms["states"].items():
            coord_guards = [g for g in state["guards"] if g[0] == "coord"]
            if coord_guards:
                reachable_events = gms._events_from_state_onward(
                    ms["states"], idx, known_satisfied_start=True
                )
                if not reachable_events:
                    continue
                x = next((v for _k, ax, v in coord_guards if ax == "x"), None)
                y = next((v for _k, ax, v in coord_guards if ax == "y"), None)
                requires = tuple(sorted(g for g in state["guards"] if g[0] == "event"))
                for ev in reachable_events:
                    out.append((ev, map_id, x, y, "coord", requires))
            elif idx == 0:
                reachable_events = gms._events_from_state_onward(
                    ms["states"], idx, known_satisfied_start=True
                )
                if not reachable_events:
                    continue
                requires = tuple(sorted(g for g in state["guards"] if g[0] == "event"))
                for ev in reachable_events:
                    out.append((ev, map_id, None, None, "entrance", requires))
    return out


MOVE_PLAYER_RE = re.compile(r"^\tcall\s+\w*MovePlayer(?:Up|Down|Left|Right)Script\s*$")


def blocking_tiles() -> list[tuple[int, int, int, str]]:
    """[(map_id, x, y, event_name), ...] -- the invisible-wall-until-this-
    event idiom (e.g. ViridianCity's Old Man blocking the road north until
    EVENT_GOT_POKEDEX, or its own gym door "locked" push-back until
    EVENT_VIRIDIAN_GYM_OPEN): `CheckEvent EVENT_X` / `ret nz` (continues
    only while EVENT_X is False -- an idempotency-shaped guard, same
    ("event", name, False) tuple pokered_asm/gen_map_scripts already use),
    then a wYCoord/wXCoord pin on one exact tile, then a call into a
    `<Map>MovePlayer<Dir>Script` helper that simulates a button press
    shoving the player back the way they came. Confirmed live bug this
    fixes: world_graph has no notion of a conditionally-blocked tile at
    all, so its BFS happily routed the compass straight through
    ViridianCity's Old Man at (19, 9) toward EVENT_BEAT_VIRIDIAN_GYM_
    TRAINER_1 even with EVENT_GOT_POKEDEX unset -- a real, physically
    correct hop count, but not an actually-walkable path (the game itself
    reverses that step every time). Reuses gen_map_scripts.parse_entry_
    guards for the guard-chain scan (same shape as a dispatch-table
    state's entry guards, just on a plain helper function instead).

    Item-gated versions of this same idiom (the Saffron guards' drink-for-
    passage gates) and single-badge-bit versions (Route22Gate's Boulder
    Badge check, Route23's all-8-badges check) are deliberately NOT modeled
    here -- items, and a single/all badge bit, would need their own guard
    vocabulary this project's static analysis doesn't have yet (same
    "prefer missing over wrong" tradeoff as everywhere else in this
    module): those tiles are left exactly as walkable as they already
    were, not silently (mis)blocked by a guess. The one badge-COUNT
    exception (all-except-one) is badge_blocking_tiles() below, since it's
    representable precisely and needed for a real, confirmed case."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for _label, (start, end) in pa.find_label_ranges(lines).items():
            guards, after = gms.parse_entry_guards(lines, start + 1, end)
            event_guard = next(
                (name for kind, name, required in guards if kind == "event" and not required),
                None,
            )
            coord_guards = [g for g in guards if g[0] == "coord"]
            if event_guard is None or not coord_guards:
                continue
            if not any(MOVE_PLAYER_RE.match(lines[i]) for i in range(after, end)):
                continue
            x = next((v for _k, ax, v in coord_guards if ax == "x"), None)
            y = next((v for _k, ax, v in coord_guards if ax == "y"), None)
            if x is None or y is None:
                continue  # need an exact tile, not a whole row/column, to block
            out.append((map_id, x, y, event_guard))
    return out


BADGE_ALL_EXCEPT_RE = re.compile(r"^\tcp ~\(1 << (BIT_\w+)\)\s*$")


def badge_blocking_tiles() -> list[tuple[int, int, int, str]]:
    """[(map_id, x, y, bit_name), ...] -- the "closed until every OTHER
    badge is obtained" idiom: `ld a, [wObtainedBadges]` / `cp ~(1 <<
    BIT_X)`, in the same function as a wYCoord/wXCoord pin and a
    MovePlayer<Dir>Script call (see blocking_tiles()'s docstring for that
    shape -- this is the same idea, just with a badge-count comparison
    standing in for CheckEvent). Confirmed live bug this fixes: Viridian
    Gym's own front door refuses entry (a scripted push-back, exactly like
    the Old Man) until the player holds all 7 OTHER badges -- badges live
    in wObtainedBadges, a completely different RAM byte than wEventFlags,
    invisible to every other guard/blocker detector in this module, so the
    compass kept routing EVENT_BEAT_VIRIDIAN_GYM_TRAINER_1 straight at a
    door the real game would never open. Confirmed via a full grep that
    `cp ~(1 << BIT_X)` appears exactly twice project-wide, both in
    ViridianCity.asm: this one (a real movement gate) and
    ViridianCityGambler1Text (flavor dialogue only, correctly excluded
    below by the same MovePlayer-call requirement)."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for _label, (start, end) in pa.find_label_ranges(lines).items():
            bit_name = next(
                (m.group(1) for line in lines[start:end] if (m := BADGE_ALL_EXCEPT_RE.match(line))),
                None,
            )
            if bit_name is None:
                continue
            if not any(MOVE_PLAYER_RE.match(lines[i]) for i in range(start, end)):
                continue
            x = y = None
            for i in range(start, end - 1):
                m = gms.COORD_LOAD_RE.match(lines[i])
                if not m:
                    continue
                cp_m = gms.CP_RE.match(lines[i + 1])
                if not cp_m:
                    continue
                if m.group(1) == "wXCoord":
                    x = int(cp_m.group(1))
                else:
                    y = int(cp_m.group(1))
            if x is None or y is None:
                continue
            out.append((map_id, x, y, bit_name))
    return out


STATIONARY_OBJECT_RE = re.compile(r"^\tobject_event\s+(\d+),\s*(\d+),\s*\w+,\s*STAY,")


def stationary_object_tiles() -> list[tuple[int, int, int]]:
    """[(map_id, x, y), ...] for every object_event with STAY movement
    (never walks) across data/maps/objects/*.asm -- a permanent physical
    obstacle in the real game (an NPC/trainer/item-ball sprite blocks its
    own tile) that pokemon.map_collision has NO way to know about at all:
    it's derived purely from tileset background data (see tools/gen_map_
    collision.py), never from sprite occupancy, so a STAY sprite sitting
    on an otherwise-open floor tile shows up as walkable there. Confirmed
    live bug this fixes: Viridian City's sleeping gambler (SPRITE_GAMBLER_
    ASLEEP) permanently occupies (18, 9), right next to the Old Man's
    invisible EVENT_GOT_POKEDEX wall at (19, 9) (see BLOCKED_TILES) --
    world_graph's BFS, once taught to avoid (19, 9), simply rerouted
    straight through (18, 9) instead, a route that's just as physically
    impossible. WALK-movement NPCs are deliberately not included: their
    actual tile varies frame to frame, so blocking their spawn coordinate
    would be neither reliable nor obviously correct (same "prefer missing
    over wrong" tradeoff as the rest of this module) -- gaps like a toggled
    -away NPC (ShowObject/HideObject) later becoming walkable again are
    accepted for the same reason."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        objects_path = gms.OBJECTS_DIR / path.name
        if not objects_path.is_file():
            continue
        for line in objects_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = STATIONARY_OBJECT_RE.match(line)
            if m:
                out.append((map_id, int(m.group(1)), int(m.group(2))))
    return out


def main() -> None:
    label_index = build_label_index()
    print(f"{len(label_index)} top-level labels indexed for call/predef/jr/jp following")

    all_triggers = (
        state_machine_triggers()
        + object_bg_triggers(label_index)
        + hidden_event_triggers(label_index)
    )

    event_triggers: dict[str, list[dict]] = {}
    seen_sites: set[tuple] = set()
    for ev, map_id, x, y, kind, requires in all_triggers:
        key = (ev, map_id, x, y, kind, requires)
        if key in seen_sites:
            continue
        seen_sites.add(key)
        event_triggers.setdefault(ev, []).append(
            {"map_id": map_id, "x": x, "y": y, "kind": kind, "requires": requires}
        )

    # event_graph.AUTO_BLACKLIST_EVENTS (dead | cyclic | resettable -- see
    # tools/gen_event_graph.py): a compass target has to STAY true once
    # reached to mean anything. Confirmed live bug this filter fixes:
    # EVENT_ROUTE22_RIVAL_WANTS_BATTLE is set at the tail of Oak's Lab's
    # Pokedex cutscene (a real, discoverable site -- exactly what this
    # module is for) but is also ResetEvent'd in three other places
    # (Route22.asm x2, PewterGym.asm) as part of the Route 22 rival-battle
    # toggle it's really tracking -- "reached" here doesn't mean "stays
    # true", so the compass could point right back at an already-passed
    # Oak's Lab tile for a flag that flips on its own. The older GOAL_
    # CANDIDATES-based compass (pokemon.navigation) already excludes this
    # whole class via the same blacklist; this module just hadn't been
    # applying it.
    blacklisted_events = sorted(set(event_triggers) & set(_event_graph.AUTO_BLACKLIST_EVENTS))
    for ev in blacklisted_events:
        del event_triggers[ev]

    # Manual overrides (tools/event_graph_manual_edges.py): extra `requires`
    # for idioms the automatic walk can't see (see module docstring).
    manual_applied = 0
    for child, parents in manual_edges.MANUAL_PARENT_EDGES.items():
        sites = event_triggers.get(child)
        if not sites:
            continue
        extra = tuple(sorted(("event", p, True) for p in parents))
        for site in sites:
            merged = tuple(sorted(set(site["requires"]) | set(extra)))
            if merged != site["requires"]:
                site["requires"] = merged
                manual_applied += 1

    blocked_tiles = blocking_tiles()
    badge_blocked_tiles = badge_blocking_tiles()
    static_obstacles = stationary_object_tiles()

    py_lines = [
        '"""Exhaustive event-trigger compass data, generated from pret/pokered',
        "static analysis -- see tools/gen_event_triggers.py for the parser and",
        'its caveats (same "hint, not proof" status as event_graph.py/',
        "map_scripts.py).",
        "",
        "EVENT_TRIGGERS: event_name -> list of TriggerSite dicts:",
        '  {"map_id": int, "x": int|None, "y": int|None, "kind": str,',
        '   "requires": tuple[("event", name, required_bool), ...]}',
        'kind is "object" (adjacent-walkable-tile interaction), "walkable"',
        '(stand-on-it bg_event/hidden_event), "coord" (per-map dispatch-',
        "table state's own coordinate guard, x or y may be None), or",
        '"entrance" (state 0 with no coordinate guard -- fires on map entry,',
        "x and y always None).",
        "",
        "BLOCKED_TILES: [(map_id, x, y, event_name), ...] -- an invisible wall",
        "(see blocking_tiles()'s docstring): that tile is impassable while",
        "event_name is NOT set, walkable once it is. Badge/item-gated versions",
        "of the same idiom are deliberately not included (see docstring).",
        "",
        "STATIC_OBSTACLES: [(map_id, x, y), ...] -- every STAY-movement",
        "object_event's own tile (see stationary_object_tiles()'s docstring):",
        "always impassable, map_collision has no notion of sprite occupancy",
        "at all so these tiles would otherwise read as walkable.",
        "",
        "BADGE_BLOCKED_TILES: [(map_id, x, y, bit_name), ...] -- same idea as",
        "BLOCKED_TILES but gated on 'every OTHER badge obtained' instead of an",
        "event flag (see badge_blocking_tiles()'s docstring).",
        "",
        "Regenerate with: python tools/gen_event_triggers.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "EVENT_TRIGGERS: dict[str, list[dict]] = " + repr(event_triggers),
        "",
        "BLOCKED_TILES: list[tuple[int, int, int, str]] = " + repr(blocked_tiles),
        "",
        "STATIC_OBSTACLES: list[tuple[int, int, int]] = " + repr(static_obstacles),
        "",
        "BADGE_BLOCKED_TILES: list[tuple[int, int, int, str]] = " + repr(badge_blocked_tiles),
        "",
    ]
    OUTPUT_PY.write_text("\n".join(py_lines), encoding="utf-8")

    total_sites = sum(len(v) for v in event_triggers.values())
    print(f"{len(event_triggers)} distinct events have at least one trigger site, {total_sites} sites total")
    print(f"{len(blacklisted_events)} events dropped as dead/cyclic/resettable (event_graph.AUTO_BLACKLIST_EVENTS)")
    print(f"manual override applied to {manual_applied} sites")
    print(f"{len(blocked_tiles)} conditionally-blocked tiles found")
    print(f"{len(static_obstacles)} static (STAY-sprite) obstacle tiles found")
    print(f"{len(badge_blocked_tiles)} badge-count-blocked tiles found")
    print(f"wrote {OUTPUT_PY}")


if __name__ == "__main__":
    main()
