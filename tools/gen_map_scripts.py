#!/usr/bin/env python
"""Build a per-map dispatch-table script graph from the pret/pokered
disassembly -- the compass's single source of truth for "where do I need to
be, and what state does this map's story need to be in, to trigger a given
event", replacing tools/gen_coord_triggers.py's blind whole-file regex scan
and tools/gen_event_graph.py's old table-order-only heuristic (item 5).
Those two, plus a special-cased ancestor-walk in goal_positions.py, were
three separate patches for the same underlying gap: a proper per-map state
graph instead of three overlapping proximity guesses. See
PalletTown_Script for the motivating example -- `PalletTownDefaultScript`
is state 0 of PalletTown_ScriptPointers, and its entry-guard chain
(`CheckEvent EVENT_FOLLOWED_OAK_INTO_LAB / ret nz` then `ld a, [wYCoord] /
cp 1 / ret nz`) is the ONLY thing gating `SetEvent
EVENT_OAK_APPEARED_IN_PALLET` -- not a proximity heuristic in any file, a
literal state machine.

For every scripts/*.asm file with a `def_script_pointers` table (98 of
them), parses one MapScript per map_id: for each dispatch state (a handler
in the table; table position = state index, matching def_script_pointers'
const_def numbering), its entry-guard chain (every check-and-bail at the
top of the handler body -- event or coordinate, correctly signed, see
parse_entry_guards), which events it SetEvents (events_set_in_range), and
which state it transitions to next (`ld a, SCRIPT_X` + `ld [w*CurScript],
a`, see parse_transition). Heuristic, not proof -- same caveats as
tools/gen_event_graph.py: only this specific "check, then bail unless
matched" shape is recognized; a `jr z, .label` branch into a *different*
code path (not a bail) is not followed, so a guard/transition written that
way is silently missed here, never guessed wrong. Only the first
def_script_pointers table in a file is used if more than one exists (rare
in practice).

Cross-file/prologue-level event dependencies (e.g. a dispatcher's own
leading side-effect check, or an unrelated file's CheckEvent) are still
tools/gen_event_graph.py's job (its general CheckEvent-before-SetEvent,
same-function, signed heuristic already covers those) -- this module is
deliberately scoped to *within one dispatch table* only.

Also covers the third, previously-unmodeled class of "where do scripts run"
location: object_event NPC/sign/item/trainer and bg_event sign/PC/etc
interactions (data/maps/objects/*.asm) -- e.g. Oak's Lab's three starter
Poke Balls are plain object_events, and walking up to one and interacting
is what actually advances wOaksLabCurScript into OaksLabChoseStarterScript
(state 8) -- nothing in the map's own def_script_pointers table leads there
at all, since states 6 and 7 just loop on each other waiting for the player
to be positioned right, and the ball's own text script is what jumps the
state forward. find_object_events/find_bg_events parse each interaction's
(x, y, TEXT_*) triple (object_event covers NPC/sign, item, and trainer
forms -- see its own docstring for why item/trainer aren't skipped
anymore); find_text_handlers resolves TEXT_* to its handler in the same
file's def_text_pointers table(s); find_reachable_set_events then does a
coarse reachability walk (not full guard-condition tracking like
pokered_asm.analyze_function_event_guards -- just "can interacting with
this eventually SetEvent X"), by following jr/jp targets across top-level
functions (e.g. OaksLabCharmanderPokeBallText jumping into the shared
OaksLabSelectedPokeBallScript) and, for trainers specifically, the one
`ld hl, <TrainerHeaderN>` operand that names the event a won battle sets
(see its own docstring). `call` targets are deliberately NOT followed
(could lead into arbitrary shared engine code, too broad/noisy to trust
here) -- same "prefer missing over wrong" policy as everywhere else in
this project. Each interaction resolves to a *routable* tile, not
necessarily its own (x, y): an object_event's own tile is almost always
solid (an NPC/item/trainer sprite blocks it), so it resolves to an
adjacent walkable tile instead (_adjacent_walkable_tiles); a bg_event's
own tile is walkable by convention (the player stands on a sign/PC to use
it) so it's used directly, falling back to an adjacent tile only if the
collision data disagrees (_bg_event_tiles).

Outputs: src/pokemon/map_scripts.py (importable at runtime):
  MAP_SCRIPTS: map_id -> {cur_script_ram, states}
  COORD_TRIGGERS: event name -> [(map_id, x_or_None, y_or_None), ...],
                  derived from every state's own coordinate guard(s) PLUS
                  every object_event/bg_event interaction that can reach
                  it -- replaces pokemon.coord_triggers.py for
                  pokemon.goal_positions.

Usage: python tools/gen_map_scripts.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import pokered_asm as pa  # noqa: E402  (shared with gen_event_graph.py -- see its own docstring for why this isn't just imported from gen_event_graph.py directly: gen_event_graph.py imports THIS module, so the reverse would be circular)
from pokemon import map_collision  # noqa: E402  (pa already put src/ on sys.path)

REPO_ROOT = pa.REPO_ROOT
POKERED_DIR = pa.POKERED_DIR
SCRIPTS_DIR = pa.SCRIPTS_DIR
OUTPUT_PY = REPO_ROOT / "src" / "pokemon" / "map_scripts.py"

SCRIPT_TABLE_START_RE = re.compile(r"^\tdef_script_pointers\s*$")
DW_CONST_RE = re.compile(r"^\tdw_const\s+(\w+),\s*(\w+)")
COORD_LOAD_RE = re.compile(r"^\tld a, \[(wYCoord|wXCoord)\]\s*(;.*)?$")
CP_RE = re.compile(r"^\tcp\s+(-?\d+)")
CHECK_EVENT_RE = re.compile(r"^\tCheckEvent\s+(EVENT_\w+)\s*$")
# `CheckBothEventsSet EVENT_A, EVENT_B[, byte_hint]` (macros/scripts/
# events.asm) sets Z when BOTH bits are set -- the OPPOSITE polarity of
# plain CheckEvent's Z-means-false, so its own bail direction is inverted:
# `jr nz` (not `jr z`) after it means "bail unless both hold", a real
# two-event AND prerequisite. Confirmed live bug: PalletTown.asm's
# PalletTownDaisyScript gates EVENT_DAISY_WALKING on `CheckBothEventsSet
# EVENT_GOT_TOWN_MAP, EVENT_ENTERED_BLUES_HOUSE, 1 / jr nz, .next` --
# without this, that gate was invisible here, so both this module's own
# COORD_TRIGGERS/state guards AND (via add_map_script_edges in
# gen_event_graph.py) EVENT_GRAPH's parent inference silently missed it,
# letting the live compass point at EVENT_DAISY_WALKING as if it needed no
# prerequisite at all. `CheckEitherEventSet` (Z means NEITHER set, an OR
# condition) has no clean representation as an extra AND'd guard tuple, so
# it's deliberately left unmatched here -- same "skip, never guess" policy
# as everywhere else in this module.
CHECK_BOTH_EVENTS_RE = re.compile(r"^\tCheckBothEventsSet\s+(EVENT_\w+),\s*(EVENT_\w+)\b")
# Raw bail-instruction families. Which one means "a real prerequisite" is
# OPPOSITE between CheckEvent and `cp` -- CheckEvent's macro sets Z when
# the event is FALSE, so `ret z`/`jr z`/`jp z` right after it means "bail
# unless the event holds" (a real prerequisite; `ret nz` is the inverse,
# an idempotency guard -- see PalletTownDefaultScript's own `CheckEvent
# EVENT_FOLLOWED_OAK_INTO_LAB / ret nz`, deliberately not a guard
# requiring that event). `cp N` sets Z when EQUAL, so it's the mirror
# image: `ret nz`/`jr nz`/`jp nz` means "bail unless it equals N" (the
# real prerequisite); `ret z`/`jr z`/`jp z` would mean "bail if it DOES
# equal N", a different shape this module doesn't model, so it's simply
# not matched (ambiguous -> skip, never guess).
_RET_Z_FAMILY = re.compile(r"^\t(?:ret z|jr z\s*,|jp z\s*,)")
_RET_NZ_FAMILY = re.compile(r"^\t(?:ret nz|jr nz\s*,|jp nz\s*,)")
BAIL_EVENT_REQUIRED = _RET_Z_FAMILY    # CheckEvent + this -> event must be True
BAIL_EVENT_FORBIDDEN = _RET_NZ_FAMILY  # CheckEvent + this -> event must be False (idempotency)
BAIL_COORD_REQUIRED = _RET_NZ_FAMILY   # cp N + this -> coordinate must equal N
TRANSITION_LD_A_RE = re.compile(r"^\tld a, (SCRIPT_\w+)\s*$")
TRANSITION_STORE_RE = re.compile(r"^\tld \[(w\w*CurScript)\], a\s*$")

MAX_GUARD_CHAIN = 10  # gates; a real chain is short (FightingDojo's 3 is the longest seen)


def find_state_tables(text: str) -> list[list[tuple[str, str]]]:
    """[[(handler_label, SCRIPT_CONST), ...], ...] -- one list per
    `def_script_pointers` table, table position = state index (matches
    that macro's own const_def numbering, 0-based)."""
    lines = text.splitlines()
    tables: list[list[tuple[str, str]]] = []
    i = 0
    while i < len(lines):
        if SCRIPT_TABLE_START_RE.match(lines[i]):
            entries: list[tuple[str, str]] = []
            j = i + 1
            while j < len(lines):
                m = DW_CONST_RE.match(lines[j])
                if not m:
                    break
                entries.append((m.group(1), m.group(2)))
                j += 1
            if entries:
                tables.append(entries)
            i = j
        else:
            i += 1
    return tables


def parse_entry_guards(
    lines: list[str], start: int, end: int
) -> tuple[list[tuple], int]:
    """Walk forward from lines[start] (0-indexed) collecting a chain of
    check+bail gates -- ("event", NAME, required_bool) or
    ("coord", "x"|"y", value). Stops at the first non-gate line (ambiguous
    -> stop collecting, never guess) or MAX_GUARD_CHAIN gates. Returns
    (guards, index of the first line after the chain)."""
    guards: list[tuple] = []
    i = start
    while i + 1 < end and len(guards) < MAX_GUARD_CHAIN:
        m_both = CHECK_BOTH_EVENTS_RE.match(lines[i])
        if m_both:
            if BAIL_EVENT_FORBIDDEN.match(lines[i + 1]):  # jr nz -> both required True
                guards.append(("event", m_both.group(1), True))
                guards.append(("event", m_both.group(2), True))
                i += 2
                continue
            break
        m_ev = CHECK_EVENT_RE.match(lines[i])
        if m_ev:
            if BAIL_EVENT_REQUIRED.match(lines[i + 1]):
                guards.append(("event", m_ev.group(1), True))
                i += 2
                continue
            if BAIL_EVENT_FORBIDDEN.match(lines[i + 1]):
                guards.append(("event", m_ev.group(1), False))
                i += 2
                continue
            break
        m_co = COORD_LOAD_RE.match(lines[i])
        if m_co and i + 2 < end:
            cp_m = CP_RE.match(lines[i + 1])
            if cp_m and BAIL_COORD_REQUIRED.match(lines[i + 2]):
                axis = "x" if m_co.group(1) == "wXCoord" else "y"
                guards.append(("coord", axis, int(cp_m.group(1))))
                i += 3
                continue
        break
    return guards, i


def events_set_in_range(lines: list[str], start: int, end: int) -> set[str]:
    """Every EVENTS-known name SetEvent*'d (or implicit-set via `trainer`)
    within lines[start:end] -- bounded by this state handler's own
    find_label_ranges span, so this never bleeds into a different state."""
    out: set[str] = set()
    for lineno in range(start, end):
        m = pa.CALL_RE.match(lines[lineno].strip())
        if m and m.group(1) in (pa.SET_MACROS | pa.IMPLICIT_SET_MACROS):
            out.update(
                n.strip() for n in m.group(2).split(",") if n.strip() in pa.EVENTS
            )
    return out


def parse_transition(
    lines: list[str], start: int, end: int, script_const_to_index: dict[str, int]
) -> int | None:
    """First `ld a, SCRIPT_X` immediately followed by `ld [w*CurScript], a`
    in lines[start:end], resolved to a state index via this table's own
    SCRIPT_CONST list. None for a terminal/noop state, a computed/indirect
    transition, or any shape this simple scan doesn't recognize."""
    for i in range(start, end - 1):
        m = TRANSITION_LD_A_RE.match(lines[i])
        if m and TRANSITION_STORE_RE.match(lines[i + 1]):
            return script_const_to_index.get(m.group(1))
    return None


def find_cur_script_var(lines: list[str]) -> str | None:
    """This map's own dedicated `w<Map>CurScript` byte, if it uses one --
    None means it dispatches through the single shared `wCurMapScript`
    byte instead, which only reflects this map's story state while the
    player is actually standing on it."""
    for line in lines:
        m = TRANSITION_STORE_RE.match(line)
        if m and m.group(1) != "wCurMapScript":
            return m.group(1)
    return None


def build_map_script(path: Path) -> tuple[int, dict] | None:
    map_id = pa.resolve_map_id(path)
    if map_id is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tables = find_state_tables(text)
    if not tables:
        return None
    table = tables[0]
    label_ranges = pa.find_label_ranges(lines)
    script_const_to_index = {const: idx for idx, (_h, const) in enumerate(table)}

    states: dict[int, dict] = {}
    for idx, (handler, _const) in enumerate(table):
        rng = label_ranges.get(handler)
        if rng is None:
            continue
        body_start = rng[0] + 1  # rng[0] is the label line itself
        guards, after_guards = parse_entry_guards(lines, body_start, rng[1])
        sets = events_set_in_range(lines, after_guards, rng[1])
        next_state = parse_transition(lines, after_guards, rng[1], script_const_to_index)
        states[idx] = {
            "handler": handler,
            "guards": guards,
            "sets": sorted(sets),
            "next_state": next_state,
        }
    return map_id, {"cur_script_ram": find_cur_script_var(lines), "states": states}


# ---------------------------------------------------------------------------
# object_event NPC/sign/trainer/item interactions and bg_event signs/PCs/etc
# (see module docstring). macros/scripts/maps.asm's object_event macro emits
# a different arg count per kind: plain NPC/sign ends right after its bare
# TEXT_* id (6 args); item adds one trailing ITEM_* (7 args); trainer adds
# trailing OPP_CLASS + level (8 args) -- three separate regexes so each
# shape only matches its own kind, never bleeds into another.
OBJECTS_DIR = POKERED_DIR / "data" / "maps" / "objects"
OBJECT_EVENT_RE = re.compile(
    r"^\tobject_event\s+(\d+),\s*(\d+),\s*\w+,\s*\w+,\s*\w+,\s*(TEXT_\w+)\s*$"
)
ITEM_OBJECT_EVENT_RE = re.compile(
    r"^\tobject_event\s+(\d+),\s*(\d+),\s*\w+,\s*\w+,\s*\w+,\s*(TEXT_\w+),\s*\w+\s*$"
)
TRAINER_OBJECT_EVENT_RE = re.compile(
    r"^\tobject_event\s+(\d+),\s*(\d+),\s*\w+,\s*\w+,\s*\w+,\s*(TEXT_\w+),\s*\w+,\s*\d+\s*$"
)
BG_EVENT_RE = re.compile(r"^\tbg_event\s+(\d+),\s*(\d+),\s*(TEXT_\w+)\s*$")
TEXT_TABLE_START_RE = re.compile(r"^\tdef_text_pointers\s*$")
# `ld hl, <Label>` immediately loading a trainer's own header address before
# handing off to the shared TalkToTrainer engine routine -- see
# find_reachable_set_events's trainer-header handling below.
LD_HL_LABEL_RE = re.compile(r"^\tld hl, (\w+)\s*(;.*)?$")


def find_object_events(text: str) -> list[tuple[int, int, str]]:
    """[(x, y, TEXT_CONST), ...] for every object_event (NPC/sign, item, or
    trainer) in a data/maps/objects/*.asm file. Item/trainer object_events
    used to be skipped entirely (their text ID isn't a bare TEXT_* the same
    way a sign's is), but that silently hid real cases -- a trainer's
    TEXT_CONST handler still `ld hl`s its own def_trainers header (see
    find_reachable_set_events), and some trainer AFTER-battle text bodies
    SetEvent an unrelated story flag directly (e.g. Route24's rocket fight
    handing out EVENT_GOT_NUGGET) that was invisible simply because the
    entry point -- the object_event itself -- was never even looked at."""
    lines = text.splitlines()
    out = []
    for line in lines:
        m = OBJECT_EVENT_RE.match(line) or ITEM_OBJECT_EVENT_RE.match(line) or TRAINER_OBJECT_EVENT_RE.match(line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def find_bg_events(text: str) -> list[tuple[int, int, str]]:
    """[(x, y, TEXT_CONST), ...] for every bg_event (sign/PC/bookshelf/etc
    -- an invisible, walkable-tile interaction, unlike object_event's
    solid NPC sprite) in a data/maps/objects/*.asm file."""
    return [
        (int(m.group(1)), int(m.group(2)), m.group(3))
        for m in (BG_EVENT_RE.match(line) for line in text.splitlines())
        if m
    ]


def find_text_handlers(text: str) -> dict[str, str]:
    """TEXT_CONST -> handler label, merged across every def_text_pointers
    table in this file (usually one)."""
    lines = text.splitlines()
    out: dict[str, str] = {}
    i = 0
    while i < len(lines):
        if TEXT_TABLE_START_RE.match(lines[i]):
            j = i + 1
            while j < len(lines):
                m = DW_CONST_RE.match(lines[j])
                if not m:
                    break
                out[m.group(2)] = m.group(1)
                j += 1
            i = j
        else:
            i += 1
    return out


def _events_from_state_onward(
    states: dict[int, dict],
    start_idx: int,
    max_steps: int = 30,
    known_satisfied_start: bool = False,
) -> set[str]:
    """Every event set by states[start_idx] or transitively via its
    next_state chain -- stops at a cycle (an already-visited state, e.g.
    OaksLab's states 6<->7 "wait for the player" loop), max_steps, or (for
    every state except possibly the first, see known_satisfied_start) a
    state with its own coordinate guard.

    A later state's own coordinate guard means reaching ITS sets needs the
    player independently at that other position too -- not implied merely
    by having reached this point in the chain, so walking further would
    silently over-attribute events to a location where they aren't actually
    reachable. Confirmed live bug this fixes: OaksLab's Poke Ball/Oak-talk
    object_event interactions transition into early dispatch states that
    eventually chain into OaksLabRivalChallengesPlayerScript (wYCoord==6)
    and beyond -- without stopping at that guard, EVENT_BATTLED_RIVAL_IN_
    OAKS_LAB (set two states further, unconditionally once the guard
    passes) got attributed to those interactions' own, already-used-up
    tiles too, alongside the real y==6 row, and a live BFS compass would
    happily lock onto whichever of the two was geometrically closer at any
    given moment -- including the wrong one.

    known_satisfied_start=True skips this check for start_idx itself only
    (used when start_idx's own coordinate guard IS the fact being modeled,
    e.g. this module's own COORD_TRIGGERS construction for a coord-guarded
    state -- see main()); find_reachable_set_events's object/bg-interaction
    call leaves it False, since merely performing an interaction never
    satisfies an unrelated coordinate gate."""
    events: set[str] = set()
    idx: int | None = start_idx
    seen: set[int] = set()
    steps = 0
    while idx is not None and idx not in seen and steps < max_steps:
        seen.add(idx)
        state = states.get(idx)
        if state is None:
            break
        skip_guard_check = known_satisfied_start and idx == start_idx
        if not skip_guard_check and any(g[0] == "coord" for g in state["guards"]):
            break
        events.update(state["sets"])
        idx = state["next_state"]
        steps += 1
    return events


def find_reachable_set_events(
    lines: list[str],
    label_ranges: dict[str, tuple[int, int]],
    start_label: str,
    states: dict[int, dict],
    script_const_to_index: dict[str, int],
    max_functions: int = 12,
) -> set[str]:
    """Every EVENTS-known name reachable from start_label's body, via two
    mechanisms: (a) a direct SetEvent*/CheckAndSetEvent* anywhere in the
    chain reached by following jr/jp targets that resolve to another known
    top-level label in the same file, and (b) a `ld a, SCRIPT_X` + `ld
    [w*CurScript], a` state transition found anywhere in that same chain --
    once the interaction hands control back to the map's own per-frame
    dispatch at a new state, whatever that state (or anything reachable
    from it via next_state) sets is just as reachable as a direct SetEvent
    in the interaction's own text (see module docstring's Oak's Lab example
    -- picking a Poke Ball doesn't SetEvent EVENT_GOT_STARTER itself, it
    transitions into OaksLabChoseStarterScript, which is what eventually
    leads there), and (c) a trainer's TEXT_CONST handler `ld hl`-ing its
    own def_trainers header (`trainer EVENT_BEAT_X, level, ...`) right
    before jumping into the shared TalkToTrainer engine routine -- the
    event name is a literal operand on that one line, so this only ever
    harvests it directly, never generically follows `ld hl` elsewhere
    (which would risk false edges from unrelated data-pointer loads).
    `call` targets are deliberately not followed (see module docstring)."""
    seen: set[str] = set()
    to_visit = [start_label]
    events: set[str] = set()
    while to_visit and len(seen) < max_functions:
        label = to_visit.pop()
        if label in seen or label not in label_ranges:
            continue
        seen.add(label)
        b_start, b_end = label_ranges[label]
        for lineno in range(b_start, b_end):
            line = lines[lineno]
            mm = pa.CALL_RE.match(line.strip())
            if mm and mm.group(1) in (pa.SET_MACROS | pa.IMPLICIT_SET_MACROS | pa.CHECK_AND_SET_MACROS):
                events.update(
                    n.strip() for n in mm.group(2).split(",") if n.strip() in pa.EVENTS
                )
            tm = TRANSITION_LD_A_RE.match(line)
            if tm and lineno + 1 < b_end and TRANSITION_STORE_RE.match(lines[lineno + 1]):
                target_idx = script_const_to_index.get(tm.group(1))
                if target_idx is not None:
                    events.update(_events_from_state_onward(states, target_idx))
            lhm = LD_HL_LABEL_RE.match(line)
            if lhm and lhm.group(1) in label_ranges:
                th_start, th_end = label_ranges[lhm.group(1)]
                for th_line in lines[th_start:th_end]:
                    tm2 = pa.CALL_RE.match(th_line.strip())
                    if tm2 and tm2.group(1) in pa.IMPLICIT_SET_MACROS:
                        events.update(
                            n.strip() for n in tm2.group(2).split(",") if n.strip() in pa.EVENTS
                        )
            cm = pa.COND_JUMP_RE.match(line)
            um = pa.UNCOND_JUMP_RE.match(line)
            target = cm.group(2).lstrip(".") if cm else (um.group(1).lstrip(".") if um else None)
            if target and target in label_ranges and target not in seen:
                to_visit.append(target)
    return events


def _adjacent_walkable_tiles(map_id: int, x: int, y: int) -> list[tuple[int, int, int]]:
    """The player interacts with an object_event by standing next to it and
    facing it, not by standing on it -- an NPC/sign sprite's own tile is
    almost always solid in the collision data (confirmed e.g. for Oak's
    Lab's Poke Balls at (6,3)/(7,3) in map 40, both is_walkable()==False),
    so routing straight to (x, y) leaves world_graph's BFS with no way to
    ever reach it and it silently falls back to some unrelated, far-away
    goal tile instead. Resolve to whichever of the 4 orthogonal neighbors
    are actually walkable (usually just one; some are approachable from
    multiple sides) so the BFS has a real node to route to."""
    return [
        (map_id, nx, ny)
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        if map_collision.is_walkable(map_id, nx, ny)
    ]


def _bg_event_tiles(map_id: int, x: int, y: int) -> list[tuple[int, int, int]]:
    """A bg_event's own tile is walkable by convention (the player stands on
    a sign/PC/bookshelf tile to read/use it, unlike a solid object_event
    NPC sprite) -- but fall back to an adjacent walkable tile on the rare
    chance the collision data disagrees, same safety net as object_events."""
    if map_collision.is_walkable(map_id, x, y):
        return [(map_id, x, y)]
    return _adjacent_walkable_tiles(map_id, x, y)


def _build_interaction_triggers(
    map_id: int,
    events: list[tuple[int, int, str]],
    tile_resolver,
    scripts_text: str,
    scripts_lines: list[str],
    states: dict[int, dict],
) -> dict[str, list[tuple[int, int, int]]]:
    """Shared by build_object_triggers/build_bg_triggers: resolve each
    (x, y, TEXT_CONST) interaction to its text handler, then to every event
    reachable from it, then to the routable tile(s) tile_resolver picks for
    that (map_id, x, y)."""
    text_handlers = find_text_handlers(scripts_text)
    label_ranges = pa.find_label_ranges(scripts_lines)
    script_const_to_index = {
        const: idx
        for table in find_state_tables(scripts_text)
        for idx, (_h, const) in enumerate(table)
    }
    out: dict[str, list[tuple[int, int, int]]] = {}
    for x, y, text_const in events:
        handler = text_handlers.get(text_const)
        if handler is None:
            continue
        tiles = tile_resolver(map_id, x, y)
        if not tiles:
            continue
        for ev in find_reachable_set_events(
            scripts_lines, label_ranges, handler, states, script_const_to_index
        ):
            out.setdefault(ev, []).extend(tiles)
    return out


def build_object_triggers(
    map_id: int,
    objects_text: str,
    scripts_text: str,
    scripts_lines: list[str],
    states: dict[int, dict],
) -> dict[str, list[tuple[int, int, int]]]:
    return _build_interaction_triggers(
        map_id, find_object_events(objects_text), _adjacent_walkable_tiles,
        scripts_text, scripts_lines, states,
    )


def build_bg_triggers(
    map_id: int,
    objects_text: str,
    scripts_text: str,
    scripts_lines: list[str],
    states: dict[int, dict],
) -> dict[str, list[tuple[int, int, int]]]:
    return _build_interaction_triggers(
        map_id, find_bg_events(objects_text), _bg_event_tiles,
        scripts_text, scripts_lines, states,
    )


def main() -> None:
    map_scripts: dict[int, dict] = {}
    files_scanned = 0
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        files_scanned += 1
        result = build_map_script(path)
        if result is not None:
            map_id, ms = result
            map_scripts[map_id] = ms

    coord_triggers: dict[str, list[tuple[int, int | None, int | None]]] = {}
    for map_id, ms in map_scripts.items():
        for idx, state in ms["states"].items():
            coord_guards = [g for g in state["guards"] if g[0] == "coord"]
            if not coord_guards:
                continue
            # Not just this state's OWN sets -- everything reachable via its
            # next_state chain too (same helper _build_interaction_triggers
            # already uses for object/bg interactions -- see
            # _events_from_state_onward). Confirmed live bug this fixes:
            # OaksLab's OaksLabRivalChallengesPlayerScript (state 10) gates
            # on wYCoord==6 but sets nothing itself -- the actual
            # `SetEvent EVENT_BATTLED_RIVAL_IN_OAKS_LAB` only happens two
            # states later (state 12, OaksLabRivalEndBattleScript, reached
            # automatically once the scripted battle resolves), so without
            # this the event had NO real coordinate trigger at all and
            # goal_positions.py's entrance-tile fallback pointed the compass
            # at Oak's Lab's door instead of the actual wYCoord==6 row.
            reachable_events = _events_from_state_onward(
                ms["states"], idx, known_satisfied_start=True
            )
            if not reachable_events:
                continue
            x = next((v for _k, ax, v in coord_guards if ax == "x"), None)
            y = next((v for _k, ax, v in coord_guards if ax == "y"), None)
            for ev in reachable_events:
                coord_triggers.setdefault(ev, []).append((map_id, x, y))

    object_events_found = 0
    maps_with_objects = 0
    bg_events_found = 0
    maps_with_bg_events = 0
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        objects_path = OBJECTS_DIR / path.name
        if not objects_path.is_file():
            continue
        scripts_text = path.read_text(encoding="utf-8", errors="replace")
        scripts_lines = scripts_text.splitlines()
        objects_text = objects_path.read_text(encoding="utf-8", errors="replace")
        states = map_scripts.get(map_id, {}).get("states", {})

        obj_events = find_object_events(objects_text)
        if obj_events:
            maps_with_objects += 1
            object_events_found += len(obj_events)
            for ev, tiles in build_object_triggers(
                map_id, objects_text, scripts_text, scripts_lines, states
            ).items():
                coord_triggers.setdefault(ev, []).extend(tiles)

        bg_events = find_bg_events(objects_text)
        if bg_events:
            maps_with_bg_events += 1
            bg_events_found += len(bg_events)
            for ev, tiles in build_bg_triggers(
                map_id, objects_text, scripts_text, scripts_lines, states
            ).items():
                coord_triggers.setdefault(ev, []).extend(tiles)

    py_lines = [
        '"""Per-map dispatch-table script graph, generated from pret/pokered',
        "static analysis -- see tools/gen_map_scripts.py for the parser and",
        'its caveats (same "hint, not proof" status as event_graph.py).',
        "",
        "MAP_SCRIPTS: map_id -> {cur_script_ram, states}. states: state_index ->",
        "{handler, guards, sets, next_state}. guards is an ordered chain of",
        '("event", NAME, required_bool) | ("coord", "x"|"y", value) gates.',
        "",
        "COORD_TRIGGERS: event name -> [(map_id, x_or_None, y_or_None), ...],",
        "derived from every state's own coordinate guard(s) PLUS every",
        "object_event (NPC/sign/item/trainer) and bg_event (sign/PC/etc)",
        "interaction that can reach it -- consumed by pokemon.goal_positions",
        "the same way pokemon.coord_triggers.py used to be.",
        "",
        "Regenerate with: python tools/gen_map_scripts.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        f"MAP_SCRIPTS: dict[int, dict] = {map_scripts!r}",
        "",
        "COORD_TRIGGERS: dict[str, list[tuple[int, int | None, int | None]]] = "
        + repr(coord_triggers),
        "",
    ]
    OUTPUT_PY.write_text("\n".join(py_lines), encoding="utf-8")

    total_states = sum(len(ms["states"]) for ms in map_scripts.values())
    dedicated = sum(1 for ms in map_scripts.values() if ms["cur_script_ram"])
    print(f"scanned {files_scanned} files")
    print(f"{len(map_scripts)} maps with a script table, {total_states} states total")
    print(f"{dedicated} maps have their own dedicated CurScript RAM var")
    print(f"{maps_with_objects} maps have object_event NPCs/signs/items/trainers, {object_events_found} of them total")
    print(f"{maps_with_bg_events} maps have bg_event signs/PCs/etc, {bg_events_found} of them total")
    print(f"{len(coord_triggers)} events have a coordinate-trigger entry (state guards + object/bg interactions)")
    print(f"wrote {OUTPUT_PY}")


if __name__ == "__main__":
    main()
