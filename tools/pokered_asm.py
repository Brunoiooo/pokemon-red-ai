"""Shared pret/pokered asm-parsing primitives for tools/gen_event_graph.py
and tools/gen_map_scripts.py -- pulled out to a common module so neither
generator has to import the other (they used to form
gen_event_graph <-> gen_map_scripts if gen_event_graph reused
gen_map_scripts' resolve_map_id/find_label_ranges the way
gen_coord_triggers.py used to reuse gen_event_graph's, since
gen_map_scripts also needs to be importable FROM gen_event_graph now for
the map-script-derived edges).

Not a generator itself -- has no `main()`, produces no output file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
POKERED_DIR = REPO_ROOT / "reference" / "pokered"
SCRIPTS_DIR = POKERED_DIR / "scripts"
ENGINE_DIR = POKERED_DIR / "engine"

sys.path.insert(0, str(REPO_ROOT / "src"))
from pokemon.event_constants import EVENTS, EVENTS_BY_INDEX  # noqa: E402
from pokemon.map_constants import MAPS  # noqa: E402

SET_MACROS = {
    "SetEvent", "SetEvents", "SetEventReuseHL",
    "SetEventAfterBranchReuseHL", "SetEventForceReuseHL",
}
CHECK_MACROS = {
    "CheckEvent", "CheckEventReuseA", "CheckEventAfterBranchReuseA",
    "CheckEventHL", "CheckEventReuseHL", "CheckEventForceReuseHL",
    "CheckEventAfterBranchReuseHL", "CheckBothEventsSet", "CheckEitherEventSet",
}
RESET_MACROS = {
    "ResetEvent", "ResetEvents", "ResetEventReuseHL",
    "ResetEventAfterBranchReuseHL", "ResetEventForceReuseHL",
}
# Both a read (test old value) and a write in one instruction.
CHECK_AND_SET_MACROS = {"CheckAndSetEvent", "CheckAndSetEventA"}
CHECK_AND_RESET_MACROS = {"CheckAndResetEvent", "CheckAndResetEventA"}
RANGE_SET_MACROS = {"SetEventRange"}
RANGE_RESET_MACROS = {"ResetEventRange"}
# Trainer-header macro: `trainer EVENT_X, level, ...` -- EVENT_X is set by
# the battle framework (CheckFightingMapTrainers) when that trainer is beaten.
IMPLICIT_SET_MACROS = {"trainer"}

ALL_EVENT_MACROS = (
    SET_MACROS | CHECK_MACROS | RESET_MACROS | CHECK_AND_SET_MACROS
    | CHECK_AND_RESET_MACROS | RANGE_SET_MACROS | RANGE_RESET_MACROS
    | IMPLICIT_SET_MACROS
)

CALL_RE = re.compile(
    r"^([A-Za-z]+)\s+(EVENT_[A-Za-z0-9_]+(?:\s*,\s*EVENT_[A-Za-z0-9_]+)*)"
)

# A top-level (function) label: unindented, not a local `.sublabel`, either
# `Name:` or the exported `Name::` form (see map_header's `\1_h::`).
TOP_LABEL_RE = re.compile(r"^([A-Za-z_]\w*)::?\s*$")


def normalize_to_map_name(stem: str) -> str:
    """PascalCase filename stem -> pokered SCREAMING_SNAKE_CASE map constant.

    e.g. 'CeladonMart1F' -> 'CELADON_MART_1F', 'SSAnne2FRooms' -> 'SS_ANNE_2F_ROOMS'.
    """
    s = stem
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)  # acronym boundary: SSAnne -> SS_Anne
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", s)  # word boundary: PewterGym -> Pewter_Gym
    s = re.sub(r"(?<=[a-z])(?=[0-9])", "_", s)  # lower->digit: Mart1F -> Mart_1F (not B1F -> B_1F)
    s = re.sub(r"(?<=[0-9])(?=[A-Z][a-z])", "_", s)  # digit->word (not digit->floor-letter)
    return s.upper()


def resolve_map_id(path: Path) -> int | None:
    if SCRIPTS_DIR not in path.parents:
        return None
    stem = path.stem
    candidates = [normalize_to_map_name(stem)]
    if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
        candidates.append(normalize_to_map_name(stem.rsplit("_", 1)[0]))
    for cand in candidates:
        if cand in MAPS:
            return MAPS[cand][0]
    return None


def find_label_ranges(lines: list[str]) -> dict[str, tuple[int, int]]:
    """label -> [start, end) 0-indexed line range, for every top-level label
    in this file -- a handler function's body is everything from its own
    label up to (not including) the next top-level label, or EOF."""
    label_lines = [
        (idx, m.group(1)) for idx, line in enumerate(lines)
        if (m := TOP_LABEL_RE.match(line))
    ]
    ranges: dict[str, tuple[int, int]] = {}
    for k, (idx, name) in enumerate(label_lines):
        end = label_lines[k + 1][0] if k + 1 < len(label_lines) else len(lines)
        ranges[name] = (idx, end)
    return ranges


def build_line_owner(lines: list[str]) -> list[str | None]:
    """1-indexed line number -> enclosing top-level label's name (None for
    lines before the first label). A CheckEvent in one top-level function
    has no business gating a SetEvent in a different one (see
    tools/gen_event_graph.py's module docstring for the real false-positive
    this caught)."""
    ranges = find_label_ranges(lines)
    owner: list[str | None] = [None] * (len(lines) + 1)
    for name, (start, end) in ranges.items():
        for i in range(start, end):
            owner[i + 1] = name  # 0-indexed range -> 1-indexed line numbers
    return owner


def parse_file_calls(path: Path) -> list[tuple[int, str, list[str]]]:
    calls = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        m = CALL_RE.match(line)
        if not m:
            continue
        macro = m.group(1)
        if macro not in ALL_EVENT_MACROS:
            continue
        names = [a.strip() for a in m.group(2).split(",")]
        calls.append((lineno, macro, names))
    return calls


def expand_calls(calls: list[tuple[int, str, list[str]]]) -> list[tuple[int, str, list[str]]]:
    """Normalize every call into (lineno, kind, [event names]), kind in
    {set, check, reset}, expanding ranges and multi-effect macros."""
    out: list[tuple[int, str, list[str]]] = []
    for lineno, macro, names in calls:
        if macro in RANGE_SET_MACROS or macro in RANGE_RESET_MACROS:
            if len(names) >= 2 and names[0] in EVENTS and names[1] in EVENTS:
                lo, hi = EVENTS[names[0]], EVENTS[names[1]]
                if lo > hi:
                    lo, hi = hi, lo
                ev_names = [EVENTS_BY_INDEX[i] for i in range(lo, hi + 1) if i in EVENTS_BY_INDEX]
            else:
                ev_names = [n for n in names if n in EVENTS]
            kind = "set" if macro in RANGE_SET_MACROS else "reset"
            if ev_names:
                out.append((lineno, kind, ev_names))
            continue

        valid = [n for n in names if n in EVENTS]
        if not valid:
            continue

        if macro in SET_MACROS or macro in IMPLICIT_SET_MACROS:
            out.append((lineno, "set", valid))
        elif macro in RESET_MACROS:
            out.append((lineno, "reset", valid))
        elif macro in CHECK_MACROS:
            out.append((lineno, "check", valid))
        elif macro in CHECK_AND_SET_MACROS:
            out.append((lineno, "check", valid))
            out.append((lineno, "set", valid))
        elif macro in CHECK_AND_RESET_MACROS:
            out.append((lineno, "check", valid))
            out.append((lineno, "reset", valid))
    return out


def iter_source_files():
    yield from sorted(SCRIPTS_DIR.glob("*.asm"))
    yield from sorted(ENGINE_DIR.rglob("*.asm"))


# ---------------------------------------------------------------------------
# Intra-function control-flow analysis: which EVENT_*/coordinate guards are
# unavoidable on every path from a function's entry to a given SetEvent,
# tracing actual branches (jr/jp/ret + their targets) instead of assuming
# code is a single straight-line "check then bail" chain. Built specifically
# because that assumption missed a real dependency: BluesHouse.asm's
# BluesHouseDaisySittingText only gives the Town Map once you already have
# the Pokedex, via `CheckEvent EVENT_GOT_POKEDEX / jr nz, .give_town_map`
# (SetEvent EVENT_GOT_TOWN_MAP lives inside .give_town_map) -- a
# check-then-jump-to-a-different-branch-that-still-SetEvents shape, not a
# linear bail. Still a heuristic, not a real compiler pass: `call`s are
# assumed to always return (no non-local exits), and any function whose
# local-label graph contains a back-edge (a loop) is skipped entirely
# (returns {} for it) rather than risk a wrong guard on a path that can
# re-enter itself -- same "prefer missing over wrong" policy as everything
# else derived from static analysis in this project.
LOCAL_LABEL_RE = re.compile(r"^\.(\w+)\s*$")
_JUMP_TARGET = r"(\.\w+|[A-Za-z_]\w*)"
COND_JUMP_RE = re.compile(r"^\t(?:jr|jp)\s+(z|nz|c|nc)\s*,\s*" + _JUMP_TARGET + r"\s*$")
UNCOND_JUMP_RE = re.compile(r"^\t(?:jr|jp)\s+" + _JUMP_TARGET + r"\s*$")
COND_RET_RE = re.compile(r"^\tret\s+(z|nz|c|nc)\s*$")
UNCOND_RET_RE = re.compile(r"^\tret(i)?\s*$")
CHECK_EVENT_RE = re.compile(r"^\tCheckEvent\s+(EVENT_\w+)\s*$")
COORD_LOAD_RE = re.compile(r"^\tld a, \[(wYCoord|wXCoord)\]\s*(;.*)?$")
CP_RE = re.compile(r"^\tcp\s+(-?\d+)")

_EXIT = object()  # sentinel: function exit (ret), not a real block


def _split_blocks(lines: list[str], start: int, end: int) -> list[int]:
    """Sorted cut points (0-indexed line numbers) splitting [start, end)
    into basic blocks -- a new block starts at every local label and at
    every line immediately after a jump/return, so each block's own last
    line is always its sole terminator (or it has none -> implicit
    fall-through to the next block)."""
    cuts = {start}
    for i in range(start, end):
        line = lines[i]
        if LOCAL_LABEL_RE.match(line):
            cuts.add(i)
        if (COND_JUMP_RE.match(line) or UNCOND_JUMP_RE.match(line)
                or COND_RET_RE.match(line) or UNCOND_RET_RE.match(line)):
            if i + 1 < end:
                cuts.add(i + 1)
    return sorted(cuts)


def _guard_for_check(lines: list[str], before_idx: int) -> tuple[str, str, int] | tuple[str, str] | None:
    """Look at the 1-2 lines immediately before a conditional jump/return
    (i.e. lines[before_idx] is that jump/return) for the CheckEvent or
    wYCoord/wXCoord+cp pattern that sets up the tested flag. Returns a
    ("event", NAME) or ("coord", "x"|"y", value) tuple (the required_bool
    for events is filled in by the caller based on which edge -- taken or
    fall-through -- is being labeled), or None if nothing recognizable
    immediately precedes it."""
    if before_idx - 1 < 0:
        return None
    m_ev = CHECK_EVENT_RE.match(lines[before_idx - 1])
    if m_ev:
        return ("event", m_ev.group(1))
    if before_idx - 2 >= 0:
        m_co = COORD_LOAD_RE.match(lines[before_idx - 2])
        cp_m = CP_RE.match(lines[before_idx - 1]) if m_co else None
        if m_co and cp_m:
            axis = "x" if m_co.group(1) == "wXCoord" else "y"
            return ("coord", axis, int(cp_m.group(1)))
    return None


def analyze_function_event_guards(
    lines: list[str], start: int, end: int
) -> dict[str, frozenset[tuple]]:
    """event_name -> frozenset of ("event", NAME, required_bool) |
    ("coord", "x"|"y", value) guards proven unavoidable, via real branch
    tracing, on every path from this function's entry to a SetEvent* of
    that name within lines[start:end]. {} if the function's local-label
    graph has a back-edge (loop) -- see module-level caveat above."""
    cut_points = _split_blocks(lines, start, end)
    block_at: dict[int, int] = {c: i for i, c in enumerate(cut_points)}  # start-line -> block index
    label_to_block: dict[str, int] = {}
    for i, c in enumerate(cut_points):
        m = LOCAL_LABEL_RE.match(lines[c])
        if m:
            label_to_block[m.group(1)] = i

    n = len(cut_points)

    def block_end(i: int) -> int:
        return cut_points[i + 1] if i + 1 < n else end

    # edges[i] = list of (target_block_index_or__EXIT, guard_or_None)
    edges: list[list[tuple]] = [[] for _ in range(n)]
    for i in range(n):
        b_start, b_end = cut_points[i], block_end(i)
        if b_end <= b_start:
            continue
        last = lines[b_end - 1]
        fallthrough = i + 1 if i + 1 < n else None
        m = COND_JUMP_RE.match(last)
        if m:
            cond, target = m.group(1), m.group(2)
            tgt_block = label_to_block.get(target.lstrip("."))
            guard = _guard_for_check(lines, b_end - 1)
            taken_guard = fall_guard = None
            if guard and guard[0] == "event":
                # CheckEvent: Z means FALSE, NZ means TRUE.
                taken_true = cond in ("z", "c")  # taken -> event is FALSE for z/c-family used after CheckEvent
                # cond is one of z/nz (event checks always use z/nz, not c/nc)
                if cond == "z":
                    taken_guard = ("event", guard[1], False)
                    fall_guard = ("event", guard[1], True)
                elif cond == "nz":
                    taken_guard = ("event", guard[1], True)
                    fall_guard = ("event", guard[1], False)
            elif guard and guard[0] == "coord":
                if cond == "z":
                    taken_guard = guard  # taken -> coordinate == value
                # nz-taken (coordinate != value) has no single-value representation; leave unguarded
            if tgt_block is not None:
                edges[i].append((tgt_block, taken_guard))
            if fallthrough is not None:
                edges[i].append((fallthrough, fall_guard))
            continue
        m = UNCOND_JUMP_RE.match(last)
        if m:
            tgt_block = label_to_block.get(m.group(1).lstrip("."))
            if tgt_block is not None:
                edges[i].append((tgt_block, None))
            continue
        m = COND_RET_RE.match(last)
        if m:
            cond = m.group(1)
            guard = _guard_for_check(lines, b_end - 1)
            fall_guard = None
            if guard and guard[0] == "event":
                if cond == "z":
                    fall_guard = ("event", guard[1], True)
                elif cond == "nz":
                    fall_guard = ("event", guard[1], False)
            edges[i].append((_EXIT, None))
            if fallthrough is not None:
                edges[i].append((fallthrough, fall_guard))
            continue
        if UNCOND_RET_RE.match(last):
            edges[i].append((_EXIT, None))
            continue
        if fallthrough is not None:
            edges[i].append((fallthrough, None))

    # Back-edge (loop) detection: a real DFS cycle in the block graph ->
    # bail out entirely for this function, per the module-level caveat.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def has_cycle(i: int) -> bool:
        color[i] = GRAY
        for tgt, _g in edges[i]:
            if tgt is _EXIT:
                continue
            if color[tgt] == GRAY:
                return True
            if color[tgt] == WHITE and has_cycle(tgt):
                return True
        color[i] = BLACK
        return False

    if n and has_cycle(0):
        return {}

    # Forward dataflow: guards_at[block] = meet (intersection) over every
    # predecessor path's (predecessor's guards | this edge's own guard).
    # Topological order via Kahn's algorithm (safe: no cycle, checked above).
    indeg = [0] * n
    preds: list[list[tuple[int, tuple | None]]] = [[] for _ in range(n)]
    for i in range(n):
        for tgt, g in edges[i]:
            if tgt is _EXIT:
                continue
            preds[tgt].append((i, g))
            indeg[tgt] += 1

    guards_at: dict[int, frozenset] = {0: frozenset()}
    order = [0]
    queue = [j for j in range(n) if j != 0]
    # simple stable topological sort by repeatedly picking a ready node
    remaining = set(queue)
    progressed = True
    while remaining and progressed:
        progressed = False
        for j in sorted(remaining):
            if all((p not in remaining) for p, _g in preds[j]):
                order.append(j)
                remaining.discard(j)
                progressed = True
    order.extend(sorted(remaining))  # any leftovers (shouldn't happen, no cycle) -- best effort

    for j in order[1:]:
        if not preds[j]:
            continue  # unreachable from entry via a traced edge -- no guard info
        options = []
        for p, g in preds[j]:
            if p not in guards_at:
                continue
            base = guards_at[p]
            options.append(base | {g} if g is not None else base)
        if options:
            acc = options[0]
            for o in options[1:]:
                acc = acc & o
            guards_at[j] = acc

    result: dict[str, frozenset[tuple]] = {}
    for i in range(n):
        if i not in guards_at:
            continue
        b_start, b_end = cut_points[i], block_end(i)
        for lineno in range(b_start, b_end):
            mm = CALL_RE.match(lines[lineno].strip())
            # CHECK_AND_SET_MACROS (e.g. `CheckAndSetEvent EVENT_X` --
            # OaksLab.asm's "give spare poke balls" text uses exactly this
            # for EVENT_GOT_POKEBALLS_FROM_OAK) perform a set as a side
            # effect too, same as expand_calls already treats them
            # elsewhere in this project -- omitting them here silently
            # blinded this pass to every event only ever set that way.
            if mm and mm.group(1) in (SET_MACROS | IMPLICIT_SET_MACROS | CHECK_AND_SET_MACROS):
                for nm in mm.group(2).split(","):
                    nm = nm.strip()
                    if nm in EVENTS:
                        result[nm] = result.get(nm, frozenset()) | guards_at[i]
    return result
