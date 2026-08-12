#!/usr/bin/env python
"""Build an event dependency graph from the pret/pokered disassembly.

For every scripts/*.asm and engine/**/*.asm file, extracts every
SetEvent*/CheckEvent*/ResetEvent*/trainer call referencing a named
EVENT_* constant (see macros/scripts/events.asm for the macro catalogue),
then:

  1. Resolves the owning map_id for scripts/<MapName>.asm files by
     normalizing the filename to pokered's SCREAMING_SNAKE_CASE map-constant
     convention and looking it up in map_constants.py. Engine-side files
     (not per-map) get map_id = None.
  2. Infers parent -> child edges: within one file, a `SetEvent CHILD` whose
     nearest preceding `CheckEvent PARENT` (a *different* event, within
     MAX_LOOKBACK lines) is treated as gated by PARENT -- i.e. PARENT is a
     prerequisite context for CHILD. This is a static-analysis heuristic
     (same pattern PewterGym.asm's Brock fight showed by hand: CheckEvent
     EVENT_BEAT_BROCK guards the TM34 hand-off script that SetEvents
     EVENT_GOT_TM34), not a proof of causality -- verify visually before
     trusting an edge.
  3. `SetEventRange`/`ResetEventRange` expand to every individually-named
     event whose global bit index falls in the given [start, end] range.
  4. `trainer EVENT_X, ...` headers are treated as an implicit "set" site
     (CheckFightingMapTrainers sets EVENT_X when that trainer is beaten,
     with no literal SetEvent call in the script).

Outputs:
  - src/pokemon/event_graph.py   (importable at runtime)
  - src/pokemon/event_graph.json (for the visualization artifact)

Usage: python tools/gen_event_graph.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POKERED_DIR = REPO_ROOT / "reference" / "pokered"
SCRIPTS_DIR = POKERED_DIR / "scripts"
ENGINE_DIR = POKERED_DIR / "engine"
OUTPUT_PY = REPO_ROOT / "src" / "pokemon" / "event_graph.py"
OUTPUT_JSON = REPO_ROOT / "src" / "pokemon" / "event_graph.json"

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

MAX_LOOKBACK = 80  # lines; how far back a CheckEvent can still "gate" a SetEvent


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


def find_cyclic_events(nodes: dict[str, dict]) -> set[str]:
    """Events on any parent->child cycle (3-color DFS over children edges).

    A cycle here is a toggle, not progress: e.g. the Vermilion Gym trash-can
    puzzle's EVENT_1ST_LOCK_OPENED / EVENT_2ND_LOCK_OPENED flip each other on
    a wrong guess (see engine/events/hidden_events/vermilion_gym_trash.asm),
    so "reaching" one is not a stable, one-way milestone.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    cyclic: set[str] = set()
    path: list[str] = []

    def dfs(n: str) -> None:
        color[n] = GRAY
        path.append(n)
        for c in nodes[n]["children"]:
            if c not in nodes:
                continue
            if color[c] == GRAY:
                idx = path.index(c)
                cyclic.update(path[idx:])
            elif color[c] == WHITE:
                dfs(c)
        path.pop()
        color[n] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return cyclic


def main() -> None:
    nodes: dict[str, dict] = {
        name: {
            "index": idx,
            "map_id": None,
            "set_in": [],
            "checked_in": [],
            "reset_in": [],
            "parents": set(),
            "children": set(),
        }
        for name, idx in EVENTS.items()
    }
    # (event, kind) -> first map_id seen, in file-scan order. "set" occurrences
    # win over "check"/"reset"-only sightings when picking each node's home map
    # (a gym script that merely CheckEvents a flag it doesn't own shouldn't
    # steal home-map credit from the map that actually SetEvents it).
    map_touches: dict[str, list[tuple[int, str]]] = {name: [] for name in EVENTS}

    unresolved_script_files: list[str] = []
    files_scanned = 0

    for path in iter_source_files():
        files_scanned += 1
        rel = path.relative_to(POKERED_DIR).as_posix()
        map_id = resolve_map_id(path)
        if SCRIPTS_DIR in path.parents and map_id is None:
            unresolved_script_files.append(rel)

        expanded = expand_calls(parse_file_calls(path))

        recent_checks: list[tuple[int, str]] = []
        for lineno, kind, names in expanded:
            for name in names:
                node = nodes[name]
                if map_id is not None:
                    map_touches[name].append((map_id, kind))
                if kind == "set":
                    node["set_in"].append(f"{rel}:{lineno}")
                elif kind == "check":
                    node["checked_in"].append(f"{rel}:{lineno}")
                elif kind == "reset":
                    node["reset_in"].append(f"{rel}:{lineno}")

            if kind == "check":
                recent_checks.extend((lineno, n) for n in names)
            elif kind == "set":
                recent_checks = [
                    (ln, n) for ln, n in recent_checks if lineno - ln <= MAX_LOOKBACK
                ]
                for child in names:
                    for _ln, parent in recent_checks:
                        if parent != child:
                            nodes[parent]["children"].add(child)
                            nodes[child]["parents"].add(parent)

    # Resolve each node's home map: prefer the map that actually SetEvents it;
    # fall back to wherever it's merely CheckEvent/ResetEvent'd if never set.
    for name, touches in map_touches.items():
        if not touches:
            continue
        set_touches = [mid for mid, kind in touches if kind == "set"]
        nodes[name]["map_id"] = set_touches[0] if set_touches else touches[0][0]

    # Freeze sets -> sorted lists for stable, JSON-serializable output.
    for node in nodes.values():
        node["parents"] = sorted(node["parents"])
        node["children"] = sorted(node["children"])

    roots = [n for n, d in nodes.items() if not d["parents"] and (d["set_in"] or d["children"])]
    dead = [n for n, d in nodes.items() if not d["set_in"] and not d["reset_in"] and not d["checked_in"]]
    cyclic = find_cyclic_events(nodes)
    resettable = [n for n, d in nodes.items() if d["reset_in"]]
    # Auto-blacklist for goal candidacy: dead (can never be satisfied),
    # cyclic (toggles, not one-way progress), resettable (can un-happen, so
    # "reached" doesn't mean "stays true" -- see e.g. EVENT_MANSION_SWITCH_ON,
    # EVENT_BILL_SAID_USE_CELL_SEPARATOR, EVENT_SAFARI_GAME_OVER audited by
    # hand earlier). Union, not intersection -- any one reason disqualifies.
    blacklist = sorted(set(dead) | cyclic | set(resettable))

    commit = subprocess.run(
        ["git", "-C", str(POKERED_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    OUTPUT_JSON.write_text(json.dumps(nodes, indent=1, sort_keys=True), encoding="utf-8")

    py_lines = [
        '"""Event dependency graph, generated from pret/pokered static analysis.',
        "",
        f"Source: https://github.com/pret/pokered @ {commit}",
        "Built by tools/gen_event_graph.py -- see that file for the parent/child",
        "inference heuristic (CheckEvent-before-SetEvent proximity within a file).",
        "This is NOT a proof of causal dependency, only a static-analysis hint --",
        "verify visually (see src/pokemon/event_graph.json + the graph viewer)",
        "before trusting an edge for reward/curriculum logic.",
        "Regenerate with: python tools/gen_event_graph.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "# event name -> {index, map_id, set_in, checked_in, reset_in, parents, children}",
        "EVENT_GRAPH: dict[str, dict] = " + repr(nodes),
        "",
        f"ROOT_EVENTS = {roots!r}  # no inferred parent, but set/referenced somewhere",
        f"DEAD_EVENTS = {dead!r}  # never set, checked, or reset anywhere in the ROM",
        f"CYCLIC_EVENTS = {sorted(cyclic)!r}  # on a parent<->child cycle -- a toggle, not one-way progress",
        f"RESETTABLE_EVENTS = {sorted(resettable)!r}  # has a ResetEvent* site somewhere -- can un-happen",
        "",
        "# Auto blacklist for goal candidacy: DEAD_EVENTS | CYCLIC_EVENTS | RESETTABLE_EVENTS.",
        "# Extend with a project-specific manual set at the call site if the auto rules",
        "# miss something (see src/pokemon/Data.py's GOAL_MANUAL_BLACKLIST).",
        f"AUTO_BLACKLIST_EVENTS = {blacklist!r}",
        "",
    ]
    OUTPUT_PY.write_text("\n".join(py_lines), encoding="utf-8")

    print(f"scanned {files_scanned} files")
    print(f"{len(nodes)} events total, {sum(1 for d in nodes.values() if d['map_id'] is not None)} with a resolved map_id")
    print(f"{len(dead)} dead events (never set/checked/reset anywhere)")
    print(f"{len(cyclic)} cyclic events (toggle loops)")
    print(f"{len(resettable)} resettable events (have a ResetEvent* site)")
    print(f"{len(blacklist)} total auto-blacklisted ({len(nodes) - len(blacklist)} remain as goal candidates)")
    print(f"{len(unresolved_script_files)} scripts/*.asm files did not resolve to a map name:")
    for f in unresolved_script_files:
        print(f"  {f}")
    print(f"wrote {OUTPUT_PY}")
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
