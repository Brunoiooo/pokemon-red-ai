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
  2. Infers parent -> child edges via real intra-function control-flow
     analysis (see pokered_asm.analyze_function_event_guards), one
     top-level function at a time: builds that function's actual basic-
     block graph from its jr/jp/ret branches (not just an assumed linear
     "check then bail" chain), propagates which CheckEvent/coordinate
     guards are unavoidable on every path reaching each SetEvent, and
     keeps only the `required=True` event guards as parent edges (a
     `required=False` guard -- an idempotency check like "don't re-fire if
     this already happened" -- has no representation in this graph's
     true-only parents list, so it's dropped, never inverted into a wrong
     edge). This replaced an earlier, much simpler line-proximity-with-
     lookback heuristic specifically because it missed real dependencies
     written as a branch rather than a bail -- e.g. BluesHouse.asm's Daisy
     only gives the Town Map once you already have the Pokedex, via
     `CheckEvent EVENT_GOT_POKEDEX / jr nz, .give_town_map` with the
     SetEvent living inside `.give_town_map`, a shape the old heuristic
     structurally could not see. Still a static-analysis heuristic, not a
     real compiler pass (`call`s are assumed to always return; any
     function whose local-label graph has a back-edge/loop is skipped
     entirely rather than risk a wrong guard) -- verify visually before
     trusting an edge for reward/curriculum logic.
  3. `SetEventRange`/`ResetEventRange` expand to every individually-named
     event whose global bit index falls in the given [start, end] range.
  4. `trainer EVENT_X, ...` headers are treated as an implicit "set" site
     (CheckFightingMapTrainers sets EVENT_X when that trainer is beaten,
     with no literal SetEvent call in the script).
  5. Also infers edges from every map's dispatch-table script graph (see
     tools/gen_map_scripts.py, imported here) -- within one
     `def_script_pointers` table, a state's own required event-guards, plus
     everything set by any earlier state, are treated as prerequisites for
     whatever that state itself sets. This catches cutscene chains with NO
     CheckEvent anywhere near the SetEvent at all (e.g. OaksLab.asm's
     EVENT_GOT_STARTER, gated purely by state order, not any check), which
     (2) can never find regardless of MAX_LOOKBACK. Same "hint, not proof"
     caveat as (2): a table whose states can legitimately jump backward
     (a puzzle that resets on a wrong answer, say) would produce a
     spurious edge here -- but find_cyclic_events already exists to catch
     and blacklist exactly that shape of contradiction, so a wrong edge
     from this heuristic is self-correcting, not silently trusted.
  6. Also merges tools/event_graph_manual_edges.py -- a hand-maintained
     safety valve for whatever (2)/(5) still can't see (a `call` into a
     shared helper that itself branches, a condition tested via something
     other than CheckEvent/wYCoord/wXCoord, ...). Empty unless a human
     spots a real gap while reading a script (tools/view_script_graph.py's
     graph viewer is built for exactly this) and adds it by hand. Goes
     through the same cycle-detection/blacklist safety net as every other
     edge -- a wrong manual entry is still caught, not silently trusted.

Outputs:
  - src/pokemon/event_graph.py   (importable at runtime)
  - src/pokemon/event_graph.json (for the visualization artifact)

Usage: python tools/gen_event_graph.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import event_graph_manual_edges as manual_edges_module  # noqa: E402
import gen_map_scripts as gms  # noqa: E402
import pokered_asm as pa  # noqa: E402

REPO_ROOT = pa.REPO_ROOT
POKERED_DIR = pa.POKERED_DIR
SCRIPTS_DIR = pa.SCRIPTS_DIR
ENGINE_DIR = pa.ENGINE_DIR
OUTPUT_PY = REPO_ROOT / "src" / "pokemon" / "event_graph.py"
OUTPUT_JSON = REPO_ROOT / "src" / "pokemon" / "event_graph.json"

EVENTS = pa.EVENTS
resolve_map_id = pa.resolve_map_id
parse_file_calls = pa.parse_file_calls
expand_calls = pa.expand_calls
iter_source_files = pa.iter_source_files


def add_map_script_edges(nodes: dict[str, dict], ms: dict) -> int:
    """Add parent/child edges derived from one map's map_scripts.py-parsed
    dispatch table (see this module's docstring item 5) directly into
    `nodes`. For each state, in table order: its own required event-guards
    plus everything set (or required) by any earlier state are parents of
    whatever it itself sets (the state machine can't reach state N without
    having already passed through state N-1's own SetEvents/guards and
    state N's own guards). Required guards propagate forward through the
    accumulator even for a state that itself sets nothing (e.g. OaksLab's
    state 0 requires EVENT_OAK_APPEARED_IN_PALLET but sets no event of its
    own -- every later state still needs that requirement to have been an
    ancestor). Returns how many new edges were added."""
    added = 0
    earlier_events: set[str] = set()
    for idx in sorted(ms["states"]):
        state = ms["states"][idx]
        own_events = set(state["sets"])
        required_guards = {
            name for kind, name, required in state["guards"]
            if kind == "event" and required
        }
        parents = earlier_events | required_guards
        for child in own_events:
            if child not in nodes:
                continue
            for parent in parents:
                if parent == child or parent not in nodes:
                    continue
                if child not in nodes[parent]["children"]:
                    added += 1
                nodes[parent]["children"].add(child)
                nodes[child]["parents"].add(parent)
        earlier_events |= own_events | required_guards
    return added


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
        file_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

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

        # Per-function control-flow-aware guard analysis (see
        # pokered_asm.analyze_function_event_guards) -- traces real branches
        # instead of assuming a linear "check then bail" chain, so it also
        # catches "check, then jump to a different branch that still
        # SetEvents" shapes a line-proximity heuristic can't (e.g.
        # BluesHouse.asm's Daisy only gives the Town Map once you already
        # have the Pokedex, via `CheckEvent EVENT_GOT_POKEDEX / jr nz,
        # .give_town_map` with the SetEvent inside .give_town_map). Only
        # required=True event guards become parent edges -- a required=False
        # guard (an idempotency check, e.g. "don't re-fire if this event
        # already happened") has no representation in this graph's
        # true-only parents list, so it's dropped, never inverted into a
        # wrong edge.
        for _label, (fn_start, fn_end) in pa.find_label_ranges(file_lines).items():
            guards_by_event = pa.analyze_function_event_guards(file_lines, fn_start, fn_end)
            for child, guards in guards_by_event.items():
                if child not in nodes:
                    continue
                for guard in guards:
                    if guard[0] != "event":
                        continue
                    _kind, parent, required = guard
                    if not required or parent == child or parent not in nodes:
                        continue
                    nodes[parent]["children"].add(child)
                    nodes[child]["parents"].add(parent)

    map_script_edges = 0
    maps_with_scripts = 0
    for path in sorted(SCRIPTS_DIR.glob("*.asm")):
        result = gms.build_map_script(path)
        if result is None:
            continue
        maps_with_scripts += 1
        _map_id, ms = result
        map_script_edges += add_map_script_edges(nodes, ms)

    manual_edges = 0
    for child, parents in manual_edges_module.MANUAL_PARENT_EDGES.items():
        if child not in nodes:
            continue
        for parent in parents:
            if parent not in nodes or parent == child:
                continue
            if child not in nodes[parent]["children"]:
                manual_edges += 1
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
        "inference heuristics (CheckEvent-before-SetEvent proximity within a file,",
        "plus tools/gen_map_scripts.py's per-map dispatch-table graph).",
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
    print(f"{maps_with_scripts} maps with a dispatch table, {map_script_edges} edges added from them")
    print(f"{manual_edges} manual edges merged from tools/event_graph_manual_edges.py")
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
