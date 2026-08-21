#!/usr/bin/env python
"""Regenerate src/pokemon/map_warps.py from the pret/pokered disassembly.

Static per-map warp graph -- the discrete, non-adjacent map transitions
(doors, staircases, cave mouths) as opposed to continuous outdoor edge-walks
(Route <-> City seams via data/maps/headers/*.asm's `connection` lines,
which this does NOT cover -- see the module docstring gap note below).

Source data:
  - data/maps/objects/<File>.asm's `def_warp_events` / `warp_event x, y, dest,
    warp_id` lines (macros/scripts/maps.asm's `warp_event` macro: dest map
    -1 means LAST_MAP -- "whichever map the player physically walked in
    from", resolved at runtime from wLastMap, not fixed by this file alone).
  - <File>.asm reuses the same file-stem <-> MAP_CONST link
    tools/gen_map_collision.py already builds from data/maps/headers/*.asm
    (reused here via that module's parse_map_header).

LAST_MAP resolution: for a LAST_MAP warp on map M, this looks at every OTHER
map's warp_events for ones that target M, and only resolves it if exactly
one distinct source map does so (the overwhelmingly common case: a building
with a single door). If zero or multiple maps warp into M, the edge is
dropped rather than guessed -- it simply won't appear as a MAP_WARPS entry,
same as an unimplemented connection. main() prints how many were dropped.

Known gap: outdoor Route<->City edge-walks (no warp_event involved at all,
just walking off a map's border) are not covered by this graph. Those are
already tracked live and empirically by Data.map_transitions once an agent
has actually crossed them; this file is only about the discrete jumps
(building/floor/cave warps) relevant to static per-map collision analysis
and cross-map routing between indoor destinations.

Usage: python tools/gen_map_warps.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POKERED_DIR = REPO_ROOT / "reference" / "pokered"
OUTPUT_PATH = REPO_ROOT / "src" / "pokemon" / "map_warps.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from pokemon.map_constants import MAPS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_map_collision import parse_map_header  # noqa: E402

WARP_EVENT_RE = re.compile(r"^\twarp_event\s+(-?\d+),\s*(-?\d+),\s*(\w+),\s*(\d+)", re.M)


def parse_warp_events(path: Path) -> list[tuple[int, int, str, int]]:
    """Returns (x, y, dest_map_const_or_'LAST_MAP', dest_warp_id (1-based)) in file order."""
    out = []
    for m in WARP_EVENT_RE.finditer(path.read_text(encoding="utf-8")):
        x, y, dest, warp_id = m.groups()
        out.append((int(x), int(y), dest, int(warp_id)))
    return out


def main() -> None:
    header_dir = POKERED_DIR / "data" / "maps" / "headers"
    map_const_to_stem: dict[str, str] = {}
    for header_path in sorted(header_dir.glob("*.asm")):
        parsed = parse_map_header(header_path)
        if parsed is None:
            continue
        map_const, _tileset = parsed
        map_const_to_stem[map_const] = header_path.stem

    objects_dir = POKERED_DIR / "data" / "maps" / "objects"

    # map_const -> raw warp list (dest may literally be the string 'LAST_MAP').
    raw_warps: dict[str, list[tuple[int, int, str, int]]] = {}
    skipped: list[str] = []
    for map_const, stem in map_const_to_stem.items():
        obj_path = objects_dir / f"{stem}.asm"
        if not obj_path.exists():
            skipped.append(f"{map_const}: no {obj_path.name}")
            continue
        raw_warps[map_const] = parse_warp_events(obj_path)

    # Reverse index keyed by the EXACT (dest_map_const, dest_warp_id) slot a
    # warp targets -- not just dest_map_const alone. A map can have several
    # unrelated inbound warps landing on different slots (e.g. a house's
    # front door lands on slot 1, its own upstairs lands on slot 3); indexing
    # by map alone would wrongly treat those as competing sources for the
    # *same* LAST_MAP exit and call it ambiguous when it isn't ($sources for
    # slot 1 is just the front door). LAST_MAP entries don't name a concrete
    # source so they're excluded from this index (they're resolved below,
    # not used as a source themselves).
    warps_into: dict[tuple[str, int], set[str]] = {}
    for src_const, warps in raw_warps.items():
        for _x, _y, dest, warp_id in warps:
            if dest == "LAST_MAP":
                continue
            warps_into.setdefault((dest, warp_id), set()).add(src_const)

    # Pass 1: resolve each LAST_MAP slot directly from warps_into, when exactly
    # one map warps into that exact slot.
    resolved_last_map: dict[tuple[str, int], str] = {}
    unresolved_slots: list[tuple[str, int, int]] = []  # (map_const, own_slot, own_warp_id)
    for map_const, warps in raw_warps.items():
        for own_slot, (_x, _y, dest, own_warp_id) in enumerate(warps, start=1):
            if dest != "LAST_MAP":
                continue
            sources = warps_into.get((map_const, own_slot), set())
            if len(sources) == 1:
                resolved_last_map[(map_const, own_slot)] = next(iter(sources))
            else:
                unresolved_slots.append((map_const, own_slot, own_warp_id))

    # Pass 2: a map's front door is frequently 2 tiles wide -- both tiles are
    # separate LAST_MAP warp_events with the SAME outgoing warp_id (pret's own
    # "land on slot N of wherever you came from" field), but only one of the
    # two tiles is the specific slot an outdoor map's warp_event actually
    # targets, so only that one gets a hit in pass 1. Its untargeted twin has
    # zero direct sources despite being the exact same door -- inherit the
    # sibling slot's resolved source when exactly one sibling (same map,
    # also LAST_MAP, same outgoing warp_id) was resolved in pass 1.
    ambiguous: set[tuple[str, int]] = set()
    for map_const, own_slot, own_warp_id in unresolved_slots:
        siblings = {
            resolved_last_map[(map_const, sib_slot)]
            for sib_slot, (_x, _y, sib_dest, sib_warp_id) in enumerate(
                raw_warps[map_const], start=1
            )
            if sib_dest == "LAST_MAP"
            and sib_warp_id == own_warp_id
            and (map_const, sib_slot) in resolved_last_map
        }
        if len(siblings) == 1:
            resolved_last_map[(map_const, own_slot)] = next(iter(siblings))
        else:
            ambiguous.add((map_const, own_slot))

    def resolve_dest_map(map_const: str, own_slot: int, dest: str) -> str | None:
        if dest != "LAST_MAP":
            return dest
        return resolved_last_map.get((map_const, own_slot))

    # Resolve every warp to a concrete (src_x, src_y, dest_map_id, dest_x, dest_y).
    edges: dict[int, list[tuple[int, int, int, int, int]]] = {}
    dropped = 0
    for src_const, warps in raw_warps.items():
        src_entry = MAPS.get(src_const)
        if src_entry is None:
            continue
        src_id = src_entry[0]
        out_edges: list[tuple[int, int, int, int, int]] = []
        for own_slot, (x, y, dest, warp_id) in enumerate(warps, start=1):
            dest_const = resolve_dest_map(src_const, own_slot, dest)
            if dest_const is None:
                dropped += 1
                continue
            dest_entry = MAPS.get(dest_const)
            dest_warps = raw_warps.get(dest_const)
            if dest_entry is None or not dest_warps or not (1 <= warp_id <= len(dest_warps)):
                dropped += 1
                continue
            dest_x, dest_y, _dest_dest, _dest_warp_id = dest_warps[warp_id - 1]
            out_edges.append((x, y, dest_entry[0], dest_x, dest_y))
        if out_edges:
            edges[src_id] = out_edges

    commit = subprocess.run(
        ["git", "-C", str(POKERED_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    lines = [
        '"""Static per-map warp graph, generated from pret/pokered.',
        "",
        f"Source: https://github.com/pret/pokered @ {commit}",
        "         data/maps/objects/*.asm, data/maps/headers/*.asm",
        "Discrete map-to-map jumps only (doors, staircases, cave mouths) --",
        "NOT the continuous outdoor Route<->City edge-walks (see",
        "tools/gen_map_warps.py's module docstring for that gap and why).",
        "See tools/gen_map_warps.py's module docstring for the LAST_MAP",
        "resolution rule.",
        "Regenerate with: python tools/gen_map_warps.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "# map_id -> list of (src_x, src_y, dest_map_id, dest_x, dest_y).",
        "MAP_WARPS: dict[int, list[tuple[int, int, int, int, int]]] = {",
    ]
    for map_id in sorted(edges):
        lines.append(f"    {map_id}: {edges[map_id]!r},")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def warps_from(map_id: int) -> list[tuple[int, int, int, int, int]]:")
    lines.append('    """(src_x, src_y, dest_map_id, dest_x, dest_y) warps leaving map_id,')
    lines.append('    or [] if it has none (or none pret/pokered\'s LAST_MAP could resolve')
    lines.append('    unambiguously -- see the module docstring)."""')
    lines.append("    return MAP_WARPS.get(map_id, [])")
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    total_warps = sum(len(v) for v in raw_warps.values())
    print(f"wrote {sum(len(v) for v in edges.values())}/{total_warps} warp edges "
          f"across {len(edges)} maps to {OUTPUT_PATH}")
    print(f"dropped {dropped} unresolved warp(s) ({len(ambiguous)} ambiguous LAST_MAP slot(s))")
    if ambiguous:
        print("ambiguous LAST_MAP slots (2+ or 0 distinct sources warp into that exact slot):")
        for m, slot in sorted(ambiguous):
            print(f"  {m} slot {slot}: sources={sorted(warps_into.get((m, slot), ()))}")
    if skipped:
        print(f"skipped {len(skipped)} maps with no object file:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
