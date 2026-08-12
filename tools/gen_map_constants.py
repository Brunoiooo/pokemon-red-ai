#!/usr/bin/env python
"""Regenerate src/pokemon/map_constants.py from the pret/pokered disassembly.

constants/map_constants.asm declares every map id via a `map_const NAME,
WIDTH, HEIGHT` macro (sequential const, so id == declaration order) plus a
handful of bare `const` entries. Width/height are in 2x2-tile map blocks, the
same unit the game itself uses for wCurMapWidth/wCurMapHeight.

Usage: python tools/gen_map_constants.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_CONSTANTS_ASM = REPO_ROOT / "reference" / "pokered" / "constants" / "map_constants.asm"
OUTPUT_PATH = REPO_ROOT / "src" / "pokemon" / "map_constants.py"

MAP_CONST_RE = re.compile(
    r"^map_const\s+(\w+),\s*(\d+),\s*(\d+)\s*(?:;.*)?$"
)
CONST_DEF_RE = re.compile(r"^const_def(?:\s+\$?([0-9A-Fa-f]+))?$")


def parse_maps(path: Path) -> dict[str, tuple[int, int, int]]:
    """Returns {name: (id, width, height)}."""
    idx = 0
    maps: dict[str, tuple[int, int, int]] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("MACRO") or line.startswith("ENDM"):
            continue
        if line.startswith("DEF ") or line.startswith("ASSERT") or line.startswith("end_indoor_group"):
            continue

        if CONST_DEF_RE.match(line):
            idx = 0
            continue

        m = MAP_CONST_RE.match(line)
        if m:
            name, width, height = m.group(1), int(m.group(2)), int(m.group(3))
            maps[name] = (idx, width, height)
            idx += 1
            continue

    return maps


def render_module(maps: dict[str, tuple[int, int, int]], pokered_commit: str) -> str:
    by_id = {v[0]: (k, v[1], v[2]) for k, v in maps.items()}

    lines = [
        '"""Map ids and block dimensions, generated from pret/pokered.',
        "",
        f"Source: https://github.com/pret/pokered @ {pokered_commit}",
        "         constants/map_constants.asm",
        "Width/height are in 2x2-tile map blocks (matches wCurMapWidth/",
        "wCurMapHeight at runtime) -- 'UNUSED_MAP_*' placeholder ids are",
        "included with (0, 0) so id lookups never KeyError on a valid byte.",
        "Regenerate with: python tools/gen_map_constants.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "# name -> (map_id, width_blocks, height_blocks).",
        "MAPS: dict[str, tuple[int, int, int]] = {",
    ]
    for name in sorted(maps, key=lambda n: maps[n][0]):
        mid, w, h = maps[name]
        lines.append(f"    {name!r}: ({mid}, {w}, {h}),")
    lines.append("}")
    lines.append("")
    lines.append("# map_id -> (name, width_blocks, height_blocks).")
    lines.append("MAPS_BY_ID: dict[int, tuple[str, int, int]] = {")
    for mid in sorted(by_id):
        name, w, h = by_id[mid]
        lines.append(f"    {mid}: ({name!r}, {w}, {h}),")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def map_area_blocks(map_id: int) -> int:")
    lines.append('    """width_blocks * height_blocks for map_id, 0 if unknown/unused."""')
    lines.append("    entry = MAPS_BY_ID.get(map_id)")
    lines.append("    return entry[1] * entry[2] if entry else 0")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    maps = parse_maps(MAP_CONSTANTS_ASM)

    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT / "reference" / "pokered"), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    OUTPUT_PATH.write_text(render_module(maps, commit), encoding="utf-8")
    print(f"wrote {len(maps)} maps to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
