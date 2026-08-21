#!/usr/bin/env python
"""Print an ASCII render of a map's grass-island BFS for eyeball verification.

Renders the full static collision grid for one map_id, with three tiers:
  '#' wall / non-walkable
  '.' walkable but not grass
  ',' grass, but not reachable from the given start (a different island)
  'o' grass and reachable from the start (pokemon.map_collision.flood_grass_island)
  '@' the start tile itself

Use this to confirm flood_grass_island's BFS does what it claims: stays
inside the connected grass patch the given tile sits in, and does NOT leak
across a path into some other patch of grass elsewhere on the map.

Usage:
  python tools/render_map_island.py ROUTE_1 8 6
  python tools/render_map_island.py 12 8 6
  python tools/render_map_island.py ROUTE_1 8 6 --legend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon import map_collision as mc  # noqa: E402
from pokemon.map_constants import MAPS, MAPS_BY_ID  # noqa: E402


def resolve_map_id(spec: str) -> int:
    if spec.isdigit():
        return int(spec)
    name = spec.upper()
    if name not in MAPS:
        raise SystemExit(
            f"Unknown map {spec!r}. Pass a numeric map_id or a name from "
            f"pokemon.map_constants.MAPS, e.g. ROUTE_1."
        )
    return MAPS[name][0]


def render(map_id: int, x0: int, y0: int) -> str:
    entry = MAPS_BY_ID.get(map_id)
    if entry is None:
        raise SystemExit(f"map_id {map_id} has no MAPS_BY_ID entry.")
    name, width_blocks, height_blocks = entry
    width_cells = width_blocks * 2
    height_cells = height_blocks * 2

    if not mc.is_walkable(map_id, x0, y0):
        print(
            f"warning: ({x0},{y0}) on {name} (id {map_id}) is not walkable at all "
            f"-- flood_grass_island will be empty regardless of grass.",
            file=sys.stderr,
        )
    elif not mc.is_grass(map_id, x0, y0):
        print(
            f"warning: ({x0},{y0}) on {name} (id {map_id}) is walkable but not the "
            f"tileset's grass tile -- flood_grass_island will be empty (this is "
            f"correct: cave/water/building encounters have no grass island).",
            file=sys.stderr,
        )

    island = mc.flood_grass_island(map_id, x0, y0)

    rows = []
    for y in range(height_cells):
        row = []
        for x in range(width_cells):
            if x == x0 and y == y0:
                row.append("@")
            elif (x, y) in island:
                row.append("o")
            elif mc.is_grass(map_id, x, y):
                row.append(",")
            elif mc.is_walkable(map_id, x, y):
                row.append(".")
            else:
                row.append("#")
        rows.append("".join(row))

    header = f"{name} (id {map_id}), {width_cells}x{height_cells} cells, start=({x0},{y0})"
    stats = f"grass island size: {len(island)} tile(s)"
    return "\n".join([header, stats, "", *rows])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", help="Map name (e.g. ROUTE_1) or numeric map_id")
    p.add_argument("x", type=int, help="Start x (wXCoord)")
    p.add_argument("y", type=int, help="Start y (wYCoord)")
    p.add_argument(
        "--legend", action="store_true", help="Print the character legend before the grid"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    map_id = resolve_map_id(args.map)
    if args.legend:
        print("# wall/non-walkable   . walkable, not grass   , grass (other island)")
        print("o grass (this island) @ start tile")
        print()
    print(render(map_id, args.x, args.y))


if __name__ == "__main__":
    main()
