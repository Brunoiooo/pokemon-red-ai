#!/usr/bin/env python
"""Find and draw a route from a start tile to one or more goal tiles, across
maps (through doors/staircases/cave mouths), for eyeball verification of
pokemon.world_graph's BFS.

Each map the route passes through is drawn as its own separate ASCII grid,
one after another -- never overlaid onto a shared canvas. That's deliberate:
a house sits "on top of" a city tile and a 2F sits "on top of" its own 1F in
the game's own map data (same-ish or unrelated local coordinates, no shared
frame), so there is no single 2D plane to draw them both on. Treating the
world as a graph of separate maps linked by warp edges sidesteps needing one.

Three modes:

  1. No --goal at all -- AUTO mode: walks every pokemon.Data.GOAL_CANDIDATES
     curriculum goal in pokemon.goal_positions.unlock_order() (dependency
     order, i.e. a goal's EVENT_GRAPH parents come before it), routing one
     leg at a time from wherever the previous leg left off. Prints one
     summary table for the whole run; add --show-maps for the ASCII
     fragments of every leg too (this is a LOT of output across all ~400
     goals -- pair with --limit while just browsing).

       python tools/render_route.py
       python tools/render_route.py --limit 15 --show-maps

  2. One or more --goal -- routes to the NEAREST reachable one (by step
     count); ties go to whichever --goal was given first.

       python tools/render_route.py --start PALLET_TOWN 5 6 \\
           --goal REDS_HOUSE_1F 2 7 --goal BLUES_HOUSE 2 7 --goal OAKS_LAB 5 5

  3. --html -- writes and opens an interactive page (tools/route_map_template.html)
     with the whole world graph embedded: pick start and goal live, from any
     curriculum goal to any other, and it draws the shortest route the same
     way (map-by-map canvas fragments), computed client-side in the browser.

       python tools/render_route.py --html
       python tools/render_route.py --html out/kanto.html --no-open

  --start defaults to Red's bedroom (REDS_HOUSE_2F, the room the game
  actually opens in) when omitted; override it to route from wherever your
  agent currently is instead. --start/--goal are ignored by --html (pick
  those in the page itself).
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon import goal_positions as gp  # noqa: E402
from pokemon import map_collision as mc  # noqa: E402
from pokemon import map_warps as mw  # noqa: E402
from pokemon import world_graph as wg  # noqa: E402
from pokemon.map_constants import MAPS, MAPS_BY_ID  # noqa: E402

HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "route_map_template.html"

# Red's bedroom, REDS_HOUSE_2F -- where a fresh game actually starts.
DEFAULT_START: wg.Pos = (7, 1, MAPS["REDS_HOUSE_2F"][0])


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


def parse_point(map_spec: str, x: str, y: str) -> wg.Pos:
    return (int(x), int(y), resolve_map_id(map_spec))


def segments(path: list[wg.Pos]) -> list[list[wg.Pos]]:
    """Split a path into contiguous same-map_id runs, in order."""
    out: list[list[wg.Pos]] = []
    for pos in path:
        if out and out[-1][-1][2] == pos[2]:
            out[-1].append(pos)
        else:
            out.append([pos])
    return out


def render_segment(seg: list[wg.Pos], is_first: bool, is_last: bool) -> str:
    map_id = seg[0][2]
    name, width_blocks, height_blocks = MAPS_BY_ID[map_id]
    width_cells, height_cells = width_blocks * 2, height_blocks * 2
    on_path = {(x, y) for x, y, _m in seg}
    start_xy = (seg[0][0], seg[0][1]) if is_first else None
    goal_xy = (seg[-1][0], seg[-1][1]) if is_last else None

    rows = []
    for y in range(height_cells):
        row = []
        for x in range(width_cells):
            if (x, y) == start_xy:
                row.append("@")
            elif (x, y) == goal_xy:
                row.append("G")
            elif (x, y) in on_path:
                row.append("*")
            elif mc.is_grass(map_id, x, y):
                row.append(",")
            elif mc.is_walkable(map_id, x, y):
                row.append(".")
            else:
                row.append("#")
        rows.append("".join(row))

    header = f"{name} (id {map_id}) -- {len(seg)} tile(s) on route"
    return "\n".join([header, "", *rows])


def print_route(path: list[wg.Pos], reached: wg.Pos, show_maps: bool) -> None:
    segs = segments(path)
    map_names = [MAPS_BY_ID[s[0][2]][0] for s in segs]
    print(f"  reached {reached} in {len(path) - 1} step(s) via: {' -> '.join(map_names)}")
    if show_maps:
        print()
        for i, seg in enumerate(segs):
            print(render_segment(seg, is_first=(i == 0), is_last=(i == len(segs) - 1)))
            print()


def run_nearest_of(start: wg.Pos, goals: list[wg.Pos], show_maps: bool) -> None:
    result = wg.shortest_path_to_nearest(start, goals)
    if result is None:
        print(
            f"No path found from {start} to any of {len(goals)} goal(s) over the "
            f"warp-connected graph (pokemon.world_graph does not cover outdoor "
            f"Route<->City edge-walks -- see its module docstring)."
        )
        raise SystemExit(1)
    path, reached = result
    print_route(path, reached, show_maps)


def run_auto(start: wg.Pos, limit: int | None, show_maps: bool) -> None:
    order = gp.unlock_order()
    if limit is not None:
        order = order[:limit]

    print(f"{len(order)} goal(s), in dependency (unlock) order:")
    print()

    current = start
    for i, goal in enumerate(order, start=1):
        tile = gp.goal_tile(goal)
        label = f"[{i}/{len(order)}] {goal} -- {gp.describe(goal)}"
        if tile is None:
            print(f"{label}\n  SKIPPED: no resolvable warp entrance for this goal's map")
            continue
        result = wg.shortest_path_to_nearest(current, [tile])
        if result is None:
            print(
                f"{label}\n  UNREACHABLE from current position "
                f"{current} over the warp graph (position unchanged for next goal)"
            )
            continue
        path, reached = result
        print(label)
        print_route(path, reached, show_maps)
        current = reached
        print()


def build_world_data() -> dict:
    """Everything route_map_template.html needs to run pokemon.world_graph's
    BFS client-side: every map's packed walkable/grass bitmaps (base64, same
    bytes as pokemon.map_collision.MAP_COLLISION/GRASS_COLLISION -- small
    enough in total, a few hundred KB at most, to just inline), the warp
    graph, and every curriculum goal's resolved tile (pokemon.goal_positions,
    in unlock order)."""
    import base64

    maps: dict[int, dict] = {}
    for map_id, (name, width_blocks, height_blocks) in MAPS_BY_ID.items():
        if width_blocks == 0 or height_blocks == 0 or map_id not in mc.MAP_COLLISION:
            continue
        maps[map_id] = {
            "name": name,
            "w": width_blocks * 2,
            "h": height_blocks * 2,
            "walk": base64.b64encode(mc.MAP_COLLISION[map_id]).decode("ascii"),
            "grass": (
                base64.b64encode(mc.GRASS_COLLISION[map_id]).decode("ascii")
                if map_id in mc.GRASS_COLLISION
                else None
            ),
        }

    order = gp.unlock_order()
    goals = []
    for i, goal in enumerate(order):
        tile = gp.goal_tile(goal)
        goals.append(
            {
                "id": goal,
                "desc": gp.describe(goal),
                "order": i,
                "tile": list(tile) if tile else None,
            }
        )

    return {"maps": maps, "warps": mw.MAP_WARPS, "goals": goals}


def write_html(output_path: Path, start_label: str, start_tile: wg.Pos) -> None:
    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    world = build_world_data()
    initial = {
        "start_label": start_label,
        "start_tile": list(start_tile),
        # Left None -- the page falls back to the first resolvable goal in
        # unlock order, and the user picks interactively from there anyway.
        "goal_key": None,
    }
    html = template.replace(
        "/*__WORLD_DATA__*/", json.dumps(world, separators=(",", ":"))
    ).replace("/*__INITIAL__*/", json.dumps(initial, separators=(",", ":")))
    output_path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--start", nargs=3, metavar=("MAP", "X", "Y"), default=None,
        help="Start map name/id and (x, y). Default: Red's bedroom (game start).",
    )
    p.add_argument(
        "--goal", nargs=3, metavar=("MAP", "X", "Y"), action="append", default=None,
        help="A candidate goal map name/id and (x, y). Repeatable -- routes to "
             "the nearest reachable one. Omit entirely for AUTO mode (see "
             "module docstring): every curriculum goal, in unlock order.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="AUTO mode only: stop after this many goals (default: all ~400).",
    )
    p.add_argument(
        "--show-maps", action="store_true",
        help="Also print the ASCII map fragment(s) for every leg, not just the "
             "one-line summary. Off by default in AUTO mode (huge output).",
    )
    p.add_argument("--legend", action="store_true", help="Print the character legend first")
    p.add_argument(
        "--html", nargs="?", const="route_map.html", default=None, metavar="PATH",
        help="Write an interactive HTML page (default: route_map.html) and open "
             "it -- pick start/goal live in the browser instead of via "
             "--start/--goal. See mode 3 above.",
    )
    p.add_argument(
        "--no-open", action="store_true",
        help="With --html, don't automatically open the generated file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    start = parse_point(*args.start) if args.start else DEFAULT_START

    if args.html is not None:
        output_path = Path(args.html)
        start_label = (
            f"{MAPS_BY_ID[start[2]][0]} ({start[0]},{start[1]})"
            if args.start
            else "Start gry (pokój Reda)"
        )
        write_html(output_path, start_label, start)
        print(f"Wrote {output_path.resolve()}")
        if not args.no_open:
            webbrowser.open(output_path.resolve().as_uri())
        return

    if args.legend:
        print("# wall/non-walkable   . walkable, not grass   , grass")
        print("* route tile   @ start   G goal reached")
        print()

    if args.goal:
        goals = [parse_point(*g) for g in args.goal]
        run_nearest_of(start, goals, args.show_maps)
    else:
        run_auto(start, args.limit, args.show_maps)


if __name__ == "__main__":
    main()
