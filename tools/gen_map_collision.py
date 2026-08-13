#!/usr/bin/env python
"""Regenerate src/pokemon/map_collision.py from the pret/pokered disassembly.

Builds two per-map, per-walking-cell bitmaps straight from the same source
data the ROM itself uses, instead of only what's visible on-screen around
the player at any given moment:

  - constants/tileset_constants.asm declares the tileset ids in order
    (OVERWORLD, REDS_HOUSE_1, ...).
  - data/tilesets/tileset_headers.asm's `tileset` macro invocations list the
    same tilesets in the same order under their CamelCase names (Overworld,
    RedsHouse1, ...) plus each one's grass tile id (-1 if none) -- the two
    lists are zipped by position to link an ALLCAPS map-header tileset name
    to its CamelCase data-file name.
  - gfx/tilesets.asm's `<Name>_Block::` labels point (via INCBIN, with
    label aliasing for tilesets that share one blockset) at each tileset's
    gfx/blocksets/*.bst file: one 4x4-tile block picture (16 tile ids) per
    possible block byte value.
  - data/tilesets/collision_tile_ids.asm's `<Name>_Coll::` labels (same
    aliasing) list the walkable tile ids for that tileset -- this is the
    exact table the live ROM loads into wD530/the collision-tile-id pointer.
  - data/maps/headers/*.asm's `map_header <File>, <MAP_CONST>, <TILESET>`
    line names each map's tileset; maps/<File>.blk holds its block ids
    (width_blocks * height_blocks bytes, row-major -- see
    pokemon.map_constants.MAPS_BY_ID for the dimensions).

A block's byte indexes 16 bytes in its tileset's .bst (a 4x4 tile-id
picture). Each block covers a 2x2 grid of walking cells (the unit
wXCoord/wYCoord move in); a walking cell is walkable iff its block's
bottom-left tile (row 2*qr+1, col 2*qc, 0-indexed within the 4x4 picture --
matches PyBoy's GameWrapperPokemonGen1._get_screen_walkable_matrix, which
samples live VRAM this same way) is in that tileset's walkable-id set, i.e.
its collision_tile_ids.asm list plus its grass tile id (grass is walkable
but not in the base list, same "if tileset has a grass tile, add it" rule
the RAM code applies via wGrassTile). Validated against a live PyBoy session
(GameWrapperPokemonGen1.game_area_collision(), walking Route 1): 163/163
sampled on-screen cells matched this static computation.

A second, stricter bitmap marks only cells whose tile is exactly the
tileset's designated grass tile (a strict subset of the walkable one) --
on an overworld map the grass patches directly abut the path with nothing
walling them off, so a flood fill over the *walkable* bitmap swallows the
entire connected route, not just the "island" of grass a wild encounter's
tile sits in (empirically: Route 1 has 553 walkable cells but only 104 are
actually grass, split into several disconnected patches). See
Data._mark_wild_grass_island, which floods the grass bitmap specifically.

Usage: python tools/gen_map_collision.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POKERED_DIR = REPO_ROOT / "reference" / "pokered"
OUTPUT_PATH = REPO_ROOT / "src" / "pokemon" / "map_collision.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from pokemon.map_constants import MAPS_BY_ID  # noqa: E402

TILESET_CONST_RE = re.compile(r"^\tconst\s+(\w+)", re.M)
TILESET_HEADER_RE = re.compile(
    r"^\ttileset\s+(\w+),\s*(-?\$?[0-9A-Fa-f]+),\s*(-?\$?[0-9A-Fa-f]+),"
    r"\s*(-?\$?[0-9A-Fa-f]+),\s*(-?\$?[0-9A-Fa-f]+),",
    re.M,
)
BLOCK_LABEL_RE = re.compile(r"^(\w+)_Block::\s*(?:INCBIN \"([^\"]+)\")?")
COLL_LABEL_RE = re.compile(r"^(\w+)_Coll::")
COLL_TILES_RE = re.compile(r"^\tcoll_tiles\s+(.+)$")
MAP_HEADER_RE = re.compile(r"^\tmap_header\s+\w+,\s*(\w+),\s*(\w+)")


def parse_num(s: str) -> int:
    s = s.strip()
    return int(s[1:], 16) if s.startswith("$") else int(s)


def parse_tileset_constants(path: Path) -> list[str]:
    """Returns ALLCAPS tileset names in id order (OVERWORLD=0, ...)."""
    return [m.group(1) for m in TILESET_CONST_RE.finditer(path.read_text(encoding="utf-8"))]


def parse_tileset_headers(path: Path) -> list[tuple[str, int | None]]:
    """Returns (CamelName, grass_tile_id_or_None) in the same id order."""
    out = []
    for m in TILESET_HEADER_RE.finditer(path.read_text(encoding="utf-8")):
        name = m.group(1)
        grass = parse_num(m.group(5))
        out.append((name, None if grass == -1 else grass))
    return out


def parse_blocksets(path: Path) -> dict[str, str]:
    """CamelName -> blockset path (e.g. 'gfx/blocksets/overworld.bst')."""
    out: dict[str, str] = {}
    pending: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = BLOCK_LABEL_RE.match(line)
        if not m:
            continue
        name, incbin_path = m.groups()
        pending.append(name)
        if incbin_path:
            for n in pending:
                out[n] = incbin_path
            pending = []
    return out


def parse_collision_ids(path: Path) -> dict[str, list[int]]:
    """CamelName -> list of walkable raw tile ids for that tileset."""
    out: dict[str, list[int]] = {}
    pending: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = COLL_LABEL_RE.match(line)
        if m:
            pending.append(m.group(1))
            continue
        m2 = COLL_TILES_RE.match(line)
        if m2 and pending:
            ids = [parse_num(x) for x in m2.group(1).split(",")]
            for name in pending:
                out[name] = ids
            pending = []
    return out


def parse_map_header(path: Path) -> tuple[str, str] | None:
    """Returns (MAP_CONST, TILESET_ALLCAPS) from a data/maps/headers/*.asm file."""
    m = MAP_HEADER_RE.match(path.read_text(encoding="utf-8"))
    return (m.group(1), m.group(2)) if m else None


def build_grids(
    blk: bytes,
    width_blocks: int,
    height_blocks: int,
    bst: bytes,
    walkable_ids: set[int],
    grass_tile: int | None,
) -> tuple[bytearray, bytearray]:
    """(walkable, grass) -- one bit per walking cell each (LSB-first,
    row-major). walkable is 1 for any tile in ``walkable_ids`` (paths,
    doors, grass, ... -- everything the player can step on); grass is 1
    only for the tileset's single designated grass tile, a strict subset
    of walkable (see the module docstring for why the two need to be
    separate bitmaps, not just one flood fill).

    See the module docstring for the bottom-left-tile sampling rule.
    """
    width_cells = width_blocks * 2
    height_cells = height_blocks * 2
    n_bits = width_cells * height_cells
    walkable_packed = bytearray((n_bits + 7) // 8)
    grass_packed = bytearray((n_bits + 7) // 8)

    for br in range(height_blocks):
        for bc in range(width_blocks):
            block_id = blk[br * width_blocks + bc]
            off = block_id * 16
            tile_ids = bst[off : off + 16]
            if len(tile_ids) < 16:
                continue  # block id past the end of this tileset's blockset -- treat as wall
            for qr in range(2):
                for qc in range(2):
                    trow, tcol = 2 * qr + 1, 2 * qc
                    tile_id = tile_ids[trow * 4 + tcol]
                    if tile_id not in walkable_ids:
                        continue
                    cell_x, cell_y = bc * 2 + qc, br * 2 + qr
                    idx = cell_y * width_cells + cell_x
                    walkable_packed[idx // 8] |= 1 << (idx % 8)
                    if grass_tile is not None and tile_id == grass_tile:
                        grass_packed[idx // 8] |= 1 << (idx % 8)
    return walkable_packed, grass_packed


def main() -> None:
    tileset_names = parse_tileset_constants(POKERED_DIR / "constants" / "tileset_constants.asm")
    tileset_headers = parse_tileset_headers(POKERED_DIR / "data" / "tilesets" / "tileset_headers.asm")
    assert len(tileset_names) == len(tileset_headers), (
        f"tileset_constants.asm has {len(tileset_names)} entries, "
        f"tileset_headers.asm has {len(tileset_headers)} -- format changed upstream?"
    )
    # ALLCAPS (as used in map_header lines) -> (CamelName, grass_tile_id_or_None)
    allcaps_to_camel = dict(zip(tileset_names, tileset_headers))

    blocksets = parse_blocksets(POKERED_DIR / "gfx" / "tilesets.asm")
    collision_ids = parse_collision_ids(POKERED_DIR / "data" / "tilesets" / "collision_tile_ids.asm")

    bst_cache: dict[str, bytes] = {}

    def load_bst(rel_path: str) -> bytes:
        if rel_path not in bst_cache:
            bst_cache[rel_path] = (POKERED_DIR / rel_path).read_bytes()
        return bst_cache[rel_path]

    header_dir = POKERED_DIR / "data" / "maps" / "headers"
    map_const_to_header: dict[str, tuple[str, str]] = {}
    for header_path in sorted(header_dir.glob("*.asm")):
        parsed = parse_map_header(header_path)
        if parsed is None:
            continue
        map_const, tileset_allcaps = parsed
        map_const_to_header[map_const] = (header_path.stem, tileset_allcaps)

    collisions: dict[int, bytes] = {}
    grass_collisions: dict[int, bytes] = {}
    skipped: list[str] = []
    for map_id in sorted(MAPS_BY_ID):
        name, width_blocks, height_blocks = MAPS_BY_ID[map_id]
        if width_blocks == 0 or height_blocks == 0:
            continue  # unused placeholder id

        header_entry = map_const_to_header.get(name)
        if header_entry is None:
            skipped.append(f"{name}: no header file")
            continue
        file_stem, tileset_allcaps = header_entry

        camel_grass = allcaps_to_camel.get(tileset_allcaps)
        if camel_grass is None:
            skipped.append(f"{name}: unknown tileset {tileset_allcaps}")
            continue
        camel_name, grass_tile = camel_grass

        blk_path = POKERED_DIR / "maps" / f"{file_stem}.blk"
        if not blk_path.exists():
            skipped.append(f"{name}: no {blk_path.name}")
            continue
        blk = blk_path.read_bytes()
        if len(blk) != width_blocks * height_blocks:
            skipped.append(
                f"{name}: {blk_path.name} is {len(blk)} bytes, "
                f"expected {width_blocks * height_blocks}"
            )
            continue

        bst_rel_path = blocksets.get(camel_name)
        if bst_rel_path is None:
            skipped.append(f"{name}: no blockset for tileset {camel_name}")
            continue
        bst = load_bst(bst_rel_path)

        walkable_ids = set(collision_ids.get(camel_name, ()))
        if grass_tile is not None:
            walkable_ids.add(grass_tile)

        walkable_grid, grass_grid = build_grids(
            blk, width_blocks, height_blocks, bst, walkable_ids, grass_tile
        )
        collisions[map_id] = bytes(walkable_grid)
        if grass_tile is not None:
            grass_collisions[map_id] = bytes(grass_grid)

    commit = subprocess.run(
        ["git", "-C", str(POKERED_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    lines = [
        '"""Per-map walkable/grass-tile bitmaps, generated from pret/pokered.',
        "",
        f"Source: https://github.com/pret/pokered @ {commit}",
        "         maps/*.blk, gfx/blocksets/*.bst, data/tilesets/*.asm,",
        "         data/maps/headers/*.asm",
        "One bit per walking cell (LSB-first, row-major); grid size for a",
        "map_id is (width_blocks*2, height_blocks*2) from",
        "pokemon.map_constants.MAPS_BY_ID. See tools/gen_map_collision.py's",
        "module docstring for how a cell's walkability/grass-ness is derived.",
        "Regenerate with: python tools/gen_map_collision.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "from collections import deque",
        "",
        "from pokemon.map_constants import MAPS_BY_ID",
        "",
        "# map_id -> packed walkable-cell bitmap (paths, doors, grass, ... --",
        "# everything the player can step on).",
        "MAP_COLLISION: dict[int, bytes] = {",
    ]
    for map_id in sorted(collisions):
        lines.append(f"    {map_id}: {collisions[map_id]!r},")
    lines.append("}")
    lines.append("")
    lines.append("# map_id -> packed grass-cell bitmap -- a strict subset of MAP_COLLISION's")
    lines.append("# walkable bits (only the tileset's single designated grass tile id).")
    lines.append("# Omitted entirely for map_ids whose tileset has no grass tile at all.")
    lines.append("GRASS_COLLISION: dict[int, bytes] = {")
    for map_id in sorted(grass_collisions):
        lines.append(f"    {map_id}: {grass_collisions[map_id]!r},")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def _lookup(grid_by_map: dict[int, bytes], map_id: int, x: int, y: int) -> bool:")
    lines.append("    grid = grid_by_map.get(map_id)")
    lines.append("    entry = MAPS_BY_ID.get(map_id)")
    lines.append("    if grid is None or entry is None:")
    lines.append("        return False")
    lines.append("    width_cells = entry[1] * 2")
    lines.append("    height_cells = entry[2] * 2")
    lines.append("    if not (0 <= x < width_cells and 0 <= y < height_cells):")
    lines.append("        return False")
    lines.append("    idx = y * width_cells + x")
    lines.append("    return bool(grid[idx // 8] & (1 << (idx % 8)))")
    lines.append("")
    lines.append("")
    lines.append("def is_walkable(map_id: int, x: int, y: int) -> bool:")
    lines.append('    """True if (x, y) on map_id is a walkable overworld tile.')
    lines.append("")
    lines.append("    False for a map with no static collision data, an out-of-bounds")
    lines.append('    (x, y), or a tile outside the walkable set."""')
    lines.append("    return _lookup(MAP_COLLISION, map_id, x, y)")
    lines.append("")
    lines.append("")
    lines.append("def is_grass(map_id: int, x: int, y: int) -> bool:")
    lines.append('    """True if (x, y) on map_id is the tileset\'s designated grass tile')
    lines.append('    (a strict subset of is_walkable -- see GRASS_COLLISION). False for a')
    lines.append('    tileset with no grass tile at all (e.g. caves), not just an')
    lines.append('    out-of-bounds/non-grass (x, y)."""')
    lines.append("    return _lookup(GRASS_COLLISION, map_id, x, y)")
    lines.append("")
    lines.append("")
    lines.append("def walkable_tile_count(map_id: int) -> int:")
    lines.append('    """Total walkable cells on map_id (population count of its packed')
    lines.append("    bitmap -- trailing pad bits past width_cells*height_cells are always")
    lines.append('    0, so they never inflate this), or 0 if map_id has no static')
    lines.append('    collision data. The natural "how much of this map is there to')
    lines.append('    discover" denominator for a live position_visit_counts coverage')
    lines.append('    stat (see run_eval_ppo.py\'s -v/-vv [step] line)."""')
    lines.append("    grid = MAP_COLLISION.get(map_id)")
    lines.append("    if grid is None:")
    lines.append("        return 0")
    lines.append("    return int.from_bytes(grid, \"little\").bit_count()")
    lines.append("")
    lines.append("")
    lines.append("def flood_grass_island(map_id: int, x0: int, y0: int) -> set[tuple[int, int]]:")
    lines.append('    """4-connected grass-tile component reachable from (x0, y0) on')
    lines.append('    map_id, i.e. the grass "island" the tile sits in -- NOT the whole')
    lines.append("    connected walkable area (see the module docstring: on an overworld")
    lines.append("    map, grass patches directly abut the path with no wall between them,")
    lines.append("    so flooding is_walkable would swallow the entire route instead of")
    lines.append('    stopping at the patch\'s edge). Empty set if (x0, y0) itself is not')
    lines.append("    grass (see is_grass -- no static data for map_id, out-of-bounds, a")
    lines.append('    non-grass tile, or a tileset with no grass tile at all). Used by')
    lines.append("    Data._mark_wild_grass_island.")
    lines.append('    """')
    lines.append("    if not is_grass(map_id, x0, y0):")
    lines.append("        return set()")
    lines.append("    seen = {(x0, y0)}")
    lines.append("    queue = deque([(x0, y0)])")
    lines.append("    while queue:")
    lines.append("        x, y = queue.popleft()")
    lines.append("        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):")
    lines.append("            nx, ny = x + dx, y + dy")
    lines.append("            if (nx, ny) not in seen and is_grass(map_id, nx, ny):")
    lines.append("                seen.add((nx, ny))")
    lines.append("                queue.append((nx, ny))")
    lines.append("    return seen")
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(collisions)} maps ({len(grass_collisions)} with grass) to {OUTPUT_PATH}")
    if skipped:
        print(f"skipped {len(skipped)} maps:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
