"""Render Pokémon Red/Blue map pixel art directly from reference/pokered
source assets, for use as a heatmap background (see
src/utils/PositionHeatmap.py).

Source chain (mirrors tools/gen_map_collision.py's asset parsing -- see that
module's docstring for the full rationale -- but renders actual pixel art
instead of a walkable/grass bitmap):
  - constants/tileset_constants.asm + data/tilesets/tileset_headers.asm link
    a map header's ALLCAPS tileset name (e.g. OVERWORLD) to its CamelCase
    data-file name (e.g. Overworld).
  - gfx/tilesets.asm's `<Name>_Block::` INCBIN lines give the blockset path
    stem (e.g. "overworld") shared by both gfx/blocksets/<stem>.bst (block
    definitions) and gfx/tilesets/<stem>.png (the decoded tile sheet --
    verified 1:1 with the .bst path in every tileset, so no separate
    `_GFX::` parse is needed).
  - data/maps/headers/*.asm's `map_header <File>, <MAP_CONST>, <TILESET>`
    names each map's tileset; maps/<File>.blk holds its block ids
    (width_blocks * height_blocks bytes, row-major).
  - A block byte indexes 16 bytes in its tileset's .bst: a 4x4 tile-id
    picture. gfx/tilesets/<stem>.png is the same tileset's tile sheet,
    16 tiles (8px each) per row.

Grid unit: one walking cell (wXCoord/wYCoord) is CELL_PX (16) pixels -- 2x2
tiles, the same layout pokemon.map_collision samples for collision. A
rendered map's pixel array is therefore always exactly
(height_cells*CELL_PX, width_cells*CELL_PX), letting callers overlay
per-cell heatmap data (average_grid/combined_view's cell-coordinate grids)
at a matching pixel scale with a plain *CELL_PX conversion.

Everything here is read from reference/pokered directly and cached
(functools.lru_cache) rather than baked into a generated module like
map_collision.py -- full pixel art for all ~250 maps would be a multi-MB
source file, and this is only ever used by dev/analysis tooling
(PositionHeatmap.py), not the training/eval hot path, so a one-time
parse+PNG-decode cost per process is fine.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokemon.map_constants import MAPS_BY_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POKERED_DIR = _REPO_ROOT / "reference" / "pokered"

TILE_PX = 8
CELL_PX = TILE_PX * 2  # one walking cell = 2x2 tiles
_BLOCK_PX = CELL_PX * 2  # one block = 4x4 tiles = 2x2 cells

_TILESET_CONST_RE = re.compile(r"^\tconst\s+(\w+)", re.M)
_TILESET_HEADER_RE = re.compile(
    r"^\ttileset\s+(\w+),\s*(-?\$?[0-9A-Fa-f]+),\s*(-?\$?[0-9A-Fa-f]+),"
    r"\s*(-?\$?[0-9A-Fa-f]+),\s*(-?\$?[0-9A-Fa-f]+),",
    re.M,
)
_BLOCK_LABEL_RE = re.compile(r"^(\w+)_Block::\s*(?:INCBIN \"([^\"]+)\")?")
_MAP_HEADER_RE = re.compile(r"^\tmap_header\s+\w+,\s*(\w+),\s*(\w+)")


@lru_cache(maxsize=1)
def _allcaps_to_camel() -> dict[str, str]:
    """Map-header tileset name (ALLCAPS, e.g. OVERWORLD) -> CamelCase data
    name (e.g. Overworld) used by gfx/tilesets.asm's `_Block::` labels."""
    names = [
        m.group(1)
        for m in _TILESET_CONST_RE.finditer(
            (_POKERED_DIR / "constants" / "tileset_constants.asm").read_text(encoding="utf-8")
        )
    ]
    camel = [
        m.group(1)
        for m in _TILESET_HEADER_RE.finditer(
            (_POKERED_DIR / "data" / "tilesets" / "tileset_headers.asm").read_text(encoding="utf-8")
        )
    ]
    return dict(zip(names, camel))


@lru_cache(maxsize=1)
def _blockset_stems() -> dict[str, str]:
    """CamelName -> shared path stem (e.g. 'overworld') for both
    gfx/blocksets/<stem>.bst and gfx/tilesets/<stem>.png."""
    out: dict[str, str] = {}
    pending: list[str] = []
    for line in (_POKERED_DIR / "gfx" / "tilesets.asm").read_text(encoding="utf-8").splitlines():
        m = _BLOCK_LABEL_RE.match(line)
        if not m:
            continue
        name, incbin_path = m.groups()
        pending.append(name)
        if incbin_path:
            stem = Path(incbin_path).stem
            for n in pending:
                out[n] = stem
            pending = []
    return out


@lru_cache(maxsize=1)
def _map_headers() -> dict[str, tuple[str, str]]:
    """MAP_CONST -> (file_stem, tileset_allcaps)."""
    out: dict[str, tuple[str, str]] = {}
    header_dir = _POKERED_DIR / "data" / "maps" / "headers"
    for path in sorted(header_dir.glob("*.asm")):
        m = _MAP_HEADER_RE.match(path.read_text(encoding="utf-8"))
        if m:
            out[m.group(1)] = (path.stem, m.group(2))
    return out


@lru_cache(maxsize=None)
def _tile_sheet(stem: str) -> np.ndarray:
    """gfx/tilesets/<stem>.png as an (rows*8, 16*8) uint8 grayscale array."""
    from PIL import Image

    path = _POKERED_DIR / "gfx" / "tilesets" / f"{stem}.png"
    with Image.open(path) as img:
        return np.array(img.convert("L"))


@lru_cache(maxsize=None)
def _blockset_bytes(stem: str) -> bytes:
    return (_POKERED_DIR / "gfx" / "blocksets" / f"{stem}.bst").read_bytes()


@lru_cache(maxsize=None)
def render_map_rgb(map_id: int) -> np.ndarray | None:
    """Full pixel-art render of map_id: (height_cells*CELL_PX,
    width_cells*CELL_PX, 3) uint8 RGB. None if map_id has no static layout
    data (unused placeholder id) or its source assets can't be resolved
    (missing header/blk/tileset file -- same failure modes
    tools/gen_map_collision.py already tolerates)."""
    entry = MAPS_BY_ID.get(map_id)
    if entry is None:
        return None
    name, width_blocks, height_blocks = entry
    if width_blocks == 0 or height_blocks == 0:
        return None

    header_entry = _map_headers().get(name)
    if header_entry is None:
        return None
    file_stem, tileset_allcaps = header_entry

    camel_name = _allcaps_to_camel().get(tileset_allcaps)
    if camel_name is None:
        return None
    stem = _blockset_stems().get(camel_name)
    if stem is None:
        return None

    blk_path = _POKERED_DIR / "maps" / f"{file_stem}.blk"
    if not blk_path.exists():
        return None
    blk = blk_path.read_bytes()
    if len(blk) != width_blocks * height_blocks:
        return None

    try:
        sheet = _tile_sheet(stem)
    except FileNotFoundError:
        return None
    bst = _blockset_bytes(stem)
    sheet_cols = sheet.shape[1] // TILE_PX
    sheet_rows = sheet.shape[0] // TILE_PX

    canvas = np.zeros((height_blocks * _BLOCK_PX, width_blocks * _BLOCK_PX), dtype=np.uint8)
    for br in range(height_blocks):
        for bc in range(width_blocks):
            block_id = blk[br * width_blocks + bc]
            off = block_id * 16
            tile_ids = bst[off : off + 16]
            if len(tile_ids) < 16:
                continue  # block id past the end of this tileset's blockset
            for trow in range(4):
                for tcol in range(4):
                    tile_id = tile_ids[trow * 4 + tcol]
                    sr, sc = divmod(tile_id, sheet_cols)
                    if sr >= sheet_rows:
                        continue
                    tile = sheet[sr * TILE_PX : (sr + 1) * TILE_PX, sc * TILE_PX : (sc + 1) * TILE_PX]
                    py = br * _BLOCK_PX + trow * TILE_PX
                    px = bc * _BLOCK_PX + tcol * TILE_PX
                    canvas[py : py + TILE_PX, px : px + TILE_PX] = tile
    return np.stack([canvas, canvas, canvas], axis=-1)


def stitched_rgb(
    offsets: dict[int, tuple[int, int]]
) -> tuple[np.ndarray, int, int] | None:
    """Paste every map_id in ``offsets`` (map_id -> (cell_x, cell_y) world
    offset, e.g. RollingHeatmapAggregator.global_offsets) into one RGB
    canvas at its world position, for the heatmap's combined-map view.

    Returns (canvas, x0, y0): (x0, y0) is the canvas's cell-space origin --
    same convention as RollingHeatmapAggregator.combined_view's own
    (x0, y0), so both can be drawn on one axes with independent
    imshow(..., extent=...) calls and still align pixel-for-pixel, even
    though this canvas covers each map's full footprint while
    combined_view's grid only covers the visited-tile bounding box. None if
    no map in ``offsets`` has renderable source assets.
    """
    entries: list[tuple[int, int, np.ndarray]] = []
    x0 = y0 = x1 = y1 = None
    for map_id, (gx, gy) in offsets.items():
        img = render_map_rgb(map_id)
        if img is None:
            continue
        h_cells = img.shape[0] // CELL_PX
        w_cells = img.shape[1] // CELL_PX
        entries.append((gx, gy, img))
        x0 = gx if x0 is None else min(x0, gx)
        y0 = gy if y0 is None else min(y0, gy)
        x1 = gx + w_cells if x1 is None else max(x1, gx + w_cells)
        y1 = gy + h_cells if y1 is None else max(y1, gy + h_cells)
    if not entries:
        return None
    canvas = np.zeros(((y1 - y0) * CELL_PX, (x1 - x0) * CELL_PX, 3), dtype=np.uint8)
    for gx, gy, img in entries:
        px = (gx - x0) * CELL_PX
        py = (gy - y0) * CELL_PX
        canvas[py : py + img.shape[0], px : px + img.shape[1]] = img
    return canvas, x0, y0
