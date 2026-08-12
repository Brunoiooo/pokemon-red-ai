#!/usr/bin/env python
"""Regenerate src/pokemon/ram_constants.py from the pret/pokered disassembly.

pret/pokered's ram/wram.asm, ram/hram.asm and ram/sram.asm declare every RAM
symbol *by name only* (SECTION + db/dw/ds) -- there are no literal addresses
in the source. The real addresses are only known after running the actual
assembler + linker (rgbasm/rgblink from rgbds) with pokered's own
layout.link, which pins section order/placement.

This script:
  1. Assembles reference/pokered/ram.asm (which just INCLUDEs the four
     ram/*.asm files) with rgbasm.
  2. Generates a throwaway "stub.asm" that declares an empty SECTION for
     every ROM0/ROMX bank layout.link references (audio/gfx/engine code we
     don't have/need object files for), so rgblink's strict layout.link
     resolves cleanly. These stubs are zero-byte and never touch RAM space.
  3. Links ram.o + stub.o with `-d` (DMG mode, expands WRAM0 to the full
     8 KiB $C000-$DFFF) and layout.link, producing a real .sym file.
  4. Parses the .sym file for w*/h*/s*-prefixed symbols (WRAM/HRAM/SRAM)
     and writes them out as a Python constants module.

Requires rgbds (rgbasm/rgblink) matching reference/pokered/.rgbds-version
on PATH, or pass --rgbds-bin <dir>.
Get it from: https://github.com/gbdev/rgbds/releases
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POKERED_DIR = REPO_ROOT / "reference" / "pokered"
OUTPUT_PATH = REPO_ROOT / "src" / "pokemon" / "ram_constants.py"

# Sections already defined for real by ram.asm (via ram/wram.asm, ram/hram.asm,
# ram/sram.asm, ram/vram.asm) -- everything else layout.link references is ROM
# code/data we don't assemble, and gets a zero-byte stub instead.
REAL_SECTIONS = {
    "Audio RAM", "Sprite State Data", "OAM Buffer", "Tilemap",
    "Overworld Map", "WRAM", "Party Data", "Main Data", "Current Box Data",
    "Stack", "VRAM", "Sprite Buffers", "Save Data", "Saved Boxes 1",
    "Saved Boxes 2", "HRAM",
}

SYM_LINE_RE = re.compile(r"^([0-9A-Fa-f]{2}):([0-9A-Fa-f]{4}) (\S+)$")


def parse_layout_link(path: Path) -> list[tuple[str, str]]:
    """Return [(region, section_name), ...] in file order."""
    region = None
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith('"'):
            entries.append((region, line.strip('"')))
        elif line.startswith("org") or line.startswith("align"):
            continue
        else:
            region = line
    return entries


def build_stub_asm(entries: list[tuple[str, str]]) -> str:
    lines = []
    for region, name in entries:
        if name in REAL_SECTIONS:
            continue
        if region == "ROM0":
            lines.append(f'SECTION "{name}", ROM0')
        elif region.startswith("ROMX"):
            bank = region.split()[1]
            lines.append(f'SECTION "{name}", ROMX, BANK[{bank}]')
        elif region == "HRAM":
            # e.g. "OAM DMA" -- declared in home.asm, not ram/hram.asm
            lines.append(f'SECTION "{name}", HRAM')
        else:
            raise ValueError(f"unhandled layout.link region {region!r} for {name!r}")
    return "\n".join(lines) + "\n"


def find_rgbds_tool(name: str, rgbds_bin: str | None) -> str:
    if rgbds_bin:
        candidate = Path(rgbds_bin) / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
        candidate = Path(rgbds_bin) / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        sys.exit(
            f"error: {name} not found on PATH. Install rgbds "
            f"(https://github.com/gbdev/rgbds/releases, version pinned in "
            f".rgbds-version) or pass --rgbds-bin <dir containing rgbasm/rgblink>."
        )
    return found


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")


def parse_sym_file(path: Path) -> dict[str, int]:
    symbols: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SYM_LINE_RE.match(line)
        if not m:
            continue
        _bank, addr, name = m.groups()
        if name[0] not in "whs":
            continue  # keep wXxx (WRAM), hXxx (HRAM), sXxx (SRAM) only
        symbols.setdefault(name, int(addr, 16))
    return symbols


def render_module(symbols: dict[str, int], pokered_commit: str) -> str:
    by_addr: dict[int, list[str]] = {}
    for name, addr in symbols.items():
        by_addr.setdefault(addr, []).append(name)

    lines = [
        '"""Named RAM addresses, generated from pret/pokered.',
        "",
        f"Source: https://github.com/pret/pokered @ {pokered_commit}",
        "Regenerate with: python tools/gen_ram_constants.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        "# name -> address. Multiple names can share an address (rgbds UNION",
        "# blocks: the same bytes are reused for different purposes depending",
        "# on game state, e.g. menu scratch space vs. battle scratch space).",
        "RAM: dict[str, int] = {",
    ]
    for name in sorted(symbols):
        lines.append(f"    {name!r}: 0x{symbols[name]:04X},")
    lines.append("}")
    lines.append("")
    lines.append("# address -> name(s), for reverse lookup / validation.")
    lines.append("RAM_BY_ADDRESS: dict[int, list[str]] = {")
    for addr in sorted(by_addr):
        names = ", ".join(repr(n) for n in sorted(by_addr[addr]))
        lines.append(f"    0x{addr:04X}: [{names}],")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbds-bin", help="directory containing rgbasm/rgblink")
    args = parser.parse_args()

    rgbasm = find_rgbds_tool("rgbasm", args.rgbds_bin)
    rgblink = find_rgbds_tool("rgblink", args.rgbds_bin)

    layout_link = POKERED_DIR / "layout.link"
    entries = parse_layout_link(layout_link)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stub_asm = tmp_path / "stub.asm"
        stub_asm.write_text(build_stub_asm(entries), encoding="utf-8")

        ram_o = tmp_path / "ram.o"
        stub_o = tmp_path / "stub.o"
        sym_out = tmp_path / "ram.sym"
        map_out = tmp_path / "ram.map"
        bin_out = tmp_path / "ram.out"

        run(
            [rgbasm, "-Q8", "-P", "includes.asm", "-o", str(ram_o), "ram.asm"],
            cwd=POKERED_DIR,
        )
        run([rgbasm, "-o", str(stub_o), str(stub_asm)])
        run([
            rgblink, "-d", "-l", str(layout_link),
            "-m", str(map_out), "-n", str(sym_out), "-o", str(bin_out),
            str(ram_o), str(stub_o),
        ])

        symbols = parse_sym_file(sym_out)

    commit = subprocess.run(
        ["git", "-C", str(POKERED_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    OUTPUT_PATH.write_text(render_module(symbols, commit), encoding="utf-8")
    print(f"wrote {len(symbols)} symbols to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
