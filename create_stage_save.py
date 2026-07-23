#!/usr/bin/env python3
"""Play Pokemon Red manually and save a mid-game state for a curriculum stage.

The bot loads ``saves/<stage>/checkpoint.state`` when that stage is active
(see ``curriculum_config.get_curriculum_saves``). Use this tool to reach a
spot yourself, then press S to dump that state.

Controls (global keyboard hook — works without focusing the terminal):
  Arrow keys  — movement
  A / B       — confirm / cancel
  Enter       — Start
  Space       — Select
  S           — save state for the target stage (overwrites)
  R           — reload from --from (discard unsaved progress)
  ESC / Q     — quit

Examples:
  python cli.py save-stage --stage stage_oaks_lab
  python cli.py save-stage --stage stage_fought_brock --from stage_oaks_lab
  python cli.py save-stage --list
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import argparse
import queue
import time
from multiprocessing import RLock
from pathlib import Path

import keyboard

from curriculum_config import CURRICULUM, STAGE_ORDER
from pokemon.Emulator import DURATION_BINS, Emulator

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down", "None"]
BIN_16_IDX = 0
NONE_ACTION = 8

# name -> (label, action_idx | None for meta)
KEY_MAP: dict[str, tuple[str, int | None]] = {
    "up": ("Up", 6),
    "down": ("Down", 7),
    "left": ("Left", 4),
    "right": ("Right", 5),
    "a": ("A", 0),
    "b": ("B", 1),
    "enter": ("Start", 2),
    "space": ("Sel", 3),
    "s": ("SAVE", None),
    "r": ("RELOAD", None),
    "esc": ("ESC", None),
    "q": ("Q", None),
}


def print_sep(char: str = "=", width: int = 60) -> None:
    print(char * width)


def list_stages() -> None:
    print_sep()
    print("  Curriculum stages (save path → goal)")
    print_sep()
    for name in STAGE_ORDER:
        cfg = CURRICULUM.get(name, {})
        path = Path(f"saves/{name}/checkpoint.state")
        mark = "✓" if path.is_file() else "·"
        goal = cfg.get("goal", "?")
        desc = cfg.get("description", "")
        print(f"  {mark} {name:<24} goal={goal:<16} {desc}")
    print_sep()
    print("  ✓ = saves/<stage>/checkpoint.state exists")
    print("  Train uses that file when the stage is active (else falls back to start).")


def _status_line(emulator: Emulator) -> str:
    mem = emulator.pyboy.memory
    mid = emulator.data.map_id(mem)
    x = emulator.data.position_x(mem)
    y = emulator.data.position_y(mem)
    badges = sum(emulator.data.badges(mem))
    return f"map={mid} pos=({x},{y}) badges={badges}"


def run(args: argparse.Namespace) -> None:
    if getattr(args, "list", False):
        list_stages()
        return

    stage = args.stage
    if not stage:
        print("Pass --stage <name> (or --list). Example: --stage stage_oaks_lab")
        raise SystemExit(2)

    if stage in CURRICULUM:
        goal = CURRICULUM[stage]["goal"]
        desc = CURRICULUM[stage].get("description", "")
    else:
        goal = "(custom — not in CURRICULUM)"
        desc = "custom save folder"
        print(f"Note: '{stage}' is not in STAGE_ORDER — still saved under saves/{stage}/")

    out_dir = f"saves/{stage}"
    from_ckpt = args.from_checkpoint

    key_queue: queue.Queue[tuple[str, int | None]] = queue.Queue()

    def on_key(event: keyboard.KeyboardEvent) -> None:
        if event.event_type != keyboard.KEY_DOWN:
            return
        entry = KEY_MAP.get(event.name.lower())
        if entry is not None:
            key_queue.put_nowait(entry)

    keyboard.hook(on_key, suppress=False)

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = True

    # Human play: disable stuck truncation so standing still does not end the run.
    emulator.data.max_useless_ticks = 10**9
    emulator.data.max_useless_dialog_ticks = 10**9

    print_sep()
    print("  Pokemon Red AI — Create stage save")
    print_sep()
    print(f"  Target stage : {stage}")
    print(f"  Goal         : {goal}")
    if desc:
        print(f"  Description  : {desc}")
    print(f"  Start from   : saves/{from_ckpt}/")
    print(f"  Will write   : {out_dir}/checkpoint.state")
    print_sep()
    print("  Controls:")
    print("    Arrows / A / B / Enter / Space  — play")
    print("    S                              — save stage state")
    print("    R                              — reload --from")
    print("    ESC / Q                        — quit")
    print_sep()
    print()

    memory, _ = emulator.reset(dir=from_ckpt)
    emulator.pyboy.tick(1, render=True, sound=False)
    print(f"  Loaded. {_status_line(emulator)}")
    print("  Play to the spot you want, then press S.\n")

    saves_count = 0
    running = True
    while running:
        try:
            label, action = key_queue.get(timeout=0.016)
        except queue.Empty:
            label, action = "-", NONE_ACTION
        else:
            if label in ("ESC", "Q"):
                print("\nQuit.")
                break
            if label == "SAVE":
                # Only game RAM state — no visited_* pickles (fresh exploration for the bot).
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                path = f"{out_dir}/checkpoint.state"
                with files_lock:
                    with open(path, "wb") as f:
                        emulator.pyboy.save_state(f)
                saves_count += 1
                print(f"  ✓ Saved → {path}  ({_status_line(emulator)})")
                continue
            if label == "RELOAD":
                memory, _ = emulator.reset(dir=from_ckpt)
                emulator.pyboy.tick(1, render=True, sound=False)
                print(f"  Reloaded from saves/{from_ckpt}/  ({_status_line(emulator)})")
                continue

        meta_action = action * len(DURATION_BINS) + BIN_16_IDX
        memory, _, _, terminated, truncated = emulator.step(
            memory=memory, meta_action=meta_action
        )
        # truncated/terminated ignored — human decides when to stop
        _ = terminated, truncated
        time.sleep(0.005)

    emulator.pyboy.stop(False)

    if saves_count:
        print_sep()
        print(f"  Done. {saves_count} save(s) at {out_dir}/checkpoint.state")
        print(f"  Train with:  python cli.py train --stage {stage}")
        print_sep()
    else:
        print("  No save written (press S next time before quitting).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manually play and save a mid-game state for a curriculum stage",
    )
    p.add_argument(
        "--stage",
        "-s",
        default=None,
        help="Stage / folder name under saves/ (e.g. stage_oaks_lab)",
    )
    p.add_argument(
        "--from",
        dest="from_checkpoint",
        default="start",
        help="Save dir to load first (default: start)",
    )
    p.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List curriculum stages and which already have saves",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
