#!/usr/bin/env python3
"""Human play with live reward / dialog / battle debug prints.

Use this to feel what the agent sees when approaching the first battle dialog:
rewards, is_dialog / is_battle flags, dialog_id, loop penalties, stuck fuses.

Controls (global keyboard hook — works without focusing the terminal):
  Arrow keys  — movement
  A / B       — confirm / cancel
  Enter       — Start
  Space       — Select
  S           — save state to saves/<save-as>/checkpoint.state
  R           — reload from --from
  H           — print help + current status
  ESC / Q     — quit

Examples:
  python cli.py debug-play --from start
  python cli.py debug-play --from stage_oaks_lab --goal oaks_lab
  python cli.py debug-play --from start --frame-skip 16 --real-truncation
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import argparse
import queue
from multiprocessing import RLock
from pathlib import Path

import keyboard

from curriculum_config import CURRICULUM, get_goal_for_stage
from pokemon.Emulator import Emulator

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down", "None"]
NONE_ACTION = 8

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
    "h": ("HELP", None),
    "esc": ("ESC", None),
    "q": ("Q", None),
}


def print_sep(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _mode_flags(emu: Emulator) -> str:
    mem = emu.pyboy.memory
    flags = []
    if emu.data.is_cutscene_locked(mem):
        flags.append("cutscene")
    if emu.data.is_world(mem):
        flags.append("world")
    if emu.data.is_dialog(mem):
        flags.append("dialog")
    if emu.data.is_menu(mem):
        flags.append("menu")
    if emu.data.is_battle(mem):
        flags.append("battle")
    return "+".join(flags) or "?"


def status_line(emu: Emulator) -> str:
    mem = emu.pyboy.memory
    mid = emu.data.map_id(mem)
    x = emu.data.position_x(mem)
    y = emu.data.position_y(mem)
    did = emu.data.dialog_id(mem)
    return (
        f"map={mid} pos=({x},{y}) mode={_mode_flags(emu)} "
        f"dialog_id={did} goal={emu.data.goal}"
    )


def fuse_line(emu: Emulator) -> str:
    data = emu.data
    pos = data.get_position()
    tile = data.visited_positions.get(pos, 0)
    return (
        f"fuses: tile={tile}/{data.max_useless_ticks} "
        f"dialog={int(data.in_dialog_ticks)}/{data.max_useless_dialog_ticks} "
        f"battle={int(data.in_battle_ticks)}/{data.max_useless_battle_ticks} "
        f"menu={int(data.in_menu_ticks)}/{data.max_useless_ticks} "
        f"loop={data.loop_streak}/{data.max_loop_streak}"
    )


def print_help() -> None:
    print_sep("-")
    print("  Arrows / A / B / Enter / Space  — play")
    print("  S                              — save state")
    print("  R                              — reload --from")
    print("  H                              — this help + status")
    print("  ESC / Q                        — quit")
    print_sep("-")


def run(args: argparse.Namespace) -> None:
    from_ckpt = args.from_checkpoint
    save_as = args.save_as or "manual_debug"
    frame_skip = int(args.frame_skip)
    goal = args.goal
    if goal is None and args.stage:
        goal = get_goal_for_stage(args.stage) or CURRICULUM.get(args.stage, {}).get(
            "goal"
        )
    if goal is None:
        goal = "oaks_lab"

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
    emulator.data.goal = goal

    if not args.real_truncation:
        # Human play: don't end the run while standing still / reading text.
        emulator.data.max_useless_ticks = 10**9
        emulator.data.max_useless_dialog_ticks = 10**9
        emulator.data.max_useless_battle_ticks = 10**9
        emulator.data.max_loop_streak = 10**9

    print_sep()
    print("  Pokemon Red AI — Debug play (human + rewards)")
    print_sep()
    print(f"  Start from   : saves/{from_ckpt}/")
    print(f"  Save as      : saves/{save_as}/checkpoint.state")
    print(f"  Goal         : {goal}")
    print(f"  Frame skip   : {frame_skip}  (PPO default is 16)")
    print(f"  Speed        : 1x (human)")
    print(
        f"  Truncation   : "
        f"{'REAL (agent fuses)' if args.real_truncation else 'disabled (human)'}"
    )
    print(f"  Log every    : {'all steps' if args.all_steps else 'button presses only'}")
    print_sep()
    print_help()
    print()

    memory, _ = emulator.reset(dir=from_ckpt)
    emulator.data.goal = goal
    # Real-time pace: PyBoy only rate-limits once per tick() call, so batched
    # tick(duration) would still run near-instant. render_each paces each frame.
    emulator.pyboy.set_emulation_speed(1)
    emulator.pyboy.tick(1, render=True, sound=False)
    print(f"  Loaded. {status_line(emulator)}")
    print(f"  {fuse_line(emulator)}")
    print("  Play toward the first battle dialog and watch reward / mode lines.\n")

    prev_mode = _mode_flags(emulator)
    prev_dialog = emulator.data.dialog_id(emulator.pyboy.memory)
    step_i = 0
    total_reward = 0.0
    saves_count = 0

    while True:
        try:
            label, action = key_queue.get(timeout=0.016)
        except queue.Empty:
            label, action = "-", NONE_ACTION
        else:
            if label in ("ESC", "Q"):
                print("\nQuit.")
                break
            if label == "HELP":
                print_help()
                print(f"  {status_line(emulator)}")
                print(f"  {fuse_line(emulator)}")
                print(f"  total_reward={total_reward:.4f} steps={step_i}")
                continue
            if label == "SAVE":
                out_dir = Path(f"saves/{save_as}")
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "checkpoint.state"
                with files_lock:
                    with open(path, "wb") as f:
                        emulator.pyboy.save_state(f)
                saves_count += 1
                print(f"  ✓ Saved → {path}  ({status_line(emulator)})")
                continue
            if label == "RELOAD":
                memory, _ = emulator.reset(dir=from_ckpt)
                emulator.data.goal = goal
                emulator.pyboy.tick(1, render=True, sound=False)
                prev_mode = _mode_flags(emulator)
                prev_dialog = emulator.data.dialog_id(emulator.pyboy.memory)
                step_i = 0
                total_reward = 0.0
                print(f"  Reloaded from saves/{from_ckpt}/  ({status_line(emulator)})")
                continue

        # Match PPO hold length, but pace frame-by-frame at 1x (human speed).
        memory, _, reward, terminated, truncated = emulator.step_discrete(
            memory=memory,
            action_idx=action,
            duration=frame_skip,
            render_each=True,
        )
        step_i += 1
        total_reward += float(reward)

        mode = _mode_flags(emulator)
        did = emulator.data.dialog_id(emulator.pyboy.memory)
        mode_changed = mode != prev_mode
        dialog_changed = did != prev_dialog
        interesting = (
            action != NONE_ACTION
            or mode_changed
            or dialog_changed
            or abs(float(reward)) >= 0.005
            or emulator.data.loop_flag
            or terminated
            or truncated
        )

        if args.all_steps or interesting:
            btn = BUTTON_NAMES[action] if 0 <= action < len(BUTTON_NAMES) else "?"
            milestone = getattr(emulator, "last_milestone", 0.0)
            step_r = getattr(emulator, "last_step", 0.0)
            marks = []
            if mode_changed:
                marks.append(f"{prev_mode}->{mode}")
            if dialog_changed:
                marks.append(f"dialog {prev_dialog}->{did}")
            if emulator.data.loop_flag:
                marks.append("LOOP")
            if getattr(emulator.data, "_last_regressed", None):
                lost = ",".join(emulator.data._last_regressed)
                marks.append(f"REGRESS({lost})")
            if terminated:
                marks.append("TERM(goal)")
            if truncated:
                marks.append("TRUNC")
            mark_s = f"  [{' | '.join(marks)}]" if marks else ""
            live_n = len(emulator.data.live_story_goals())
            peak_n = int(getattr(emulator.data, "_peak_live_goals", 0) or 0)
            print(
                f"#{step_i:<5} {btn:<6} r={reward:+.4f} "
                f"(m={milestone:+.4f} s={step_r:+.4f})  "
                f"{status_line(emulator)}  "
                f"goals={live_n}/{peak_n} loop={emulator.data.loop_streak}{mark_s}"
            )
            if mode_changed or dialog_changed or truncated or terminated:
                print(f"       {fuse_line(emulator)}  Σr={total_reward:.4f}")

        prev_mode = mode
        prev_dialog = did

        if terminated:
            print(f"\n  Goal reached ({goal}). Press R to reload or Q to quit.")
        if truncated and args.real_truncation:
            print(
                f"\n  Truncated (stuck fuse / loop). {fuse_line(emulator)}"
                f"\n  Press R to reload or Q to quit."
            )

    emulator.pyboy.stop(False)
    print_sep()
    print(f"  Done. steps={step_i} Σreward={total_reward:.4f} saves={saves_count}")
    if saves_count:
        print(f"  Last save dir: saves/{save_as}/")
    print_sep()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Human play with live reward/dialog/battle debug prints",
    )
    p.add_argument(
        "--from",
        dest="from_checkpoint",
        default="start",
        help="Save dir to load first (default: start)",
    )
    p.add_argument(
        "--save-as",
        default="manual_debug",
        help="Folder under saves/ for S key (default: manual_debug)",
    )
    p.add_argument(
        "--goal",
        default=None,
        help="Active curriculum goal (affects milestone scaling / terminated)",
    )
    p.add_argument(
        "--stage",
        default=None,
        help="Optional stage name — sets --goal from curriculum if --goal omitted",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=16,
        help="Hold duration per action, same as PPO (default: 16)",
    )
    p.add_argument(
        "--real-truncation",
        action="store_true",
        help="Keep agent stuck/loop fuses (default: disabled for human play)",
    )
    p.add_argument(
        "--all-steps",
        action="store_true",
        help="Print every step including idle None actions",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
