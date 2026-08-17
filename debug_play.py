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
  python cli.py debug-play --from stage_route1
  python cli.py debug-play --from start --frame-skip 16 --real-truncation
  python cli.py debug-play --model  # + live model button-probability table

Parity with PPO training:
  - same discrete actions (0–8) via Emulator.step_discrete
  - same frame-skip hold (default 16)
  - same Data.goal / reward / fuse path
  - auto-curriculum advance on goal (disable with --no-auto-curriculum)
  - emulator is fully paused while idle (no key held) — it only advances
    when you press something, so nothing can happen off-screen that reward
    tracking would miss. Unrecognized keys still advance the game, just
    with ACTION_NONE as the action.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import argparse
import queue
from multiprocessing import RLock
from pathlib import Path

import keyboard

from curriculum_config import (
    CURRICULUM,
    DEFAULT_STAGE,
    GOAL_ORDER,
    get_goal_for_stage,
    stage_for_goal,
)
from pokemon import event_compass as _event_compass
from pokemon.Emulator import Emulator

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down", "None"]
NONE_ACTION = 8

# Same discrete indices as Emulator.buttons / PokemonRedEnv (N_ACTIONS=9).
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

# Global hook: ignore modifiers / lock keys so they never become ACTION_NONE.
_IGNORE_KEYS = frozenset(
    {
        "shift",
        "left shift",
        "right shift",
        "ctrl",
        "control",
        "left ctrl",
        "right ctrl",
        "alt",
        "left alt",
        "right alt",
        "alt gr",
        "windows",
        "left windows",
        "right windows",
        "caps lock",
        "num lock",
        "scroll lock",
        "menu",
        "pause",
        "print screen",
    }
)


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


def _policy_action_probs(model, obs, action_masks):
    """The policy's actual action distribution (post action-masking) for the
    current obs -- what the trained model would do right now, for comparison
    against whatever button the human just pressed."""
    import torch as th

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=action_masks)
        probs = distribution.distribution.probs[0].cpu().numpy()
    return probs


def _probs_table(probs, action_names) -> str:
    """%-per-button, sorted highest share first."""
    order = sorted(range(len(action_names)), key=lambda i: probs[i], reverse=True)
    return " ".join(f"{action_names[i]}={100.0 * probs[i]:5.1f}%" for i in order)


def legal_actions_str(emu: Emulator) -> str:
    """Human-readable Data.legal_action_mask() -- same mask MaskablePPO gets
    from PokemonRedEnv.action_masks() before it picks an action, so a human
    can see live which buttons the policy is even allowed to press."""
    mask = emu.data.legal_action_mask(emu.pyboy.memory)
    allowed = [BUTTON_NAMES[i] for i, ok in enumerate(mask) if ok]
    return ",".join(allowed) if allowed else "(none)"


def status_line(emu: Emulator) -> str:
    mem = emu.pyboy.memory
    mid = emu.data.map_id(mem)
    x = emu.data.position_x(mem)
    y = emu.data.position_y(mem)
    did = emu.data.dialog_id(mem)
    state = emu.data.current_map_script_state(mem)
    state_s = f" script_state={state}" if state is not None else ""
    return (
        f"map={mid} pos=({x},{y}) mode={_mode_flags(emu)} "
        f"dialog_id={did} goal={emu.data.goal} legal=[{legal_actions_str(emu)}]{state_s}"
    )


# Gen 1 item IDs (constants/item_constants.asm in pret/pokered).
ITEM_NAMES: dict[int, str] = {
    1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poké Ball",
    5: "Town Map", 6: "Bicycle", 7: "Surfboard", 8: "Safari Ball",
    9: "Pokédex", 10: "Moon Stone", 11: "Antidote", 12: "Burn Heal",
    13: "Ice Heal", 14: "Awakening", 15: "Parlyz Heal", 16: "Full Restore",
    17: "Max Potion", 18: "Hyper Potion", 19: "Super Potion", 20: "Potion",
    21: "Boulderbadge", 22: "Cascadebadge", 23: "Thunderbadge",
    24: "Rainbowbadge", 25: "Soulbadge", 26: "Marshbadge", 27: "Volcanobadge",
    28: "Earthbadge", 29: "Escape Rope", 30: "Repel", 31: "Old Amber",
    32: "Fire Stone", 33: "Thunder Stone", 34: "Water Stone", 35: "HP Up",
    36: "Protein", 37: "Iron", 38: "Carbos", 39: "Calcium", 40: "Rare Candy",
    41: "Dome Fossil", 42: "Helix Fossil", 43: "Secret Key",
    45: "Bike Voucher", 46: "X Accuracy", 47: "Leaf Stone", 48: "Card Key",
    49: "Nugget", 51: "Poké Doll", 52: "Full Heal", 53: "Revive",
    54: "Max Revive", 55: "Guard Spec.", 56: "Super Repel", 57: "Max Repel",
    58: "Dire Hit", 59: "Coin", 60: "Fresh Water", 61: "Soda Pop",
    62: "Lemonade", 63: "S.S. Ticket", 64: "Gold Teeth", 65: "X Attack",
    66: "X Defend", 67: "X Speed", 68: "X Special", 69: "Coin Case",
    70: "Oak's Parcel", 71: "Itemfinder", 72: "Silph Scope", 73: "Poké Flute",
    74: "Lift Key", 75: "Exp. All", 76: "Old Rod", 77: "Good Rod",
    78: "Super Rod", 79: "PP Up", 80: "Ether", 81: "Max Ether",
    82: "Elixer", 83: "Max Elixer",
    **{0xC4 + i: f"HM{i + 1:02d}" for i in range(5)},
    **{0xC9 + i: f"TM{i + 1:02d}" for i in range(50)},
}


def item_name(item_id: int) -> str:
    return ITEM_NAMES.get(item_id, f"Item#{item_id}")


def bag_items(emu: Emulator) -> tuple[tuple[int, int], ...]:
    """Non-empty (id, qty) bag slots, same layout PokemonRedEnv rewards off."""
    mem = emu.pyboy.memory
    ids = emu.data.items_ids(mem)
    qtys = emu.data.items_quantities(mem)
    out = []
    for item_id, qty in zip(ids, qtys):
        if item_id in (0, 255):
            break
        out.append((item_id, qty))
    return tuple(out)


def bag_line(emu: Emulator) -> str:
    items = bag_items(emu)
    if not items:
        return "items: (empty)"
    return "items: " + ", ".join(
        f"{item_name(item_id)} x{qty}" for item_id, qty in items
    )


def map_counter_str(emu: Emulator) -> str:
    """Current map's world_map_step_counts vs its truncate budget (see
    Data.map_truncate_budget) -- the same per-map dwell counter Data.clean() /
    set_curriculum(clear_visits=True) zero out."""
    data = emu.data
    mid = data.map_id(emu.pyboy.memory)
    truncate_budget = data.map_truncate_budget(mid)
    used = data.world_map_step_counts.get(mid, 0)
    return f"{used}/{truncate_budget}"


# Unicode arrows -- fine for normal interactive play (same risk profile as
# print_help's existing em-dashes), but will UnicodeEncodeError if stdout is
# ever piped/redirected under the cp1250 console codepage (see
# MilestoneCallback's curriculum-advance print for the same gotcha hit
# elsewhere in this project). debug_play.py is always run directly in a
# console, never redirected, so that's an accepted tradeoff here.
_COMPASS_ARROWS = {(1, 0): "→", (-1, 0): "←", (0, 1): "↓", (0, -1): "↑", (0, 0): "•"}


def compass_line(emu: Emulator) -> str:
    """Live pokemon.event_compass.nearest_unlockable_event() reading -- a
    diagnostic-only display, deliberately NOT the same signal
    Data.compass_progress() feeds the model (that one still runs the older
    pokemon.navigation/pokemon.goal_positions BFS over the curated
    GOAL_CANDIDATES pool, untouched here). Switched debug_play's own display
    to the exhaustive, static-analysis-derived event_compass instead, after
    live play repeatedly caught the older compass pointing at goals with no
    real walkable trigger at all (an entrance-tile fallback guess, not a
    verified location) -- event_compass never guesses: an event only
    becomes a candidate once tools/gen_event_triggers.py's asm resolver has
    actually found a real SetEvent site for it, so there is nothing to
    manually blacklist here."""
    data = emu.data
    memory = emu.pyboy.memory
    if not data.is_world(memory):
        return "compass: (not in world)"
    result = _event_compass.nearest_unlockable_event(data.get_position(), memory)
    if result is None:
        return "compass: no reachable target"
    dx, dy, dist, target = result
    arrow = _COMPASS_ARROWS.get((dx, dy), f"({dx:+d},{dy:+d})")
    return f"compass: {arrow} dist={dist} target={target}"


def fuse_line(emu: Emulator) -> str:
    data = emu.data
    pos = data.get_position()
    tile = data.visited_positions.get(pos, 0)
    line = (
        f"fuses: tile={tile}/{data.max_useless_ticks} "
        f"dialog={int(data.in_dialog_ticks)}/{data.max_useless_dialog_ticks} "
        f"battle={int(data.in_battle_ticks)}/{data.max_useless_battle_ticks} "
        f"menu={int(data.in_menu_ticks)}/{data.max_useless_ticks} "
        f"loop={data.loop_streak}/{data.max_loop_streak}"
    )
    # Size-scaled per-map world-step budget (see Data.map_truncate_budget) —
    # the episode is truncated once it's exceeded (see
    # Data.map_truncate_exceeded), unlike the old flat/informational-only
    # map_dwell_budget display this replaces.
    mid = data.map_id(emu.pyboy.memory)
    line += f" map_budget[{mid}]={map_counter_str(emu)}"
    return line


def print_help() -> None:
    print_sep("-")
    print("  Arrows / A / B / Enter / Space  — play (same indices as PPO)")
    print("  Other keys                     — ACTION_NONE (idx 8)")
    print("  (idle)                         — emulator paused, no tick")
    print("  S                              — save state")
    print("  R                              — reload --from")
    print("  H                              — this help + status")
    print("  ESC / Q                        — quit")
    print("  status line 'legal=[...]'      — Data.legal_action_mask() right now:")
    print("    the buttons MaskablePPO is allowed to sample this tick (see")
    print("    PokemonRedEnv.action_masks). Dialog -> A/B only; battle/menu ->")
    print("    everything but NONE; world -> everything. Pressing a button")
    print("    outside that set prints a 'MASKED(btn)' tag on that step -- it")
    print("    still executes here (this is human play, nothing is enforced),")
    print("    it's just what the trained policy could never have picked.")
    print("  compass line                   — pokemon.event_compass.nearest_unlockable_event():")
    print("    diagnostic only, NOT what feeds the model's reward (that's still")
    print("    Data.compass_progress) -- exhaustive asm-resolver-derived compass,")
    print("    never guesses a location (arrow = next-hop direction, dist = hop")
    print("    count, target = which event it's routing to).")
    print_sep("-")


def _resolve_stage_goal(args: argparse.Namespace) -> tuple[str, str]:
    """Match train_ppo defaults: unset --stage/--goal -> DEFAULT_STAGE."""
    if args.goal:
        goal = str(args.goal)
        if goal not in GOAL_ORDER:
            # A bare goal id (e.g. "gave_parcel") was expected, but this looks
            # like a stage name (e.g. "stage_gave_parcel") got passed instead.
            # stage_for_goal() now warns (see curriculum_config.py) rather
            # than silently falling back for any unrecognized goal, which
            # used to make data.goal permanently unmatchable (no milestone
            # ever equals it -> stuck on muted rewards and auto-advance
            # never fires) with no error at all.
            hint = ""
            if goal in CURRICULUM:
                hint = (
                    f" Did you mean --goal {CURRICULUM[goal]['goal']!r} "
                    f"(or drop --goal and pass --stage {goal!r} instead)?"
                )
            raise SystemExit(
                f"--goal {goal!r} is not a known goal id. Valid values: "
                f"{', '.join(GOAL_ORDER)}.{hint}"
            )
        stage = args.stage or stage_for_goal(goal)
        return stage, goal
    stage = args.stage
    if stage is None and args.from_checkpoint in CURRICULUM:
        stage = args.from_checkpoint
    if stage is None:
        stage = DEFAULT_STAGE
    return stage, get_goal_for_stage(stage) or GOAL_ORDER[0]


def _clear_visits(data) -> None:
    """Same exploration reset as PokemonRedEnv.set_curriculum(clear_visits=True)."""
    data.visited_positions.clear()
    data.position_visit_counts.clear()
    data.direction_counts.clear()
    data.map_transitions.clear()
    data.reward_sums.clear()
    data.reward_component_sums.clear()
    data.battle_outcome_counts.clear()
    data.milestone_hit_counts.clear()
    data.dialog_hit_counts.clear()
    data.truncate_hit_counts.clear()
    data.world_map_step_counts.clear()
    data.dialog_id_visit_counts.clear()
    data.map_id_visit_counts.clear()
    data.menu_noop_streak = 0
    data.dialog_wrong_streak = 0
    data._last_heatmap_pos = None
    data.recent_positions.clear()
    data.recent_actions.clear()
    data.loop_flag = False
    data.loop_streak = 0
    data.in_menu_ticks = 0
    data.in_battle_ticks = 0
    data.in_dialog_ticks = 0
    data._dialog_screens_seen = set()
    data._completed_dialogs = set()
    data._dialog_reopen_counts = {}


def _advance_after_goal(emu: Emulator, stage: str, goal: str) -> tuple[bool, str, str]:
    """In-place curriculum advance — mirrors PokemonRedEnv._advance_after_goal.

    stage/goal are deliberately left unchanged (see that docstring for why:
    reward doesn't gate on the assigned goal, so reassigning it mid-episode
    only risks handing out a goal nowhere near reachable this run). Only the
    per-leg counters get a fresh start.
    """
    emu.data.mark_goal_cleared(goal)
    _clear_visits(emu.data)
    return True, stage, goal


def run(args: argparse.Namespace) -> None:
    from_ckpt = args.from_checkpoint
    save_as = args.save_as or "manual_debug"
    frame_skip = int(args.frame_skip)
    stage, goal = _resolve_stage_goal(args)
    auto_advance = bool(getattr(args, "auto_curriculum", True))

    key_queue: queue.Queue[tuple[str, int | None]] = queue.Queue()

    def on_key(event: keyboard.KeyboardEvent) -> None:
        if event.event_type != keyboard.KEY_DOWN:
            return
        name = (event.name or "").lower()
        if not name or name in _IGNORE_KEYS:
            return
        entry = KEY_MAP.get(name)
        if entry is not None:
            key_queue.put_nowait(entry)
        else:
            # ACTION_NONE only for unrecognized keys — never on idle timeout.
            key_queue.put_nowait(("?", NONE_ACTION))

    keyboard.hook(on_key, suppress=False)

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = True
    emulator.data.goal = goal

    # --model: load a PPO checkpoint purely to print what it would do from
    # each state you land in (post-step, same obs timing the next
    # model.predict() call would see) -- lets you compare its live
    # button-probability table against what you just pressed, to spot where
    # its policy disagrees with sensible play. It never touches the controls.
    model = None
    obs_env = None
    if args.model is not None:
        from sb3_contrib import MaskablePPO

        from env.pokemon_red_env import PokemonRedEnv
        from ppo.checkpoints import resolve_model_path

        model_path = resolve_model_path(args.model or None)
        print(f"  Loading model: {model_path}")
        model = MaskablePPO.load(str(model_path), device="cpu")
        obs_env = PokemonRedEnv(save_state=from_ckpt, goal=goal, stage=stage)
        obs_env._emu = emulator

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
    print(f"  Stage / goal : {stage} / {goal}")
    print(f"  Frame skip   : {frame_skip}  (PPO default is 16)")
    print(f"  Speed        : 1x (human)")
    print(
        f"  Curriculum   : "
        f"{'auto-advance (train parity)' if auto_advance else 'fixed goal'}"
    )
    print(
        f"  Truncation   : "
        f"{'REAL (agent fuses)' if args.real_truncation else 'disabled (human)'}"
    )
    print(f"  Log every    : {'all steps' if args.all_steps else 'button presses only'}")
    print("  Idle         : emulator paused (no tick) until a key is pressed")
    print_sep()
    print_help()
    print()

    def model_line(inputs) -> str | None:
        """What the loaded checkpoint would do right now, sorted highest first.
        None if --model wasn't passed."""
        if model is None:
            return None
        obs_env.goal = goal
        obs = obs_env._torch_inputs_to_obs(inputs)
        action_masks = obs_env.action_masks()
        probs = _policy_action_probs(model, obs, action_masks)
        return f"  [model] {_probs_table(probs, BUTTON_NAMES)}"

    memory, inputs = emulator.reset(dir=from_ckpt)
    emulator.data.goal = goal
    emulator.data.seed_cleared_goals(goal)
    # Real-time pace: PyBoy only rate-limits once per tick() call, so batched
    # tick(duration) would still run near-instant. render_each paces each frame.
    emulator.pyboy.set_emulation_speed(1)
    emulator.pyboy.tick(1, render=True, sound=False)
    print(f"  Loaded. {status_line(emulator)}")
    print(f"  {fuse_line(emulator)}")
    print(f"  {compass_line(emulator)}")
    print(f"  {bag_line(emulator)}")
    line = model_line(inputs)
    if line:
        print(line)
    print("  Play toward the first battle dialog and watch reward / mode lines.\n")

    prev_mode = _mode_flags(emulator)
    prev_dialog = emulator.data.dialog_id(emulator.pyboy.memory)
    prev_bag = bag_items(emulator)
    step_i = 0
    total_reward = 0.0
    saves_count = 0
    base_stage, base_goal = stage, goal
    goal_done_announced = False

    while True:
        try:
            label, action = key_queue.get(timeout=0.016)
        except queue.Empty:
            # Idle: emulator is fully paused (no tick at all). Nothing can
            # change in-game while you're not pressing anything, so there is
            # nothing to miss. tick(0) just pumps the SDL window's OS events
            # (keeps it from looking unresponsive) without advancing a frame.
            emulator.pyboy.tick(0, render=False, sound=False)
            continue

        if label in ("ESC", "Q"):
            print("\nQuit.")
            break
        if label == "HELP":
            print_help()
            print(f"  {status_line(emulator)}")
            print(f"  {fuse_line(emulator)}")
            print(f"  {compass_line(emulator)}")
            print(f"  {bag_line(emulator)}")
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
            memory, inputs = emulator.reset(dir=from_ckpt)
            stage, goal = base_stage, base_goal
            emulator.data.goal = goal
            emulator.data.seed_cleared_goals(goal)
            emulator.pyboy.tick(1, render=True, sound=False)
            prev_mode = _mode_flags(emulator)
            prev_dialog = emulator.data.dialog_id(emulator.pyboy.memory)
            prev_bag = bag_items(emulator)
            step_i = 0
            total_reward = 0.0
            goal_done_announced = False
            print(
                f"  Reloaded from saves/{from_ckpt}/  "
                f"stage={stage} ({status_line(emulator)})"
            )
            print(f"  {compass_line(emulator)}")
            print(f"  {bag_line(emulator)}")
            line = model_line(inputs)
            if line:
                print(line)
            continue

        # Mask as computed *before* the tick -- the same point in time
        # MaskablePPO reads it at (action_masks() is called on the
        # pre-action observation, see PokemonRedEnv.action_masks). Only
        # meaningful for real button presses (label is a KEY_MAP entry, i.e.
        # action is one of 0-8) -- SAVE/RELOAD/HELP never reach here.
        pre_mask = emulator.data.legal_action_mask(emulator.pyboy.memory)
        was_masked = action is not None and not pre_mask[action]

        # Match PPO hold length, but pace frame-by-frame at 1x (human speed).
        memory, inputs, reward, terminated, truncated = emulator.step_discrete(
            memory=memory,
            action_idx=action,
            duration=frame_skip,
            render_each=True,
        )
        step_i += 1
        total_reward += float(reward)

        # Always shown, every environment step (not gated by --all-steps /
        # "interesting", unlike the rest of this log) -- see map_counter_str.
        print(f"       map={map_counter_str(emulator)}")

        cleared_goal: str | None = None
        advanced = False
        # Train/eval parity: success advances in-place instead of ending forever.
        if terminated and auto_advance and not truncated:
            cleared_goal = goal
            advanced, stage, goal = _advance_after_goal(emulator, stage, goal)
            if advanced:
                terminated = False
                goal_done_announced = False

        mode = _mode_flags(emulator)
        did = emulator.data.dialog_id(emulator.pyboy.memory)
        mode_changed = mode != prev_mode
        dialog_changed = did != prev_dialog
        cur_bag = bag_items(emulator)
        bag_changed = cur_bag != prev_bag
        interesting = (
            action != NONE_ACTION
            or mode_changed
            or dialog_changed
            or bag_changed
            or abs(float(reward)) >= 0.005
            or emulator.data.loop_flag
            or advanced
            or terminated
            or truncated
            or was_masked
        )

        if args.all_steps or interesting:
            btn = BUTTON_NAMES[action] if 0 <= action < len(BUTTON_NAMES) else "?"
            milestone = getattr(emulator, "last_milestone", 0.0)
            step_r = getattr(emulator, "last_step", 0.0)
            marks = []
            if was_masked:
                # This is the exact button MaskablePPO could never sample here
                # (Data.legal_action_mask) -- e.g. NONE in dialog/battle/menu,
                # or any non-A/B button while a dialog textbox is open.
                marks.append(f"MASKED({btn})")
            if mode_changed:
                marks.append(f"{prev_mode}->{mode}")
            if dialog_changed:
                marks.append(f"dialog {prev_dialog}->{did}")
            if bag_changed:
                marks.append("BAG")
            if emulator.data.loop_flag:
                causes = ",".join(sorted(emulator.data.loop_causes)) or "?"
                marks.append(f"LOOP[{causes}]")
            # _last_regressed mixes real clawbacks with the ordinary "left a
            # map whose location goal is curriculum-cleared" case (ambiguous
            # for a human reading the log) — only _last_hard_regressed
            # actually docked reward. See Data._last_hard_regressed.
            hard_regressed = getattr(emulator.data, "_last_hard_regressed", None) or []
            soft_regressed = [
                g
                for g in (getattr(emulator.data, "_last_regressed", None) or [])
                if g not in hard_regressed
            ]
            if hard_regressed:
                marks.append(f"REGRESS({','.join(hard_regressed)})")
            if soft_regressed:
                marks.append(f"goal_off_map({','.join(soft_regressed)})")
            if advanced and cleared_goal:
                marks.append(f"CLEAR({cleared_goal})->{goal}")
            elif terminated:
                marks.append("TERM(goal)")
            if truncated:
                marks.append("TRUNC")
            exit_info = getattr(emulator.data, "last_battle_exit_info", None) or getattr(
                emulator.data, "last_flee_info", None
            )
            if exit_info:
                kind = str(exit_info.get("kind", "?")).upper()
                if kind in ("SMART", "COWARD"):
                    tag = f"FLEE_{kind}"
                    # Flee risk is judged on the whole team's strongest mon.
                    cmp_lv = exit_info.get("party_max")
                    cmp_tag = "p"
                elif kind == "WIN":
                    tag = "BATTLE_WIN"
                    # Win reward is judged on the mon that actually fought.
                    cmp_lv = exit_info.get("active_level")
                    cmp_tag = "a"
                elif kind == "LOSE":
                    tag = "BATTLE_LOSE"
                    cmp_lv = exit_info.get("active_level")
                    cmp_tag = "a"
                else:
                    tag = f"BATTLE_{kind}"
                    cmp_lv = exit_info.get("active_level")
                    cmp_tag = "a"
                scale = exit_info.get("difficulty_scale")
                scale_s = f"*{scale:.2f}" if scale is not None else ""
                marks.append(
                    f"{tag}"
                    f"(e{exit_info.get('enemy_level')}"
                    f"/{cmp_tag}{cmp_lv}"
                    f"{scale_s}"
                    f"={exit_info.get('reward', 0):+.2f})"
                )
            hp_dbg = getattr(emulator.data, "last_enemy_hp_debug", None)
            if hp_dbg and emulator.data.is_battle(emulator.pyboy.memory) and abs(
                hp_dbg.get("scaled", 0.0)
            ) >= 0.005:
                marks.append(
                    f"HIT(e{hp_dbg.get('enemy_level')}/a{hp_dbg.get('active_level')}"
                    f"*{hp_dbg.get('difficulty_scale', 1.0):.2f}"
                    f"={hp_dbg.get('scaled', 0.0):+.3f})"
                )
            mark_s = f"  [{' | '.join(marks)}]" if marks else ""
            live_n = len(emulator.data.live_story_goals())
            peak_n = int(getattr(emulator.data, "_peak_live_goals", 0) or 0)
            print(
                f"#{step_i:<5} {btn:<6} r={reward:+.4f} "
                f"(m={milestone:+.4f} s={step_r:+.4f})  "
                f"{status_line(emulator)}  "
                f"goals={live_n}/{peak_n} loop={emulator.data.loop_streak}{mark_s}"
            )
            print(f"       {compass_line(emulator)}")
            line = model_line(inputs)
            if line:
                print(line)
            if exit_info:
                kind = str(exit_info.get("kind", "?"))
                label = (
                    f"FLEE {kind}"
                    if kind in ("smart", "coward")
                    else f"BATTLE {kind.upper()}"
                )
                print(
                    f"       {label}: enemy_lv={exit_info.get('enemy_level')} "
                    f"active_lv={exit_info.get('active_level')} "
                    f"party={exit_info.get('party_levels')} max={exit_info.get('party_max')} "
                    f"scale={exit_info.get('difficulty_scale')} "
                    f"CF0B={exit_info.get('battle_result')} "
                    f"→ {exit_info.get('reward', 0):+.2f}"
                )
            if (
                mode_changed
                or dialog_changed
                or truncated
                or terminated
                or advanced
                or exit_info
            ):
                print(f"       {fuse_line(emulator)}  Σr={total_reward:.4f}")
            if bag_changed:
                print(f"       {bag_line(emulator)}")

        prev_mode = mode
        prev_dialog = did
        prev_bag = cur_bag

        if advanced and cleared_goal:
            print(
                f"\n  ✓ Cleared {cleared_goal} → stage={stage} goal={goal} "
                f"(auto-curriculum, same as train)\n"
            )
        elif terminated and not goal_done_announced:
            goal_done_announced = True
            print(
                f"\n  Goal reached ({goal}) — end of curriculum or fixed goal.\n"
                f"  Press R to reload or Q to quit.\n"
            )
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
        help="Active curriculum goal (default: left_house, same as train)",
    )
    p.add_argument(
        "--stage",
        default=None,
        help="Optional stage — sets --goal from curriculum if --goal omitted",
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
        "--auto-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On goal success, advance stage like train (default: on)",
    )
    p.add_argument(
        "--all-steps",
        action="store_true",
        help="Print every agent step (incl. unrecognized-key None)",
    )
    p.add_argument(
        "--model",
        nargs="?",
        const="",
        default=None,
        help="Load a PPO checkpoint and print its live button-probability "
             "table (sorted highest first) after every step, next to what "
             "you actually pressed -- read-only, never touches the controls. "
             "Bare --model auto-picks the newest models/ppo_*/best/best_model.zip.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
