#!/usr/bin/env python3
"""Evaluate a Stable-Baselines3 PPO checkpoint on Pokemon Red."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from ppo.checkpoints import resolve_model_path


def _raw_env(env):
    """Unwrap Monitor / wrappers to reach PokemonRedEnv."""
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def _sane(value, lo, hi):
    return lo <= value <= hi


def _mode_flags(emu) -> str:
    """Same mode string as debug_play.py's _mode_flags, for eval -v/-vv parity."""
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


def _mon_line(label, lv, hp, maxhp, atk, dfn, spd, spc):
    """One battler's stats + a plausibility flag (Gen1 ranges: lv 1-100,

    hp 0-714, other stats 1-999). A flagged line means an address/decoding
    bug, not just a "weak Pokemon" — the difficulty-scale bugs (wrong
    enemy_level address, little- vs big-endian stats) both showed up first
    as absurd numbers here (e.g. hp=4352, level via a field that's always 0).
    """
    ok = (
        _sane(lv, 1, 100)
        and _sane(hp, 0, 714)
        and _sane(maxhp, 1, 714)
        and _sane(atk, 1, 999)
        and _sane(dfn, 1, 999)
        and _sane(spd, 1, 999)
        and _sane(spc, 1, 999)
    )
    flag = "" if ok else " <-- SUSPECT"
    return (
        f"{label}(lv={lv} hp={hp}/{maxhp} atk={atk} def={dfn} spd={spd} spc={spc}){flag}"
    )


def run(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        resolve_stage_name,
    )
    from env.pokemon_red_env import PokemonRedEnv
    from pokemon.Data import GOAL_ALLOWED_MAPS

    model_path = resolve_model_path(getattr(args, "model", None))
    auto = bool(getattr(args, "auto_curriculum", True))
    stage = resolve_stage_name(args.stage)

    if auto:
        goal = get_goal_for_stage(stage)
        max_steps = args.max_steps if args.max_steps else get_stage_max_steps(stage)
        saves = get_curriculum_saves(stage)
        save_state = args.checkpoint if args.checkpoint != "start" else saves[-1]
    else:
        goal = args.goal
        max_steps = args.max_steps if args.max_steps else 5000
        save_state = args.checkpoint
        saves = [save_state]

    heatmap_proc = heatmap_queue = None
    if args.heatmap:
        from utils.PositionHeatmap import start_heatmap_process

        heatmap_proc, heatmap_queue = start_heatmap_process(
            args.heatmap_frames, title="Pokemon Red AI - Position Heatmap (eval)"
        )

    env = PokemonRedEnv(
        save_state=save_state,
        max_steps=max_steps,
        frame_skip=args.frame_skip,
        goal=goal,
        stage=stage,
        render_mode="human" if args.gui else None,
        curriculum_mix=0.0,
        curriculum_saves=saves,
        auto_advance=auto,
        worker_rank=0,
        n_workers=1,
        collect_heatmap=args.heatmap,
    )
    # Monitor forbids step() after terminated; in-place advance keeps the
    # episode alive, so Monitor is fine. Still skip when auto for clarity.
    if not auto:
        env = Monitor(env)
    raw = _raw_env(env)

    print(f"Loading {model_path}")
    print(f"Auto curriculum: {'ON' if auto else 'OFF'}")
    print(f"Save checkpoints on goal success: {'ON' if args.save_checkpoints else 'OFF'}")
    print(f"Start stage: {stage}  goal={goal}  max_steps={max_steps}")
    model = PPO.load(str(model_path), device="cpu" if args.cpu else "auto")
    data = raw.emu.data

    verbose = int(getattr(args, "verbose", 0) or 0)
    ACTION_NAMES = ("NONE", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT")

    for ep in range(args.episodes):
        stage = resolve_stage_name(args.stage)
        if auto:
            saves = get_curriculum_saves(stage)
            raw.set_curriculum(
                stage=stage,
                goal=get_goal_for_stage(stage),
                max_steps=get_stage_max_steps(stage) if not args.max_steps else args.max_steps,
                save_state=args.checkpoint if args.checkpoint != "start" else saves[-1],
                curriculum_saves=saves,
                curriculum_mix=0.0,
                reset_steps=True,
            )

        obs, info = env.reset()
        total = 0.0
        steps = 0
        stages_cleared: list[str] = []
        done = False
        last_map = info.get("map_id")
        last_milestone = info.get("milestone")
        maps_seen: list[int] = [int(last_map)] if last_map is not None else []
        loop_hits = 0
        blocked_printed: set[str] = set()
        # -v: off-goal-map tracking (mirrors reward_off_goal_camp's
        # GOAL_ALLOWED_MAPS check in Data.py) — surfaces whether the agent is
        # camping outside the current goal's critical path, not just whether
        # loop_flag fired (which also covers spatial/menu loops on-path).
        off_path_steps = 0
        last_in_allowed_map: bool | None = None
        if verbose:
            print(
                f"\n=== Episode {ep + 1}/{args.episodes} start "
                f"stage={info.get('stage', stage)} goal={info.get('goal')} "
                f"map={info.get('map_id')} milestone={info.get('milestone')} ==="
            )

        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action_i = int(action)
            obs, reward, terminated, truncated, info = env.step(action_i)
            total += float(reward)
            steps += 1
            if info.get("loop_flag"):
                loop_hits += 1

            if args.heatmap and info.get("heatmap_positions"):
                from utils.PositionHeatmap import push_episode

                push_episode(
                    heatmap_queue,
                    info["heatmap_positions"],
                    info.get("heatmap_directions"),
                    info.get("heatmap_transitions"),
                    info.get("heatmap_rewards"),
                    info.get("heatmap_steps") or 0,
                    info.get("stage", stage),
                    info.get("heatmap_wild"),
                )

            map_id = info.get("map_id")
            milestone = info.get("milestone")
            if map_id is not None and map_id != last_map:
                maps_seen.append(int(map_id))
                if verbose:
                    print(
                        f"  [map] step={steps} {last_map} -> {map_id} "
                        f"rew={reward:+.3f} total={total:.2f} "
                        f"stage={info.get('stage')} goal={info.get('goal')}"
                    )
                last_map = map_id
            if milestone and milestone != last_milestone:
                if verbose:
                    print(
                        f"  [milestone] step={steps} {last_milestone} -> {milestone} "
                        f"total={total:.2f}"
                    )
                last_milestone = milestone

            # -v: are we inside GOAL_ALLOWED_MAPS for the *current* goal? Only
            # print on a status flip (like [map]/[milestone]) so a long camp
            # doesn't spam a line per step; the [step] line below still shows
            # the instantaneous flag on every sampled/printed step.
            cur_goal = info.get("goal")
            allowed_maps = GOAL_ALLOWED_MAPS.get(cur_goal)
            in_allowed_map = allowed_maps is None or (
                map_id is not None and int(map_id) in allowed_maps
            )
            if not in_allowed_map:
                off_path_steps += 1
            if verbose and in_allowed_map != last_in_allowed_map:
                status = "on-path" if in_allowed_map else "OFF-PATH"
                print(
                    f"  [goal-map] step={steps:4d} map={map_id} goal={cur_goal} "
                    f"-> {status} (off_path_steps={off_path_steps})"
                )
            last_in_allowed_map = in_allowed_map

            # -v/-vv: goal-payout bookkeeping — proves the regress/re-pay farm
            # loop is closed (paid once, then blocked on re-entry) and shows
            # any regression clawback (see reward_goal_regression).
            if verbose >= 1:
                for name, payout in data.last_milestone_payouts:
                    print(
                        f"  [goal-pay] step={steps:4d} {name} +{payout:.3f} total={total:.2f}"
                    )
                for name in data.last_milestone_blocked:
                    # Fires every step the condition stays true (e.g. standing
                    # on the map) — print once per name per episode, not a
                    # per-step flood.
                    if name in blocked_printed:
                        continue
                    blocked_printed.add(name)
                    print(
                        f"  [goal-blocked] step={steps:4d} {name} "
                        f"(already regressed-and-spent this episode, no re-pay)"
                    )
                # "hard" = a payout was actually clawed back (goal wasn't
                # yet curriculum-cleared). Plain goals_regressed also lists
                # goals that just fell out of the live-set because the map
                # was already legitimately cleared and left behind — that's
                # expected curriculum progress, not a real regression, so
                # it's shown separately (and only at -vv) to avoid implying
                # a clawback that didn't happen.
                hard = info.get("goals_regressed_hard")
                if hard:
                    print(
                        f"  [goal-regress] step={steps:4d} {hard} total={total:.2f}"
                    )
                if verbose >= 2:
                    regressed = info.get("goals_regressed") or []
                    soft = [n for n in regressed if n not in (hard or [])]
                    if soft:
                        print(
                            f"    [goal-live-lost] step={steps:4d} {soft} "
                            f"(already cleared, no clawback — just left the map)"
                        )
            # Level 1: sampled (every 50 steps, or a notable reward).
            # Level 2 (-vv): every single step, i.e. every button press.
            show_step = verbose >= 2 or (
                verbose >= 1 and (steps % 50 == 0 or abs(float(reward)) >= 0.5)
            )
            if show_step:
                aname = ACTION_NAMES[action_i] if 0 <= action_i < len(ACTION_NAMES) else str(action_i)
                print(
                    f"  [step] {steps:4d} act={aname:6s} rew={reward:+.4f} "
                    f"total={total:7.2f} map={map_id} mode={_mode_flags(raw.emu):9s} "
                    f"loop={int(bool(info.get('loop_flag')))} "
                    f"in_goal_map={int(in_allowed_map)} "
                    f"ms={info.get('milestone')}"
                )

            # -vv: per-hit damage-reward scale (which levels it's judged against),
            # plus the raw stats behind it so a bad address/endianness decode
            # (absurd HP, level stuck at 0, etc.) is visible immediately.
            in_battle_now = data.is_battle(raw.emu.pyboy.memory)
            if verbose >= 2 and in_battle_now:
                hp_dbg = data.last_enemy_hp_debug
                if hp_dbg and abs(hp_dbg.get("scaled", 0.0)) >= 0.005:
                    print(
                        f"    [hit] step={steps:4d} enemy_lv={hp_dbg['enemy_level']} "
                        f"active_lv={hp_dbg['active_level']} "
                        f"scale={hp_dbg['difficulty_scale']:.2f} "
                        f"dmg_frac={hp_dbg['frac']:+.3f} reward={hp_dbg['scaled']:+.3f}"
                    )
                    mem = raw.emu.pyboy.memory
                    enemy_line = _mon_line(
                        "enemy",
                        data.enemy_level(mem),
                        data.enemy_hp(mem),
                        data.enemy_max_hp(mem),
                        data.enemy_attack(mem),
                        data.enemy_defense(mem),
                        data.enemy_speed(mem),
                        data.enemy_special(mem),
                    )
                    active_line = _mon_line(
                        "active",
                        data.pokemon_level(mem),
                        data.pokemon_current_hp(mem),
                        data.pokemon_max_hp(mem),
                        data.pokemon_attack(mem),
                        data.pokemon_defense(mem),
                        data.pokemon_speed(mem),
                        data.pokemon_special(mem),
                    )
                    print(f"    [stats] {enemy_line} | {active_line}")

            # -v/-vv: battle outcome (win/lose/flee/blackout) and the levels it
            # was judged against. "blackout" = full party wipe, detected from
            # wIsInBattle=0xFF directly (see reward_battle_exit) instead of
            # trusting wBattleResult, which HandlePlayerBlackOut never writes
            # — before this fix a wipe could misread as a stale "win".
            exit_info = data.last_battle_exit_info
            if verbose >= 1 and exit_info:
                print(
                    f"  [battle] step={steps:4d} {exit_info['kind']:8s} "
                    f"blacked_out={int(bool(exit_info.get('blacked_out')))} "
                    f"enemy_lv={exit_info['enemy_level']} "
                    f"active_lv={exit_info.get('active_level')} "
                    f"party_max={exit_info['party_max']} "
                    f"scale={exit_info.get('difficulty_scale')} "
                    f"reward={exit_info['reward']:+.3f}"
                )
                if exit_info.get("blacked_out"):
                    # Proves the free warp-heal (HP/status/PP -> full) isn't
                    # being paid: reward above should be ~battle_lost_penalty,
                    # not offset by a same-step [+hp-heal] near +1.0.
                    mem = raw.emu.pyboy.memory
                    print(
                        f"    [post-blackout hp] "
                        f"{data.pokemon_current_hp(mem)}/{data.pokemon_max_hp(mem)} "
                        f"(free heal — excluded from reward_core this step)"
                    )

            if info.get("goal_success"):
                cleared = info.get("cleared_stage") or stage
                stages_cleared.append(cleared)
                stage = info.get("stage") or stage
                print(
                    f"  [ok] {cleared} cleared (goal={get_goal_for_stage(cleared)}) "
                    f"-> advancing to {stage} "
                    f"(goal={info.get('goal')})"
                )
                if args.save_checkpoints:
                    out_dir = Path(f"saves/{stage}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "checkpoint.state"
                    with raw.emu.files_lock:
                        with open(out_path, "wb") as f:
                            raw.emu.pyboy.save_state(f)
                    print(f"  [checkpoint] saved -> {out_path}")

            if truncated or terminated:
                done = True

        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"return={total:.3f} steps={steps} "
            f"map={info.get('map_id')} badges={info.get('badges')} "
            f"milestone={info.get('milestone')} "
            f"stage={info.get('stage', stage)} cleared={stages_cleared or '-'} "
            f"loop={info.get('loop_flag')} loop_hits={loop_hits} "
            f"off_path_steps={off_path_steps} "
            f"maps={maps_seen} "
            f"term={info.get('terminated')} trunc={info.get('truncated')}"
        )

    env.close()
    if heatmap_proc is not None:
        from utils.PositionHeatmap import stop_heatmap_process

        stop_heatmap_process(heatmap_proc, heatmap_queue)


def main():
    p = argparse.ArgumentParser(description="Evaluate PPO Pokemon Red agent")
    p.add_argument(
        "--model", "-m", default=None,
        help="PPO .zip checkpoint (default: newest models/ppo_*/best/best_model.zip, "
             "else ppo_latest.zip)",
    )
    p.add_argument("--episodes", "-e", type=int, default=3)
    p.add_argument("--checkpoint", "-c", default="start")
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max steps (default: per-stage curriculum limit)",
    )
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument("--stage", default="stage_left_house", help="Starting curriculum stage")
    p.add_argument(
        "--goal", default="left_house",
        help="Fixed goal when --no-auto-curriculum",
    )
    p.add_argument(
        "--auto-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On goal success, advance stage in-place (default: on)",
    )
    p.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=False,
        help="On goal success, overwrite saves/<new_stage>/checkpoint.state with "
             "the reached state (opt-in; off by default)",
    )
    p.add_argument("--gui", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="-v: sampled step log + goal/off-path/battle events. "
             "-vv: every step, plus per-hit damage debug.",
    )
    p.add_argument(
        "--heatmap", action="store_true",
        help="Open a live position-heatmap window (avg ticks/run per map tile, "
             "rolling window of --heatmap-frames). <-/-> switches map.",
    )
    p.add_argument(
        "--heatmap-frames", type=int, default=300_000,
        help="Rolling window size in frames, pooled across all runs (default: 300000)",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
