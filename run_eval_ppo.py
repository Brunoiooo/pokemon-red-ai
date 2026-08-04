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
            # Level 1: sampled (every 50 steps, or a notable reward).
            # Level 2 (-vv): every single step, i.e. every button press.
            show_step = verbose >= 2 or (
                verbose >= 1 and (steps % 50 == 0 or abs(float(reward)) >= 0.5)
            )
            if show_step:
                aname = ACTION_NAMES[action_i] if 0 <= action_i < len(ACTION_NAMES) else str(action_i)
                print(
                    f"  [step] {steps:4d} act={aname:6s} rew={reward:+.4f} "
                    f"total={total:7.2f} map={map_id} "
                    f"loop={int(bool(info.get('loop_flag')))} "
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

            # -v/-vv: battle outcome (win/lose/flee) and the levels it was judged against.
            exit_info = data.last_battle_exit_info
            if verbose >= 1 and exit_info:
                print(
                    f"  [battle] step={steps:4d} {exit_info['kind']:5s} "
                    f"enemy_lv={exit_info['enemy_level']} "
                    f"active_lv={exit_info.get('active_level')} "
                    f"party_max={exit_info['party_max']} "
                    f"scale={exit_info.get('difficulty_scale')} "
                    f"reward={exit_info['reward']:+.3f}"
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
            f"maps={maps_seen} "
            f"term={info.get('terminated')} trunc={info.get('truncated')}"
        )

    env.close()


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
    p.add_argument("--verbose", "-v", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
