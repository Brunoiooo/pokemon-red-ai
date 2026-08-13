#!/usr/bin/env python3
"""Evaluate a Stable-Baselines3 PPO checkpoint on Pokemon Red."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from multiprocessing import set_start_method
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
    from sb3_contrib import MaskablePPO
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
        save_checkpoints=args.save_checkpoints,
    )
    # Monitor forbids step() after terminated; in-place advance keeps the
    # episode alive, so Monitor is fine. Still skip when auto for clarity.
    if not auto:
        env = Monitor(env)
    raw = _raw_env(env)

    print(f"Loading {model_path}")
    print(f"Auto curriculum: {'ON' if auto else 'OFF'}")
    print(f"Save checkpoints (any goal, order-free): {'ON' if args.save_checkpoints else 'OFF'}")
    print(f"Start stage: {stage}  goal={goal}  max_steps={max_steps}")
    model = MaskablePPO.load(str(model_path), device="cpu" if args.cpu else "auto")
    data = raw.emu.data

    verbose = int(getattr(args, "verbose", 0) or 0)
    ACTION_NAMES = ("NONE", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT")

    for ep in range(args.episodes):
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
        # -v: off-goal-map tracking — surfaces whether the agent is on a map
        # with no reachable, unfinished progress (see Data.active_map_events/
        # reward_active_map_presence), not just whether loop_flag fired
        # (which also covers spatial/menu loops on an otherwise fine map).
        off_path_steps = 0
        last_has_active_event: bool | None = None
        last_over_map_budget: bool | None = None
        if verbose:
            print(
                f"\n=== Episode {ep + 1}/{args.episodes} start "
                f"stage={info.get('stage', stage)} goal={info.get('goal')} "
                f"map={info.get('map_id')} milestone={info.get('milestone')} ==="
            )

        while not done:
            action, _ = model.predict(
                obs,
                action_masks=raw.action_masks(),
                deterministic=not args.stochastic,
            )
            action_i = int(action)
            obs, reward, terminated, truncated, info = env.step(action_i)
            total += float(reward)
            steps += 1
            if info.get("loop_flag"):
                loop_hits += 1

            # Dynamic, order-free checkpoint capture (see
            # PokemonRedEnv._save_milestone_checkpoints, called from inside
            # env.step() above when save_checkpoints=True): every
            # GOAL_CANDIDATES event/badge newly satisfied this step already
            # got its own saves/<name>/checkpoint.state written, whatever
            # event it is -- not just the curriculum's currently-assigned
            # goal or STAGE_ORDER's heuristic order. This is just the
            # -v/-vv visibility into that; the write itself already happened.
            if verbose and args.save_checkpoints:
                for milestone_name, _payout in data.last_milestone_payouts:
                    print(
                        f"  [checkpoint] saved -> saves/{milestone_name}/checkpoint.state"
                    )

            if args.heatmap and info.get("heatmap_positions"):
                from utils.PositionHeatmap import push_episode

                push_episode(
                    heatmap_queue,
                    info["heatmap_positions"],
                    info.get("heatmap_directions"),
                    info.get("heatmap_transitions"),
                    info.get("heatmap_rewards"),
                    info.get("heatmap_battle_outcomes"),
                    info.get("heatmap_milestones"),
                    info.get("heatmap_dialogs"),
                    info.get("heatmap_truncations"),
                    info.get("heatmap_steps") or 0,
                    info.get("stage", stage),
                    info.get("party_count"),
                    info.get("party_avg_level"),
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

            # -v: does the current map have a reachable, unfinished
            # GOAL_CANDIDATES event right now (see active_map_events)? Only
            # print on a status flip (like [map]/[milestone]) so a long camp
            # doesn't spam a line per step.
            cur_goal = info.get("goal")
            has_active_event = data.has_active_map_event()
            if not has_active_event:
                off_path_steps += 1
            if verbose and has_active_event != last_has_active_event:
                status = "on-path" if has_active_event else "OFF-PATH"
                print(
                    f"  [goal-map] step={steps:4d} map={map_id} goal={cur_goal} "
                    f"-> {status} (off_path_steps={off_path_steps})"
                )
            last_has_active_event = has_active_event

            # -v: size-scaled per-map step budget flip (see
            # Data.map_step_budget/map_budget_exceeded) — exceeding this
            # ramps a per-step penalty; the episode is only truncated at 2x
            # that, Data.map_truncate_budget, unlike the old flat/
            # informational-only map_dwell_budget this replaces.
            if verbose and map_id is not None:
                budget = data.map_step_budget(int(map_id))
                truncate_budget = data.map_truncate_budget(int(map_id))
                used = data.world_map_step_counts.get(int(map_id), 0)
                over_budget = used > budget
                if over_budget != last_over_map_budget:
                    print(
                        f"  [map-budget] step={steps:4d} map={map_id} "
                        f"used={used}/{budget}(/{truncate_budget}) "
                        f"-> {'OVER BUDGET' if over_budget else 'within budget'}"
                    )
                last_over_map_budget = over_budget

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
                # -vv only: current map's world_map_step_counts vs its step/
                # truncate budget (see Data.map_step_budget/map_truncate_budget),
                # every step -- same counter as the [map-budget] transition log
                # above and debug_play.py's map_counter_str, but unconditional.
                map_ctr = ""
                if verbose >= 2 and map_id is not None:
                    ctr_budget = data.map_step_budget(int(map_id))
                    ctr_truncate_budget = data.map_truncate_budget(int(map_id))
                    ctr_used = data.world_map_step_counts.get(int(map_id), 0)
                    map_ctr = f" map_ctr={ctr_used}/{ctr_budget}(/{ctr_truncate_budget})"
                print(
                    f"  [step] {steps:4d} act={aname:6s} rew={reward:+.4f} "
                    f"total={total:7.2f} map={map_id} mode={_mode_flags(raw.emu):9s} "
                    f"loop={int(bool(info.get('loop_flag')))} "
                    f"in_goal_map={int(has_active_event)} "
                    f"ms={info.get('milestone')}{map_ctr}"
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


def run_batch(args):
    """Headless-workers batch mode: run --batch total episodes/curriculum-legs
    across --batch-workers parallel PyBoy workers, pool their heatmap data
    into one unbounded RollingHeatmapAggregator per stage (+ "all"), and save
    static PNGs. Also usable with --heatmap: unlike plain eval, the vec-env
    workers churn through runs far faster than the live window can redraw,
    so rather than block on it every step, the window is fed the same way
    and left running — showing whatever it's managed to draw at any given
    moment — and, unlike plain eval, is *not* auto-closed at the end: a 300k-run
    batch is exactly the case where you want to sit down and read the result
    afterward, not have it vanish the instant the last worker finishes.
    """
    set_start_method("spawn", force=True)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        resolve_stage_name,
    )
    from env.pokemon_red_env import make_pokemon_env
    from utils.PositionHeatmap import RollingHeatmapAggregator, save_heatmap_images

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

    n_workers = max(1, args.batch_workers)
    target_runs = args.batch
    out_dir = (
        Path(args.batch_out)
        if args.batch_out
        else Path("heatmaps") / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    )

    print("=" * 70)
    print("  Pokemon Red AI - Batch Eval (heatmap export)")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Runs:       {target_runs:,}")
    print(f"Workers:    {n_workers}")
    print(f"Stage:      {stage}  goal={goal}  max_steps={max_steps}")
    print(f"Auto curr.: {'ON' if auto else 'OFF'}")
    print(f"Save ckpts: {'ON' if args.save_checkpoints else 'OFF'} (any goal, order-free)")
    print(f"Metrics:    {', '.join(args.batch_metrics)}")
    print(f"Out dir:    {out_dir}")
    print(f"Live view:  {'ON' if args.heatmap else 'OFF'}")
    print("=" * 70)

    heatmap_proc = heatmap_queue = None
    if args.heatmap:
        from utils.PositionHeatmap import start_heatmap_process

        heatmap_proc, heatmap_queue = start_heatmap_process(
            args.heatmap_frames, title="Pokemon Red AI - Position Heatmap (eval)"
        )

    def _make(rank: int):
        return make_pokemon_env(
            save_state=save_state,
            max_steps=max_steps,
            frame_skip=args.frame_skip,
            goal=goal,
            rank=rank,
            seed=args.seed,
            curriculum_mix=0.0,
            curriculum_saves=saves,
            auto_advance=auto,
            stage=stage,
            render_mode=None,
            n_workers=n_workers,
            collect_heatmap=True,
            save_checkpoints=args.save_checkpoints,
        )

    if n_workers <= 1:
        vec_env = DummyVecEnv([_make(0)])
    else:
        vec_env = SubprocVecEnv([_make(i) for i in range(n_workers)], start_method="spawn")

    model = MaskablePPO.load(str(model_path), device="cpu" if args.cpu else "auto")

    agg_all = RollingHeatmapAggregator(window_frames=None)
    stage_aggs: dict[str, RollingHeatmapAggregator] = {}

    def _record(info: dict) -> None:
        payload = (
            info["heatmap_positions"],
            info.get("heatmap_directions"),
            info.get("heatmap_transitions"),
            info.get("heatmap_rewards"),
            info.get("heatmap_battle_outcomes"),
            info.get("heatmap_milestones"),
            info.get("heatmap_dialogs"),
            info.get("heatmap_truncations"),
            info.get("heatmap_steps") or 0,
            info.get("party_count"),
            info.get("party_avg_level"),
        )
        agg_all.add_episode(*payload)
        stage_label = info.get("stage") or "unknown"
        stage_aggs.setdefault(
            stage_label, RollingHeatmapAggregator(window_frames=None)
        ).add_episode(*payload)
        if heatmap_queue is not None:
            from utils.PositionHeatmap import push_episode

            push_episode(heatmap_queue, *payload[:9], stage_label, *payload[9:])

    runs_done = 0
    goal_successes = 0
    obs = vec_env.reset()
    t0 = time.time()

    from tqdm.auto import tqdm

    pbar = tqdm(total=target_runs, unit="run", desc="batch eval")
    try:
        while runs_done < target_runs:
            action_masks = get_action_masks(vec_env)
            actions, _ = model.predict(
                obs, action_masks=action_masks, deterministic=not args.stochastic
            )
            obs, rewards, dones, infos = vec_env.step(actions)
            before = runs_done
            for info in infos:
                if info.get("heatmap_positions"):
                    _record(info)
                    runs_done += 1
                if info.get("goal_success"):
                    goal_successes += 1
            if runs_done > before:
                pbar.update(runs_done - before)
                pbar.set_postfix(successes=goal_successes, refresh=False)
    except KeyboardInterrupt:
        print("\nInterrupted - rendering heatmap from partial data...")
    finally:
        pbar.close()
        vec_env.close()

    elapsed = time.time() - t0
    print(
        f"Done: {runs_done:,} runs in {elapsed:.0f}s "
        f"({runs_done / max(elapsed, 1e-9):.1f} runs/s)"
    )
    if runs_done > 0:
        print(f"Rendering heatmap images to {out_dir} ...")
        aggregators = {"all": agg_all, **stage_aggs}
        written = save_heatmap_images(
            aggregators, out_dir,
            metrics=tuple(args.batch_metrics), top_maps=args.batch_top_maps,
        )
        print(f"Wrote {len(written)} image(s) to {out_dir}")
    else:
        print("No runs completed - nothing to render.")

    if heatmap_proc is not None:
        print(
            "Heatmap window stays open - close it manually when you're done "
            "(Ctrl+C here closes it now)."
        )
        try:
            heatmap_proc.join()
        except KeyboardInterrupt:
            from utils.PositionHeatmap import stop_heatmap_process

            print("\nClosing heatmap window...")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On goal success, overwrite saves/<new_stage>/checkpoint.state with "
             "the reached state (default: on; pass --no-save-checkpoints to disable)",
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
    p.add_argument(
        "--batch", type=int, default=None, metavar="N",
        help="Batch mode: run N total episodes/curriculum-legs across "
             "--batch-workers parallel workers and save static heatmap PNGs "
             "to --batch-out at the end. Pass --heatmap too to also open the "
             "same live window as plain eval, fed live from the batch run — "
             "it just won't auto-close when the batch finishes (close it "
             "yourself, or Ctrl+C). E.g. --batch 300000. All other eval "
             "params (--model, --stage, --goal, --checkpoint, --max-steps, "
             "--frame-skip, --stochastic, --auto-curriculum, --cpu) still "
             "apply.",
    )
    p.add_argument(
        "--batch-workers", type=int, default=8,
        help="Parallel PyBoy worker processes for --batch (default: 8)",
    )
    p.add_argument(
        "--batch-out", default=None,
        help="Output directory for --batch heatmap PNGs "
             "(default: heatmaps/eval_<timestamp>)",
    )
    p.add_argument(
        "--batch-metrics", nargs="+",
        choices=[
            "ticks", "reward", "winrate", "fleerate", "recency", "dialog_recency",
            "milestones", "truncations",
        ],
        default=["ticks"],
        help="Which heatmap metric(s) to render in --batch mode (default: ticks)",
    )
    p.add_argument(
        "--batch-top-maps", type=int, default=6,
        help="Per stage, also render single-map views for the N most-visited "
             "maps, in addition to the combined view (default: 6)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Base worker seed for --batch mode (each worker gets seed + rank)",
    )
    args = p.parse_args()
    if args.batch:
        run_batch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
