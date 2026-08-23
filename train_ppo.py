#!/usr/bin/env python3
"""Train Pokemon Red with Stable-Baselines3 PPO."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from multiprocessing import set_start_method

import curriculum_config
import tunables
from pokemon.Data import Data
from pokemon.Emulator import Emulator


def _int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run(args):
    set_start_method("spawn", force=True)

    # Apply --advance-success-threshold / --default-max-steps / etc. before
    # anything else touches curriculum_config (resolve_stage_name,
    # get_stage_max_steps, CURRICULUM, ...) so every lookup below sees the
    # override, not the hardcoded default.
    curriculum_config.configure_curriculum(
        **tunables.kwargs_from_args(args, curriculum_config.CurriculumConfig)
    )
    # Reward-shaping (Data) / emulator-timing (Emulator) overrides, splatted
    # into every PokemonRedEnv() below via its data_kwargs/emulator_kwargs
    # params (kept separate -- Data(...) and Emulator(...) have disjoint
    # constructor signatures).
    data_kwargs = tunables.kwargs_from_args(args, Data)
    emulator_kwargs = tunables.kwargs_from_args(args, Emulator)

    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from curriculum_config import (
        GOAL_INDEX,
        GOAL_ORDER,
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        resolve_stage_name,
    )
    from env.pokemon_red_env import PokemonRedEnv
    from ppo.callbacks import MilestoneCallback
    from ppo.checkpoints import atomic_model_save, resolve_model_path
    from ppo.features import PokemonFeaturesExtractor
    from ppo.migrate import migrate_state_dict

    if args.resume is not None and args.migrate is not None:
        raise SystemExit("Pass either --resume or --migrate, not both.")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"ppo_{session}"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models") / f"ppo_{session}"
    model_dir.mkdir(parents=True, exist_ok=True)

    stage = resolve_stage_name(args.stage)
    goal = args.goal or get_goal_for_stage(stage)
    max_steps = args.max_steps or get_stage_max_steps(stage)
    # get_curriculum_saves always puts "start" first (see its docstring) --
    # training resets from curriculum_saves[-1] (the frontier checkpoint)
    # most episodes, and from "start" with probability args.curriculum_mix
    # (see PokemonRedEnv._pick_save), so the true game start stays in the
    # mix instead of only ever drilling forward from the frontier.
    curriculum_saves = get_curriculum_saves(stage)
    train_save_state = curriculum_saves[-1]

    # SubprocVecEnv requires identical render_mode on every worker, so --gui
    # enables SDL on all training workers. Eval stays headless: a second SDL
    # PyBoy in the main process (DummyVecEnv) conflicts with the first and
    # crashes on the next EvalCallback run if the window is left open.
    if args.gui and args.workers > 1:
        print(
            f"Note: --gui opens an SDL window on each of {args.workers} workers."
        )

    render_mode = "human" if args.gui else None
    if args.gui:
        from utils.gui_layout import window_slot

        slot0 = window_slot(0, args.workers)
        print(
            f"GUI layout: {slot0.cols}x{slot0.rows} grid, "
            f"scale x{slot0.scale} ({args.workers} window(s))"
        )

    print("=" * 70)
    print("  Pokemon Red AI - PPO Training (Stable-Baselines3)")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Workers:    {args.workers}")
    print(f"Stage:      {stage}")
    print(f"Goal:       {goal}")
    print(f"Max steps:  {max_steps}")
    print(f"Frame skip: {args.frame_skip}")
    if len(curriculum_saves) > 1:
        print(
            f"Train save: {train_save_state} "
            f"({args.curriculum_mix:.0%} mix-in from {curriculum_saves[:-1]})"
        )
    else:
        print(f"Train save: {train_save_state}")
    print("Eval save:  start")
    print(f"Saves:      {curriculum_saves}")
    print(f"Log dir:    {log_dir}")
    print(f"Total steps:{args.timesteps:,}")
    print(f"GUI:        {'ON' if args.gui else 'OFF'}")
    print(f"Auto curr.: {'ON' if args.auto_curriculum else 'OFF'}")
    print(
        f"Save ckpts: eval=ON train="
        f"{'ON' if args.save_checkpoints else 'OFF'}"
    )
    print(f"Heatmap:    {'ON (' + format(args.heatmap_frames, ',') + ' frame window)' if args.heatmap else 'OFF'}")
    print("=" * 70)

    heatmap_proc = heatmap_queue = None
    if args.heatmap:
        from utils.PositionHeatmap import start_heatmap_process

        heatmap_proc, heatmap_queue = start_heatmap_process(args.heatmap_frames)

    def _make(rank: int):
        def _thunk():
            env = PokemonRedEnv(
                save_state=train_save_state,
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                stage=stage,
                curriculum_mix=args.curriculum_mix,
                curriculum_saves=curriculum_saves,
                render_mode=render_mode,
                auto_advance=args.auto_curriculum,
                worker_rank=rank,
                n_workers=args.workers,
                collect_heatmap=args.heatmap,
                save_checkpoints=args.save_checkpoints,
                data_kwargs=data_kwargs,
                emulator_kwargs=emulator_kwargs,
            )
            env.reset(seed=args.seed + rank)
            return env

        return _thunk

    if args.workers <= 1:
        train_env = DummyVecEnv([_make(0)])
    else:
        train_env = SubprocVecEnv(
            [_make(i) for i in range(args.workers)], start_method="spawn"
        )
    train_env = VecMonitor(train_env, str(log_dir / "monitor"))

    eval_env = DummyVecEnv(
        [
            lambda: PokemonRedEnv(
                # Always the true game start, regardless of --stage/--goal or
                # curriculum_saves -- eval measures whole-game performance,
                # not the narrow segment training happens to be drilling.
                save_state="start",
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                stage=stage,
                curriculum_mix=0.0,
                curriculum_saves=["start"],
                render_mode=None,
                auto_advance=args.auto_curriculum,
                # Eval episodes are deterministic (near-)greedy rollouts, so
                # a goal they reach is one the policy can reliably repeat --
                # a much stronger "mastered" signal than a noisy exploratory
                # training episode stumbling into it once. This is now the
                # primary writer of saves/<goal>/checkpoint.state (see
                # curriculum_config.pick_new_goal's docstring); training
                # workers default to --no-save-checkpoints so there's only
                # one writer per goal directory.
                save_checkpoints=True,
                data_kwargs=data_kwargs,
                emulator_kwargs=emulator_kwargs,
            )
        ]
    )
    eval_env = VecMonitor(eval_env, str(log_dir / "eval_monitor"))

    policy_kwargs = dict(
        features_extractor_class=PokemonFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim,
            screen_cnn_channels=args.screen_cnn_channels,
            visit_cnn_channels=args.visit_cnn_channels,
            vector_mlp_hidden=args.vector_mlp_hidden,
        ),
        net_arch=dict(pi=args.net_arch_pi, vf=args.net_arch_vf),
    )

    if args.resume is not None:
        # nargs='?': bare --resume → "AUTO"; --resume path.zip → path
        resume_arg = None if args.resume in ("", "AUTO") else args.resume
        resume_path = resolve_model_path(resume_arg, prefer_latest=True)
        print(f"Resuming from {resume_path}")
        # A checkpoint saved before the switch to MaskablePPO (plain PPO /
        # MultiInputActorCriticPolicy) will not load here -- the action
        # distribution class changed (MaskableCategoricalDistribution), not
        # just weight shapes. Use --migrate for those (transplants
        # shape-matching weights by name into a fresh MaskablePPO instead of
        # deserializing the old policy class directly).
        model = MaskablePPO.load(str(resume_path), env=train_env, device=device)
        # Override frozen checkpoint hyperparams (e.g. raise entropy after collapse).
        model.ent_coef = float(args.ent_coef)
        print(f"ent_coef override: {model.ent_coef}")
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=device,
            verbose=1,
            seed=args.seed,
        )
        if args.migrate is not None:
            # A checkpoint from before a curriculum goal was added won't
            # load via PPO.load() — its "vector" observation width no
            # longer matches (goal one-hot grew). Build a fresh model
            # against the *current* env/spaces above, then transplant every
            # weight that still matches shape-for-shape, remapping just the
            # goal one-hot columns by name (see ppo/migrate.py).
            migrate_arg = None if args.migrate in ("", "AUTO") else args.migrate
            migrate_path = resolve_model_path(migrate_arg, prefer_latest=True)
            print(f"Migrating weights from {migrate_path}")
            from stable_baselines3.common.save_util import load_from_zip_file

            _, old_params, _ = load_from_zip_file(
                str(migrate_path), device=device, load_data=False
            )
            merged = migrate_state_dict(
                old_params["policy"],
                model.policy.state_dict(),
                GOAL_ORDER,
                GOAL_INDEX,
            )
            model.policy.load_state_dict(merged)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.workers, 1), 1),
        save_path=str(model_dir),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.workers, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    milestone_cb = MilestoneCallback(
        window=args.milestone_window,
        auto_curriculum=args.auto_curriculum,
        start_stage=stage,
        curriculum_mix=args.curriculum_mix,
        eval_env=eval_env,
    )

    callbacks = [checkpoint_cb, eval_cb, milestone_cb]
    if args.heatmap:
        from ppo.callbacks import HeatmapCallback

        callbacks.append(HeatmapCallback(heatmap_queue))
    if args.auto_entropy:
        from ppo.callbacks import EntropyCoefScheduler

        callbacks.append(
            EntropyCoefScheduler(
                ent_coef_min=args.ent_coef_min,
                ent_coef_max=args.ent_coef_max,
                window=args.entropy_window,
                check_every=args.entropy_check_every,
                loop_rate_high=args.loop_rate_high,
                loop_rate_low=args.loop_rate_low,
                increase_factor=args.entropy_increase_factor,
                decay_factor=args.entropy_decay_factor,
                cooldown_checks=args.entropy_cooldown_checks,
                verbose=1,
            )
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            progress_bar=not args.no_progress,
        )
    except KeyboardInterrupt:
        print("\nInterrupted - saving latest model...")
    finally:
        out = model_dir / "ppo_latest.zip"
        atomic_model_save(model, out)
        print(f"Saved: {out}")
        train_env.close()
        eval_env.close()
        if heatmap_proc is not None:
            from utils.PositionHeatmap import stop_heatmap_process

            stop_heatmap_process(heatmap_proc, heatmap_queue)


def print_stage_list() -> None:
    """List every curriculum stage (--stage accepts any of these ids)."""
    from curriculum_config import CURRICULUM, STAGE_ORDER, _save_exists

    name_w = max(len(s) for s in STAGE_ORDER)
    goal_w = max(len(CURRICULUM[s]["goal"]) for s in STAGE_ORDER)
    saves = {
        name: (cfg["checkpoint"] if _save_exists(cfg["checkpoint"]) else "MISSING")
        for name, cfg in ((n, CURRICULUM[n]) for n in STAGE_ORDER)
    }
    save_w = max(len(s) for s in saves.values())
    print(
        f"{'#':>3}  {'stage':<{name_w}}  {'goal':<{goal_w}}  "
        f"{'max_steps':>9}  {'save':<{save_w}}  description"
    )
    for i, name in enumerate(STAGE_ORDER, start=1):
        cfg = CURRICULUM[name]
        print(
            f"{i:>3}  {name:<{name_w}}  {cfg['goal']:<{goal_w}}  "
            f"{cfg['max_steps']:>9}  {saves[name]:<{save_w}}  {cfg['description']}"
        )


def build_argparser(parent=None):
    p = argparse.ArgumentParser(
        description="Pokemon Red PPO training",
        parents=[parent] if parent else [],
    )
    p.add_argument("--workers", "-w", type=int, default=8)
    p.add_argument("--timesteps", "-t", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument(
        "--auto-entropy", action="store_true",
        help="Auto-adjust ent_coef during training based on the rolling "
             "action-loop rate (bumps toward --ent-coef-max on collapse, "
             "decays back toward --ent-coef-min once it clears)",
    )
    p.add_argument("--ent-coef-min", type=float, default=0.01,
                   help="Floor for --auto-entropy (default: 0.01)")
    p.add_argument("--ent-coef-max", type=float, default=0.3,
                   help="Ceiling for --auto-entropy (default: 0.3)")
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None)
    # Must match curriculum_config.DEFAULT_STAGE -- "stage_left_house" used
    # to be here, but that's a dead pre-generic-goal-system id with no
    # wEventFlags equivalent (see _LEGACY_STAGE_ALIASES' docstring), so it
    # silently resolved to an arbitrary, unreachable, checkpoint-less goal.
    p.add_argument("--stage", default="EVENT_GOT_STARTER",
                   help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
    p.add_argument("--list-stages", action="store_true",
                   help="List all curriculum stages (id, goal, max_steps, save) and exit")
    p.add_argument("--goal", default=None,
                   help="Override goal (auto-curriculum still advances stages)")
    p.add_argument("--auto-curriculum", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-advance through STAGE_ORDER when goal success rate is high (default: on)")
    p.add_argument(
        "--save-checkpoints", action=argparse.BooleanOptionalAction, default=False,
        help="On every GOAL_CANDIDATES event/badge newly satisfied, overwrite "
             "saves/<name>/checkpoint.state with the reached state -- whichever "
             "goal an episode clears first, not just the currently-assigned "
             "curriculum goal. Default off for training workers: the eval_env "
             "(deterministic rollouts, always writes) is the intended source "
             "of truth, since a goal reached under exploration noise isn't "
             "necessarily one the policy can reliably repeat. Pass "
             "--save-checkpoints to also let (noisier) training episodes "
             "write -- both writers use atomic replace, so this is safe, just "
             "noisier -- e.g. for faster bootstrapping very early in a run.",
    )
    p.add_argument("--curriculum-mix", type=float, default=0.3,
                   help="Probability of resetting to an earlier curriculum save")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        nargs="?",
        const="AUTO",
        default=None,
        help="Resume from .zip; omit path to auto-pick newest ppo_latest.zip (else best)",
    )
    p.add_argument(
        "--migrate",
        nargs="?",
        const="AUTO",
        default=None,
        help="Like --resume, but for a checkpoint whose 'vector' observation "
             "width no longer matches (e.g. after a new curriculum goal grew "
             "GOAL_ORDER). Builds a fresh model and transplants every "
             "shape-matching weight, remapping the goal one-hot columns by "
             "name (see ppo/migrate.py). New goal(s) start at random init — "
             "expect a short readjustment before training looks normal again.",
    )
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--heatmap", action="store_true",
        help="Open a live position-heatmap window (avg ticks/run per map tile, "
             "rolling window of --heatmap-frames). <-/-> switches map.",
    )
    p.add_argument(
        "--heatmap-frames", type=int, default=300_000,
        help="Rolling window size in frames, pooled across all runs (default: 300000)",
    )

    # PPO / policy-network hyperparameters that used to be hardcoded in
    # MaskablePPO(...)/PokemonFeaturesExtractor construction below.
    model_group = p.add_argument_group("PPO / model hyperparameters")
    model_group.add_argument("--gae-lambda", type=float, default=0.95)
    model_group.add_argument("--clip-range", type=float, default=0.2)
    model_group.add_argument("--vf-coef", type=float, default=0.5)
    model_group.add_argument("--max-grad-norm", type=float, default=0.5)
    model_group.add_argument("--features-dim", type=int, default=256)
    model_group.add_argument(
        "--net-arch-pi", type=_int_list, default=[256, 256], metavar="N,N,...",
        help="Policy-head hidden layer sizes, comma-separated (default: 256,256)",
    )
    model_group.add_argument(
        "--net-arch-vf", type=_int_list, default=[256, 256], metavar="N,N,...",
        help="Value-head hidden layer sizes, comma-separated (default: 256,256)",
    )
    model_group.add_argument(
        "--screen-cnn-channels", type=_int_list, default=[32, 64, 64], metavar="N,N,...",
        help="screen_tiles CNN channel widths, comma-separated (default: 32,64,64)",
    )
    model_group.add_argument(
        "--visit-cnn-channels", type=_int_list, default=[16, 32], metavar="N,N,...",
        help="visit_mask CNN channel widths, comma-separated (default: 16,32)",
    )
    model_group.add_argument(
        "--vector-mlp-hidden", type=_int_list, default=[256, 128], metavar="N,N,...",
        help="Flat-vector MLP hidden layer sizes, comma-separated (default: 256,128)",
    )
    model_group.add_argument(
        "--milestone-window", type=int, default=100,
        help="MilestoneCallback's rolling episode window (default: 100)",
    )

    entropy_group = p.add_argument_group("--auto-entropy scheduler")
    entropy_group.add_argument("--entropy-window", type=int, default=50)
    entropy_group.add_argument("--entropy-check-every", type=int, default=20_000)
    entropy_group.add_argument("--loop-rate-high", type=float, default=0.3)
    entropy_group.add_argument("--loop-rate-low", type=float, default=0.08)
    entropy_group.add_argument("--entropy-increase-factor", type=float, default=1.5)
    entropy_group.add_argument("--entropy-decay-factor", type=float, default=0.9)
    entropy_group.add_argument("--entropy-cooldown-checks", type=int, default=5)

    tunables.add_dataclass_args(p, Data, group_title="reward shaping (Data)")
    tunables.add_dataclass_args(p, Emulator, group_title="emulator timing")
    tunables.add_dataclass_args(
        p, curriculum_config.CurriculumConfig, group_title="curriculum advance"
    )
    return p


def main():
    args = build_argparser().parse_args()
    if args.list_stages:
        print_stage_list()
        return
    run(args)


if __name__ == "__main__":
    main()
