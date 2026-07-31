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


def run(args):
    set_start_method("spawn", force=True)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        resolve_stage_name,
    )
    from env.pokemon_red_env import PokemonRedEnv
    from ppo.callbacks import MilestoneCallback
    from ppo.checkpoints import atomic_model_save, resolve_model_path
    from ppo.features import PokemonFeaturesExtractor

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
    curriculum_saves = get_curriculum_saves(stage)

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
    print(f"Saves:      {curriculum_saves}")
    print(f"Log dir:    {log_dir}")
    print(f"Total steps:{args.timesteps:,}")
    print(f"GUI:        {'ON' if args.gui else 'OFF'}")
    print(f"Auto curr.: {'ON' if args.auto_curriculum else 'OFF'}")
    print("=" * 70)

    def _make(rank: int):
        def _thunk():
            env = PokemonRedEnv(
                save_state=curriculum_saves[-1],
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
                save_state=curriculum_saves[-1],
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                stage=stage,
                curriculum_mix=0.0,
                curriculum_saves=curriculum_saves[-1:],
                render_mode=None,
                auto_advance=args.auto_curriculum,
            )
        ]
    )
    eval_env = VecMonitor(eval_env, str(log_dir / "eval_monitor"))

    policy_kwargs = dict(
        features_extractor_class=PokemonFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.resume is not None:
        # nargs='?': bare --resume → "AUTO"; --resume path.zip → path
        resume_arg = None if args.resume in ("", "AUTO") else args.resume
        resume_path = resolve_model_path(resume_arg, prefer_latest=True)
        print(f"Resuming from {resume_path}")
        model = PPO.load(str(resume_path), env=train_env, device=device)
        # Override frozen checkpoint hyperparams (e.g. raise entropy after collapse).
        model.ent_coef = float(args.ent_coef)
        print(f"ent_coef override: {model.ent_coef}")
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=device,
            verbose=1,
            seed=args.seed,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.workers, 1), 1),
        save_path=str(model_dir),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.workers, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    milestone_cb = MilestoneCallback(
        window=100,
        auto_curriculum=args.auto_curriculum,
        start_stage=stage,
        curriculum_mix=args.curriculum_mix,
        eval_env=eval_env,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_cb, eval_cb, milestone_cb]),
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
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stage", default="stage_left_house",
                   help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
    p.add_argument("--list-stages", action="store_true",
                   help="List all curriculum stages (id, goal, max_steps, save) and exit")
    p.add_argument("--goal", default=None,
                   help="Override goal (auto-curriculum still advances stages)")
    p.add_argument("--auto-curriculum", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-advance through STAGE_ORDER when goal success rate is high (default: on)")
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
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if args.list_stages:
        print_stage_list()
        return
    run(args)


if __name__ == "__main__":
    main()
