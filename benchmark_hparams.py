#!/usr/bin/env python3
"""Hyperparameter benchmark: races curriculum-milestone candidates to build a
target chain, tuning PPO hyperparameters per chain segment via Optuna.

Design (see memory/hparam_benchmark_design.md from the planning session):

  Starting from a "frontier" checkpoint (initially "start"), candidates for
  the next chain segment are discovered with a LIVE reachability query --
  pokemon.event_compass.nearest_unlockable_event, the same map-connection +
  event/badge-prerequisite BFS Data.compass_progress feeds the model --
  called repeatedly (each call excluding what it just found) from the
  frontier's own checkpoint until --max-candidates GOAL_CANDIDATES-mapped
  goals are collected. See _discover_candidates.

  NOTE: an earlier version of this script used curriculum_config.goal_status
  ("practicable" = EVENT_GRAPH parents satisfied) for candidate discovery.
  A smoke test caught this early: EVENT_GRAPH's parent/child edges mostly
  point OUTSIDE GOAL_CANDIDATES (NPC dialogue flags, etc.), so the parents-
  satisfied check came back vacuously true for ~449 of ~450 GOAL_CANDIDATES
  from "start" alone -- no geography signal at all, exactly the trap
  pick_new_goal's docstring already warns about ("a goal three towns away...
  eligible from step one"), just far worse in scale than expected. Switched
  to event_compass, the only real reachability source in this repo.

  Each candidate gets its OWN full Optuna sweep (TPE + MedianPruner) racing
  to reach a rolling-window success rate >= --success-threshold in the fewest
  env steps ("saturation"). The candidate whose best config saturates
  fastest -- with a bonus for incidentally saturating other goals along the
  way -- wins the step; its checkpoint (written by a short deterministic
  confirmation rollout) becomes the new frontier, and the walk repeats.

  This deliberately races every candidate with its own full sweep rather
  than a cheap shared baseline config -- more expensive, but each candidate
  is judged under its own best hyperparameters rather than ones tuned for
  someone else's segment. See the three agreed cost mitigations below.

Cost mitigations (agreed during planning, keep this tractable):
  - Smaller --n-trials per sweep + MedianPruner does most of the work: a bad
    trial is cut early rather than running to --timesteps-cap.
  - The expensive multi-seed confirmation pass (--n-confirm x
    --confirm-seeds) only runs once, on the FINAL winner at the end of the
    whole walk -- not after every step.
  - Losing candidates keep their Optuna study (SQLite under --storage-dir)
    across script runs -- study.optimize resumes instead of restarting if
    that goal comes up again on a later frontier.

NOT implemented in this version (left for later, see the design memory):
  parallel candidate racing (splitting --workers across simultaneously
  racing candidates), and a separate reward-shaping (Data.py) sweep pass.

Prerequisite: run benchmark_workers.py first and pass its result as
--workers -- worker count must stay IDENTICAL across every trial/candidate,
or step-based Optuna pruning comparisons stop meaning the same wall-clock
cost.

Usage:
  pip install optuna
  python benchmark_hparams.py --depth 3 --n-trials 40 --workers 8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from multiprocessing import set_start_method

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Saturation callback: tracks the same noisy-training-episode rolling-window
# success rate curriculum_config/MilestoneCallback already trusts for live
# curriculum-advance decisions (see ADVANCE_SUCCESS_THRESHOLD) -- reused here
# as the "has this segment saturated" trigger, plus Optuna trial.report/
# should_prune wiring and env-step-scoped bonus-goal tracking.
# ---------------------------------------------------------------------------
class SaturationCallback(BaseCallback):
    def __init__(
        self,
        threshold: float,
        min_attempts: int,
        window: int,
        check_every: int,
        timesteps_cap: int,
        trial: "optuna.Trial | None" = None,
        mastered_before: set[str] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.threshold = threshold
        self.min_attempts = min_attempts
        self.check_every = max(check_every, 1)
        self.timesteps_cap = timesteps_cap
        self.trial = trial
        self.mastered_before = set(mastered_before or ())

        self._successes: deque[int] = deque(maxlen=window)
        self._bonus_goals: set[str] = set()
        self.saturated_at: int | None = None

    def _on_training_start(self) -> None:
        self._ep_hit = [False] * self.training_env.num_envs

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not info:
                continue
            for m in info.get("milestones_hit", []) or []:
                if m not in self.mastered_before:
                    self._bonus_goals.add(m)
            if info.get("goal_success") or info.get("cleared_stage"):
                self._ep_hit[i] = True

            if i < len(dones) and dones[i]:
                success = bool(self._ep_hit[i] or info.get("terminated", False))
                self._successes.append(1 if success else 0)
                self._ep_hit[i] = False

        if self.n_calls % self.check_every == 0 and len(self._successes) >= self.min_attempts:
            rate = float(np.mean(self._successes))
            if self.trial is not None:
                self.trial.report(rate, self.num_timesteps)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            if rate >= self.threshold:
                self.saturated_at = self.num_timesteps
                return False

        return self.num_timesteps < self.timesteps_cap


# ---------------------------------------------------------------------------
# Search space (warstwa A + B from the plan -- PPO core + curriculum_mix;
# network architecture and Data.py reward-shaping stay fixed at their
# train_ppo.py defaults, deferred to a later, narrower sweep pass).
# ---------------------------------------------------------------------------
def _suggest_hparams(trial: optuna.Trial, workers: int) -> dict:
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    buffer_size = n_steps * workers
    batch_candidates = [b for b in (64, 128, 256, 512, 1024, 2048) if buffer_size % b == 0]
    if not batch_candidates:
        batch_candidates = [buffer_size]
    batch_size = trial.suggest_categorical("batch_size", batch_candidates)
    return dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        ent_coef=trial.suggest_float("ent_coef", 1e-3, 0.3, log=True),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=trial.suggest_int("n_epochs", 2, 10),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        curriculum_mix=trial.suggest_float("curriculum_mix", 0.0, 0.5),
    )


def _policy_kwargs():
    from ppo.features import PokemonFeaturesExtractor

    return dict(
        features_extractor_class=PokemonFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            screen_cnn_channels=[32, 64, 64],
            visit_cnn_channels=[16, 32],
            vector_mlp_hidden=[256, 128],
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )


def run_trial(
    trial: "optuna.Trial | None",
    goal: str,
    frontier: str,
    hp: dict,
    args: argparse.Namespace,
    device: str,
    mastered_before: set[str],
) -> dict:
    """Train one config from ``frontier`` toward ``goal`` until it saturates
    or hits --timesteps-cap. ``trial=None`` skips Optuna reporting/pruning
    (used for the post-race retrain-to-confirm and the final confirmation
    pass, neither of which is part of a live sweep)."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from curriculum_config import get_stage_max_steps, stage_for_goal
    from env.pokemon_red_env import PokemonRedEnv

    stage = stage_for_goal(goal)
    max_steps = args.max_steps or get_stage_max_steps(stage)
    # Keep "start" in the mix pool alongside the frontier checkpoint, same
    # regularization rationale as curriculum_config.get_curriculum_saves --
    # otherwise curriculum_mix has nothing earlier than frontier to ever
    # reset to once frontier != "start".
    saves = ["start", frontier] if frontier != "start" else ["start"]

    def _make(rank: int):
        def _thunk():
            env = PokemonRedEnv(
                save_state=frontier,
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                stage=stage,
                curriculum_mix=hp["curriculum_mix"],
                curriculum_saves=saves,
                render_mode=None,
                auto_advance=False,
                worker_rank=rank,
                n_workers=args.workers,
                collect_heatmap=False,
                # Sweep trials never write production saves/<goal>/ -- only
                # the post-race confirmation rollout does (see
                # _confirm_checkpoint), same "noisy exploration isn't a
                # trusted writer" split train_ppo.py already uses.
                save_checkpoints=False,
            )
            env.reset(seed=args.seed + rank)
            return env

        return _thunk

    if args.workers <= 1:
        train_env = DummyVecEnv([_make(0)])
    else:
        train_env = SubprocVecEnv([_make(i) for i in range(args.workers)], start_method="spawn")
    train_env = VecMonitor(train_env)

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=hp["lr"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        policy_kwargs=_policy_kwargs(),
        device=device,
        verbose=0,
        seed=args.seed,
    )

    check_every = max(args.eval_freq // max(args.workers, 1), 1)
    cb = SaturationCallback(
        threshold=args.success_threshold,
        min_attempts=args.min_attempts,
        window=args.min_attempts,
        check_every=check_every,
        timesteps_cap=args.timesteps_cap,
        trial=trial,
        mastered_before=mastered_before,
    )
    try:
        model.learn(total_timesteps=args.timesteps_cap, callback=cb, progress_bar=False)
    finally:
        train_env.close()

    return dict(
        saturated=cb.saturated_at is not None,
        steps=cb.saturated_at if cb.saturated_at is not None else args.timesteps_cap,
        bonus_goals=cb._bonus_goals,
        model=model,
    )


def objective(
    trial: optuna.Trial,
    goal: str,
    frontier: str,
    args: argparse.Namespace,
    device: str,
    mastered_before: set[str],
) -> float:
    hp = _suggest_hparams(trial, args.workers)
    result = run_trial(trial, goal, frontier, hp, args, device, mastered_before)
    trial.set_user_attr("bonus_goals", len(result["bonus_goals"]))
    trial.set_user_attr("saturated", result["saturated"])
    if not result["saturated"]:
        # Penalized rather than inf so TPE can still tell "close" from
        # "hopeless" among failed trials instead of treating them all alike.
        return float(args.timesteps_cap) * 1.5
    return float(result["steps"])


def _confirm_checkpoint(
    model,
    goal: str,
    frontier: str,
    max_steps: int,
    frame_skip: int,
    n_episodes: int,
    seed: int,
) -> bool:
    """Deterministic rollout(s) from ``frontier`` toward ``goal``, writing
    saves/<goal>/checkpoint.state on the first success (save_checkpoints=
    True) -- the frontier can only advance to a goal with a real emulator
    checkpoint behind it, and sweep trials themselves never write one (see
    run_trial's save_checkpoints=False)."""
    from curriculum_config import stage_for_goal
    from env.pokemon_red_env import PokemonRedEnv

    env = PokemonRedEnv(
        save_state=frontier,
        max_steps=max_steps,
        frame_skip=frame_skip,
        goal=goal,
        stage=stage_for_goal(goal),
        curriculum_mix=0.0,
        curriculum_saves=[frontier],
        auto_advance=False,
        save_checkpoints=True,
    )
    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = truncated = False
            while not (done or truncated):
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, _reward, done, truncated, info = env.step(int(action))
            if info.get("goal_success") or info.get("cleared_stage"):
                return True
        return False
    finally:
        env.close()


def _discover_candidates(
    frontier: str, k: int, max_hops: int, seed: int
) -> list[str]:
    """Live BFS reachability from ``frontier``'s own checkpoint via
    pokemon.event_compass.nearest_unlockable_event (map connections +
    event/badge prerequisites) -- see the module docstring for why this
    replaced an earlier curriculum_config.goal_status-based filter that
    turned out to carry no real geography signal.

    Loads a throwaway PokemonRedEnv at ``frontier`` just to read its live
    position/memory (same mechanism Data.compass_progress uses every
    training step) -- one env construction per frontier STEP, not per
    trial, so the extra cost is negligible next to the sweep itself.
    Repeatedly calls nearest_unlockable_event, excluding each found event
    name, until ``k`` GOAL_CANDIDATES-mapped (and not yet globally
    checkpointed) goals are collected or nothing further is reachable
    within ``max_hops``. A beat-gym-leader event also yields its anchored
    badge (see curriculum_config._BADGE_ANCHOR_EVENT) as a second candidate,
    since both are won in the same in-game moment.
    """
    from curriculum_config import GOAL_CANDIDATES, _BADGE_ANCHOR_EVENT, _save_exists
    from env.pokemon_red_env import PokemonRedEnv
    from pokemon import event_compass

    badge_for_anchor: dict[str, str] = {}
    for badge, anchor in _BADGE_ANCHOR_EVENT.items():
        badge_for_anchor.setdefault(anchor, badge)

    env = PokemonRedEnv(save_state=frontier, auto_advance=False, save_checkpoints=False)
    try:
        env.reset(seed=seed)
        pos = env.emu.data.get_position()
        memory = env.emu.pyboy.memory

        found: list[str] = []
        excluded: set[str] = set()
        # Safety cap on BFS calls -- most of event_triggers' ~386 events
        # don't map onto a tracked GOAL_CANDIDATE (NPC dialogue-only flags
        # etc.), so collecting k real candidates can take well more than k
        # calls.
        for _ in range(max(k * 20, 40)):
            if len(found) >= k:
                break
            result = event_compass.nearest_unlockable_event(
                pos, memory, max_hops=max_hops, excluded_events=frozenset(excluded)
            )
            if result is None:
                break
            _dx, _dy, _dist, event_name = result
            excluded.add(event_name)
            if event_name in GOAL_CANDIDATES and not _save_exists(event_name) and event_name not in found:
                found.append(event_name)
            badge = badge_for_anchor.get(event_name)
            if badge and not _save_exists(badge) and badge not in found:
                found.append(badge)
        return found
    finally:
        env.close()


def _study_for(goal: str, args: argparse.Namespace) -> optuna.Study:
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(storage_dir / f'{goal}.db').as_posix()}"
    return optuna.create_study(
        study_name=goal,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )


@dataclass
class ChainStep:
    frontier_from: str
    goal: str
    steps: float
    bonus_goals: int
    params: dict = field(default_factory=dict)


def race_step(frontier: str, args: argparse.Namespace, device: str) -> ChainStep | None:
    """One frontier-walk step: give every event_compass-reachable candidate
    its own full Optuna sweep, pick the winner by steps-to-saturation
    (credited for incidental extra goals reached), retrain + confirm its
    checkpoint."""
    from curriculum_config import STAGE_ORDER, _save_exists

    candidates = _discover_candidates(frontier, args.max_candidates, args.compass_max_hops, args.seed)
    if not candidates:
        print("  No event_compass-reachable candidates from this frontier -- stopping the walk.")
        return None

    # Fixed for the whole step so every candidate's bonus-goal credit is
    # judged against the same baseline, not one that drifts as earlier
    # candidates in this same step happen to write new checkpoints.
    mastered_before = {g for g in STAGE_ORDER if _save_exists(g)}

    print(f"\n=== Frontier: {frontier} -- racing {len(candidates)} candidate(s): {candidates} ===")

    best_goal: str | None = None
    best_score = float("inf")
    best_value = 0.0
    best_bonus = 0
    for cand in candidates:
        study = _study_for(cand, args)
        n_done = sum(1 for t in study.trials if t.state.is_finished())
        n_remaining = max(args.n_trials - n_done, 0)
        if n_remaining:
            print(f"  [{cand}] running {n_remaining} trial(s) ({n_done} already recorded)...")
            study.optimize(
                lambda t: objective(t, cand, frontier, args, device, mastered_before),
                n_trials=n_remaining,
            )
        if not study.best_trials:
            print(f"  [{cand}] no trial saturated within --timesteps-cap -- skipping")
            continue
        bonus = int(study.best_trial.user_attrs.get("bonus_goals", 0))
        score = study.best_value / (1 + bonus)
        print(f"  [{cand}] best={study.best_value:,.0f} steps  bonus_goals={bonus}  score={score:,.0f}")
        if score < best_score:
            best_score, best_goal, best_value, best_bonus = score, cand, study.best_value, bonus

    if best_goal is None:
        print("  No candidate saturated at all -- stopping the walk here.")
        return None

    print(f"  -> winner: {best_goal} ({best_value:,.0f} steps, score={best_score:,.0f})")
    best_params = _study_for(best_goal, args).best_params

    print("  Retraining winner with its best config to confirm + write its checkpoint...")
    from curriculum_config import get_stage_max_steps, stage_for_goal

    result = run_trial(None, best_goal, frontier, best_params, args, device, mastered_before=set())
    max_steps = args.max_steps or get_stage_max_steps(stage_for_goal(best_goal))
    confirmed = _confirm_checkpoint(
        result["model"], best_goal, frontier, max_steps, args.frame_skip,
        args.confirm_episodes, args.seed,
    )
    if not confirmed:
        print(
            f"  WARNING: could not confirm a deterministic checkpoint for {best_goal} -- "
            f"the frontier can't advance without a real saves/{best_goal}/checkpoint.state. "
            f"Stopping the walk here."
        )
        return None

    return ChainStep(frontier, best_goal, best_value, best_bonus, best_params)


def _confirm_top_configs(step: ChainStep, args: argparse.Namespace, device: str) -> None:
    """Multi-seed confirmation pass -- run once, only on the final chain
    winner, per the agreed cost mitigation (not per frontier step)."""
    study = _study_for(step.goal, args)
    finished = [t for t in study.trials if t.value is not None]
    finished.sort(key=lambda t: t.value)
    top = finished[: args.n_confirm]

    for rank, t in enumerate(top, 1):
        steps_seen = []
        for seed_offset in range(args.confirm_seeds):
            seeded_args = argparse.Namespace(**vars(args))
            seeded_args.seed = args.seed + 1000 * rank + seed_offset
            result = run_trial(
                None, step.goal, step.frontier_from, t.params, seeded_args, device,
                mastered_before=set(),
            )
            steps_seen.append(result["steps"] if result["saturated"] else None)
        ok = [s for s in steps_seen if s is not None]
        mean = sum(ok) / len(ok) if ok else float("nan")
        print(
            f"  #{rank} trial {t.number} (sweep value={t.value:,.0f}): "
            f"{len(ok)}/{args.confirm_seeds} seeds saturated, mean={mean:,.0f} steps -- {t.params}"
        )


def run(args: argparse.Namespace) -> None:
    set_start_method("spawn", force=True)

    import torch

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print("=" * 70)
    print("  Pokemon Red AI - Hyperparameter Benchmark (Optuna frontier walk)")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Workers:     {args.workers}  (fixed across every trial -- see benchmark_workers.py)")
    print(f"Depth:       {args.depth}")
    print(f"Trials/step: {args.n_trials}")
    print(f"Steps cap:   {args.timesteps_cap:,}")
    print(f"Threshold:   {args.success_threshold:.0%} over >= {args.min_attempts} episodes")
    print(f"Start goal:  {args.start_goal}")
    print(f"Storage:     {args.storage_dir}")
    print("=" * 70)

    frontier = args.start_goal
    chain: list[ChainStep] = []
    for _ in range(args.depth):
        step = race_step(frontier, args, device)
        if step is None:
            break
        chain.append(step)
        frontier = step.goal

    print("\n" + "=" * 70)
    print("  Chain result")
    print("=" * 70)
    if not chain:
        print("  (empty -- no candidate ever saturated)")
    for i, s in enumerate(chain, 1):
        print(f"{i}. {s.frontier_from} -> {s.goal}: {s.steps:,.0f} steps (+{s.bonus_goals} bonus goal(s))")
        print(f"   params: {s.params}")

    report_path = Path("logs") / f"benchmark_hparams_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(s) for s in chain], f, indent=2)
    print(f"\nSaved: {report_path}")

    if chain and args.n_confirm > 0:
        print("\n" + "=" * 70)
        print(
            f"  Confirmation pass: top {args.n_confirm} config(s) x {args.confirm_seeds} seed(s) "
            f"for the final winner ({chain[-1].goal})"
        )
        print("=" * 70)
        _confirm_top_configs(chain[-1], args, device)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Race curriculum-milestone candidates with per-candidate Optuna "
        "PPO-hyperparameter sweeps to build a tuned target chain."
    )
    p.add_argument("--depth", type=int, default=3, help="Frontier-walk steps to take (default: 3)")
    p.add_argument("--n-trials", type=int, default=40, help="Optuna trials per candidate sweep (default: 40)")
    p.add_argument("--timesteps-cap", type=int, default=400_000,
                    help="Hard env-step budget per trial (default: 400000)")
    p.add_argument("--success-threshold", type=float, default=0.9,
                    help="Rolling-window success rate counted as 'saturated' (default: 0.9)")
    p.add_argument("--min-attempts", type=int, default=15,
                    help="Rolling-window size / min episodes before the rate is trusted (default: 15)")
    p.add_argument("--eval-freq", type=int, default=20_000,
                    help="Env-step interval between saturation checks / Optuna reports (default: 20000)")
    p.add_argument("--workers", type=int, default=8,
                    help="Fixed worker count for every trial -- see benchmark_workers.py (default: 8)")
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None,
                    help="Override per-episode step cap (default: curriculum_config's per-goal default)")
    p.add_argument("--start-goal", default="start", help="Initial frontier checkpoint (default: start)")
    p.add_argument("--max-candidates", type=int, default=6,
                    help="Max event_compass-reachable candidates raced per frontier step (default: 6)")
    p.add_argument("--compass-max-hops", type=int, default=300,
                    help="BFS hop budget for event_compass candidate discovery (default: 300)")
    p.add_argument("--n-confirm", type=int, default=3,
                    help="Top-N configs of the final winner to re-run multi-seed (default: 3, 0 to skip)")
    p.add_argument("--confirm-seeds", type=int, default=3,
                    help="Seeds per confirmed config (default: 3)")
    p.add_argument("--confirm-episodes", type=int, default=3,
                    help="Deterministic episodes per post-race checkpoint confirmation (default: 3)")
    p.add_argument("--storage-dir", default="logs/optuna",
                    help="Dir for per-goal Optuna SQLite studies (default: logs/optuna)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
