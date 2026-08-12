"""Training callbacks: milestone / loop-rate logging + auto curriculum."""
from __future__ import annotations

import json
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

# Cumulative per-goal hit counts (see MilestoneCallback._goal_hit_counts),
# persisted here so it survives resumed training runs and is readable
# outside the training process -- e.g. create_stage_save.py --list shows it
# alongside each goal's checkpoint status. Lives under saves/ (not the
# per-session logs/ dir) because, like saves/<goal>/checkpoint.state, it's
# global "what has this project learned so far" state, not one run's logs.
GOAL_HIT_COUNTS_PATH = Path("saves") / "goal_hit_counts.json"

# Every map_id any episode has ever set foot on this training run (see
# MilestoneCallback._visited_maps), persisted the same way and for the same
# reason as GOAL_HIT_COUNTS_PATH -- fed into curriculum_config.pick_new_goal
# as ``visited_maps`` so the free pick in _try_reassign never hands out a
# goal on a map training hasn't reached yet. Recorded once and reused for
# the rest of the run instead of recomputed live, since it only ever grows.
VISITED_MAPS_PATH = Path("saves") / "visited_maps.json"


class HeatmapCallback(BaseCallback):
    """Forward per-run (map,x,y)->ticks snapshots to the live --heatmap window.

    Envs only populate info["heatmap_positions"] when collect_heatmap=True,
    so this is a no-op cost-wise when the flag is off.
    """

    def __init__(self, queue, verbose: int = 0):
        super().__init__(verbose)
        self.queue = queue

    def _on_step(self) -> bool:
        from utils.PositionHeatmap import push_episode

        for info in self.locals.get("infos", []):
            if not info:
                continue
            positions = info.get("heatmap_positions")
            if positions:
                push_episode(
                    self.queue,
                    positions,
                    info.get("heatmap_directions"),
                    info.get("heatmap_transitions"),
                    info.get("heatmap_rewards"),
                    info.get("heatmap_battle_outcomes"),
                    info.get("heatmap_milestones"),
                    info.get("heatmap_dialogs"),
                    info.get("heatmap_steps") or 0,
                    info.get("stage"),
                    info.get("party_count"),
                    info.get("party_avg_level"),
                )
        return True


class EntropyCoefScheduler(BaseCallback):
    """Bump/decay ``model.ent_coef`` off the rolling action-loop rate.

    Uses the same ``info["loop_flag"]`` signal as ``MilestoneCallback``'s
    ``loop_episode_rate`` (an episode where the agent got stuck repeating a
    short action sequence). When that rate runs hot, raise entropy to break
    the collapse; once it's been quiet for a while, decay back toward
    ``ent_coef_min`` so the policy isn't kept artificially noisy forever.
    """

    def __init__(
        self,
        ent_coef_min: float,
        ent_coef_max: float,
        window: int = 50,
        check_every: int = 20_000,
        loop_rate_high: float = 0.3,
        loop_rate_low: float = 0.08,
        increase_factor: float = 1.5,
        decay_factor: float = 0.9,
        cooldown_checks: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        self.window = window
        self.check_every = check_every
        self.loop_rate_high = loop_rate_high
        self.loop_rate_low = loop_rate_low
        self.increase_factor = increase_factor
        self.decay_factor = decay_factor
        self.cooldown_checks = cooldown_checks

        self._loops: deque[int] = deque(maxlen=window)
        self._ep_loop: list[bool] | None = None
        self._check_count = 0
        self._last_change_check = 0

    def _on_training_start(self) -> None:
        self._ep_loop = [False] * self.training_env.num_envs
        self.model.ent_coef = float(
            np.clip(self.model.ent_coef, self.ent_coef_min, self.ent_coef_max)
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not info:
                continue
            if info.get("loop_flag"):
                self._ep_loop[i] = True
            if i < len(dones) and dones[i]:
                self._loops.append(1 if self._ep_loop[i] else 0)
                self._ep_loop[i] = False

        if len(self._loops) >= max(self.window // 2, 5) and self.n_calls % self.check_every == 0:
            self._check_count += 1
            loop_rate = float(np.mean(self._loops))
            current = float(self.model.ent_coef)

            if loop_rate > self.loop_rate_high:
                new = min(self.ent_coef_max, current * self.increase_factor)
                if new > current:
                    self.model.ent_coef = new
                    self._last_change_check = self._check_count
                    if self.verbose:
                        print(
                            f"[entropy] loop_rate={loop_rate:.2f} > {self.loop_rate_high} "
                            f"-> ent_coef {current:.4f} -> {new:.4f}"
                        )
            elif (
                loop_rate < self.loop_rate_low
                and current > self.ent_coef_min
                and (self._check_count - self._last_change_check) >= self.cooldown_checks
            ):
                new = max(self.ent_coef_min, current * self.decay_factor)
                self.model.ent_coef = new
                self._last_change_check = self._check_count
                if self.verbose:
                    print(
                        f"[entropy] loop_rate={loop_rate:.2f} < {self.loop_rate_low} "
                        f"-> ent_coef {current:.4f} -> {new:.4f}"
                    )

            self.logger.record("pokemon/ent_coef", float(self.model.ent_coef))
            self.logger.record("pokemon/action_loop_rate", loop_rate)

        return True


class MilestoneCallback(BaseCallback):
    """Track episode milestone hit-rate and loop episode rate.

    When ``auto_curriculum`` is True, periodically hands training a fresh,
    randomly-picked goal (see curriculum_config.pick_new_goal) once the
    rolling success rate -- did each recent episode clear *some* goal, not
    a specific one -- clears the threshold. There is no story order to walk:
    whichever goal(s) an episode actually reaches already get their own
    saves/<goal>/checkpoint.state (see
    PokemonRedEnv._save_milestone_checkpoints); this just keeps refreshing
    what new resets aim for instead of grinding the same goal forever.
    """

    def __init__(
        self,
        window: int = 100,
        auto_curriculum: bool = True,
        start_stage: str = "stage_left_house",
        success_threshold: float | None = None,
        min_episodes: int | None = None,
        check_every: int | None = None,
        curriculum_mix: float = 0.3,
        eval_env: VecEnv | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window = window
        self.auto_curriculum = auto_curriculum
        self.stage = start_stage
        self.curriculum_mix = curriculum_mix
        self.eval_env = eval_env

        # Imported lazily-safe defaults from curriculum_config
        from curriculum_config import (
            ADVANCE_CHECK_EVERY,
            ADVANCE_MIN_EPISODES,
            ADVANCE_SUCCESS_THRESHOLD,
        )

        self.success_threshold = (
            ADVANCE_SUCCESS_THRESHOLD if success_threshold is None else success_threshold
        )
        self.min_episodes = ADVANCE_MIN_EPISODES if min_episodes is None else min_episodes
        self.check_every = ADVANCE_CHECK_EVERY if check_every is None else check_every

        self._returns: deque[float] = deque(maxlen=window)
        self._loops: deque[int] = deque(maxlen=window)
        self._loop_causes: deque[frozenset] = deque(maxlen=window)
        self._truncate_causes: deque[frozenset] = deque(maxlen=window)
        self._successes: deque[int] = deque(maxlen=window)
        self._badges: deque[int] = deque(maxlen=window)
        self._goals_live: deque[int] = deque(maxlen=window)
        self._goals_peak: deque[int] = deque(maxlen=window)
        self._regressions: deque[int] = deque(maxlen=window)
        # Per-episode mean of Data.map_id_visit_counts (avg dwell-steps per
        # map touched) — how close episodes run to map_dwell_budget on
        # average, see PokemonRedEnv._info's map_dwell_avg.
        self._map_dwell_avgs: deque[float] = deque(maxlen=window)
        # Cumulative (whole run, not windowed) count of completed episodes
        # that touched each goal at least once -- see info["milestones_hit"]
        # below. Doubles as a per-goal mastery signal: fed into
        # pick_new_goal's ``weights`` in _try_reassign so the next base goal
        # is biased toward whatever the policy has practiced least, letting
        # a de facto goal order emerge from what it's actually learned
        # rather than a fixed sequence.
        self._goal_hit_counts: Counter[str] = Counter()
        # Every map_id any episode has touched this run, whole-run cumulative
        # like _goal_hit_counts -- fed into pick_new_goal's ``visited_maps``
        # in _try_reassign so the free pick only ever lands on a goal whose
        # home map training has actually reached, not one still unseen.
        self._visited_maps: set[int] = set()
        # Every goal_success hit this window, whichever goal it was --
        # cleared_stage no longer needs to equal self.stage: goals clear
        # order-free now (see PokemonRedEnv._advance_after_goal), there is
        # no single "current goal" to match against. Reset whenever the
        # base goal is reassigned (see _try_reassign).
        self._goal_hit_count = 0
        self._goal_hit_reward_sum = 0.0
        self._ep_loop = None
        self._ep_loop_causes = None
        self._ep_milestones = None
        self._ep_goal_hit = None
        self._ep_regressed = None
        self._ep_goals_peak = None
        self._ep_map_dwell_avg = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_loop = [False] * n
        self._ep_loop_causes = [set() for _ in range(n)]
        self._ep_milestones = [set() for _ in range(n)]
        self._ep_goal_hit = [False] * n
        self._ep_regressed = [False] * n
        self._ep_goals_peak = [0] * n
        self._ep_map_dwell_avg = [0.0] * n
        self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
        self._reset_goal_hits()
        self._load_goal_hit_counts()
        self._load_visited_maps()

    def _on_training_end(self) -> None:
        self._save_goal_hit_counts()
        self._save_visited_maps()

    def _load_goal_hit_counts(self) -> None:
        """Seed cumulative per-goal counts from a prior run, if any, so
        resuming training doesn't reset the saturation picture to zero."""
        if not GOAL_HIT_COUNTS_PATH.is_file():
            return
        try:
            with open(GOAL_HIT_COUNTS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self._goal_hit_counts = Counter({k: int(v) for k, v in data.items()})

    def _save_goal_hit_counts(self) -> None:
        GOAL_HIT_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = GOAL_HIT_COUNTS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(self._goal_hit_counts), f, indent=2, sort_keys=True)
        os.replace(tmp, GOAL_HIT_COUNTS_PATH)

    def _load_visited_maps(self) -> None:
        """Seed the cumulative visited-map set from a prior run, if any --
        same resume story as _load_goal_hit_counts."""
        if not VISITED_MAPS_PATH.is_file():
            return
        try:
            with open(VISITED_MAPS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self._visited_maps = {int(m) for m in data}

    def _save_visited_maps(self) -> None:
        VISITED_MAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = VISITED_MAPS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(self._visited_maps), f)
        os.replace(tmp, VISITED_MAPS_PATH)

    def _reset_goal_hits(self) -> None:
        self._goal_hit_count = 0
        self._goal_hit_reward_sum = 0.0
        self.logger.record("pokemon/goal_hits", 0.0)
        self.logger.record("pokemon/goal_hit_avg_reward", 0.0)

    def _stage_idx(self) -> int:
        """STAGE_ORDER position of self.stage -- informational TensorBoard
        gauge only, not used to pick what comes next (see pick_new_goal)."""
        from curriculum_config import stage_index

        return float(stage_index(self.stage))

    def _apply_stage_to_env(self, env: VecEnv, stage: str) -> None:
        from curriculum_config import get_curriculum_saves, get_goal_for_stage, get_stage_max_steps

        goal = get_goal_for_stage(stage)
        max_steps = get_stage_max_steps(stage)
        saves = get_curriculum_saves(stage)
        env.env_method(
            "set_curriculum",
            goal=goal,
            max_steps=max_steps,
            save_state=saves[-1],
            curriculum_saves=saves,
            curriculum_mix=self.curriculum_mix,
            stage=stage,
            clear_visits=True,
        )

    def _try_reassign(self) -> None:
        """Periodically hand training a fresh, order-free goal pick.

        No demote counterpart anymore: there's no single "current stage" to
        fall back from, and a stall just means the next random pick (which
        can land on an easier goal as readily as a harder one) gets tried
        once enough episodes confirm the current one is going nowhere.
        """
        if not self.auto_curriculum:
            return
        if len(self._successes) < self.min_episodes:
            return

        rate = float(np.mean(self._successes))
        self.logger.record("pokemon/goal_success_rate", rate)
        if rate < self.success_threshold:
            return

        from curriculum_config import get_goal_for_stage, get_stage_max_steps, pick_new_goal

        # Inverse-hit-count weighting: a goal no completed episode has ever
        # touched (count 0) gets weight 1.0, same as one seen exactly once,
        # so brand-new discoveries aren't favored purely for being unseen --
        # weight only decays as a goal accumulates *practiced* hits.
        weights = {g: 1.0 / (1.0 + c) for g, c in self._goal_hit_counts.items()}
        nxt = pick_new_goal(weights=weights, visited_maps=self._visited_maps)
        if nxt is None or nxt == self.stage:
            return

        old = self.stage
        self.stage = nxt
        self._apply_stage_to_env(self.training_env, nxt)
        if self.eval_env is not None:
            self._apply_stage_to_env(self.eval_env, nxt)

        # Fresh window so we don't immediately leap again on stale successes.
        self._successes.clear()
        self._returns.clear()
        self._reset_goal_hits()

        goal = get_goal_for_stage(nxt)
        max_steps = get_stage_max_steps(nxt)
        nxt_hits = self._goal_hit_counts.get(nxt, 0)
        msg = (
            f"[curriculum] {old} -> {nxt} (order-free pick, "
            f"practiced {nxt_hits}x so far) "
            f"(goal={goal}, max_steps={max_steps}, "
            f"success_rate={rate:.2f} >= {self.success_threshold:.2f})"
        )
        # ASCII only: Windows cp1250 + PowerShell redirect crashes on arrows.
        print(f"\n{msg}\n")
        self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
        self.logger.record("pokemon/goal_success_rate", 0.0)
        self.logger.record("pokemon/goal_practice_count", float(nxt_hits))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if not info:
                continue
            map_id = info.get("map_id")
            if map_id is not None:
                self._visited_maps.add(int(map_id))
            if info.get("loop_flag"):
                self._ep_loop[i] = True
            for c in info.get("loop_causes", []) or []:
                self._ep_loop_causes[i].add(c)
            for m in info.get("milestones_hit", []) or []:
                self._ep_milestones[i].add(m)
            cur = info.get("milestone")
            if cur and cur != "start":
                self._ep_milestones[i].add(cur)

            done = bool(dones[i]) if i < len(dones) else False
            cleared_stage = info.get("cleared_stage")
            if info.get("goal_success") or cleared_stage:
                self._ep_goal_hit[i] = True
            # Every hit counts, whichever goal it was -- envs advance
            # order-free now (see PokemonRedEnv._advance_after_goal), there
            # is no single "current goal" to match against anymore.
            if info.get("goal_success"):
                self._goal_hit_count += 1
                if i < len(rewards):
                    self._goal_hit_reward_sum += float(rewards[i])
            # "hard" = a payout was actually clawed back (goal not yet
            # curriculum-cleared). Plain "goals_regressed" also fires on
            # every ordinary "left a map already cleared en route to the
            # next stage" step, which is expected progress, not backsliding
            # — using it here made goal_regression_rate read close to 1.0
            # for essentially any episode that advanced past one stage.
            if info.get("goals_regressed_hard"):
                self._ep_regressed[i] = True
            peak = int(info.get("goals_peak_count", 0) or 0)
            if peak > self._ep_goals_peak[i]:
                self._ep_goals_peak[i] = peak
            live = int(info.get("goals_live_count", 0) or 0)
            # Cumulative for the episode so far — just keep the latest
            # reading, appended to the rolling window at done below.
            self._ep_map_dwell_avg[i] = float(info.get("map_dwell_avg", 0.0) or 0.0)

            if done:
                self._loops.append(1 if self._ep_loop[i] else 0)
                self._loop_causes.append(frozenset(self._ep_loop_causes[i]))
                self._truncate_causes.append(
                    frozenset(info.get("truncate_causes", []) or ())
                )
                # auto_advance clears terminated on the goal step, so rely on
                # _ep_goal_hit (set when goal_success was True earlier).
                success = bool(
                    self._ep_goal_hit[i]
                    or info.get("goal_success")
                    or info.get("cleared_stage")
                    or info.get("terminated", False)
                )
                self._successes.append(1 if success else 0)
                self._badges.append(int(info.get("badges", 0) or 0))
                self._goals_live.append(live)
                self._goals_peak.append(self._ep_goals_peak[i])
                self._regressions.append(1 if self._ep_regressed[i] else 0)
                self._map_dwell_avgs.append(self._ep_map_dwell_avg[i])
                for name in self._ep_milestones[i]:
                    self._goal_hit_counts[name] += 1

                if "episode" in info:
                    self._returns.append(float(info["episode"]["r"]))

                self._ep_loop[i] = False
                self._ep_loop_causes[i] = set()
                self._ep_milestones[i] = set()
                self._ep_goal_hit[i] = False
                self._ep_regressed[i] = False
                self._ep_goals_peak[i] = 0
                self._ep_map_dwell_avg[i] = 0.0

        if len(self._loops) >= 10 and self.n_calls % self.check_every == 0:
            loop_rate = float(np.mean(self._loops))
            self.logger.record("pokemon/loop_episode_rate", loop_rate)
            if self._loop_causes:
                from pokemon.Data import LOOP_CAUSES

                n_eps = len(self._loop_causes)
                for cause in LOOP_CAUSES:
                    rate = sum(1 for s in self._loop_causes if cause in s) / n_eps
                    self.logger.record(f"pokemon/loop_cause_{cause}", rate)
            if self._truncate_causes:
                from pokemon.Data import TRUNCATE_CAUSES

                n_eps = len(self._truncate_causes)
                for cause in TRUNCATE_CAUSES:
                    rate = sum(1 for s in self._truncate_causes if cause in s) / n_eps
                    self.logger.record(f"pokemon/truncate_cause_{cause}", rate)
            if self._successes:
                self.logger.record(
                    "pokemon/goal_success_rate", float(np.mean(self._successes))
                )
            if self._badges:
                self.logger.record(
                    "pokemon/badges_mean", float(np.mean(self._badges))
                )
            if self._goals_live:
                self.logger.record(
                    "pokemon/goals_live_mean", float(np.mean(self._goals_live))
                )
            if self._goals_peak:
                self.logger.record(
                    "pokemon/goals_peak_mean", float(np.mean(self._goals_peak))
                )
            if self._map_dwell_avgs:
                self.logger.record(
                    "pokemon/map_dwell_avg_count", float(np.mean(self._map_dwell_avgs))
                )
            if self._regressions:
                self.logger.record(
                    "pokemon/goal_regression_rate", float(np.mean(self._regressions))
                )
            if self._returns:
                self.logger.record(
                    "pokemon/ep_return_mean", float(np.mean(self._returns))
                )
            self.logger.record(
                "pokemon/goal_hits", float(self._goal_hit_count)
            )
            if self._goal_hit_count:
                self.logger.record(
                    "pokemon/goal_hit_avg_reward",
                    self._goal_hit_reward_sum / self._goal_hit_count,
                )
            self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
            if self._goal_hit_counts:
                from curriculum_config import N_GOALS

                counts = list(self._goal_hit_counts.values())
                self.logger.record(
                    "pokemon/goal_saturation_coverage",
                    len(self._goal_hit_counts) / N_GOALS,
                )
                self.logger.record(
                    "pokemon/goal_saturation_mean", float(np.mean(counts))
                )
                self.logger.record(
                    "pokemon/goal_saturation_max", float(np.max(counts))
                )
            if self._visited_maps:
                self.logger.record(
                    "pokemon/maps_visited_count", float(len(self._visited_maps))
                )
            self._save_goal_hit_counts()
            self._save_visited_maps()
            self._try_reassign()

        return True
