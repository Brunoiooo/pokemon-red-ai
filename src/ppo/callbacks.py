"""Training callbacks: milestone / loop-rate logging + auto curriculum."""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


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

    When ``auto_curriculum`` is True, advances stage (goal / max_steps / saves)
    once the rolling success rate on the *current* goal exceeds the threshold.
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
            DEMOTE_STALL_CHECKS,
            DEMOTE_SUCCESS_CEILING,
        )

        self.success_threshold = (
            ADVANCE_SUCCESS_THRESHOLD if success_threshold is None else success_threshold
        )
        self.min_episodes = ADVANCE_MIN_EPISODES if min_episodes is None else min_episodes
        self.check_every = ADVANCE_CHECK_EVERY if check_every is None else check_every
        self.demote_stall_checks = DEMOTE_STALL_CHECKS
        self.demote_success_ceiling = DEMOTE_SUCCESS_CEILING
        self._stall_checks = 0

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
        # Hits on the *currently set* goal only (cleared_stage == self.stage).
        # Reset whenever the stage changes, forward (advance) or back (demote).
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

    def _reset_goal_hits(self) -> None:
        self._goal_hit_count = 0
        self._goal_hit_reward_sum = 0.0
        self.logger.record("pokemon/goal_hits_current_stage", 0.0)
        self.logger.record("pokemon/goal_hit_avg_reward", 0.0)

    def _stage_idx(self) -> int:
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

    def _try_advance(self) -> None:
        if not self.auto_curriculum:
            return
        if len(self._successes) < self.min_episodes:
            return

        rate = float(np.mean(self._successes))
        self.logger.record("pokemon/goal_success_rate", rate)
        if rate < self.success_threshold:
            self._try_demote(rate)
            return

        self._stall_checks = 0
        from curriculum_config import get_goal_for_stage, get_stage_max_steps, next_stage

        nxt = next_stage(self.stage)
        if nxt is None:
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
        msg = (
            f"[curriculum] Advanced {old} -> {nxt} "
            f"(goal={goal}, max_steps={max_steps}, "
            f"success_rate={rate:.2f} >= {self.success_threshold:.2f})"
        )
        # ASCII only: Windows cp1250 + PowerShell redirect crashes on arrows.
        print(f"\n{msg}\n")
        self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
        self.logger.record("pokemon/goal_success_rate", 0.0)

    def _try_demote(self, rate: float) -> None:
        """Step back one stage after a long, near-total stall.

        Forward advance has no counterpart: once the current stage grinds to
        a halt (e.g. policy drift on an earlier sub-goal it no longer
        rehearses), ``curriculum_stage_idx`` would otherwise sit on a stage
        the agent can no longer actually clear, forever.
        """
        if rate > self.demote_success_ceiling:
            self._stall_checks = 0
            return

        self._stall_checks += 1
        if self._stall_checks < self.demote_stall_checks:
            return

        from curriculum_config import get_goal_for_stage, get_stage_max_steps, prev_stage

        prv = prev_stage(self.stage)
        self._stall_checks = 0
        if prv is None:
            return

        old = self.stage
        self.stage = prv
        self._apply_stage_to_env(self.training_env, prv)
        if self.eval_env is not None:
            self._apply_stage_to_env(self.eval_env, prv)

        self._successes.clear()
        self._returns.clear()
        self._reset_goal_hits()

        goal = get_goal_for_stage(prv)
        max_steps = get_stage_max_steps(prv)
        msg = (
            f"[curriculum] Stalled on {old} (success_rate<={self.demote_success_ceiling:.2f} "
            f"for {self.demote_stall_checks} checks) -> demoting to {prv} "
            f"(goal={goal}, max_steps={max_steps})"
        )
        print(f"\n{msg}\n")
        self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
        self.logger.record("pokemon/goal_success_rate", 0.0)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if not info:
                continue
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
            if info.get("goal_success") or info.get("cleared_stage"):
                self._ep_goal_hit[i] = True
            # Only count a hit toward the *currently set* goal — envs can
            # auto-advance in-place past the callback's stage within one
            # episode, and those later hits belong to a different goal.
            if info.get("goal_success") and info.get("cleared_stage") == self.stage:
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
                "pokemon/goal_hits_current_stage", float(self._goal_hit_count)
            )
            if self._goal_hit_count:
                self.logger.record(
                    "pokemon/goal_hit_avg_reward",
                    self._goal_hit_reward_sum / self._goal_hit_count,
                )
            self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
            self._try_advance()

        return True
