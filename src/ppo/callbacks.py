"""Training callbacks: milestone / loop-rate logging + auto curriculum."""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


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
        )

        self.success_threshold = (
            ADVANCE_SUCCESS_THRESHOLD if success_threshold is None else success_threshold
        )
        self.min_episodes = ADVANCE_MIN_EPISODES if min_episodes is None else min_episodes
        self.check_every = ADVANCE_CHECK_EVERY if check_every is None else check_every

        self._returns: deque[float] = deque(maxlen=window)
        self._loops: deque[int] = deque(maxlen=window)
        self._successes: deque[int] = deque(maxlen=window)
        self._badges: deque[int] = deque(maxlen=window)
        self._ep_loop = None
        self._ep_milestones = None
        self._ep_goal_hit = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_loop = [False] * n
        self._ep_milestones = [set() for _ in range(n)]
        self._ep_goal_hit = [False] * n
        self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())

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
            return

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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if not info:
                continue
            if info.get("loop_flag"):
                self._ep_loop[i] = True
            for m in info.get("milestones_hit", []) or []:
                self._ep_milestones[i].add(m)
            cur = info.get("milestone")
            if cur and cur != "start":
                self._ep_milestones[i].add(cur)

            done = bool(dones[i]) if i < len(dones) else False
            if info.get("goal_success") or info.get("cleared_stage"):
                self._ep_goal_hit[i] = True

            if done:
                self._loops.append(1 if self._ep_loop[i] else 0)
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

                if "episode" in info:
                    self._returns.append(float(info["episode"]["r"]))
                elif rewards is not None and i < len(rewards):
                    pass

                self._ep_loop[i] = False
                self._ep_milestones[i] = set()
                self._ep_goal_hit[i] = False

        if len(self._loops) >= 10 and self.n_calls % self.check_every == 0:
            loop_rate = float(np.mean(self._loops))
            self.logger.record("pokemon/loop_episode_rate", loop_rate)
            if self._successes:
                self.logger.record(
                    "pokemon/goal_success_rate", float(np.mean(self._successes))
                )
            if self._badges:
                self.logger.record(
                    "pokemon/badges_mean", float(np.mean(self._badges))
                )
            if self._returns:
                self.logger.record(
                    "pokemon/ep_return_mean", float(np.mean(self._returns))
                )
            self.logger.record("pokemon/curriculum_stage_idx", self._stage_idx())
            self._try_advance()

        return True
