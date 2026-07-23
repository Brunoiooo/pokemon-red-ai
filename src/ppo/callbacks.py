"""Training callbacks: milestone / loop-rate logging for PPO."""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MilestoneCallback(BaseCallback):
    """Track episode milestone hit-rate and loop episode rate."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._loops: deque[int] = deque(maxlen=window)
        self._left_house: deque[int] = deque(maxlen=window)
        self._route1: deque[int] = deque(maxlen=window)
        self._badge1: deque[int] = deque(maxlen=window)
        self._ep_loop = None
        self._ep_milestones = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_loop = [False] * n
        self._ep_milestones = [set() for _ in range(n)]

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
            # Also track current milestone string.
            cur = info.get("milestone")
            if cur and cur != "start":
                self._ep_milestones[i].add(cur)

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                self._loops.append(1 if self._ep_loop[i] else 0)
                ms = self._ep_milestones[i]
                self._left_house.append(1 if "left_house" in ms else 0)
                self._route1.append(1 if "route1" in ms else 0)
                self._badge1.append(
                    1 if ("badge1" in ms or info.get("badges", 0) >= 1) else 0
                )
                if "episode" in info:
                    self._returns.append(float(info["episode"]["r"]))
                elif rewards is not None and i < len(rewards):
                    pass

                self._ep_loop[i] = False
                self._ep_milestones[i] = set()

        if len(self._loops) >= 10 and self.n_calls % 2048 == 0:
            loop_rate = float(np.mean(self._loops))
            self.logger.record("pokemon/loop_episode_rate", loop_rate)
            self.logger.record(
                "pokemon/left_house_rate", float(np.mean(self._left_house) if self._left_house else 0)
            )
            self.logger.record(
                "pokemon/route1_rate", float(np.mean(self._route1) if self._route1 else 0)
            )
            self.logger.record(
                "pokemon/badge1_rate", float(np.mean(self._badge1) if self._badge1 else 0)
            )
            if self._returns:
                self.logger.record("pokemon/ep_return_mean", float(np.mean(self._returns)))

        return True
