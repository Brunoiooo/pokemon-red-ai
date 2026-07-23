"""Gymnasium environment wrapping PyBoy + Data for PPO training."""
from __future__ import annotations

from threading import RLock
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon.Data import (
    GOAL_BADGE_1,
    GOAL_LEFT_HOUSE,
    GOAL_OAKS_LAB,
    GOAL_ROUTE_1,
    Data,
)
from pokemon.Emulator import MEMORY_SNAPSHOT_END, N_ACTIONS, Emulator

# Flattened float feature groups from Data.inputs() (excluding images / raw ids).
_VECTOR_FLOAT_KEYS = (
    "core",
    "battle",
    "menu_battle_dialog",
    "mode",
    "progress",
    "nav",
    "inv",
    "party",
)
_ID_SCALAR_KEYS = (
    "map_id",
    "dialog_id",
    "index_of_current_pokemon_send_out",
    "type_of_battle",
    "move_menu_type",
)
_ID_SEQ_KEYS = (
    "move_id",
    "move_type",
    "pokemon_id",
    "pokemon_type",
    "sprite_id",
    "item_id",
)

# Sizes measured from a live reset (must stay in sync with Data.py).
VECTOR_DIM = 793
VISIT_MASK_SIZE = 2 * 5 + 1  # map_vision_radius default


class PokemonRedEnv(gym.Env):
    """Pokemon Red as a Gymnasium Dict-observation Discrete-action env."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        save_state: str = "start",
        max_steps: int = 5000,
        frame_skip: int = 24,
        goal: str = GOAL_BADGE_1,
        render_mode: str | None = None,
        curriculum_mix: float = 0.0,
        curriculum_saves: list[str] | None = None,
    ):
        super().__init__()
        self.save_state = save_state
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.goal = goal
        self.render_mode = render_mode
        self.curriculum_mix = float(curriculum_mix)
        self.curriculum_saves = list(curriculum_saves or [save_state])

        self._lock = RLock()
        self._emu: Emulator | None = None
        self._memory: bytes | None = None
        self._step_count = 0
        self._episode_loop = False

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "screen_tiles": spaces.Box(
                    low=0.0, high=1.0, shape=(1, 18, 20), dtype=np.float32
                ),
                "visit_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, VISIT_MASK_SIZE, VISIT_MASK_SIZE),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(VECTOR_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def emu(self) -> Emulator:
        if self._emu is None:
            self._emu = Emulator(files_lock=self._lock)
            self._emu.use_sdl = self.render_mode == "human"
        return self._emu

    def _torch_inputs_to_obs(self, inputs: dict) -> dict[str, np.ndarray]:
        floats: list[np.ndarray] = []
        for key in _VECTOR_FLOAT_KEYS:
            floats.append(inputs[key].detach().cpu().numpy().astype(np.float32).ravel())

        for key in _ID_SCALAR_KEYS:
            val = float(inputs[key].detach().cpu().item())
            floats.append(np.array([val / 255.0], dtype=np.float32))

        for key in _ID_SEQ_KEYS:
            arr = inputs[key].detach().cpu().numpy().astype(np.float32).ravel() / 255.0
            floats.append(arr)

        vector = np.concatenate(floats, axis=0)
        if vector.shape[0] != VECTOR_DIM:
            # Pad / trim defensively if Data feature sizes drift.
            out = np.zeros(VECTOR_DIM, dtype=np.float32)
            n = min(VECTOR_DIM, vector.shape[0])
            out[:n] = vector[:n]
            vector = out

        screen = inputs["screen_tiles"].detach().cpu().numpy().astype(np.float32)
        if "visit_mask" in inputs:
            visit = inputs["visit_mask"].detach().cpu().numpy().astype(np.float32)
        else:
            visit = np.zeros((1, VISIT_MASK_SIZE, VISIT_MASK_SIZE), dtype=np.float32)

        return {
            "screen_tiles": screen,
            "visit_mask": visit,
            "vector": vector,
        }

    def set_curriculum(
        self,
        goal: str | None = None,
        max_steps: int | None = None,
        save_state: str | None = None,
        curriculum_saves: list[str] | None = None,
        curriculum_mix: float | None = None,
        reset_steps: bool = False,
    ) -> dict[str, Any]:
        """Hot-update goal / episode length / saves (for auto-curriculum)."""
        if goal is not None:
            self.goal = str(goal)
            if self._emu is not None:
                self.emu.data.goal = self.goal
        if max_steps is not None:
            self.max_steps = int(max_steps)
        if save_state is not None:
            self.save_state = str(save_state)
        if curriculum_saves is not None:
            self.curriculum_saves = list(curriculum_saves)
        if curriculum_mix is not None:
            self.curriculum_mix = float(curriculum_mix)
        if reset_steps:
            self._step_count = 0
            self._episode_loop = False
            if self._emu is not None:
                self.emu.data.loop_flag = False
        return {
            "goal": self.goal,
            "max_steps": self.max_steps,
            "save_state": self.save_state,
            "curriculum_saves": list(self.curriculum_saves),
            "curriculum_mix": self.curriculum_mix,
        }

    def _pick_save(self, options: dict | None) -> str:
        if options and options.get("save"):
            return str(options["save"])
        if self.curriculum_mix > 0 and len(self.curriculum_saves) > 1:
            import random

            if random.random() < self.curriculum_mix:
                return random.choice(self.curriculum_saves[:-1])
            return self.curriculum_saves[-1]
        return self.save_state

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        save = self._pick_save(options)
        memory, inputs = self.emu.reset(dir=save)
        self.emu.data.goal = self.goal
        if options and options.get("goal"):
            self.emu.data.goal = str(options["goal"])
        self._memory = memory
        self._step_count = 0
        self._episode_loop = False
        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(terminated=False, truncated=False)
        return obs, info

    def step(self, action: int):
        assert self._memory is not None, "Call reset() before step()"
        memory, inputs, reward, terminated, truncated = self.emu.step_discrete(
            memory=self._memory,
            action_idx=int(action),
            duration=self.frame_skip,
            render_each=False,
        )
        self._memory = memory
        self._step_count += 1
        if self.emu.data.loop_flag:
            self._episode_loop = True

        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(terminated=terminated, truncated=truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _info(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        data: Data = self.emu.data
        mem = self.emu.pyboy.memory
        badges = data.badges(mem)
        return {
            "badges": int(sum(badges)),
            "badge_bits": badges,
            "map_id": int(data.map_id(mem)),
            "loop_flag": bool(self._episode_loop or data.loop_flag),
            "milestone": data.current_milestone(),
            "milestones_hit": sorted(data._milestones_hit),
            "menu_ticks": float(data.in_menu_ticks),
            "steps": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            "goal": data.goal,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return np.asarray(self.emu.pyboy.screen.ndarray)
        return None

    def close(self):
        if self._emu is not None:
            try:
                pyboy = getattr(self._emu, "_Emulator__pyboy", None)
                if pyboy is not None:
                    pyboy.stop(False)
            except Exception:
                pass
            self._emu = None


def make_pokemon_env(
    save_state: str = "start",
    max_steps: int = 5000,
    frame_skip: int = 24,
    goal: str = GOAL_BADGE_1,
    rank: int = 0,
    seed: int = 0,
    curriculum_mix: float = 0.3,
    curriculum_saves: list[str] | None = None,
):
    """Factory for SubprocVecEnv workers."""

    def _init():
        env = PokemonRedEnv(
            save_state=save_state,
            max_steps=max_steps,
            frame_skip=frame_skip,
            goal=goal,
            curriculum_mix=curriculum_mix,
            curriculum_saves=curriculum_saves,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


# Re-export goals for trainers / curriculum.
__all__ = [
    "PokemonRedEnv",
    "make_pokemon_env",
    "VECTOR_DIM",
    "GOAL_BADGE_1",
    "GOAL_LEFT_HOUSE",
    "GOAL_ROUTE_1",
    "GOAL_OAKS_LAB",
]
