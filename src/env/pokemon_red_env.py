"""Gymnasium environment wrapping PyBoy + Data for PPO training."""
from __future__ import annotations

import sys
from pathlib import Path
from threading import RLock
from typing import Any

# curriculum_config lives at repo root (not under src/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from curriculum_config import (
    GOAL_INDEX,
    N_GOALS,
    get_curriculum_saves,
    get_goal_for_stage,
    get_stage_max_steps,
    next_stage,
    stage_for_goal,
)
from pokemon.Data import (
    GOAL_BADGE_1,
    GOAL_LEFT_HOUSE,
    GOAL_OAKS_LAB,
    GOAL_ROUTE_1,
    Data,
)
from pokemon.Emulator import N_ACTIONS, Emulator

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

# Base feature vector + curriculum goal one-hot.
# player_pokemons_level now reports all 6 party slots (was 1) so the model can
# see bench levels for the smart/coward flee comparison: +5.
_BASE_VECTOR_DIM = 798
VECTOR_DIM = _BASE_VECTOR_DIM + N_GOALS
VISIT_MASK_SIZE = 2 * 5 + 1  # map_vision_radius default


class PokemonRedEnv(gym.Env):
    """Pokemon Red as a Gymnasium Dict-observation Discrete-action env."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        save_state: str = "start",
        max_steps: int = 5000,
        frame_skip: int = 16,
        goal: str = GOAL_BADGE_1,
        render_mode: str | None = None,
        curriculum_mix: float = 0.0,
        curriculum_saves: list[str] | None = None,
        auto_advance: bool = True,
        stage: str | None = None,
        worker_rank: int = 0,
        n_workers: int = 1,
        collect_heatmap: bool = False,
    ):
        super().__init__()
        self.save_state = save_state
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.goal = goal
        self.stage = stage or stage_for_goal(goal)
        self.render_mode = render_mode
        self.curriculum_mix = float(curriculum_mix)
        self.curriculum_saves = list(curriculum_saves or [save_state])
        # When True, goal success advances curriculum in-place (train == eval).
        self.auto_advance = bool(auto_advance)
        self.worker_rank = int(worker_rank)
        self.n_workers = max(1, int(n_workers))
        # When True, snapshot Data.visited_positions (map,x,y)->ticks into info
        # at every run boundary (episode end or mid-episode curriculum-leg
        # clear) for the --heatmap live visualizer. Off by default: it copies
        # a dict every boundary, which is wasted work when nothing reads it.
        self.collect_heatmap = bool(collect_heatmap)

        self._lock = RLock()
        self._emu: Emulator | None = None
        self._memory: bytes | None = None
        self._step_count = 0
        self._episode_loop = False
        # Base curriculum owned by the trainer/callback. In-episode auto_advance
        # is ephemeral — reset() restores these so workers don't permanently
        # drift to later stages and stop counting the current goal.
        self._store_base_curriculum()

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
                "wild_visit_mask": spaces.Box(
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
            if self.render_mode == "human":
                from utils.gui_layout import window_slot

                slot = window_slot(self.worker_rank, self.n_workers)
                self._emu.sdl_scale = slot.scale
                self._emu.window_x = slot.x
                self._emu.window_y = slot.y
                self._emu.window_title = f"Pokemon Red AI - worker {self.worker_rank}"
            self._emu.use_sdl = self.render_mode == "human"
        return self._emu

    def _goal_one_hot(self) -> np.ndarray:
        oh = np.zeros(N_GOALS, dtype=np.float32)
        idx = GOAL_INDEX.get(self.goal)
        if idx is not None:
            oh[idx] = 1.0
        return oh

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

        floats.append(self._goal_one_hot())

        vector = np.concatenate(floats, axis=0)
        if vector.shape[0] != VECTOR_DIM:
            out = np.zeros(VECTOR_DIM, dtype=np.float32)
            n = min(VECTOR_DIM, vector.shape[0])
            out[:n] = vector[:n]
            vector = out

        screen = inputs["screen_tiles"].detach().cpu().numpy().astype(np.float32)
        if "visit_mask" in inputs:
            visit = inputs["visit_mask"].detach().cpu().numpy().astype(np.float32)
        else:
            visit = np.zeros((1, VISIT_MASK_SIZE, VISIT_MASK_SIZE), dtype=np.float32)
        if "wild_visit_mask" in inputs:
            wild_visit = (
                inputs["wild_visit_mask"].detach().cpu().numpy().astype(np.float32)
            )
        else:
            wild_visit = np.zeros((1, VISIT_MASK_SIZE, VISIT_MASK_SIZE), dtype=np.float32)

        return {
            "screen_tiles": screen,
            "visit_mask": visit,
            "wild_visit_mask": wild_visit,
            "vector": vector,
        }

    def _store_base_curriculum(self) -> None:
        self._base_stage = self.stage
        self._base_goal = self.goal
        self._base_max_steps = self.max_steps
        self._base_save_state = self.save_state
        self._base_curriculum_saves = list(self.curriculum_saves)
        self._base_curriculum_mix = self.curriculum_mix

    def _restore_base_curriculum(self) -> None:
        self.stage = self._base_stage
        self.goal = self._base_goal
        self.max_steps = self._base_max_steps
        self.save_state = self._base_save_state
        self.curriculum_saves = list(self._base_curriculum_saves)
        self.curriculum_mix = self._base_curriculum_mix

    def set_curriculum(
        self,
        goal: str | None = None,
        max_steps: int | None = None,
        save_state: str | None = None,
        curriculum_saves: list[str] | None = None,
        curriculum_mix: float | None = None,
        reset_steps: bool = False,
        stage: str | None = None,
        clear_visits: bool = False,
        permanent: bool = True,
    ) -> dict[str, Any]:
        """Hot-update goal / episode length / saves (for auto-curriculum).

        ``permanent=True`` (default) updates the base stage used on ``reset``
        — for MilestoneCallback / eval episode starts. ``permanent=False`` is
        for in-episode auto_advance only (restored on the next reset).
        """
        if stage is not None:
            self.stage = str(stage)
        if goal is not None:
            self.goal = str(goal)
            if stage is None:
                self.stage = stage_for_goal(self.goal)
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
        if permanent:
            self._store_base_curriculum()
        if reset_steps:
            self._step_count = 0
            self._episode_loop = False
            if self._emu is not None:
                self.emu.data.loop_flag = False
        if clear_visits and self._emu is not None:
            # Fresh exploration pressure toward the next goal.
            data = self.emu.data
            data.visited_positions.clear()
            data.position_visit_counts.clear()
            data.wild_visit_counts.clear()
            data.direction_counts.clear()
            data.map_transitions.clear()
            data.reward_sums.clear()
            data._last_heatmap_pos = None
            data.recent_positions.clear()
            data.recent_actions.clear()
            data.loop_flag = False
            data.loop_streak = 0
            data.in_menu_ticks = 0
            data.in_battle_ticks = 0
            data.in_dialog_ticks = 0
            data._dialog_screens_seen = set()
            data._completed_dialogs = set()
            data._dialog_reopen_counts = {}
            data._dialog_reopen_truncate = False
        return {
            "goal": self.goal,
            "stage": self.stage,
            "max_steps": self.max_steps,
            "save_state": self.save_state,
            "curriculum_saves": list(self.curriculum_saves),
            "curriculum_mix": self.curriculum_mix,
            "base_stage": self._base_stage,
        }

    def _advance_after_goal(self) -> tuple[bool, str | None]:
        """Advance curriculum in-place. Returns (advanced, cleared_stage).

        Ephemeral: does not change the base stage, so ``reset()`` returns to
        the trainer-assigned curriculum until MilestoneCallback advances it.
        """
        cleared = self.stage
        cleared_goal = self.goal
        nxt = next_stage(
            self.stage,
            is_satisfied=self.emu.data.is_goal_satisfied,
        )
        if nxt is None:
            return False, cleared
        # Mark before leaving the map so location clawback does not undo a
        # legitimately cleared curriculum beat (lab → Route 1).
        self.emu.data.mark_goal_cleared(cleared_goal)
        self.set_curriculum(
            stage=nxt,
            goal=get_goal_for_stage(nxt),
            max_steps=get_stage_max_steps(nxt),
            # Keep base saves; only the live goal/max_steps change this episode.
            reset_steps=True,
            clear_visits=True,
            permanent=False,
        )
        return True, cleared

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
        # Drop any in-episode auto_advance; train on the assigned stage.
        self._restore_base_curriculum()
        save = self._pick_save(options)
        memory, inputs = self.emu.reset(dir=save)
        self.emu.data.collect_heatmap = self.collect_heatmap
        self.emu.data.goal = self.goal
        if options and options.get("goal"):
            self.emu.data.goal = str(options["goal"])
            self.goal = str(options["goal"])
            self.stage = stage_for_goal(self.goal)
            self._store_base_curriculum()
        self._memory = memory
        self._step_count = 0
        self._episode_loop = False
        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(
            terminated=False,
            truncated=False,
            goal_success=False,
            cleared_stage=None,
        )
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

        goal_success = False
        cleared_stage: str | None = None
        # (positions, directions, transitions, rewards, steps) snapshot of the
        # run that's ending — either the whole episode or one curriculum leg
        # (auto_advance clears visits mid-episode on goal success). Grabbed
        # before Data.clean()/clear_visits wipes them, so it's the "last X
        # frames" heatmap unit.
        heatmap_run: tuple[dict, dict, dict, dict, int] | None = None
        # Train/eval parity: success advances in-place instead of ending the episode.
        if terminated and self.auto_advance and not truncated:
            if self.collect_heatmap:
                heatmap_run = (
                    dict(self.emu.data.visited_positions),
                    dict(self.emu.data.direction_counts),
                    dict(self.emu.data.map_transitions),
                    dict(self.emu.data.reward_sums),
                    self._step_count,
                )
            advanced, cleared_stage = self._advance_after_goal()
            if advanced:
                goal_success = True
                terminated = False
                # Obs must reflect the new goal one-hot immediately.
                inputs = self.emu.data.inputs()

        if (terminated or truncated) and self.collect_heatmap and heatmap_run is None:
            heatmap_run = (
                dict(self.emu.data.visited_positions),
                dict(self.emu.data.direction_counts),
                dict(self.emu.data.map_transitions),
                dict(self.emu.data.reward_sums),
                self._step_count,
            )

        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(
            terminated=terminated,
            truncated=truncated,
            goal_success=goal_success,
            cleared_stage=cleared_stage,
            heatmap_run=heatmap_run,
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _info(
        self,
        terminated: bool,
        truncated: bool,
        goal_success: bool = False,
        cleared_stage: str | None = None,
        heatmap_run: tuple[dict, dict, dict, dict, int] | None = None,
    ) -> dict[str, Any]:
        data: Data = self.emu.data
        mem = self.emu.pyboy.memory
        badges = data.badges(mem)
        live_goals = data.live_story_goals()
        return {
            "heatmap_positions": heatmap_run[0] if heatmap_run else None,
            "heatmap_directions": heatmap_run[1] if heatmap_run else None,
            "heatmap_transitions": heatmap_run[2] if heatmap_run else None,
            "heatmap_rewards": heatmap_run[3] if heatmap_run else None,
            "heatmap_steps": heatmap_run[4] if heatmap_run else None,
            "badges": int(sum(badges)),
            "badge_bits": badges,
            "map_id": int(data.map_id(mem)),
            "loop_flag": bool(self._episode_loop or data.loop_flag),
            "milestone": data.current_milestone(),
            "milestones_hit": sorted(data._milestones_hit),
            "goals_live": live_goals,
            "goals_live_count": len(live_goals),
            "goals_peak_count": int(data._peak_live_goals),
            "goals_regressed": list(data._last_regressed),
            "goals_regressed_hard": list(data._last_hard_regressed),
            "goals_cleared": sorted(data._cleared_goals),
            "menu_ticks": float(data.in_menu_ticks),
            "steps": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            "goal": data.goal,
            "stage": self.stage,
            "goal_success": bool(goal_success),
            "cleared_stage": cleared_stage,
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
    frame_skip: int = 16,
    goal: str = GOAL_BADGE_1,
    rank: int = 0,
    seed: int = 0,
    curriculum_mix: float = 0.3,
    curriculum_saves: list[str] | None = None,
    auto_advance: bool = True,
    stage: str | None = None,
    render_mode: str | None = None,
    n_workers: int = 1,
    collect_heatmap: bool = False,
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
            auto_advance=auto_advance,
            stage=stage,
            render_mode=render_mode,
            worker_rank=rank,
            n_workers=n_workers,
            collect_heatmap=collect_heatmap,
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
