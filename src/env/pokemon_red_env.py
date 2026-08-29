"""Gymnasium environment wrapping PyBoy + Data for PPO training."""
from __future__ import annotations

import dataclasses
import os
import sys
import time
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
    stage_for_goal,
)
from pokemon.Data import (
    BADGE_GOALS,
    GOAL_CANDIDATES,
    REWARD_COMPONENT_NAMES,
    REWARD_MODE_NAMES,
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
    "dialog_id_visit_counts",
    "map_id_visit_counts",
    "reward_component_sums",
    "map_budget_progress",
    "stuck_tile_progress",
    "loop_streak_progress",
    "compass_progress",
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
# +256 for dialog_id_visit_counts: a per-map histogram over the full byte
# range dialog_id is read from (see Data.dialog_id_visit_grid).
# +256 for map_id_visit_counts: an episode-wide histogram over the full byte
# range map_id is read from (see Data.map_id_visit_grid).
# +len(REWARD_COMPONENT_NAMES) for reward_component_sums: cumulative
# per-episode sum of each named reward sub-component, incl. "total" (see
# Data.reward_component_vector / REWARD_COMPONENT_NAMES).
# +256 for map_budget_progress: episode-wide per-map_id histogram of
# world_map_step_counts / map_truncate_budget, the map_budget truncate
# cause's "distance to fuse" (see Data.map_budget_progress).
# +1 for stuck_tile_progress: current tile's visited_positions / the
# stuck_tile truncate fuse (max_useless_ticks) (see Data.stuck_tile_progress).
# +1 for loop_streak_progress: loop_streak / the loop_streak truncate fuse
# (max_loop_streak) (see Data.loop_streak_progress).
# +3 for compass_progress: [dx, dy, has_target] toward the nearest
# currently-unlockable event -- a deterministic BFS "compass" over
# pret/pokered-derived map data (see pokemon.event_compass.
# nearest_unlockable_event / Data.compass_progress), not a learned signal.
# has_target=0 (dx=dy=0) whenever no candidate is reachable right now,
# deliberately never a guessed direction.
_BASE_VECTOR_DIM = (
    798 + 256 + 256 + len(REWARD_COMPONENT_NAMES) + 256 + 1 + 1 + 3
)
VECTOR_DIM = _BASE_VECTOR_DIM + N_GOALS
# Derived from Data's own map_vision_radius field default (not a separately
# hand-kept "5" literal) so the two never drift apart. A per-env override
# (see PokemonRedEnv's data_kwargs) recomputes this per instance -- see
# _visit_mask_size below; this module-level constant stays the *default*.
DEFAULT_MAP_VISION_RADIUS: int = next(
    f.default for f in dataclasses.fields(Data) if f.name == "map_vision_radius"
)
VISIT_MASK_SIZE = 2 * DEFAULT_MAP_VISION_RADIUS + 1


class PokemonRedEnv(gym.Env):
    """Pokemon Red as a Gymnasium Dict-observation Discrete-action env."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        save_state: str = "start",
        max_steps: int = 5000,
        frame_skip: int = 16,
        goal: str = BADGE_GOALS[0],
        render_mode: str | None = None,
        curriculum_mix: float = 0.0,
        curriculum_saves: list[str] | None = None,
        auto_advance: bool = True,
        stage: str | None = None,
        worker_rank: int = 0,
        n_workers: int = 1,
        collect_heatmap: bool = False,
        save_checkpoints: bool = False,
        data_kwargs: dict | None = None,
        emulator_kwargs: dict | None = None,
    ):
        super().__init__()
        # Splatted into Data(...) / Emulator(...) respectively -- lets a
        # CLI/GUI reward-shaping (data_kwargs) or emulator-timing
        # (emulator_kwargs) override (see tunables.py / cli.py) reach the
        # actual running env. Recompute the visit-mask size from data_kwargs
        # too so a map_vision_radius override doesn't drift out of sync
        # with observation_space's declared shape (see VISIT_MASK_SIZE).
        self.data_kwargs = dict(data_kwargs or {})
        self.emulator_kwargs = dict(emulator_kwargs or {})
        self._visit_mask_size = (
            2 * int(self.data_kwargs.get("map_vision_radius", DEFAULT_MAP_VISION_RADIUS)) + 1
        )
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
        # When True, on every step that newly satisfies one or more
        # GOAL_CANDIDATES events (see Data.reward_generic_progress), write
        # saves/<name>/checkpoint.state for each -- whatever event it is,
        # regardless of the curriculum's currently-assigned goal or
        # STAGE_ORDER's heuristic position for it. See _save_milestone_checkpoints.
        self.save_checkpoints = bool(save_checkpoints)

        self._lock = RLock()
        self._emu: Emulator | None = None
        self._memory: bytes | None = None
        self._step_count = 0
        self._episode_loop = False
        self._episode_loop_causes: set[str] = set()
        self._goals_peak_count = 0
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
                    shape=(1, self._visit_mask_size, self._visit_mask_size),
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
            self._emu = Emulator(
                files_lock=self._lock,
                data_kwargs=self.data_kwargs,
                **self.emulator_kwargs,
            )
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
            visit = np.zeros(
                (1, self._visit_mask_size, self._visit_mask_size), dtype=np.float32
            )

        return {
            "screen_tiles": screen,
            "visit_mask": visit,
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
            self._episode_loop_causes = set()
            if self._emu is not None:
                self.emu.data.loop_flag = False
        if clear_visits and self._emu is not None:
            # Fresh exploration pressure toward the next goal.
            data = self.emu.data
            data.visited_positions.clear()
            data.position_visit_counts.clear()
            data.direction_counts.clear()
            data.map_transitions.clear()
            data.reward_sums.clear()
            data.reward_component_sums.clear()
            data.reward_mode_sums.clear()
            data.battle_outcome_counts.clear()
            data.milestone_hit_counts.clear()
            data.dialog_hit_counts.clear()
            data.truncate_hit_counts.clear()
            data.battle_outcome_tally.clear()
            data.battle_step_count = 0
            data._wild_decay_sum = 0.0
            data._wild_decay_count = 0
            data._wild_decay_floored_count = 0
            # Reward/truncation-affecting, same "fresh pressure" reasoning as
            # position_visit_counts above -- without this a map budget or
            # loop streak partly spent on the previous curriculum leg would
            # ramp/trip a penalty attributed to the brand-new goal.
            data.world_map_step_counts.clear()
            data.dialog_id_visit_counts.clear()
            # Reward-affecting too (reward_new_map_presence) despite reading
            # like a purely informational dwell histogram -- same "fresh
            # pressure toward the new goal" reasoning as the counters above;
            # previously missing here, so a map visited heavily on the prior
            # curriculum leg kept its reward_new_map_presence bonus suppressed
            # into the new leg instead of resetting like its siblings.
            data.map_id_visit_counts.clear()
            data.menu_noop_streak = 0
            data.dialog_wrong_streak = 0
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
        """Keep the episode going in-place after a milestone. Returns
        (advanced, cleared_stage).

        Not gated on the specifically-assigned self.goal: Data.terminated()
        fires on ANY GOAL_CANDIDATES event/badge newly satisfied this step
        (see last_milestone_payouts). Whichever goal(s) actually landed this
        step count as "cleared"; self.goal/self.stage are deliberately left
        alone here -- reward_generic_progress already pays for any newly
        satisfied goal regardless of assignment, and max_steps is a flat
        constant for every goal, so there is nothing to gain by reassigning
        self.goal mid-episode, only risk: a random reassignment (the old
        curriculum_config.pick_new_goal call this used to make) could hand
        out a goal nowhere near the map the episode is actually standing on,
        with no way to get there this episode -- wasting the rest of the run
        wandering back toward familiar ground instead of continuing to
        explore forward. Per-leg counters get a fresh start (see
        clear_visits), so budgets/penalties from before this goal cleared
        don't leak into the fresh pressure the episode still has ahead --
        except self._step_count, deliberately NOT reset here (unlike
        set_curriculum's reset_steps=True, which this used to pass): a
        policy that keeps chaining newly-satisfied milestones re-triggers
        this method every time, and each call would otherwise hand max_steps
        a brand new lease on life. self._step_count must stay a true,
        never-reset ceiling on the *whole episode*, not just the current
        leg -- a hung eval episode (a deterministic policy stringing
        together enough milestones to keep resetting its own truncation
        fuses) blocked MaskableEvalCallback for minutes before this was
        caught (evaluate_policy runs synchronously in the main training
        process, so one runaway eval episode stalls every worker).
        """
        hits = [name for name, _ in self.emu.data.last_milestone_payouts]
        cleared = hits[0] if hits else self.stage
        # Mark before leaving the map so location clawback does not undo a
        # legitimately cleared curriculum beat (lab → Route 1).
        for name in hits:
            self.emu.data.mark_goal_cleared(name)
        self._episode_loop = False
        self._episode_loop_causes = set()
        self.emu.data.loop_flag = False
        self.set_curriculum(clear_visits=True, permanent=False)
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
        self.emu.data.seed_cleared_goals(self.emu.data.goal)
        self._memory = memory
        self._step_count = 0
        self._episode_loop = False
        self._episode_loop_causes = set()
        self._goals_peak_count = 0
        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(
            terminated=False,
            truncated=False,
            goal_success=False,
            cleared_stage=None,
        )
        return obs, info

    def action_masks(self) -> np.ndarray:
        """Legal-action mask for sb3-contrib MaskablePPO (see get_action_masks).

        Called by the trainer right before predicting the *next* action, so
        it must reflect live emulator state, same as the reward/mode checks
        in Emulator.step_discrete -- not the stale pre-step self._memory
        snapshot.
        """
        assert self._emu is not None, "Call reset() before action_masks()"
        return np.asarray(
            self.emu.data.legal_action_mask(self.emu.pyboy.memory), dtype=bool
        )

    def _save_milestone_checkpoints(self) -> None:
        """Dynamic, order-free checkpoint capture.

        Every GOAL_CANDIDATES event/badge newly satisfied this step (see
        Data.reward_generic_progress -- self.emu.data.last_milestone_payouts)
        gets its own saves/<name>/checkpoint.state, whichever event it
        happens to be and regardless of whether it's the curriculum's
        currently-assigned goal or where STAGE_ORDER's heuristic sort
        happens to place it. In-game event order is not guaranteed to match
        STAGE_ORDER (see curriculum_config.py's module docstring), so tying
        saves to the single assigned goal meant a run could clear several
        other goals along the way and never save any of them -- whichever
        goal actually lands first is the good checkpoint, not whichever one
        STAGE_ORDER says should be next.
        """
        for name, _payout in self.emu.data.last_milestone_payouts:
            out_dir = Path("saves") / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "checkpoint.state"
            # Training runs several PokemonRedEnv workers as separate
            # SubprocVecEnv processes -- two of them can clear the same
            # goal in the same wall-clock window and race to write this
            # exact path. self.emu.files_lock only guards this one
            # process, so write to a per-process temp file and os.replace()
            # into place (atomic on POSIX and Windows on the same volume):
            # the file a later reset() loads is always one complete
            # writer's state, never an interleaved/corrupt mix of two.
            tmp_path = out_dir / f".checkpoint.state.tmp.{os.getpid()}"
            with self.emu.files_lock:
                with open(tmp_path, "wb") as f:
                    self.emu.pyboy.save_state(f)
                # os.replace can transiently raise WinError 5 (access
                # denied) on Windows when another process -- another
                # worker's os.replace into this same path, or the AV's
                # real-time scanner -- briefly holds the destination file
                # without FILE_SHARE_DELETE. Retry instead of crashing the
                # whole SubprocVecEnv worker over a checkpoint write.
                for attempt in range(5):
                    try:
                        os.replace(tmp_path, out_path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            print(
                                f"Warning: giving up replacing {out_path} "
                                f"after repeated PermissionError; leaving "
                                f"existing checkpoint in place."
                            )
                            tmp_path.unlink(missing_ok=True)
                        else:
                            time.sleep(0.05 * (2**attempt))

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
        if self.save_checkpoints and self.emu.data.last_milestone_payouts:
            self._save_milestone_checkpoints()
        if self.emu.data.loop_flag:
            self._episode_loop = True
            self._episode_loop_causes |= self.emu.data.loop_causes

        # Which fuse ended the episode (see Data.truncated / TRUNCATE_CAUSES) —
        # "max_steps" covers running out of the step budget without any fuse
        # firing first, the one truncation cause Data itself can't see.
        truncate_causes = set(self.emu.data.last_truncate_causes) if truncated else set()
        if self._step_count >= self.max_steps:
            truncated = True
            if not truncate_causes:
                truncate_causes.add("max_steps")

        # --heatmap "truncations" overlay: where episodes actually run out,
        # not just where time is spent. Attributed here (not in Data.truncated
        # itself) since "max_steps" above is only known at this level. Same
        # _last_heatmap_pos anchor as the other mid-dialog-safe overlays —
        # a stuck_dialog ending has no reliable is_world() position of its own.
        if truncated and self.collect_heatmap:
            pos = self.emu.data._last_heatmap_pos
            if pos is not None:
                self.emu.data.truncate_hit_counts[pos] = (
                    self.emu.data.truncate_hit_counts.get(pos, 0) + 1
                )

        goal_success = False
        cleared_stage: str | None = None
        # (positions, directions, transitions, rewards, steps) snapshot of the
        # run that's ending — either the whole episode or one curriculum leg
        # (auto_advance clears visits mid-episode on goal success). Grabbed
        # before Data.clean()/clear_visits wipes them, so it's the "last X
        # frames" heatmap unit.
        heatmap_run: tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict, int] | None = None
        # Train/eval parity: success advances in-place instead of ending the episode.
        if terminated and self.auto_advance and not truncated:
            if self.collect_heatmap:
                heatmap_run = (
                    dict(self.emu.data.visited_positions),
                    dict(self.emu.data.direction_counts),
                    dict(self.emu.data.map_transitions),
                    dict(self.emu.data.reward_sums),
                    dict(self.emu.data.battle_outcome_counts),
                    dict(self.emu.data.milestone_hit_counts),
                    dict(self.emu.data.dialog_hit_counts),
                    dict(self.emu.data.truncate_hit_counts),
                    dict(self.emu.data.position_visit_counts),
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
                dict(self.emu.data.battle_outcome_counts),
                dict(self.emu.data.milestone_hit_counts),
                dict(self.emu.data.dialog_hit_counts),
                dict(self.emu.data.truncate_hit_counts),
                dict(self.emu.data.position_visit_counts),
                self._step_count,
            )

        obs = self._torch_inputs_to_obs(inputs)
        info = self._info(
            terminated=terminated,
            truncated=truncated,
            goal_success=goal_success,
            cleared_stage=cleared_stage,
            heatmap_run=heatmap_run,
            truncate_causes=truncate_causes,
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _info(
        self,
        terminated: bool,
        truncated: bool,
        goal_success: bool = False,
        cleared_stage: str | None = None,
        heatmap_run: tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict, int] | None = None,
        truncate_causes: set[str] | None = None,
    ) -> dict[str, Any]:
        data: Data = self.emu.data
        mem = self.emu.pyboy.memory
        badges = data.badges(mem)
        live_goals = data.live_story_goals()
        # Peak live-goal count seen this episode (see reset()'s init) — this
        # bookkeeping used to live on Data (_peak_live_goals), tied to the
        # old reward_goal_regression; it's purely a monitoring concern now,
        # so it lives on the env wrapper instead.
        self._goals_peak_count = max(self._goals_peak_count, len(live_goals))
        # Mean dwell-steps per map touched so far this episode (see
        # map_id_visit_counts / Data.map_dwell_budget) — how close episodes
        # run to the flat per-map budget on average, for the training
        # callback's pokemon/map_dwell_avg_count metric.
        dwell_counts = data.map_id_visit_counts
        map_dwell_avg = (
            sum(dwell_counts.values()) / len(dwell_counts) if dwell_counts else 0.0
        )
        # Mean visits-per-tile over every distinct (x, y, map_id) touched so
        # far this episode (see position_visit_counts) — how heavily the
        # episode is revisiting its own footprint on average, for the
        # training callback's pokemon/position_visit_avg_count metric.
        visit_counts = data.position_visit_counts
        position_visit_avg = (
            sum(visit_counts.values()) / len(visit_counts) if visit_counts else 0.0
        )
        party_count: int | None = None
        party_avg_level: float | None = None
        if heatmap_run:
            party_count = int(data.party_count(mem))
            levels = data.all_party_levels(mem)
            party_avg_level = sum(levels) / len(levels) if levels else 0.0
        return {
            "heatmap_positions": heatmap_run[0] if heatmap_run else None,
            "heatmap_directions": heatmap_run[1] if heatmap_run else None,
            "heatmap_transitions": heatmap_run[2] if heatmap_run else None,
            "heatmap_rewards": heatmap_run[3] if heatmap_run else None,
            "heatmap_battle_outcomes": heatmap_run[4] if heatmap_run else None,
            "heatmap_milestones": heatmap_run[5] if heatmap_run else None,
            "heatmap_dialogs": heatmap_run[6] if heatmap_run else None,
            "heatmap_truncations": heatmap_run[7] if heatmap_run else None,
            "heatmap_visit_counts": heatmap_run[8] if heatmap_run else None,
            "heatmap_steps": heatmap_run[9] if heatmap_run else None,
            "party_count": party_count,
            "party_avg_level": party_avg_level,
            "badges": int(sum(badges)),
            "badge_bits": badges,
            "map_id": int(data.map_id(mem)),
            "map_dwell_avg": float(map_dwell_avg),
            "position_visit_avg": float(position_visit_avg),
            "loop_flag": bool(self._episode_loop or data.loop_flag),
            "loop_causes": sorted(self._episode_loop_causes | data.loop_causes),
            "milestone": data.current_milestone(),
            "milestones_hit": sorted(data._milestones_hit),
            # GOAL_CANDIDATES already satisfied by the resume checkpoint this
            # episode loaded from (see Data._episode_start_milestones), and
            # the ones newly cleared *this step* (last_milestone_payouts,
            # filtered to real goals — it also carries HEALED_AT_* pseudo
            # milestones). MilestoneCallback uses these to score each goal's
            # rolling success window only over episodes that actually started
            # before that goal, crediting a 1/0 for reached/not-reached
            # instead of a permanent 1 the first time it ever landed.
            "milestones_at_start": sorted(data._episode_start_milestones),
            "milestones_newly_hit": [
                n for n, _ in data.last_milestone_payouts if n in GOAL_CANDIDATES
            ],
            "goals_live": live_goals,
            "goals_live_count": len(live_goals),
            "goals_peak_count": self._goals_peak_count,
            "goals_regressed": list(data.last_regressed),
            # No more soft/hard distinction (see Data.last_regressed) —
            # every regression is real now, so this is the same list.
            "goals_regressed_hard": list(data.last_regressed),
            # Vestigial, always empty — the curriculum-clear-exemption
            # bookkeeping this reported is gone (see Data.mark_goal_cleared).
            "goals_cleared": [],
            "menu_ticks": float(data.in_menu_ticks),
            "steps": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            "truncate_causes": sorted(truncate_causes or ()),
            "goal": data.goal,
            "stage": self.stage,
            "goal_success": bool(goal_success),
            "cleared_stage": cleared_stage,
            # Cumulative per-episode sum of each REWARD_COMPONENT_NAMES entry
            # so far (see Data.reward_component_vector) — read by
            # MilestoneCallback at episode end to log one TensorBoard chart
            # per reward sub-component (pokemon/reward/<name>).
            "reward_component_sums": dict(
                zip(REWARD_COMPONENT_NAMES, data.reward_component_vector())
            ),
            # Same, bucketed by which game mode (world/dialog/menu/battle/
            # cutscene) was active when each step's reward landed (see
            # Data.reward_mode_vector) — MilestoneCallback logs one
            # TensorBoard chart per mode (pokemon/reward_mode/<mode>).
            "reward_mode_sums": dict(
                zip(REWARD_MODE_NAMES, data.reward_mode_vector())
            ),
            # Always-on battle telemetry (see Data.battle_outcome_tally /
            # battle_step_count / _wild_decay_* / last_map_budget_trunc_at_start
            # for what each backs) -- read by MilestoneCallback for
            # pokemon/battle_*_rate, pokemon/battles_per_episode,
            # pokemon/battle_ticks_frac, pokemon/wild_encounter_decay_mean,
            # pokemon/wild_encounter_decay_floored_rate and
            # pokemon/map_budget_trunc_at_start_rate.
            "battle_outcome_tally": dict(data.battle_outcome_tally),
            "battle_step_count": data.battle_step_count,
            "wild_decay_sum": data._wild_decay_sum,
            "wild_decay_count": data._wild_decay_count,
            "wild_decay_floored_count": data._wild_decay_floored_count,
            "map_budget_trunc_at_start": (
                data.last_map_budget_trunc_at_start
                if truncate_causes and "map_budget" in truncate_causes
                else None
            ),
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
    goal: str = BADGE_GOALS[0],
    rank: int = 0,
    seed: int = 0,
    curriculum_mix: float = 0.3,
    curriculum_saves: list[str] | None = None,
    auto_advance: bool = True,
    stage: str | None = None,
    render_mode: str | None = None,
    n_workers: int = 1,
    collect_heatmap: bool = False,
    save_checkpoints: bool = False,
    data_kwargs: dict | None = None,
    emulator_kwargs: dict | None = None,
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
            save_checkpoints=save_checkpoints,
            data_kwargs=data_kwargs,
            emulator_kwargs=emulator_kwargs,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


# Re-export for trainers / curriculum.
__all__ = [
    "PokemonRedEnv",
    "make_pokemon_env",
    "VECTOR_DIM",
    "BADGE_GOALS",
]
