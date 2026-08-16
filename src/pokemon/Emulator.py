from dataclasses import dataclass
import io
from multiprocessing.synchronize import RLock
import os
from typing import Any
import uuid

import numpy as np
from pyboy import PyBoy

from pokemon.Data import Data


DURATION_BINS = [16, 32, 64, 128, 255]
N_ACTIONS = 9       # 8 buttons + none
N_META_ACTIONS = N_ACTIONS * len(DURATION_BINS)   # 45

# Every RAM address Data.py reads falls within WRAM bank 0xC000-0xE000 (highest
# literal is stored_pokemon_* around 0xDAB0, plus a menu-index offset bounded by
# the in-game box/bag size of 20 items — max observed ~0xDD69, well under this
# cutoff). Snapshotting only this window instead of the full 64KB address space
# avoids copying ROM banks, echo RAM, OAM, and I/O registers that nothing reads.
MEMORY_SNAPSHOT_END = 0xE000


@dataclass
class Emulator:
    files_lock: RLock
    saves = "saves"
    buttons = [
        ["a"],       # 0
        ["b"],       # 1
        ["start"],   # 2
        ["select"],  # 3
        ["left"],    # 4
        ["right"],   # 5
        ["up"],      # 6
        ["down"],    # 7
        [],          # 8 — none (no button pressed)
    ]

    ALL_BUTTONS = ["a", "b", "start", "select", "left", "right", "up", "down"]

    # button_press()/button_release() only take effect on the next tick() —
    # releasing at the end of a step and pressing again at the start of the
    # next one (with zero ticks in between) never actually produces a
    # released frame, so repeating the same button across consecutive steps
    # looks like one continuous hold to the game. Gen-1 input handling (e.g.
    # dialog advance) triggers on a fresh press edge, not on "still held", so
    # a held A stops advancing text after the first press. Ticking a couple
    # of frames with everything released between action holds restores that
    # edge.
    RELEASE_GAP_TICKS = 2

    __use_sdl: bool = False
    # SDL layout (set by PokemonRedEnv before first pyboy access).
    sdl_scale: int = 3
    window_x: int | None = None
    window_y: int | None = None
    window_title: str | None = None

    @property
    def use_sdl(self) -> bool:
        return self.__use_sdl

    @use_sdl.setter
    def use_sdl(self, use_sdl: bool):
        if use_sdl == self.__use_sdl:
            return

        self.__use_sdl = bool(use_sdl)

        if self.__pyboy is None:
            return

        with io.BytesIO() as f:
            self.pyboy.save_state(f)
            f.seek(0)
            self.pyboy.stop(False)
            self.__pyboy = None
            self.pyboy.load_state(f)

    __pyboy: None | PyBoy = None

    def _apply_sdl_layout(self) -> None:
        if self.__pyboy is None or not self.use_sdl:
            return
        from utils.gui_layout import apply_pyboy_window

        apply_pyboy_window(
            self.__pyboy,
            x=self.window_x,
            y=self.window_y,
            title=self.window_title,
        )

    @property
    def pyboy(self):
        if self.__pyboy is None:
            window_str = "SDL2" if self.use_sdl else "null"
            kwargs: dict[str, Any] = dict(
                sound_emulated=False, window=window_str, cgb=False
            )
            if self.use_sdl:
                kwargs["scale"] = max(1, int(self.sdl_scale))
            self.__pyboy = PyBoy(f"rom.gb", **kwargs)
            if not self.use_sdl:
                self.__pyboy.set_emulation_speed(0)
            else:
                self._apply_sdl_layout()
            if self.__data is not None:
                self.__data.pyboy = self.__pyboy

        return self.__pyboy

    __data: None | Data = None

    @property
    def data(self):
        if self.__data is None:
            self.__data = Data(pyboy=self.pyboy, files_lock=self.files_lock)

        return self.__data

    def reset(self, dir: str | None = None):
        path = f"{self.saves}/{dir}"

        try:
            with self.files_lock:
                with open(f"{path}/checkpoint.state", "rb") as f:
                    self.pyboy.load_state(f)
        except Exception:
            with self.files_lock:
                with open(f"{self.saves}/start/checkpoint.state", "rb") as f:
                    self.pyboy.load_state(f)
            path = f"{self.saves}/start"

        self.data.clean()

        try:
            self.data.load(path=path)
        except Exception:
            pass

        return (bytes(self.pyboy.memory[0:MEMORY_SNAPSHOT_END]), self.data.inputs())

    def step(self, memory: bytes, meta_action: int, render_each: bool = False):
        action_idx = meta_action // len(DURATION_BINS)
        duration = DURATION_BINS[meta_action % len(DURATION_BINS)]

        self.ticks(meta_action, render_each=render_each)

        milestone, step = self.data.reward(memory=memory, action=action_idx)

        min_d = DURATION_BINS[0]
        if step > 0:
            step *= min_d / duration
        elif step < 0:
            step *= duration / min_d

        reward = milestone + step

        self.data.count(reward=reward, action=action_idx, memory=memory, duration=duration)

        terminated = self.data.terminated(memory)

        truncated = self.data.truncated(memory)

        if truncated:
            reward = self.data.truncated_reward
            milestone = reward
            step = 0.0

        self.last_milestone = milestone
        self.last_step = step

        # Do NOT call data.clean() on event/badge flips. That wiped visit
        # counts, loop streak, and curriculum _cleared_goals mid-episode,
        # which let the policy farm door-warp flag noise then reset anti-loop.

        return (
            bytes(self.pyboy.memory[0:MEMORY_SNAPSHOT_END]),
            self.data.inputs(),
            reward,
            terminated,
            truncated,
        )

    def step_discrete(
        self,
        memory: bytes,
        action_idx: int,
        duration: int = 16,
        render_each: bool = False,
    ):
        """PPO-friendly step: discrete button + fixed hold duration (no meta-actions)."""
        action_idx = int(action_idx) % N_ACTIONS
        duration = max(1, int(duration))

        self._skip_locked_frames(duration)
        self._skip_battle_messages(duration)
        self._press_action(action_idx)

        if render_each:
            for _ in range(duration):
                self.pyboy.tick(1, render=True, sound=False)
        else:
            self.pyboy.tick(duration, render=self.use_sdl, sound=False)

        for button in self.ALL_BUTTONS:
            self.pyboy.button_release(button)
        self._release_gap(render_each=render_each)
        self._skip_battle_messages(duration)

        milestone, step = self.data.reward(memory=memory, action=action_idx)

        min_d = DURATION_BINS[0]
        if step > 0:
            step *= min_d / duration
        elif step < 0:
            step *= duration / min_d

        reward = milestone + step
        self.data.count(
            reward=reward, action=action_idx, memory=memory, duration=duration
        )

        terminated = self.data.terminated(memory)
        truncated = self.data.truncated(memory)

        if truncated:
            reward = self.data.truncated_reward
            milestone = reward
            step = 0.0

        self.last_milestone = milestone
        self.last_step = step

        # Intentionally no data.clean() here — see step() above.

        return (
            bytes(self.pyboy.memory[0:MEMORY_SNAPSHOT_END]),
            self.data.inputs(),
            float(reward),
            bool(terminated),
            bool(truncated),
        )

    def is_new_episode(self, memory: bytes):
        return ()

    # Absolute safety-valve cap on a single _skip_locked_frames() call, in
    # emulator frames (~34s at 60fps) -- only reached if the per-call cap
    # argument passed in is left at its default. See _skip_locked_frames.
    MAX_CUTSCENE_SKIP = 2048

    def _skip_locked_frames(self, max_frames: int = MAX_CUTSCENE_SKIP) -> int:
        """Jump straight through (a bounded chunk of) a forced-walk / warp-
        transition window.

        Reads pokered's own frame-exact countdown registers (see
        Data.cutscene_skip_frames) so a cutscene collapses into a few big
        pyboy.tick() calls instead of being polled away frame_skip-frames-
        at-a-time across many separate env.step() calls -- every one of
        which discards its action anyway per _press_action.

        max_frames caps a single call to roughly the current step's own
        duration (train_ppo.py's --frame-skip, 16 by default) rather than a
        large fixed constant: training runs several PokemonRedEnv workers as
        separate SubprocVecEnv processes, and step() is synchronous across
        all of them -- the batch can't advance until every worker's step()
        returns. A worker mid-cutscene ticking hundreds/thousands of frames
        in one call is a massive outlier next to every other worker's
        ~frame_skip-sized step that same round, so it single-handedly stalls
        the whole batch (visible as a global FPS drop and workers freezing
        then resuming together in GUI mode) for as long as it keeps winning
        that outlier race. Capping each call to the same order of magnitude
        as a normal step keeps a locked worker statistically indistinguishable
        from any other worker's step; is_cutscene_locked (checked again at
        the top of the next step_discrete()/ticks() call) keeps consuming the
        remainder across as many subsequent env.step() calls as it takes.
        """
        max_frames = max(1, int(max_frames))
        total = 0
        while total < max_frames and self.data.is_cutscene_locked(self.pyboy.memory):
            n = self.data.cutscene_skip_frames(self.pyboy.memory)
            n = max(1, min(n, max_frames - total))
            self.pyboy.tick(n, render=self.use_sdl, sound=False)
            total += n
        return total

    # Absolute safety-valve cap on a single _skip_battle_messages() call, in
    # emulator frames -- only reached if the per-call cap argument passed in
    # is left at its default. See _skip_battle_messages.
    MAX_BATTLE_MESSAGE_SKIP = 600

    def _skip_battle_messages(self, max_frames: int = MAX_BATTLE_MESSAGE_SKIP) -> int:
        """EXPERIMENTAL: auto-mash A through non-actionable battle text.

        Data.is_battle_message() ("X used TACKLE!", "It's super effective!",
        "Wild RATTATA fainted!", ...) is the battle-mode analog of plain
        dialog text, and Data.legal_action_mask() already hard-restricts the
        model to A/B for these frames (see its docstring) -- there is no
        decision being skipped by clearing them here instead of spending
        real env.step() calls (each paying the full reward/observation-
        building cost) mashing A into them one frame_skip window at a time,
        same reasoning as _skip_locked_frames for overworld cutscenes,
        including the max_frames per-call cap (see its docstring for why:
        SubprocVecEnv's synchronous step() means a big outlier call here
        stalls every other worker until it returns). Unlike that method,
        pokered exposes no countdown register for battle text (animations.asm's
        DelayFrames busy-waits internally, nothing readable from RAM) -- so
        this polls a press/tick/release/tick cycle instead of jumping
        straight to a known frame count.
        """
        max_frames = max(1, int(max_frames))
        total = 0
        while total < max_frames and self.data.is_battle_message(self.pyboy.memory):
            self.pyboy.button_press("a")
            self.pyboy.tick(2, render=self.use_sdl, sound=False)
            self.pyboy.button_release("a")
            self.pyboy.tick(2, render=self.use_sdl, sound=False)
            total += 4
        return total

    def _press_action(self, action_idx: int) -> None:
        # During forced walks the engine ignores (or may override) input — skip
        # presses so we don't disturb scripted sequences. Dialog/menu still need A/B.
        if self.data.is_cutscene_locked(self.pyboy.memory):
            return
        for button in self.buttons[action_idx]:
            self.pyboy.button_press(button)

    def _release_gap(self, render_each: bool = False) -> None:
        """Tick a couple of frames with every button released.

        button_release() only takes effect on the *next* tick() — without
        this, releasing at the end of one step and pressing the same button
        again at the start of the next (zero ticks in between) never
        produces an actually-released frame, so repeating an action across
        steps reads as one continuous hold. See RELEASE_GAP_TICKS.
        """
        if render_each:
            for _ in range(self.RELEASE_GAP_TICKS):
                self.pyboy.tick(1, render=True, sound=False)
        else:
            self.pyboy.tick(self.RELEASE_GAP_TICKS, render=self.use_sdl, sound=False)

    def ticks(self, meta_action: int, render_each: bool = False):
        action_idx = meta_action // len(DURATION_BINS)
        duration = DURATION_BINS[meta_action % len(DURATION_BINS)]

        self._skip_locked_frames(duration)
        self._press_action(action_idx)

        if render_each:
            for _ in range(duration):
                self.pyboy.tick(1, render=True, sound=False)
        else:
            # Observations are built purely from RAM (Data.py), never from the
            # framebuffer, so skip PyBoy's render pass entirely unless the SDL
            # window is actually showing (PyBoy's own docs recommend render=False
            # for AI training, calling it a substantial and otherwise-needless cost).
            self.pyboy.tick(duration, render=self.use_sdl, sound=False)

        for button in self.ALL_BUTTONS:
            self.pyboy.button_release(button)
        self._release_gap(render_each=render_each)

    def save_last_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        with self.files_lock:
            with open(f"{path}/checkpoint.state", "wb") as f:
                self.pyboy.save_state(f)

        self.data.save(path=path)
