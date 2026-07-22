from collections import deque
from dataclasses import dataclass
import io
from multiprocessing.synchronize import RLock
import os
from queue import Queue
from typing import Any
import uuid

import numpy as np
from pyboy import PyBoy
import torch

from pokemon.Data import Data
from pokemon.ModelPokemon import get_model
import keyboard
import time


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

    __use_sdl: bool = False

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

    @property
    def pyboy(self):
        if self.__pyboy is None:
            window_str = "SDL2" if self.use_sdl else "null"
            self.__pyboy = PyBoy(f"rom.gb", sound_emulated=False, window=window_str, cgb=False)
            if not self.use_sdl:
                self.__pyboy.set_emulation_speed(0)
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

        if self.is_milestone(memory=memory):
            self.data.clean()

        return (
            bytes(self.pyboy.memory[0:MEMORY_SNAPSHOT_END]),
            self.data.inputs(),
            reward,
            terminated,
            truncated,
        )

    def is_milestone(self, memory: bytes):
        return 0 < self.data.reward_event_flags(memory) or 0 < self.data.reward_badges(
            memory
        )

    def is_new_episode(self, memory: bytes):
        return ()

    def auto_mode(self, queue_logs: Queue):
        self.use_sdl = True

        self.pyboy.set_emulation_speed(0)

        memory, inputs = self.reset(dir="start")

        flags = self.data.map_id(self.pyboy.memory), self.data.event_flags_data(
            self.pyboy.memory
        )

        while True:
            action_idx = 8  # none

            key = keyboard.read_key()
            if key == "up":
                action_idx = 6
            elif key == "down":
                action_idx = 7
            elif key == "left":
                action_idx = 4
            elif key == "right":
                action_idx = 5
            elif key == "a":
                action_idx = 0
            elif key == "b":
                action_idx = 1
            elif key == "space":
                action_idx = 2
            elif key == "enter":
                action_idx = 3
            elif key == "q":
                break
            elif key == "e":
                self.save_last_checkpoint("saves/manual")

            meta_action = action_idx * len(DURATION_BINS) + 0  # 16-tick bin

            memory, inputs, reward, terminated, truncated = self.step(
                memory=memory, meta_action=meta_action
            )

            # queue_logs.put_nowait(
            #     f"useless_count: {self.data.useless_count, len(self.data.visited_screens)}"
            # )
            # queue_logs.put_nowait(f"visited_dialogs: {self.data.visited_dialogs}")
            # queue_logs.put_nowait(
            #     f"sprite_data_ids: {self.data.sprite_data_ids(self.pyboy.memory)}"
            # )
            # queue_logs.put_nowait(
            #     f"visited_dialogs.get: {self.data.visited_dialogs.get(self.data.get_dialog(), 0)}"
            # )
            queue_logs.put_nowait(f"key: {key}, reward: {reward:.5f}")
            queue_logs.put_nowait(
                f"is_dialog: {self.data.is_dialog(self.pyboy.memory)}, is_world: {self.data.is_world(self.pyboy.memory)}, is_menu: {self.data.is_menu(self.pyboy.memory)}, is_battle: {self.data.is_battle(self.pyboy.memory)}"
            )
            queue_logs.put_nowait(f"visited_maps: {self.data.visited_maps}")
            queue_logs.put_nowait(
                f"visited_dialogs: {self.data.visited_dialogs.get(self.data.get_dialog(), 0)}"
            )
            queue_logs.put_nowait(
                f"visited_positions: {self.data.visited_positions.get(self.data.get_position(), 0)}"
            )

            if truncated:
                break

            if flags != (
                self.data.map_id(self.pyboy.memory),
                self.data.event_flags_data(self.pyboy.memory),
            ):
                flags = self.data.map_id(self.pyboy.memory), self.data.event_flags_data(
                    self.pyboy.memory
                )
                queue_logs.put_nowait(
                    f"{self.data.position_x(self.pyboy.memory), self.data.position_y(self.pyboy.memory),flags}"
                )

            # queue_logs.put_nowait(f"{reward:.5f}")

            time.sleep(0.1)

        self.pyboy.stop(False)

    def ticks(self, meta_action: int, render_each: bool = False):
        action_idx = meta_action // len(DURATION_BINS)
        duration = DURATION_BINS[meta_action % len(DURATION_BINS)]

        for button in self.buttons[action_idx]:
            self.pyboy.button_press(button)

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

    def evaluate_greedy(
        self,
        model_state_dict: dict[str, Any],
        queue_logs: Queue,
        is_debug: bool,
        is_evaluation_window: bool,
        save_name: str = "last",
    ):
        checkpoint = f"saves/{save_name}"

        self.use_sdl = is_evaluation_window

        model = get_model(device="cpu", files_lock=self.files_lock)
        model.load_state_dict(model_state_dict)
        model.eval()

        total_reward = 0.0

        memory, inputs = self.reset(dir="start")

        self.save_last_checkpoint(checkpoint)

        state_buffer = deque(maxlen=64)
        count = 0
        while True:
            count += 1

            model_inputs = inputs
            if state_buffer:
                model_inputs = {
                    **inputs,
                    "state_sequence": torch.stack(list(state_buffer)).unsqueeze(0),
                }

            with torch.inference_mode():
                out = model(model_inputs)
                if isinstance(out, dict) and "z" in out:
                    state_buffer.append(out["z"].squeeze(0).detach())
                q = out["q"] if isinstance(out, dict) else out
                q = q.squeeze(0)

            action = int(torch.argmax(q).item())

            next_memory, next_inputs, reward, terminated, truncated = self.step(
                memory=memory, meta_action=action
            )

            if (
                self.data.is_world(self.pyboy.memory)
                and self.data.visited_positions.get(self.data.get_position(), 0) == 0
            ):
                self.save_last_checkpoint(checkpoint)
                queue_logs.put_nowait(f"saved checkpoint {count}")

            if terminated:
                queue_logs.put_nowait(
                    f"Evaluation terminated successfully with total reward: {total_reward:.2f}"
                )

            total_reward += reward

            if truncated:
                break

            memory, inputs = (next_memory, next_inputs)

        badges_collected = sum(self.data.badges(self.pyboy.memory))

        self.pyboy.stop(False)

        return total_reward, count, badges_collected

    def save_last_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        with self.files_lock:
            with open(f"{path}/checkpoint.state", "wb") as f:
                self.pyboy.save_state(f)

        self.data.save(path=path)
