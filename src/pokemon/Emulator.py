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


@dataclass
class Emulator:
    files_lock: RLock
    saves = "saves"
    buttons = [
        ["a"],
        ["b"],
        ["start"],
        ["select"],
        ["left"],
        ["right"],
        ["up"],
        ["down"],
    ]

    ticks_per_step_on_press = 16
    ticks_per_step_after_press = 16
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

        with self.files_lock:
            with open(f"{path}/checkpoint.state", "rb") as f:
                self.pyboy.load_state(f)

        self.data.clean()

        try:
            self.data.load(path=path)
        except FileNotFoundError:
            pass

        return (bytes(self.pyboy.memory[0:0x10000]), self.data.inputs())

    def step(self, memory: bytes, action: int):
        self.ticks(action)

        reward = self.data.reward(memory=memory, action=action)

        self.data.count(reward=reward, action=action, memory=memory)

        terminated = self.data.terminated(memory)

        truncated = self.data.truncated(memory)

        if truncated:
            reward = self.data.truncated_reward

        if self.is_milestone(memory=memory):
            self.data.clean()

        return (
            bytes(self.pyboy.memory[0:0x10000]),
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
            action = 0

            key = keyboard.read_key()
            if key == "up":
                action = 6
            elif key == "down":
                action = 7
            elif key == "left":
                action = 4
            elif key == "right":
                action = 5
            elif key == "a":
                action = 0
            elif key == "b":
                action = 1
            elif key == "space":
                action = 2
            elif key == "enter":
                action = 3
            elif key == "q":
                break
            elif key == "e":
                self.save_last_checkpoint("saves/manual")

            memory, inputs, reward, terminated, truncated = self.step(
                memory=memory, action=action
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

    def ticks(self, action: int):
        for button in self.buttons[action]:
            self.pyboy.button_press(button)

        self.pyboy.tick(self.ticks_per_step_on_press)

        for i in range(len(self.ALL_BUTTONS)):
            self.pyboy.button_release(self.ALL_BUTTONS[i])

        self.pyboy.tick(self.ticks_per_step_after_press)

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
                memory=memory, action=action
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

        self.pyboy.stop(False)

        return total_reward, count

    def save_last_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        with self.files_lock:
            with open(f"{path}/checkpoint.state", "wb") as f:
                self.pyboy.save_state(f)

        self.data.save(path=path)
