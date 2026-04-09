from dataclasses import dataclass
import io
from multiprocessing.synchronize import RLock
import os
from queue import Queue
from typing import Any

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
    ticks_per_step_after_press = 256
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
            self.__pyboy = PyBoy(f"rom.gb", sound_emulated=False, window=window_str)
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
        self.data.progress = 0
        self.data.last_progress = 0

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

            queue_logs.put_nowait(f"{reward:.5f}")

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
    ):
        self.use_sdl = is_evaluation_window

        model = get_model(device="cpu", files_lock=self.files_lock)
        model.load_state_dict(model_state_dict)
        model.eval()

        total_reward = 0.0

        memory, inputs = self.reset(dir="start")

        self.save_last_checkpoint("saves/last")

        count = 0
        while True:
            count += 1

            with torch.inference_mode():
                q = model(inputs)
                q = q.squeeze(0)

            action = int(torch.argmax(q).item())

            next_memory, next_inputs, reward, terminated, truncated = self.step(
                memory=memory, action=action
            )

            if self.is_milestone(memory):
                self.save_last_checkpoint("saves/last")
                queue_logs.put_nowait(
                    f"saved checkpoint {count}, with progress {self.data.progress}"
                )

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
