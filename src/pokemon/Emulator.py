from dataclasses import dataclass
import hashlib
import io
import os
from queue import Queue
from typing import Any

from pyboy import PyBoy
import torch

from pokemon.Data import Data
from pokemon.ModelPokemon import ModelPokemon, get_model
import keyboard
import time


@dataclass
class Emulator:
    saves = "saves"
    truncated_count_file_name = "truncated_count"
    terminated_count_file_name = "terminated_count"
    buttons = [
        [],
        ["a"],
        ["b"],
        ["start"],
        ["select"],
        ["left"],
        ["right"],
        ["up"],
        ["down"],
    ]
    ticks_per_step = 32
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
            self.__data = Data(pyboy=self.pyboy)

        return self.__data

    def reset(self, dir: str | None = None):
        path = f"{self.saves}/{dir}"

        with open(f"{path}/checkpoint.state", "rb") as f:
            self.pyboy.load_state(f)

        self.data.clean()

        return (bytes(self.pyboy.memory[0:0x10000]), self.data.inputs())

    def step(self, memory: bytes, action: int):
        self.ticks(action)

        reward = self.data.reward(memory)

        terminated = self.data.terminated(memory)

        truncated = self.data.truncated()

        self.data.count(memory, reward)

        if terminated:
            self.data.clean()

        return (
            bytes(self.pyboy.memory[0:0x10000]),
            self.data.inputs(),
            reward,
            terminated,
            truncated,
        )

    def auto_mode(self, queue_logs: Queue):
        self.use_sdl = True

        self.pyboy.set_emulation_speed(0)

        memory, inputs = self.reset(dir="start")

        while True:
            action = 0

            key = keyboard.read_key()
            if key == "up":
                action = 7
            elif key == "down":
                action = 8
            elif key == "left":
                action = 5
            elif key == "right":
                action = 6
            elif key == "a":
                action = 1
            elif key == "b":
                action = 2
            elif key == "space":
                action = 3
            elif key == "enter":
                action = 4
            elif key == "q":
                break

            memory, inputs, reward, terminated, truncated = self.step(
                memory=memory, action=action
            )

            if truncated:
                break

            queue_logs.put_nowait("==================================")
            queue_logs.put_nowait(f"Reward: {reward:.2f}")
            queue_logs.put_nowait(f"Terminated: {terminated}")
            queue_logs.put_nowait(f"Truncated: {truncated}")
            queue_logs.put_nowait("==================================")

            time.sleep(0.1)

        self.pyboy.stop(False)

    def ticks(self, action: int):
        for button in self.buttons[action]:
            self.pyboy.button_press(button)

        self.pyboy.tick(self.ticks_per_step / 2)

        for i in range(len(self.ALL_BUTTONS)):
            self.pyboy.button_release(self.ALL_BUTTONS[i])

        self.pyboy.tick(self.ticks_per_step / 2)

    def mask_action(self, action: int) -> int:

        return action

    def evaluate_greedy(
        self,
        model_state_dict: dict[str, Any],
        evaluate_greedy_times: int,
        queue_logs: Queue,
        is_debug: bool,
        is_evaluation_window: bool,
    ):
        self.use_sdl = is_evaluation_window

        model = get_model(device="cpu")
        model.load_state_dict(model_state_dict)
        model.eval()

        total_reward = 0.0
        for i in range(evaluate_greedy_times):
            memory, inputs = self.reset(dir="start")

            self.save_last_checkpoint("saves/last")

            while True:
                with torch.inference_mode():
                    q = model(inputs)
                    q = q.squeeze(0)

                action = int(torch.argmax(q).item())

                next_memory, next_inputs, reward, terminated, truncated = self.step(
                    memory=memory, action=action
                )

                total_reward += reward

                if is_debug:
                    queue_logs.put_nowait(
                        f"Episode: {i + 1}, Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}"
                    )

                if terminated:
                    self.save_last_checkpoint("saves/last")

                if truncated:
                    break

                memory, inputs = (next_memory, next_inputs)

        self.pyboy.stop(False)

        return total_reward / evaluate_greedy_times

    def save_last_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/checkpoint.state", "wb") as f:
            self.pyboy.save_state(f)
