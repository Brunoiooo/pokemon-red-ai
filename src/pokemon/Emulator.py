from dataclasses import dataclass
import hashlib
import io
import os
import random
from typing import Literal

from pyboy import PyBoy
import torch

from pokemon.Data import Data
from pokemon.ModelPokemon import ModelPokemon


@dataclass
class Emulator:
    saves = "saves"
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
        ["left", "b"],
        ["right", "b"],
        ["up", "b"],
        ["down", "b"],
    ]
    ticks_per_step = 32
    ALL_BUTTONS = ["a", "b", "start", "select", "left", "right", "up", "down"]

    @property
    def random_save(self):
        return random.choice(os.listdir(self.saves))

    __pyboy: None | PyBoy = None

    __window: Literal["null", "SDL2"] = "null"

    @property
    def window(self):
        return self.__window

    @window.setter
    def window(self, window: Literal["null", "SDL2"]):
        if window == self.__window:
            return

        self.__window = window

        if self.__pyboy is None:
            return

        with io.BytesIO() as f:
            self.pyboy.save_state(f)
            f.seek(0)
            self.pyboy.stop(False)
            self.__pyboy = None
            self.pyboy.load_state(f)

    @property
    def pyboy(self):
        if self.__pyboy is None:
            self.__pyboy = PyBoy(f"rom.gb", sound_emulated=False, window=self.window)
            if self.__data is not None:
                self.__data.pyboy = self.__pyboy

        return self.__pyboy

    __data: None | Data = None

    @property
    def data(self):
        if self.__data is None:
            self.__data = Data(pyboy=self.pyboy)

        return self.__data

    def reset(self, random=True):
        if random:
            dir = self.random_save
        else:
            dir = "start"

        with open(f"{self.saves}/{dir}/checkpoint.state", "rb") as f:
            self.pyboy.load_state(f)

        self.data.clean()

        return (bytes(self.pyboy.memory[0:0x10000]), self.data.inputs())

    def step(self, memory: bytes, action: int):
        self.ticks(action)

        reward = self.data.reward(memory)

        terminated = self.data.terminated(memory)

        truncated = self.data.truncated()

        self.data.count(memory)

        return (
            bytes(self.pyboy.memory[0:0x10000]),
            self.data.inputs(),
            reward,
            terminated,
            truncated,
        )

    def ticks(self, action: int):
        for button in self.buttons[action]:
            self.pyboy.button_press(button)

        self.pyboy.tick(self.ticks_per_step / 2)

        for i in range(len(self.ALL_BUTTONS)):
            self.pyboy.button_release(self.ALL_BUTTONS[i])

        self.pyboy.tick(self.ticks_per_step / 2)

    def mask_action(self, action: int) -> int:
        if self.data.is_battle():
            return action

        if self.data.is_dialog():
            return 2 if action not in {0, 1, 2} else action

        if self.data.is_menu():
            return 2 if action not in {0, 1, 2, 5, 6, 7, 8} else action

        if self.data.is_blocked():
            return 0

        return action

    def evaluate_greedy(self, model: ModelPokemon, evaluate_greedy_times: int):
        total = 0.0
        for _ in range(evaluate_greedy_times):
            memory, inputs = self.reset(random=False)

            ep_ret = 0.0

            while True:
                with torch.inference_mode():
                    q = model(inputs)
                    q = q.squeeze(0)

                action = int(torch.argmax(q).item())

                next_memory, next_inputs, reward, terminated, truncated = self.step(
                    memory=memory, action=action
                )

                ep_ret += float(reward)

                if truncated:
                    break

                if terminated:
                    self.data.clean()

                next_memory, inputs = (next_memory, next_inputs)

            total += ep_ret

        self.pyboy.stop(False)

        return total / evaluate_greedy_times

    def save(self):
        path = f"{self.saves}/{self.get_hash()}"

        os.makedirs(path, exist_ok=True)
        with open(f"{path}/checkpoint.state", "wb") as f:
            self.pyboy.save_state(f)

    def get_hash(self):
        return hashlib.sha256(
            self.data.badges(self.pyboy.memory) + self.data.event_flags_data()
        ).hexdigest()
