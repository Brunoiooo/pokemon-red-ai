from dataclasses import dataclass
import hashlib
import io
import os
from queue import Queue
import random

from pyboy import PyBoy
import torch

from pokemon.Data import Data
from pokemon.ModelPokemon import ModelPokemon
import keyboard
import time


@dataclass
class Emulator:
    id: int = 0
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

    def __post_init__(self):
        path = f"{self.saves}/{self.id}/start"

        if not os.path.exists(f"{path}/checkpoint.state"):
            os.makedirs(path, exist_ok=True)

            with open(f"{self.saves}/start/checkpoint.state", "rb") as f:
                data = f.read()

            with open(f"{path}/checkpoint.state", "wb") as f:
                f.write(data)

    @property
    def random_save(self):
        return random.choice(os.listdir(f"{self.saves}/{self.id}"))

    __pyboy: None | PyBoy = None

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
        if dir is None:
            dir = self.random_save

        with open(f"{self.saves}/{self.id}/{dir}/checkpoint.state", "rb") as f:
            self.pyboy.load_state(f)

        self.data.clean()

        return (bytes(self.pyboy.memory[0:0x10000]), self.data.inputs())

    def step(self, memory: bytes, action: int):
        self.ticks(action)

        reward = self.data.reward(memory)

        terminated = self.data.terminated(memory)

        if terminated:
            self.data.clean()

        truncated = self.data.truncated()

        self.data.count(memory)

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
            queue_logs.put_nowait(
                f"is_world: {int(self.data.is_world())} is_battle: {int(self.data.is_battle())} is_dialog: {int(self.data.is_dialog())} is_menu: {int(self.data.is_menu())} is_blocked: {int(self.data.is_blocked())}"
            )
            queue_logs.put_nowait(f"Badges: {self.data.badges(self.pyboy.memory)}")
            queue_logs.put_nowait(
                f"Event Flags: {self.data.event_flags_data(self.pyboy.memory)}"
            )
            queue_logs.put_nowait(f"Battle Count: {self.data.battle_count}")
            queue_logs.put_nowait(f"Position: {self.data.get_position()}")
            queue_logs.put_nowait(f"Menu Count: {self.data.menu_count}")
            queue_logs.put_nowait(
                f"Visited Dialogs Count: {self.data.visited_dialogs_count.get(self.data.dialog_id(self.pyboy.memory), 0)}"
            )
            queue_logs.put_nowait(
                f"Visited Positions Count: {self.data.visited_positions_count.get(self.data.get_position(), 0)}"
            )
            queue_logs.put_nowait("==================================")

            time.sleep(0.25)

        self.pyboy.stop(False)

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

    def evaluate_greedy(
        self,
        model: ModelPokemon,
        evaluate_greedy_times: int,
        queue_logs: Queue,
        is_debug: bool,
        is_evaluation_window: bool,
    ):

        self.use_sdl = is_evaluation_window

        total_episodes = 0
        for i in range(evaluate_greedy_times):
            memory, inputs = self.reset(dir="start")

            while True:
                with torch.inference_mode():
                    q = model(inputs)
                    q = q.squeeze(0)

                action = int(torch.argmax(q).item())

                next_memory, next_inputs, reward, terminated, truncated = self.step(
                    memory=memory, action=action
                )

                if is_debug:
                    queue_logs.put_nowait(
                        f"Episode: {i + 1}, Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}"
                    )

                if truncated:
                    break

                memory, inputs = (next_memory, next_inputs)

            total_episodes += self.data.badges(self.pyboy.memory).count(
                1
            ) + self.data.event_flags_data(self.pyboy.memory).count(1)

        self.pyboy.stop(False)

        return total_episodes / evaluate_greedy_times

    def save(self):
        path = f"{self.saves}/{self.id}/{self.get_hash()}"

        os.makedirs(path, exist_ok=True)
        with open(f"{path}/checkpoint.state", "wb") as f:
            self.pyboy.save_state(f)

    def get_hash(self):
        return hashlib.sha256(
            bytes(
                self.data.badges(self.pyboy.memory)
                + self.data.event_flags_data(self.pyboy.memory)
            )
        ).hexdigest()
