from collections import deque
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
import os
from queue import Full
import random
import time
import traceback
from typing import Any
import torch
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import ModelPokemon, get_model


@dataclass
class ExperienceWorker:
    queue_logs: Queue
    queue_data: Queue
    window: Synchronized
    gamma: float
    model_state_dict: dict[str, Any]
    epsilon: float
    td_error_steps = 5
    start_save_chance = 0.0

    __last_save_path = "last"

    @property
    def last_save_path(self):
        return (
            self.__last_save_path
            if os.path.exists(f"saves/{self.__last_save_path}")
            else "start"
        )

    __model: None | ModelPokemon = None

    @property
    def model(self):
        if not self.__model:
            self.__model = get_model(device="cpu")

            self.__model.load_state_dict(self.model_state_dict)

            self.__model.eval()

        return self.__model

    __buffer: deque | None = None

    @property
    def buffer(self):
        if self.__buffer is None:
            self.__buffer = deque(maxlen=self.td_error_steps)

        return self.__buffer

    __emulator: Emulator | None = None

    @property
    def emulator(self):
        if self.__emulator is None:
            self.__emulator = Emulator()

        return self.__emulator

    def run(self):
        try:
            self.queue_logs.put_nowait(f"Worker started epsilon: {self.epsilon}.")

            memory, inputs = self.emulator.reset(
                dir=(
                    "start"
                    if random.random() < self.start_save_chance
                    else self.last_save_path
                )
            )

            while True:
                action = self.get_action(inputs)

                next_memory, next_inputs, reward, terminated, truncated = (
                    self.emulator.step(memory=memory, action=action)
                )

                self.buffer.append(
                    {
                        "inputs": self.detach_to_cpu(inputs),
                        "action": action,
                        "next_inputs": self.detach_to_cpu(next_inputs),
                        "reward": reward,
                    }
                )

                self.put_to_queue_data(terminated=terminated, truncated=truncated)

                if truncated:
                    break

                memory, inputs = next_memory, next_inputs

                self.emulator.use_sdl = bool(self.window.get())

        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.emulator.pyboy.stop(False)
            self.queue_logs.put_nowait(f"Worker stopped epsilon: {self.epsilon}.")

    def get_action(self, inputs: dict[float]):
        if random.random() < self.epsilon:
            action = random.randint(0, len(self.emulator.buttons) - 1)
        else:
            with torch.inference_mode():
                q = self.model(inputs)
                q = q.squeeze(0)

            action = int(torch.argmax(q).item())

        return action

    def put_to_queue_data(self, terminated: bool, truncated: bool):
        if self.buffer.maxlen <= len(self.buffer):
            while len(self.buffer):
                reward, discount = 0.0, 1.0

                for item in self.buffer:
                    reward += discount * item["reward"]
                    discount *= self.gamma

                try:
                    self.queue_data.put_nowait(
                        (
                            self.detach_to_cpu(self.buffer[0]["inputs"]),
                            self.buffer[0]["action"],
                            self.detach_to_cpu(self.buffer[-1]["next_inputs"]),
                            reward,
                            terminated,
                            truncated,
                            len(self.buffer),
                        )
                    )
                except Full:
                    time.sleep(0.01)
                    pass

                if truncated:
                    self.buffer.popleft()
                else:
                    break

    def detach_to_cpu(self, inputs):
        out = {}

        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()

                if k == "continuous":
                    out[k] = v.numpy().copy()
                else:
                    if v.numel() == 1:
                        out[k] = int(v.item())
                    else:
                        out[k] = v.tolist()
            else:
                out[k] = v

        return out
