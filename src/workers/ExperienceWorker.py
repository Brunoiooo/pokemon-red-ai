from collections import deque
from dataclasses import dataclass
import io
import math
from multiprocessing import Queue
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from queue import Full
import random
import time
import traceback
import torch
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import ModelPokemon


@dataclass
class ExperienceWorker:
    id: int
    event_stop: Event
    queue_logs: Queue
    connection_epsilon: Connection
    connection_state_dict: Connection
    queue_data: Queue
    window: Synchronized
    gamma: float
    epsilon: float = 1
    td_error_steps = 5

    __model: None | ModelPokemon = None

    @property
    def model(self):
        if not self.__model:
            emulator = Emulator()
            self.__model = ModelPokemon(
                len(emulator.data.data()), len(emulator.buttons)
            )
            emulator.pyboy.stop(False)

            self.model.eval()

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
            self.__emulator = Emulator(id=self.id)

        return self.__emulator

    def start(self):
        try:
            self.sync()

            memory, inputs = self.emulator.reset()

            while not self.event_stop.is_set():
                self.poll()

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

                if terminated:
                    self.emulator.save()

                memory, inputs = (
                    self.emulator.reset() if truncated else (next_memory, next_inputs)
                )

                with self.window.get_lock():
                    self.emulator.use_sdl = bool(self.window.value)

        except Exception as e:
            self.event_stop.set()
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.emulator.pyboy.stop(False)
            self.queue_logs.put_nowait("Worker stopped.")

    def sync(self):
        self.load_state_dict(self.connection_state_dict.recv())

        self.model.eval()

        self.epsilon = self.connection_epsilon.recv()

    def poll(self):
        if self.connection_state_dict.poll(0):
            self.load_state_dict(self.connection_state_dict.recv())
        if self.connection_epsilon.poll(0):
            self.epsilon = self.connection_epsilon.recv()

    def load_state_dict(self, recv: bytes):
        with torch.no_grad():
            self.model.load_state_dict(
                torch.load(io.BytesIO(recv), map_location="cpu"), strict=False
            )

    def get_action(self, inputs: dict[float]):
        if random.random() < self.epsilon:
            action = random.randint(0, len(self.emulator.buttons) - 1)
        else:
            with torch.inference_mode():
                q = self.model(inputs)
                q = q.squeeze(0)

            action = int(torch.argmax(q).item())

        return self.emulator.mask_action(action)

    def put_to_queue_data(self, terminated: bool, truncated: bool):
        if self.buffer.maxlen <= len(self.buffer):
            while len(self.buffer):
                reward, discount = 0.0, 1.0

                for item in self.buffer:
                    reward += discount * item["reward"]
                    discount *= self.gamma

                try:
                    if (
                        terminated
                        or truncated
                        or random.random() * len(self.buffer) <= math.fabs(reward)
                    ):
                        self.queue_data.put_nowait(
                            (
                                self.detach_to_cpu(self.buffer[0]["inputs"]),
                                self.buffer[0]["action"],
                                self.detach_to_cpu(self.buffer[-1]["next_inputs"]),
                                reward,
                                terminated,
                                len(self.buffer),
                            )
                        )
                except Full:
                    time.sleep(0.1)
                    pass

                if truncated:
                    self.buffer.popleft()
                else:
                    break

    def detach_to_cpu(self, inputs):
        out = {}

        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k == "continuous":
                    out[k] = v.detach().cpu().numpy().copy()
                else:
                    out[k] = int(v.item())
            else:
                out[k] = v

        return out
