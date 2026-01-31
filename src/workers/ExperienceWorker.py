from collections import deque
from dataclasses import dataclass
import hashlib
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
import os
from queue import Full
import random
import time
import traceback
from typing import Any
import numpy as np
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
    gru_hidden: int
    gru_layers: int
    epsilon: float = 1.0
    td_error_steps = 10
    start_save_chance = 0.25
    max_stuck_epsilon = 0.25

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
            self.__model = get_model(
                device="cpu", gru_hidden=self.gru_hidden, gru_layers=self.gru_layers
            )

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
            self.queue_logs.put_nowait(f"Worker started epsilon: {self.epsilon:.3f}.")

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
            self.queue_logs.put_nowait(
                f"Worker stopped epsilon: {self.epsilon:.3f} | {'success' if terminated else 'failure'}"
            )

    def get_action(self, inputs: dict[float]):
        if (
            random.random()
            < self.emulator.data.useless_count
            / self.emulator.data.max_useless_count
            * self.max_stuck_epsilon
        ):
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
                    inputs0 = self.detach_to_cpu(self.buffer[0]["inputs"])
                    next_inputs = self.detach_to_cpu(self.buffer[-1]["next_inputs"])
                    action = self.buffer[0]["action"]

                    hkey = self.transition_hash(
                        inputs0,
                        action,
                        next_inputs,
                        reward,
                        terminated,
                        truncated,
                        eps=1e-3,
                    )

                    self.queue_data.put_nowait(
                        (
                            inputs0,
                            action,
                            next_inputs,
                            reward,
                            terminated,
                            truncated,
                            len(self.buffer),
                            hkey,
                        )
                    )
                except Full:
                    time.sleep(0.01)
                    pass

                if truncated or terminated:
                    self.buffer.popleft()
                else:
                    break

    def _stable_hash_obj(self, x, eps: float | None = 1e-3, h=None):
        if h is None:
            h = hashlib.blake2b(digest_size=16)

        # torch tensor
        if torch is not None and hasattr(x, "detach"):
            x = x.detach().to("cpu")
            x = x.contiguous().numpy()

        # numpy array
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            if eps is not None and np.issubdtype(a.dtype, np.floating):
                a = np.round(a / eps).astype(np.int32)
            h.update(b"nd")
            h.update(str(a.shape).encode("utf-8"))
            h.update(str(a.dtype).encode("utf-8"))
            h.update(a.tobytes())
            return h

        # dict
        if isinstance(x, dict):
            h.update(b"{")
            for k in sorted(x.keys(), key=lambda k: repr(k)):
                self._stable_hash_obj(k, eps=eps, h=h)
                h.update(b":")
                self._stable_hash_obj(x[k], eps=eps, h=h)
                h.update(b",")
            h.update(b"}")
            return h

        # list/tuple
        if isinstance(x, (list, tuple)):
            h.update(b"[")
            for it in x:
                self._stable_hash_obj(it, eps=eps, h=h)
                h.update(b",")
            h.update(b"]")
            return h

        # primitives
        if isinstance(x, (str, int, float, bool)) or x is None:
            h.update(b"p")
            h.update(repr(x).encode("utf-8"))
            return h

        # fallback
        h.update(b"o")
        h.update(repr(x).encode("utf-8"))
        return h

    def transition_hash(
        self,
        inputs,
        action,
        next_inputs,
        reward,
        terminated,
        truncated,
        eps: float | None = 1e-3,
    ) -> str:
        h = hashlib.blake2b(digest_size=16)
        self._stable_hash_obj(inputs, eps=eps, h=h)
        h.update(b"|a|")
        self._stable_hash_obj(action, eps=eps, h=h)
        h.update(b"|n|")
        self._stable_hash_obj(next_inputs, eps=eps, h=h)
        h.update(b"|r|")
        self._stable_hash_obj(reward, eps=eps, h=h)
        h.update(b"|t|")
        self._stable_hash_obj(terminated, eps=eps, h=h)
        h.update(b"|tr|")
        self._stable_hash_obj(truncated, eps=eps, h=h)
        return h.hexdigest()

    def detach_to_cpu(self, inputs):
        out = {}

        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()

                if k in self.model.FLOAT_INPUTS:
                    out[k] = v.numpy().copy()
                else:
                    if v.numel() == 1:
                        out[k] = int(v.item())
                    else:
                        out[k] = v.tolist()
            else:
                out[k] = v

        return out
