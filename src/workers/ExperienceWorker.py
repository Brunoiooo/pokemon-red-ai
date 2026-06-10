from collections import deque
from dataclasses import dataclass, field
import hashlib
from multiprocessing import Queue
from multiprocessing.connection import Pipe, PipeConnection
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, RLock
import os
from queue import Full
import random
import time
import traceback
from typing import Any
from pathlib import Path
import keyboard
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
    recv_conn: PipeConnection
    event_start: Event
    files_lock: RLock
    td_error_steps = 5
    start_save_chance = 0.1
    max_stuck_epsilon = 0.8
    min_stuck_epsilon = 0.5
    epsilon_decay_steps: int = field(default=500_000, init=False)
    _total_steps: int = field(default=0, init=False)
    _state_buffer: deque = field(default_factory=lambda: deque(maxlen=64), init=False)
    init_model_state_dict: dict[str, Any]
    max_episode_steps: int = 5000

    __last_save_path = "last"

    @property
    def last_save_path(self):
        return (
            self.__last_save_path
            if os.path.exists(f"saves/{self.__last_save_path}")
            else "start"
        )

    @property
    def random_save_path(self):
        return random.choice([p for p in Path("saves/").iterdir() if p.is_dir()]).name

    __model_state_dict: dict[str, Any] | None = None

    @property
    def model_state_dict(self):
        if self.__model_state_dict is None:
            self.__model_state_dict = self.init_model_state_dict

        return self.__model_state_dict

    @model_state_dict.setter
    def model_state_dict(self, value: dict[str, Any]):
        self.__model_state_dict = value
        self.__model = None

    __model: None | ModelPokemon = None

    @property
    def model(self):
        if not self.__model:
            self.__model = get_model(device="cpu", files_lock=self.files_lock)

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
            self.__emulator = Emulator(files_lock=self.files_lock)

        return self.__emulator

    def run(self, focused: bool = False):
        try:
            while self.event_start.is_set():
                self.run_game(focused=focused)

                if self.recv_conn.poll(0.1):
                    self.model_state_dict = self.recv_conn.recv()
        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.emulator.pyboy.stop(False)
            self.event_start.clear()
            self.queue_logs.put_nowait("Worker stopped.")

    def run_game(self, focused: bool = False):
        self._state_buffer.clear()
        memory, inputs = self.emulator.reset(dir=self.random_save_path)

        while self.event_start.is_set():
            action = self.get_action(inputs)

            if focused and keyboard.is_pressed("ctrl"):
                key = keyboard.read_key()
                time.sleep(0.1)
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

            self.put_to_queue_data(
                terminated=terminated,
                truncated=truncated,
            )

            if truncated:
                break

            memory, inputs = next_memory, next_inputs

            self.emulator.use_sdl = bool(self.window.get())

    def get_action(self, inputs: dict[float]):
        frac = min(1.0, self._total_steps / self.epsilon_decay_steps)
        epsilon = self.max_stuck_epsilon - frac * (self.max_stuck_epsilon - self.min_stuck_epsilon)
        self._total_steps += 1

        model_inputs = inputs
        if self._state_buffer:
            model_inputs = {**inputs, "state_sequence": torch.stack(list(self._state_buffer)).unsqueeze(0)}

        with torch.inference_mode():
            out = self.model(model_inputs)

        if isinstance(out, dict) and "z" in out:
            self._state_buffer.append(out["z"].squeeze(0).detach().cpu())

        if random.random() < epsilon:
            action = random.randint(0, len(self.emulator.buttons) - 1)
        else:
            q = out["q"] if isinstance(out, dict) else out
            q = q.squeeze(0)
            action = int(torch.argmax(q).item())

        return action

    def put_to_queue_data(self, terminated: bool, truncated: bool, flush: bool = False):
        if self.buffer.maxlen <= len(self.buffer) or terminated or truncated or flush:
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

                if truncated or terminated or flush:
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
