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
from pokemon.Emulator import Emulator, N_META_ACTIONS, DURATION_BINS
from pokemon.ModelPokemon import ModelPokemon, get_model

try:
    from curriculum_config import get_checkpoint_for_episode, get_current_stage
    CURRICULUM_ENABLED = True
except ImportError:
    CURRICULUM_ENABLED = False


@dataclass
class ExperienceWorker:
    queue_logs: Queue
    queue_data: Queue
    stats_queue: Queue
    window: Synchronized
    gamma: float
    recv_conn: PipeConnection
    event_start: Event
    files_lock: RLock
    init_model_state_dict: dict[str, Any]
    worker_id: int = 0
    max_episode_steps: int = 5000

    td_error_steps = 10
    start_save_chance = 0.8
    max_stuck_epsilon = 0.15
    min_stuck_epsilon = 0.0

    epsilon_decay_steps: int = field(default=500_000, init=False)
    _total_steps: int = field(default=0, init=False)
    _last_epsilon: float = field(default=0.0, init=False)
    _state_buffer: deque = field(default_factory=lambda: deque(maxlen=64), init=False)
    _last_curriculum_stage: str | None = field(default=None, init=False)

    # Checkpoint state
    _checkpoint_attempts: int = field(default=0, init=False)
    _fallback_to_start: bool = field(default=False, init=False)
    _new_checkpoint_this_episode: bool = field(default=False, init=False)

    @property
    def worker_checkpoint_path(self) -> str:
        return f"worker_{self.worker_id}"

    @property
    def random_save_path(self):
        if not self._fallback_to_start and os.path.exists(f"saves/{self.worker_checkpoint_path}"):
            return self.worker_checkpoint_path

        if CURRICULUM_ENABLED:
            checkpoint = get_checkpoint_for_episode(self._total_steps)
            if checkpoint:
                current_stage = get_current_stage(self._total_steps)
                if current_stage != self._last_curriculum_stage:
                    self._last_curriculum_stage = current_stage
                    self.queue_logs.put(f"Curriculum: Advanced to {current_stage} (using checkpoint: {checkpoint})")
                return checkpoint

        return "start"

    def _save_checkpoint(self, reason: str):
        try:
            path = f"saves/{self.worker_checkpoint_path}"
            self.emulator.save_last_checkpoint(path)
            self._checkpoint_attempts = 0
            self._fallback_to_start = False
            self._new_checkpoint_this_episode = True
            self.queue_logs.put(f"Worker {self.worker_id} checkpoint: {reason} | attempts reset to 0")
        except Exception as e:
            self.queue_logs.put(f"Worker {self.worker_id} failed to save checkpoint: {e}")

    def _handle_truncate(self):
        if self._new_checkpoint_this_episode:
            return

        self._checkpoint_attempts += 1
        if self._checkpoint_attempts >= 10:
            self._fallback_to_start = True
            self._checkpoint_attempts = 0
            self.queue_logs.put(f"Worker {self.worker_id}: 10 failed attempts → fallback to start")
        else:
            self.queue_logs.put(f"Worker {self.worker_id}: attempt {self._checkpoint_attempts}/10 failed")

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

            self.__model.train()

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
        self._new_checkpoint_this_episode = False
        memory, inputs = self.emulator.reset(dir=self.random_save_path)

        visited_maps = set(self.emulator.data.visited_maps)
        visited_dialogs = set(self.emulator.data.visited_dialogs.keys())

        action_counter = [0] * N_META_ACTIONS
        total_episode_reward = 0.0
        step_count = 0

        while self.event_start.is_set():
            action = self.get_action(inputs)
            action_counter[action] += 1

            if focused and keyboard.is_pressed("ctrl"):
                key = keyboard.read_key()
                time.sleep(0.1)
                action_idx = None
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
                if action_idx is not None:
                    action = action_idx * len(DURATION_BINS) + 0  # 16-tick bin

            next_memory, next_inputs, reward, terminated, truncated = (
                self.emulator.step(memory=memory, meta_action=action)
            )

            total_episode_reward += reward
            step_count += 1

            if self.emulator.last_milestone > 0:
                self._save_checkpoint(f"Milestone reward")

            current_map = self.emulator.data.map_id(next_memory)
            if current_map not in visited_maps:
                visited_maps.add(current_map)
                self._save_checkpoint(f"New map #{current_map}")

            if self.emulator.data.is_dialog(next_memory):
                current_dialog = self.emulator.data.get_dialog(next_memory)
                if current_dialog not in visited_dialogs:
                    visited_dialogs.add(current_dialog)
                    self._save_checkpoint(f"New dialog {current_dialog}")

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
                self._handle_truncate()
                try:
                    self.stats_queue.put_nowait({
                        "type": "episode",
                        "episode_length": step_count,
                        "epsilon": self._last_epsilon,
                        "action_counts": action_counter,
                        "total_reward": total_episode_reward,
                    })
                except Exception:
                    pass
                break

            memory, inputs = next_memory, next_inputs

            if self._total_steps % 200 == 0 and self.recv_conn.poll():
                self.model_state_dict = self.recv_conn.recv()

            self.emulator.use_sdl = bool(self.window.get())

    def get_action(self, inputs: dict[float]):
        frac = min(1.0, self._total_steps / self.epsilon_decay_steps)
        epsilon = self.max_stuck_epsilon - frac * (self.max_stuck_epsilon - self.min_stuck_epsilon)
        self._last_epsilon = epsilon
        self._total_steps += 1

        model_inputs = inputs
        if self._state_buffer:
            model_inputs = {**inputs, "state_sequence": torch.stack(list(self._state_buffer)).unsqueeze(0)}

        with torch.inference_mode():
            out = self.model(model_inputs)

        if isinstance(out, dict) and "z" in out:
            self._state_buffer.append(out["z"].squeeze(0).detach().cpu())

        if random.random() < epsilon:
            action = random.randint(0, N_META_ACTIONS - 1)
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
