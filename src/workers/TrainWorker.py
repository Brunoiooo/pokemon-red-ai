from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from multiprocessing import Queue, Manager
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
import os
import queue
import random
from threading import RLock, Thread
from time import sleep
import traceback
from typing import Any
import torch.optim as optim
import numpy as np
import torch
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import ModelPokemon, get_model
from pokemon.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from workers.ExperienceWorker import ExperienceWorker
from math import inf


@dataclass
class TrainWorker:
    max_workers: int = field(default_factory=lambda: 5)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    queue_data: Queue = field(default_factory=lambda: Manager().Queue())
    event_start: Event = field(default_factory=lambda: Manager().Event())
    queue_logs: Queue = field(default_factory=lambda: Manager().Queue())
    queue_dots: Queue = field(default_factory=lambda: Manager().Queue())
    model_lock: RLock = field(default_factory=lambda: Manager().RLock())
    is_debug: Synchronized = field(default_factory=lambda: Manager().Value("b", False))
    hash_buffer: deque = field(default_factory=lambda: deque(maxlen=200000))
    buffer: deque = field(default_factory=lambda: deque(maxlen=200000))
    buffer_lock: RLock = field(default_factory=lambda: Manager().RLock())
    is_evaluation_window: Synchronized = field(
        default_factory=lambda: Manager().Value("b", False)
    )
    train_use_sdl: Synchronized = field(
        default_factory=lambda: Manager().Value("b", False)
    )

    batch_size = 512
    grad_accum_steps = 1
    lr = 0.0001
    weight_decay = 1e-5
    gamma = 0.99
    criterion: torch.nn.SmoothL1Loss = field(default_factory=torch.nn.SmoothL1Loss)
    tau = 0.005
    target_update_interval = 1000
    _opt_steps: int = 0

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    era: int = 10
    era_count: int = 0

    max_epsilon: float = 0.01

    count: int = 0

    __model: None | ModelPokemon = None

    @property
    def model(self):
        if not self.__model:
            name = None
            if os.path.exists("models/latest.pth"):
                name = "latest"
            elif os.path.exists("models/best.pth"):
                name = "best"

            self.__model = get_model(device=self.device, name=name)

            self.__model.train()

        return self.__model

    __target_model: None | ModelPokemon = None

    @property
    def target_model(self):
        if not self.__target_model:
            self.__target_model = get_model(self.device)

            with self.model_lock:
                self.__target_model.load_state_dict(self.model.state_dict())

            self.__target_model.eval()

        return self.__target_model

    __optimizer: None | optim.AdamW = None

    @property
    def optimizer(self):
        if not self.__optimizer:
            with self.model_lock:
                self.__optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

        return self.__optimizer

    __gamma_tensor: torch.Tensor | None = None

    @property
    def gamma_tensor(self):
        if not self.__gamma_tensor:
            self.__gamma_tensor = torch.tensor(
                self.gamma, device=self.device, dtype=torch.float32
            )

        return self.__gamma_tensor

    __best_eval_return: float | None = None

    @property
    def best_eval_return(self):
        if self.__best_eval_return is None:
            try:
                model_best = torch.load(
                    "models/best.pth", map_location=torch.device("cpu")
                )

                self.__best_eval_return = float(
                    model_best["best_return"]
                    if isinstance(model_best, dict) and "best_return" in model_best
                    else -float("inf")
                )

            except Exception:
                self.__best_eval_return = -inf

        return self.__best_eval_return

    @best_eval_return.setter
    def best_eval_return(self, best_eval_return: float):
        self.__best_eval_return = best_eval_return

    __run_queue_thread: Thread | None = None

    @property
    def run_queue_thread(self):
        if not self.__run_queue_thread:
            self.__run_queue_thread = Thread(target=self.run_queue, daemon=True)

        return self.__run_queue_thread

    @run_queue_thread.setter
    def run_queue_thread(self, value: Thread | None):
        self.__run_queue_thread = value

    __run_train_thread: Thread | None = None

    @property
    def run_train_thread(self):
        if not self.__run_train_thread:
            self.__run_train_thread = Thread(target=self.run_train, daemon=True)

        return self.__run_train_thread

    @run_train_thread.setter
    def run_train_thread(self, value: Thread | None):
        self.__run_train_thread = value

    __run_workers_thread: Thread | None = None

    @property
    def run_workers_thread(self):
        if not self.__run_workers_thread:
            self.__run_workers_thread = Thread(target=self.run_workers, daemon=True)

        return self.__run_workers_thread

    @run_workers_thread.setter
    def run_workers_thread(self, value: Thread | None):
        self.__run_workers_thread = value

    __experienceWorkers: list[ExperienceWorker] | None = None

    @property
    def experienceWorkers(self):
        with self.model_lock:
            model_state_dict = self.model.state_dict()

        if self.__experienceWorkers is None:
            self.__experienceWorkers = [
                ExperienceWorker(
                    queue_logs=self.queue_logs,
                    queue_data=self.queue_data,
                    gamma=self.gamma,
                    model_state_dict=model_state_dict,
                    window=self.train_use_sdl,
                )
                for _ in range(self.max_workers)
            ]

        with self.buffer_lock:
            buffer_len = len(self.buffer)
            buffer_maxlen = self.buffer.maxlen

        epsilon = max(1.0 - buffer_len / buffer_maxlen, self.max_epsilon)

        for x in self.__experienceWorkers:
            x.model_state_dict = model_state_dict
            x.epsilon = epsilon

        return self.__experienceWorkers

    def run(self):
        try:
            self.queue_logs.put_nowait("TrainWorker starting up.")

            self.event_start.set()

            if not self.run_queue_thread.is_alive():
                self.run_queue_thread = None
                self.run_queue_thread.start()

            if not self.run_train_thread.is_alive():
                self.run_train_thread = None
                self.run_train_thread.start()

            if not self.run_workers_thread.is_alive():
                self.run_workers_thread = None
                self.run_workers_thread.start()

        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
            self.event_start.clear()
        finally:
            self.queue_logs.put_nowait("TrainWorker setted up.")

    def run_train(self):
        try:
            self.queue_logs.put_nowait("Training started.")

            while self.event_start.is_set():
                self.optimize_batch()

                if self.is_debug.value and self.count % 10 == 0:
                    with self.buffer_lock:
                        self.queue_logs.put_nowait(
                            f"Count: {self.count} | Buffer: {len(self.buffer)}"
                        )

                self.count += 1
        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.event_start.clear()
            self.queue_logs.put_nowait("Train stopped.")

    def run_workers(self):
        try:
            self.queue_logs.put_nowait("Workers started.")

            while self.event_start.is_set():
                if self.era_count % self.era == 0:
                    self.evaluate_greedy()

                self.queue_logs.put_nowait(f"Episode {self.era_count}.")

                with ProcessPoolExecutor(
                    max_workers=self.max_workers, mp_context=mp.get_context("spawn")
                ) as pool:
                    futures = [pool.submit(x.run) for x in self.experienceWorkers]

                    for future in futures:
                        future.result()

                self.era_count += 1

        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.event_start.clear()
            self.queue_logs.put_nowait("Workers stopped.")

    def run_queue(self):
        try:
            self.queue_logs.put_nowait("Queue handler started.")

            count = 0

            while self.event_start.is_set():
                try:
                    item = self.queue_data.get_nowait()
                except queue.Empty:
                    continue

                if item[7] in self.hash_buffer:
                    count += 1
                    if count % 1000 == 0:
                        self.queue_logs.put_nowait(
                            f"Duplicate experience skipped: {count}"
                        )
                    continue

                with self.buffer_lock:
                    self.buffer.append(item)
                    self.hash_buffer.append(item[7])
        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
        finally:
            self.event_start.clear()
            self.queue_logs.put_nowait("Queue handler stopped.")

    def evaluate_greedy(self):
        self.queue_logs.put_nowait(f"Starting evaluation.")

        with self.model_lock:
            model_state_dict = self.model.state_dict()

        self.save_latest()

        avg_ret = Emulator().evaluate_greedy(
            model_state_dict=model_state_dict,
            queue_logs=self.queue_logs,
            is_debug=False,
            is_evaluation_window=self.is_evaluation_window.value,
        )

        if self.best_eval_return < avg_ret:
            self.save_best(avg_ret)

        self.queue_dots.put_nowait((avg_ret, self.era_count))

        self.queue_logs.put_nowait(f"Finished evaluation {avg_ret:.6f}.")

        self.queue_logs.put_nowait("Evaluation stopped.")

    def optimize_batch(self):
        try:
            with self.buffer_lock:
                if len(self.buffer) < self.buffer.maxlen // 100:
                    raise ValueError(
                        f"Buffer too small: expected at least {self.buffer.maxlen // 100}, got {len(self.buffer)}"
                    )
        except ValueError as e:
            self.queue_logs.put_nowait(str(e))
            sleep(1.0)
            return

        with self.buffer_lock:
            batch = random.sample(list(self.buffer), k=self.batch_size)

        (
            inputs,
            actions,
            next_inputs,
            rewards,
            terminateds,
            truncateds,
            steps,
            keys,
        ) = zip(*batch)

        inputs = self.collate_states(inputs)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        next_inputs = self.collate_states(next_inputs)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminateds = torch.tensor(terminateds, device=self.device, dtype=torch.bool)
        truncateds = torch.tensor(truncateds, device=self.device, dtype=torch.bool)
        steps = torch.tensor(steps, device=self.device, dtype=torch.long)

        micro = self.batch_size // self.grad_accum_steps
        assert self.batch_size % self.grad_accum_steps == 0

        self.optimizer.zero_grad(set_to_none=True)

        total_td_errors = np.zeros(self.batch_size)

        for i in range(self.grad_accum_steps):
            sl = slice(i * micro, (i + 1) * micro)
            s = {k: v[sl] for k, v in inputs.items()}
            ns = {k: v[sl] for k, v in next_inputs.items()}
            a = actions[sl]
            rN = rewards[sl]
            te = terminateds[sl]
            tr = truncateds[sl]
            n = steps[sl]

            with self.model_lock:
                q_all = self.model(s)
            q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                with self.model_lock:
                    was_training = self.model.training
                    self.model.eval()
                    next_q_online = self.model(ns)
                    next_a = torch.argmax(next_q_online, dim=1)
                    if was_training:
                        self.model.train()

                    next_q_target = (
                        self.target_model(ns).gather(1, next_a.view(-1, 1)).squeeze(1)
                    )

                gamma_pow_n = torch.pow(self.gamma_tensor, n)
                bootstrap_mask = (~(te | tr)).float()
                target = rN + bootstrap_mask * gamma_pow_n * next_q_target

            loss = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction="mean")
            (loss / self.grad_accum_steps).backward()

            td_error = torch.abs(q_sa - target).detach()
            total_td_errors[sl] = td_error.cpu().numpy()

        with self.model_lock:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_interval == 0:
            self.hard_update_target()

        self.soft_update_target()

    def collate_states(self, list_of_dicts: tuple[Any, ...]):
        batch = {}
        keys = list(list_of_dicts[0].keys())
        for k in keys:
            vals = [d[k] for d in list_of_dicts]
            if k in self.model.FLOAT_INPUTS:
                t = torch.from_numpy(np.stack(vals, axis=0)).float()
                batch[k] = t.to(self.device, non_blocking=True)
            else:
                batch[k] = torch.tensor(vals, dtype=torch.long, device=self.device)

        return batch

    def soft_update_target(self):
        with torch.no_grad(), self.model_lock:
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def hard_update_target(self):
        with torch.no_grad(), self.model_lock:
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(p.data)

    def save_latest(self):
        os.makedirs("models", exist_ok=True)
        with self.model_lock:
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "target_state": self.target_model.state_dict(),
                },
                "models/latest.pth",
            )

    def save_best(self, avg_return: float):
        self.best_eval_return = avg_return

        os.makedirs("models", exist_ok=True)
        with self.model_lock:
            torch.save(
                {
                    "best_return": avg_return,
                    "model_state": self.model.state_dict(),
                },
                "models/best.pth",
            )
