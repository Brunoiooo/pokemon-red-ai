from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Queue, Manager
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
import os
from queue import Empty
import random
from threading import RLock, Thread
import traceback
from typing import Any
import torch.optim as optim
import numpy as np
import torch
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import ModelPokemon, get_model
from workers.ExperienceWorker import ExperienceWorker
from math import inf


@dataclass
class TrainWorker:
    max_workers: int = field(default_factory=lambda: max(1, os.cpu_count() - 1 or 1))
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    queue_data: Queue = field(default_factory=lambda: Manager().Queue())
    event_start: Event = field(default_factory=lambda: Manager().Event())
    queue_logs: Queue = field(default_factory=lambda: Manager().Queue())
    model_lock: RLock = field(default_factory=lambda: Manager().RLock())
    is_debug: Synchronized = field(default_factory=lambda: Manager().Value("b", False))
    is_evaluation_window: Synchronized = field(
        default_factory=lambda: Manager().Value("b", False)
    )
    train_use_sdl: Synchronized = field(
        default_factory=lambda: Manager().Value("b", False)
    )
    deque_buffer_maxlen = 50000
    ckpt_every = 100

    batch_size = 512
    grad_accum_steps = 1
    lr = 0.00001
    weight_decay = 1e-5
    gamma = 0.99
    criterion: torch.nn.SmoothL1Loss = field(default_factory=torch.nn.SmoothL1Loss)
    tau = 0.005
    evaluate_greedy_times = 1
    epsilon: float = 0.1
    # loss tracking
    running_loss_ema: float = 0.0
    loss_ema_alpha: float = 0.001
    last_loss: float | None = None
    target_update_interval = 1000
    _opt_steps: int = 0

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

    def run_era(self):
        try:
            with self.model_lock:
                model_state_dict = self.model.state_dict()

            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=ctx
            ) as pool:
                futures = [
                    pool.submit(
                        ExperienceWorker(
                            queue_logs=self.queue_logs,
                            queue_data=self.queue_data,
                            gamma=self.gamma,
                            model_state_dict=model_state_dict,
                            epsilon=self.epsilon,
                            window=self.train_use_sdl,
                        ).run
                    )
                    for _ in range(self.max_workers)
                ]

                for future in futures:
                    future.result()

        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")
            self.event_start.clear()

    def run_train(self):
        try:
            self.event_start.set()

            deque_buffer = deque(maxlen=self.deque_buffer_maxlen)

            age = 1

            while self.event_start.is_set():
                count = 0

                thread = Thread(target=self.run_era, daemon=True)
                thread.start()

                if len(deque_buffer) < self.batch_size:
                    thread.join()

                while thread.is_alive():
                    self.optimize_batch(deque_buffer=deque_buffer)
                    count += 1

                while True:
                    try:
                        deque_buffer.append(self.queue_data.get_nowait())
                    except Empty:
                        break

                self.queue_logs.put_nowait(f"Age: {age} | Count: {count}")

                if age % self.ckpt_every == 0 and age > 0:
                    self.evaluate_greedy()

                age += 1
        except Exception as e:
            self.queue_logs.put_nowait(f"{e}\n{traceback.print_exc()}")

    def optimize_batch(self, deque_buffer: deque):
        batch = random.sample(deque_buffer, self.batch_size)
        (inputs, actions, next_inputs, rewards, terminateds, truncateds, steps) = zip(
            *batch
        )

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
        batch_loss = 0.0

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
                bootstrap_mask = (~te | ~tr).float()
                target = rN + bootstrap_mask * gamma_pow_n * next_q_target

            loss = self.criterion(q_sa, target) / self.grad_accum_steps
            loss.backward()
            batch_loss += float(loss.detach().item())

        with self.model_lock:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update_interval == 0:
            self.hard_update_target()

        self.soft_update_target()

        self.last_loss = batch_loss
        if self.running_loss_ema == 0.0:
            self.running_loss_ema = batch_loss
        else:
            self.running_loss_ema = (
                self.loss_ema_alpha * batch_loss
                + (1 - self.loss_ema_alpha) * self.running_loss_ema
            )

        try:
            self.queue_logs.put_nowait(
                f"Loss: {batch_loss:.6f} | EMA: {self.running_loss_ema:.6f}"
            )
        except Exception:
            pass

    def collate_states(self, list_of_dicts: tuple[Any, ...]):
        batch = {}
        keys = list(list_of_dicts[0].keys())
        for k in keys:
            vals = [d[k] for d in list_of_dicts]
            if k == "continuous":
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

    def evaluate_greedy(self):
        self.queue_logs.put_nowait(f"Starting evaluation.")

        self.save_latest()

        with self.model_lock:
            self.model.eval()
            avg_ret = Emulator().evaluate_greedy(
                model=self.model,
                evaluate_greedy_times=self.evaluate_greedy_times,
                queue_logs=self.queue_logs,
                is_debug=self.is_debug.value,
                is_evaluation_window=self.is_evaluation_window.value,
            )
            self.model.train()

        if self.best_eval_return < avg_ret:
            self.save_best(avg_ret)

        self.queue_logs.put_nowait(f"Finished evaluation {avg_ret:.2f}.")
