from collections import deque
from dataclasses import dataclass, field
import io
from multiprocessing import Pipe, Process, Queue
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
import os
from queue import Empty
import random
import traceback
from typing import Any
import torch.optim as optim
import numpy as np
import torch
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import ModelPokemon
from workers.ExperienceWorker import ExperienceWorker


@dataclass
class TrainWorker:
    is_debug = False
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    queue_data_maxsize = 1000
    deque_buffer_maxlen = 50000
    optimize_every = 4
    updates_per_optimize = 2
    batch_size = 512
    grad_accum_steps = 1
    lr = 0.0001
    weight_decay = 0.0001
    gamma = 0.99
    criterion: torch.nn.SmoothL1Loss = field(default_factory=torch.nn.SmoothL1Loss)
    tau = 0.005
    sync_interval = 2000
    epsilon: float = 1
    epsilon_burn = 100000
    epsilon_end = 0.05
    epsilon_decay_count = 2000000
    ckpt_every = 25000
    evaluate_greedy_times = 5

    __model: None | ModelPokemon = None

    __event_stop: None | Event = None

    @property
    def event_stop(self):
        if not self.__event_stop:
            raise RuntimeError("event_stop is not set")

        return self.__event_stop

    @event_stop.setter
    def event_stop(self, event_stop: Event):
        self.__event_stop = event_stop

    __queue_logs: None | MPQueue = None

    @property
    def queue_logs(self):
        if not self.__queue_logs:
            raise RuntimeError("queue_logs is not set")

        return self.__queue_logs

    @queue_logs.setter
    def queue_logs(self, queue_logs: MPQueue):
        self.__queue_logs = queue_logs

    __count: None | Synchronized = None

    @property
    def count(self):
        if not self.__count:
            raise RuntimeError("count is not set")

        return self.__count

    @count.setter
    def count(self, count: Synchronized):
        self.__count = count

    @property
    def model(self):
        if not self.__model:
            emulator = Emulator()
            self.__model = ModelPokemon(
                len(emulator.data.data()), len(emulator.buttons)
            ).to(self.device)
            emulator.pyboy.stop(False)

            ckpt_path = None
            if os.path.exists("models/latest.pth"):
                ckpt_path = "models/latest.pth"
            elif os.path.exists("models/best.pth"):
                ckpt_path = "models/best.pth"

            if ckpt_path:
                state = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(
                    (
                        state["model_state"]
                        if isinstance(state, dict) and "model_state" in state
                        else state
                    ),
                    strict=True,
                )

            self.model.train()

        return self.__model

    __target_model: None | ModelPokemon = None

    @property
    def target_model(self):
        if not self.__target_model:
            emulator = Emulator()
            self.__target_model = ModelPokemon(
                len(emulator.data.data()), len(emulator.buttons)
            ).to(self.device)
            emulator.pyboy.stop(False)

            self.target_model.load_state_dict(self.model.state_dict())

            self.target_model.eval()

        return self.__target_model

    __optimizer: None | optim.AdamW = None

    @property
    def optimizer(self):
        if not self.__optimizer:
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

    @property
    def current_epsilon(self):
        with self.count.get_lock():
            if self.count.value < self.epsilon_burn:
                return self.epsilon
            t = min(self.count.value - self.epsilon_burn, self.epsilon_decay_count)
            frac = 1.0 - (t / self.epsilon_decay_count)
            return self.epsilon_end + (self.epsilon - self.epsilon_end) * max(frac, 0.0)

    __best_eval_return: float | None = None

    @property
    def best_eval_return(self):

        if self.__best_eval_return is None:

            model_best = torch.load("models/best.pth", map_location=torch.device("cpu"))

            self.__best_eval_return = float(
                model_best["best_return"]
                if isinstance(model_best, dict) and "best_return" in model_best
                else -float("inf")
            )

        return self.__best_eval_return

    @best_eval_return.setter
    def best_eval_return(self, best_eval_return: float):
        self.__best_eval_return = best_eval_return

    def run(self, event_stop: Event, queue_logs: MPQueue, count: Synchronized):
        try:
            self.event_stop = event_stop
            self.queue_logs = queue_logs
            self.count = count

            queue_data = Queue(maxsize=self.queue_data_maxsize)

            processes: dict[int, Process] = {}
            connections_epsilon: dict[int, Connection] = {}
            connections_state_dict: dict[int, Connection] = {}

            for i in range(self.workers):
                (connection_epsilon, connection_state_dict, process) = (
                    self.create_process(queue_data=queue_data)
                )

                connections_epsilon.setdefault(i, connection_epsilon)
                connections_state_dict.setdefault(i, connection_state_dict)
                processes.setdefault(i, process)

            for i in processes:
                processes[i].start()

            self.setup_experience_workers(
                connections_epsilon=connections_epsilon,
                connections_state_dict=connections_state_dict,
            )

            deque_buffer = deque(maxlen=self.deque_buffer_maxlen)

            while not self.event_stop.is_set():
                with self.count.get_lock():
                    if self.count.value % 1000 == 0:
                        self.queue_logs.put_nowait(
                            f"Count: {self.count.value} | Epsilon: {self.current_epsilon:.2f} | Progress: {(self.count.value / self.epsilon_end * 100):.2f}% | queue_data: {queue_data.qsize()} | deque_buffer: {len(deque_buffer)}"
                        )

                    try:
                        deque_buffer.append(queue_data.get_nowait())
                    except Empty as e:
                        self.event_stop.wait(0.001)

                    if self.count.value % self.optimize_every == 0:
                        for _ in range(self.updates_per_optimize):
                            self.optimize_batch(deque_buffer=deque_buffer)

                    if self.count.value % self.sync_interval == 0:
                        self.setup_experience_workers(
                            connections_epsilon=connections_epsilon,
                            connections_state_dict=connections_state_dict,
                        )

                    if self.count.value % self.ckpt_every == 0:
                        self.evaluate_greedy()

                    print("test")

                    self.count.value += 1
        except Exception as e:
            self.queue_logs.put_nowait(e)
            print(traceback.print_exc())
            self.event_stop.set()

    def optimize_batch(self, deque_buffer: deque):
        if len(deque_buffer) < self.batch_size:
            return

        batch = random.sample(deque_buffer, self.batch_size)
        (
            inputs,
            actions,
            next_inputs,
            rewards,
            terminateds,
            steps,
        ) = zip(*batch)

        inputs = self.collate_states(inputs)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        next_inputs = self.collate_states(next_inputs)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminateds = torch.tensor(terminateds, device=self.device, dtype=torch.bool)
        steps = torch.tensor(steps, device=self.device, dtype=torch.long)

        micro = self.batch_size // self.grad_accum_steps
        assert self.batch_size % self.grad_accum_steps == 0

        self.optimizer.zero_grad(set_to_none=True)

        for i in range(self.grad_accum_steps):
            sl = slice(i * micro, (i + 1) * micro)

            s = {k: v[sl] for k, v in inputs.items()}
            ns = {k: v[sl] for k, v in next_inputs.items()}

            a = actions[sl]
            rN = rewards[sl]
            te = terminateds[sl]
            n = steps[sl]

            q_all = self.model(s)
            q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                next_q_online = self.model(ns)
                next_a = torch.argmax(next_q_online, dim=1)

                next_q_target = (
                    self.target_model(ns).gather(1, next_a.view(-1, 1)).squeeze(1)
                )

                gamma_pow_n = torch.pow(self.gamma_tensor, n)

                bootstrap_mask = (~te).float()
                target = rN + bootstrap_mask * gamma_pow_n * next_q_target

            loss = self.criterion(q_sa, target) / self.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update_target()

    def collate_states(self, list_of_dicts: tuple[Any, ...]):
        batch = {}
        keys = list(list_of_dicts[0].keys())
        for k in keys:
            vals = [d[k] for d in list_of_dicts]
            if k == "continuous":
                t = torch.from_numpy(np.stack(vals, axis=0)).float()
                batch[k] = t.to(self.device, non_blocking=True)
            else:
                t = torch.tensor(vals, dtype=torch.long)
                batch[k] = t.to(self.device, non_blocking=True)

        return batch

    def soft_update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save_latest(self):
        os.makedirs("models", exist_ok=True)
        with self.count.get_lock():
            torch.save(
                {
                    "step": self.count.value,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "target_state": self.target_model.state_dict(),
                },
                "models/latest.pth",
            )

    def save_best(self, avg_return: float):
        self.best_eval_return = avg_return

        with self.count.get_lock():
            torch.save(
                {
                    "step": self.count.value,
                    "best_return": avg_return,
                    "model_state": self.model.state_dict(),
                },
                "models/best.pth",
            )

    def create_process(self, queue_data: MPQueue):
        connection_epsilon_parent, connection_epsilon_child = Pipe(duplex=True)
        connection_state_dict_parent, connection_state_dict_child = Pipe(duplex=True)

        process = Process(
            target=ExperienceWorker(
                event_stop=self.event_stop,
                queue_logs=self.queue_logs,
                connection_epsilon=connection_epsilon_child,
                connection_state_dict=connection_state_dict_child,
                queue_data=queue_data,
                gamma=self.gamma,
            ).start,
            daemon=True,
        )

        return (connection_epsilon_parent, connection_state_dict_parent, process)

    def setup_experience_workers(
        self,
        connections_epsilon: dict[int, Connection],
        connections_state_dict: dict[int, Connection],
    ):
        for i in connections_epsilon:
            connections_epsilon[i].send(self.current_epsilon)

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)

        for i in connections_state_dict:
            connections_state_dict[i].send(buffer.getvalue())

    def evaluate_greedy(self):
        self.save_latest()

        avg_ret = Emulator().evaluate_greedy(
            model=self.model, evaluate_greedy_times=self.evaluate_greedy_times
        )

        if avg_ret > self.best_eval_return:
            self.save_best(avg_ret)
