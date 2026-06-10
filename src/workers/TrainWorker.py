from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from multiprocessing import Process, Queue, Manager
import multiprocessing as mp
from multiprocessing.connection import Pipe, PipeConnection
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
import os
import queue
import random
import shutil
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
from utils.BufferManager import BufferManager
from math import inf


@dataclass
class TrainWorker:
    max_workers: int = field(default_factory=lambda: 5)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    manager: any = field(init=False, repr=False)

    queue_data: any = field(init=False)
    event_start: any = field(init=False)
    queue_logs: any = field(init=False)
    queue_dots: any = field(init=False)
    model_lock: any = field(init=False)
    files_lock: any = field(init=False)
    eval_lock: any = field(init=False)
    is_debug: any = field(init=False)
    buffer_lock: any = field(init=False)
    is_evaluation_window: any = field(init=False)
    train_use_sdl: any = field(init=False)

    buffer_capacity: int = 200000
    buffer: PrioritizedReplayBuffer | None = field(default=None, init=False, repr=False)
    buffer_manager: BufferManager = field(default=None, init=False, repr=False)
    last_buffer_save: int = field(default=0, init=False, repr=False)

    max_last_saves: int = 10
    max_best_saves: int = 3
    eval_interval_seconds: int = 120

    def __post_init__(self):
        self.manager = mp.Manager()

        self.queue_data = self.manager.Queue(maxsize=2000)
        self.event_start = self.manager.Event()
        self.queue_logs = self.manager.Queue()
        self.queue_dots = self.manager.Queue()
        self.model_lock = self.manager.RLock()
        self.files_lock = self.manager.RLock()
        self.eval_lock = self.manager.RLock()
        self.is_debug = self.manager.Value("b", False)
        self.buffer_lock = self.manager.RLock()
        self.is_evaluation_window = self.manager.Value("b", False)
        self.train_use_sdl = self.manager.Value("b", False)

        self.buffer_manager = BufferManager()

        loaded_buffer = self.buffer_manager.load_buffer(
            capacity=self.buffer_capacity,
            alpha=self.per_alpha
        )

        if loaded_buffer is not None:
            self.buffer = loaded_buffer
            buffer_size_mb = self.buffer_manager.get_buffer_size()
            self.queue_logs.put_nowait(
                f"Loaded replay buffer from disk ({buffer_size_mb:.1f}MB, {len(self.buffer)} entries)"
            )
        else:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_capacity,
                alpha=self.per_alpha,
            )

    batch_size = 512
    grad_accum_steps = 1
    lr = 5e-4
    weight_decay = 1e-4
    gamma = 0.99
    criterion: torch.nn.SmoothL1Loss = field(default_factory=torch.nn.SmoothL1Loss)
    tau = 0.001
    target_update_interval = 500
    _opt_steps: int = 0

    per_alpha: float = 0.7
    per_beta_start: float = 0.6
    per_beta_frames: int = 2_000_000

    era: int = 1000

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

            self.__model = get_model(
                device=self.device, files_lock=self.files_lock, name=name
            )

            self.__model.train()

        return self.__model

    __target_model: None | ModelPokemon = None

    @property
    def target_model(self):
        if not self.__target_model:
            self.__target_model = get_model(self.device, files_lock=self.files_lock)

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
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

        return self.__optimizer

    __scheduler: any = None

    @property
    def scheduler(self):
        if self.__scheduler is None:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=5000,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=1_000_000,
                eta_min=1e-6,
            )
            self.__scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[5000],
            )
        return self.__scheduler

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

    __experienceWorkerPipes: list[tuple[PipeConnection, PipeConnection]] | None = None

    @property
    def experienceWorkerPipes(self):
        if self.__experienceWorkerPipes is None:
            self.__experienceWorkerPipes = [
                Pipe(duplex=False) for x in range(self.max_workers)
            ]

        return self.__experienceWorkerPipes

    __experienceWorkers: list[ExperienceWorker] | None = None

    @property
    def experienceWorkers(self):
        if self.__experienceWorkers is None:
            with self.model_lock:
                model_state_dict = self.model.state_dict()

            self.__experienceWorkers = [
                ExperienceWorker(
                    queue_logs=self.queue_logs,
                    queue_data=self.queue_data,
                    gamma=self.gamma,
                    window=self.train_use_sdl,
                    recv_conn=recv_conn,
                    event_start=self.event_start,
                    init_model_state_dict=model_state_dict,
                    files_lock=self.files_lock,
                )
                for recv_conn, send_conn in self.experienceWorkerPipes
            ]

        return self.__experienceWorkers

    def _safe_log(self, msg: str):
        try:
            self.queue_logs.put_nowait(msg)
        except (BrokenPipeError, EOFError, OSError):
            pass

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
            self._safe_log(f"{e}\n{traceback.print_exc()}")
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            self._safe_log("TrainWorker setted up.")

    def run_train(self):
        try:
            self.queue_logs.put_nowait("Training started.")

            while self.event_start.is_set():
                self.optimize_batch()

                with self.eval_lock:
                    if self.is_debug.value and self.count % 10 == 0:
                        with self.buffer_lock:
                            self.queue_logs.put_nowait(
                                f"Count: {self.count} | Buffer: {len(self.buffer)}"
                            )

                    if self.count % 500 == 0:
                        self.save_buffer_to_disk()

                    self.count += 1
        except Exception as e:
            self._safe_log(f"{e}\n{traceback.print_exc()}")
        finally:
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.save_buffer_to_disk()
            self._safe_log("Train stopped.")

    def run_workers(self):
        try:
            self.queue_logs.put_nowait("Workers started.")

            processes: list[Process] = []
            for x in self.experienceWorkers:
                p = Process(
                    target=x.run, kwargs={"focused": len(processes) <= 0}, daemon=True
                )
                p.start()
                processes.append(p)

            threads: list[Thread] = []
            for recv_conn, send_conn in self.experienceWorkerPipes:
                t = Thread(
                    target=self.send_model_state_dict_to_worker,
                    args=[send_conn],
                )
                t.start()
                threads.append(t)

            t = Thread(target=self.evaluate_greedy)
            t.start()
            threads.append(t)

            while self.event_start.is_set():
                for p in processes:
                    p.join()
                for t in threads:
                    t.join()

        except Exception as e:
            self._safe_log(f"{e}\n{traceback.print_exc()}")
        finally:
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
            if processes:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
            self._safe_log("Workers stopped.")

    def run_queue(self):
        try:
            self.queue_logs.put_nowait("Queue handler started.")

            while self.event_start.is_set():
                try:
                    item = self.queue_data.get(timeout=0.1)
                except queue.Empty:
                    continue

                with self.buffer_lock:
                    self.buffer.add(item)
        except Exception as e:
            self._safe_log(f"{e}\n{traceback.print_exc()}")
        finally:
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._safe_log("Queue handler stopped.")

    def send_model_state_dict_to_worker(self, send_conn: PipeConnection):
        try:
            while self.event_start.is_set():
                with self.model_lock:
                    model_state_dict = self.model.state_dict()

                send_conn.send(model_state_dict)
                sleep(30)
        except Exception as e:
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._safe_log(f"{e}\n{traceback.print_exc()}")
        finally:
            self._safe_log("Stopped sending model state dict to worker.")

    def evaluate_greedy(self):
        try:
            self.queue_logs.put_nowait(f"Starting evaluation.")

            evaluation_count = 0

            while self.event_start.is_set():

                with self.model_lock:
                    model_state_dict = self.model.state_dict()

                self.save_latest()

                with self.eval_lock:
                    count = self.count

                save_last = f"last_{evaluation_count % self.max_last_saves}"
                save_best = f"best_{evaluation_count % self.max_best_saves}"

                (avg_ret, steps) = Emulator(files_lock=self.files_lock).evaluate_greedy(
                    model_state_dict=model_state_dict,
                    queue_logs=self.queue_logs,
                    is_debug=False,
                    is_evaluation_window=self.is_evaluation_window.value,
                    save_name=save_last,
                )

                if self.best_eval_return < avg_ret:
                    self.save_best(avg_ret)
                    with self.files_lock:
                        shutil.copytree(
                            f"saves/{save_last}",
                            f"saves/{save_best}",
                            dirs_exist_ok=True,
                        )

                with self.eval_lock:
                    self.queue_dots.put_nowait((count, avg_ret))

                self.queue_logs.put_nowait(f"Finished evaluation {avg_ret:.6f}.")

                evaluation_count += 1

                for _ in range(self.eval_interval_seconds):
                    if not self.event_start.is_set():
                        break
                    sleep(1)

        except Exception as e:
            self._safe_log(f"{e}\n{traceback.print_exc()}")
            try:
                self.event_start.clear()
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            self._safe_log("Evaluation stopped.")

    __is_freeze: bool = False

    @property
    def is_freeze(self):
        return self.__is_freeze

    @is_freeze.setter
    def is_freeze(self, value: bool):
        if self.__is_freeze == value:
            return

        self.__is_freeze = value
        with self.model_lock:
            if value:
                self.model.freeze()
            else:
                self.model.unfreeze_all()
        self.freeze_steps = 0
        self.freeze_delay = 0
        self.__optimizer = None

    freeze_steps: int = 0
    freeze_max_steps: int = 10
    freeze_delay: int = 0
    freeze_max_delay: int = 10

    def freeze_model(self, avg_return: float):
        if self.is_freeze and self.freeze_steps < self.freeze_max_steps:
            self.freeze_steps += 1
            return
        elif self.is_freeze:
            self.queue_logs.put_nowait(f"Unfreezing model.")
            self.is_freeze = False
            return
        elif not self.is_freeze and self.freeze_delay < self.freeze_max_delay:
            self.freeze_delay += 1
            return
        elif not self.is_freeze and avg_return < self.best_eval_return:
            self.queue_logs.put_nowait(
                f"Freezing model for {self.freeze_max_steps} steps."
            )
            self.is_freeze = True
            return

    def save_buffer_to_disk(self):
        """Saves the replay buffer to disk. Call this periodically during training."""
        try:
            with self.buffer_lock:
                if self.buffer_manager.save_buffer(self.buffer):
                    buffer_size_mb = self.buffer_manager.get_buffer_size()
                    self.queue_logs.put_nowait(
                        f"Saved replay buffer ({buffer_size_mb:.1f}MB, {len(self.buffer)} entries)"
                    )
        except Exception as e:
            self.queue_logs.put_nowait(f"Error saving buffer: {e}")

    def per_beta(self):
        with self.eval_lock:
            frac = min(1.0, self.count / max(1, self.per_beta_frames))
        return self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def optimize_batch(self):
        try:
            with self.buffer_lock:
                if len(self.buffer) < self.batch_size:
                    raise ValueError(
                        f"Buffer too small: expected at least {self.batch_size}, got {len(self.buffer)}"
                    )
                beta = self.per_beta()
                batch, idxs, is_weights = self.buffer.sample(self.batch_size, beta=beta)
        except ValueError as e:
            self.queue_logs.put_nowait(str(e))
            sleep(0.1)
            return

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
        next_inputs = self.collate_states(next_inputs)

        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminateds = torch.tensor(terminateds, device=self.device, dtype=torch.bool)
        truncateds = torch.tensor(truncateds, device=self.device, dtype=torch.bool)
        steps = torch.tensor(steps, device=self.device, dtype=torch.long)
        is_weights = torch.tensor(is_weights, device=self.device, dtype=torch.float32)

        micro = self.batch_size // self.grad_accum_steps
        assert self.batch_size % self.grad_accum_steps == 0

        self.optimizer.zero_grad(set_to_none=True)

        total_td_errors = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.grad_accum_steps):
            sl = slice(i * micro, (i + 1) * micro)

            s = {k: v[sl] for k, v in inputs.items()}
            ns = {k: v[sl] for k, v in next_inputs.items()}
            a = actions[sl]
            rN = rewards[sl]
            te = terminateds[sl]
            tr = truncateds[sl]
            n = steps[sl]
            w = is_weights[sl]

            with self.model_lock:
                out = self.model(s)

            if isinstance(out, dict):
                # C51 distributional RL: Bellman projection + cross-entropy loss
                dist_sa = out["dist"].gather(
                    1,
                    a.view(-1, 1, 1).expand(-1, 1, self.model.num_atoms),
                ).squeeze(1)  # (micro, num_atoms)

                with torch.no_grad():
                    with self.model_lock:
                        was_training = self.model.training
                        self.model.eval()
                        next_out_online = self.model(ns)
                        next_a = torch.argmax(next_out_online["q"], dim=1)
                        if was_training:
                            self.model.train()

                        next_out_target = self.target_model(ns)
                        next_dist = next_out_target["dist"].gather(
                            1,
                            next_a.view(-1, 1, 1).expand(-1, 1, self.model.num_atoms),
                        ).squeeze(1)  # (micro, num_atoms)

                    support = self.model.support  # (num_atoms,)
                    v_min = self.model.v_min
                    v_max = self.model.v_max
                    num_atoms = self.model.num_atoms
                    delta_z = (v_max - v_min) / (num_atoms - 1)

                    gamma_pow_n = torch.pow(self.gamma_tensor, n.float())
                    bootstrap_mask = (~te).float()
                    Tz = rN.unsqueeze(1) + bootstrap_mask.unsqueeze(1) * gamma_pow_n.unsqueeze(1) * support.unsqueeze(0)
                    Tz = Tz.clamp(v_min, v_max)
                    b = (Tz - v_min) / delta_z
                    l = b.floor().long().clamp(0, num_atoms - 1)
                    u = b.ceil().long().clamp(0, num_atoms - 1)

                    m = torch.zeros_like(next_dist)
                    m.scatter_add_(1, l, next_dist * (u.float() - b))
                    m.scatter_add_(1, u, next_dist * (b - l.float()))

                log_p = torch.log(dist_sa + 1e-8)
                per_sample_loss = -(m * log_p).sum(dim=1)
                total_td_errors[sl] = per_sample_loss.detach().cpu().numpy()
            else:
                # Standard DQN fallback
                q_sa = out.gather(1, a.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    with self.model_lock:
                        was_training = self.model.training
                        self.model.eval()
                        next_q_online = self.model(ns)
                        if isinstance(next_q_online, dict):
                            next_q_online = next_q_online["q"]
                        next_a = torch.argmax(next_q_online, dim=1)
                        if was_training:
                            self.model.train()

                        next_out_target = self.target_model(ns)
                        if isinstance(next_out_target, dict):
                            next_out_target = next_out_target["q"]
                        next_q_target = next_out_target.gather(1, next_a.view(-1, 1)).squeeze(1)

                    gamma_pow_n = torch.pow(self.gamma_tensor, n.float())
                    bootstrap_mask = (~te).float()
                    target = rN + bootstrap_mask * gamma_pow_n * next_q_target

                td = target - q_sa
                total_td_errors[sl] = td.detach().abs().cpu().numpy()
                per_sample_loss = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction="none")

            loss = (per_sample_loss * w).mean()
            loss = loss / self.grad_accum_steps
            loss.backward()

        with self.model_lock:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        with self.buffer_lock:
            self.buffer.update_priorities(idxs, total_td_errors.tolist())

        self._opt_steps += 1
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
