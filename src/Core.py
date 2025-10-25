import torch
import torch.optim as optim
from Emulator import Emulator
from ModelPokemon import ModelPokemon
import os, io, keyboard, random, sys
from collections import deque
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np


class Core:
    def __init__(
        self,
        game="PokemonRed",
        epsilon=0.2,
        epsilonBurn=100000,
        epsilonEnd=0.05,
        epsilonDecayCount=2000000,
        tmpEpsilon=0.2,
        tmpEpsilonSteps=100000,
        ticksPerStep=32,
        maxMenuSelect=5,
        maxMenuPosition=10,
        maxMenuIn=15,
        maxSameAction=20,
        worldIllegalMovesMax=5,
        menuIllegalMovesMax=20,
        ckpt_every=25000,
        lr=0.0001,
        weight_decay=0.0001,
        sync_interval=2000,
        wrongDialogActionMax=5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.game = game
        self.epsilon = epsilon
        self.epsilonBurn = epsilonBurn
        self.epsilonEnd = epsilonEnd
        self.epsilonDecayCount = epsilonDecayCount
        self.ticksPerStep = ticksPerStep
        self.maxMenuSelect = maxMenuSelect
        self.maxMenuPosition = maxMenuPosition
        self.maxMenuIn = maxMenuIn
        self.maxSameAction = maxSameAction
        self.tmpEpsilon = tmpEpsilon
        self.tmpEpsilonSteps = tmpEpsilonSteps
        self.worldIllegalMovesMax = worldIllegalMovesMax
        self.menuIllegalMovesMax = menuIllegalMovesMax
        self.ckpt_every = ckpt_every
        self.next_ckpt = self.ckpt_every
        self.best_eval_return = -float("inf")
        self.count = 0
        self.sync_interval = sync_interval
        self.wrongDialogActionMax = wrongDialogActionMax

        ckpt_path = None
        if os.path.exists(f"roms/{self.game}/best.pth"):
            ckpt_path = f"roms/{self.game}/best.pth"
        elif os.path.exists(f"roms/{self.game}/latest.pth"):
            ckpt_path = f"roms/{self.game}/latest.pth"

        emulator = Emulator(
            0,
            False,
            self.game,
            self.ticksPerStep,
            self.maxMenuSelect,
            self.maxMenuPosition,
            self.maxMenuIn,
            self.maxSameAction,
            self.worldIllegalMovesMax,
            self.menuIllegalMovesMax,
            self.tmpEpsilonSteps,
            self.epsilon,
            self.tmpEpsilon,
            self.wrongDialogActionMax,
        )
        emulator.pyboy_init()
        self.modelPokemon = ModelPokemon(
            len(emulator.data()),
            len(emulator.buttons),
        ).to(self.device)

        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location=self.device)
            self.modelPokemon.load_state_dict(
                (
                    state["model_state"]
                    if isinstance(state, dict) and "model_state" in state
                    else state
                ),
                strict=True,
            )
        self.modelPokemon.train()

        self.targetPokemon = ModelPokemon(
            len(emulator.data()),
            len(emulator.buttons),
        ).to(self.device)
        self.targetPokemon.load_state_dict(self.modelPokemon.state_dict())
        self.targetPokemon.eval()

        emulator.pyboy.stop(False)
        emulator.pyboy = None

        self.tau = 0.005
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(
            self.modelPokemon.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.batch_size = 512
        self.buffer = deque(maxlen=50000)
        self.optimize_every = 4
        self.updates_per_opt = 3
        self.grad_accum_steps = 1
        self.replay_start = 5000

        self.gamma = 0.99
        self._gamma_tensor = torch.tensor(
            self.gamma, device=self.device, dtype=torch.float32
        )

    def start(self):
        self.dataQ = mp.Queue(maxsize=1000)
        self.stop_event = mp.Event()

        n_cores = os.cpu_count() or 1
        n_workers = max(1, n_cores - 1)

        self.conns = {}
        self.procs = {}

        window = False

        for actorId in range(n_workers):
            parent_conn, child_conn = mp.Pipe(duplex=True)
            emulator = Emulator(
                actorId,
                window,
                self.game,
                self.ticksPerStep,
                self.maxMenuSelect,
                self.maxMenuPosition,
                self.maxMenuIn,
                self.maxSameAction,
                self.worldIllegalMovesMax,
                self.menuIllegalMovesMax,
                self.tmpEpsilonSteps,
                self.epsilon,
                self.tmpEpsilon,
                self.wrongDialogActionMax,
            )
            p = mp.Process(
                target=emulator.start,
                args=(self.dataQ, child_conn, self.stop_event),
                daemon=False,
            )
            p.start()
            self.conns[actorId] = parent_conn
            self.procs[actorId] = p

        while True:
            if keyboard.is_pressed("q"):
                self.stop_event.set()
                break
            elif keyboard.is_pressed("w"):
                for wid, conn in self.conns.items():
                    conn.send({"type": "window", "value": False})
                window = False
                print("w")
            elif keyboard.is_pressed("e"):
                for wid, conn in self.conns.items():
                    conn.send({"type": "window", "value": True})
                window = True
                print("e")
            elif keyboard.is_pressed("r"):
                emulator = Emulator(
                    0,
                    True,
                    self.game,
                    self.ticksPerStep,
                    self.maxMenuSelect,
                    self.maxMenuPosition,
                    self.maxMenuIn,
                    self.maxSameAction,
                    self.worldIllegalMovesMax,
                    self.menuIllegalMovesMax,
                    self.tmpEpsilonSteps,
                    self.epsilon,
                    self.tmpEpsilon,
                    self.wrongDialogActionMax,
                )
                emulator.auto()
                print("r")
            elif keyboard.is_pressed("t"):
                emulator = Emulator(
                    0,
                    True,
                    self.game,
                    self.ticksPerStep,
                    self.maxMenuSelect,
                    self.maxMenuPosition,
                    self.maxMenuIn,
                    self.maxSameAction,
                    self.worldIllegalMovesMax,
                    self.menuIllegalMovesMax,
                    self.tmpEpsilonSteps,
                    self.epsilon,
                    self.tmpEpsilon,
                    self.wrongDialogActionMax,
                )
                emulator.manual()
                print("t")

            try:
                item = self.dataQ.get()
                self.count += 1
            except Exception as e:
                print(e)
                break

            self.buffer.append(item)

            if len(self.buffer) >= self.replay_start and (
                self.count % self.optimize_every == 0
            ):
                for _ in range(self.updates_per_opt):
                    self.optimize_batch()

            if self.count % self.sync_interval == 0:
                buf = io.BytesIO()
                torch.save(self.modelPokemon.state_dict(), buf)

                for wid, conn in self.conns.items():
                    conn.send({"type": "load_state_dict", "value": buf.getvalue()})
                    conn.send({"type": "epsilon", "value": self.currentEpsilon()})

            if self.count >= self.next_ckpt:
                self.save_latest()
                emulator = Emulator(
                    0,
                    True,
                    self.game,
                    self.ticksPerStep,
                    self.maxMenuSelect,
                    self.maxMenuPosition,
                    self.maxMenuIn,
                    self.maxSameAction,
                    self.worldIllegalMovesMax,
                    self.menuIllegalMovesMax,
                    self.tmpEpsilonSteps,
                    self.epsilon,
                    self.tmpEpsilon,
                    self.wrongDialogActionMax,
                    False,
                )
                avg_ret = emulator.evaluate_greedy(5)
                # self.plot_done_graph(emulator.doneGraph)

                if avg_ret > self.best_eval_return + 0.5:
                    self.save_best(avg_ret)

                self.next_ckpt += self.ckpt_every

            if self.count % 1000 == 0:
                sys.stdout.write(
                    f"\rEpsilon: {self.currentEpsilon():.2f} | Count: {self.count} | Progress: {(self.count / self.epsilonDecayCount * 100):.2f}% dataQ: {self.dataQ.qsize()}"
                )
                sys.stdout.flush()

        for p in self.procs.values():
            try:
                p.join(timeout=3.0)
            except Exception:
                pass

        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

    def currentEpsilon(self):
        if self.count < self.epsilonBurn:
            return self.epsilon
        t = min(self.count - self.epsilonBurn, self.epsilonDecayCount)
        frac = 1.0 - (t / self.epsilonDecayCount)
        return self.epsilonEnd + (self.epsilon - self.epsilonEnd) * max(frac, 0.0)

    def optimize_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        (
            states_cpu,
            actions,
            Rn_list,
            next_states_cpu,
            terminateds,
            truncateds,
            n_used_list,
        ) = zip(*batch)

        states = self.collate_states(states_cpu, self.device)
        next_states = self.collate_states(next_states_cpu, self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        Rn = torch.tensor(Rn_list, device=self.device, dtype=torch.float32)
        terminateds = torch.tensor(terminateds, device=self.device, dtype=torch.bool)
        truncateds = torch.tensor(
            truncateds, device=self.device, dtype=torch.bool
        )  # obecnie nieużywane w target
        n_used = torch.tensor(n_used_list, device=self.device, dtype=torch.float32)

        micro = self.batch_size // self.grad_accum_steps
        assert self.batch_size % self.grad_accum_steps == 0

        self.optimizer.zero_grad(set_to_none=True)

        for i in range(self.grad_accum_steps):
            sl = slice(i * micro, (i + 1) * micro)

            s = {k: v[sl] for k, v in states.items()}
            ns = {k: v[sl] for k, v in next_states.items()}

            a, rN, te, tr, n = (
                actions[sl],
                Rn[sl],
                terminateds[sl],
                truncateds[sl],
                n_used[sl],
            )

            # Q(s,a)
            q_all = self.modelPokemon(s)
            q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                # Double-DQN bootstrap na końcu okna (s_{t+n})
                next_q_online = self.modelPokemon(ns)
                next_a = torch.argmax(next_q_online, dim=1)

                next_q_target = (
                    self.targetPokemon(ns).gather(1, next_a.view(-1, 1)).squeeze(1)
                )

                # gamma ** n_used (wektorowo) – używamy prealokowanego gamma-tensora
                gamma_pow_n = torch.pow(self._gamma_tensor, n)

                # KLUCZ: tnij bootstrap TYLKO na terminated (te), nie na pełnym tensorze!
                target = torch.where(te, rN, rN + gamma_pow_n * next_q_target)

            loss = self.criterion(q_sa, target) / self.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.modelPokemon.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update_target()

    def soft_update_target(self):
        with torch.no_grad():
            for p, tp in zip(
                self.modelPokemon.parameters(), self.targetPokemon.parameters()
            ):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save_latest(self):
        torch.save(
            {
                "step": self.count,
                "model_state": self.modelPokemon.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "target_state": self.targetPokemon.state_dict(),
            },
            f"roms/{self.game}/latest.pth",
        )

    def save_best(self, avg_return):
        self.best_eval_return = avg_return
        torch.save(
            {
                "step": self.count,
                "best_return": float(avg_return),
                "model_state": self.modelPokemon.state_dict(),
            },
            f"roms/{self.game}/best.pth",
        )

    def plot_done_graph(self, doneGraph):
        if not doneGraph:
            print("Brak danych w doneGraph")
            return

        keys = list(doneGraph.keys())
        values = list(doneGraph.values())

        plt.figure(figsize=(12, 6))
        plt.bar(keys, values)
        plt.xticks(rotation=90)
        plt.ylabel("Ilość wystąpień")
        plt.title("DoneGraph - zakończenia epizodów")
        plt.tight_layout()
        plt.show()

    def collate_states(self, list_of_dicts, device):
        batch = {}
        keys = list(list_of_dicts[0].keys())
        for k in keys:
            vals = [d[k] for d in list_of_dicts]
            if k == "continuous":
                t = torch.from_numpy(np.stack(vals, axis=0)).float()
                batch[k] = t.to(device, non_blocking=True)
            else:
                t = torch.tensor(vals, dtype=torch.long)
                batch[k] = t.to(device, non_blocking=True)

        return batch
