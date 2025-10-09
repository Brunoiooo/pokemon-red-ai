from collections import deque
import time
import keyboard
from pyboy import PyBoy
import os, struct, torch, random, sys, io, math, json, hashlib
import numpy as np
from multiprocessing import Queue
from queue import Full
import torch
from ModelPokemon import ModelPokemon
from threading import Event


class Emulator:
    def __init__(
        self,
        actorId,
        window,
        game,
        maxResetCount,
        ticksPerStep,
        maxMenuSelect,
        maxMenuPosition,
        maxMenuIn,
        maxSameAction,
        worldIllegalMovesMax,
        menuIllegalMovesMax,
        tmpEpsilonSteps,
        epsilon,
        tmpEpsilon,
        wrongDialogActionMax,
        isBest=True,
        td_error_steps=5,
        gamma=0.99,
    ):
        self.ALL_BUTTONS = ["a", "b", "start", "select", "left", "right", "up", "down"]
        self.game = game
        self.window = window
        self.pyboy_init()
        self.maxResetCount = maxResetCount
        self.buttons = [
            [],
            ["a"],
            ["b"],
            ["start"],
            ["select"],
            ["left"],
            ["right"],
            ["up"],
            ["down"],
            ["left", "b"],
            ["right", "b"],
            ["up", "b"],
            ["down", "b"],
        ]
        self.ticksPerStep = ticksPerStep
        self.maxMenuSelect = maxMenuSelect
        self.maxMenuPosition = maxMenuPosition
        self.maxMenuIn = maxMenuIn
        self.averages = np.zeros(1000, dtype=np.float32)
        self.maxSameAction = maxSameAction
        self.worldIllegalMovesMax = worldIllegalMovesMax
        self.menuIllegalMovesMax = menuIllegalMovesMax
        self.actorId = actorId
        self.epsilon = epsilon
        self.tmpEpsilonOn = False
        self.tmpEpsilonStepsCount = 0
        self.tmpEpsilonCooldown = 0
        self.tmpEpsilonSteps = tmpEpsilonSteps
        self.tmpEpsilon = tmpEpsilon
        self.count = 0
        self.wrongDialogActionMax = wrongDialogActionMax
        self.visitedPositionsCount = {}
        self.visitedMaps = []
        self.visitedDialogCount = {}

        ckpt_path = None
        if os.path.exists(f"roms/{self.game}/best.pth") and isBest:
            ckpt_path = f"roms/{self.game}/best.pth"
        elif os.path.exists(f"roms/{self.game}/latest.pth"):
            ckpt_path = f"roms/{self.game}/latest.pth"

        self.modelPokemon = ModelPokemon(len(self.data()), len(self.buttons)).to("cpu")

        self.pyboy.stop(False)
        self.pyboy = None

        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            self.modelPokemon.load_state_dict(
                (
                    state["model_state"]
                    if isinstance(state, dict) and "model_state" in state
                    else state
                ),
                strict=True,
            )
        self.modelPokemon.eval()

        seed = struct.unpack("I", os.urandom(4))[0]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.set_num_threads(1)

        self.doneGraph = {}

        self.gamma = gamma
        self.buffer = deque(maxlen=td_error_steps)

    def pyboy_init(self):
        if self.window:
            self.pyboy = PyBoy(f"roms/{self.game}/rom.gb", sound_emulated=False)
        else:
            self.pyboy = PyBoy(
                f"roms/{self.game}/rom.gb", sound_emulated=False, window="null"
            )

    def counts(self):
        self.dialogCount()
        self.positionCount()
        self.count += 1

    def manual(self):
        self.pyboy_init()
        self.reset(True)
        self.pyboy.set_emulation_speed(1.0)

        while True:
            event = keyboard.read_event(suppress=False)  # blokujące
            if event.event_type != "down":
                continue
            if event.name == "q":
                break

            action = {
                "a": 1,
                "b": 2,
                "enter": 3,
                "shift": 4,
                "left": 5,
                "right": 6,
                "up": 7,
                "down": 8,
            }.get(event.name, 0)

            # TRZYMAJ akację dopóki klawisz trzymany – ALE tykaj pyboy w pętli
            while keyboard.is_pressed(event.name):
                primary_memo_before = bytes(self.pyboy.memory[0x0000:0x10000])
                self.tick(action)  # render + logika w czasie rzeczywistym
                r = self.reward(primary_memo_before, action)
                time.sleep(1)

            self.counts()

            # po puszczeniu można zrobić jeszcze jeden tick z action=0, jeśli chcesz
            # self.tick(0)

            print(
                f"{r:.2f} {self.dialogData()} {self.mapData()} {self.modeFlags()} {self.visitedDialogCount.get(self.mapId(self.pyboy.memory), 0)} {self.terminated} {self.truncated}"
            )
            # print(
            #     f"{event.name} {action} {self.reward(primary_memo_before, action):.2} {self.terminated} {self.truncated}"
            # )

        self.pyboy.stop(False)

    def start(self, dataQ: Queue, conn):
        self.pyboy_init()

        self.reset()

        inputs = self.inputs()

        while True:
            if conn.poll(0.1):
                self.getMsg(conn.recv())
            else:
                inputs = self.train(inputs, dataQ)

                if self.tmpEpsilonOn:
                    self.tmpEpsilonStepsCount += 1

                if not self.tmpEpsilonOn and 0 < self.tmpEpsilonCooldown:
                    self.tmpEpsilonCooldown -= 1

                if self.tmpEpsilonStepsCount >= self.tmpEpsilonSteps:
                    self.tmpEpsilonOn = False
                    self.tmpEpsilonStepsCount = 0
                    self.tmpEpsilonCooldown = self.tmpEpsilonSteps

                if (
                    not self.tmpEpsilonOn
                    and self.epsilon < self.tmpEpsilon
                    and self.average() < 0.5
                    and self.tmpEpsilonCooldown <= 0
                ):
                    self.tmpEpsilonOn = True

    def currentEpsilon(self):
        return self.tmpEpsilon if self.tmpEpsilonOn else self.epsilon

    def average(self):
        return sum(self.averages) / len(self.averages)

    def getMsg(self, msg):
        type = msg["type"]

        if type == "epsilon":
            self.epsilon = msg["value"]
        elif type == "load_state_dict":
            blob = msg["value"]
            buf = io.BytesIO(blob)
            state = torch.load(buf, map_location="cpu")
            with torch.no_grad():
                self.modelPokemon.load_state_dict(state, strict=False)
            self.modelPokemon.eval()
        elif type == "window" and msg["value"] == False:
            self.pyboy.stop(False)
            self.pyboy = PyBoy(
                f"roms/{self.game}/rom.gb", sound_emulated=False, window="null"
            )
            self.reset()
        elif type == "window" and msg["value"] == True:
            self.pyboy.stop(False)
            self.pyboy = PyBoy(f"roms/{self.game}/rom.gb", sound_emulated=False)
            self.reset()

    def auto(self):
        self.pyboy_init()

        self.reset(True)

        self.pyboy.set_emulation_speed(1.0)

        obs = self.inputs()

        while True:
            if keyboard.is_pressed("q"):
                break

            self.doAction(obs, False)

            self.counts()

            obs = self.inputs()

        self.pyboy.stop(False)

    def doAction(self, obs, isEpsilon):
        with torch.inference_mode():
            q = self.modelPokemon(obs)
            q = q.squeeze(0)

        action = int(torch.argmax(q).item())

        if isEpsilon and random.random() < self.currentEpsilon():
            action = random.randint(0, len(self.buttons) - 1)

        action = self.mask_action(action)

        self.tick(action)

        return action

    def evaluate_greedy(self, episodes):
        self.pyboy_init()

        total = 0.0

        for _ in range(episodes):
            self.reset(True)
            obs = self.inputs()
            ep_ret = 0.0

            while True:
                before = self.pyboy.memory[0x0000:0x10000]

                # print(
                #     f"{self.dialogData()} {self.mapData()} {self.modeFlags()} {self.visitedDialogCount.get(self.mapId(self.pyboy.memory), 0)}"
                # )

                action = self.doAction(obs, False)

                r = self.reward(before, action)

                # print(f"{r:.2f}")

                # if self.done:
                #     print("================================")

                ep_ret += float(r)

                if self.truncated:
                    break

                self.counts()

                obs = self.inputs()

                sys.stdout.write(
                    f"\rAvg: {((total / episodes) * 100):.2f}% ep_ret: {ep_ret:.2f} button: {self.buttons[action]} episode: {_}"
                )
                sys.stdout.flush()

            total += ep_ret

        self.pyboy.stop(False)

        return total / episodes

    def train(self, inputs, dataQ: Queue):
        primary_memo_before = bytes(self.pyboy.memory[0x0000:0x10000])

        action = self.doAction(inputs, True)

        reward = self.reward(primary_memo_before, action)

        self.counts()

        next_state = self.inputs()

        self.averages = np.roll(self.averages, -1, axis=0)
        self.averages[-1] = reward

        self.buffer.append(
            (
                self.detach_to_cpu(inputs),
                action,
                float(reward),
                self.detach_to_cpu(next_state),
            )
        )

        if self.done or len(self.buffer) >= self.buffer.maxlen:
            try:
                while True:
                    R, disc = 0.0, 1.0
                    s_0, a_0 = self.buffer[0][0], self.buffer[0][1]

                    for s, a, r, _ in list(self.buffer):
                        R += disc * r
                        disc *= self.gamma

                    dataQ.put_nowait(
                        (
                            self.detach_to_cpu(s_0),
                            a_0,
                            float(R),
                            self.detach_to_cpu(self.buffer[len(self.buffer) - 1][3]),
                            bool(self.terminated),
                            bool(self.truncated),
                            len(self.buffer),
                        )
                    )

                    if self.done and len(self.buffer) >= 2:
                        self.buffer.popleft()
                    else:
                        break
            except Full:
                time.sleep(0.001)
                pass

        if self.done:
            if self.terminated:
                self.saveGameState()

            self.reset()
            return self.inputs()

        return next_state

    def saveGameState(self):
        hashPath = self.getHashPath()

        if os.path.isdir(hashPath):
            return

        os.makedirs(hashPath, exist_ok=True)
        with open(f"{hashPath}/checkpoint.state", "wb") as f:
            self.pyboy.save_state(f)

        meta_path = f"{hashPath}/meta.json"
        meta = {
            "visitedPositionsCount": {
                str(position): int(count)
                for position, count in self.visitedPositionsCount.items()
            },
            "visitedMaps": self.visitedMaps,
            "visitedDialogCount": {
                int(map_id): int(count)
                for map_id, count in self.visitedDialogCount.items()
            },
        }
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False)

    def getHashPath(self):
        return f"roms/{self.game}/saves/{self.getHash()}"

    def getHash(self):
        return hashlib.sha256(
            bytes(
                [
                    self.pyboy.memory[0xD5AB],
                    self.pyboy.memory[0xD5F3],
                    self.pyboy.memory[0xD60D],
                    self.pyboy.memory[0xD710],
                    self.pyboy.memory[0xD72E],
                    self.pyboy.memory[0xD751],
                    self.pyboy.memory[0xD755],
                    self.pyboy.memory[0xD75E],
                    self.pyboy.memory[0xD773],
                    self.pyboy.memory[0xD77C],
                    self.pyboy.memory[0xD782],
                    self.pyboy.memory[0xD792],
                    self.pyboy.memory[0xD79A],
                    self.pyboy.memory[0xD7B3],
                    self.pyboy.memory[0xD7D4],
                    self.pyboy.memory[0xD7D8],
                    self.pyboy.memory[0xD7E0],
                    self.pyboy.memory[0xD7EE],
                    *[self.pyboy.memory[i] for i in range(0xD2F7, 0xD31C)],
                    self.pyboy.memory[0xD356],
                    self.mapId(self.pyboy.memory),
                    self.positionX(self.pyboy.memory),
                    self.positionY(self.pyboy.memory),
                ]
            )
        ).hexdigest()

    def reward(self, primary_memo_before, action):
        reward = 0

        # Player's Substitute HP
        if primary_memo_before[0xCCD7] < self.pyboy.memory[0xCCD7]:
            reward += 1
        elif primary_memo_before[0xCCD7] > self.pyboy.memory[0xCCD7]:
            reward -= 1

        # Enemy Substitute HP
        if primary_memo_before[0xCCD8] > self.pyboy.memory[0xCCD8]:
            reward += 1
        elif primary_memo_before[0xCCD8] < self.pyboy.memory[0xCCD8]:
            reward -= 1

        # Player move that the enemy disabled
        if primary_memo_before[0xCCEE] != self.pyboy.memory[0xCCEE]:
            reward += 1 if self.pyboy.memory[0xCCEE] == 0 else -1

        # Enemy move that the player disabled
        if (
            primary_memo_before[0xCCEF] != self.pyboy.memory[0xCCEF]
            and self.pyboy.memory[0xCCEF] != 0
        ):
            reward += 1

        # Enemy's HP
        if (
            primary_memo_before[0xCFE7] << 8 | primary_memo_before[0xCFE6]
            > self.pyboy.memory[0xCFE7] << 8 | self.pyboy.memory[0xCFE6]
        ):
            reward += 1

        # Enemy's Status
        reward += self.rewardStatus(
            primary_memo_before[0xCFE8], self.pyboy.memory[0xCFE8]
        )

        # Pokémon 1st Slot (In-Battle)
        if (
            primary_memo_before[0xD016] << 8 | primary_memo_before[0xD015]
            < self.pyboy.memory[0xD016] << 8 | self.pyboy.memory[0xD015]
        ):  # Current HP
            reward += 1

        # Status
        reward -= self.rewardStatus(
            primary_memo_before[0xD018], self.pyboy.memory[0xD018]
        )

        # Critical Hit / OHKO Flag
        if primary_memo_before[0xD05E] != self.pyboy.memory[0xD05E]:
            reward += self.pyboy.memory[0xD05E]

        if (
            primary_memo_before[0xD05F] != self.pyboy.memory[0xD05F]
            and self.pyboy.memory[0xD05F] == 1
        ):
            reward += 1

        # Pokémon 1
        if (
            primary_memo_before[0xD16D] << 8 | primary_memo_before[0xD16C]
            < self.pyboy.memory[0xD16D] << 8 | self.pyboy.memory[0xD16C]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD16D] << 8 | primary_memo_before[0xD16C]
            > self.pyboy.memory[0xD16D] << 8 | self.pyboy.memory[0xD16C]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD16F], self.pyboy.memory[0xD16F]
        )  # Status

        if primary_memo_before[0xD18C] < self.pyboy.memory[0xD18C]:  # Level
            reward += 1
        elif primary_memo_before[0xD18C] > self.pyboy.memory[0xD18C]:
            reward -= 1

        # Pokémon 2
        if (
            primary_memo_before[0xD199] << 8 | primary_memo_before[0xD198]
            < self.pyboy.memory[0xD199] << 8 | self.pyboy.memory[0xD198]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD199] << 8 | primary_memo_before[0xD198]
            > self.pyboy.memory[0xD199] << 8 | self.pyboy.memory[0xD198]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD19B], self.pyboy.memory[0xD19B]
        )  # Status

        if primary_memo_before[0xD1B8] < self.pyboy.memory[0xD1B8]:  # Level
            reward += 1
        elif primary_memo_before[0xD1B8] > self.pyboy.memory[0xD1B8]:
            reward -= 1

        # Pokémon 3
        if (
            primary_memo_before[0xD1C5] << 8 | primary_memo_before[0xD1C4]
            < self.pyboy.memory[0xD1C5] << 8 | self.pyboy.memory[0xD1C4]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD1C5] << 8 | primary_memo_before[0xD1C4]
            > self.pyboy.memory[0xD1C5] << 8 | self.pyboy.memory[0xD1C4]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD1C7], self.pyboy.memory[0xD1C7]
        )  # Status

        if primary_memo_before[0xD1E4] < self.pyboy.memory[0xD1E4]:  # Level
            reward += 1
        elif primary_memo_before[0xD1E4] > self.pyboy.memory[0xD1E4]:
            reward -= 1

        # Pokémon 4
        if (
            primary_memo_before[0xD1F1] << 8 | primary_memo_before[0xD1F0]
            < self.pyboy.memory[0xD1F1] << 8 | self.pyboy.memory[0xD1F0]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD1F1] << 8 | primary_memo_before[0xD1F0]
            > self.pyboy.memory[0xD1F1] << 8 | self.pyboy.memory[0xD1F0]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD1F3], self.pyboy.memory[0xD1F3]
        )  # Status

        if primary_memo_before[0xD210] < self.pyboy.memory[0xD210]:  # Level
            reward += 1
        elif primary_memo_before[0xD210] > self.pyboy.memory[0xD210]:
            reward -= 1

        # Pokémon 5
        if (
            primary_memo_before[0xD21D] << 8 | primary_memo_before[0xD21C]
            < self.pyboy.memory[0xD21D] << 8 | self.pyboy.memory[0xD21C]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD21D] << 8 | primary_memo_before[0xD21C]
            > self.pyboy.memory[0xD21D] << 8 | self.pyboy.memory[0xD21C]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD21F], self.pyboy.memory[0xD21F]
        )  # Status

        if primary_memo_before[0xD23C] < self.pyboy.memory[0xD23C]:  # Level
            reward += 1
        elif primary_memo_before[0xD23C] > self.pyboy.memory[0xD23C]:
            reward -= 1

        # Pokémon 6
        if (
            primary_memo_before[0xD249] << 8 | primary_memo_before[0xD248]
            < self.pyboy.memory[0xD249] << 8 | self.pyboy.memory[0xD248]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD249] << 8 | primary_memo_before[0xD248]
            > self.pyboy.memory[0xD249] << 8 | self.pyboy.memory[0xD248]
        ):
            reward -= 1

        reward -= self.rewardStatus(
            primary_memo_before[0xD24B], self.pyboy.memory[0xD24B]
        )  # Status

        if primary_memo_before[0xD268] < self.pyboy.memory[0xD268]:  # Level
            reward += 1
        elif primary_memo_before[0xD268] > self.pyboy.memory[0xD268]:
            reward -= 1

        # Pokedex
        for i in range(0xD2F7, 0xD31C):
            reward += self.rewardPokedex(primary_memo_before[i], self.pyboy.memory[i])

        # Miscellaneous
        reward += self.rewardBadges(
            primary_memo_before[0xD356], self.pyboy.memory[0xD356]
        )

        # Starters Back?
        if (
            primary_memo_before[0xD5AB] != self.pyboy.memory[0xD5AB]
            and self.pyboy.memory[0xD5AB]
        ):
            reward += 30
            self.updateDoneGraph("Starters Back", True)

        # Have Town map?
        if (
            primary_memo_before[0xD5F3] != self.pyboy.memory[0xD5F3]
            and self.pyboy.memory[0xD5F3]
        ):
            reward += 30
            self.updateDoneGraph("Have Town map?", True)

        # Have Oak's Parcel?
        if (
            primary_memo_before[0xD60D] != self.pyboy.memory[0xD60D]
            and self.pyboy.memory[0xD60D]
        ):
            reward += 30
            self.updateDoneGraph("Have Oak's Parcel?", True)

        # Fossilized Pokémon?
        if (
            primary_memo_before[0xD710] != self.pyboy.memory[0xD710]
            and self.pyboy.memory[0xD710]
        ):
            reward += 30
            self.updateDoneGraph("Fossilized Pokémon?", True)

        # Did you get Lapras Yet?
        if (
            primary_memo_before[0xD72E] != self.pyboy.memory[0xD72E]
            and self.pyboy.memory[0xD72E]
        ):
            reward += 30
            self.updateDoneGraph("Did you get Lapras Yet?", True)

        # Fought Giovanni Yet?
        if (
            primary_memo_before[0xD751] != self.pyboy.memory[0xD751]
            and self.pyboy.memory[0xD751]
        ):
            reward += 30
            self.updateDoneGraph("Fought Giovanni Yet?", True)

        # Fought Brock Yet?
        if (
            primary_memo_before[0xD755] != self.pyboy.memory[0xD755]
            and self.pyboy.memory[0xD755]
        ):
            reward += 30
            self.updateDoneGraph("Fought Brock Yet?", True)

        # Fought Misty Yet?
        if (
            primary_memo_before[0xD75E] != self.pyboy.memory[0xD75E]
            and self.pyboy.memory[0xD75E]
        ):
            reward += 30
            self.updateDoneGraph("Fought Misty Yet?", True)

        # Fought Lt. Surge Yet?
        if (
            primary_memo_before[0xD773] != self.pyboy.memory[0xD773]
            and self.pyboy.memory[0xD773]
        ):
            reward += 30
            self.updateDoneGraph("Fought Lt. Surge Yet?", True)

        # Fought Erika Yet?
        if (
            primary_memo_before[0xD77C] != self.pyboy.memory[0xD77C]
            and self.pyboy.memory[0xD77C]
        ):
            reward += 30
            self.updateDoneGraph("Fought Erika Yet?", True)

        # Fought Articuno Yet?
        if (
            primary_memo_before[0xD782] != self.pyboy.memory[0xD782]
            and self.pyboy.memory[0xD782]
        ):
            reward += 30
            self.updateDoneGraph("Fought Articuno Yet?", True)

        # Fought Koga Yet?
        if (
            primary_memo_before[0xD792] != self.pyboy.memory[0xD792]
            and self.pyboy.memory[0xD792]
        ):
            reward += 30
            self.updateDoneGraph("Fought Koga Yet?", True)

        # Fought Blaine Yet?
        if (
            primary_memo_before[0xD79A] != self.pyboy.memory[0xD79A]
            and self.pyboy.memory[0xD79A]
        ):
            reward += 30
            self.updateDoneGraph("Fought Blaine Yet?", True)

        # Fought Sabrina Yet?
        if (
            primary_memo_before[0xD7B3] != self.pyboy.memory[0xD7B3]
            and self.pyboy.memory[0xD7B3]
        ):
            reward += 30
            self.updateDoneGraph("Fought Sabrina Yet?", True)

        # Fought Zapdos Yet?
        if (
            primary_memo_before[0xD7D4] != self.pyboy.memory[0xD7D4]
            and self.pyboy.memory[0xD7D4]
        ):
            reward += 30
            self.updateDoneGraph("Fought Zapdos Yet?", True)

        # Fought Snorlax Yet (Vermilion)
        if (
            primary_memo_before[0xD7D8] != self.pyboy.memory[0xD7D8]
            and self.pyboy.memory[0xD7D8]
        ):
            reward += 30
            self.updateDoneGraph("Fought Snorlax Yet (Vermilion)", True)

        # Fought Snorlax Yet? (Celadon)
        if (
            primary_memo_before[0xD7E0] != self.pyboy.memory[0xD7E0]
            and self.pyboy.memory[0xD7E0]
        ):
            reward += 30
            self.updateDoneGraph("Fought Snorlax Yet? (Celadon)", True)

        # Fought Moltres Yet?
        if (
            primary_memo_before[0xD7EE] != self.pyboy.memory[0xD7EE]
            and self.pyboy.memory[0xD7EE]
        ):
            reward += 30
            self.updateDoneGraph("Fought Moltres Yet", True)

        # Opponent Trainer’s Pokémon
        # Pokémon 1
        if (
            primary_memo_before[0xD8A6] << 8 | primary_memo_before[0xD8A5]
            > self.pyboy.memory[0xD8A6] << 8 | self.pyboy.memory[0xD8A5]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD8A6] << 8 | primary_memo_before[0xD8A5]
            < self.pyboy.memory[0xD8A6] << 8 | self.pyboy.memory[0xD8A5]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD8A8], self.pyboy.memory[0xD8A8]
        )  # Status

        # Pokémon 2
        if (
            primary_memo_before[0xD8D2] << 8 | primary_memo_before[0xD8D1]
            > self.pyboy.memory[0xD8D2] << 8 | self.pyboy.memory[0xD8D1]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD8D2] << 8 | primary_memo_before[0xD8D1]
            < self.pyboy.memory[0xD8D2] << 8 | self.pyboy.memory[0xD8D1]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD8D4], self.pyboy.memory[0xD8D4]
        )  # Status

        # Pokémon 3
        if (
            primary_memo_before[0xD8FE] << 8 | primary_memo_before[0xD8FD]
            > self.pyboy.memory[0xD8FE] << 8 | self.pyboy.memory[0xD8FD]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD8FE] << 8 | primary_memo_before[0xD8FD]
            < self.pyboy.memory[0xD8FE] << 8 | self.pyboy.memory[0xD8FD]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD900], self.pyboy.memory[0xD900]
        )  # Status

        # Pokémon 4
        if (
            primary_memo_before[0xD92A] << 8 | primary_memo_before[0xD929]
            > self.pyboy.memory[0xD92A] << 8 | self.pyboy.memory[0xD929]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD92A] << 8 | primary_memo_before[0xD929]
            < self.pyboy.memory[0xD92A] << 8 | self.pyboy.memory[0xD929]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD92C], self.pyboy.memory[0xD92C]
        )  # Status

        # Pokémon 5
        if (
            primary_memo_before[0xD956] << 8 | primary_memo_before[0xD955]
            > self.pyboy.memory[0xD956] << 8 | self.pyboy.memory[0xD955]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD956] << 8 | primary_memo_before[0xD955]
            < self.pyboy.memory[0xD956] << 8 | self.pyboy.memory[0xD955]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD958], self.pyboy.memory[0xD958]
        )  # Status

        # Pokémon 6
        if (
            primary_memo_before[0xD982] << 8 | primary_memo_before[0xD981]
            > self.pyboy.memory[0xD982] << 8 | self.pyboy.memory[0xD981]
        ):  # Current HP
            reward += 1
        elif (
            primary_memo_before[0xD982] << 8 | primary_memo_before[0xD981]
            < self.pyboy.memory[0xD982] << 8 | self.pyboy.memory[0xD981]
        ):
            reward -= 1

        reward += self.rewardStatus(
            primary_memo_before[0xD984], self.pyboy.memory[0xD984]
        )  # Status

        reward += self.rewardDialog()
        reward += self.rewardPosition()
        reward += self.rewardMap()
        reward += self.panishWorldIllegalMoves(primary_memo_before, reward)
        reward += self.panishMenuIllegalMoves(primary_memo_before)
        reward += self.panishMenuSelect(primary_memo_before)
        reward += self.panishMenuPosition(primary_memo_before)
        reward += self.panishMenuIn(reward)
        reward += self.panishSameAction(action, reward)
        reward += self.panishWrongDialogAction(primary_memo_before, action)
        reward += self.panishBacktrack()
        self.countingReward(reward)
        reward -= 0.01

        return reward

    def mask_action(self, action: int) -> int:
        # jeśli mamy cooldown po zmianie menu – nie pozwalaj na kolejne przełączenie
        if self.menuCooldown > 0:
            # „start” jest u Ciebie pod action==3 (["start"])
            if action == 3:  # start
                # zamień na „nic” lub bezpieczną akcję
                return 0  # [] -> no-op (naciskasz nic)
        # gdy jesteśmy w menu, preferuj nawigację lub wyjście 'b'
        if self.isMenu():
            # dopuść tylko: a(1), b(2), up(7), down(8), left(5), right(6)
            allowed = {0, 1, 2, 5, 6, 7, 8}
            if action not in allowed:
                return 2  # 'b' żeby cofać menu zamiast robić dziwne rzeczy
        # gdy jesteśmy w świecie, ogranicz spam „start”
        if self.isWorld() and self.menuCooldown > 0 and action == 3:
            return 0

        return action

    def getPosition(self):
        return f"{self.positionX(self.pyboy.memory)}x{self.positionY(self.pyboy.memory)}x{self.mapId(self.pyboy.memory)}"

    def panishBacktrack(self):
        if not self.isWorld():
            return 0

        if self.lastPosition0 == self.getPosition():
            reward = -0.2
        else:
            reward = 0

        self.lastPosition0 = self.lastPosition1
        self.lastPosition1 = self.getPosition()

        return reward

    def panishWrongDialogAction(self, primary_memo_before, action):
        if not self.isDialog() or action in (0, 1, 2):
            self.wrongDialogActionCount = 0
            return 0
        if not self.isSameMenuSelected(primary_memo_before):
            self.wrongDialogActionCount = 0
            return 0
        self.wrongDialogActionCount += 1
        if self.wrongDialogActionMax < self.wrongDialogActionCount:
            self.updateDoneGraph(f"panishWrongDialogAction-{action}")
        return -1.0

    def mapId(self, memory):
        return memory[0xD35E]

    def rewardDialog(self):
        if not self.isDialog():
            return 0

        return 0.2 - 0.04 * self.visitedDialogCount.get(
            self.mapId(self.pyboy.memory), 0
        )

    def dialogCount(self):
        if not self.isDialog():
            return

        map = self.mapId(self.pyboy.memory)

        if map not in self.visitedDialogCount:
            self.visitedDialogCount[map] = 0

        self.visitedDialogCount[map] += 1

    def positionCount(self):
        if not self.isWorld():
            return

        position = self.getPosition()

        if position not in self.visitedPositionsCount:
            self.visitedPositionsCount[position] = 0

        self.visitedPositionsCount[position] += 1

    def rewardPokedex(self, before, after):
        reward = 0

        for i in range(8):
            if after & (1 << i) and before & (1 << i) != after & (1 << i):
                reward += 20
                self.updateDoneGraph(f"rewardPokedex-{i}", True)

        return reward

    def rewardBadges(self, before, after):
        reward = 0

        for i in range(8):
            if after & (1 << i) and before & (1 << i) != after & (1 << i):
                reward += 40
                self.updateDoneGraph(f"rewardBadges-{i}", True)

        return reward

    def panishSameAction(self, action, reward):
        if reward > 0:
            self.sameActionCount = 0
            self.lastAction = action
            return 0.0
        if action == self.lastAction:
            self.sameActionCount += 1
        else:
            self.sameActionCount = 0
            self.lastAction = action

        return -0.2 if self.sameActionCount > self.maxSameAction else 0.0

    def panishMenuIn(self, reward):
        if not self.isMenu() or 0 < reward:
            self.menuInCount = 0
            self.menuSelectCount = 0
            self.menuPositionCount = 0
            return 0

        self.menuInCount += 1

        return -0.4 if self.maxMenuIn < self.menuInCount else 0

    def panishMenuPosition(self, primary_memo_before):
        if not self.isMenu():
            self.menuPositionCount = 0
            return 0

        if self.isSameMenuPosition(primary_memo_before):
            self.menuPositionCount += 1
        else:
            self.menuSelectCount = 0
            self.menuPositionCount = 0

        return -0.3 if self.maxMenuPosition < self.menuPositionCount else 0

    def isSameMenuSelected(self, primary_memo_before):
        return (
            True
            if primary_memo_before[0xCC26] == self.pyboy.memory[0xCC26]
            and self.isSameMenuPosition(primary_memo_before)
            else False
        )

    def panishMenuSelect(self, primary_memo_before):
        if not self.isMenu():
            self.menuSelectCount = 0
            return 0

        if self.isSameMenuSelected(primary_memo_before):
            self.menuSelectCount += 1
        else:
            self.menuSelectCount = 0

        return -0.2 if self.maxMenuSelect < self.menuSelectCount else 0

    def panishSwitchMenu(self):
        wasMenu = self.lastMenu
        isMenu = self.isMenu()
        self.lastMenu = isMenu

        # rosnąca kara za siedzenie w menu – żeby uciekał
        in_menu_penalty = -0.2 if isMenu else 0.0  # było -0.1

        if wasMenu != isMenu:
            self.menuToggleStreak += 1
            self.menuCooldown = self.MENU_COOLDOWN_STEPS
            # rosnąca kara za przełączanie (np. -0.5, -1.0, -1.5, …)
            return in_menu_penalty + (-0.5 * self.menuToggleStreak)
        else:
            # jeśli nie przełączono – streak maleje powoli
            if self.menuToggleStreak > 0:
                self.menuToggleStreak -= 1
            # cooldown „tyka”
            if self.menuCooldown > 0:
                self.menuCooldown -= 1
            return in_menu_penalty

    def positionX(self, memory):
        return memory[0xD361]

    def positionY(self, memory):
        return memory[0xD362]

    def rewardPosition(self):
        if not self.isWorld():
            return 0.0

        return 0.2 - 0.04 * self.visitedPositionsCount.get(self.getPosition(), 0)

    def rewardMap(self):
        mapId = self.mapId(self.pyboy.memory)

        if not self.isWorld() or mapId == 0:
            return 0.0

        if len(self.visitedMaps) <= 0:
            self.visitedMaps.append(mapId)
            return 0.0

        if mapId in self.visitedMaps:
            return 0.0

        self.visitedMaps.append(mapId)
        self.updateDoneGraph(f"rewardMap-{mapId}", True)
        return 10.0

    def isSameMenuPosition(self, primary_memo_before):
        return (
            True
            if primary_memo_before[0xCC24] == self.pyboy.memory[0xCC24]
            and primary_memo_before[0xCC25] == self.pyboy.memory[0xCC25]
            else False
        )

    def panishMenuIllegalMoves(self, primary_memo_before):
        if not self.isMenu() or not self.isSameMenuPosition(primary_memo_before):
            self.menuIllegalMovesCount = 0
            return 0

        self.menuIllegalMovesCount += 1

        if self.menuIllegalMovesMax < self.menuIllegalMovesCount:
            self.updateDoneGraph("panishMenuIllegalMoves")

        if 1 < self.menuIllegalMovesCount:
            return -3
        else:
            return 0

    def isSameWorldPosition(self, primary_memo_before):
        return (
            True
            if self.mapId(primary_memo_before) == self.mapId(self.pyboy.memory)
            and self.positionX(primary_memo_before) == self.positionX(self.pyboy.memory)
            and self.positionY(primary_memo_before) == self.positionY(self.pyboy.memory)
            else False
        )

    def panishWorldIllegalMoves(self, primary_memo_before, reward):
        if (
            not self.isWorld()
            or not self.isSameWorldPosition(primary_memo_before)
            or 0 < reward
        ):
            self.worldIllegalMovesCount = 0
            return 0

        self.worldIllegalMovesCount += 1

        if self.worldIllegalMovesMax < self.worldIllegalMovesCount:
            self.updateDoneGraph("panishWorldIllegalMoves")

        if 1 < self.worldIllegalMovesCount:
            return -3
        else:
            return 0

    def rewardStatus(self, before, after):
        reward = 0

        if after & (1 << 6) and before & (1 << 6) != after & (1 << 6):  # Paralyzed
            reward += 1
        if after & (1 << 5) and before & (1 << 5) != after & (1 << 5):  # Frozen
            reward += 1
        if after & (1 << 4) and before & (1 << 4) != after & (1 << 4):  # Burned
            reward += 1
        if after & (1 << 3) and before & (1 << 3) != after & (1 << 3):  # Poisoned
            reward += 1

        return reward

    def reset(self, stateStart=False):
        base = f"roms/{self.game}/saves"
        dir = "start"

        if not stateStart:
            dir = random.choice(os.listdir(base))

        try:
            with open(f"{base}/{dir}/checkpoint.state", "rb") as load_file:
                self.pyboy.load_state(load_file)
        except Exception:
            dir = "start"
            with open(f"{base}/{dir}/checkpoint.state", "rb") as load_file:
                self.pyboy.load_state(load_file)

        try:
            with open(f"{base}/{dir}/meta.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            vp = data.get("visitedPositionsCount", {})
            self.visitedPositionsCount = {
                str(position): int(count) for position, count in vp.items()
            }

            self.visitedMaps = data.get("visitedMaps", [])

            vdc = data.get("visitedDialogCount", {})
            self.visitedDialogCount = {
                int(map_id): int(count) for map_id, count in vdc.items()
            }
        except FileNotFoundError:
            self.visitedPositionsCount = {}
            self.visitedMaps = []
            self.visitedDialogCount = {}

        self.resetCount = 0
        self.menuSelectCount = 0
        self.menuPositionCount = 0
        self.menuInCount = 0
        self.sameActionCount = 0
        self.lastAction = -1
        self.worldIllegalMovesCount = 0
        self.menuIllegalMovesCount = 0
        self.terminated = False
        self.truncated = False
        self.wrongDialogActionCount = 0
        self.lastMenu = self.isMenu()
        self.lastPosition0 = self.getPosition()
        self.lastPosition1 = self.getPosition()
        self.MENU_COOLDOWN_STEPS = math.ceil(0.8 * 60 / self.ticksPerStep)
        self.menuCooldown = 0
        self.menuToggleStreak = 0
        self.buffer.clear()

    def data(self):
        data = []

        data += self.dialogData()

        data += self.mapData()

        data += self.spriteData()

        data += self.menuData()

        data += self.battleData()

        data += self.pokeMartData()

        data += self.playerData()

        data += self.pokedexData()

        data += self.itemsData()

        data += self.moneyData()

        data += self.rivalData()

        data += self.miscellaneousData()

        data += self.storedItemsData()

        data += self.gameCoinsData()

        data += self.eventFlagsData()

        data += self.opponentTrainersPokemonData()

        return data

    def modeFlags(self):
        return [
            int(self.isWorld()),
            int(self.isMenu()),
            int(self.isDialog()),
            int(self.isBattle()),
        ]

    def dialogId(self, memory):
        return memory[0xCF13]

    def inputs(self):
        arr = np.asarray(self.data(), dtype=np.float32)
        arr = np.clip(arr, 0, 255) / 255.0
        continuous = torch.from_numpy(arr).unsqueeze(0).to("cpu")

        map_id = torch.tensor([int(self.mapId(self.pyboy.memory))], dtype=torch.long)
        dialog_id = torch.tensor(
            [int(self.dialogId(self.pyboy.memory))], dtype=torch.long
        )
        pos_x = torch.tensor([int(self.positionX(self.pyboy.memory))], dtype=torch.long)
        pos_y = torch.tensor([int(self.positionY(self.pyboy.memory))], dtype=torch.long)

        mode = torch.tensor([int(np.argmax(self.modeFlags()))], dtype=torch.long)

        return {
            "map_id": map_id,
            "dialog_id": dialog_id,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "mode": mode,
            "continuous": continuous,
        }

    def dialogData(self):
        data = [
            self.visitedDialogCount.get(self.mapId(self.pyboy.memory), 0),
        ]

        return data if self.isDialog() else [0] * len(data)

    def mapData(self):
        data = [
            self.visitedPositionsCount.get(self.getPosition(), 0),
        ]

        return data if self.isWorld() else [0] * len(data)

    def isBattle(self):
        return True if self.pyboy.memory[0xD057] else False

    def isMenu(self):
        return (
            True
            if self.isBlocked() and self.dialogId(self.pyboy.memory) == 0
            else False
        )

    def isBlocked(self):
        return True if self.pyboy.memory[0xCFC4] else False

    def isDialog(self):
        return (
            True
            if self.isBlocked() and self.dialogId(self.pyboy.memory) != 0
            else False
        )

    def isWorld(self):
        return (
            True
            if not self.isBlocked() and not self.isBattle() and not self.isMenu()
            else False
        )

    def spriteData(self):
        return (
            [self.pyboy.memory[i] for i in range(0xC100, 0xC300)]
            if self.isWorld()
            else [0] * (0xC300 - 0xC100)
        )

    def menuData(self):
        data = [self.pyboy.memory[i] for i in range(0xCC24, 0xCC30)]

        return data if self.isMenu() else [0] * len(data)

    def battleData(self):
        data = (
            [self.pyboy.memory[0xCCD5]]
            + [self.pyboy.memory[i] for i in range(0xCCD7, 0xCCD9)]
            + [self.pyboy.memory[i] for i in range(0xCCDB, 0xCCDE)]
            + [self.pyboy.memory[i] for i in range(0xCCE8, 0xCCEA)]
            + [self.pyboy.memory[i] for i in range(0xCCED, 0xCCF0)]
            + [self.pyboy.memory[0xCCF6]]
            + [self.pyboy.memory[i] for i in range(0xCD05, 0xCD07)]
            + [self.pyboy.memory[i] for i in range(0xCD1A, 0xCD20)]
            + [self.pyboy.memory[i] for i in range(0xCD2D, 0xCD34)]
            + [self.pyboy.memory[i] for i in range(0xCFCC, 0xCFE8)]
            + self.bitsExtractor(self.pyboy.memory[0xCFE9], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xCFEA, 0xD031)]
            + [self.pyboy.memory[0xD057], self.pyboy.memory[0xD05A]]
            + [self.pyboy.memory[i] for i in range(0xD05C, 0xD060)]
            + self.bitsExtractor(self.pyboy.memory[0xD062])
            + self.bitsExtractor(self.pyboy.memory[0xD063])
            + self.bitsExtractor(self.pyboy.memory[0xD064], 0, 3)
        )

        return data if self.isBattle() else [0] * len(data)

    def pokeMartData(self):
        data = [self.pyboy.memory[i] for i in range(0xCF7B, 0xCF86)]

        return data if self.isMenu() else [0] * len(data)

    def playerData(self):
        data = (
            [self.pyboy.memory[i] for i in range(0xD163, 0xD16F)]
            + self.bitsExtractor(self.pyboy.memory[0xCFE9], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD170, 0xD19B)]
            + self.bitsExtractor(self.pyboy.memory[0xD19B], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD19C, 0xD1C7)]
            + self.bitsExtractor(self.pyboy.memory[0xD1C7], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD1C8, 0xD1F3)]
            + self.bitsExtractor(self.pyboy.memory[0xD1F3], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD1F4, 0xD21F)]
            + self.bitsExtractor(self.pyboy.memory[0xD21F], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD220, 0xD24B)]
            + self.bitsExtractor(self.pyboy.memory[0xD24B], 3, 6)
            + [self.pyboy.memory[i] for i in range(0xD24C, 0xD273)]
        )

        return (
            data
            if self.isBattle() or self.isMenu() or self.isWorld()
            else [0] * len(data)
        )

    def pokedexData(self):
        data = [self.pyboy.memory[i] for i in range(0xD2F7, 0xD31D)]

        return data

    def itemsData(self):
        data = [self.pyboy.memory[i] for i in range(0xD31D, 0xD347)]

        return data if self.isMenu() or self.isBattle() else [0] * len(data)

    def moneyData(self):
        b0 = self.pyboy.memory[0xD347]
        b1 = self.pyboy.memory[0xD348]
        b2 = self.pyboy.memory[0xD349]

        money = (
            (b0 & 0x0F) * 10**5
            + (b0 >> 4) * 10**6
            + (b1 & 0x0F) * 10**3
            + (b1 >> 4) * 10**4
            + (b2 & 0x0F) * 10**1
            + (b2 >> 4) * 10**2
        )

        return [money] if self.isMenu() else [0]

    def rivalData(self):
        data = [self.pyboy.memory[i] for i in range(0xD347, 0xD350)]

        return data if self.isBattle() else [0] * len(data)

    def miscellaneousData(self):
        return self.bitsExtractor(self.pyboy.memory[0xD356]) + [
            self.pyboy.memory[i] for i in range(0xD35F, 0xD361)
        ]

    def storedItemsData(self):
        data = [self.pyboy.memory[i] for i in range(0xD53A, 0xD560)]

        return data if self.isMenu() else [0] * len(data)

    def gameCoinsData(self):
        b0 = self.pyboy.memory[0xD5A4]
        b1 = self.pyboy.memory[0xD5A5]

        coins = (
            (b0 & 0x0F) * 10**1
            + (b0 >> 4) * 10**2
            + (b1 & 0x0F) * 10**3
            + (b1 >> 4) * 10**4
        )

        return [coins] if self.isMenu() else [0]

    def eventFlagsData(self):
        return [
            self.pyboy.memory[0xD5AB],
            self.pyboy.memory[0xD5C0],
            self.pyboy.memory[0xD5F3],
            self.pyboy.memory[0xD60D],
            self.pyboy.memory[0xD700],
            self.pyboy.memory[0xD70B],
            self.pyboy.memory[0xD70C],
            self.pyboy.memory[0xD70D],
            self.pyboy.memory[0xD70E],
            self.pyboy.memory[0xD710],
            self.pyboy.memory[0xD714],
            self.pyboy.memory[0xD72E],
            self.pyboy.memory[0xD732],
            self.pyboy.memory[0xD751],
            self.pyboy.memory[0xD755],
            self.pyboy.memory[0xD75E],
            self.pyboy.memory[0xD773],
            self.pyboy.memory[0xD77C],
            self.pyboy.memory[0xD782],
            self.pyboy.memory[0xD790],
            self.pyboy.memory[0xD792],
            self.pyboy.memory[0xD79A],
            self.pyboy.memory[0xD7B3],
            self.pyboy.memory[0xD7D4],
            self.pyboy.memory[0xD7D8],
            self.pyboy.memory[0xD7E0],
            self.pyboy.memory[0xD7EE],
            self.pyboy.memory[0xD803],
            self.pyboy.memory[0xD85F],
        ]

    def opponentTrainersPokemonData(self):
        data = [self.pyboy.memory[i] for i in range(0xD89C, 0xDA30)]

        return data if self.isBattle() else [0] * len(data)

    def bitsExtractor(self, byte, start_bit=0, end_bit=7):
        if start_bit < 0 or end_bit > 7 or start_bit > end_bit:
            raise ValueError("Invalid bit range")

        return [1 if (byte & (1 << i)) else 0 for i in range(start_bit, end_bit + 1)]

    def countingReward(self, reward):
        if reward > 0.2:
            self.resetCount = 0
        else:
            self.resetCount += 1
        if self.resetCount > self.maxResetCount:
            self.updateDoneGraph("countingReward")

    def tick(self, action):
        for button in self.buttons[action]:
            self.pyboy.button_press(button)

        self.pyboy.tick(self.ticksPerStep / 2)

        for i in range(len(self.ALL_BUTTONS)):
            self.pyboy.button_release(self.ALL_BUTTONS[i])

        self.pyboy.tick(self.ticksPerStep / 2)

    def updateDoneGraph(self, key: str, terminated: bool = False):
        self.doneGraph[key] = self.doneGraph.get(key, 0) + 1
        if terminated:
            self.terminated = True
        else:
            self.truncated = True

    def detach_to_cpu(self, obs_dict):
        out = {}
        for k, v in obs_dict.items():
            if torch.is_tensor(v):
                out[k] = v.detach().to("cpu")
            else:
                out[k] = v
        return out

    @property
    def done(self):
        return self.terminated or self.truncated
