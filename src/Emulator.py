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
        self._battleCount = 0

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

    def counts(self, before, after):
        self.dialogCount()
        self.positionCount()
        self.battleCount(before, after)
        self.count += 1

    def manual(self):
        self.pyboy_init()
        self.reset(path="start_manual")
        self.pyboy.set_emulation_speed(1.0)

        while True:
            event = keyboard.read_event(suppress=False)
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
                "t": 9,
            }.get(event.name, 0)

            if action == 9:
                action = 0
                os.makedirs(f"roms/{self.game}/saves/start_manual", exist_ok=True)
                with open(
                    f"roms/{self.game}/saves/start_manual/checkpoint.state", "wb"
                ) as f:
                    self.pyboy.save_state(f)

            while keyboard.is_pressed(event.name):
                primary_memo_before = bytes(self.pyboy.memory[0x0000:0x10000])
                self.tick(action)
                r = self.reward(primary_memo_before, action)
                time.sleep(1)

            self.counts(primary_memo_before, self.pyboy.memory)

            print(self.modeFlags())
            if self.isWorld():
                print(
                    f"{r:.2f} isWorld mapData: {self.mapData()} position: {self.getPosition()}"
                )
            elif self.isDialog:
                print(
                    f"{r:.2f} isDialog {self.dialogData()} mapId: {self.mapId(self.pyboy.memory)}"
                )
            elif self.isBattle():
                print(f"{r:.2f} isBattle mapData: {self.battleData()}")
            elif self.isBlocked():
                print(f"{r:.2f} isBlocked")

        self.pyboy.stop(False)

    def start(self, dataQ: Queue, conn, stop_event: Event = None):
        self.pyboy_init()

        self.reset()

        inputs = self.inputs()

        while True:
            if conn.poll(0.1):
                self.getMsg(conn.recv())
            elif stop_event is not None and stop_event.is_set():
                break
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

        self.pyboy.stop(False)

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

            self.counts(obs, self.pyboy.memory)

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

                self.counts(before, self.pyboy.memory)

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

        self.counts(primary_memo_before, self.pyboy.memory)

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
                    self.startersBack(self.pyboy.memory),
                    self.haveTownMap(self.pyboy.memory),
                    self.haveOaksParcel(self.pyboy.memory),
                    *self.flyAnywhere(self.pyboy.memory),
                    self.fossilizedPokemon(self.pyboy.memory),
                    self.positionInAir(self.pyboy.memory),
                    self.didYouGetLaprasYet(self.pyboy.memory),
                    self.debugNewGame(self.pyboy.memory),
                    self.foughtGiovanniYet(self.pyboy.memory),
                    self.foughtBrockYet(self.pyboy.memory),
                    self.foughtMistyYet(self.pyboy.memory),
                    self.foughtLtSurgeYet(self.pyboy.memory),
                    self.foughtErikaYet(self.pyboy.memory),
                    self.foughtArticunoYet(self.pyboy.memory),
                    self.foughtKogaYet(self.pyboy.memory),
                    self.foughtBlaineYet(self.pyboy.memory),
                    self.foughtSabrinaYet(self.pyboy.memory),
                    self.foughtZapdosYet(self.pyboy.memory),
                    self.foughtSnorlaxYetVermilion(self.pyboy.memory),
                    self.foughtSnorlaxYetCeladon(self.pyboy.memory),
                    self.foughtMoltresYet(self.pyboy.memory),
                    self.isSSAnneHere(self.pyboy.memory),
                    *self.badges(self.pyboy.memory),
                    self.mapId(self.pyboy.memory),
                    self.positionX(self.pyboy.memory),
                    self.positionY(self.pyboy.memory),
                ]
            )
        ).hexdigest()

    def reward(self, primary_memo_before, action):
        reward = 0

        if self.isBattle():
            reward += self.rewardPlayersSubstituteHp(
                primary_memo_before, self.pyboy.memory
            )
            reward += self.rewardEnemySubstituteHp(
                primary_memo_before, self.pyboy.memory
            )
            reward += self.rewardEnemyHp(primary_memo_before, self.pyboy.memory)
            reward += self.rewardEnemyStatus(primary_memo_before, self.pyboy.memory)
            reward += self.rewardPokemonCurrentHP1(
                primary_memo_before, self.pyboy.memory
            )
            reward += self.rewardPokemonStatus1(primary_memo_before, self.pyboy.memory)
            reward += self.rewardCriticalHitFlag(primary_memo_before, self.pyboy.memory)
            reward += self.rewardOneHitKOFlag(primary_memo_before, self.pyboy.memory)

        reward += self.rewardPokedex(primary_memo_before, self.pyboy.memory)
        reward += self.rewardBadges(primary_memo_before, self.pyboy.memory)

        reward += self.rewardEventFlag(
            self.startersBack(primary_memo_before),
            self.startersBack(self.pyboy.memory),
            "Starters Back",
        )

        reward += self.rewardEventFlag(
            self.haveTownMap(primary_memo_before),
            self.haveTownMap(self.pyboy.memory),
            "Have Town map?",
        )

        reward += self.rewardEventFlag(
            self.haveOaksParcel(primary_memo_before),
            self.haveOaksParcel(self.pyboy.memory),
            "Have Oak's Parcel?",
        )

        reward += self.rewardEventFlag(
            self.fossilizedPokemon(primary_memo_before),
            self.fossilizedPokemon(self.pyboy.memory),
            "Fossilized Pokémon?",
        )

        reward += self.rewardEventFlag(
            self.positionInAir(primary_memo_before),
            self.positionInAir(self.pyboy.memory),
            "Position in Air",
        )

        reward += self.rewardEventFlag(
            self.didYouGetLaprasYet(primary_memo_before),
            self.didYouGetLaprasYet(self.pyboy.memory),
            "Did you get Lapras Yet?",
        )

        reward += self.rewardEventFlag(
            self.debugNewGame(primary_memo_before),
            self.debugNewGame(self.pyboy.memory),
            "Debug New Game",
        )

        reward += self.rewardEventFlag(
            self.foughtGiovanniYet(primary_memo_before),
            self.foughtGiovanniYet(self.pyboy.memory),
            "Fought Giovanni Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtBrockYet(primary_memo_before),
            self.foughtBrockYet(self.pyboy.memory),
            "Fought Brock Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtMistyYet(primary_memo_before),
            self.foughtMistyYet(self.pyboy.memory),
            "Fought Misty Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtLtSurgeYet(primary_memo_before),
            self.foughtLtSurgeYet(self.pyboy.memory),
            "Fought Lt. Surge Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtErikaYet(primary_memo_before),
            self.foughtErikaYet(self.pyboy.memory),
            "Fought Erika Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtArticunoYet(primary_memo_before),
            self.foughtArticunoYet(self.pyboy.memory),
            "Fought Articuno Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtKogaYet(primary_memo_before),
            self.foughtKogaYet(self.pyboy.memory),
            "Fought Koga Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtBlaineYet(primary_memo_before),
            self.foughtBlaineYet(self.pyboy.memory),
            "Fought Blaine Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtSabrinaYet(primary_memo_before),
            self.foughtSabrinaYet(self.pyboy.memory),
            "Fought Sabrina Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtZapdosYet(primary_memo_before),
            self.foughtZapdosYet(self.pyboy.memory),
            "Fought Zapdos Yet?",
        )

        reward += self.rewardEventFlag(
            self.foughtSnorlaxYetVermilion(primary_memo_before),
            self.foughtSnorlaxYetVermilion(self.pyboy.memory),
            "Fought Snorlax Yet (Vermilion)",
        )

        reward += self.rewardEventFlag(
            self.foughtSnorlaxYetCeladon(primary_memo_before),
            self.foughtSnorlaxYetCeladon(self.pyboy.memory),
            "Fought Snorlax Yet? (Celadon)",
        )

        reward += self.rewardEventFlag(
            self.foughtMoltresYet(primary_memo_before),
            self.foughtMoltresYet(self.pyboy.memory),
            "Fought Moltres Yet?",
        )

        reward += self.rewardEventFlag(
            self.isSSAnneHere(primary_memo_before),
            self.isSSAnneHere(self.pyboy.memory),
            "Is SS Anne here?",
        )

        reward += self.rewardDialog()
        reward += self.rewardBattle()
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
        reward -= 0.01

        return reward

    def rewardBattle(self):
        if not self.isBattle():
            return 0.0

        reward = 0.2 - 0.01 * self._battleCount

        if reward < -0.5:
            self.updateDoneGraph("rewardBattle")

        return reward

    def mask_action(self, action: int) -> int:
        if self.menuCooldown > 0:
            if action == 3:
                return 0 
            
        if self.isBattle():
            return action

        if self.isDialog():
            return 2 if action not in {0, 1, 2} else action
            
        if self.isMenu():
            return 2 if action not in {0, 1, 2, 5, 6, 7, 8} else action
                
        if self.isWorld() and self.menuCooldown > 0 and action == 3:
            return 0
        
        if self.isBlocked():
            return 0

        return action

    def getPosition(self):
        return f"{self.positionX(self.pyboy.memory)}x{self.positionY(self.pyboy.memory)}x{self.mapId(self.pyboy.memory)}"

    def panishBacktrack(self):
        if not self.isWorld():
            return 0

        if (
            self.lastPosition0 == self.getPosition()
            and self.lastPosition0 != self.lastPosition1
        ):
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
            return 0.0

        reward = 0.2 - 0.01 * self.visitedDialogCount.get(
            self.mapId(self.pyboy.memory), 0
        )

        if reward < -0.5:
            self.updateDoneGraph("rewardDialog")

        return reward

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

    def battleCount(self, before, after):
        if not self.isBattle() or self.numberOfTurnsInCurrentBattle(
            before
        ) < self.numberOfTurnsInCurrentBattle(after):
            self._battleCount = 0
        else:
            self._battleCount += 1

    def rewardPokedexOwn(self, before, after):
        reward = 0

        for bit_before, bit_after in zip(self.pokedexOwn(before), self.pokedexOwn(after)):
            if bit_before == 0 and bit_after == 1:
                reward += 100

        return reward

    def rewardPokedexSeen(self, before, after):
        reward = 0

        for bit_before, bit_after in  zip(self.pokedexSeen(before), self.pokedexSeen(after)):
            if bit_before == 0 and bit_after == 1:
                reward += 50

        return reward
    
    def rewardPokedex(self, before, after):
        return self.rewardPokedexOwn(before, after) + self.rewardPokedexSeen(before, after)

    def rewardBadges(self, before, after):
        reward = 0

        for i, (bit_before, bit_after) in enumerate(
            zip(self.badges(before), self.badges(after))
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1000
                self.updateDoneGraph(f"rewardBadges-{i}", True)

        return reward

    def rewardEventFlag(self, before, after, name):
        if before == 0 and after == 1:
            self.updateDoneGraph(name, True)
            return 500

        return 0

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

        reward = 0.2 - 0.01 * self.visitedPositionsCount.get(self.getPosition(), 0)

        if reward < -0.5:
            self.updateDoneGraph("rewardPosition")

        return reward

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
        return 50.0

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

    def reset(self, stateStart=False, path=None):
        base = f"roms/{self.game}/saves"
        dir = "start"

        if path is not None:
            dir = path
        elif not stateStart:
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
        self._battleCount = 0

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

        data += self.miscellaneousData()

        data += self.storedItemsData()

        data += self.gameCoinsData()

        data += self.eventFlagsData()

        data += self.opponentTrainersPokemonData()

        data += self.storedPokemonData()

        return data

    def storedPokemonData(self):
        data = [self.pyboy.memory[i] for i in range(0xDA80, 0xDD2A)]

        return data if self.isMenu() else [0] * len(data)

    def modeFlags(self):
        if self.isBattle():
            return 1
        elif self.isDialog():
            return 2
        elif self.isMenu():
            return 3
        else:
            return 0

    def dialogId(self, memory):
        return memory[0xCF13]

    def inputs(self):
        mode = torch.tensor([self.modeFlags()], dtype=torch.long)

        continuous = torch.tensor(self.data(), dtype=torch.float32)

        map_id = torch.tensor([self.mapId(self.pyboy.memory)], dtype=torch.long)
        pos_x = torch.tensor([self.positionX(self.pyboy.memory)], dtype=torch.long)
        pos_y = torch.tensor([self.positionY(self.pyboy.memory)], dtype=torch.long)
        dialog_id = torch.tensor(
            [self.dialogId(self.pyboy.memory)], dtype=torch.long
        )

        return {
            "mode": mode,
            "continuous": continuous,
            "map_id": map_id,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "dialog_id": dialog_id,
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
        return True if self.typeOfBattle(self.pyboy.memory) else False

    def isMenu(self):
        return (
            True
            if self.isBlocked()
            and self.dialogId(self.pyboy.memory) == 0
            and not self.isBattle()
            else False
        )

    def isBlocked(self):
        return True if self.pyboy.memory[0xCFC4] else False

    def isDialog(self):
        return (
            True
            if self.isBlocked()
            and self.dialogId(self.pyboy.memory) != 0
            and not self.isBattle()
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
        data = [self.pyboy.memory[i] for i in range(0xCC24, 0xCC36)]

        return data if self.isMenu() or self.isBattle() else [0] * len(data)

    def battleData(self):
        data = (
            [
                self.numberOfTurnsInCurrentBattle(self.pyboy.memory),
                self.playersSubstituteHp(self.pyboy.memory),
                self.enemySubstituteHp(self.pyboy.memory),
                self.moveMenuType(self.pyboy.memory),
                self.playerSelectedMove(self.pyboy.memory),
                self.enemySelectedMove(self.pyboy.memory),
                self.yourMoveUsed(self.pyboy.memory),
                self.yourMoveType(self.pyboy.memory),
                self.yourMoveEffect(self.pyboy.memory),
                self.enemyMoveId(self.pyboy.memory),
                self.enemyMoveEffect(self.pyboy.memory),
                self.enemyMovePower(self.pyboy.memory),
                self.enemyMoveType(self.pyboy.memory),
                self.enemyMoveAccuracy(self.pyboy.memory),
                self.enemyMoveMaxPP(self.pyboy.memory),
                self.playerMoveId(self.pyboy.memory),
                self.playerMovePower(self.pyboy.memory),
                self.playerMoveAccuracy(self.pyboy.memory),
                self.playerMoveMaxPP(self.pyboy.memory),
                self.enemyPokemonInternalId1(self.pyboy.memory),
                self.playerPokemonInterna2Id(self.pyboy.memory),
                self.enemyPokemonInternalId2(self.pyboy.memory),
                self.enemyHp(self.pyboy.memory),
                self.enemyLevel1(self.pyboy.memory),
            ]
            + self.enemyStatus(self.pyboy.memory)
            + [
                self.enemyType1(self.pyboy.memory),
                self.enemyType2(self.pyboy.memory),
                self.enemyMove1(self.pyboy.memory),
                self.enemyMove2(self.pyboy.memory),
                self.enemyMove3(self.pyboy.memory),
                self.enemyMove4(self.pyboy.memory),
                self.enemyAttackAndDefenseIVs(self.pyboy.memory),
                self.enemyAttackAndSpecialIVs(self.pyboy.memory),
                self.enemyLevel2(self.pyboy.memory),
                self.enemyMaxHp(self.pyboy.memory),
                self.enemyAttack(self.pyboy.memory),
                self.enemyDefense(self.pyboy.memory),
                self.enemySpeed(self.pyboy.memory),
                self.enemySpecial(self.pyboy.memory),
                self.enemyPPFirstSlot(self.pyboy.memory),
                self.enemyPPSecondSlot(self.pyboy.memory),
                self.enemyPPThirdSlot(self.pyboy.memory),
                self.enemyPPFourthSlot(self.pyboy.memory),
            ]
            + self.enemyBaseStats(self.pyboy.memory)
            + [
                self.enemyCatchRate(self.pyboy.memory),
                self.enemyBaseExperience(self.pyboy.memory),
                self.pokemonNumber1(self.pyboy.memory),
                self.pokemonCurrentHP1(self.pyboy.memory),
            ]
            + self.pokemonStatus1(self.pyboy.memory)
            + [
                self.pokemonType11(self.pyboy.memory),
                self.pokemonType21(self.pyboy.memory),
                self.pokemonMoveFirstSlot1(self.pyboy.memory),
                self.pokemonMoveSecondSlot1(self.pyboy.memory),
                self.pokemonMoveThirdSlot1(self.pyboy.memory),
                self.pokemonMoveFourthSlot1(self.pyboy.memory),
                self.pokemonAttackAndDefenseIVs1(self.pyboy.memory),
                self.pokemonSpeedAndSpecialIVs1(self.pyboy.memory),
                self.pokemonLevel1(self.pyboy.memory),
                self.pokemonMaxHp1(self.pyboy.memory),
                self.pokemonAttack1(self.pyboy.memory),
                self.pokemonDefense1(self.pyboy.memory),
                self.pokemonSpeed1(self.pyboy.memory),
                self.pokemonSpecial1(self.pyboy.memory),
                self.pokemonPPFirstSlot1(self.pyboy.memory),
                self.pokemonPPSecondSlot1(self.pyboy.memory),
                self.pokemonPPThirdSlot1(self.pyboy.memory),
                self.pokemonPPFourthSlot1(self.pyboy.memory),
                self.typeOfBattle(self.pyboy.memory),
                self.battleType(self.pyboy.memory),
                self.isGymLeaderBattleMusicPlaying(self.pyboy.memory),
                self.criticalHitFlag(self.pyboy.memory),
                self.oneHitKOFlag(self.pyboy.memory),
                self.hookedPokemonFlag(self.pyboy.memory),
            ]
            + self.battleStatusPlayer(self.pyboy.memory)
            + [self._battleCount]
        )

        return data if self.isBattle() else [0] * len(data)

    def pokeMartData(self):
        data = [self.pyboy.memory[i] for i in range(0xCF7B, 0xCF86)]

        return data if self.isMenu() else [0] * len(data)

    def playerData(self):
        data = (
            [self.pyboy.memory[i] for i in range(0xD163, 0xD16F)]
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
        return self.pokedexOwn(self.pyboy.memory) + self.pokedexSeen(self.pyboy.memory)

    def itemsData(self):
        data = [self.pyboy.memory[i] for i in range(0xD31D, 0xD347)]

        return data if self.isMenu() or self.isBattle() else [0] * len(data)

    def moneyData(self):
        return (
            [self.playerMoney(self.pyboy.memory)]
            if self.isMenu() or self.isDialog()
            else [0]
        )

    def miscellaneousData(self):
        return self.badges(self.pyboy.memory)

    def storedItemsData(self):
        data = self.storedItems(self.pyboy.memory)

        return data if self.isMenu() else [0] * len(data)

    def gameCoinsData(self):
        return [self.gameCoins(self.pyboy.memory)] if self.isMenu() else [0]

    def eventFlagsData(self):
        return (
            [
                self.startersBack(self.pyboy.memory),
                self.pyboy.memory[0xD5C0] & 1,
                self.haveTownMap(self.pyboy.memory),
                self.haveOaksParcel(self.pyboy.memory),
                self.bikeSpeed(self.pyboy.memory),
            ]
            + self.flyAnywhere(self.pyboy.memory)
            + [
                self.safariZoneTime(self.pyboy.memory),
                self.fossilizedPokemon(self.pyboy.memory),
                self.positionInAir(self.pyboy.memory),
                self.didYouGetLaprasYet(self.pyboy.memory),
                self.debugNewGame(self.pyboy.memory),
                self.foughtGiovanniYet(self.pyboy.memory),
                self.foughtBrockYet(self.pyboy.memory),
                self.foughtMistyYet(self.pyboy.memory),
                self.foughtLtSurgeYet(self.pyboy.memory),
                self.foughtErikaYet(self.pyboy.memory),
                self.foughtArticunoYet(self.pyboy.memory),
                self.foughtKogaYet(self.pyboy.memory),
                self.foughtBlaineYet(self.pyboy.memory),
                self.foughtSabrinaYet(self.pyboy.memory),
                self.foughtZapdosYet(self.pyboy.memory),
                self.foughtSnorlaxYetVermilion(self.pyboy.memory),
                self.foughtSnorlaxYetCeladon(self.pyboy.memory),
                self.foughtMoltresYet(self.pyboy.memory),
                self.isSSAnneHere(self.pyboy.memory),
                self.mewtwoCanBeCaught(self.pyboy.memory),
            ]
        )

    def opponentTrainersPokemonData(self):
        data = [self.pyboy.memory[i] for i in range(0xD89C, 0xD9AC)]

        return data if self.isBattle() else [0] * len(data)

    def bitsExtractor(self, byte, start_bit=0, end_bit=7):
        if start_bit < 0 or end_bit > 7 or start_bit > end_bit:
            raise ValueError("Invalid bit range")

        return [1 if (byte & (1 << i)) else 0 for i in range(start_bit, end_bit + 1)]

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
                if k == "continuous":
                    out[k] = v.detach().cpu().numpy().copy()
                else:
                    out[k] = int(v.item())
            else:
                out[k] = v
        return out

    @property
    def done(self):
        return self.terminated or self.truncated

    def startersBack(self, memory):
        return memory[0xD5AB] & 1

    def haveTownMap(self, memory):
        return memory[0xD5F3] & 1

    def haveOaksParcel(self, memory):
        return memory[0xD60D] & 1

    def bikeSpeed(self, memory):
        return memory[0xD700]

    def flyAnywhere(self, memory):
        return self.bitsExtractor(memory[0xD70B]) + self.bitsExtractor(memory[0xD70C])

    def safariZoneTime(self, memory):
        return memory[0xD70D] | (memory[0xD70E] << 8)

    def fossilizedPokemon(self, memory):
        return memory[0xD710] & 1

    def positionInAir(self, memory):
        return memory[0xD714] & 1

    def didYouGetLaprasYet(self, memory):
        return memory[0xD72E] & 1

    def debugNewGame(self, memory):
        return memory[0xD732] & 1

    def foughtGiovanniYet(self, memory):
        return memory[0xD751] & 1

    def foughtBrockYet(self, memory):
        return memory[0xD755] & 1

    def foughtMistyYet(self, memory):
        return memory[0xD75E] & 1

    def foughtLtSurgeYet(self, memory):
        return memory[0xD773] & 1

    def foughtErikaYet(self, memory):
        return memory[0xD77C] & 1

    def foughtArticunoYet(self, memory):
        return memory[0xD782] & 1

    def safariGameover(self, memory):
        return memory[0xD790] & 0x80

    def foughtKogaYet(self, memory):
        return memory[0xD792] & 1

    def foughtBlaineYet(self, memory):
        return memory[0xD79A] & 1

    def foughtSabrinaYet(self, memory):
        return memory[0xD7B3] & 1

    def foughtZapdosYet(self, memory):
        return memory[0xD7D4] & 1

    def foughtSnorlaxYetVermilion(self, memory):
        return memory[0xD7D8] & 1

    def foughtSnorlaxYetCeladon(self, memory):
        return memory[0xD7E0] & 1

    def foughtMoltresYet(self, memory):
        return memory[0xD7EE] & 1

    def isSSAnneHere(self, memory):
        return memory[0xD803] & 1

    def mewtwoCanBeCaught(self, memory):
        return memory[0xD85F] & 1

    def numberOfTurnsInCurrentBattle(self, memory):
        return memory[0xCCD5]

    def playersSubstituteHp(self, memory):
        return memory[0xCCD7]

    def rewardPlayersSubstituteHp(self, before, after):
        return (
            self.playersSubstituteHp(after) - self.playersSubstituteHp(before)
        ) / 255

    def enemySubstituteHp(self, memory):
        return memory[0xCCD8]

    def rewardEnemySubstituteHp(self, before, after):
        return (self.enemySubstituteHp(before) - self.enemySubstituteHp(after)) / 255

    def moveMenuType(self, memory):
        return memory[0xCCDB]

    def playerSelectedMove(self, memory):
        return memory[0xCCDC]

    def enemySelectedMove(self, memory):
        return memory[0xCCDD]

    def yourMoveUsed(self, memory):
        return memory[0xCCDC]

    def yourMoveType(self, memory):
        return memory[0xCFD5]

    def yourMoveEffect(self, memory):
        return memory[0xCFD3]

    def enemyMoveId(self, memory):
        return memory[0xCFCC]

    def enemyMoveEffect(self, memory):
        return memory[0xCFCD]

    def enemyMovePower(self, memory):
        return memory[0xCFCE]

    def enemyMoveType(self, memory):
        return memory[0xCFCF]

    def enemyMoveAccuracy(self, memory):
        return memory[0xCFD0]

    def enemyMoveMaxPP(self, memory):
        return memory[0xCFD1]

    def playerMoveId(self, memory):
        return memory[0xCFD2]

    def playerMovePower(self, memory):
        return memory[0xCFD4]

    def playerMoveAccuracy(self, memory):
        return memory[0xCFD6]

    def playerMoveMaxPP(self, memory):
        return memory[0xCFD7]

    def enemyPokemonInternalId1(self, memory):
        return memory[0xCFD8]

    def playerPokemonInterna2Id(self, memory):
        return memory[0xCFD9]

    def enemyPokemonInternalId2(self, memory):
        return memory[0xCFE5]

    def enemyHp(self, memory):
        return memory[0xCFE6] << 8 | memory[0xCFE7]

    def rewardEnemyHp(self, before, after):
        return (self.enemyHp(before) - self.enemyHp(after)) / self.enemyMaxHp(after)

    def enemyLevel1(self, memory):
        return memory[0xCFE8]

    def enemyStatus(self, memory):
        return self.bitsExtractor(memory[0xCFE9], end_bit=6)

    def rewardEnemyStatus(self, before, after):
        reward = 0

        for bit_before, bit_after in zip(
            self.enemyStatus(before), self.enemyStatus(after)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1

        return reward

    def enemyType1(self, memory):
        return memory[0xCFEA]

    def enemyType2(self, memory):
        return memory[0xCFEB]

    def enemyMove1(self, memory):
        return memory[0xCFED]

    def enemyMove2(self, memory):
        return memory[0xCFEE]

    def enemyMove3(self, memory):
        return memory[0xCFEF]

    def enemyMove4(self, memory):
        return memory[0xCFF0]

    def enemyAttackAndDefenseIVs(self, memory):
        return memory[0xCFF1]

    def enemyAttackAndSpecialIVs(self, memory):
        return memory[0xCFF2]

    def enemyLevel2(self, memory):
        return memory[0xCFF3]

    def enemyMaxHp(self, memory):
        return memory[0xCFF4] | (memory[0xCFF5] << 8)

    def enemyAttack(self, memory):
        return memory[0xCFF6] | (memory[0xCFF7] << 8)

    def enemyDefense(self, memory):
        return memory[0xCFF8] | (memory[0xCFF9] << 8)

    def enemySpeed(self, memory):
        return memory[0xCFFA] | (memory[0xCFFB] << 8)

    def enemySpecial(self, memory):
        return memory[0xCFFC] | (memory[0xCFFD] << 8)

    def enemyPPFirstSlot(self, memory):
        return memory[0xCFFE]

    def enemyPPSecondSlot(self, memory):
        return memory[0xCFFF]

    def enemyPPThirdSlot(self, memory):
        return memory[0xD000]

    def enemyPPFourthSlot(self, memory):
        return memory[0xD001]

    def enemyBaseStats(self, memory):
        return [memory[0xD002 + i] for i in range(5)]

    def enemyCatchRate(self, memory):
        return memory[0xD007]

    def enemyBaseExperience(self, memory):
        return memory[0xD008]

    def pokemonNumber1(self, memory):
        return memory[0xD014]

    def pokemonCurrentHP1(self, memory):
        return memory[0xD015] | (memory[0xD016] << 8)

    def rewardPokemonCurrentHP1(self, before, after):
        return (
            self.pokemonCurrentHP1(after) - self.pokemonCurrentHP1(before)
        ) / self.pokemonMaxHp1(after)

    def pokemonStatus1(self, memory):
        return self.bitsExtractor(memory[0xD018], end_bit=6)

    def rewardPokemonStatus1(self, before, after):
        reward = 0

        for bit_before, bit_after in zip(
            self.pokemonStatus1(before), self.pokemonStatus1(after)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1

        return reward

    def pokemonType11(self, memory):
        return memory[0xD019]

    def pokemonType21(self, memory):
        return memory[0xD01A]

    def pokemonMoveFirstSlot1(self, memory):
        return memory[0xD01C]

    def pokemonMoveSecondSlot1(self, memory):
        return memory[0xD01D]

    def pokemonMoveThirdSlot1(self, memory):
        return memory[0xD01E]

    def pokemonMoveFourthSlot1(self, memory):
        return memory[0xD01F]

    def pokemonAttackAndDefenseIVs1(self, memory):
        return memory[0xD020]

    def pokemonSpeedAndSpecialIVs1(self, memory):
        return memory[0xD021]

    def pokemonLevel1(self, memory):
        return memory[0xD022]

    def pokemonMaxHp1(self, memory):
        return memory[0xD023] | (memory[0xD024] << 8)

    def pokemonAttack1(self, memory):
        return memory[0xD025] | (memory[0xD026] << 8)

    def pokemonDefense1(self, memory):
        return memory[0xD027] | (memory[0xD028] << 8)

    def pokemonSpeed1(self, memory):
        return memory[0xD029] | (memory[0xD02A] << 8)

    def pokemonSpecial1(self, memory):
        return memory[0xD02B] | (memory[0xD02C] << 8)

    def pokemonPPFirstSlot1(self, memory):
        return memory[0xD02D]

    def pokemonPPSecondSlot1(self, memory):
        return memory[0xD02E]

    def pokemonPPThirdSlot1(self, memory):
        return memory[0xD02F]

    def pokemonPPFourthSlot1(self, memory):
        return memory[0xD030]

    def typeOfBattle(self, memory):
        return memory[0xD057]

    def battleType(self, memory):
        return memory[0xD05A]

    def isGymLeaderBattleMusicPlaying(self, memory):
        return memory[0xD05C] & 1

    def criticalHitFlag(self, memory):
        return memory[0xD05E] & 1

    def rewardCriticalHitFlag(self, before, after):
        return (
            15
            if self.criticalHitFlag(before) is 0 and self.criticalHitFlag(after) is 0
            else 0.0
        )

    def oneHitKOFlag(self, memory):
        return memory[0xD05E] & 2

    def rewardOneHitKOFlag(self, before, after):
        return (
            25
            if self.oneHitKOFlag(before) is 0 and self.oneHitKOFlag(after) is 0
            else 0.0
        )

    def hookedPokemonFlag(self, memory):
        return memory[0xD05F] & 1

    def battleStatusPlayer(self, memory):
        return (
            self.bitsExtractor(memory[0xD062])
            + self.bitsExtractor(memory[0xD063])
            + self.bitsExtractor(memory[0xD064], 0, 3)
        )

    def pokedexOwn(self, memory):
        data = memory[0xD2F7:0xD30A]

        bits = []
        for byte in data:
            bits.extend(self.bitsExtractor(byte))

        return bits

    def pokedexSeen(self, memory):
        data = memory[0xD30A:0xD31D]

        bits = []
        for byte in data:
            bits.extend(self.bitsExtractor(byte))

        return bits

    def playerMoney(self, memory):
        return memory[0xD347] + (memory[0xD348] << 8) + (memory[0xD349] << 16)

    def badges(self, memory):
        return self.bitsExtractor(memory[0xD356])

    def storedItems(self, memory):
        return [memory[i] for i in range(0xD53A, 0xD5A0)]

    def gameCoins(self, memory):
        return memory[0xD5A4] + (memory[0xD5A5] << 8)
