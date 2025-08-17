from pyboy import PyBoy
import torch
import torch.optim as optim
from ModelPokemon import ModelPokemon
import os
import keyboard
import random
import math
import numpy as np
import sys
from collections import deque

class Core:
    def __init__(
            self, 
            game, 
            epsilon=1, 
            epsilonBurn=100000, 
            epsilonEnd=0.05,
            epsilonDecayCount=2000000,
            tmpEpsilon = 0.2,
            tmpEpsilonSteps = 100000,
            maxResetCount = 100, 
            ticksPerStep = 20, 
            maxMenuSelect = 5,
            maxMenuPosition = 10,
            maxMenuIn = 15,
            maxSameAction = 20,
            worldIllegalMovesMax = 5,
            menuIllegalMovesMax = 20,
            ckpt_every = 100000,
            lr=0.0001,
            weight_decay=0.0001
        ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.game = game
        self.epsilon = epsilon
        self.pyboy = PyBoy(f"roms/{self.game}/rom.gb", sound_emulated = False, window="null")

        ckpt_path = None
        if os.path.exists(f"roms/{self.game}/latest.pth"):
            ckpt_path = f"roms/{self.game}/latest.pth"
        elif os.path.exists(f"roms/{self.game}/best.pth"):
            ckpt_path = f"roms/{self.game}/best.pth"

        self.modelPokemon = ModelPokemon().to(self.device)

        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location=self.device)
            self.modelPokemon.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)

        self.targetPokemon = ModelPokemon().to(self.device)
        self.targetPokemon.load_state_dict(self.modelPokemon.state_dict())
        self.tau = 0.005
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.modelPokemon.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = 512       
        self.buffer = deque(maxlen=50000)
        self.optimize_every = 4
        self.updates_per_opt = 2 
        self.grad_accum_steps = 1
        self.replay_start = 20_000

        self.visitedPositions = {}
        self.isStarted = False
        self.gamma = 0.99 
        self.epsilon = epsilon
        self.epsilonBurn = epsilonBurn
        self.epsilonEnd = epsilonEnd
        self.epsilonDecayCount = epsilonDecayCount
        self.resetCount = 0
        self.maxResetCount = maxResetCount
        self.count = 0
        self.buttons = ["a", "b", "start", "select", "left", "right", "up", "down"]
        self.historyInputs = np.zeros((16, 1559), dtype=np.float32)
        self.ticksPerStep = ticksPerStep
        self.maxMenuSelect = maxMenuSelect
        self.menuSelectCount = 0
        self.maxMenuPosition = maxMenuPosition
        self.menuPositionCount = 0
        self.maxMenuIn = maxMenuIn
        self.menuInCount = 0
        self.sameActionCount = 0
        self.maxSameAction = maxSameAction
        self.lastAction = -1
        self.averages = np.zeros(1000, dtype=np.float32)
        self.tmpEpsilon = tmpEpsilon
        self.tmpEpsilonOn = False
        self.tmpEpsilonSteps = tmpEpsilonSteps
        self.tmpEpsilonStepsCount = 0
        self.tmpEpsilonCooldown = 0
        self.menuReward = -0.5
        self.worldIllegalMovesCount = 0
        self.worldIllegalMovesMax = worldIllegalMovesMax
        self.menuIllegalMovesCount = 0
        self.menuIllegalMovesMax = menuIllegalMovesMax
        self.done = False
        self.ckpt_every = ckpt_every
        self.next_ckpt = self.ckpt_every
        self.need_game_state_ckpt = False
        self.best_eval_return = -float("inf")

    def save_latest(self):
        torch.save({
            "step": self.count,
            "model_state": self.modelPokemon.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "target_state": self.targetPokemon.state_dict(),
        }, f"roms/{self.game}/latest.pth")

    def save_best(self, avg_return):
        self.best_eval_return = avg_return
        torch.save({
            "step": self.count,
            "best_return": float(avg_return),
            "model_state": self.modelPokemon.state_dict(),
        }, f"roms/{self.game}/best.pth")

    def evaluate_greedy(self, episodes=50):
        total = 0.0
        was_training = self.modelPokemon.training
        self.modelPokemon.eval()

        for _ in range(episodes):
            self.reset("start")
            obs = self.inputs()
            ep_ret = 0.0

            while True:
                with torch.inference_mode():
                    q = self.modelPokemon(obs)
                    a = int(torch.argmax(q).item())

                self.pyboy.button_press(self.buttons[a])
                before = self.pyboy.memory[0x0000:0x10000]
                for __ in range(self.ticksPerStep):
                    self.pyboy.tick()
                r = self.reward(before, a)
                self.pyboy.button_release(self.buttons[a])

                ep_ret += float(r)

                if self.done:
                    self.done = False
                    break

                obs = self.inputs()
                
                sys.stdout.write(f"\rAvg: {((total / episodes) * 100):.2f}% ep_ret: {ep_ret:.2f} button: {self.buttons[a]} episode: {_}")
                sys.stdout.flush()
                
            total += ep_ret

        if was_training:
            self.modelPokemon.train()

        return total / episodes

    def soft_update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.modelPokemon.parameters(), self.targetPokemon.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def currentEpsilon(self):
        if self.tmpEpsilonOn:
            return self.tmpEpsilon

        if self.count < self.epsilonBurn:
            return self.epsilon
        t = min(self.count - self.epsilonBurn, self.epsilonDecayCount)
        frac = 1.0 - (t / self.epsilonDecayCount)
        return self.epsilonEnd + (self.epsilon - self.epsilonEnd) * max(frac, 0.0)

    def reset(self, stateName):
        if os.path.exists(f"roms/{self.game}/{stateName}.state"):
            with open(f"roms/{self.game}/{stateName}.state", "rb") as load_file:
                self.pyboy.load_state(load_file)
        self.resetCount = 0
        self.historyInputs = np.zeros((16, 1559), dtype=np.float32)
        self.visitedPositions = {}
        self.menuSelectCount = 0
        self.menuPositionCount = 0
        self.menuInCount = 0
        for i in range(8):
            self.pyboy.button_release(self.buttons[i])
        self.sameActionCount = 0
        self.lastAction = -1
        self.averages = np.zeros(1000, dtype=np.float32)
        self.menuReward = -0.5
        self.worldIllegalMovesCount = 0
        self.menuIllegalMovesCount = 0
        self.done = False

    def saveModel(self, name):
        torch.save(self.modelPokemon.state_dict(), f"roms/{self.game}/{name}.pth")

    def saveGameState(self, name):
        with open(f"roms/{self.game}/{name}.state", "wb") as save_file:
            self.pyboy.save_state(save_file)

    def isBattle(self):
        return True if self.pyboy.memory[0xD057] else False
    
    def isMenu(self):
        return True if self.pyboy.memory[0xCFC4] else False
    
    def isWorld(self):
        return True if not self.isBattle() and not self.isMenu() else False

    def spriteData(self):
        return (
            [self.pyboy.memory[i] for i in range(0xC100, 0xC300)] 
            if self.isWorld() 
            else [0] * (0xC300 - 0xC100)
        )
    
    def menuData(self):
        data = [self.pyboy.memory[i] for i in range(0xCC24, 0xCC30)]

        return (
             data
            if self.isMenu() 
            else [0] * len(data)
        )
    
    def battleData(self):
        data = [self.pyboy.memory[0xCCD5]] + \
            [self.pyboy.memory[i] for i in range(0xCCD7, 0xCCD9)] + \
            [self.pyboy.memory[i] for i in range(0xCCDB, 0xCCDE)] + \
            [self.pyboy.memory[i] for i in range(0xCCE8, 0xCCEA)] + \
            [self.pyboy.memory[i] for i in range(0xCCED, 0xCCF0)]  + \
            [self.pyboy.memory[0xCCF6]] + \
            [self.pyboy.memory[i] for i in range(0xCD05, 0xCD07)] + \
            [self.pyboy.memory[i] for i in range(0xCD1A, 0xCD20)] + \
            [self.pyboy.memory[i] for i in range(0xCD2D, 0xCD34)] + \
            [self.pyboy.memory[i] for i in range(0xCFCC, 0xCFE8)] + \
            self.bitsExtractor(self.pyboy.memory[0xCFE9], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xCFEA, 0xD031)] + \
            [self.pyboy.memory[0xD057], self.pyboy.memory[0xD05A]] + \
            [self.pyboy.memory[i] for i in range(0xD05C, 0xD060)] + \
            self.bitsExtractor(self.pyboy.memory[0xD062]) + \
            self.bitsExtractor(self.pyboy.memory[0xD063]) + \
            self.bitsExtractor(self.pyboy.memory[0xD064], 0, 3)
        
        return (
            data
            if self.isBattle() 
            else [0] * len(data)
        )
    
    def pokeMartData(self):
        data = [self.pyboy.memory[i] for i in range(0xCF7B, 0xCF86)]
        
        return (
            data
            if self.isMenu() 
            else [0] * len(data)
        )
    
    def playerData(self):
        data = [self.pyboy.memory[i] for i in range(0xD163, 0xD16F)] + \
            self.bitsExtractor(self.pyboy.memory[0xCFE9], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD170, 0xD19B)] + \
            self.bitsExtractor(self.pyboy.memory[0xD19B], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD19C, 0xD1C7)] + \
            self.bitsExtractor(self.pyboy.memory[0xD1C7], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD1C8, 0xD1F3)] + \
            self.bitsExtractor(self.pyboy.memory[0xD1F3], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD1F4, 0xD21F)] + \
            self.bitsExtractor(self.pyboy.memory[0xD21F], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD220, 0xD24B)] + \
            self.bitsExtractor(self.pyboy.memory[0xD24B], 3, 6) + \
            [self.pyboy.memory[i] for i in range(0xD24C, 0xD273)]
        
        return (
            data
            if self.isBattle()  or self.isMenu()
            else [0] * len(data)
        )
    
    def pokedexData(self):
        data = [self.pyboy.memory[i] for i in range(0xD2F7, 0xD31D)]
        
        return (
            data
            if self.isBattle()
            else [0] * len(data)
        )
    
    def itemsData(self):
        data = [self.pyboy.memory[i] for i in range(0xD31D, 0xD347)]
        
        return (
            data
            if self.isMenu()
            else [0] * len(data)
        )
    
    def moneyData(self):
        b0 = self.pyboy.memory[0xD347]
        b1 = self.pyboy.memory[0xD348]
        b2 = self.pyboy.memory[0xD349]

        money = (
            (b0 & 0x0F) * 10**5 + (b0 >> 4) * 10**6 +
            (b1 & 0x0F) * 10**3 + (b1 >> 4) * 10**4 +
            (b2 & 0x0F) * 10**1 + (b2 >> 4) * 10**2
        )

        return [money] if self.isMenu() else [0]
    
    def rivalData(self):
        data = [self.pyboy.memory[i] for i in range(0xD347, 0xD350)]
        
        return (
            data
            if self.isBattle()
            else [0] * len(data)
        )
    
    def miscellaneousData(self):
        return self.bitsExtractor(self.pyboy.memory[0xD356]) + \
            [self.pyboy.memory[i] for i in range(0xD35E, 0xD366)]
    
    def storedItemsData(self):
        data = [self.pyboy.memory[i] for i in range(0xD53A, 0xD560)]
        
        return (
            data
            if self.isMenu()
            else [0] * len(data)
        )
    
    def gameCoinsData(self):
        b0 = self.pyboy.memory[0xD5A4]
        b1 = self.pyboy.memory[0xD5A5]

        coins = (
            (b0 & 0x0F) * 10**1 + (b0 >> 4) * 10**2 +
            (b1 & 0x0F) * 10**3 + (b1 >> 4) * 10**4
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
            self.pyboy.memory[0xD85F]
        ]
    
    def opponentTrainersPokemonData(self):
        data = [self.pyboy.memory[i] for i in range(0xD89C, 0xDA30)]
        
        return (
            data
            if self.isBattle()
            else [0] * len(data)
        )
        
    def inputs(self):
        data = self.spriteData()
            
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

        self.historyInputs = np.roll(self.historyInputs, -1, axis=0)
        self.historyInputs[-1] = data

        arr = np.asarray(self.log_transform([item for sublist in self.historyInputs for item in sublist]), dtype=np.float32)

        arr = np.clip(arr, 0, 255) / 255.0

        return torch.from_numpy(arr).to(self.device)
    
    def log_transform(self, data):
        return [math.log(x+1) if x > 3 else float(x) for x in data]

    def bitsExtractor(self, byte, start_bit = 0, end_bit = 7):
        if start_bit < 0 or end_bit > 7 or start_bit > end_bit:
            raise ValueError("Invalid bit range")

        return [1 if (byte & (1<<i)) else 0 for i in range(start_bit, end_bit+1)]
    
    def start(self):
        self.modelPokemon.train()
        self.targetPokemon.eval()

        self.reset("checkpoint")
        self.isStarted = True

        inputs = self.inputs()
        torch.set_printoptions(threshold=float('inf'))
        while True:
            if keyboard.is_pressed('q'):
                break
            elif keyboard.is_pressed('w'):
                self.pyboy.stop(False)
                self.pyboy = PyBoy(f"roms/{self.game}/rom.gb", sound_emulated = False, window="null")
                self.reset("checkpoint")
                print("w")
            elif keyboard.is_pressed('e'):
                self.pyboy.stop(False)
                self.pyboy = PyBoy(f"roms/{self.game}/rom.gb", sound_emulated = False)
                self.reset("start")
                print("e")
            elif keyboard.is_pressed('['):
                self.isStarted = False
                print("stop")
            elif keyboard.is_pressed(']'):
                print("start")
                self.isStarted = True

            if not self.isStarted:
                self.pyboy.tick()
                continue

            inputs = self.train(inputs)

            if self.tmpEpsilonOn:
                self.tmpEpsilonStepsCount += 1

            if not self.tmpEpsilonOn and 0 < self.tmpEpsilonCooldown:
                self.tmpEpsilonCooldown -= 1

            if self.tmpEpsilonStepsCount >= self.tmpEpsilonSteps:
                self.tmpEpsilonOn = False
                self.tmpEpsilonStepsCount = 0
                self.tmpEpsilonCooldown = self.tmpEpsilonSteps

            if not self.tmpEpsilonOn and self.count > self.epsilonDecayCount and self.average() < 0.5 and self.tmpEpsilonCooldown <= 0:
                self.tmpEpsilonOn = True

    def action(self, inputs):
        with torch.inference_mode():
            output = self.modelPokemon(inputs)
            greedy_action = int(torch.argmax(output).item())
        if random.random() < self.currentEpsilon():
            return random.randint(0, 7)
        
        return greedy_action
        
    def optimize_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states_cpu, actions, rewards, next_states_cpu, dones = zip(*batch)

        states      = torch.stack(states_cpu).to(self.device, non_blocking=True)
        next_states = torch.stack(next_states_cpu).to(self.device, non_blocking=True)
        actions     = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards     = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones       = torch.tensor(dones,   device=self.device, dtype=torch.bool)

        rewards = rewards.clamp_(-1.0, 1.0)

        micro = self.batch_size // self.grad_accum_steps
        assert self.batch_size % self.grad_accum_steps == 0

        self.optimizer.zero_grad(set_to_none=True)

        for i in range(self.grad_accum_steps):
            sl = slice(i*micro, (i+1)*micro)
            s, ns = states[sl], next_states[sl]
            a, r, d = actions[sl], rewards[sl], dones[sl]

            q_all = self.modelPokemon(s)                  
            q_sa  = q_all.gather(1, a.view(-1,1)).squeeze(1)    

            with torch.no_grad():
                next_q_online = self.modelPokemon(ns)                  
                next_a        = torch.argmax(next_q_online, dim=1) 
                next_q_target = self.targetPokemon(ns).gather(1, next_a.view(-1,1)).squeeze(1)
                target = torch.where(d, r, r + self.gamma * next_q_target)

            loss = self.criterion(q_sa, target) / self.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.modelPokemon.parameters(), 10.0)
        self.optimizer.step()

        self.soft_update_target()

    def train(self, inputs):
        action = self.action(inputs)

        self.pyboy.button_press(self.buttons[action])

        primary_memo_before = self.pyboy.memory[0x0000:0x10000]
        for _ in range(self.ticksPerStep):
            self.pyboy.tick()

        reward = self.reward(primary_memo_before, action) 
        
        self.pyboy.button_release(self.buttons[action])
        
        next_state = self.inputs()

        self.count += 1

        self.averages = np.roll(self.averages, -1, axis=0)
        self.averages[-1] = reward   

        self.buffer.append((
            inputs.detach().to("cpu"),
            action,
            float(reward),
            next_state.detach().to("cpu"),
            bool(self.done)
        ))

        if len(self.buffer) >= self.replay_start and (self.count % self.optimize_every == 0):
            for _ in range(self.updates_per_opt):
                self.optimize_batch()

        if self.count >= self.next_ckpt:
            self.save_latest()
            avg_ret = self.evaluate_greedy(episodes=50)
            if avg_ret > self.best_eval_return + 0.5: 
                self.save_best(avg_ret)
            self.need_game_state_ckpt = True
            self.next_ckpt += self.ckpt_every
        
        if self.count % 10 == 0:
            sys.stdout.write(f"\risWorld: {self.isWorld()} isMenu: {self.isMenu()} isBattle: {self.isBattle()} Reward {reward:.2f} Epsilon: {self.currentEpsilon():.2f} | Count: {self.count} | Progress: {(self.count / self.epsilonDecayCount * 100):.2f}% | Average: {self.average():.2f} Button {self.buttons[action]}")
            sys.stdout.flush()

        if self.done:
            if self.need_game_state_ckpt:
                self.saveGameState("checkpoint")
                self.need_game_state_ckpt = False
            self.reset("checkpoint")
            return self.inputs()
        
        return next_state
    
    def countingReward(self, reward):
        self.resetCount = 0 if reward > 0 else self.resetCount + 1
        if self.resetCount > self.maxResetCount:
            self.done = True
    
    def panishWorldIllegalMoves(self, primary_memo_before, reward):
        if not self.isWorld() or 0 < reward or not self.isSameWorldPosition(primary_memo_before):
            self.worldIllegalMovesCount = 0
            return 0
        
        self.worldIllegalMovesCount += 1

        if self.worldIllegalMovesMax < self.worldIllegalMovesCount:
            self.done = True

        if 1 < self.worldIllegalMovesCount:
            return -3
        else:
            return 0

    def panishMenuIllegalMoves(self, primary_memo_before, reward):
        if not self.isMenu() or 0 < reward or not self.isSameMenuPosition(primary_memo_before) or not self.isSameMenuPosition(primary_memo_before):
            self.menuIllegalMovesCount = 0
            return 0
        
        self.menuIllegalMovesCount += 1

        if self.menuIllegalMovesMax < self.menuIllegalMovesCount:
            self.done = True

        if 1 < self.menuIllegalMovesCount:
            return -3
        else:
            return 0
    
    def panishSwitchMenu(self, reward):
        if self.isMenu() and reward <= 0:
            if -0.2 < self.menuReward:
                self.menuReward += -0.05
            return self.menuReward
        else:
            return 0

    def average(self):
        return sum(self.averages) / len(self.averages)

    def isSameWorldPosition(self, primary_memo_before):
        return True if primary_memo_before[0xD35E] == self.pyboy.memory[0xD35E] and primary_memo_before[0xD361] == self.pyboy.memory[0xD361] and primary_memo_before[0xD362] == self.pyboy.memory[0xD362] else False

    def isSameMenuPosition(self, primary_memo_before):
        return True if primary_memo_before[0xCC24] == self.pyboy.memory[0xCC24] and primary_memo_before[0xCC25] == self.pyboy.memory[0xCC25] else False
    
    def isSameMenuSelect(self, primary_memo_before):
        return True if primary_memo_before[0xCC26] == self.pyboy.memory[0xCC26] else False

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
        
    def panishMenuSelect(self, primary_memo_before, reward):
        if not self.isMenu() or 0 < reward:
            self.menuSelectCount = 0
            return 0

        if primary_memo_before[0xCC26] == self.pyboy.memory[0xCC26] and self.isSameMenuPosition(primary_memo_before):
            self.menuSelectCount += 1
        else:
            self.menuSelectCount = 0
            
        return -0.2 if self.maxMenuSelect < self.menuSelectCount else 0
    
    def panishMenuPosition(self, primary_memo_before, reward):
        if not self.isMenu() or 0 < reward:
            self.menuPositionCount = 0
            return 0
        
        if self.isSameMenuPosition(primary_memo_before):
            self.menuPositionCount += 1
        else:
            self.menuSelectCount = 0
            self.menuPositionCount = 0
        
        return -0.3 if self.maxMenuPosition < self.menuPositionCount else 0
   
    def panishMenuIn(self, reward):
        if not self.isMenu() or 0 < reward:
            self.menuInCount = 0
            self.menuSelectCount = 0
            self.menuPositionCount = 0
            return 0

        self.menuInCount += 1
        
        return -0.4 if self.maxMenuIn < self.menuInCount else 0

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
        if primary_memo_before[0xCCEE] != self.pyboy.memory[0xCCEE] and self.pyboy.memory[0xCCEE] == 0: 
            reward += 1
        elif primary_memo_before[0xCCEE] != self.pyboy.memory[0xCCEE]:
            reward -= 1

        # Enemy move that the player disabled
        if primary_memo_before[0xCCEF] != self.pyboy.memory[0xCCEF] and self.pyboy.memory[0xCCEF] != 0:
            reward += 1

        # Enemy's HP
        if primary_memo_before[0xCFE7] << 8 | primary_memo_before[0xCFE6] > self.pyboy.memory[0xCFE7] << 8 | self.pyboy.memory[0xCFE6]:
            reward += 1

        # Enemy's Status
        reward += self.rewardStatus(primary_memo_before[0xCFE8], self.pyboy.memory[0xCFE8])

        # Pokémon 1st Slot (In-Battle)
        if primary_memo_before[0xD016] << 8 | primary_memo_before[0xD015] < self.pyboy.memory[0xD016] << 8 | self.pyboy.memory[0xD015]: # Current HP
            reward += 1

        # Status
        reward -= self.rewardStatus(primary_memo_before[0xD018], self.pyboy.memory[0xD018])

        #Critical Hit / OHKO Flag
        if primary_memo_before[0xD05E] != self.pyboy.memory[0xD05E]:
            reward += self.pyboy.memory[0xD05E]

        if primary_memo_before[0xD05F] != self.pyboy.memory[0xD05F] and self.pyboy.memory[0xD05F] == 1:
            reward += 1

        # Pokémon 1
        if primary_memo_before[0xD16D] << 8 | primary_memo_before[0xD16C] < self.pyboy.memory[0xD16D] << 8 | self.pyboy.memory[0xD16C]: # Current HP
            reward += 1
        elif primary_memo_before[0xD16D] << 8 | primary_memo_before[0xD16C] > self.pyboy.memory[0xD16D] << 8 | self.pyboy.memory[0xD16C]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD16F], self.pyboy.memory[0xD16F]) # Status

        if primary_memo_before[0xD18C] < self.pyboy.memory[0xD18C]: # Level
            reward += 1
        elif primary_memo_before[0xD18C] > self.pyboy.memory[0xD18C]:
            reward -= 1

        # Pokémon 2
        if primary_memo_before[0xD199] << 8 | primary_memo_before[0xD198] < self.pyboy.memory[0xD199] << 8 | self.pyboy.memory[0xD198]: # Current HP
            reward += 1
        elif primary_memo_before[0xD199] << 8 | primary_memo_before[0xD198] > self.pyboy.memory[0xD199] << 8 | self.pyboy.memory[0xD198]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD19B], self.pyboy.memory[0xD19B]) # Status

        if primary_memo_before[0xD1B8] < self.pyboy.memory[0xD1B8]: # Level
            reward += 1
        elif primary_memo_before[0xD1B8] > self.pyboy.memory[0xD1B8]:
            reward -= 1

        # Pokémon 3
        if primary_memo_before[0xD1C5] << 8 | primary_memo_before[0xD1C4] < self.pyboy.memory[0xD1C5] << 8 | self.pyboy.memory[0xD1C4]: # Current HP
            reward += 1
        elif primary_memo_before[0xD1C5] << 8 | primary_memo_before[0xD1C4] > self.pyboy.memory[0xD1C5] << 8 | self.pyboy.memory[0xD1C4]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD1C7], self.pyboy.memory[0xD1C7]) # Status

        if primary_memo_before[0xD1E4] < self.pyboy.memory[0xD1E4]: # Level
            reward += 1
        elif primary_memo_before[0xD1E4] > self.pyboy.memory[0xD1E4]:
            reward -= 1

        # Pokémon 4
        if primary_memo_before[0xD1F1] << 8 | primary_memo_before[0xD1F0] < self.pyboy.memory[0xD1F1] << 8 | self.pyboy.memory[0xD1F0]: # Current HP
            reward += 1
        elif primary_memo_before[0xD1F1] << 8 | primary_memo_before[0xD1F0] > self.pyboy.memory[0xD1F1] << 8 | self.pyboy.memory[0xD1F0]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD1F3], self.pyboy.memory[0xD1F3]) # Status

        if primary_memo_before[0xD210] < self.pyboy.memory[0xD210]: # Level
            reward += 1
        elif primary_memo_before[0xD210] > self.pyboy.memory[0xD210]:
            reward -= 1

        # Pokémon 5
        if primary_memo_before[0xD21D] << 8 | primary_memo_before[0xD21C] < self.pyboy.memory[0xD21D] << 8 | self.pyboy.memory[0xD21C]: # Current HP
            reward += 1
        elif primary_memo_before[0xD21D] << 8 | primary_memo_before[0xD21C] > self.pyboy.memory[0xD21D] << 8 | self.pyboy.memory[0xD21C]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD21F], self.pyboy.memory[0xD21F]) # Status

        if primary_memo_before[0xD23C] < self.pyboy.memory[0xD23C]: # Level
            reward += 1
        elif primary_memo_before[0xD23C] > self.pyboy.memory[0xD23C]:
            reward -= 1

        # Pokémon 6
        if primary_memo_before[0xD249] << 8 | primary_memo_before[0xD248] < self.pyboy.memory[0xD249] << 8 | self.pyboy.memory[0xD248]: # Current HP
            reward += 1
        elif primary_memo_before[0xD249] << 8 | primary_memo_before[0xD248] > self.pyboy.memory[0xD249] << 8 | self.pyboy.memory[0xD248]:
            reward -= 1

        reward -= self.rewardStatus(primary_memo_before[0xD24B], self.pyboy.memory[0xD24B]) # Status

        if primary_memo_before[0xD268] < self.pyboy.memory[0xD268]: # Level
            reward += 1
        elif primary_memo_before[0xD268] > self.pyboy.memory[0xD268]:
            reward -= 1

        # Pokedex
        for i in range(0xD2F7, 0xD31C):
            reward += self.rewardPokedex(primary_memo_before[i], self.pyboy.memory[i])

        # Miscellaneous
        reward += self.rewardBadges(primary_memo_before[0xD356], self.pyboy.memory[0xD356])

        # Starters Back?
        if primary_memo_before[0xD5AB] != self.pyboy.memory[0xD5AB] and self.pyboy.memory[0xD5AB]:
            reward += 1

        # Have Town map?
        if primary_memo_before[0xD5F3] != self.pyboy.memory[0xD5F3] and self.pyboy.memory[0xD5F3]:
            reward += 1

        # Have Oak's Parcel?
        if primary_memo_before[0xD60D] != self.pyboy.memory[0xD60D] and self.pyboy.memory[0xD60D]:
            reward += 1

        # Fossilized Pokémon?
        if primary_memo_before[0xD710] != self.pyboy.memory[0xD710] and self.pyboy.memory[0xD710]:
            reward += 1

        # Did you get Lapras Yet?
        if primary_memo_before[0xD72E] != self.pyboy.memory[0xD72E] and self.pyboy.memory[0xD72E]:
            reward += 1

        # Fought Giovanni Yet?
        if primary_memo_before[0xD751] != self.pyboy.memory[0xD751] and self.pyboy.memory[0xD751]:
            reward += 1

        # Fought Brock Yet?
        if primary_memo_before[0xD755] != self.pyboy.memory[0xD755] and self.pyboy.memory[0xD755]:
            reward += 1

        # Fought Misty Yet?
        if primary_memo_before[0xD75E] != self.pyboy.memory[0xD75E] and self.pyboy.memory[0xD75E]:
            reward += 1

        # Fought Lt. Surge Yet?
        if primary_memo_before[0xD773] != self.pyboy.memory[0xD773] and self.pyboy.memory[0xD773]:
            reward += 1

        # Fought Erika Yet?
        if primary_memo_before[0xD77C] != self.pyboy.memory[0xD77C] and self.pyboy.memory[0xD77C]:
            reward += 1

        # Fought Articuno Yet?
        if primary_memo_before[0xD782] != self.pyboy.memory[0xD782] and self.pyboy.memory[0xD782]:
            reward += 1

        # Fought Koga Yet?
        if primary_memo_before[0xD792] != self.pyboy.memory[0xD792] and self.pyboy.memory[0xD792]:
            reward += 1

        # Fought Blaine Yet?
        if primary_memo_before[0xD79A] != self.pyboy.memory[0xD79A] and self.pyboy.memory[0xD79A]:
            reward += 1

        # Fought Sabrina Yet?
        if primary_memo_before[0xD7B3] != self.pyboy.memory[0xD7B3] and self.pyboy.memory[0xD7B3]:
            reward += 1

        # Fought Zapdos Yet?
        if primary_memo_before[0xD7D4] != self.pyboy.memory[0xD7D4] and self.pyboy.memory[0xD7D4]:
            reward += 1

        # Fought Snorlax Yet (Vermilion)
        if primary_memo_before[0xD7D8] != self.pyboy.memory[0xD7D8] and self.pyboy.memory[0xD7D8]:
            reward += 1

        # Fought Snorlax Yet? (Celadon)
        if primary_memo_before[0xD7E0] != self.pyboy.memory[0xD7E0] and self.pyboy.memory[0xD7E0]:
            reward += 1

        # Fought Moltres Yet?
        if primary_memo_before[0xD7EE] != self.pyboy.memory[0xD7EE] and self.pyboy.memory[0xD7EE]:
            reward += 1

        # Opponent Trainer’s Pokémon
        # Pokémon 1
        if primary_memo_before[0xD8A6] << 8 | primary_memo_before[0xD8A5] > self.pyboy.memory[0xD8A6] << 8 | self.pyboy.memory[0xD8A5]: # Current HP
            reward += 1
        elif primary_memo_before[0xD8A6] << 8 | primary_memo_before[0xD8A5] < self.pyboy.memory[0xD8A6] << 8 | self.pyboy.memory[0xD8A5]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD8A8], self.pyboy.memory[0xD8A8]) # Status

        # Pokémon 2
        if primary_memo_before[0xD8D2] << 8 | primary_memo_before[0xD8D1] > self.pyboy.memory[0xD8D2] << 8 | self.pyboy.memory[0xD8D1]: # Current HP
            reward += 1
        elif primary_memo_before[0xD8D2] << 8 | primary_memo_before[0xD8D1] < self.pyboy.memory[0xD8D2] << 8 | self.pyboy.memory[0xD8D1]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD8D4], self.pyboy.memory[0xD8D4]) # Status

        # Pokémon 3
        if primary_memo_before[0xD8FE] << 8 | primary_memo_before[0xD8FD] > self.pyboy.memory[0xD8FE] << 8 | self.pyboy.memory[0xD8FD]: # Current HP
            reward += 1
        elif primary_memo_before[0xD8FE] << 8 | primary_memo_before[0xD8FD] < self.pyboy.memory[0xD8FE] << 8 | self.pyboy.memory[0xD8FD]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD900], self.pyboy.memory[0xD900]) # Status

        # Pokémon 4
        if primary_memo_before[0xD92A] << 8 | primary_memo_before[0xD929] > self.pyboy.memory[0xD92A] << 8 | self.pyboy.memory[0xD929]: # Current HP
            reward += 1
        elif primary_memo_before[0xD92A] << 8 | primary_memo_before[0xD929] < self.pyboy.memory[0xD92A] << 8 | self.pyboy.memory[0xD929]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD92C], self.pyboy.memory[0xD92C]) # Status

        # Pokémon 5
        if primary_memo_before[0xD956] << 8 | primary_memo_before[0xD955] > self.pyboy.memory[0xD956] << 8 | self.pyboy.memory[0xD955]: # Current HP
            reward += 1
        elif primary_memo_before[0xD956] << 8 | primary_memo_before[0xD955] < self.pyboy.memory[0xD956] << 8 | self.pyboy.memory[0xD955]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD958], self.pyboy.memory[0xD958]) # Status

        # Pokémon 6
        if primary_memo_before[0xD982] << 8 | primary_memo_before[0xD981] > self.pyboy.memory[0xD982] << 8 | self.pyboy.memory[0xD981]: # Current HP
            reward += 1
        elif primary_memo_before[0xD982] << 8 | primary_memo_before[0xD981] < self.pyboy.memory[0xD982] << 8 | self.pyboy.memory[0xD981]:
            reward -= 1

        reward += self.rewardStatus(primary_memo_before[0xD984], self.pyboy.memory[0xD984]) # Status

        reward += self.panishWorldIllegalMoves(primary_memo_before, reward)
        reward += self.panishMenuIllegalMoves(primary_memo_before, reward)
        reward += self.rewardPosition()
        reward += self.panishSwitchMenu(reward)
        reward += self.panishMenuSelect(primary_memo_before, reward)
        reward += self.panishMenuPosition(primary_memo_before, reward)
        reward += self.panishMenuIn(reward)
        reward += self.panishSameAction(action, reward)

        return reward

    def rewardStatus(self, before, after):
        reward = 0

        if after & (1 << 6) and before & (1 << 6) != after & (1 << 6): # Paralyzed
            reward += 1
        if after & (1 << 5) and before & (1 << 5) != after & (1 << 5): # Frozen
            reward += 1 
        if after & (1 << 4) and before & (1 << 4) != after & (1 << 4): # Burned
            reward += 1
        if after & (1 << 3) and before & (1 << 3) != after & (1 << 3): # Poisoned
            reward += 1

        return reward

    def rewardPokedex(self, before, after):
        reward = 0

        for i in range(8):
            if after & (1 << i) and before & (1 << i) != after & (1 << i):
                reward += 1

        return reward
    
    def rewardBadges(self, before, after):
        reward = 0

        for i in range(8):
            if after & (1 << i) and before & (1 << i) != after & (1 << i):
                reward += 1

        return reward
    
    def rewardPosition(self):
        if not self.isWorld():
            return 0
        
        map = self.pyboy.memory[0xD35E]
        position = f"{self.pyboy.memory[0xD361]}x{self.pyboy.memory[0xD362]}"

        if map not in self.visitedPositions:
            self.visitedPositions[map] = {}
            self.menuReward += 0.2
            return 2

        if position not in self.visitedPositions[map]:
            self.visitedPositions[map][position] = 0.2
            self.menuReward += 0.05
            return self.visitedPositions[map][position]
        
        if self.visitedPositions[map][position] > -0.2:
            self.visitedPositions[map][position] += -0.05

        return self.visitedPositions[map][position]
