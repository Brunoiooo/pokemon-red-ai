"""Headless diagnostic: verify flee path of reward_battle_exit.

Prefer ``diag_battle_outcome.py`` for win/lose/flee + level soft-cap coverage.
This script keeps the original flee-focused checks for quick regression.
"""
from __future__ import annotations

import sys
from multiprocessing import RLock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pokemon.Data import Data  # noqa: E402
from pokemon.Emulator import MEMORY_SNAPSHOT_END  # noqa: E402

# Addresses used by flee reward / party max level.
ADDR_BATTLE_RESULT = 0xCF0B
ADDR_ENEMY_LEVEL = 0xCFE8
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_LEVEL_0 = 0xD18C
ADDR_BATTLE_TYPE = 0xD057
PARTY_MON_SIZE = 0x2C


class FakeMem(bytearray):
    """bytearray that also supports PyBoy-style single-address reads."""

    pass


def make_data(mem: FakeMem) -> Data:
    pyboy = SimpleNamespace(memory=mem)
    return Data(pyboy=pyboy, files_lock=RLock())


def set_party(mem: FakeMem, levels: list[int]) -> None:
    mem[ADDR_PARTY_COUNT] = len(levels)
    for i, lv in enumerate(levels):
        mem[ADDR_PARTY_LEVEL_0 + PARTY_MON_SIZE * i] = lv


def build_prev(
    *,
    in_battle: bool,
    enemy_level: int,
    party_levels: list[int],
) -> FakeMem:
    mem = FakeMem(MEMORY_SNAPSHOT_END)
    mem[ADDR_BATTLE_TYPE] = 1 if in_battle else 0  # wild
    mem[ADDR_ENEMY_LEVEL] = enemy_level
    set_party(mem, party_levels)
    return mem


def build_curr(*, fled: bool, battle_result: int = 2) -> FakeMem:
    mem = FakeMem(MEMORY_SNAPSHOT_END)
    mem[ADDR_BATTLE_TYPE] = 0  # left battle
    mem[ADDR_BATTLE_RESULT] = battle_result if fled else 0
    # Party/enemy may be stale after exit; flee reads those from *prev*.
    return mem


def check_snapshot_coverage() -> None:
    needed = {
        "wBattleResult (0xCF0B)": ADDR_BATTLE_RESULT,
        "enemy_level (0xCFE8)": ADDR_ENEMY_LEVEL,
        "party_count (0xD163)": ADDR_PARTY_COUNT,
        "party_level0 (0xD18C)": ADDR_PARTY_LEVEL_0,
        "party_level5 (0xD18C+5*0x2C)": ADDR_PARTY_LEVEL_0 + PARTY_MON_SIZE * 5,
    }
    print(f"MEMORY_SNAPSHOT_END = 0x{MEMORY_SNAPSHOT_END:X} ({MEMORY_SNAPSHOT_END})")
    ok = True
    for name, addr in needed.items():
        included = addr < MEMORY_SNAPSHOT_END
        print(f"  {name}: 0x{addr:X} included={included}")
        ok = ok and included
    print(f"SNAPSHOT_OK={ok}")
    print()


def run_mock_cases() -> None:
    print("=== MOCK UNIT TESTS ===")

    # Coward: enemy <= max party
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8, 3, 1])
    curr = build_curr(fled=True, battle_result=2)
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    milestone, step = data.reward(bytes(prev), action=0)
    print(
        f"COWARD enemy=5 party_max=8: flee={flee} milestone={milestone} step={step}"
    )
    assert flee == data.flee_coward_penalty, f"expected {data.flee_coward_penalty}, got {flee}"
    assert flee == -1.0
    # Blank fake overworld still pays new_screen_reward (+0.1) via reward_map.
    assert abs(milestone - flee - data.new_screen_reward) < 1e-6, (
        f"milestone {milestone} expected flee {flee} + new_screen {data.new_screen_reward}"
    )

    # Smart: enemy > max party
    prev = build_prev(in_battle=True, enemy_level=12, party_levels=[8, 3, 1])
    curr = build_curr(fled=True, battle_result=2)
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    milestone, step = data.reward(bytes(prev), action=0)
    print(
        f"SMART  enemy=12 party_max=8: flee={flee} milestone={milestone} step={step}"
    )
    assert flee == data.flee_smart_reward
    assert flee == 0.4
    assert abs(milestone - flee - data.new_screen_reward) < 1e-6

    # Equal levels → coward (enemy <= party)
    prev = build_prev(in_battle=True, enemy_level=8, party_levels=[8])
    curr = build_curr(fled=True, battle_result=2)
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    print(f"EQUAL  enemy=8 party_max=8: flee={flee}")
    assert flee == -1.0

    # Win exit (CF0B=0): now pays battle_won_reward (not flee)
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8])
    curr = build_curr(fled=True, battle_result=0)
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    print(f"WIN exit (CF0B=0): exit={flee} (expect battle_won={data.battle_won_reward})")
    assert flee == data.battle_won_reward == 2.0

    # Still in battle: no exit
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8])
    curr = FakeMem(MEMORY_SNAPSHOT_END)
    curr[ADDR_BATTLE_TYPE] = 1
    curr[ADDR_BATTLE_RESULT] = 2
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    print(f"STILL IN BATTLE: flee={flee}")
    assert flee == 0.0

    # Weak slot cannot fake smart flee (max of all party slots)
    prev = build_prev(in_battle=True, enemy_level=10, party_levels=[1, 15])
    curr = build_curr(fled=True, battle_result=2)
    data = make_data(curr)
    flee = data.reward_battle_flee(bytes(prev))
    print(f"MULTI party=[1,15] enemy=10: flee={flee} (expect coward -1.0)")
    assert flee == -1.0

    print("MOCK_OK=True")
    print()


def try_live_flee(max_steps: int = 200) -> None:
    """Load a real save on PyBoy; force battle→flee RAM transition and score it.

    Walking for wild encounters is unreliable (stage_route1 save is Oak's Lab).
    Forcing the same RAM edge the env sees still exercises real PyBoy memory.
    """
    print("=== PYBOY FORCED FLEE (real save party) ===")
    rom = ROOT / "rom.gb"
    save = ROOT / "saves" / "stage_route1" / "checkpoint.state"
    if not rom.exists() or not save.exists():
        print(f"SKIP live: rom={rom.exists()} save={save.exists()}")
        return

    from pyboy import PyBoy

    pyboy = PyBoy(str(rom), window="null", sound_emulated=False, cgb=False)
    pyboy.set_emulation_speed(0)
    with open(save, "rb") as f:
        pyboy.load_state(f)

    data = Data(pyboy=pyboy, files_lock=RLock())
    data.clean()
    party_max = data.max_party_level(pyboy.memory)
    party = data.all_party_levels(pyboy.memory)
    test_party = party if party else [6]
    print(
        f"save map={data.map_id(pyboy.memory)} party_levels={party} "
        f"party_max={party_max} test_party={test_party}"
    )

    def force_and_score(enemy_lv: int, label: str) -> float:
        prev = bytearray(pyboy.memory[0:MEMORY_SNAPSHOT_END])
        prev[ADDR_BATTLE_TYPE] = 1
        prev[ADDR_ENEMY_LEVEL] = enemy_lv
        set_party(prev, test_party)
        # Current = left battle with fled result
        pyboy.memory[ADDR_BATTLE_TYPE] = 0
        pyboy.memory[ADDR_BATTLE_RESULT] = 2
        flee = data.reward_battle_flee(bytes(prev))
        milestone, step_r = data.reward(bytes(prev), action=0)
        print(
            f"{label}: enemy={enemy_lv} party_max={data.max_party_level(prev)} "
            f"CF0B={data.battle_result(pyboy.memory)} flee={flee} "
            f"milestone={milestone} step={step_r}"
        )
        return flee

    pmax = max(test_party)
    coward = force_and_score(max(1, pmax - 1), "COWARD")
    smart = force_and_score(pmax + 5, "SMART")
    assert coward == -1.0, coward
    assert smart == 0.4, smart
    print("PYBOY_FORCED_OK=True")
    pyboy.stop(save=False)
    print()


def main() -> None:
    check_snapshot_coverage()
    run_mock_cases()
    try:
        try_live_flee()
    except Exception as exc:
        print(f"LIVE ERROR (non-fatal): {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
