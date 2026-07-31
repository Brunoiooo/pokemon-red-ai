"""Headless diagnostic: verify reward_battle_exit (win / lose / flee)."""
from __future__ import annotations

import sys
from multiprocessing import RLock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pokemon.Data import Data  # noqa: E402
from pokemon.Emulator import MEMORY_SNAPSHOT_END  # noqa: E402

ADDR_BATTLE_RESULT = 0xCF0B
ADDR_ENEMY_LEVEL = 0xCFE8
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_LEVEL_0 = 0xD18C
ADDR_BATTLE_TYPE = 0xD057
PARTY_MON_SIZE = 0x2C


class FakeMem(bytearray):
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
    mem[ADDR_BATTLE_TYPE] = 1 if in_battle else 0
    mem[ADDR_ENEMY_LEVEL] = enemy_level
    set_party(mem, party_levels)
    return mem


def build_curr(*, battle_result: int) -> FakeMem:
    mem = FakeMem(MEMORY_SNAPSHOT_END)
    mem[ADDR_BATTLE_TYPE] = 0
    mem[ADDR_BATTLE_RESULT] = battle_result
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
    print("=== MOCK UNIT TESTS (battle_exit) ===")

    # Coward flee
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8, 3, 1])
    curr = build_curr(battle_result=2)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    milestone, step = data.reward(bytes(prev), action=0)
    print(f"COWARD flee enemy=5 party_max=8: exit={exit_r} milestone={milestone}")
    assert exit_r == data.flee_coward_penalty == -1.0
    assert abs(milestone - exit_r - data.new_screen_reward) < 1e-6

    # Smart flee
    prev = build_prev(in_battle=True, enemy_level=12, party_levels=[8, 3, 1])
    curr = build_curr(battle_result=2)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"SMART  flee enemy=12 party_max=8: exit={exit_r}")
    assert exit_r == data.flee_smart_reward == 0.4

    # Equal levels → coward
    prev = build_prev(in_battle=True, enemy_level=8, party_levels=[8])
    curr = build_curr(battle_result=2)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"EQUAL  flee enemy=8 party_max=8: exit={exit_r}")
    assert exit_r == -1.0

    # Win
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8])
    curr = build_curr(battle_result=0)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"WIN    CF0B=0: exit={exit_r} info={data.last_battle_exit_info}")
    assert exit_r == data.battle_won_reward == 2.0
    assert data.last_battle_exit_info is not None
    assert data.last_battle_exit_info["kind"] == "win"

    # Lose (blackout path — no separate blackout signal)
    prev = build_prev(in_battle=True, enemy_level=20, party_levels=[8])
    curr = build_curr(battle_result=1)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"LOSE   CF0B=1: exit={exit_r} info={data.last_battle_exit_info}")
    assert exit_r == data.battle_lost_penalty == -1.0
    assert data.last_battle_exit_info["kind"] == "lose"

    # Still in battle
    prev = build_prev(in_battle=True, enemy_level=5, party_levels=[8])
    curr = FakeMem(MEMORY_SNAPSHOT_END)
    curr[ADDR_BATTLE_TYPE] = 1
    curr[ADDR_BATTLE_RESULT] = 0
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"STILL IN BATTLE: exit={exit_r}")
    assert exit_r == 0.0

    # Weak slot cannot fake smart flee
    prev = build_prev(in_battle=True, enemy_level=10, party_levels=[1, 15])
    curr = build_curr(battle_result=2)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"MULTI party=[1,15] enemy=10: exit={exit_r} (expect coward)")
    assert exit_r == -1.0

    # Alias still works
    prev = build_prev(in_battle=True, enemy_level=12, party_levels=[8])
    curr = build_curr(battle_result=2)
    data = make_data(curr)
    assert data.reward_battle_flee(bytes(prev)) == 0.4

    print("MOCK_OK=True")
    print()


def run_level_soft_cap_cases() -> None:
    print("=== MOCK LEVEL SOFT-CAP ===")

    def levels_mem(levels: list[int]) -> FakeMem:
        mem = FakeMem(MEMORY_SNAPSHOT_END)
        set_party(mem, levels)
        return mem

    # One level-up below threshold: +0.5
    prev = levels_mem([5])
    curr = levels_mem([6])
    data = make_data(curr)
    r = data.reward_party_levels(bytes(prev))
    print(f"LEVEL 5->6 (sum=6 <=22): r={r}")
    assert abs(r - 0.5) < 1e-6

    # Crossing threshold: first level at/under full, next /4
    # sum 21->23 = two level-ups: 22 gets 0.5, 23 gets 0.125
    prev = levels_mem([21])
    curr = levels_mem([23])
    data = make_data(curr)
    r = data.reward_party_levels(bytes(prev))
    print(f"LEVEL sum 21->23: r={r} (expect 0.5+0.125=0.625)")
    assert abs(r - 0.625) < 1e-6

    # Well above threshold
    prev = levels_mem([30])
    curr = levels_mem([31])
    data = make_data(curr)
    r = data.reward_party_levels(bytes(prev))
    print(f"LEVEL sum 30->31: r={r} (expect 0.125)")
    assert abs(r - 0.125) < 1e-6

    print("LEVEL_SOFT_CAP_OK=True")
    print()


def try_live_outcomes() -> None:
    """Force win/lose/flee on real PyBoy save party."""
    print("=== PYBOY FORCED OUTCOMES ===")
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
    # Empty party (pre-starter saves) still exercises RAM edges with a synthetic slot.
    test_party = party if party else [6]
    print(
        f"save map={data.map_id(pyboy.memory)} party={party} max={party_max} "
        f"test_party={test_party}"
    )

    def force(enemy_lv: int, result: int, label: str) -> float:
        prev = bytearray(pyboy.memory[0:MEMORY_SNAPSHOT_END])
        prev[ADDR_BATTLE_TYPE] = 1
        prev[ADDR_ENEMY_LEVEL] = enemy_lv
        set_party(prev, test_party)
        pyboy.memory[ADDR_BATTLE_TYPE] = 0
        pyboy.memory[ADDR_BATTLE_RESULT] = result
        exit_r = data.reward_battle_exit(bytes(prev))
        print(
            f"{label}: enemy={enemy_lv} CF0B={result} exit={exit_r} "
            f"info={data.last_battle_exit_info}"
        )
        return exit_r

    pmax = max(test_party)
    assert force(max(1, pmax - 1), 2, "COWARD") == -1.0
    assert force(pmax + 5, 2, "SMART") == 0.4
    assert force(pmax, 0, "WIN") == 2.0
    assert force(pmax, 1, "LOSE") == -1.0
    print("PYBOY_FORCED_OK=True")
    pyboy.stop(save=False)
    print()


def main() -> None:
    check_snapshot_coverage()
    run_mock_cases()
    run_level_soft_cap_cases()
    try:
        try_live_outcomes()
    except Exception as exc:
        print(f"LIVE ERROR (non-fatal): {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
