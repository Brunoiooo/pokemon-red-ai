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
ADDR_ENEMY_LEVEL = 0xCFF3  # wEnemyMonLevel (wEnemyMon base 0xCFE5 + offset 0x0E)
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_LEVEL_0 = 0xD18C
ADDR_BATTLE_TYPE = 0xD057
ADDR_ACTIVE_LEVEL = 0xD022  # wBattleMonLevel — the mon actually fighting
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
    active_level: int | None = None,
) -> FakeMem:
    """``active_level`` defaults to the party's max — matches the old mocks
    (single-mon parties) where "active" and "strongest" were the same mon.
    Pass it explicitly to simulate a weaker/leveling mon in the front slot.
    """
    mem = FakeMem(MEMORY_SNAPSHOT_END)
    mem[ADDR_BATTLE_TYPE] = 1 if in_battle else 0
    mem[ADDR_ENEMY_LEVEL] = enemy_level
    set_party(mem, party_levels)
    mem[ADDR_ACTIVE_LEVEL] = (
        active_level if active_level is not None else max(party_levels, default=0)
    )
    return mem


def build_curr(*, battle_result: int) -> FakeMem:
    mem = FakeMem(MEMORY_SNAPSHOT_END)
    mem[ADDR_BATTLE_TYPE] = 0
    mem[ADDR_BATTLE_RESULT] = battle_result
    return mem


def check_snapshot_coverage() -> None:
    needed = {
        "wBattleResult (0xCF0B)": ADDR_BATTLE_RESULT,
        "enemy_level (0xCFF3)": ADDR_ENEMY_LEVEL,
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

    # Win against an equal-or-tougher opponent → full battle_won_reward.
    prev = build_prev(in_battle=True, enemy_level=8, party_levels=[8])
    curr = build_curr(battle_result=0)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(f"WIN    CF0B=0 (fair): exit={exit_r} info={data.last_battle_exit_info}")
    assert exit_r == data.battle_won_reward == 2.0
    assert data.last_battle_exit_info is not None
    assert data.last_battle_exit_info["kind"] == "win"
    assert data.last_battle_exit_info["difficulty_scale"] == 1.0

    # Win against a much weaker opponent, fought by the strong mon itself
    # (active == party_max == 10 vs enemy 2) → discounted by
    # enemy_lv/active_lv (floored at battle_difficulty_floor). This is the
    # reported exploit: a lvl-10 one-shotting a lvl-2 wild Pokemon.
    prev = build_prev(
        in_battle=True, enemy_level=2, party_levels=[10], active_level=10
    )
    curr = build_curr(battle_result=0)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    expect_scale = max(data.battle_difficulty_floor, 2 / 10)
    print(
        f"WIN    CF0B=0 (active lvl10 vs lvl2): exit={exit_r} "
        f"info={data.last_battle_exit_info}"
    )
    assert abs(exit_r - data.battle_won_reward * expect_scale) < 1e-9
    assert data.last_battle_exit_info["difficulty_scale"] == expect_scale

    # Deliberately leveling a weaker bench mon: it's the active battler and
    # the fight is fair *for it* (enemy_lv == its own level), even though a
    # much stronger mon (lvl 10) sits on the bench. Must NOT be discounted —
    # that would punish legitimate grinding of the whole team.
    prev = build_prev(
        in_battle=True, enemy_level=3, party_levels=[10, 3], active_level=3
    )
    curr = build_curr(battle_result=0)
    data = make_data(curr)
    exit_r = data.reward_battle_exit(bytes(prev))
    print(
        f"WIN    CF0B=0 (leveling weak mon, active lvl3 vs lvl3): exit={exit_r} "
        f"info={data.last_battle_exit_info}"
    )
    assert exit_r == data.battle_won_reward == 2.0
    assert data.last_battle_exit_info["difficulty_scale"] == 1.0

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

    def force(enemy_lv: int, result: int, label: str, active_lv: int | None = None) -> float:
        prev = bytearray(pyboy.memory[0:MEMORY_SNAPSHOT_END])
        prev[ADDR_BATTLE_TYPE] = 1
        prev[ADDR_ENEMY_LEVEL] = enemy_lv
        set_party(prev, test_party)
        prev[ADDR_ACTIVE_LEVEL] = active_lv if active_lv is not None else max(test_party)
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


def run_enemy_level_cache_case() -> None:
    """A hit landing on a frame where wEnemyMonLevel misreads 0 must still

    be scaled using the level last seen this battle, not fall through to
    the floor as if the fight were unreadable/unknown.
    """
    print("=== ENEMY LEVEL CACHE (bad-frame fallback) ===")
    ADDR_ENEMY_HP = 0xCFE6
    ADDR_ENEMY_MAXHP = 0xCFF4
    ADDR_POKEMON_MAXHP = 0xD023

    def mem_with(enemy_level: int, enemy_hp: int, enemy_maxhp: int, active_level: int) -> FakeMem:
        mem = FakeMem(MEMORY_SNAPSHOT_END)
        mem[ADDR_BATTLE_TYPE] = 1
        mem[ADDR_ENEMY_LEVEL] = enemy_level
        mem[ADDR_ENEMY_HP] = enemy_hp & 0xFF
        mem[ADDR_ENEMY_HP + 1] = (enemy_hp >> 8) & 0xFF
        mem[ADDR_ENEMY_MAXHP] = enemy_maxhp & 0xFF
        mem[ADDR_ENEMY_MAXHP + 1] = (enemy_maxhp >> 8) & 0xFF
        mem[ADDR_ACTIVE_LEVEL] = active_level
        mem[ADDR_POKEMON_MAXHP] = 50
        return mem

    data = make_data(mem_with(2, 100, 100, 10))
    expect_scale = max(data.battle_difficulty_floor, 2 / 10)

    # Step 1: good frame — enemy_lv=2 readable, real damage dealt (100->60).
    prev1 = mem_with(2, 100, 100, 10)
    data.pyboy.memory = mem_with(2, 60, 100, 10)
    r1 = data.reward_enemy_hp(bytes(prev1))
    dbg1 = data.last_enemy_hp_debug
    print(f"step1 (good frame): reward={r1} dbg={dbg1}")
    assert dbg1["enemy_level"] == 2
    assert abs(dbg1["difficulty_scale"] - expect_scale) < 1e-9

    # Step 2: bad frame — enemy_lv misreads 0 in both prev and current, but
    # HP is still clearly dropping (a real, finishing hit: 60->0). Must
    # reuse the level cached in step 1, not silently default to the floor.
    prev2 = mem_with(0, 60, 100, 10)
    data.pyboy.memory = mem_with(0, 0, 100, 10)
    r2 = data.reward_enemy_hp(bytes(prev2))
    dbg2 = data.last_enemy_hp_debug
    print(f"step2 (bad frame, enemy_lv misreads 0): reward={r2} dbg={dbg2}")
    assert dbg2["enemy_level"] == 2, "must fall back to cached enemy level, not 0"
    assert abs(dbg2["difficulty_scale"] - expect_scale) < 1e-9
    print("ENEMY_LEVEL_CACHE_OK=True")
    print()


def main() -> None:
    check_snapshot_coverage()
    run_mock_cases()
    run_level_soft_cap_cases()
    run_enemy_level_cache_case()
    try:
        try_live_outcomes()
    except Exception as exc:
        print(f"LIVE ERROR (non-fatal): {type(exc).__name__}: {exc}")
    print("DONE")


if __name__ == "__main__":
    main()
