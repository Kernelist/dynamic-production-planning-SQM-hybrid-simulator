# -*- coding: utf-8 -*-
"""
mold_utils.py
Mold 수량 제약 유틸리티

각 Mold 타입의 보유 수량만큼만 동시에 라인에 배정 가능.
동일 Mold가 서로 다른 라인에서 같은 슬롯 구간에 겹치면 제약 위반.
"""

# 기본 Mold 보유 수량 (시뮬레이터 기본값)
DEFAULT_MOLD_QUANTITIES = {
    1: 2,   # Mold 1 (Model 1,2)
    2: 3,   # Mold 2 (Model 3,4) — 가장 수요 많음
    3: 2,   # Mold 3 (Model 5,6)
    4: 3,   # Mold 4 (Model 7,8) — 수요 많음
    5: 1,   # Mold 5 (Model 9,10)
}


def is_mold_feasible(line: int, mold_id: int, start_slot: int, n_slots_needed: int,
                     mold_schedule: dict, mold_quantities: dict) -> bool:
    """
    특정 라인에 Mold 배정이 수량 제약을 만족하는지 검사.

    Args:
        line:           배정하려는 라인 번호
        mold_id:        배정하려는 Mold ID
        start_slot:     배정 시작 슬롯
        n_slots_needed: 필요한 슬롯 수
        mold_schedule:  {line: [(mold_id, start, end), ...]} — 현재까지의 Mold 배정 이력
        mold_quantities:{mold_id: qty} — Mold 보유 수량

    Returns:
        True  — 제약 만족 (배정 가능)
        False — 제약 위반 (배정 불가)
    """
    qty_limit = mold_quantities.get(mold_id, 99)
    end_slot = start_slot + n_slots_needed

    # 같은 시간 구간에 동일 Mold를 사용하는 다른 라인 수 계산
    concurrent = 0
    for other_line, ranges in mold_schedule.items():
        if other_line == line:
            continue
        for (mid, s, e) in ranges:
            if mid == mold_id and s < end_slot and e > start_slot:
                concurrent += 1
                break  # 해당 라인에서 겹침 확인 → 다음 라인으로

    return concurrent < qty_limit


def register_mold(line: int, mold_id: int, start_slot: int, n_slots_needed: int,
                  mold_schedule: dict):
    """Mold 배정 이력에 등록."""
    if line not in mold_schedule:
        mold_schedule[line] = []
    mold_schedule[line].append((mold_id, start_slot, start_slot + n_slots_needed))


def filter_by_mold_constraint(valid_lines: list, mold_id: int, start_slots: dict,
                               n_slots_needed: int, mold_schedule: dict,
                               mold_quantities: dict) -> list:
    """
    valid_lines 중 Mold 수량 제약을 만족하는 라인만 반환.
    만족하는 라인이 없으면 원본 리스트 반환 (fallback).
    """
    feasible = [
        l for l in valid_lines
        if is_mold_feasible(l, mold_id, start_slots[l], n_slots_needed,
                            mold_schedule, mold_quantities)
    ]
    return feasible if feasible else valid_lines


def count_mold_violations(schedule: list, n_lines: int, n_slots: int) -> int:
    """
    스케줄에서 Mold 수량 제약 위반 슬롯 수 계산.
    동일 슬롯에서 같은 Mold를 사용하는 라인 쌍 수를 반환.
    """
    violations = 0
    for t in range(n_slots):
        mold_at_slot = {}
        for r in range(n_lines):
            cell = schedule[r][t]
            if cell is not None:
                mid = cell[3]
                mold_at_slot.setdefault(mid, []).append(r)
        # 동일 Mold가 여러 라인에서 사용되는 경우는 위반으로 표시 (단순 카운트)
        for mid, lines in mold_at_slot.items():
            if len(lines) > 1:
                violations += len(lines) - 1
    return violations
