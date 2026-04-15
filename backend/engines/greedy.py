# -*- coding: utf-8 -*-
"""
greedy.py
AS-IS 그리디 엔진: due 날짜 순 정렬 후 부하 최소 라인 배정
"""

import math
import numpy as np
from engines.mold_utils import (
    DEFAULT_MOLD_QUANTITIES, register_mold,
    filter_by_mold_constraint
)


SLOT_QTY = 50  # 슬롯당 생산 수량


def run_greedy(orders, n_lines, n_slots, line_valid_models,
               mold_quantities=None):
    """
    Args:
        orders: list of dict {model, qty, due, mold_id, color_id}
        n_lines: int
        n_slots: int
        line_valid_models: dict {line_idx: [model_list]}
        mold_quantities: dict {mold_id: qty} — None이면 기본값 사용
    Returns:
        schedule: list[list[list|None]]  n_lines x n_slots, each cell = [model,qty,due,mold,color] or None
        metrics: dict
    """
    if mold_quantities is None:
        mold_quantities = DEFAULT_MOLD_QUANTITIES

    # due 순 정렬 (AS-IS 방식)
    sorted_orders = sorted(enumerate(orders), key=lambda x: x[1]["due"])

    # 스케줄 초기화 (None = 빈 슬롯)
    schedule = [[None] * n_slots for _ in range(n_lines)]
    # 각 라인의 다음 빈 슬롯 포인터
    line_next_slot = [0] * n_lines
    mold_schedule = {}  # Mold 배정 이력

    for orig_idx, order in sorted_orders:
        model = order["model"]
        qty = order["qty"]
        due = order["due"]
        mold_id = order["mold_id"]
        color_id = order["color_id"]

        n_slots_needed = max(1, math.ceil(qty / SLOT_QTY))

        # 이 모델을 처리할 수 있는 라인 중 부하 최소 라인 선택
        valid_lines = [l for l in range(n_lines) if model in line_valid_models.get(l, [])]
        if not valid_lines:
            valid_lines = list(range(n_lines))

        # Mold 수량 제약 필터
        start_slots = {l: line_next_slot[l] for l in valid_lines}
        valid_lines = filter_by_mold_constraint(
            valid_lines, mold_id, start_slots, n_slots_needed,
            mold_schedule, mold_quantities
        )

        # 부하 = 현재 배정된 슬롯 수
        chosen_line = min(valid_lines, key=lambda l: line_next_slot[l])

        # 슬롯 배정
        placed = 0
        col = line_next_slot[chosen_line]
        while placed < n_slots_needed and col < n_slots:
            schedule[chosen_line][col] = [model, qty, due, mold_id, color_id]
            placed += 1
            col += 1

        register_mold(chosen_line, mold_id, line_next_slot[chosen_line],
                      n_slots_needed, mold_schedule)
        line_next_slot[chosen_line] = col

    metrics = _calc_metrics(schedule, n_lines, n_slots)
    return schedule, metrics


def _calc_metrics(schedule, n_lines, n_slots):
    due_violations = 0
    mold_changes = 0
    color_changes = 0
    ct_per_line = []

    for r in range(n_lines):
        line_ct = 0
        for c in range(n_slots):
            cell = schedule[r][c]
            if cell is None:
                continue
            model, qty, due, mold_id, color_id = cell
            line_ct += qty
            # 납기 위반: 슬롯 인덱스(0-based week proxy) > due
            if c >= due:
                due_violations += 1
            # Mold/Color 변경
            if c > 0:
                prev = schedule[r][c - 1]
                if prev is not None:
                    if prev[3] != mold_id:
                        mold_changes += 1
                    if prev[4] != color_id:
                        color_changes += 1
        ct_per_line.append(line_ct)

    ct_std = float(np.std(ct_per_line)) if ct_per_line else 0.0

    slot_counts = []
    for r in range(n_lines):
        cnt = sum(1 for c in range(n_slots) if schedule[r][c] is not None)
        slot_counts.append(cnt)
    balance_std = float(np.std(slot_counts)) if slot_counts else 0.0

    return {
        "due_violations": due_violations,
        "mold_changes": mold_changes,
        "color_changes": color_changes,
        "ct_std": round(ct_std, 2),
        "balance_std": round(balance_std, 2),
    }
