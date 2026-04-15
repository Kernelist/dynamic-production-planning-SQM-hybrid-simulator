# -*- coding: utf-8 -*-
"""
heuristic.py
SQM 계획 평준화 휴리스틱 엔진
정렬: mold_id → color_id → due 순
라인 선택: 같은 mold 마지막 배치 라인 우선 → 같은 color → 부하 최소 라인
"""

import math
import numpy as np
from engines.mold_utils import (
    DEFAULT_MOLD_QUANTITIES, register_mold,
    filter_by_mold_constraint
)


SLOT_QTY = 50  # 슬롯당 생산 수량


def run_heuristic(orders, n_lines, n_slots, line_valid_models,
                  mold_quantities=None):
    """
    Args:
        orders: list of dict {model, qty, due, mold_id, color_id}
        n_lines: int
        n_slots: int
        line_valid_models: dict {line_idx: [model_list]}
        mold_quantities: dict {mold_id: qty} — None이면 기본값 사용
    Returns:
        schedule: list[list[list|None]]  n_lines x n_slots
        metrics: dict
    """
    if mold_quantities is None:
        mold_quantities = DEFAULT_MOLD_QUANTITIES

    # mold_id → color_id → due 순 정렬
    sorted_orders = sorted(
        orders,
        key=lambda o: (o["mold_id"], o["color_id"], o["due"])
    )

    # 스케줄 초기화
    schedule = [[None] * n_slots for _ in range(n_lines)]
    line_next_slot = [0] * n_lines
    # 각 라인의 마지막 mold_id, color_id 추적
    line_last_mold = [0] * n_lines
    line_last_color = [0] * n_lines
    mold_schedule = {}  # Mold 배정 이력

    for order in sorted_orders:
        model = order["model"]
        qty = order["qty"]
        due = order["due"]
        mold_id = order["mold_id"]
        color_id = order["color_id"]

        n_slots_needed = max(1, math.ceil(qty / SLOT_QTY))

        # 이 모델을 처리할 수 있는 라인 필터
        valid_lines = [l for l in range(n_lines) if model in line_valid_models.get(l, [])]
        if not valid_lines:
            valid_lines = list(range(n_lines))

        # 슬롯이 남아있는 라인만 대상
        available = [l for l in valid_lines if line_next_slot[l] + n_slots_needed <= n_slots]
        if not available:
            available = valid_lines  # 슬롯 초과라도 배정 시도

        # Mold 수량 제약 필터
        start_slots = {l: line_next_slot[l] for l in available}
        available = filter_by_mold_constraint(
            available, mold_id, start_slots, n_slots_needed,
            mold_schedule, mold_quantities
        )

        # 우선순위 1: 같은 mold를 마지막으로 배치한 라인
        same_mold = [l for l in available if line_last_mold[l] == mold_id]
        if same_mold:
            # 그 중 같은 color 우선
            same_color = [l for l in same_mold if line_last_color[l] == color_id]
            if same_color:
                chosen_line = min(same_color, key=lambda l: line_next_slot[l])
            else:
                chosen_line = min(same_mold, key=lambda l: line_next_slot[l])
        else:
            # 우선순위 2: 같은 color를 마지막으로 배치한 라인
            same_color = [l for l in available if line_last_color[l] == color_id]
            if same_color:
                chosen_line = min(same_color, key=lambda l: line_next_slot[l])
            else:
                # 우선순위 3: 부하 최소 라인
                chosen_line = min(available, key=lambda l: line_next_slot[l])

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
        line_last_mold[chosen_line] = mold_id
        line_last_color[chosen_line] = color_id

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
            if c >= due:
                due_violations += 1
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
