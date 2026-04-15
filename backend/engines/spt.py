# -*- coding: utf-8 -*-
"""
spt.py
SPT (Shortest Processing Time) 스케줄링 알고리즘

처리 시간(수량)이 짧은 오더를 먼저 배정
→ 평균 완료 시간 최소화, 라인 처리량 극대화에 집중
"""

import math
import numpy as np
from engines.greedy import _calc_metrics
from engines.mold_utils import (
    DEFAULT_MOLD_QUANTITIES, register_mold,
    filter_by_mold_constraint
)

SLOT_QTY = 50


def run_spt(orders, n_lines, n_slots, line_valid_models,
            mold_quantities=None):
    """
    Args:
        orders: list of dict {model, qty, due, mold_id, color_id}
        n_lines: int
        n_slots: int
        line_valid_models: dict {line_idx: [model_list]}
        mold_quantities: dict {mold_id: qty} — None이면 기본값 사용
    Returns:
        schedule: list[list[list|None]]
        metrics: dict
    """
    if mold_quantities is None:
        mold_quantities = DEFAULT_MOLD_QUANTITIES

    # 처리 시간(qty) 오름차순 정렬 → 짧은 작업 먼저
    sorted_orders = sorted(orders, key=lambda o: (o["qty"], o["due"]))

    schedule = [[None] * n_slots for _ in range(n_lines)]
    line_next_slot = [0] * n_lines
    mold_schedule = {}  # Mold 배정 이력

    for order in sorted_orders:
        model    = order["model"]
        qty      = order["qty"]
        due      = order["due"]
        mold_id  = order["mold_id"]
        color_id = order["color_id"]

        n_slots_needed = max(1, math.ceil(qty / SLOT_QTY))

        valid_lines = [l for l in range(n_lines)
                       if model in line_valid_models.get(l, [])]
        if not valid_lines:
            valid_lines = list(range(n_lines))

        # Mold 수량 제약 필터
        start_slots = {l: line_next_slot[l] for l in valid_lines}
        valid_lines = filter_by_mold_constraint(
            valid_lines, mold_id, start_slots, n_slots_needed,
            mold_schedule, mold_quantities
        )

        # 최소 부하 라인 선택 (SPT는 라인 균등 배분 + 짧은 작업 우선)
        chosen_line = min(valid_lines, key=lambda l: line_next_slot[l])

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
