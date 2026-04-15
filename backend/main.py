# -*- coding: utf-8 -*-
"""
main.py
SQM Hybrid 생산 계획 시뮬레이터 - FastAPI 백엔드
실행: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from engines.heuristic import run_heuristic
from engines.greedy import run_greedy
from engines.edd import run_edd
from engines.spt import run_spt
from engines.rl_engine import run_rl_agent
from engines.mold_utils import DEFAULT_MOLD_QUANTITIES

app = FastAPI(title="SQM Hybrid Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 기본 데이터 ───────────────────────────────────────────────────────────────
DEFAULT_ORDERS = [
    {"model": 6, "qty": 200, "due": 2,  "mold_id": 3, "color_id": 1},
    {"model": 3, "qty": 300, "due": 2,  "mold_id": 2, "color_id": 1},
    {"model": 2, "qty": 300, "due": 3,  "mold_id": 1, "color_id": 1},
    {"model": 2, "qty": 300, "due": 3,  "mold_id": 1, "color_id": 2},
    {"model": 1, "qty": 250, "due": 4,  "mold_id": 1, "color_id": 1},
    {"model": 3, "qty": 200, "due": 4,  "mold_id": 2, "color_id": 2},
    {"model": 4, "qty": 300, "due": 2,  "mold_id": 2, "color_id": 1},
    {"model": 4, "qty": 250, "due": 3,  "mold_id": 2, "color_id": 2},
    {"model": 7, "qty": 250, "due": 5,  "mold_id": 4, "color_id": 1},
    {"model": 8, "qty": 250, "due": 5,  "mold_id": 4, "color_id": 1},
    {"model": 3, "qty": 150, "due": 5,  "mold_id": 2, "color_id": 3},
    {"model": 4, "qty": 250, "due": 6,  "mold_id": 2, "color_id": 3},
    {"model": 8, "qty": 250, "due": 6,  "mold_id": 4, "color_id": 2},
    {"model": 5, "qty": 100, "due": 6,  "mold_id": 3, "color_id": 1},
    {"model": 4, "qty": 200, "due": 7,  "mold_id": 2, "color_id": 4},
    {"model": 2, "qty": 300, "due": 7,  "mold_id": 1, "color_id": 3},
    {"model": 6, "qty": 250, "due": 7,  "mold_id": 3, "color_id": 2},
    {"model": 4, "qty": 100, "due": 7,  "mold_id": 2, "color_id": 5},
    {"model": 7, "qty": 100, "due": 8,  "mold_id": 4, "color_id": 2},
    {"model": 3, "qty": 100, "due": 9,  "mold_id": 2, "color_id": 4},
    {"model": 6, "qty": 150, "due": 10, "mold_id": 3, "color_id": 3},
    {"model": 2, "qty": 200, "due": 8,  "mold_id": 1, "color_id": 4},
    {"model": 5, "qty": 150, "due": 11, "mold_id": 3, "color_id": 2},
    {"model": 2, "qty": 200, "due": 13, "mold_id": 1, "color_id": 5},
    {"model": 8, "qty": 150, "due": 12, "mold_id": 4, "color_id": 3},
    {"model": 8, "qty": 250, "due": 11, "mold_id": 4, "color_id": 4},
    {"model": 4, "qty": 200, "due": 12, "mold_id": 2, "color_id": 6},
    {"model": 1, "qty": 150, "due": 12, "mold_id": 1, "color_id": 2},
    {"model": 7, "qty": 100, "due": 13, "mold_id": 4, "color_id": 3},
    {"model": 7, "qty": 300, "due": 14, "mold_id": 4, "color_id": 5},
    {"model": 6, "qty": 150, "due": 14, "mold_id": 3, "color_id": 4},
    {"model": 6, "qty": 250, "due": 15, "mold_id": 3, "color_id": 5},
    {"model": 3, "qty": 200, "due": 13, "mold_id": 2, "color_id": 5},
    {"model": 8, "qty": 300, "due": 16, "mold_id": 4, "color_id": 5},
    {"model": 7, "qty": 200, "due": 16, "mold_id": 4, "color_id": 6},
    {"model": 3, "qty": 200, "due": 17, "mold_id": 2, "color_id": 6},
    {"model": 3, "qty": 100, "due": 17, "mold_id": 2, "color_id": 7},
    {"model": 7, "qty": 300, "due": 18, "mold_id": 4, "color_id": 7},
    {"model": 3, "qty": 250, "due": 19, "mold_id": 2, "color_id": 8},
    {"model": 8, "qty": 200, "due": 18, "mold_id": 4, "color_id": 6},
    {"model": 3, "qty": 100, "due": 20, "mold_id": 2, "color_id": 8},
    {"model": 1, "qty": 250, "due": 21, "mold_id": 1, "color_id": 3},
    {"model": 3, "qty": 100, "due": 23, "mold_id": 2, "color_id": 7},
    {"model": 2, "qty": 250, "due": 22, "mold_id": 1, "color_id": 6},
    {"model": 4, "qty": 100, "due": 23, "mold_id": 2, "color_id": 6},
    {"model": 4, "qty": 250, "due": 24, "mold_id": 2, "color_id": 7},
    {"model": 6, "qty": 200, "due": 23, "mold_id": 3, "color_id": 6},
    {"model": 3, "qty": 150, "due": 24, "mold_id": 2, "color_id": 8},
    {"model": 2, "qty": 100, "due": 22, "mold_id": 1, "color_id": 7},
]

DEFAULT_LINE_VALID_MODELS = {
    0: [1, 3, 4, 5, 6, 7, 8, 9, 10],
    1: [1, 2, 5, 6, 8, 9, 10],
    2: [1, 2, 3, 4, 5, 6, 7, 8],
    3: [1, 2, 3, 4, 6, 7, 9, 10],
    4: [1, 2, 3, 4, 5, 7, 8, 10],
    5: [1, 2, 3, 4, 5, 6, 8, 9, 10],
    6: [2, 3, 4, 5, 6, 7, 8, 9],
    7: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    8: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    9: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────
class OrderItem(BaseModel):
    model: int
    qty: int
    due: int
    mold_id: int
    color_id: int


class SimulateRequest(BaseModel):
    orders: list[OrderItem]
    n_lines: int = 10
    n_slots: int = 60
    line_valid_models: dict[str, list[int]]
    mold_quantities: Optional[Dict[str, int]] = None


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/api/default-config")
def get_default_config():
    return {
        "orders": DEFAULT_ORDERS,
        "line_valid_models": {str(k): v for k, v in DEFAULT_LINE_VALID_MODELS.items()},
        "n_lines": 10,
        "n_slots": 60,
        "mold_quantities": {str(k): v for k, v in DEFAULT_MOLD_QUANTITIES.items()},
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    orders = [o.model_dump() for o in req.orders]
    n_lines = req.n_lines
    n_slots = req.n_slots
    # key를 int로 변환
    lvm = {int(k): v for k, v in req.line_valid_models.items()}
    # mold_quantities: str key → int key 변환, 없으면 기본값 사용
    mold_qty = (
        {int(k): v for k, v in req.mold_quantities.items()}
        if req.mold_quantities else None
    )

    # EDD
    edd_schedule, edd_metrics = run_edd(orders, n_lines, n_slots, lvm, mold_qty)

    # SPT
    spt_schedule, spt_metrics = run_spt(orders, n_lines, n_slots, lvm, mold_qty)

    # Greedy (AS-IS)
    greedy_schedule, greedy_metrics = run_greedy(orders, n_lines, n_slots, lvm, mold_qty)

    # Heuristic (SQM)
    heuristic_schedule, heuristic_metrics = run_heuristic(orders, n_lines, n_slots, lvm, mold_qty)

    # RL Agent
    rl_schedule, rl_metrics, rl_available, rl_note = run_rl_agent(orders, n_lines, lvm, mold_qty)
    if not rl_available:
        rl_schedule = [[None] * n_slots for _ in range(n_lines)]

    return {
        "edd": {
            "schedule": _serialize_schedule(edd_schedule),
            "metrics": edd_metrics,
        },
        "spt": {
            "schedule": _serialize_schedule(spt_schedule),
            "metrics": spt_metrics,
        },
        "greedy": {
            "schedule": _serialize_schedule(greedy_schedule),
            "metrics": greedy_metrics,
        },
        "heuristic": {
            "schedule": _serialize_schedule(heuristic_schedule),
            "metrics": heuristic_metrics,
        },
        "rl_agent": {
            "schedule": _serialize_schedule(rl_schedule),
            "metrics": rl_metrics,
            "available": rl_available,
            "note": rl_note,
        },
    }


def _serialize_schedule(schedule):
    """None → null 직렬화 (FastAPI가 자동 처리하므로 그대로 반환)"""
    return schedule
