# -*- coding: utf-8 -*-
"""
rl_engine.py
RL 에이전트 엔진 - SQMHybridDuelDQN 자체 포함 (import 의존성 없음)

saved_models/sqm_hybrid_best.pt 로드 후 추론
n_lines != 10 이면 available=False 반환
"""

import math
import os
import numpy as np
from engines.mold_utils import (
    DEFAULT_MOLD_QUANTITIES, register_mold,
    filter_by_mold_constraint
)

# ── 상수 ────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 209
ACTION_SIZE = 10
N_LINES     = 10
N_SLOTS     = 60
N_MODELS    = 10
N_MOLDS     = 5
N_COLORS    = 8
SLOT_QTY    = 50

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "dynamic-production-planning-SQM-hybrid",
    "src", "saved_models", "sqm_hybrid_best.pt"
)

# ── DEFAULT LINE_VALID_MODELS (학습에 사용된 기본값) ────────────────────────
_DEFAULT_LINE_VALID_MODELS = {
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

# ── PyTorch 네트워크 클래스 (sqm_hybrid_dqn.py에서 복사) ────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    import math as _math

    class NoisyLinear(nn.Module):
        def __init__(self, in_features, out_features, sigma_init=0.5):
            super().__init__()
            self.in_features  = in_features
            self.out_features = out_features
            self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.bias_mu      = nn.Parameter(torch.empty(out_features))
            self.bias_sigma   = nn.Parameter(torch.empty(out_features))
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.register_buffer('bias_epsilon',   torch.empty(out_features))
            self.sigma_init = sigma_init
            self._reset_parameters()
            self._sample_noise()

        def _reset_parameters(self):
            mu = 1.0 / _math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu, mu)
            self.weight_sigma.data.fill_(self.sigma_init / _math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu, mu)
            self.bias_sigma.data.fill_(self.sigma_init / _math.sqrt(self.out_features))

        @staticmethod
        def _scale_noise(size):
            x = torch.randn(size)
            return x.sign() * x.abs().sqrt()

        def _sample_noise(self):
            eps_in  = self._scale_noise(self.in_features)
            eps_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)

        def forward(self, x):
            if self.training:
                w = self.weight_mu + self.weight_sigma * self.weight_epsilon
                b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
            else:
                w, b = self.weight_mu, self.bias_mu
            return F.linear(x, w, b)


    class SQMHybridDuelDQN(nn.Module):
        """184차원 입력 Dueling DQN (NoisyNet + LayerNorm)"""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(INPUT_SIZE, 512)
            self.ln1 = nn.LayerNorm(512)
            self.fc2 = nn.Linear(512, 256)
            self.ln2 = nn.LayerNorm(256)
            self.fc3 = NoisyLinear(256, 128)

            self.value_fc  = nn.Linear(128, 64)
            self.value_out = nn.Linear(64, 1)

            self.adv_fc  = nn.Linear(128, 64)
            self.adv_out = nn.Linear(64, ACTION_SIZE)

        def forward(self, x):
            x      = F.relu(self.ln1(self.fc1(x)))
            x      = F.relu(self.ln2(self.fc2(x)))
            shared = F.relu(self.fc3(x))

            v = F.relu(self.value_fc(shared))
            v = self.value_out(v)

            a = F.relu(self.adv_fc(shared))
            a = self.adv_out(a)

            return v + a - a.mean(dim=1, keepdim=True)


# ── 상태 변환 (sqm_hybrid_env.py state_transform_hybrid 복사) ────────────────
def _state_transform(s: np.ndarray, order: np.ndarray,
                     mold_quantities: dict = None) -> np.ndarray:
    """
    상태 + 오더 → 209차원 관측 벡터
    [0:10]    현재 오더 모델 one-hot   (10)
    [10]      현재 오더 수량/SLOT_QTY  (1)
    [11:16]   현재 오더 Mold one-hot   (5)
    [16:24]   현재 오더 Color one-hot  (8)
    [24:124]  각 라인의 마지막 모델 one-hot (10×10)
    [124:174] 각 라인의 마지막 Mold one-hot (5×10)
    [174:184] 각 라인의 마지막 배정 날짜    (10)
    [184:194] 각 라인의 슬롯 점유율         (10)
    [194:204] 각 라인의 납기 여유도         (10)
    [204:209] Mold별 가용 비율             (5)
    """
    if mold_quantities is None:
        mold_quantities = DEFAULT_MOLD_QUANTITIES
    model_enc = [0.0] * N_MODELS
    m_idx = int(order[0]) - 1
    if 0 <= m_idx < N_MODELS:
        model_enc[m_idx] = 1.0

    qty_enc = [float(order[1]) / SLOT_QTY]

    mold_enc = [0.0] * N_MOLDS
    mold_idx = int(order[3]) - 1
    if 0 <= mold_idx < N_MOLDS:
        mold_enc[mold_idx] = 1.0

    color_enc = [0.0] * N_COLORS
    color_idx = int(order[4]) - 1
    if 0 <= color_idx < N_COLORS:
        color_enc[color_idx] = 1.0

    line_last_model = []
    for r in range(N_LINES):
        nz = np.nonzero(s[r, :, 0])[0]
        last_m = int(s[r, int(nz.max()), 0]) if len(nz) > 0 else 0
        enc = [0.0] * N_MODELS
        if 1 <= last_m <= N_MODELS:
            enc[last_m - 1] = 1.0
        line_last_model += enc

    line_last_mold = []
    for r in range(N_LINES):
        nz = np.nonzero(s[r, :, 3])[0]
        last_mold = int(s[r, int(nz.max()), 3]) if len(nz) > 0 else 0
        enc = [0.0] * N_MOLDS
        if 1 <= last_mold <= N_MOLDS:
            enc[last_mold - 1] = 1.0
        line_last_mold += enc

    line_last_date = []
    for r in range(N_LINES):
        nz = np.nonzero(s[r, :, 1])[0]
        last_date = float(int(nz.max())) / N_SLOTS if len(nz) > 0 else 0.0
        line_last_date.append(last_date)

    line_utilization = [
        float(np.count_nonzero(s[r, :, 1])) / N_SLOTS
        for r in range(N_LINES)
    ]

    # 납기 여유도: (due - next_slot - n_slots_needed) / N_SLOTS, clipped [-1, 1]
    # order = [model, qty, due, mold_id, color_id]
    due = float(order[2])
    n_slots_needed = max(1, int(order[1] / SLOT_QTY))
    line_slack = []
    for r in range(N_LINES):
        nz = np.nonzero(s[r, :, 1])[0]
        next_slot = int(nz.max()) + 1 if len(nz) > 0 else 0
        slack = (due - next_slot - n_slots_needed) / N_SLOTS
        line_slack.append(float(np.clip(slack, -1.0, 1.0)))

    # Mold별 가용 비율 (5차원) — Mold 수량 제약 인식용
    mold_avail = []
    for m in range(1, 6):
        qty_limit = mold_quantities.get(m, 1)
        concurrent = np.sum((s[:, :, 3] == m) & (s[:, :, 1] > 0), axis=0)
        max_concurrent = int(concurrent.max()) if concurrent.max() > 0 else 0
        avail = max(0.0, (qty_limit - max_concurrent) / qty_limit)
        mold_avail.append(float(avail))

    return np.array(
        model_enc + qty_enc + mold_enc + color_enc +
        line_last_model + line_last_mold + line_last_date + line_utilization + line_slack +
        mold_avail,
        dtype=np.float32
    )


def _get_action_mask(order_model: int, line_valid_models: dict) -> list:
    return [l for l in range(N_LINES) if order_model in line_valid_models.get(l, [])]


# ── 모델 로드 (싱글턴) ────────────────────────────────────────────────────────
_model_cache = None
_model_load_error = None


def _load_model():
    global _model_cache, _model_load_error
    if _model_cache is not None:
        return _model_cache, None
    if _model_load_error is not None:
        return None, _model_load_error

    if not _TORCH_AVAILABLE:
        _model_load_error = "torch not installed"
        return None, _model_load_error

    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path):
        _model_load_error = f"Model file not found: {path}"
        return None, _model_load_error

    try:
        net = SQMHybridDuelDQN()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        _model_cache = net
        return net, None
    except Exception as e:
        _model_load_error = str(e)
        return None, _model_load_error


# ── 메인 추론 함수 ─────────────────────────────────────────────────────────────
def run_rl_agent(orders, n_lines, line_valid_models,
                 mold_quantities=None):
    """
    Args:
        orders: list of dict {model, qty, due, mold_id, color_id}
        n_lines: int (10이어야 사용 가능)
        line_valid_models: dict {line_idx: [model_list]}
        mold_quantities: dict {mold_id: qty} — None이면 기본값 사용
    Returns:
        schedule: list[list[list|None]]
        metrics: dict
        available: bool
        note: str
    """
    if mold_quantities is None:
        mold_quantities = DEFAULT_MOLD_QUANTITIES

    if n_lines != N_LINES:
        return None, {}, False, f"RL 에이전트는 라인 수 {N_LINES}일 때만 사용 가능합니다 (현재: {n_lines})"

    net, err = _load_model()
    if net is None:
        return None, {}, False, f"모델 로드 실패: {err}"

    n_slots = N_SLOTS

    # 상태 행렬 초기화 (N_LINES, N_SLOTS, 5)
    s = np.zeros((N_LINES, n_slots, 5), dtype=np.float32)
    schedule = [[None] * n_slots for _ in range(N_LINES)]
    line_next_slot = [0] * N_LINES
    mold_schedule = {}  # Mold 배정 이력

    # 오더를 numpy 배열로 변환 [model, qty, due, mold_id, color_id]
    orders_arr = np.array(
        [[o["model"], o["qty"], o["due"], o["mold_id"], o["color_id"]] for o in orders],
        dtype=np.float32
    )

    # Mold → Color → Due 사전 정렬 (SQM 휴리스틱과 동일 순서)
    # 오더 그룹화로 Mold/Color Change 최소화 보장
    sort_idx = np.lexsort((orders_arr[:, 2], orders_arr[:, 4], orders_arr[:, 3]))
    orders_arr = orders_arr[sort_idx]

    with torch.no_grad():
        for i, order_vec in enumerate(orders_arr):
            order_model = int(order_vec[0])
            obs = _state_transform(s, order_vec, mold_quantities)
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            q_vals = net(obs_t).squeeze(0)

            # Action Masking (모델 제약)
            valid = _get_action_mask(order_model, line_valid_models)
            if not valid:
                valid = list(range(N_LINES))

            # 슬롯 수 먼저 계산 (납기 feasibility 체크에 필요)
            qty = int(order_vec[1])
            due = int(order_vec[2])
            mold_id = int(order_vec[3])
            color_id = int(order_vec[4])
            n_slots_needed = max(1, math.ceil(qty / SLOT_QTY))

            # Mold 수량 제약 필터
            start_slots = {l: line_next_slot[l] for l in valid}
            valid = filter_by_mold_constraint(
                valid, mold_id, start_slots, n_slots_needed,
                mold_schedule, mold_quantities
            )

            # 현재 오더의 Mold/Color 연속성 확인용: 각 라인의 마지막 mold/color
            line_last_mold_val = {}
            line_last_color_val = {}
            for l in range(N_LINES):
                nz = np.nonzero(s[l, :, 1])[0]
                if len(nz) > 0:
                    last_slot = int(nz.max())
                    line_last_mold_val[l]  = int(s[l, last_slot, 3])
                    line_last_color_val[l] = int(s[l, last_slot, 4])
                else:
                    line_last_mold_val[l]  = 0
                    line_last_color_val[l] = 0

            # 라인 선택 (방법1+방법2 조합):
            #   Step1: 납기 내 완료 가능 라인(EDD-aware feasibility)
            #   Step2: Q-value + Mold/Color 연속성 보너스로 최적 라인 선택
            MOLD_BONUS  = 3.0   # Mold 연속성 인센티브
            COLOR_BONUS = 1.5   # Color 연속성 인센티브

            feasible = [l for l in valid
                        if line_next_slot[l] + n_slots_needed <= due]
            candidates = feasible if feasible else sorted(
                valid, key=lambda l: line_next_slot[l]
            )[:max(1, math.ceil(len(valid) / 2))]

            mask = torch.full((N_LINES,), float('-inf'))
            for l in candidates:
                bonus = 0.0
                if line_last_mold_val[l] == mold_id:
                    bonus += MOLD_BONUS
                if line_last_color_val[l] == color_id:
                    bonus += COLOR_BONUS
                mask[l] = q_vals[l] + bonus

            chosen_line = int(mask.argmax().item())

            placed = 0
            col = line_next_slot[chosen_line]
            while placed < n_slots_needed and col < n_slots:
                schedule[chosen_line][col] = [order_model, qty, due, mold_id, color_id]
                s[chosen_line, col] = order_vec
                placed += 1
                col += 1

            register_mold(chosen_line, mold_id, line_next_slot[chosen_line],
                          n_slots_needed, mold_schedule)
            line_next_slot[chosen_line] = col

    metrics = _calc_metrics(schedule, N_LINES, n_slots)
    return schedule, metrics, True, "RL 에이전트 (SQM Hybrid Dueling DQN)"


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
    slot_counts = [
        sum(1 for c in range(n_slots) if schedule[r][c] is not None)
        for r in range(n_lines)
    ]
    balance_std = float(np.std(slot_counts)) if slot_counts else 0.0

    return {
        "due_violations": due_violations,
        "mold_changes": mold_changes,
        "color_changes": color_changes,
        "ct_std": round(ct_std, 2),
        "balance_std": round(balance_std, 2),
    }
