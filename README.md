# SQM Hybrid 생산 계획 시뮬레이터

SQM 계획 평준화(휴리스틱) vs RL 에이전트 성능을 비교하는 Web 기반 시뮬레이터입니다.

## 실행 방법

### 1. 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

### 2. 백엔드 서버 실행

```bash
cd backend
uvicorn main:app --reload --port 8000
```

서버가 http://localhost:8000 에서 실행됩니다.

### 3. 프론트엔드 실행

`frontend/index.html` 파일을 브라우저에서 직접 열거나:

```bash
cd frontend
python -m http.server 3000
```

http://localhost:3000 으로 접속합니다.

## 기능

- **AS-IS 그리디**: 납기 순 정렬 후 부하 최소 라인 배정
- **SQM 휴리스틱**: Mold → Color → Due 순 정렬, Mold/Color 연속성 우선 배정
- **RL 에이전트**: SQM Hybrid Dueling DQN 모델 추론 (n_lines=10 필요)

## API 엔드포인트

- `GET /api/default-config` - 기본 오더 및 제약 설정 반환
- `POST /api/simulate` - 시뮬레이션 실행

## 프로젝트 구조

```
dynamic-production-planning-SQM-hybrid-simulator/
├── backend/
│   ├── main.py              # FastAPI 앱
│   ├── engines/
│   │   ├── heuristic.py     # SQM 휴리스틱 엔진
│   │   ├── greedy.py        # AS-IS 그리디 엔진
│   │   └── rl_engine.py     # RL 에이전트 엔진
│   └── requirements.txt
├── frontend/
│   └── index.html           # 단일 파일 UI
└── README.md
```
