# Space Robot Planning

## Version 2 (Current)

### 개요
- **CVAE (Conditional Variational Autoencoder)** 모델과 **MLP (Baseline)** 모델을 사용한 로봇 궤적 계획 학습
- 고정된 시작점에서 다양한 목표점으로의 궤적 생성 학습

### 주요 변경사항 (v1 → v2)
- **Trajectory 파라미터**: 5분절 → **4분절** (각 관절당 3개 파라미터)
- **Spline Interpolation**: Linear → **3차 스플라인** (각 waypoint에서 미분=0)
- **Evaluation**: 쿼터니언 RK4 적분 기반 최종 오차 평가

### 주요 특징

#### 학습 설정
- **고정 시작점**: `q0_start = [0, 0, 0, 1]` (단위 쿼터니언)
- **랜덤 목표점**: 매 epoch마다 랜덤하게 생성된 정규화된 쿼터니언
- **물리 시뮬레이션**: PhysicsLayer를 통한 궤적 검증
- **Validation**: 10 epoch마다 고정된 목표점에 대한 검증 수행

#### CVAE 모델 설정
- **Batch Size**: 512
- **Epochs**: 15,000
- **Latent Dimension**: 8
- **Condition Dimension**: 8 (Start(4) + Goal(4))
- **Waypoints**: 3 (4분절: 시작점 + 중간 3개 + 끝점)
- **Total Time**: 1.0초
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Hidden Dimension**: 256

#### MLP 모델 설정
- **Batch Size**: 256
- **Epochs**: 15,000
- **Condition Dimension**: 8 (Start(4) + Goal(4))
- **Waypoints**: 3 (4분절)
- **Total Time**: 1.0초
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Hidden Dimension**: 128

### Trajectory Generation (v2)

#### 4분절 3차 스플라인
- **구조**: 시작점(0) + 중간 waypoint 3개 + 끝점(0) = 총 5개 점으로 4분절 구성
- **3차 Hermite 스플라인**: 각 분절에서 `q(t) = q_start + (q_end - q_start) * t² * (3 - 2*t)`
- **속도**: `q'(t) = (q_end - q_start) * 6*t*(1-t)`
- **특징**: 각 waypoint에서 미분이 0이 되어 부드러운 궤적 보장

### 저장 파일

#### 학습 데이터
- **CSV 파일**: `plots/{model}_training_curve/v2.csv`
  - 컬럼: `epoch`, `train_loss`, `epoch_duration`, `val_loss`
- **플롯 이미지**: `plots/{model}_training_curve/v2.png`
  - Train Loss와 Validation Loss 곡선
- **모델 가중치**: 
  - CVAE: `weights/cvae_debug/v2.pth`
  - MLP: `weights/mlp_debug/v2.pth`

#### TensorBoard 로그
- **CVAE**: `runs/cvae_debug_v1/`
- **MLP**: `runs/mlp_debug_v1/`

### 모델 아키텍처

#### CVAE
- **Encoder**: Condition + Trajectory → (μ, log σ²)
- **Decoder**: Condition + Latent z → Waypoints
- **Inference**: Decoder만 사용하여 랜덤 샘플링된 z로 궤적 생성

#### MLP (Baseline)
- **구조**: 4층 MLP (128 hidden units)
- **입력**: Condition (Start + Goal)
- **출력**: Waypoints (결정론적)

### Physics Simulation

#### 학습/최적화
- **쿼터니언 Euler 적분**: `q_{k+1} = q_k + 0.5 * q_k * wb * dt`
- **Non-holonomic Constraint**: SPART 동역학 기반 각속도 계산

#### 평가 (Evaluation)
- **쿼터니언 RK4 적분**: 4차 Runge-Kutta로 더 정확한 자세 추적
- **최종 오차**: 각도 오차 `θ²` (쿼터니언 내적 기반)

### 학습 프로세스
1. 매 epoch마다 랜덤 목표점 생성
2. CVAE: 랜덤 샘플링된 z로 waypoints 예측
3. MLP: 조건부로 waypoints 예측
4. PhysicsLayer를 통한 궤적 시뮬레이션 및 손실 계산
5. 역전파 및 가중치 업데이트
6. 10 epoch마다 고정 목표점에 대한 검증 및 시각화


**적분 방법**
- 학습/최적화: Euler 적분 (`q_{k+1} = q_k + 0.5 * q_k * wb * dt`)
- 평가: RK4 적분 (4차 Runge-Kutta)

#### 최적화 하이퍼파라미터

**직접 최적화** (`optimize_direct.py`)
- `OPTIMIZER`: Adam
- `LEARNING_RATE`: 0.05
- `MAX_ITERATIONS`: 200
- `EARLY_STOP_THRESHOLD`: 1e-4
- `INIT_STD`: 0.1 (초기 waypoint 랜덤 초기화 표준편차)

**NN 기반 최적화** (`optimize_nn.py`)
- `OPTIMIZER`: LBFGS
- `LBFGS_LR`: 1.0
- `LBFGS_MAX_ITER`: 20 (LBFGS 내부 반복 횟수)
- `LBFGS_HISTORY_SIZE`: 10
- `LBFGS_LINE_SEARCH`: "strong_wolfe"
- `CVAE_NUM_SAMPLES`: 10 (CVAE inference 후 최적 샘플 선택)
- `NUM_WAYPOINTS`: 3
- `TOTAL_TIME`: 1.0초

#### 평가 하이퍼파라미터 (`evaluate.py`)

**CVAE 평가**
- `NUM_SAMPLES`: 100 (평가용 샘플 수)
- `LATENT_DIM`: 8
- 평가 방법: RK4 적분 기반 최종 오차 계산

**MLP 평가**
- 평가 방법: Single shot (결정론적)
- 평가 방법: RK4 적분 기반 최종 오차 계산

#### 궤적 생성 파라미터

**3차 스플라인**
- 분절 수: 4 (시작점 + 중간 3개 waypoint + 끝점)
- 각 분절당 스텝 수: `seg_steps = num_steps // num_waypoints` (약 2-3 스텝)
- 스플라인 함수: `q(t) = q_start + (q_end - q_start) * t² * (3 - 2*t)`
- 속도 함수: `q'(t) = (q_end - q_start) * 6*t*(1-t)`
- 경계 조건: 각 waypoint에서 미분 = 0

#### 고정 테스트 케이스

**시작점**
- `q0_start`: `[0, 0, 0, 1]` (단위 쿼터니언, 회전 없음)

**목표점**
- `q0_goal`: `[0, 0, 0.7071, 0.7071]` (Z축 기준 90도 회전)
- 또는: `[π/100, π/100, π/100]` (Euler angles, 평가용)

### 사용 방법
```bash
# CVAE 학습
python train_cvae.py

# MLP 학습
python train_mlp.py

# 평가 (RK4 기반)
python evaluate.py

# 직접 최적화
python optimize_direct.py

# CVAE warm start + LBFGS 최적화
python optimize_nn.py
```

---

## Version 1 (Legacy)

### 주요 알고리즘
- **5분절 Linear Spline**: 시작점 + 중간 waypoint 4개 + 끝점
- **Linear Interpolation**: 각 분절에서 선형 보간
- **쿼터니언 Euler 적분**: 물리 시뮬레이션

### 저장 파일
- **CSV**: `plots/{model}_training_curve/v1.csv`
- **플롯**: `plots/{model}_training_curve/v1.png`
- **가중치**: `weights/{model}_debug/v1.pth`
