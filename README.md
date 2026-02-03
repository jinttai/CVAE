# Space Robot Planning with ConditionalDecoder and MLP

로봇 궤적 계획을 위한 Conditional Decoder 및 Multi-Layer Perceptron (MLP) 기반 학습 시스템입니다.

## 개요

이 프로젝트는 우주 로봇의 자세 제어를 위한 궤적 생성 모델을 학습합니다. ConditionalDecoder와 MLP 두 가지 모델을 사용하여 고정된 시작점에서 다양한 목표 자세로의 궤적을 생성하며, 물리 시뮬레이션을 통한 손실 함수를 사용합니다.

## 설치

```bash
# 의존성 설치
pip install -e .

# 또는 requirements.txt 사용
pip install -r requirements.txt

# 개발 의존성 (테스트, 린터 등)
pip install -e ".[dev]"
```

---

## 주요 구성 요소

### 모델 아키텍처

#### ConditionalDecoder (Latent-Conditioned Generator)
- **목적**: 조건부 확률적 궤적 생성
- **입력**: Condition (Start(4) + Goal(4) = 8차원)
- **Latent Dimension**: 3
- **Hidden Dimension**: 256
- **구조**:
  - Decoder: Condition + Latent z → Waypoints (ResNet-style residual blocks)
  - Encoder: (학습 시에만 사용, 현재는 physics loss만 사용)
- **특징**: 랜덤 샘플링을 통한 다양한 궤적 생성 가능

> **Note**: 이전 버전에서는 CVAE로 명명되었으나, 실제로 KL divergence loss 없이 decoder만 학습하므로 ConditionalDecoder로 이름 변경됨. 하위 호환성을 위해 `CVAE` 별칭 유지.

#### MLP (Baseline Model)
- **목적**: 결정론적 궤적 생성 (Baseline)
- **입력**: Condition (Start(4) + Goal(4) = 8차원)
- **Hidden Dimension**: 128
- **구조**: 4층 MLP (ResNet-style residual blocks, 2개)
- **특징**: 동일 조건에서 항상 동일한 출력

### 물리 시뮬레이션 (PhysicsLayer)

- **방법**: 회전행렬(Rotation Matrix) 기반 동역학
- **적분**: R_{k+1} = R_k @ R_delta(wb, dt)
- **동역학**: SPART (Space Robot Dynamics) 기반
- **Non-holonomic Constraint**: 모멘텀 보존 제약 조건
- **손실 함수**: Chordal distance (log scale)
  ```
  loss = log(3 - trace(R_goal^T @ R_final) + ε)
  ```

### 궤적 생성

- **방법**: 4분절 half-cosine (ease-in-out)
- **구조**: 시작점(0) + 중간 waypoint 3개 + 끝점(0) = 총 5개 점
- **basis 함수**:
  - 위치: `b(t) = 0.5 * (1 - cos(pi * t))`
  - 속도: `b'(t) = 0.5 * pi * sin(pi * t)`
  - 따라서: `q(t) = q_start + (q_end - q_start) * b(t)`
- **특징**: 각 waypoint에서 속도(미분)가 0이 되어 부드러운 궤적 보장

---

## 학습 프로세스

### CVAE 학습 (`NN_opt/training/train_cvae.py`)

#### 하이퍼파라미터
- **Batch Size**: 1024
- **Epochs**: 2000
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Total Time**: 10.0초
- **Time Step (dt)**: 0.1초
- **Num Steps**: 100 (total_time / dt)

#### 데이터 생성
- **시작 자세**: 고정 `[0, 0, 0, 1]` (단위 쿼터니언, Identity)
- **목표 자세**: 매 epoch마다 랜덤 생성
  - 방법: Axis-Angle 방식
  - 범위: 랜덤 회전축 + 0~60도 각도
  ```python
  # 랜덤 회전축 생성 (Unit Vector)
  rand_axis = torch.randn(batch_size, 3)
  rand_axis = rand_axis / ||rand_axis||
  
  # 회전 각도 (0 ~ 60도)
  rand_theta = torch.rand(batch_size, 1) * 60° (rad)
  
  # Axis-Angle → Quaternion
  q = [sin(θ/2)*axis, cos(θ/2)]
  ```

#### 학습 과정
1. 매 epoch마다 랜덤 목표점 생성
2. 랜덤 샘플링된 z로 waypoints 예측 (Decoder only)
3. PhysicsLayer를 통한 궤적 시뮬레이션 및 손실 계산
4. 역전파 및 가중치 업데이트
5. 10 epoch마다 고정 목표점에 대한 검증 및 TensorBoard 시각화

#### 저장 파일
- **모델 가중치**: `NN_opt/weight/cvae_debug/v5_joint_change.pth` (v5)
- **학습 곡선 CSV**: `NN_opt/plots/cvae_training_curve/v4.csv`
  - 컬럼: `epoch`, `train_loss`, `epoch_duration`, `val_loss`
- **학습 곡선 이미지**: `NN_opt/plots/cvae_training_curve/v4.png`
- **TensorBoard 로그**: `NN_opt/logs/cvae_v4/`

#### 손실 함수 (v5)
- **Physics Loss**: 물리 시뮬레이션 기반 자세 오차 (Chordal distance, log scale)
- **Joint Squared Penalty**: 관절 각도의 제곱 평균
  ```
  joint_squared_penalty = mean(waypoints^2) * JOINT_SQUARED_WEIGHT
  ```
  - `JOINT_SQUARED_WEIGHT = 0.01`
- **Joint Change Penalty**: 연속된 waypoint 간 관절 변화량의 제곱 평균
  ```
  joint_diff = waypoints[:, 1:, :] - waypoints[:, :-1, :]
  joint_change_penalty = mean(joint_diff^2) * JOINT_CHANGE_WEIGHT
  ```
  - `JOINT_CHANGE_WEIGHT = 0.01`
- **Total Loss**:
  ```
  loss = physics_loss + joint_squared_penalty + joint_change_penalty
  ```

**변경 이력 (v5)**:
- 기존: 최대 관절 각도 패널티 (`max(|waypoints|) * MAX_JOINT_WEIGHT`)
- 변경: 관절 제곱 평균 패널티 + 연속 waypoint 간 변화량 패널티
- 목적: 더 부드러운 궤적 생성 및 작은 관절 각도 유도

### MLP 학습 (`NN_opt/training/train_mlp.py`)

#### 하이퍼파라미터
- **Batch Size**: 1024
- **Epochs**: 2000
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Total Time**: 10.0초
- **Time Step (dt)**: 0.1초
- **Num Steps**: 100

#### 데이터 생성
- **시작 자세**: 고정 `[0, 0, 0, 1]`
- **목표 자세**: CVAE와 동일한 방식 (Axis-Angle, 0~60도)

#### 학습 과정
1. 매 epoch마다 랜덤 목표점 생성
2. MLP로 waypoints 예측 (결정론적)
3. PhysicsLayer를 통한 궤적 시뮬레이션 및 손실 계산
4. 역전파 및 가중치 업데이트
5. 10 epoch마다 고정 목표점에 대한 검증 및 TensorBoard 시각화

#### 저장 파일
- **모델 가중치**: `NN_opt/weight/mlp_debug/v4.pth`
- **학습 곡선 CSV**: `NN_opt/plots/mlp_training_curve/v4.csv`
- **학습 곡선 이미지**: `NN_opt/plots/mlp_training_curve/v4.png`
- **TensorBoard 로그**: `NN_opt/logs/mlp_v4/`

---

## 최적화 프로세스

### CVAE 기반 최적화 (`NN_opt/optimization/optimize_cvae.py`)

#### 단계
1. **Warm Start (CUDA)**:
   - 학습된 CVAE 모델 로드
   - 여러 개의 랜덤 latent z 샘플 생성 (기본 10개)
   - 각 샘플에 대해 waypoints 생성 및 물리 시뮬레이션
   - 손실이 가장 작은 샘플 선택

2. **Refinement (CPU)**:
   - 선택된 waypoints를 CPU로 전환
   - AdamW 또는 LBFGS 최적화
   - 하이퍼파라미터:
     - AdamW: `lr=1e-2`
     - LBFGS: `lr=1.0`, `max_iter=20`, `history_size=10`, `line_search_fn="strong_wolfe"`

#### 특징 (v5)
- CUDA에서 전체 프로세스 수행 (inference + optimization)
- 여러 샘플(기본 1024개) 중 최적 초기값 선택으로 더 나은 결과
- 동일한 손실 함수로 학습과 최적화 일관성 유지

#### 저장 파일
- **궤적 플롯**: `NN_opt/result/opt_nn_lbfgs/cvae_lbfgs_traj_gpu_opt.png`
- **CSV 파일들**:
  - `q_traj.csv`: Joint position trajectory
  - `q_dot_traj.csv`: Joint velocity trajectory
  - `body_orientation.csv`: Body orientation (Euler angles)
  - `waypoints.csv`: 최적화된 waypoints
  - `q0_start.csv`, `q0_goal.csv`: 시작/목표 쿼터니언
  - `meta.csv`: 메타 정보 (dt, total_time)

### MLP 기반 최적화 (`NN_opt/optimization/optimize_mlp.py`)

#### 단계
1. **Warm Start (CUDA/CPU)**:
   - 학습된 MLP 모델 로드
   - 조건부로 waypoints 직접 예측 (결정론적)

2. **Refinement (CPU)**:
   - 예측된 waypoints를 CPU로 전환
   - AdamW 또는 LBFGS 최적화 (CVAE와 동일한 하이퍼파라미터)

#### 특징
- 결정론적 예측 (동일 조건에서 항상 동일한 초기값)
- 더 빠른 inference (샘플링 불필요)

#### 저장 파일
- **궤적 플롯**: `NN_opt/result/opt_nn_lbfgs/mlp_lbfgs_traj.png`
- **CSV 파일들**: CVAE와 동일한 구조 (`*_mlp.csv` 접미사)

### 직접 최적화 (`NN_opt/optimization/optimize_direct.py`)

#### 방법
- 랜덤 초기화된 waypoints에서 시작
- LBFGS 최적화 직접 수행

#### 하이퍼파라미터
- **Optimizer**: LBFGS
- **Learning Rate**: 1.0
- **Max Iterations**: 50
- **History Size**: 100
- **Line Search**: "strong_wolfe"

---

## 실행 방법

### 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 학습 실행

```bash
# CVAE 학습
python -m NN_opt.training.train_cvae

# MLP 학습
python -m NN_opt.training.train_mlp

# Joint-space CVAE 학습
python -m NN_opt.training.train_cvae_joint

# Joint-space MLP 학습
python -m NN_opt.training.train_mlp_joint
```

### 최적화 실행

```bash
# CVAE 기반 최적화
python -m NN_opt.optimization.optimize_cvae

# MLP 기반 최적화
python -m NN_opt.optimization.optimize_mlp

# 직접 최적화 (랜덤 초기화)
python -m NN_opt.optimization.optimize_direct

# iLQR/DDP 최적화
python -m ilqr.scripts.run_ddp_casadi
```

### TensorBoard 실행

```bash
# CVAE 학습 곡선 확인
tensorboard --logdir NN_opt/logs/cvae_v4

# MLP 학습 곡선 확인
tensorboard --logdir NN_opt/logs/mlp_v4
```

---

## 주요 파라미터 요약

### 학습 파라미터

| 파라미터 | CVAE | MLP |
|---------|------|-----|
| Batch Size | 1024 | 1024 |
| Epochs | 2000 | 2000 |
| Learning Rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam |
| Hidden Dim | 256 | 128 |
| Latent Dim | 8 | - |
| Condition Dim | 8 | 8 |
| Waypoints | 3 | 3 |
| Total Time | 10.0s | 10.0s |
| Time Step (dt) | 0.1s | 0.1s |

### 물리 시뮬레이션 파라미터

| 파라미터 | 값 |
|---------|-----|
| 동역학 | SPART (Space Robot Dynamics) |
| 적분 방법 | Rotation Matrix (R_{k+1} = R_k @ R_delta) |
| 손실 함수 | Chordal distance (log scale) |
| Damping | 1e-6 |
| 궤적 생성 | 4분절 half-cosine (ease-in-out) |

### 최적화 파라미터

| 파라미터 | AdamW | LBFGS |
|---------|-------|-------|
| Learning Rate | 1e-2 | 1e-3 |
| Max Iterations | - | 20 |
| History Size | - | - |
| Line Search | - | "strong_wolfe" |
| Device | GPU | GPU (v5) |
| Num Samples | - | 1024 (기본값) |

**참고 (v5)**:
- CVAE 기반 최적화는 GPU에서 전체 프로세스 수행 (CUDA)
- LBFGS 최적화의 learning rate가 1e-3으로 변경 (더 안정적)

---

## 디렉토리 구조

```
CVAE/
├── assets/                      # URDF 파일 및 메시
│   ├── meshes/                  # UR10e 로봇 메시
│   ├── a1_description/          # A1 쿼드러펫 로봇
│   └── SC_ur10e.urdf            # 우주 로봇 URDF
│
├── physics/                     # 물리 시뮬레이션 모듈
│   ├── utils.py                 # 공통 유틸리티 (quaternion, rotation 등)
│   └── dynamics/                # SPART 동역학 모듈
│       ├── spart_functions.py   # SPART 함수 (NumPy)
│       ├── spart_functions_torch.py  # SPART 함수 (PyTorch)
│       ├── spart_casadi.py      # SPART 함수 (CasADi)
│       └── urdf2robot*.py       # URDF 파서
│
├── ilqr/                        # iLQR/DDP 최적화 모듈
│   ├── scripts/
│   │   └── run_ddp_casadi.py    # DDP 실행 스크립트
│   └── src/
│       ├── ddp_casadi.py        # CasADi 기반 DDP 구현
│       └── trajectory_utils.py  # 궤적 유틸리티
│
├── NN_opt/                      # 신경망 기반 최적화 모듈
│   ├── model/                   # 신경망 모델 정의
│   │   ├── cvae.py              # CVAE 및 MLP 모델
│   │   ├── spline.py            # 스플라인 궤적 생성
│   │   └── reachability_predictor.py
│   ├── training/                # 학습 스크립트 및 유틸리티
│   │   ├── physics_layer.py     # 물리 시뮬레이션 레이어
│   │   ├── train_cvae.py        # CVAE 학습
│   │   ├── train_mlp.py         # MLP 학습
│   │   ├── train_cvae_joint.py  # Joint-space CVAE 학습
│   │   └── train_mlp_joint.py   # Joint-space MLP 학습
│   ├── optimization/            # 최적화 스크립트
│   │   ├── optimize_cvae.py     # CVAE 기반 최적화
│   │   ├── optimize_mlp.py      # MLP 기반 최적화
│   │   ├── optimize_direct.py   # 직접 최적화
│   │   └── optimize_pso.py      # PSO 최적화
│   ├── utils/                   # 유틸리티 함수
│   │   ├── data_generation.py   # 데이터 생성
│   │   ├── losses.py            # 손실 함수
│   │   └── visualization.py     # 시각화
│   ├── weight/                  # 학습된 모델 가중치
│   ├── result/                  # 최적화 결과 (CSV, 플롯)
│   ├── logs/                    # TensorBoard 로그
│   └── plots/                   # 학습 곡선 플롯
│
├── mujoco_sim.py                # MuJoCo 시뮬레이션
├── test.py                      # 테스트 스크립트
├── requirements.txt             # Python 의존성
└── README.md                    # 이 파일
```

---

## 기술적 세부사항

### Rotation Matrix 기반 동역학

기존 쿼터니언 적분 방식 대신 회전행렬을 사용하여 자세를 추적합니다:

```python
# 각 스텝마다
wb = compute_body_angular_velocity(qm, qd)  # SPART 동역학
R_delta = rot_from_omega(wb, dt)           # Rodrigues 공식
R_curr = R_curr @ R_delta                  # 회전행렬 곱셈
```

### SPART 동역학 계산

각 타임스텝에서:
1. Forward kinematics (RJ, RL, rJ, rL, e, g)
2. Differential kinematics (Bij, Bi0, P0, pm)
3. Inertia projection (I0, Im)
4. Composite body mass (M0_t, Mm_t)
5. Generalized inertia matrix (H0, H0m)
6. Non-holonomic constraint solver:
   ```
   H0 * u0 = -H0m * qd
   wb = u0[:3]  # Body angular velocity
   ```

### 손실 함수

#### Physics Loss (물리 시뮬레이션 손실)
Chordal distance 기반 손실:
```
R_diff = R_final - R_goal
R_diff_sq = R_diff^T @ R_diff
trace_val = 0.5 * trace(R_diff_sq)
physics_loss = log(ε + trace_val)
```
- `ε = 1e-8`

#### 정규화 손실 (v5: Joint Squared + Joint Change)
**1. Joint Squared Penalty** (관절 각도의 제곱 평균):
```
mean_joint_squared = mean(waypoints^2)
joint_squared_penalty = mean_joint_squared * JOINT_SQUARED_WEIGHT
```
- 모든 waypoint의 모든 관절 각도 제곱의 평균
- 작은 관절 각도를 유도

**2. Joint Change Penalty** (연속 waypoint 간 변화량):
```
joint_diff = waypoints[:, 1:, :] - waypoints[:, :-1, :]
mean_joint_change_squared = mean(joint_diff^2)
joint_change_penalty = mean_joint_change_squared * JOINT_CHANGE_WEIGHT
```
- 인접한 waypoint 간 관절 각도 차이의 제곱 평균
- 부드러운 궤적(작은 변화량) 유도

**총 손실 (v5)**:
```
total_loss = physics_loss + joint_squared_penalty + joint_change_penalty
```

**가중치**:
- `JOINT_SQUARED_WEIGHT = 0.01`
- `JOINT_CHANGE_WEIGHT = 0.01`

---

## 버전 정보

### v5 (현재 버전): Joint Change Penalty
**변경 사항**:
- **손실 함수 개선**:
  - 기존: 최대 관절 각도 패널티 (`max(|waypoints|)`)
  - 변경: 관절 제곱 평균 + 연속 waypoint 간 변화량 패널티
  - 목적: 더 부드러운 궤적 생성 및 작은 관절 각도 유도
- **최적화 프로세스**:
  - 전체 프로세스를 GPU(CUDA)에서 수행
  - LBFGS learning rate: 1e-3 (더 안정적)
  - 샘플 수: 기본 1024개
- **학습 및 최적화 간 손실 함수 일관성 유지**

**파일명**:
- 모델 가중치: `NN_opt/weight/cvae_debug/v5_joint_change.pth`
- 최적화 결과: `NN_opt/result/opt_nn_lbfgs/cvae_lbfgs_traj_gpu_opt.png`

### v4 (이전 버전)
- 최대 관절 각도 패널티 사용 (`MAX_JOINT_WEIGHT`)
- CPU에서 최종 최적화 수행
- 모델 가중치: `v4.pth`

---

## 참고사항

- 학습 및 최적화는 CUDA GPU를 사용합니다 (v5).
- Joint limits는 모델에 자동으로 적용되어 출력이 관절 한계 내에 있도록 보장됩니다.
- 검증은 10 epoch마다 수행되며, TensorBoard에 궤적 시각화가 기록됩니다.
- v5에서는 학습과 최적화에서 동일한 손실 함수를 사용하여 일관성을 보장합니다.
