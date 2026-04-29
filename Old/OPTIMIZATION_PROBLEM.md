# 최적화 문제 정의 (Optimization Problem Formulation)

## 문제 개요

우주 로봇의 자세 제어를 위한 궤적 계획 최적화 문제입니다. 주어진 시작 자세에서 목표 자세로 이동하는 관절 궤적을 생성하며, 물리 동역학 제약 조건을 만족시키면서 목표 자세 오차를 최소화합니다.

---

## 수학적 정의

### 결정 변수 (Decision Variables)

관절 궤적을 정의하는 waypoint들:

\[
\mathbf{W} = \begin{bmatrix} \mathbf{w}_1 \\ \mathbf{w}_2 \\ \mathbf{w}_3 \end{bmatrix} \in \mathbb{R}^{N_w \times n_q}
\]

또는 flattened 형태:

\[
\mathbf{w} = \text{vec}(\mathbf{W}) \in \mathbb{R}^{N_w \cdot n_q}
\]

여기서:
- \(N_w = 3\): 중간 waypoint 개수
- \(n_q = 6\): 관절 개수 (UR10e 로봇)
- \(\mathbf{w}_i \in \mathbb{R}^{n_q}\): \(i\)번째 waypoint의 관절 각도 벡터

---

## 목적 함수 (Objective Function)

총 손실 함수는 물리 시뮬레이션 손실과 정규화 항들의 합입니다:

\[
\mathcal{L}_{\text{total}}(\mathbf{W}) = \mathcal{L}_{\text{physics}}(\mathbf{W}) + \mathcal{L}_{\text{joint}^2}(\mathbf{W}) + \mathcal{L}_{\text{change}}(\mathbf{W}) + \mathcal{L}_{\text{max}}(\mathbf{W})
\]

### 1. 물리 시뮬레이션 손실 (Physics Loss)

최종 자세와 목표 자세 간의 chordal distance (log scale):

\[
\mathcal{L}_{\text{physics}}(\mathbf{W}) = \log\left(\varepsilon + \frac{1}{2} \text{tr}\left((\mathbf{R}_{\text{final}} - \mathbf{R}_{\text{goal}})^T (\mathbf{R}_{\text{final}} - \mathbf{R}_{\text{goal}})\right)\right)
\]

여기서:
- \(\mathbf{R}_{\text{final}} \in \mathbb{SO}(3)\): 시뮬레이션 종료 시점의 회전 행렬
- \(\mathbf{R}_{\text{goal}} \in \mathbb{SO}(3)\): 목표 회전 행렬
- \(\varepsilon = 10^{-8}\): 수치 안정성을 위한 작은 상수

**시뮬레이션 과정**:
1. Waypoint로부터 quintic polynomial interpolation으로 궤적 생성: \(\mathbf{q}(t), \dot{\mathbf{q}}(t)\)
2. 각 시간 스텝 \(t_k\)에서 SPART 동역학 계산:
   - Forward kinematics: \(\mathbf{R}_J, \mathbf{r}_J, \mathbf{R}_L, \mathbf{r}_L\)
   - Generalized inertia matrix: \(\mathbf{H}_0, \mathbf{H}_{0m}\)
3. Non-holonomic constraint solver:
   \[
   \mathbf{H}_0 \mathbf{u}_0 + \mathbf{H}_{0m} \dot{\mathbf{q}} = \mathbf{0}
   \]
   해: \(\mathbf{u}_0 = -\mathbf{H}_0^{-1} \mathbf{H}_{0m} \dot{\mathbf{q}}\)
4. 각속도 추출: \(\boldsymbol{\omega}_b = \mathbf{u}_0[:3]\)
5. 회전 행렬 업데이트:
   \[
   \mathbf{R}_{k+1} = \mathbf{R}_k \cdot \mathbf{R}_{\Delta}(\boldsymbol{\omega}_b, \Delta t)
   \]
   여기서 \(\mathbf{R}_{\Delta}\)는 Rodrigues 공식으로 계산:
   \[
   \mathbf{R}_{\Delta}(\boldsymbol{\omega}, \Delta t) = \mathbf{I} + \sin(\theta) [\mathbf{a}]_\times + (1-\cos(\theta)) [\mathbf{a}]_\times^2
   \]
   \(\theta = \|\boldsymbol{\omega}\| \Delta t\), \(\mathbf{a} = \boldsymbol{\omega} / \|\boldsymbol{\omega}\|\)

### 2. 관절 제곱 평균 패널티 (Joint Squared Penalty)

작은 관절 각도를 유도:

\[
\mathcal{L}_{\text{joint}^2}(\mathbf{W}) = \lambda_1 \cdot \frac{1}{N_w \cdot n_q} \sum_{i=1}^{N_w} \sum_{j=1}^{n_q} w_{ij}^2 = \lambda_1 \cdot \text{mean}(\mathbf{W}^2)
\]

여기서 \(\lambda_1 = 0.01\) (가중치)

### 3. 관절 변화량 패널티 (Joint Change Penalty)

연속된 waypoint 간 부드러운 변화를 유도:

\[
\mathcal{L}_{\text{change}}(\mathbf{W}) = \lambda_2 \cdot \frac{1}{(N_w-1) \cdot n_q} \sum_{i=1}^{N_w-1} \sum_{j=1}^{n_q} (w_{i+1,j} - w_{i,j})^2 = \lambda_2 \cdot \text{mean}((\mathbf{W}[1:] - \mathbf{W}[:-1])^2)
\]

여기서 \(\lambda_2 = 0.01\) (가중치)

### 4. 최대 관절 각도 패널티 (Max Joint Penalty)

관절 각도의 최대값을 제한:

\[
\mathcal{L}_{\text{max}}(\mathbf{W}) = \lambda_3 \cdot \max_{i,j} |w_{ij}| = \lambda_3 \cdot \|\mathbf{W}\|_\infty
\]

여기서 \(\lambda_3 = 0.1\) (가중치)

---

## 제약 조건 (Constraints)

### 1. 물리 동역학 제약 (Physics Dynamics Constraint)

SPART (Space Robot Dynamics) 기반 동역학을 만족해야 합니다:

**Forward Kinematics**:
\[
\mathbf{R}_J, \mathbf{r}_J, \mathbf{R}_L, \mathbf{r}_L, \mathbf{e}, \mathbf{g} = \text{kinematics}(\mathbf{R}_0, \mathbf{r}_0, \mathbf{q}, \text{robot})
\]

**Differential Kinematics**:
\[
\mathbf{B}_{ij}, \mathbf{B}_{i0}, \mathbf{P}_0, \mathbf{p}_m = \text{diff\_kinematics}(\mathbf{R}_0, \mathbf{r}_0, \mathbf{r}_L, \mathbf{e}, \mathbf{g}, \text{robot})
\]

**Inertia Projection**:
\[
\mathbf{I}_0, \mathbf{I}_m = \text{inertia\_projection}(\mathbf{R}_0, \mathbf{R}_L, \text{robot})
\]

**Generalized Inertia Matrix**:
\[
\mathbf{H}_0, \mathbf{H}_{0m}, \mathbf{H}_m = \text{generalized\_inertia}(\mathbf{M}_0, \mathbf{M}_m, \mathbf{B}_{ij}, \mathbf{B}_{i0}, \mathbf{P}_0, \mathbf{p}_m, \text{robot})
\]

### 2. Non-holonomic Constraint (모멘텀 보존)

우주 로봇의 모멘텀 보존 제약:

\[
\mathbf{H}_0 \mathbf{u}_0 + \mathbf{H}_{0m} \dot{\mathbf{q}} = \mathbf{0}
\]

이 제약은 각 시간 스텝에서 자동으로 만족되도록 해를 구합니다:
\[
\mathbf{u}_0 = -(\mathbf{H}_0 + \varepsilon_d \mathbf{I}_6)^{-1} \mathbf{H}_{0m} \dot{\mathbf{q}}
\]
여기서 \(\varepsilon_d = 10^{-6}\)는 damping term입니다.

### 3. 관절 한계 제약 (Joint Limits)

각 관절은 물리적 한계 내에 있어야 합니다:

\[
\mathbf{q}_{\min} \leq \mathbf{w}_i \leq \mathbf{q}_{\max}, \quad \forall i \in \{1, \ldots, N_w\}
\]

또는 flattened 형태:
\[
\mathbf{q}_{\min} \leq \mathbf{w} \leq \mathbf{q}_{\max}
\]

**참고**: CVAE 모델의 출력 레이어에서 자동으로 joint limits를 적용합니다 (tanh + scaling).

### 4. 궤적 생성 제약 (Trajectory Generation Constraint)

Waypoint로부터 시간 연속 궤적은 quintic polynomial interpolation으로 생성됩니다:

각 분절 \(s \in \{1, \ldots, N_w+1\}\)에 대해:

**위치**:
\[
q_s(t) = q_{\text{start}}^{(s)} + (q_{\text{end}}^{(s)} - q_{\text{start}}^{(s)}) \cdot b(t)
\]

**속도**:
\[
\dot{q}_s(t) = (q_{\text{end}}^{(s)} - q_{\text{start}}^{(s)}) \cdot b'(t) / T_s
\]

여기서:
- \(b(t) = 6t^5 - 15t^4 + 10t^3\): quintic basis 함수 (\(t \in [0,1]\))
- \(b'(t) = 30t^4 - 60t^3 + 30t^2\): 1차 미분
- \(T_s = T_{\text{total}} / (N_w + 1)\): 분절 시간
- \(q_{\text{start}}^{(s)}, q_{\text{end}}^{(s)}\): 분절의 시작/끝 waypoint

**경계 조건**:
- 각 waypoint에서 속도와 가속도가 0: \(b'(0) = b'(1) = 0\), \(b''(0) = b''(1) = 0\)

---

## 최적화 문제 (Optimization Problem)

### 일반 형태

\[
\begin{aligned}
\min_{\mathbf{W}} \quad & \mathcal{L}_{\text{total}}(\mathbf{W}) \\
\text{s.t.} \quad & \mathbf{q}_{\min} \leq \mathbf{w}_i \leq \mathbf{q}_{\max}, \quad \forall i \\
& \text{Physics dynamics satisfied} \\
& \text{Non-holonomic constraint satisfied}
\end{aligned}
\]

### 상세 형태

\[
\begin{aligned}
\min_{\mathbf{W} \in \mathbb{R}^{N_w \times n_q}} \quad & \log\left(\varepsilon + \frac{1}{2} \text{tr}\left((\mathbf{R}_{\text{final}}(\mathbf{W}) - \mathbf{R}_{\text{goal}})^T (\mathbf{R}_{\text{final}}(\mathbf{W}) - \mathbf{R}_{\text{goal}})\right)\right) \\
& \quad + \lambda_1 \cdot \text{mean}(\mathbf{W}^2) \\
& \quad + \lambda_2 \cdot \text{mean}((\mathbf{W}[1:] - \mathbf{W}[:-1])^2) \\
& \quad + \lambda_3 \cdot \|\mathbf{W}\|_\infty \\
\text{s.t.} \quad & \mathbf{q}_{\min} \leq \mathbf{w}_i \leq \mathbf{q}_{\max}, \quad \forall i \in \{1, \ldots, N_w\} \\
& \mathbf{R}_{\text{final}}(\mathbf{W}) = \text{simulate}(\mathbf{W}, \mathbf{q}_0^{\text{init}}, T_{\text{total}})
\end{aligned}
\]

여기서:
- \(\mathbf{R}_{\text{final}}(\mathbf{W})\): waypoint \(\mathbf{W}\)로부터 생성된 궤적을 시뮬레이션한 최종 회전 행렬
- \(\mathbf{q}_0^{\text{init}}\): 초기 자세 (쿼터니언)
- \(T_{\text{total}} = 10.0\): 총 시뮬레이션 시간 (초)
- \(\Delta t = 0.1\): 시간 스텝 (초)
- \(N_{\text{steps}} = T_{\text{total}} / \Delta t = 100\): 시뮬레이션 스텝 수

---

## 해법 (Solution Methods)

### 1. CVAE 기반 학습 (Training)

**목적**: 다양한 조건에 대해 좋은 초기 추정치를 제공하는 생성 모델 학습

**손실 함수**: 위의 \(\mathcal{L}_{\text{total}}\) 사용

**학습 과정**:
1. 랜덤 목표 자세 생성: \(\mathbf{q}_0^{\text{goal}} \sim \text{Uniform}(\text{SO}(3), \theta \in [0, 60°])\)
2. CVAE로 waypoint 샘플링: \(\mathbf{W} = \text{Decoder}(\mathbf{c}, \mathbf{z})\), \(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\)
3. Physics simulation으로 손실 계산
4. 역전파 및 가중치 업데이트

### 2. LBFGS 최적화 (Refinement)

**목적**: CVAE 초기값을 기반으로 로컬 최적화 수행

**방법**: Limited-memory BFGS (Quasi-Newton method)

**초기값**: CVAE 샘플 중 최소 손실을 가진 waypoint

**최적화 과정**:
1. CVAE로 \(N_{\text{samples}} = 1024\)개 샘플 생성
2. 각 샘플의 손실 평가
3. 최소 손실 샘플 선택
4. LBFGS로 로컬 최적화 (선택적)

---

## 파라미터 요약

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| \(N_w\) | 3 | 중간 waypoint 개수 |
| \(n_q\) | 6 | 관절 개수 |
| \(T_{\text{total}}\) | 10.0 s | 총 시뮬레이션 시간 |
| \(\Delta t\) | 0.1 s | 시간 스텝 |
| \(N_{\text{steps}}\) | 100 | 시뮬레이션 스텝 수 |
| \(\lambda_1\) | 0.01 | Joint squared weight |
| \(\lambda_2\) | 0.01 | Joint change weight |
| \(\lambda_3\) | 0.1 | Max joint weight |
| \(\varepsilon\) | \(10^{-8}\) | Physics loss epsilon |
| \(\varepsilon_d\) | \(10^{-6}\) | Damping term |

---

## 참고사항

1. **회전 행렬 vs 쿼터니언**: 내부적으로는 회전 행렬을 사용하지만, 입출력 인터페이스는 쿼터니언을 사용합니다.

2. **Vectorization**: `vmap`을 사용하여 배치 차원과 시간 스텝 차원을 병렬 처리합니다.

3. **Damping**: Non-holonomic constraint solver에서 수치 안정성을 위해 작은 damping term을 추가합니다.

4. **Joint Limits**: 모델 출력 레이어에서 자동으로 적용되므로, 최적화 문제의 명시적 제약으로 포함하지 않을 수 있습니다.

