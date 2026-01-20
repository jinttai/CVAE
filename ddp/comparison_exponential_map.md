# Exponential Map 구현 비교: dynamics_torch.py vs physics_layer.py

## 1. `rot_from_omega` 함수 비교

### `dynamics_torch.py` - `rot_from_omega_exponential()`
```python
def rot_from_omega_exponential(wb, dt):
    wb_norm = torch.linalg.norm(wb)
    theta = wb_norm * dt
    eps = 1e-8
    I = torch.eye(3, device=device, dtype=dtype)
    
    if theta < eps:  # Python if (not vmap-safe)
        # Small angle: R ≈ I + [ω*dt]_×
        K = skew_symmetric(wb * dt)
        R_delta = I + K
    else:
        # Rodrigues' formula
        axis = wb / (wb_norm + 1e-12)
        K = skew_symmetric(axis)
        K_squared = K @ K
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        R_delta = I + sin_theta * K + (1.0 - cos_theta) * K_squared
    
    return R_delta
```

**특징:**
- Python `if` 문 사용 (단일 입력용)
- `theta` 클램핑 없음
- 작은 각도 근사와 일반 케이스를 분리

### `physics_layer.py` - `_rot_from_omega()`
```python
def _rot_from_omega(self, wb, dt):
    wb_norm = torch.linalg.norm(wb)
    max_theta = 3.141592653589793  # π
    theta = torch.clamp(wb_norm * dt, max=max_theta)  # ✅ Clamp 적용
    eps = 1e-8
    
    axis = wb / (wb_norm + 1e-12)
    K = self._skew(axis)
    I = self.eye3.to(dtype=wb.dtype)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    K_squared = K @ K
    R_big = I + sin_theta * K + (1.0 - cos_theta) * K_squared
    
    wb_dt = wb * dt
    K_small = self._skew(wb_dt)
    R_small = I + K_small
    
    small = theta < eps
    # ✅ torch.where 사용 (vmap-safe)
    R_delta = torch.where(small, R_small, R_big)
    return R_delta
```

**특징:**
- `torch.where` 사용 (vmap 호환)
- `theta`를 π로 클램핑 (큰 각속도 보호)
- 모든 케이스를 먼저 계산 후 선택
- 메모리 최적화 (`eye3` 미리 할당)

## 2. `rot_to_quat` 함수 비교

### `dynamics_torch.py` - `rot_to_quat()`
```python
def rot_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:  # Python if
        # Case 1
        ...
    else:
        if R[0, 0] > R[1, 1] and ...:  # Python if
            # Case 2
        elif ...:  # Python if
            # Case 3
        else:
            # Case 4
    
    q = torch.stack([x, y, z, w])
    return normalize_quat(q)
```

**특징:**
- Python `if-elif-else` 사용
- 단일 입력용 (배치 불가)

### `physics_layer.py` - `_rot_to_quat()`
```python
def _rot_to_quat(self, R):
    trace = r00 + r11 + r22
    
    # 모든 케이스를 먼저 계산
    q1 = ...  # Case 1
    q2 = ...  # Case 2
    q3 = ...  # Case 3
    q4 = ...  # Case 4
    
    # torch.where로 선택 (vmap-safe)
    cond1 = trace > 0
    cond2 = (r00 > r11) & (r00 > r22)
    cond3 = (r11 > r22)
    q_out = torch.where(c1, q1, torch.where(c2, q2, torch.where(c3, q3, q4)))
    
    return q_out / (torch.linalg.norm(q_out, dim=-1, keepdim=True) + 1e-8)
```

**특징:**
- `torch.where` 중첩 사용 (vmap 호환)
- 배치 입력 지원 (`[..., 3, 3]` → `[..., 4]`)
- 모든 케이스를 사전 계산 후 선택

## 3. 주요 차이점 요약

| 항목 | dynamics_torch.py | physics_layer.py |
|------|------------------|------------------|
| **분기 방법** | Python `if` | `torch.where` (텐서 기반) |
| **Vmap 호환성** | ❌ No | ✅ Yes |
| **배치 처리** | ❌ No | ✅ Yes |
| **Theta 클램핑** | ❌ No | ✅ Yes (max=π) |
| **메모리 최적화** | ⚠️ 기본 | ✅ 최적화됨 |
| **사용 목적** | 단일 스텝 | 벡터화된 시뮬레이션 |

## 4. 권장 사항

### `dynamics_torch.py` 개선 방향:
1. **Theta 클램핑 추가**: 큰 각속도로 인한 수치 불안정성 방지
2. **`torch.where` 사용** (선택사항): vmap 호환성이 필요한 경우

### 현재 `dynamics_torch.py`의 장점:
- 구현이 단순하고 읽기 쉬움
- 단일 스텝 시뮬레이션에는 충분
- DDP에서 gradient 계산 시 문제 없음

### 현재 `physics_layer.py`의 장점:
- 벡터화된 배치 처리 가능
- vmap과 호환
- 큰 각속도에 대해 더 안정적

