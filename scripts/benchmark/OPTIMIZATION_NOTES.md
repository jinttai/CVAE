# Performance Optimization Notes for simulate_single

## 현재 적용된 최적화

### 1. Pre-allocated Constant Tensors
- `R0`, `r0`, `eye3`, `eye6` 등 상수 텐서를 미리 할당
- `_damping_term = 1e-6 * eye6`를 미리 계산하여 매 step마다 재계산 방지

### 2. 메모리 할당 최소화
- 매 step마다 생성되던 `1e-6 * eye6` 계산을 사전 계산된 `_damping_term`으로 대체

## 추가 최적화 가능한 부분

### 1. SPART 함수 내부 최적화 (고려사항)
- SPART 함수들이 매번 많은 텐서를 할당함 (130개 이상의 torch.zeros/stack/cat 호출)
- 이는 `src/dynamics/spart_functions_torch.py` 내부에서 발생
- 직접 수정이 어려울 수 있지만, 주요 함수들에 버퍼를 전달하는 방식으로 개선 가능

### 2. GPU 동기화 최소화
- CUDA 사용 시 불필요한 CPU-GPU 동기화 제거
- `torch.cuda.synchronize()` 호출 최소화

### 3. JIT Compilation
- `torch.jit.script` 또는 `torch.compile` 사용 고려
- 특히 `simulate_single` 함수를 JIT 컴파일하면 성능 향상 가능

### 4. Batch Processing 최적화
- `vmap` 사용 시 배치 크기에 따른 최적화
- 메모리 사용량과 성능의 트레이드오프 고려

### 5. 수치 연산 최적화
- `_rot_from_omega`에서 `K @ K` 계산을 재사용
- 불필요한 중간 텐서 생성 최소화

## 벤치마크 방법

`test_time.py`를 실행하여 최적화 전후 성능 비교:
```bash
python scripts/benchmark/test_time.py
```

## 예상 성능 향상

- Pre-allocated damping term: ~1-2% 향상
- 전체적인 최적화 적용 시: ~5-10% 향상 가능
- JIT compilation 적용 시: ~20-30% 향상 가능 (첫 실행 제외)

