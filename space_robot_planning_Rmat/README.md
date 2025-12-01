## 공간 로봇 계획 (Rotation-Matrix Variant)

이 폴더는 기존 `space_robot_planning` 프로젝트에서 **쿼터니언 미분 기반 적분** 대신  
각 타임스텝에서의 **회전행렬 곱(Rotation Matrix Multiplication)** 으로 진행하도록 분리한 변형 버전입니다.

현재는 핵심 물리 엔진(`PhysicsLayer`)만 회전행렬 기반으로 구현되어 있으며,  
인터페이스(`calculate_loss`, `generate_trajectory` 등)는 원본과 동일하게 맞춰져 있습니다.

- 원본: `space_robot_planning/src/training/physics_layer.py`
- 회전행렬 버전: `space_robot_planning_Rmat/src/training/physics_layer.py`

학습 / 최적화 스크립트에서 다음과 같이 import 경로만 바꾸면 회전행렬 버전을 사용할 수 있습니다:

```python
from space_robot_planning_Rmat.src.training.physics_layer import PhysicsLayer
```

나머지 코드 (CVAE/MLP 등)는 그대로 사용하며,  

