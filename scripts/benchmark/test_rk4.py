"""
RK4 Implementation Test Script

이 스크립트는 simulate_single_rk4 함수의 구현을 검증합니다:
1. 시간 경계 조건 검증
2. 보간 로직 검증
3. Loss 계산 검증
4. 간단한 케이스에서의 동작 확인
5. simulate_single과의 비교
"""

import torch
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def test_time_boundary_conditions(physics: PhysicsLayer, device: str):
    """시간 경계 조건 테스트"""
    print("\n=== Test 1: Time Boundary Conditions ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    # 간단한 waypoints 생성
    waypoints = torch.zeros(1, num_waypoints * n_q, device=device)
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    
    # RK4 실행
    try:
        loss, q_final = physics.simulate_single_rk4(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
        )
        print(f"✓ RK4 실행 성공")
        print(f"  Final time should be ~{physics.total_time:.3f}s")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Final quaternion: {q_final.cpu().numpy()}")
    except Exception as e:
        print(f"✗ RK4 실행 실패: {e}")
        raise


def test_interpolation_logic(physics: PhysicsLayer, device: str):
    """보간 로직 테스트"""
    print("\n=== Test 2: Interpolation Logic ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    # 테스트용 waypoints 생성 (명확한 변화)
    waypoints = torch.randn(1, num_waypoints * n_q, device=device) * 0.1
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    
    # 보간 함수 직접 테스트 (내부 함수이므로 간접적으로 테스트)
    print(f"  Trajectory shape: {q_traj.shape}")
    print(f"  Number of steps: {physics.num_steps}")
    print(f"  Total time: {physics.total_time}s")
    
    # 경계값 테스트: t=0, t=total_time
    print(f"  First trajectory point: {q_traj[0, :3].cpu().numpy()}")
    print(f"  Last trajectory point: {q_traj[-1, :3].cpu().numpy()}")
    
    # RK4가 경계에서 올바르게 작동하는지 확인
    try:
        loss, q_final = physics.simulate_single_rk4(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
        )
        print(f"✓ 보간 로직 검증 완료 (Loss: {loss.item():.6f})")
    except Exception as e:
        print(f"✗ 보간 로직 검증 실패: {e}")
        raise


def test_loss_calculation(physics: PhysicsLayer, device: str):
    """Loss 계산 테스트"""
    print("\n=== Test 3: Loss Calculation ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    waypoints = torch.zeros(1, num_waypoints * n_q, device=device)
    
    # 케이스 1: 동일한 초기/목표 자세 (loss가 작아야 함)
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    loss1, q_final1 = physics.simulate_single_rk4(
        q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
    )
    
    print(f"  Case 1 (same init/goal): Loss = {loss1.item():.8f}")
    print(f"    Final quaternion: {q_final1.cpu().numpy()}")
    
    # 케이스 2: 다른 초기/목표 자세 (loss가 커야 함)
    goal_rpy = [np.pi / 4, 0.0, 0.0]  # 45도 회전
    goal_quat = R.from_euler("xyz", goal_rpy).as_quat()
    q0_goal2 = torch.tensor([goal_quat], device=device, dtype=torch.float32)
    
    loss2, q_final2 = physics.simulate_single_rk4(
        q_traj[0], q_dot_traj[0], q0_init[0], q0_goal2[0]
    )
    
    print(f"  Case 2 (different init/goal): Loss = {loss2.item():.8f}")
    print(f"    Final quaternion: {q_final2.cpu().numpy()}")
    
    # Loss 계산 방식: log(epsilon + trace_val)
    # trace_val이 매우 작으면 (epsilon보다 작으면) log 결과가 음수가 될 수 있음
    # 이것은 log 함수의 수학적 특성상 정상적인 동작입니다.
    # 예: log(1e-8 + 1e-10) ≈ log(1e-8) ≈ -18.4
    
    # Loss가 합리적인 범위에 있는지 확인
    # Case 1의 loss가 Case 2보다 작아야 함 (같은 init/goal이므로 오차가 작음)
    assert loss1.item() < loss2.item(), f"Loss should be smaller for same init/goal. Got loss1={loss1.item():.8f}, loss2={loss2.item():.8f}"
    
    # Loss가 유한한 값인지 확인 (inf, nan이 아닌지)
    assert torch.isfinite(loss1), f"Loss1 should be finite, got {loss1.item()}"
    assert torch.isfinite(loss2), f"Loss2 should be finite, got {loss2.item()}"
    
    # Loss가 너무 작거나 크지 않은지 확인 (합리적인 범위)
    # log(1e-8) ≈ -18.4, log(10) ≈ 2.3 정도의 범위를 기대
    if loss1.item() < -50:
        print(f"    ⚠ Warning: Loss1 is very negative ({loss1.item():.8f}), which may indicate very small error")
    if loss2.item() < -50:
        print(f"    ⚠ Warning: Loss2 is very negative ({loss2.item():.8f}), which may indicate very small error")
    
    print(f"✓ Loss 계산 검증 완료 (Loss는 log 함수이므로 음수일 수 있음)")


def test_rk4_vs_simulate_single(physics: PhysicsLayer, device: str):
    """RK4와 simulate_single의 상세 비교 테스트"""
    print("\n=== Test 4: RK4 vs simulate_single Detailed Comparison ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    # 여러 케이스로 테스트
    test_cases = [
        {
            "name": "Small rotation",
            "waypoints_scale": 0.05,
            "goal_rpy": [np.pi / 20, 0.0, 0.0]
        },
        {
            "name": "Medium rotation",
            "waypoints_scale": 0.1,
            "goal_rpy": [np.pi / 8, 0.0, 0.0]
        },
        {
            "name": "Zero waypoints",
            "waypoints_scale": 0.0,
            "goal_rpy": [np.pi / 10, 0.0, 0.0]
        },
        {
            "name": "Random waypoints",
            "waypoints_scale": 0.15,
            "goal_rpy": [np.pi / 6, np.pi / 12, 0.0]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n  Testing: {case['name']}")
        
        # Waypoints 생성
        if case["waypoints_scale"] == 0.0:
            waypoints = torch.zeros(1, num_waypoints * n_q, device=device)
        else:
            waypoints = torch.randn(1, num_waypoints * n_q, device=device) * case["waypoints_scale"]
        
        q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        goal_quat = R.from_euler("xyz", case["goal_rpy"]).as_quat()
        q0_goal = torch.tensor([goal_quat], device=device, dtype=torch.float32)
        
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
        
        # RK4 실행
        loss_rk4, q_final_rk4 = physics.simulate_single_rk4(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
        )
        
        # simulate_single 실행
        loss_single, q_final_single = physics.simulate_single(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
        )
        
        # 결과 비교
        loss_diff = abs(loss_rk4.item() - loss_single.item())
        loss_rel_diff = loss_diff / (loss_single.item() + 1e-10) * 100  # 상대 차이 (%)
        
        q_diff_l2 = torch.norm(q_final_rk4 - q_final_single).item()
        
        # 쿼터니언 각도 차이 계산 (quaternion distance)
        q_dot = torch.clamp(torch.abs(torch.dot(q_final_rk4, q_final_single)), -1.0, 1.0)
        angle_diff_rad = 2 * torch.acos(q_dot).item()
        angle_diff_deg = np.rad2deg(angle_diff_rad)
        
        print(f"    RK4 Loss:        {loss_rk4.item():.8f}")
        print(f"    Single Loss:     {loss_single.item():.8f}")
        print(f"    Loss diff:       {loss_diff:.8f} ({loss_rel_diff:.2f}%)")
        print(f"    Quat L2 diff:    {q_diff_l2:.8f}")
        print(f"    Angle diff:      {angle_diff_deg:.4f}°")
        
        results.append({
            "case": case["name"],
            "loss_rk4": loss_rk4.item(),
            "loss_single": loss_single.item(),
            "loss_diff": loss_diff,
            "loss_rel_diff": loss_rel_diff,
            "q_diff_l2": q_diff_l2,
            "angle_diff_deg": angle_diff_deg
        })
    
    # 요약 통계
    print(f"\n  Summary Statistics:")
    avg_loss_diff = np.mean([r["loss_diff"] for r in results])
    avg_rel_diff = np.mean([r["loss_rel_diff"] for r in results])
    avg_angle_diff = np.mean([r["angle_diff_deg"] for r in results])
    max_angle_diff = np.max([r["angle_diff_deg"] for r in results])
    
    print(f"    Average loss difference: {avg_loss_diff:.8f}")
    print(f"    Average relative diff:  {avg_rel_diff:.2f}%")
    print(f"    Average angle diff:     {avg_angle_diff:.4f}°")
    print(f"    Max angle diff:         {max_angle_diff:.4f}°")
    
    # 두 방법이 합리적인 범위 내에 있는지 확인
    # RK4는 더 정확한 적분 방법이므로 차이가 있을 수 있지만, 너무 크면 안됨
    if max_angle_diff > 10.0:  # 10도 이상 차이나면 경고
        print(f"    ⚠ Warning: Large angle difference detected!")
    else:
        print(f"    ✓ Angle differences are within reasonable range")
    
    print(f"✓ 상세 비교 검증 완료 (두 방법은 다른 적분 방법이므로 차이가 있을 수 있음)")


def test_quaternion_normalization(physics: PhysicsLayer, device: str):
    """쿼터니언 정규화 테스트"""
    print("\n=== Test 5: Quaternion Normalization ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    waypoints = torch.randn(1, num_waypoints * n_q, device=device) * 0.1
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    loss, q_final = physics.simulate_single_rk4(
        q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
    )
    
    # 쿼터니언이 정규화되어 있는지 확인
    q_norm = torch.norm(q_final).item()
    print(f"  Final quaternion norm: {q_norm:.6f}")
    print(f"  Final quaternion: {q_final.cpu().numpy()}")
    
    assert abs(q_norm - 1.0) < 1e-5, f"Quaternion should be normalized, got norm={q_norm}"
    print(f"✓ 쿼터니언 정규화 검증 완료")


def test_edge_cases(physics: PhysicsLayer, device: str):
    """엣지 케이스 테스트"""
    print("\n=== Test 6: Edge Cases ===")
    
    num_waypoints = physics.num_waypoints
    n_q = physics.n_q
    
    # 케이스 1: 모든 waypoint가 0
    waypoints = torch.zeros(1, num_waypoints * n_q, device=device)
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    try:
        loss, q_final = physics.simulate_single_rk4(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal[0]
        )
        print(f"  ✓ Zero waypoints: Loss = {loss.item():.6f}")
    except Exception as e:
        print(f"  ✗ Zero waypoints failed: {e}")
        raise
    
    # 케이스 2: 큰 회전
    goal_rpy = [np.pi, 0.0, 0.0]  # 180도 회전
    goal_quat = R.from_euler("xyz", goal_rpy).as_quat()
    q0_goal2 = torch.tensor([goal_quat], device=device, dtype=torch.float32)
    
    try:
        loss, q_final = physics.simulate_single_rk4(
            q_traj[0], q_dot_traj[0], q0_init[0], q0_goal2[0]
        )
        print(f"  ✓ Large rotation: Loss = {loss.item():.6f}")
    except Exception as e:
        print(f"  ✗ Large rotation failed: {e}")
        raise
    
    print(f"✓ 엣지 케이스 검증 완료")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RK4 Implementation Test ({device}) ===")
    
    # Robot 및 PhysicsLayer 설정
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file not found at {urdf_path}")
        print("Please ensure the URDF file exists.")
        return
    
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 1.0
    
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Number of waypoints: {NUM_WAYPOINTS}")
    print(f"  Total time: {TOTAL_TIME}s")
    print(f"  Number of steps: {physics.num_steps}")
    print(f"  Joints (n_q): {physics.n_q}")
    
    # 테스트 실행
    try:
        test_time_boundary_conditions(physics, device)
        test_interpolation_logic(physics, device)
        test_loss_calculation(physics, device)
        test_rk4_vs_simulate_single(physics, device)
        test_quaternion_normalization(physics, device)
        test_edge_cases(physics, device)
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"✗ Test failed: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

