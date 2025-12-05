import torch
import numpy as np
import time
import math
from torch.profiler import profile, record_function, ProfilerActivity

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.training.physics_layer import PhysicsLayer   # Rmat 버전
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    # Half angles
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    # Quaternion components
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_goal(max_angle_deg: float = 40.0, device: str = "cpu") -> torch.Tensor:
    """
    Generate random quaternion goal from Euler angles within specified range
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    roll = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    
    # Convert to quaternion
    roll_t = torch.tensor([roll], device=device)
    pitch_t = torch.tensor([pitch], device=device)
    yaw_t = torch.tensor([yaw], device=device)
    
    q0_goal = euler_to_quaternion(roll_t, pitch_t, yaw_t)
    # euler_to_quaternion already returns [1, 4] shape, so no need for unsqueeze
    return q0_goal  # [1, 4]


def test_loss_calculation_time(physics: PhysicsLayer, q0_start: torch.Tensor, q0_goal: torch.Tensor, 
                                num_trials: int = 100, use_profiler: bool = False) -> dict:
    """
    Test loss calculation time with random waypoints
    
    Args:
        physics: PhysicsLayer instance
        q0_start: Initial quaternion [1, 4]
        q0_goal: Goal quaternion [1, 4]
        num_trials: Number of trials to run
        use_profiler: Whether to use PyTorch profiler
    
    Returns:
        Dictionary with timing statistics
    """
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    device = physics.device
    
    times = []
    losses = []
    
    print(f"Running {num_trials} trials...")
    
    # Setup profiler activities based on device
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    # -------------------------------------------------------------------------
    # Warm up (Critical for torch.compile)
    # 컴파일된 함수는 첫 실행 때 컴파일 비용이 발생하므로 Warm-up이 필수입니다.
    # -------------------------------------------------------------------------
    print("Warming up (compilation happens here)...")
    for i in range(5):
        waypoints = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
        _ = physics.calculate_loss(waypoints, q0_start, q0_goal)
        # print(f"Warm-up {i+1}/5 done") # 컴파일 진행 상황 확인용
    
    if use_profiler:
        print("Running with profiler (this may take longer)...")
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("loss_calculation_batch"):
                for i in range(num_trials):
                    # Generate random waypoints
                    waypoints = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
                    
                    with record_function(f"trial_{i}"):
                        # Measure loss calculation time
                        # 비동기 실행인 GPU 시간을 정확히 측정하기 위해 synchronize 추가
                        if device == "cuda": torch.cuda.synchronize()
                        start_time = time.time()
                        
                        loss = physics.calculate_loss(waypoints, q0_start, q0_goal)
                        
                        if device == "cuda": torch.cuda.synchronize()
                        end_time = time.time()
                        
                        calc_time = end_time - start_time
                        times.append(calc_time)
                        losses.append(loss.item())
                        
                        if (i + 1) % 10 == 0:
                            print(f"  Trial {i+1}/{num_trials}: Time = {calc_time:.6f}s, Loss = {loss.item():.6f}")
        
        # Export trace
        trace_file = "trace.json"
        prof.export_chrome_trace(trace_file)
        print(f"\nProfiler trace saved to {trace_file}")
        
        # Print summary
        print("\nProfiler Summary:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total", row_limit=20))
    else:
        # Regular timing without profiler
        for i in range(num_trials):
            # Generate random waypoints
            waypoints = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
            
            # Measure loss calculation time
            # 정확한 측정을 위한 synchronize 추가
            if device == "cuda": torch.cuda.synchronize()
            start_time = time.time()
            
            loss = physics.calculate_loss(waypoints, q0_start, q0_goal)
            
            if device == "cuda": torch.cuda.synchronize()
            end_time = time.time()
            
            calc_time = end_time - start_time
            times.append(calc_time)
            losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                print(f"  Trial {i+1}/{num_trials}: Time = {calc_time:.6f}s, Loss = {loss.item():.6f}")
    
    # Calculate statistics
    times_array = np.array(times)
    losses_array = np.array(losses)
    
    stats = {
        "num_trials": num_trials,
        "times": times,
        "losses": losses,
        "mean_time": float(np.mean(times_array)),
        "std_time": float(np.std(times_array)),
        "min_time": float(np.min(times_array)),
        "max_time": float(np.max(times_array)),
        "mean_loss": float(np.mean(losses_array)),
        "std_loss": float(np.std(losses_array)),
    }
    
    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"=== Loss Calculation Time Test on {device} ===")
    
    # [Optimization 1] MatMul Precision 설정 (GPU 사용 시 속도 향상)
    if device == "cuda":
        torch.set_float32_matmul_precision('high')
        print("MatMul Precision set to 'high'")
    print()
    
    # Setup (done once, not included in timing)
    print("Setting up physics layer...")
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # =========================================================================
    # [Optimization 2] torch.compile 적용 (핵심)
    # calculate_loss 메서드를 컴파일하여 Python Loop Overhead를 제거합니다.
    # mode="reduce-overhead"는 작은 연산이 많은 현재 상황에 최적입니다.
    # =========================================================================
    if hasattr(torch, "compile"):
        print("\n[INFO] Applying torch.compile(mode='reduce-overhead')...")
        physics.calculate_loss = torch.compile(physics.calculate_loss, mode="reduce-overhead")
    else:
        print("\n[WARNING] torch.compile not available. Running in Eager mode.")

    
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    # Generate random goal (40 deg limit)
    print("Generating random goal...")
    q0_goal = generate_random_goal(max_angle_deg=40.0, device=device)
    print(f"Goal quaternion: {q0_goal}")
    print()
    
    # Test parameters
    num_trials = 20
    use_profiler = False  # Set to True to generate trace.json
    
    # Run test
    print("="*60)
    stats = test_loss_calculation_time(physics, q0_start, q0_goal, num_trials, use_profiler=use_profiler)
    print("="*60)
    
    # Print results
    print("\nResults:")
    print(f"  Number of trials: {stats['num_trials']}")
    print(f"\n  Time Statistics:")
    print(f"    Mean: {stats['mean_time']:.6f} s")
    print(f"    Std:  {stats['std_time']:.6f} s")
    print(f"    Min:  {stats['min_time']:.6f} s")
    print(f"    Max:  {stats['max_time']:.6f} s")
    print(f"\n  Loss Statistics:")
    print(f"    Mean: {stats['mean_loss']:.6f}")
    print(f"    Std:  {stats['std_loss']:.6f}")



if __name__ == "__main__":
    main()