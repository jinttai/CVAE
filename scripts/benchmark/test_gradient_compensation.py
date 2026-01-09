import torch
import os
import sys
import math

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_quaternion_from_euler(batch_size, max_angle_deg=30.0, device='cpu'):
    """
    Generate random quaternions from Euler angles within specified range
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    
    quaternions = euler_to_quaternion(roll, pitch, yaw)
    return quaternions


def main():
    print("=== Gradient Compensation Test ===")
    
    # 1. Setup & Loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load robot
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    print(f"Robot loaded: {robot['name']}, n_q: {robot['n_q']}")
    
    # Model parameters (matching train_cvae.py)
    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 3
    TOTAL_TIME = 10.0
    
    # Joint regularization weights (matching train_cvae.py)
    JOINT_SQUARED_WEIGHT = 0.01
    JOINT_CHANGE_WEIGHT = 0.01
    MAX_JOINT_WEIGHT = 0.1
    
    # Initialize model
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM, joint_limits=robot['joint_limits']).to(device)
    
    # Load pre-trained weights
    weights_path = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Model weights loaded from: {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}, using random initialization")
    
    model.eval()  # Set to evaluation mode
    
    # Initialize PhysicsLayer
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    print("PhysicsLayer initialized")
    
    # 2. Data Generation
    print("\n=== Generating Test Data ===")
    batch_size = 1
    
    # Create random condition (start/goal)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # Identity quaternion
    q0_goal = generate_random_quaternion_from_euler(1, max_angle_deg=10.0, device=device)
    condition = torch.cat([q0_start, q0_goal], dim=1)  # [1, 8]
    
    print(f"Start quaternion: {q0_start[0].cpu().numpy()}")
    print(f"Goal quaternion: {q0_goal[0].cpu().numpy()}")
    
    # Generate random latent vector
    z = torch.randn(batch_size, LATENT_DIM, device=device)
    print(f"Latent vector z shape: {z.shape}")
    
    # Run model inference to get waypoints_pred
    with torch.no_grad():
        waypoints_pred = model.decode(condition, z)  # [1, OUTPUT_DIM]
    
    print(f"Predicted waypoints shape: {waypoints_pred.shape}")
    print(f"Predicted waypoints (first 10 values): {waypoints_pred[0, :10].cpu().numpy()}")
    
    # Detach and create new tensor with requires_grad=True
    waypoints_opt = waypoints_pred.detach().clone().requires_grad_(True)
    print(f"Created waypoints_opt with requires_grad={waypoints_opt.requires_grad}")
    
    # 3. Gradient Calculation
    print("\n=== Calculating Gradients ===")
    
    # Calculate loss
    loss, loss_dict = physics.calculate_total_loss(
        waypoints_opt, q0_start, q0_goal,
        joint_squared_weight=JOINT_SQUARED_WEIGHT,
        joint_change_weight=JOINT_CHANGE_WEIGHT,
        max_joint_weight=MAX_JOINT_WEIGHT
    )
    
    print(f"Original Loss: {loss.item():.6f}")
    print(f"  - Physics Loss: {loss_dict['physics_loss'].item():.6f}")
    print(f"  - Joint Squared Penalty: {loss_dict['joint_squared_penalty'].item():.6f}")
    print(f"  - Joint Change Penalty: {loss_dict['joint_change_penalty'].item():.6f}")
    print(f"  - Max Joint Penalty: {loss_dict['max_joint_penalty'].item():.6f}")
    
    # Compute gradients
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=waypoints_opt,
        create_graph=False,
        retain_graph=False
    )[0]  # [1, OUTPUT_DIM]
    
    grad_flat = grad[0]  # [OUTPUT_DIM] - flatten for easier indexing
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient norm: {torch.norm(grad_flat).item():.6f}")
    print(f"Gradient (first 10 values): {grad_flat[:10].cpu().numpy()}")
    
    # 4. Compensation Logic
    print("\n=== Applying Compensation Logic ===")
    
    # Select target index to perturb
    target_idx = 10
    delta_val = 0.01
    
    print(f"Target index: {target_idx}")
    print(f"Perturbation delta: {delta_val}")
    
    # Get gradient at target index
    grad_target = grad_flat[target_idx]
    print(f"Gradient at target index: {grad_target.item():.6f}")
    
    # Create mask for other indices (all except target_idx)
    mask = torch.ones(OUTPUT_DIM, dtype=torch.bool, device=device)
    mask[target_idx] = False
    
    # Get gradients for other indices
    grad_others = grad_flat[mask]  # [OUTPUT_DIM - 1]
    print(f"Other gradients shape: {grad_others.shape}")
    print(f"Other gradients norm: {torch.norm(grad_others).item():.6f}")
    
    # Calculate required change for other indices to cancel out loss change
    # Formula: delta_others = - (grad_target * delta_val / ||grad_others||^2) * grad_others
    grad_others_norm_sq = torch.sum(grad_others ** 2)
    
    if grad_others_norm_sq < 1e-12:
        print("Warning: ||grad_others||^2 is too small, compensation may be unstable")
        print("Using small regularization term")
        grad_others_norm_sq = grad_others_norm_sq + 1e-12
    
    # Calculate compensation vector
    scale_factor = - (grad_target * delta_val) / grad_others_norm_sq
    delta_others = scale_factor * grad_others
    
    print(f"Scale factor: {scale_factor.item():.6f}")
    print(f"Compensation vector norm: {torch.norm(delta_others).item():.6f}")
    print(f"Compensation vector (first 10 values): {delta_others[:10].cpu().numpy()}")
    
    # Create full modification vector
    modification = torch.zeros(OUTPUT_DIM, device=device)
    modification[target_idx] = delta_val  # Perturbation at target index
    modification[mask] = delta_others  # Compensation at other indices
    
    print(f"Full modification vector norm: {torch.norm(modification).item():.6f}")
    
    # Verify the math: expected loss change should be approximately zero
    expected_loss_change = grad_target * delta_val + torch.dot(grad_others, delta_others)
    print(f"\nExpected loss change (should be ~0): {expected_loss_change.item():.10f}")
    
    # 5. Verification
    print("\n=== Verification ===")
    
    # Create new trajectory with modification
    new_waypoints = waypoints_opt.detach().clone() + modification.unsqueeze(0)  # [1, OUTPUT_DIM]
    new_waypoints.requires_grad_(False)  # No need for gradients on this
    
    # Calculate new loss
    with torch.no_grad():
        new_loss, new_loss_dict = physics.calculate_total_loss(
            new_waypoints, q0_start, q0_goal,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT
        )
    
    # Calculate loss without compensation (only target index perturbed)
    waypoints_no_comp = waypoints_opt.detach().clone()
    waypoints_no_comp[0, target_idx] += delta_val
    waypoints_no_comp.requires_grad_(False)
    
    with torch.no_grad():
        loss_no_comp, _ = physics.calculate_total_loss(
            waypoints_no_comp, q0_start, q0_goal,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT
        )
    
    # Results
    print(f"Original Loss: {loss.item():.10f}")
    print(f"New Loss (with compensation): {new_loss.item():.10f}")
    print(f"Loss without compensation (only target perturbed): {loss_no_comp.item():.10f}")
    print(f"\nLoss Difference (with compensation): {abs(new_loss.item() - loss.item()):.10f}")
    print(f"Loss Difference (without compensation): {abs(loss_no_comp.item() - loss.item()):.10f}")
    print(f"\nRelative change (with compensation): {abs(new_loss.item() - loss.item()) / (abs(loss.item()) + 1e-10):.10e}")
    print(f"Relative change (without compensation): {abs(loss_no_comp.item() - loss.item()) / (abs(loss.item()) + 1e-10):.10e}")
    
    # Success criteria: compensated loss should be much closer to original
    improvement_factor = abs(loss_no_comp.item() - loss.item()) / (abs(new_loss.item() - loss.item()) + 1e-12)
    print(f"\nImprovement factor: {improvement_factor:.2f}x")
    
    if abs(new_loss.item() - loss.item()) < abs(loss_no_comp.item() - loss.item()) / 10:
        print("✓ SUCCESS: Compensation significantly reduces loss change!")
    else:
        print("⚠ WARNING: Compensation may not be working as expected")
    
    # Additional verification: check individual loss components
    print("\n=== Loss Component Analysis ===")
    print("Original Loss Components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val.item():.6f}")
    
    print("New Loss Components (with compensation):")
    for key, val in new_loss_dict.items():
        print(f"  {key}: {val.item():.6f}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()

