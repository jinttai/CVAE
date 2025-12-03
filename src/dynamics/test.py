import torch
import time
from urdf2robot_torch import urdf2robot
from spart_functions_torch import kinematics

# 1. 로봇 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

start_load = time.perf_counter()
robot, _ = urdf2robot("assets/SC_ur10e.urdf", device=device)
if device == 'cuda': torch.cuda.synchronize()
end_load = time.perf_counter()
print(f"Robot Load Time: {end_load - start_load:.6f}s")

# 2. 가상의 관절 각도 (Requires Grad!)
qm = torch.randn(robot['n_q'], requires_grad=True, device=device)

# 3. Forward Kinematics
# R0, r0는 Identity/Zero로 가정
R0 = torch.eye(3, device=device)
r0 = torch.zeros(3, device=device)

# Warmup (optional, but good for GPU)
if device == 'cuda':
    kinematics(R0, r0, qm, robot)
    torch.cuda.synchronize()

start_fw = time.perf_counter()
RJ, RL, rJ, rL, e, g = kinematics(R0, r0, qm, robot)
if device == 'cuda': torch.cuda.synchronize()
end_fw = time.perf_counter()
print(f"Forward Kinematics Time: {end_fw - start_fw:.6f}s")

# 4. Backward Test
loss = rL.sum() # 임의의 Loss

start_bw = time.perf_counter()
loss.backward()
if device == 'cuda': torch.cuda.synchronize()
end_bw = time.perf_counter()
print(f"Backward Pass Time: {end_bw - start_bw:.6f}s")

print("Gradient Check:", qm.grad) # None이 아니면 성공!
