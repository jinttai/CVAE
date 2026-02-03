"""Tests for physics utility functions."""

import pytest
import torch
import numpy as np
from physics.utils import (
    euler_to_quaternion_np,
    euler_to_quaternion_torch,
    quat_to_rot_np,
    quat_to_rot_torch,
    geodesic_distance_so3_np,
    geodesic_distance_so3_torch,
    skew_np,
    skew_torch,
    euler_to_quaternion,
    quat_to_rot,
)


class TestEulerToQuaternion:
    """Tests for Euler to quaternion conversion."""

    def test_identity_np(self):
        """Test zero angles give identity quaternion."""
        q = euler_to_quaternion_np(0, 0, 0)
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_almost_equal(q, expected, decimal=6)

    def test_identity_torch(self):
        """Test zero angles give identity quaternion (torch)."""
        q = euler_to_quaternion_torch(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0)
        )
        expected = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        assert torch.allclose(q, expected, atol=1e-6)

    def test_90_deg_yaw_np(self):
        """Test 90 degree yaw rotation."""
        q = euler_to_quaternion_np(0, 0, np.pi / 2)
        # For 90 deg yaw: qz = sin(45) ≈ 0.707, qw = cos(45) ≈ 0.707
        assert abs(q[2]) > 0.7  # qz should be significant
        assert abs(q[3]) > 0.7  # qw should be significant
        assert abs(q[0]) < 0.01  # qx should be near zero
        assert abs(q[1]) < 0.01  # qy should be near zero

    def test_batch_torch(self):
        """Test batch processing in torch version."""
        roll = torch.tensor([0.0, 0.1, 0.2])
        pitch = torch.tensor([0.0, 0.1, 0.2])
        yaw = torch.tensor([0.0, 0.1, 0.2])
        q = euler_to_quaternion_torch(roll, pitch, yaw)
        assert q.shape == (3, 4)

    def test_unified_interface_np(self):
        """Test unified interface with numpy input."""
        q = euler_to_quaternion(0.1, 0.2, 0.3)
        assert isinstance(q, np.ndarray)
        assert q.shape == (4,)

    def test_unified_interface_torch(self):
        """Test unified interface with torch input."""
        q = euler_to_quaternion(
            torch.tensor(0.1),
            torch.tensor(0.2),
            torch.tensor(0.3)
        )
        assert isinstance(q, torch.Tensor)


class TestQuatToRot:
    """Tests for quaternion to rotation matrix conversion."""

    def test_identity_np(self):
        """Test identity quaternion gives identity matrix."""
        q = np.array([0, 0, 0, 1])
        R = quat_to_rot_np(q)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=6)

    def test_identity_torch(self):
        """Test identity quaternion gives identity matrix (torch)."""
        q = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        R = quat_to_rot_torch(q)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_orthogonal_np(self):
        """Test rotation matrix is orthogonal."""
        q = euler_to_quaternion_np(0.1, 0.2, 0.3)
        R = quat_to_rot_np(q)
        # R @ R^T should be identity
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=6)
        # det(R) should be 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_orthogonal_torch(self):
        """Test rotation matrix is orthogonal (torch)."""
        q = euler_to_quaternion_torch(
            torch.tensor(0.1),
            torch.tensor(0.2),
            torch.tensor(0.3)
        )
        R = quat_to_rot_torch(q)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
        assert abs(torch.det(R) - 1.0) < 1e-6

    def test_batch_torch(self):
        """Test batch processing in torch version."""
        q = torch.randn(5, 4)
        q = q / q.norm(dim=1, keepdim=True)  # Normalize
        R = quat_to_rot_torch(q)
        assert R.shape == (5, 3, 3)


class TestGeodesicDistance:
    """Tests for geodesic distance on SO(3)."""

    def test_identity_np(self):
        """Test distance between identical matrices is zero."""
        R = np.eye(3)
        dist = geodesic_distance_so3_np(R, R)
        assert abs(dist) < 1e-6

    def test_identity_torch(self):
        """Test distance between identical matrices is zero (torch)."""
        R = torch.eye(3)
        dist = geodesic_distance_so3_torch(R, R)
        assert abs(dist.item()) < 1e-6

    def test_90_deg_rotation_np(self):
        """Test distance for 90 degree rotation."""
        R1 = np.eye(3)
        # 90 degree rotation around z-axis
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        dist = geodesic_distance_so3_np(R1, R2)
        # Should be pi/2 radians
        assert abs(dist - np.pi / 2) < 1e-6

    def test_symmetric_np(self):
        """Test distance is symmetric."""
        q1 = euler_to_quaternion_np(0.1, 0.2, 0.3)
        q2 = euler_to_quaternion_np(0.4, 0.5, 0.6)
        R1 = quat_to_rot_np(q1)
        R2 = quat_to_rot_np(q2)
        dist12 = geodesic_distance_so3_np(R1, R2)
        dist21 = geodesic_distance_so3_np(R2, R1)
        assert abs(dist12 - dist21) < 1e-6


class TestSkew:
    """Tests for skew-symmetric matrix."""

    def test_skew_np(self):
        """Test skew matrix properties."""
        v = np.array([1, 2, 3])
        S = skew_np(v)
        # Skew matrix should be antisymmetric
        np.testing.assert_array_almost_equal(S, -S.T)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(S), [0, 0, 0])

    def test_skew_torch(self):
        """Test skew matrix properties (torch)."""
        v = torch.tensor([1.0, 2.0, 3.0])
        S = skew_torch(v)
        # Skew matrix should be antisymmetric
        assert torch.allclose(S, -S.T, atol=1e-6)
        # Diagonal should be zero
        assert torch.allclose(torch.diag(S), torch.zeros(3), atol=1e-6)

    def test_cross_product_np(self):
        """Test skew matrix gives cross product."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        S_a = skew_np(a)
        # S_a @ b should equal a × b
        cross = np.cross(a, b)
        np.testing.assert_array_almost_equal(S_a @ b, cross)


class TestConsistency:
    """Test consistency between numpy and torch implementations."""

    def test_euler_to_quat_consistency(self):
        """Test numpy and torch give same results."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q_np = euler_to_quaternion_np(roll, pitch, yaw)
        q_torch = euler_to_quaternion_torch(
            torch.tensor(roll),
            torch.tensor(pitch),
            torch.tensor(yaw)
        ).numpy()
        np.testing.assert_array_almost_equal(q_np, q_torch, decimal=5)

    def test_quat_to_rot_consistency(self):
        """Test numpy and torch give same rotation matrices."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)  # Normalize
        R_np = quat_to_rot_np(q)
        R_torch = quat_to_rot_torch(torch.tensor(q, dtype=torch.float32)).numpy()
        np.testing.assert_array_almost_equal(R_np, R_torch, decimal=5)
