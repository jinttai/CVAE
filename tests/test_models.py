"""Tests for neural network models."""

import pytest
import torch
from NN_opt.model.cvae import ConditionalDecoder, MLP, CVAE


class TestMLP:
    """Tests for MLP model."""

    @pytest.fixture
    def mlp_model(self):
        """Create MLP model for testing."""
        return MLP(input_dim=8, output_dim=18)  # 3 waypoints * 6 joints

    def test_forward_shape(self, mlp_model):
        """Test output shape is correct."""
        batch_size = 4
        condition = torch.randn(batch_size, 8)
        output = mlp_model(condition)
        assert output.shape == (batch_size, 18)

    def test_output_range_default(self, mlp_model):
        """Test output is within default range."""
        condition = torch.randn(10, 8)
        output = mlp_model(condition)
        # Default range is approximately [-2.44, 2.44] rad (140 degrees)
        assert output.abs().max() <= 2.5

    def test_with_joint_limits(self):
        """Test MLP with joint limits."""
        joint_limits = torch.tensor([
            [-3.14, 3.14],
            [-1.57, 1.57],
            [-3.14, 3.14],
            [-1.57, 1.57],
            [-3.14, 3.14],
            [-1.57, 1.57],
        ])
        model = MLP(input_dim=8, output_dim=18, joint_limits=joint_limits)
        condition = torch.randn(4, 8)
        output = model(condition)

        # Reshape to check per-joint limits
        output_reshaped = output.view(4, 3, 6)
        for j in range(6):
            assert (output_reshaped[:, :, j] >= joint_limits[j, 0]).all()
            assert (output_reshaped[:, :, j] <= joint_limits[j, 1]).all()

    def test_deterministic(self, mlp_model):
        """Test MLP produces deterministic output."""
        condition = torch.randn(2, 8)
        output1 = mlp_model(condition)
        output2 = mlp_model(condition)
        assert torch.allclose(output1, output2)


class TestConditionalDecoder:
    """Tests for ConditionalDecoder (formerly CVAE) model."""

    @pytest.fixture
    def decoder_model(self):
        """Create ConditionalDecoder model for testing."""
        return ConditionalDecoder(
            condition_dim=8,
            output_dim=18,
            latent_dim=3
        )

    def test_decode_shape(self, decoder_model):
        """Test decode output shape."""
        batch_size = 4
        condition = torch.randn(batch_size, 8)
        z = torch.randn(batch_size, 3)
        output = decoder_model.decode(condition, z)
        assert output.shape == (batch_size, 18)

    def test_encode_shape(self, decoder_model):
        """Test encode output shapes."""
        batch_size = 4
        condition = torch.randn(batch_size, 8)
        trajectory = torch.randn(batch_size, 18)
        mu, logvar = decoder_model.encode(condition, trajectory)
        assert mu.shape == (batch_size, 3)
        assert logvar.shape == (batch_size, 3)

    def test_inference_shape(self, decoder_model):
        """Test inference output shape."""
        batch_size = 4
        condition = torch.randn(batch_size, 8)
        output = decoder_model.inference(condition)
        assert output.shape == (batch_size, 18)

    def test_stochastic_inference(self, decoder_model):
        """Test inference produces different outputs (stochastic)."""
        condition = torch.randn(1, 8)

        # Run inference multiple times
        outputs = [decoder_model.inference(condition) for _ in range(5)]

        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Inference should be stochastic"

    def test_forward_shape(self, decoder_model):
        """Test full forward pass output shapes."""
        batch_size = 4
        condition = torch.randn(batch_size, 8)
        trajectory = torch.randn(batch_size, 18)
        recon, mu, logvar = decoder_model(condition, trajectory)
        assert recon.shape == (batch_size, 18)
        assert mu.shape == (batch_size, 3)
        assert logvar.shape == (batch_size, 3)

    def test_reparameterize(self, decoder_model):
        """Test reparameterization produces valid samples."""
        mu = torch.zeros(4, 3)
        logvar = torch.zeros(4, 3)  # std = 1
        z = decoder_model.reparameterize(mu, logvar)
        assert z.shape == (4, 3)
        # With std=1 and mu=0, values should be roughly standard normal
        assert z.abs().mean() < 2.0


class TestBackwardCompatibility:
    """Test backward compatibility with CVAE alias."""

    def test_cvae_alias(self):
        """Test CVAE is alias for ConditionalDecoder."""
        assert CVAE is ConditionalDecoder

    def test_cvae_instantiation(self):
        """Test CVAE can be instantiated."""
        model = CVAE(condition_dim=8, output_dim=18, latent_dim=3)
        assert isinstance(model, ConditionalDecoder)


class TestGradients:
    """Test gradient flow through models."""

    def test_mlp_gradients(self):
        """Test gradients flow through MLP."""
        model = MLP(input_dim=8, output_dim=18)
        condition = torch.randn(4, 8, requires_grad=True)
        output = model(condition)
        loss = output.sum()
        loss.backward()
        assert condition.grad is not None
        assert not torch.isnan(condition.grad).any()

    def test_decoder_gradients(self):
        """Test gradients flow through ConditionalDecoder."""
        model = ConditionalDecoder(condition_dim=8, output_dim=18, latent_dim=3)
        condition = torch.randn(4, 8, requires_grad=True)
        z = torch.randn(4, 3, requires_grad=True)
        output = model.decode(condition, z)
        loss = output.sum()
        loss.backward()
        assert condition.grad is not None
        assert z.grad is not None
