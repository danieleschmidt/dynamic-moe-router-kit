"""Tests for DynamicRouter."""

import torch
import pytest
from dynamic_moe_router import DynamicRouter, RouterOutput


@pytest.fixture
def router():
    return DynamicRouter(input_dim=32, num_experts=8, top_k=2, capacity_factor=1.5)


class TestDynamicRouter:
    def test_output_types(self, router):
        x = torch.randn(4, 8, 32)
        out = router(x)
        assert isinstance(out, RouterOutput)
        assert isinstance(out.dispatch_indices, torch.Tensor)
        assert isinstance(out.dispatch_weights, torch.Tensor)
        assert isinstance(out.aux_loss, torch.Tensor)
        assert isinstance(out.overflow_count, int)

    def test_dispatch_indices_shape(self, router):
        B, T = 3, 10
        x = torch.randn(B, T, 32)
        out = router(x)
        assert out.dispatch_indices.shape == (B * T, 2)

    def test_dispatch_weights_sum_to_one_or_zero(self, router):
        """Weights per token should sum to 1.0 (normal) or 0.0 (fully overflowed token)."""
        x = torch.randn(4, 6, 32)
        out = router(x)
        sums = out.dispatch_weights.sum(dim=-1)
        ones = torch.ones_like(sums)
        zeros = torch.zeros_like(sums)
        ok = torch.isclose(sums, ones, atol=1e-5) | torch.isclose(sums, zeros, atol=1e-5)
        assert ok.all(), f"Weight sums not in {{0, 1}}: {sums[~ok]}"

    def test_expert_indices_in_range(self, router):
        x = torch.randn(4, 6, 32)
        out = router(x)
        assert out.dispatch_indices.min() >= 0
        assert out.dispatch_indices.max() < router.num_experts

    def test_aux_loss_positive(self, router):
        x = torch.randn(4, 8, 32)
        out = router(x)
        assert out.aux_loss.item() >= 0

    def test_aux_loss_backprop(self, router):
        x = torch.randn(4, 8, 32)
        out = router(x)
        out.aux_loss.backward()
        for param in router.gate.parameters():
            assert param.grad is not None

    def test_top_k_validation(self):
        with pytest.raises(ValueError, match="top_k"):
            DynamicRouter(input_dim=8, num_experts=4, top_k=5)

    def test_capacity_factor_setter(self, router):
        router.set_capacity_factor(2.0)
        assert router.capacity_factor == 2.0

    def test_capacity_factor_too_low(self, router):
        with pytest.raises(ValueError):
            router.set_capacity_factor(0.5)

    def test_flat_2d_input(self, router):
        x = torch.randn(16, 32)  # (S, D) — already flat
        out = router(x)
        assert out.dispatch_indices.shape == (16, 2)

    def test_top_k_1(self):
        router = DynamicRouter(input_dim=16, num_experts=4, top_k=1)
        x = torch.randn(2, 5, 16)
        out = router(x)
        assert out.dispatch_indices.shape == (10, 1)
        # Weights should all be 1.0 since top_k=1
        assert torch.allclose(out.dispatch_weights, torch.ones_like(out.dispatch_weights), atol=1e-5)

    def test_overflow_count_nonnegative(self, router):
        x = torch.randn(8, 16, 32)
        out = router(x)
        assert out.overflow_count >= 0
