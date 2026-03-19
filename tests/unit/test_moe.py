"""Tests for DynamicMoE."""

import torch
import pytest
from dynamic_moe_router import DynamicMoE, RouterOutput


@pytest.fixture
def moe():
    return DynamicMoE(
        input_dim=32,
        expert_hidden_dim=64,
        num_experts=4,
        top_k=2,
        capacity_factor=1.5,
        aux_loss_coeff=1e-2,
    )


class TestDynamicMoE:
    def test_output_shape_3d(self, moe):
        x = torch.randn(2, 8, 32)
        out, router_out = moe(x)
        assert out.shape == (2, 8, 32)

    def test_output_shape_2d(self, moe):
        x = torch.randn(16, 32)
        out, router_out = moe(x)
        assert out.shape == (16, 32)

    def test_returns_router_output(self, moe):
        x = torch.randn(2, 4, 32)
        _, router_out = moe(x)
        assert isinstance(router_out, RouterOutput)

    def test_task_and_aux_loss_backprop(self, moe):
        x = torch.randn(2, 6, 32)
        out, router_out = moe(x)
        loss = out.mean() + router_out.aux_loss
        loss.backward()
        # All parameters should have gradients
        for name, param in moe.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_set_capacity_factor(self, moe):
        moe.set_capacity_factor(2.0)
        assert moe.router.capacity_factor == 2.0

    def test_num_experts(self, moe):
        assert len(moe.experts) == 4

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        moe = DynamicMoE(input_dim=16, expert_hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 16)
        out1, _ = moe(x)

        torch.manual_seed(0)
        moe2 = DynamicMoE(input_dim=16, expert_hidden_dim=32, num_experts=4, top_k=2)
        out2, _ = moe2(x)
        assert torch.allclose(out1, out2)
