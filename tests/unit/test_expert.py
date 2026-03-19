"""Tests for Expert module."""

import torch
import pytest
from dynamic_moe_router import Expert


class TestExpert:
    def test_output_shape(self):
        expert = Expert(input_dim=16, hidden_dim=64, output_dim=16)
        x = torch.randn(4, 16)
        out = expert(x)
        assert out.shape == (4, 16)

    def test_different_input_output_dims(self):
        expert = Expert(input_dim=16, hidden_dim=32, output_dim=8)
        x = torch.randn(4, 16)
        out = expert(x)
        assert out.shape == (4, 8)

    def test_3d_input(self):
        expert = Expert(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(2, 5, 16)
        out = expert(x)
        assert out.shape == (2, 5, 16)

    def test_gradients_flow(self):
        expert = Expert(input_dim=8, hidden_dim=16, output_dim=8)
        x = torch.randn(2, 8, requires_grad=True)
        out = expert(x).sum()
        out.backward()
        assert x.grad is not None
