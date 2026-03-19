"""Tests for RouterAnalytics."""

import torch
import pytest
from dynamic_moe_router import DynamicRouter, RouterAnalytics


@pytest.fixture
def router():
    return DynamicRouter(input_dim=16, num_experts=4, top_k=2, capacity_factor=1.5)


@pytest.fixture
def analytics():
    return RouterAnalytics(num_experts=4, top_k=2, window=50)


class TestRouterAnalytics:
    def test_record_returns_step_stats(self, router, analytics):
        x = torch.randn(4, 8, 16)
        out = router(x)
        stats = analytics.record(step=0, router_output=out)
        assert stats.step == 0
        assert len(stats.expert_load) == 4
        assert stats.overflow_count >= 0

    def test_summary_empty(self, analytics):
        result = analytics.summary()
        assert "error" in result

    def test_summary_after_records(self, router, analytics):
        for step in range(10):
            x = torch.randn(2, 4, 16)
            out = router(x)
            analytics.record(step, out)

        s = analytics.summary()
        assert "steps_recorded" in s
        assert s["steps_recorded"] == 10
        assert len(s["per_expert_avg_load"]) == 4
        assert len(s["per_expert_utilisation"]) == 4
        assert "load_std" in s

    def test_window_respected(self, router):
        analytics = RouterAnalytics(num_experts=4, top_k=2, window=5)
        for step in range(20):
            x = torch.randn(2, 4, 16)
            out = router(x)
            analytics.record(step, out)
        s = analytics.summary()
        assert s["steps_recorded"] == 5  # window=5

    def test_load_history_length(self, router, analytics):
        for step in range(8):
            x = torch.randn(2, 4, 16)
            out = router(x)
            analytics.record(step, out)
        history = analytics.load_history()
        assert len(history) == 8

    def test_ideal_load(self, analytics):
        # top_k=2, num_experts=4 → ideal fraction = 2/4 = 0.5 per expert
        assert analytics.ideal_load == pytest.approx(0.5)

    def test_load_imbalance_score(self, router, analytics):
        for step in range(5):
            x = torch.randn(4, 8, 16)
            out = router(x)
            analytics.record(step, out)
        s = analytics.summary()
        assert s["load_std"] >= 0.0
