"""Expert modules for Mixture-of-Experts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single FFN expert in the MoE layer.

    Architecture: Linear → GELU → Linear → Dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., input_dim] → [..., output_dim]"""
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
