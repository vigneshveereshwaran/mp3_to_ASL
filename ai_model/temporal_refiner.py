"""
TemporalRefiner Module
Enforces temporal continuity, velocity smoothing, acceleration refinement,
and minimum-jerk filtering across sign sequences to prevent snapping or T-pose resets.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .config import AIModelConfig

class TemporalRefiner(nn.Module):
    """
    Temporal Motion Refinement Network (1D TCN + Smoothing Residual).
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config

        self.tcn = nn.Sequential(
            nn.Conv1d(config.motion_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, config.motion_dim, kernel_size=5, padding=2)
        )

    def forward(self, motion_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refine temporal sequence, compute velocity and acceleration tensors.

        Args:
            motion_tensor: Raw motion tensor of shape [B, T, 218]

        Returns:
            Tuple of (refined_motion_tensor [B, T, 218], velocity [B, T-1, 218], acceleration [B, T-2, 218])
        """
        # Conv1D expects shape [B, Channels, Length]
        x = motion_tensor.transpose(1, 2)
        residual = self.tcn(x)
        refined_x = x + 0.1 * residual # Residual smoothing step
        refined_motion = refined_x.transpose(1, 2)

        # Compute numerical velocity (v_t = x_t - x_{t-1})
        velocity = refined_motion[:, 1:, :] - refined_motion[:, :-1, :]

        # Compute numerical acceleration (a_t = v_t - v_{t-1})
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]

        return refined_motion, velocity, acceleration
