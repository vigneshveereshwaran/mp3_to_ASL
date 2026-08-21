"""
Pose & Non-Manual Marker (NMM) Decoders Module
Contains BodyPoseDecoder, HandPoseDecoder (5-finger independent articulation),
FacePoseDecoder, and NMMDecoder (WH-questions, Yes/No questions, Negation).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .config import AIModelConfig

class BodyPoseDecoder(nn.Module):
    """
    Decodes full-body 21 joint rotation parameters (63 dims) plus root translation & rotation (6 dims).
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, config.root_dim + config.body_pose_dim) # 6 + 63 = 69
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, d_model]
        out = self.net(x)
        root = out[..., :6]           # [B, T, 6] (3 trans + 3 rot)
        body_pose = out[..., 6:]      # [B, T, 63] (21 joints * 3)
        return root, body_pose


class HandPoseDecoder(nn.Module):
    """
    Decodes 5-finger articulated hand poses for both Left and Right hands.
    Thumb (CMC, MCP, IP) + Index/Middle/Ring/Pinky (MCP, PIP, DIP) = 15 joints * 3 = 45 dims per hand.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.net_left = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, config.hand_pose_dim) # 45 dims
        )
        self.net_right = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, config.hand_pose_dim) # 45 dims
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, d_model]
        left_hand = self.net_left(x)    # [B, T, 45]
        right_hand = self.net_right(x)  # [B, T, 45]
        return left_hand, right_hand


class FacePoseDecoder(nn.Module):
    """
    Decodes facial expression blendshapes (50 dims), jaw pose (3 dims), and eye gaze (6 dims).
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.SiLU(),
            nn.Linear(256, config.jaw_pose_dim + config.expression_dim + config.gaze_dim) # 3 + 50 + 6 = 59
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        jaw = out[..., :3]
        expression = torch.sigmoid(out[..., 3:53]) # Blendshapes bounded [0, 1]
        gaze = out[..., 53:59]
        return jaw, expression, gaze


class NMMDecoder(nn.Module):
    """
    Decodes synchronized Non-Manual Markers (NMM) for ASL facial grammar:
    - WH-questions: Eyebrow furrowing (-0.8), slight head tilt.
    - Yes/No questions: Eyebrow raising (+0.8), head forward.
    - Negation: Head shaking sequence, tight lips.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # [Declarative, WH-Q, YesNo-Q, Negation]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: [B, T, d_model]
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)

        wh_prob = probs[..., 1:2]
        yesno_prob = probs[..., 2:3]
        neg_prob = probs[..., 3:4]

        # Compute NMM facial offset parameters
        eyebrow_offset = (yesno_prob * 0.8) - (wh_prob * 0.8)
        head_tilt_offset = (wh_prob * 0.15) + (yesno_prob * 0.12)
        head_shake_offset = neg_prob * torch.sin(torch.linspace(0, 12*3.14159, x.size(1), device=x.device)).view(1, -1, 1)

        return {
            "nmm_logits": logits,
            "eyebrow_offset": eyebrow_offset,
            "head_tilt_offset": head_tilt_offset,
            "head_shake_offset": head_shake_offset
        }
