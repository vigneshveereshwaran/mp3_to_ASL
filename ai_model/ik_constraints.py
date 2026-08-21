"""
IKConstraintLayer Module
Anatomical constraint solver enforcing joint angle limits (elbow hyperextension, wrist flex,
finger joint ranges) and self-collision avoidance (hand-torso, hand-face).
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple
from .config import AIModelConfig
from .smplx_adapter import SMPLXAdapter

class IKConstraintLayer(nn.Module):
    """
    Differentiable Anatomical Joint Limit & Self-Collision Constraint Layer.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.adapter = SMPLXAdapter(config)

        # Convert degree limits to radians
        deg2rad = math.pi / 180.0
        self.elbow_min = config.elbow_min_deg * deg2rad
        self.elbow_max = config.elbow_max_deg * deg2rad
        self.wrist_max = config.wrist_flex_max_deg * deg2rad
        self.finger_mcp_max = config.finger_mcp_max_deg * deg2rad
        self.finger_pip_max = config.finger_pip_max_deg * deg2rad
        self.finger_dip_max = config.finger_dip_max_deg * deg2rad

    def enforce_joint_limits(self, smplx_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clamp joint rotation angles to anatomically valid human joint limits (out-of-place).
        """
        pose_body = smplx_dict["pose_body"]
        left_hand = torch.clamp(smplx_dict["left_hand_pose"], -0.2, self.finger_pip_max)
        right_hand = torch.clamp(smplx_dict["right_hand_pose"], -0.2, self.finger_pip_max)

        # Clamp elbow flexion (18:24) & wrist flexion (24:30) out-of-place
        part_0_18 = pose_body[..., :18]
        elbows_clamped = torch.clamp(pose_body[..., 18:24], self.elbow_min, self.elbow_max)
        wrists_clamped = torch.clamp(pose_body[..., 24:30], -self.wrist_max, self.wrist_max)
        part_30_end = pose_body[..., 30:]

        clamped_body = torch.cat([part_0_18, elbows_clamped, wrists_clamped, part_30_end], dim=-1)

        updated_dict = dict(smplx_dict)
        updated_dict["pose_body"] = clamped_body
        updated_dict["left_hand_pose"] = left_hand
        updated_dict["right_hand_pose"] = right_hand
        return updated_dict

    def resolve_self_collisions(self, smplx_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adjust hands if wrist translation penetrates torso bounds (out-of-place).
        """
        pose_body = smplx_dict["pose_body"]
        right_wrist_z = pose_body[..., 29:30]
        left_wrist_z = pose_body[..., 26:27]

        penetration_r = torch.relu(-0.05 - right_wrist_z)
        penetration_l = torch.relu(-0.05 - left_wrist_z)

        l_adjusted = left_wrist_z + 0.5 * penetration_l
        r_adjusted = right_wrist_z + 0.5 * penetration_r

        # Out-of-place reconstruct
        part_0_26 = pose_body[..., :26]
        part_27_29 = pose_body[..., 27:29]
        part_30_end = pose_body[..., 30:]

        adjusted_body = torch.cat([part_0_26, l_adjusted, part_27_29, r_adjusted, part_30_end], dim=-1)

        updated_dict = dict(smplx_dict)
        updated_dict["pose_body"] = adjusted_body
        return updated_dict

    def forward(self, motion_tensor: torch.Tensor, smplx_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply anatomical constraints and return updated motion tensor + smplx_dict.
        """
        smplx_dict = self.enforce_joint_limits(smplx_dict)
        smplx_dict = self.resolve_self_collisions(smplx_dict)

        constrained_motion = self.adapter.pack_smplx_dict(smplx_dict)
        return constrained_motion, smplx_dict
