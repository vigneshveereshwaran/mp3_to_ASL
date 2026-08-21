"""
Combined Multi-Objective ASL Motion Loss Function
Computes L_total = L_motion + L_hand + L_body + L_temporal + L_velocity + L_acceleration + L_NMM + L_anatomical + L_reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .config import AIModelConfig

class CombinedASLLoss(nn.Module):
    """
    Multi-Objective Loss Function balancing ASL linguistic accuracy, handshape precision,
    facial NMM alignment, temporal smoothness, and anatomical plausibility.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.weights = config.loss_weights

    def forward(
        self,
        pred_motion: torch.Tensor,
        target_motion: torch.Tensor,
        smplx_dict: Dict[str, torch.Tensor],
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        target_nmm_labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-objective loss.

        Args:
            pred_motion: Predicted motion tensor [B, T, 218]
            target_motion: Ground truth motion tensor [B, T, 218]
            smplx_dict: Unpacked predicted SMPL-X dict
            velocity: Temporal velocity tensor [B, T-1, 218]
            acceleration: Temporal acceleration tensor [B, T-2, 218]
            target_nmm_labels: Optional ground truth NMM labels [B, T]

        Returns:
            Dict containing total loss and individual loss components
        """
        # 1. Overall Pose Accuracy Loss (L_motion)
        l_motion = F.smooth_l1_loss(pred_motion, target_motion)

        # 2. Hand Pose Accuracy Loss (L_hand) - Higher priority for finger shapes
        pred_left_hand = smplx_dict["left_hand_pose"]
        pred_right_hand = smplx_dict["right_hand_pose"]
        target_left_hand = target_motion[..., 69:114]
        target_right_hand = target_motion[..., 114:159]

        l_hand = F.mse_loss(pred_left_hand, target_left_hand) + F.mse_loss(pred_right_hand, target_right_hand)

        # 3. Body Pose Accuracy Loss (L_body)
        pred_body = smplx_dict["pose_body"]
        target_body = target_motion[..., 6:69]
        l_body = F.mse_loss(pred_body, target_body)

        # 4. Temporal Velocity Smoothness Loss (L_velocity)
        l_velocity = torch.mean(velocity ** 2)

        # 5. Temporal Acceleration Smoothness Loss (L_acceleration - prevents jerky jumps)
        l_acceleration = torch.mean(acceleration ** 2)

        # 6. Non-Manual Marker Loss (L_NMM)
        l_nmm = torch.tensor(0.0, device=pred_motion.device)
        if target_nmm_labels is not None and "nmm" in smplx_dict:
            nmm_logits = smplx_dict["nmm"]["nmm_logits"].view(-1, 4)
            nmm_targets = target_nmm_labels.view(-1)
            l_nmm = F.cross_entropy(nmm_logits, nmm_targets)

        # 7. Anatomical Limit Penalty Loss (L_anatomical)
        elbow_rot_r = pred_body[..., 18:21]
        elbow_rot_l = pred_body[..., 21:24]
        l_anatomical = torch.mean(F.relu(-elbow_rot_r)) + torch.mean(F.relu(-elbow_rot_l))

        # 8. VAE KL Divergence Loss (L_reconstruction)
        mu = smplx_dict.get("mu", torch.zeros(1, device=pred_motion.device))
        logvar = smplx_dict.get("logvar", torch.zeros(1, device=pred_motion.device))
        l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / pred_motion.size(0)

        # Total Weighted Loss
        w = self.weights
        l_total = (
            w["motion"] * l_motion +
            w["hand"] * l_hand +
            w["body"] * l_body +
            w["velocity"] * l_velocity +
            w["acceleration"] * l_acceleration +
            w["nmm"] * l_nmm +
            w["anatomical"] * l_anatomical +
            w["reconstruction"] * l_kl
        )

        return {
            "loss": l_total,
            "l_motion": l_motion.detach(),
            "l_hand": l_hand.detach(),
            "l_body": l_body.detach(),
            "l_velocity": l_velocity.detach(),
            "l_acceleration": l_acceleration.detach(),
            "l_nmm": l_nmm.detach() if isinstance(l_nmm, torch.Tensor) else l_nmm,
            "l_kl": l_kl.detach()
        }
