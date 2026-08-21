"""
Evaluation Metrics Module for ASL AI Signing Motion
Computes MPJPE (Mean Per Joint Position Error), Hand Joint Error,
Velocity Smoothness, Acceleration Smoothness, and Jerk metrics.
"""

import torch
from typing import Dict, Any

class ASLMotionMetrics:
    """
    Quantitative Evaluation Suite for 3D ASL Motion Quality.
    """

    @staticmethod
    def compute_mpjpe(pred_body_pose: torch.Tensor, target_body_pose: torch.Tensor) -> float:
        """
        Compute Mean Per Joint Position Error (MPJPE) in millimeters/radians.
        """
        diff = pred_body_pose - target_body_pose
        mpjpe = torch.mean(torch.norm(diff.view(-1, 21, 3), dim=-1))
        return float(mpjpe.item())

    @staticmethod
    def compute_hand_error(pred_hand: torch.Tensor, target_hand: torch.Tensor) -> float:
        """
        Compute Hand Joint Error across 15 finger joints (Thumb, Index, Middle, Ring, Pinky).
        """
        diff = pred_hand - target_hand
        hand_err = torch.mean(torch.norm(diff.view(-1, 15, 3), dim=-1))
        return float(hand_err.item())

    @staticmethod
    def compute_temporal_smoothness(velocity: torch.Tensor, acceleration: torch.Tensor) -> Dict[str, float]:
        """
        Compute Velocity Smoothness, Acceleration Smoothness, and Jerk.
        """
        vel_smoothness = float(torch.mean(torch.norm(velocity, dim=-1)).item())
        acc_smoothness = float(torch.mean(torch.norm(acceleration, dim=-1)).item())

        # Jerk (3rd derivative of position: a_t - a_{t-1})
        jerk = acceleration[:, 1:, :] - acceleration[:, :-1, :]
        jerk_val = float(torch.mean(torch.norm(jerk, dim=-1)).item())

        return {
            "velocity_smoothness": vel_smoothness,
            "acceleration_smoothness": acc_smoothness,
            "jerk_metric": jerk_val
        }

    @classmethod
    def evaluate(cls, pred_motion: torch.Tensor, target_motion: torch.Tensor, smplx_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run complete evaluation suite over batch.
        """
        pred_body = smplx_dict["pose_body"]
        target_body = target_motion[..., 6:69]

        pred_r_hand = smplx_dict["right_hand_pose"]
        target_r_hand = target_motion[..., 114:159]

        vel = pred_motion[:, 1:, :] - pred_motion[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]

        mpjpe = cls.compute_mpjpe(pred_body, target_body)
        hand_err = cls.compute_hand_error(pred_r_hand, target_r_hand)
        smoothness = cls.compute_temporal_smoothness(vel, acc)

        return {
            "mpjpe": round(mpjpe, 4),
            "hand_joint_error": round(hand_err, 4),
            "velocity_smoothness": round(smoothness["velocity_smoothness"], 4),
            "acceleration_smoothness": round(smoothness["acceleration_smoothness"], 4),
            "jerk_metric": round(smoothness["jerk_metric"], 4)
        }
