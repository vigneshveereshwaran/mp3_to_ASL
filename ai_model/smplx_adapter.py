"""
SMPL-X Parameter Adapter Module
Converts raw 218-dim temporal motion vectors into standard SMPL-X parameter format:
root_orient, trans, pose_body (21 joints), left_hand_pose (15 joints),
right_hand_pose (15 joints), jaw_pose, expression (50 blendshapes), and gaze.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .config import AIModelConfig

class SMPLXAdapter(nn.Module):
    """
    Adapter converting 218-dim frame features to structured SMPL-X parameter tensors.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.motion_dim = config.motion_dim

    def unpack_motion_tensor(self, motion_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Unpack a tensor of shape [B, T, 218] or [T, 218] into SMPL-X parameter dictionary.

        Args:
            motion_tensor: Motion vector tensor of shape [B, T, 218]

        Returns:
            Dictionary of SMPL-X parameters
        """
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dim [1, T, 218]

        batch_size, seq_len, _ = motion_tensor.shape

        # Slice 218-dim vector into SMPL-X components
        trans = motion_tensor[..., 0:3]
        root_orient = motion_tensor[..., 3:6]
        pose_body = motion_tensor[..., 6:69]              # 63 dims = 21 joints * 3
        left_hand_pose = motion_tensor[..., 69:114]         # 45 dims = 15 joints * 3
        right_hand_pose = motion_tensor[..., 114:159]       # 45 dims = 15 joints * 3
        jaw_pose = motion_tensor[..., 159:162]              # 3 dims
        expression = motion_tensor[..., 162:212]            # 50 dims
        gaze = motion_tensor[..., 212:218]                  # 6 dims

        return {
            "trans": trans,
            "root_orient": root_orient,
            "pose_body": pose_body,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "jaw_pose": jaw_pose,
            "expression": expression,
            "gaze": gaze,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "fps": self.config.fps
        }

    def pack_smplx_dict(self, smplx_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pack SMPL-X parameter dictionary back into a unified [B, T, 218] motion tensor.
        """
        return torch.cat([
            smplx_dict["trans"],
            smplx_dict["root_orient"],
            smplx_dict["pose_body"],
            smplx_dict["left_hand_pose"],
            smplx_dict["right_hand_pose"],
            smplx_dict["jaw_pose"],
            smplx_dict["expression"],
            smplx_dict["gaze"]
        ], dim=-1)

    def export_to_json_format(self, smplx_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Export batch item 0 to JSON-serializable dictionary format for WebGL/Three.js renderers.
        """
        trans = smplx_dict["trans"][0].detach().cpu().numpy().tolist()
        root_orient = smplx_dict["root_orient"][0].detach().cpu().numpy().tolist()
        pose_body = smplx_dict["pose_body"][0].detach().cpu().numpy().tolist()
        left_hand = smplx_dict["left_hand_pose"][0].detach().cpu().numpy().tolist()
        right_hand = smplx_dict["right_hand_pose"][0].detach().cpu().numpy().tolist()
        jaw = smplx_dict["jaw_pose"][0].detach().cpu().numpy().tolist()
        expression = smplx_dict["expression"][0].detach().cpu().numpy().tolist()
        gaze = smplx_dict["gaze"][0].detach().cpu().numpy().tolist()

        seq_len = smplx_dict["seq_len"]
        frames = []
        for t in range(seq_len):
            frames.append({
                "timestamp": round(t / self.config.fps, 4),
                "trans": trans[t],
                "root_orient": root_orient[t],
                "body_pose": pose_body[t],
                "left_hand_pose": left_hand[t],
                "right_hand_pose": right_hand[t],
                "jaw_pose": jaw[t],
                "expression": expression[t],
                "gaze": gaze[t]
            })

        return {
            "fps": self.config.fps,
            "duration": round(seq_len / self.config.fps, 2),
            "frame_count": seq_len,
            "frames": frames
        }
