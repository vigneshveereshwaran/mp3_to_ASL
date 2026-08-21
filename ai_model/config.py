"""
AI Modeling Configuration for ASL 3D Motion Generation
Specifies SMPL-X parametric dimensions, Transformer dimensions, loss weights, and IK parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class AIModelConfig:
    # Frame Rate & Temporal Sequence Specs
    fps: int = 60
    max_seq_len: int = 120            # Frames per clip (2.0s at 60 FPS)
    default_frame_duration: float = 0.65  # Seconds per sign token

    # Vocab & Language Encoder Specs
    vocab_size: int = 2000
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # SMPL-X Parametric Output Dimensions
    root_dim: int = 6                 # 3 trans + 3 rot
    body_joints: int = 21             # Spine, chest, neck, head, shoulders, arms, wrists, hips, legs
    body_pose_dim: int = 63           # 21 joints * 3 rot
    hand_joints: int = 15             # 5 fingers * 3 joints per hand (Thumb CMC/MCP/IP, fingers MCP/PIP/DIP)
    hand_pose_dim: int = 45           # 15 joints * 3 rot per hand
    jaw_pose_dim: int = 3
    expression_dim: int = 50          # ARKit 52 / SMPL-X facial expression blendshapes
    gaze_dim: int = 6                 # 2 eyes * 3 rot
    
    # Total Motion Vector Dim per Frame
    # 6 (root) + 63 (body) + 45 (L hand) + 45 (R hand) + 3 (jaw) + 50 (face) + 6 (gaze) = 218
    motion_dim: int = 218

    # Loss Weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "motion": 1.0,
        "hand": 2.5,                  # Higher priority on hand articulation accuracy
        "body": 1.2,
        "temporal": 1.0,
        "velocity": 0.8,
        "acceleration": 0.5,
        "nmm": 1.5,                   # Facial grammar priority
        "contact": 0.5,
        "anatomical": 1.0,
        "reconstruction": 1.0
    })

    # Anatomical & IK Limits (in Radians)
    elbow_min_deg: float = 0.0        # No elbow hyperextension
    elbow_max_deg: float = 145.0
    wrist_flex_max_deg: float = 85.0
    finger_mcp_max_deg: float = 90.0
    finger_pip_max_deg: float = 100.0
    finger_dip_max_deg: float = 80.0
