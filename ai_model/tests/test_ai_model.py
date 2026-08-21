"""
Unit and Integration Tests for ASL Signing AI Motion Generation Model.
"""

import sys
import torch
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_model.config import AIModelConfig
from ai_model.smplx_adapter import SMPLXAdapter
from ai_model.encoders import GlossEncoder
from ai_model.decoders import BodyPoseDecoder, HandPoseDecoder, FacePoseDecoder, NMMDecoder
from ai_model.motion_generator import MotionGenerator
from ai_model.temporal_refiner import TemporalRefiner
from ai_model.ik_constraints import IKConstraintLayer
from ai_model.loss_functions import CombinedASLLoss
from ai_model.pipeline import ASLSigningAIModel
from ai_model.metrics import ASLMotionMetrics


def test_config_dimensions():
    """Verify SMPL-X motion parameter dimension calculation."""
    config = AIModelConfig()
    # 6 (root) + 63 (body) + 45 (L hand) + 45 (R hand) + 3 (jaw) + 50 (face) + 6 (gaze) = 218
    assert config.motion_dim == 218
    assert config.body_pose_dim == 63
    assert config.hand_pose_dim == 45


def test_gloss_encoder():
    """Test Transformer GlossEncoder forward pass."""
    config = AIModelConfig()
    encoder = GlossEncoder(config)
    tokens = torch.tensor([[10, 25, 42, 5]], dtype=torch.long) # Batch 1, Seq 4
    memory = encoder(tokens)
    assert memory.shape == (1, 4, config.d_model)


def test_decoders_and_nmm():
    """Test HandPoseDecoder (5 fingers, 15 joints) and NMMDecoder."""
    config = AIModelConfig()
    x = torch.randn(2, 60, config.d_model) # Batch 2, Frames 60

    hand_dec = HandPoseDecoder(config)
    left_hand, right_hand = hand_dec(x)
    assert left_hand.shape == (2, 60, 45)  # 15 joints * 3
    assert right_hand.shape == (2, 60, 45) # 15 joints * 3

    nmm_dec = NMMDecoder(config)
    nmm_out = nmm_dec(x)
    assert "eyebrow_offset" in nmm_out
    assert "head_tilt_offset" in nmm_out
    assert "head_shake_offset" in nmm_out


def test_motion_generator_and_adapter():
    """Test MotionGenerator and SMPLXAdapter packing/unpacking."""
    config = AIModelConfig()
    gen = MotionGenerator(config)
    adapter = SMPLXAdapter(config)

    memory = torch.randn(1, 5, config.d_model)
    motion_tensor, smplx_dict = gen(memory, target_seq_len=60)

    assert motion_tensor.shape == (1, 60, 218)
    assert smplx_dict["pose_body"].shape == (1, 60, 63)
    assert smplx_dict["left_hand_pose"].shape == (1, 60, 45)
    assert smplx_dict["right_hand_pose"].shape == (1, 60, 45)
    assert smplx_dict["expression"].shape == (1, 60, 50)

    repacked = adapter.pack_smplx_dict(smplx_dict)
    assert repacked.shape == (1, 60, 218)


def test_ik_constraints_layer():
    """Test joint limit clamping and collision avoidance."""
    config = AIModelConfig()
    gen = MotionGenerator(config)
    ik_layer = IKConstraintLayer(config)

    memory = torch.randn(1, 4, config.d_model)
    motion_tensor, smplx_dict = gen(memory, target_seq_len=30)

    constrained_motion, updated_dict = ik_layer(motion_tensor, smplx_dict)
    assert constrained_motion.shape == (1, 30, 218)


def test_combined_loss_computation():
    """Test multi-objective loss calculation."""
    config = AIModelConfig()
    loss_fn = CombinedASLLoss(config)

    pred = torch.randn(2, 30, 218)
    target = torch.randn(2, 30, 218)
    vel = pred[:, 1:, :] - pred[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]

    adapter = SMPLXAdapter(config)
    smplx_dict = adapter.unpack_motion_tensor(pred)

    loss_dict = loss_fn(pred, target, smplx_dict, vel, acc)
    assert "loss" in loss_dict
    assert "l_hand" in loss_dict
    assert "l_motion" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_evaluation_metrics():
    """Test MPJPE and hand joint error metrics."""
    config = AIModelConfig()
    pred = torch.randn(1, 30, 218)
    target = torch.randn(1, 30, 218)

    adapter = SMPLXAdapter(config)
    smplx_dict = adapter.unpack_motion_tensor(pred)

    metrics = ASLMotionMetrics.evaluate(pred, target, smplx_dict)
    assert "mpjpe" in metrics
    assert "hand_joint_error" in metrics
    assert "velocity_smoothness" in metrics


def test_end_to_end_ai_model_inference():
    """Test master ASLSigningAIModel generate_motion_from_gloss for 'WHAT IS YOUR NAME'."""
    config = AIModelConfig()
    model = ASLSigningAIModel(config)

    # High-level inference pass
    export = model.generate_motion_from_gloss("WHAT IS YOUR NAME?", target_seq_len=60)

    assert export["frame_count"] == 60
    assert export["fps"] == 60
    assert export["duration"] == 1.0
    assert len(export["frames"]) == 60

    first_frame = export["frames"][0]
    assert len(first_frame["body_pose"]) == 63
    assert len(first_frame["left_hand_pose"]) == 45
    assert len(first_frame["right_hand_pose"]) == 45
    assert len(first_frame["expression"]) == 50
