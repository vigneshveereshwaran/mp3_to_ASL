"""
MotionGenerator Module
Transformer VAE & Temporal Motion Decoder generating continuous frame sequences (T x D).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .config import AIModelConfig
from .decoders import BodyPoseDecoder, HandPoseDecoder, FacePoseDecoder, NMMDecoder
from .smplx_adapter import SMPLXAdapter

class MotionGenerator(nn.Module):
    """
    Temporal Motion VAE Backbone generating full SMPL-X parameter sequences from continuous Gloss embeddings.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config

        # Temporal Positional Encodings for Decoder
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_decoder_layers
        )

        # VAE Latent Projections
        self.fc_mu = nn.Linear(config.d_model, config.d_model)
        self.fc_logvar = nn.Linear(config.d_model, config.d_model)

        # Sub-Decoders
        self.body_decoder = BodyPoseDecoder(config)
        self.hand_decoder = HandPoseDecoder(config)
        self.face_decoder = FacePoseDecoder(config)
        self.nmm_decoder = NMMDecoder(config)
        self.adapter = SMPLXAdapter(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE latent sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, memory: torch.Tensor, target_seq_len: int = 120) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate continuous motion sequence from encoder memory.

        Args:
            memory: GlossEncoder output of shape [B, S, d_model]
            target_seq_len: Desired temporal sequence length T (e.g., 120)

        Returns:
            Tuple of (raw_motion_tensor [B, T, 218], smplx_dict)
        """
        batch_size = memory.size(0)

        # 1. Latent Sampling
        memory_mean = memory.mean(dim=1) # [B, d_model]
        mu = self.fc_mu(memory_mean)
        logvar = self.fc_logvar(memory_mean)
        z = self.reparameterize(mu, logvar).unsqueeze(1) # [B, 1, d_model]

        # 2. Target Query Sequence Generation
        tgt_queries = self.pe[:, :target_seq_len, :].repeat(batch_size, 1, 1) + z # [B, T, d_model]

        # 3. Transformer Decoder
        decoded_features = self.transformer_decoder(tgt_queries, memory) # [B, T, d_model]

        # 4. Sub-Decoders Execution
        root, body_pose = self.body_decoder(decoded_features)
        left_hand, right_hand = self.hand_decoder(decoded_features)
        jaw, expression, gaze = self.face_decoder(decoded_features)
        nmm = self.nmm_decoder(decoded_features)

        # Apply NMM facial / head offsets out-of-place
        head_tilt = nmm["head_tilt_offset"]
        body_pose = torch.cat([body_pose[..., :9], body_pose[..., 9:10] + head_tilt, body_pose[..., 10:]], dim=-1)
        expression = torch.cat([expression[..., :1] + nmm["eyebrow_offset"].clamp(0, 1), expression[..., 1:]], dim=-1)

        # 5. Pack into 218-dim unified motion tensor
        smplx_dict = {
            "trans": root[..., :3],
            "root_orient": root[..., 3:],
            "pose_body": body_pose,
            "left_hand_pose": left_hand,
            "right_hand_pose": right_hand,
            "jaw_pose": jaw,
            "expression": expression,
            "gaze": gaze,
            "mu": mu,
            "logvar": logvar,
            "nmm": nmm,
            "batch_size": batch_size,
            "seq_len": target_seq_len
        }

        motion_tensor = self.adapter.pack_smplx_dict(smplx_dict)
        return motion_tensor, smplx_dict
