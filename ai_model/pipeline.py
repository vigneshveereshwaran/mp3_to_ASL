"""
ASLSigningAIModel Master Pipeline
End-to-End Deep Learning Architecture converting ASL Gloss tokens to continuous SMPL-X 3D Signing Motion.
Pipeline: GlossEncoder -> MotionGenerator (VAE) -> TemporalRefiner -> IKConstraintLayer -> SMPLXAdapter.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .config import AIModelConfig
from .encoders import GlossEncoder
from .motion_generator import MotionGenerator
from .temporal_refiner import TemporalRefiner
from .ik_constraints import IKConstraintLayer
from .smplx_adapter import SMPLXAdapter
from .loss_functions import CombinedASLLoss

class ASLSigningAIModel(nn.Module):
    """
    Master AI Motion Generation Model for English/ASL Gloss -> Continuous 3D Human Signing Motion.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = GlossEncoder(config)
        self.generator = MotionGenerator(config)
        self.refiner = TemporalRefiner(config)
        self.ik_layer = IKConstraintLayer(config)
        self.adapter = SMPLXAdapter(config)
        self.loss_fn = CombinedASLLoss(config)

        # Simple Vocabulary Mapping
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def register_vocab(self, vocab_dict: Dict[str, int]):
        """Register custom vocabulary mapping."""
        self.vocab = vocab_dict
        self.inv_vocab = {v: k for k, v in vocab_dict.items()}

    def forward(
        self,
        gloss_tokens: torch.Tensor,
        mask: torch.Tensor = None,
        target_seq_len: int = 120
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass converting Gloss tokens to continuous 3D motion.

        Args:
            gloss_tokens: LongTensor of shape [B, S]
            mask: Padding mask [B, S]
            target_seq_len: Target frames T (e.g. 120)

        Returns:
            Tuple of (constrained_motion [B, T, 218], smplx_dict, velocity, acceleration)
        """
        # 1. Encode Gloss into continuous semantic representations
        memory = self.encoder(gloss_tokens, mask=mask) # [B, S, d_model]

        # 2. Generate Temporal Motion Sequence
        raw_motion, smplx_dict = self.generator(memory, target_seq_len=target_seq_len) # [B, T, 218]

        # 3. Temporal Refinement & Smooth Velocity/Acceleration
        refined_motion, velocity, acceleration = self.refiner(raw_motion)

        # 4. Anatomical IK Constraints & Self-Collision Avoidance
        constrained_motion, smplx_dict = self.ik_layer(refined_motion, smplx_dict)

        return constrained_motion, smplx_dict, velocity, acceleration

    @torch.no_grad()
    def generate_motion_from_gloss(
        self,
        gloss_text: str,
        target_seq_len: int = 120,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        High-Level Inference API: Converts ASL Gloss string (e.g., "WHAT IS YOUR NAME")
        into standard SMPL-X parameter export format.

        Args:
            gloss_text: String of ASL Gloss tokens
            target_seq_len: Number of temporal frames
            device: Torch execution device ("cpu" or "cuda")

        Returns:
            JSON-serializable SMPL-X parameter export dictionary
        """
        self.to(device)
        self.eval()

        tokens = gloss_text.upper().strip().split()
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        if not token_ids:
            token_ids = [self.vocab["<PAD>"]]

        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device) # [1, S]

        # Run pipeline
        motion_tensor, smplx_dict, _, _ = self.forward(token_tensor, target_seq_len=target_seq_len)

        # Format into SMPL-X export dictionary
        export_dict = self.adapter.export_to_json_format(smplx_dict)
        export_dict["gloss_input"] = gloss_text
        export_dict["tokens"] = tokens

        return export_dict
