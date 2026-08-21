"""
GlossEncoder Module
Transformer Language Encoder converting ASL Gloss tokens into continuous semantic representations.
"""

import math
import torch
import torch.nn as nn
from .config import AIModelConfig

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence modeling."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D]
        return x + self.pe[:, :x.size(1)]


class GlossEncoder(nn.Module):
    """
    Multi-head Self-Attention Transformer Encoder for ASL Gloss sequences.
    """

    def __init__(self, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=500)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, gloss_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode discrete Gloss tokens into continuous semantic representations.

        Args:
            gloss_tokens: LongTensor of shape [B, S] (batch size, sequence length)
            mask: Optional attention mask [B, S]

        Returns:
            Semantic feature tensor of shape [B, S, d_model]
        """
        x = self.embedding(gloss_tokens) * math.sqrt(self.config.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=mask)
        return self.norm(memory)
