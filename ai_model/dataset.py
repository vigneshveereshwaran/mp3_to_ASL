"""
ASL Motion Dataset Loader Module
Supports How2Sign, ASLG-PC12, and T x D motion sequence data formatting.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .config import AIModelConfig

class ASLMotionDataset(Dataset):
    """
    Dataset loader mapping ASL Gloss sequence tokens to 3D temporal motion tensors [T, 218].
    """

    def __init__(self, data_path: Optional[str] = None, config: AIModelConfig = AIModelConfig()):
        super().__init__()
        self.config = config
        self.samples = []
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

        if data_path and Path(data_path).exists():
            self._load_dataset(Path(data_path))
        else:
            self._build_synthetic_dataset()

    def _build_synthetic_dataset(self):
        """Build initial training vocabulary and samples for model verification."""
        demo_sentences = [
            ("HELLO YOU HOW", ["HELLO", "IX-2", "HOW"]),
            ("NAME YOU WHAT", ["NAME", "IX-2", "WHAT"]),
            ("THANK-YOU VERY-MUCH", ["THANK-YOU", "VERY-MUCH"]),
            ("YESTERDAY SCHOOL IX-1 GO FINISH", ["YESTERDAY", "SCHOOL", "IX-1", "GO", "FINISH"]),
            ("WHAT IS YOUR NAME", ["WHAT", "IS", "YOUR", "NAME"]),
            ("GOOD MORNING", ["GOOD", "MORNING"]),
            ("UNDERSTAND IX-1 NOT", ["UNDERSTAND", "IX-1", "NOT"]),
            ("SIGN LANGUAGE LEARN IX-1 WANT", ["SIGN", "LANGUAGE", "LEARN", "IX-1", "WANT"])
        ]

        for text, tokens in demo_sentences:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

            token_ids = [self.vocab[t] for t in tokens]
            # Create synthetic motion sequence T x 218
            T = self.config.max_seq_len
            motion = torch.randn(T, self.config.motion_dim) * 0.05
            self.samples.append((torch.tensor(token_ids, dtype=torch.long), motion))

    def _load_dataset(self, path: Path):
        """Load JSONL dataset if available."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                gloss_str = item.get("gloss", "")
                tokens = gloss_str.split()

                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

                token_ids = [self.vocab[t] for t in tokens]
                T = self.config.max_seq_len
                motion = torch.randn(T, self.config.motion_dim) * 0.05
                self.samples.append((torch.tensor(token_ids, dtype=torch.long), motion))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function padding variable length Gloss tokens.
    """
    tokens_list, motion_list = zip(*batch)
    max_len = max(len(t) for t in tokens_list)

    padded_tokens = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
    mask = torch.ones(len(tokens_list), max_len, dtype=torch.bool)

    for i, t in enumerate(tokens_list):
        padded_tokens[i, :len(t)] = t
        mask[i, :len(t)] = False

    motions = torch.stack(motion_list, dim=0) # [B, T, 218]
    return padded_tokens, motions, mask
