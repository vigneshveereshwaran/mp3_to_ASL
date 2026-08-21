"""
HearLink ASL — Gloss Translator
Inference wrapper for the trained English → ASL Gloss model.
Supports both CTranslate2 (fast) and HuggingFace (fallback) backends.
"""

import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GlossTranslator:
    """
    English to ASL Gloss translation inference engine.
    Uses CTranslate2 for production speed, falls back to HF Transformers.
    """

    def __init__(self, model_path: str = None,
                 ct2_model_path: str = None,
                 task_prefix: str = "translate English to ASL gloss: "):
        """
        Initialize the translator.

        Args:
            model_path: Path to HuggingFace model directory
            ct2_model_path: Path to CTranslate2 model directory (preferred)
            task_prefix: T5 task prefix
        """
        self.task_prefix = task_prefix
        self.tokenizer = None
        self.ct2_translator = None
        self.hf_model = None
        self.backend = None

        # Default paths
        if model_path is None:
            model_path = str(PROJECT_ROOT / "app" / "models" / "best_gloss_model")
        if ct2_model_path is None:
            ct2_model_path = str(PROJECT_ROOT / "app" / "models" / "ct2_gloss_model")

        # Try CTranslate2 first (fastest)
        if Path(ct2_model_path).exists():
            try:
                import ctranslate2
                from transformers import AutoTokenizer

                self.ct2_translator = ctranslate2.Translator(
                    ct2_model_path,
                    compute_type="int8",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.backend = "ctranslate2"
                print(f"[GlossTranslator] Loaded CTranslate2 model from {ct2_model_path}")
                return
            except Exception as e:
                print(f"[GlossTranslator] CTranslate2 load failed: {e}")

        # Try HuggingFace Transformers
        if Path(model_path).exists():
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.hf_model.eval()

                if torch.cuda.is_available():
                    self.hf_model = self.hf_model.cuda()

                self.backend = "transformers"
                print(f"[GlossTranslator] Loaded HF model from {model_path}")
                return
            except Exception as e:
                print(f"[GlossTranslator] HF model load failed: {e}")

        # Fallback: rule-based translator
        self.backend = "rule_based"
        print("[GlossTranslator] Using rule-based fallback translator")

    def translate(self, text: str, num_beams: int = 4) -> dict:
        """
        Translate English text to ASL Gloss.

        Args:
            text: English input text
            num_beams: Beam search width

        Returns:
            Dict with 'gloss' (string), 'tokens' (list), 'latency_ms' (float)
        """
        text = text.strip()
        if not text:
            return {"gloss": "", "tokens": [], "latency_ms": 0}

        start = time.perf_counter()

        if self.backend == "ctranslate2":
            result = self._translate_ct2(text, num_beams)
        elif self.backend == "transformers":
            result = self._translate_hf(text, num_beams)
        else:
            result = self._translate_rule_based(text)

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = elapsed_ms
        return result

    def _translate_ct2(self, text: str, num_beams: int) -> dict:
        """Translate using CTranslate2."""
        input_text = self.task_prefix + text
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(input_text)
        )

        results = self.ct2_translator.translate_batch(
            [input_tokens],
            beam_size=num_beams,
            max_decoding_length=64,
        )

        output_tokens = results[0].hypotheses[0]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        gloss = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        tokens = gloss.split()

        return {"gloss": gloss, "tokens": tokens}

    def _translate_hf(self, text: str, num_beams: int) -> dict:
        """Translate using HuggingFace Transformers."""
        import torch

        input_text = self.task_prefix + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )

        device = next(self.hf_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_length=64,
                num_beams=num_beams,
                early_stopping=True,
            )

        gloss = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        tokens = gloss.split()

        return {"gloss": gloss, "tokens": tokens}

    def _translate_rule_based(self, text: str) -> dict:
        """
        Rule-based fallback translation.
        Uses the text normalizer's heuristic grammar transformer.
        """
        from datasets.text_normalizer import english_to_asl_grammar
        gloss = english_to_asl_grammar(text)
        tokens = gloss.split() if gloss else []
        return {"gloss": gloss, "tokens": tokens}

    def translate_streaming(self, text_buffer: str,
                             min_words: int = 3) -> Optional[dict]:
        """
        Streaming-aware translation with sliding window.
        Only translates when enough words have accumulated.

        Args:
            text_buffer: Accumulated text from ASR
            min_words: Minimum words before translating

        Returns:
            Translation result or None if not enough words
        """
        words = text_buffer.strip().split()
        if len(words) < min_words:
            return None

        # Check for sentence boundaries
        has_boundary = any(
            text_buffer.rstrip().endswith(p)
            for p in ['.', '?', '!', ',']
        )

        if has_boundary or len(words) >= min_words:
            return self.translate(text_buffer)

        return None

    @property
    def is_neural(self) -> bool:
        """Whether a neural model is loaded."""
        return self.backend in ("ctranslate2", "transformers")
