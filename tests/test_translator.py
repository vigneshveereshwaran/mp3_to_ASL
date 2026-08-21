"""
Unit tests for app/gloss_translator.py
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.gloss_translator import GlossTranslator


def test_gloss_translator_fallback():
    translator = GlossTranslator()
    res = translator.translate("I want to learn sign language")

    assert "gloss" in res
    assert "tokens" in res
    assert isinstance(res["tokens"], list)
    assert len(res["tokens"]) > 0
    assert "latency_ms" in res
