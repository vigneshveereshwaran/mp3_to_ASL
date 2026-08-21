"""
Unit tests for datasets/text_normalizer.py
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.text_normalizer import (
    normalize_english,
    normalize_gloss,
    english_to_asl_grammar,
    handle_fingerspelling,
    clean_pair
)


def test_normalize_english():
    assert normalize_english("I'm going to the store.") == "i am going to the store"
    assert normalize_english("She doesn't like tea!!") == "she does not like tea"
    assert normalize_english("  HELLO   WORLD  ") == "hello world"


def test_normalize_gloss():
    assert normalize_gloss("hello world") == "HELLO WORLD"
    assert normalize_gloss("fs-john ix-1") == "FS-JOHN IX-1"


def test_english_to_asl_grammar():
    res = english_to_asl_grammar("Tomorrow I will go to the store")
    assert "TOMORROW" in res
    assert "STORE" in res
    assert "IX-1" in res


def test_handle_fingerspelling():
    assert handle_fingerspelling("mary") == "FS-MARY"


def test_clean_pair():
    pair = clean_pair("I am happy", "HAPPY IX-1")
    assert pair == ("i am happy", "HAPPY IX-1")

    assert clean_pair("", "HAPPY") is None
