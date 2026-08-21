"""
Unit tests for app/pose_dispatcher.py
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pose_dispatcher import PoseDispatcher


def test_pose_dispatcher_lookup():
    dispatcher = PoseDispatcher()
    frames = dispatcher.dispatch(["HELLO", "NAME"])

    assert isinstance(frames, list)
    assert len(frames) > 0
    assert "pose" in frames[0]
    assert "right_hand" in frames[0]


def test_pose_dispatcher_fingerspelling():
    dispatcher = PoseDispatcher()
    frames = dispatcher.dispatch(["FS-JOHN"])

    assert isinstance(frames, list)
    assert len(frames) > 0
