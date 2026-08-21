"""
MP4-to-ASL — Test Suite
Tests video processor, ASR engine, Gloss translator, Pose dispatcher, and FastAPI endpoints.
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.video_processor import VideoProcessor
from app.asr_engine import ASREngine
from app.gloss_translator import GlossTranslator
from app.pose_dispatcher import PoseDispatcher

client = TestClient(app)

def test_health_check_endpoint():
    """Test health check API endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "available_signs" in data


def test_translate_text_endpoint():
    """Test text-to-ASL translation endpoint."""
    payload = {"text": "Hello, how are you?"}
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["english"] == "Hello, how are you?"
    assert "gloss" in data
    assert len(data["tokens"]) > 0
    assert data["frame_count"] > 0
    assert len(data["frames"]) == data["frame_count"]


def test_signs_endpoint():
    """Test signs listing endpoint."""
    response = client.get("/signs")
    assert response.status_code == 200
    data = response.json()
    assert "signs" in data
    assert "alphabet" in data


def test_gloss_translator_grammar():
    """Test English to ASL Gloss grammar transformation logic."""
    translator = GlossTranslator()

    # WH-Question
    res1 = translator.translate("What is your name?")
    assert "WHAT" in res1["gloss"]
    assert "NAME" in res1["gloss"]

    # Time fronting & Negation
    res2 = translator.translate("Yesterday I did not go to school")
    assert "YESTERDAY" in res2["gloss"]
    assert "NOT" in res2["gloss"]


def test_pose_dispatcher():
    """Test pose dispatcher keypoint dispatch."""
    dispatcher = PoseDispatcher()

    # Known token dispatch
    frames = dispatcher.dispatch(["HELLO", "IX-1"])
    assert len(frames) > 0
    first_frame = frames[0]
    assert "pose" in first_frame
    assert "right_hand" in first_frame

    # Unknown token fingerspelling dispatch
    fs_frames = dispatcher.dispatch(["FS-TESTING"])
    assert len(fs_frames) > 0


def test_video_processor_dummy():
    """Test video processor audio extraction fallback."""
    processor = VideoProcessor()

    # Create dummy video path for testing
    dummy_path = PROJECT_ROOT / "tests" / "dummy_sample.mp4"
    dummy_path.parent.mkdir(exist_ok=True)
    with open(dummy_path, "wb") as f:
        f.write(b"\x00" * 1024)

    try:
        meta = processor.get_video_metadata(str(dummy_path))
        assert "filename" in meta

        pcm_bytes, wav_path = processor.extract_audio_pcm(str(dummy_path))
        assert isinstance(pcm_bytes, bytes)
        assert Path(wav_path).exists()
    finally:
        if dummy_path.exists():
            dummy_path.unlink()
        processor.cleanup()
