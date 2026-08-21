"""
End-to-End integration test for HearLink ASL FastAPI endpoints
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"


def test_translate_endpoint():
    with TestClient(app) as client:
        response = client.post("/translate", json={"text": "Hello, how are you?"})
        assert response.status_code == 200
        data = response.json()

        assert "english" in data
        assert "gloss" in data
        assert "tokens" in data
        assert "frame_count" in data
        assert "frames" in data
        assert data["frame_count"] > 0


def test_get_signs_endpoint():
    with TestClient(app) as client:
        response = client.get("/signs")
        assert response.status_code == 200
        data = response.json()
        assert "signs" in data
        assert "alphabet" in data
