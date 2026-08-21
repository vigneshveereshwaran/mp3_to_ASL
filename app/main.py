"""
HearLink ASL — Main FastAPI Application Server
Provides REST APIs, WebSocket streaming endpoints, and serves static frontend content.
"""

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.asr_engine import ASREngine
from app.gloss_translator import GlossTranslator
from app.pose_dispatcher import PoseDispatcher
from app.websocket_handler import ConnectionManager, handle_websocket_stream

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hearlink")

# FastAPI App
app = FastAPI(
    title="HearLink ASL API",
    description="Real-time Speech & Text to American Sign Language Translation Pipeline",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
ws_manager = ConnectionManager()
asr_engine = None
gloss_translator = None
pose_dispatcher = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI engines on server startup."""
    global asr_engine, gloss_translator, pose_dispatcher
    logger.info("Initializing HearLink ASL core pipeline...")

    # 1. Pose Dispatcher
    pose_dispatcher = PoseDispatcher()

    # 2. Gloss Translator
    gloss_translator = GlossTranslator()

    # 3. ASR Engine
    asr_engine = ASREngine()

    logger.info("HearLink ASL Core Pipeline successfully initialized!")


# Request Models
class TranslationRequest(BaseModel):
    text: str


class TranslationResponse(BaseModel):
    english: str
    gloss: str
    tokens: list[str]
    frame_count: int
    frames: list[dict]
    latency_ms: float


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "neural_translator": gloss_translator.is_neural if gloss_translator else False,
        "backend": gloss_translator.backend if gloss_translator else "none",
        "available_signs": len(pose_dispatcher.signs) if pose_dispatcher else 0,
        "available_alphabet": len(pose_dispatcher.alphabet) if pose_dispatcher else 0
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    """Translate English text to Gloss and Pose Keypoints."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    trans_result = gloss_translator.translate(req.text)
    gloss = trans_result.get("gloss", "")
    tokens = trans_result.get("tokens", [])

    pose_frames = pose_dispatcher.dispatch(tokens) if tokens else []

    return TranslationResponse(
        english=req.text,
        gloss=gloss,
        tokens=tokens,
        frame_count=len(pose_frames),
        frames=pose_frames,
        latency_ms=trans_result.get("latency_ms", 0.0)
    )


@app.get("/signs")
async def get_signs():
    """List all available pre-built sign tokens."""
    return {
        "signs": pose_dispatcher.get_available_signs(),
        "alphabet": pose_dispatcher.get_available_letters()
    }


# WebSocket Endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint."""
    await handle_websocket_stream(
        websocket=websocket,
        manager=ws_manager,
        asr_engine=asr_engine,
        gloss_translator=gloss_translator,
        pose_dispatcher=pose_dispatcher
    )


# Static File Serving
PUBLIC_DIR = PROJECT_ROOT / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_file = PUBLIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>HearLink ASL Frontend Build Pending</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
