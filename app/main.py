"""
MP4-to-ASL — FastAPI Server Application
Provides REST endpoints and WebSocket interfaces for MP4 video translation,
speech transcription, ASL gloss translation, pose dispatching, and WebGL frontend rendering.
"""

import os
import sys
import time
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.video_processor import VideoProcessor
from app.asr_engine import ASREngine
from app.gloss_translator import GlossTranslator
from app.pose_dispatcher import PoseDispatcher
from app.websocket_handler import ConnectionManager, handle_websocket_stream

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mp4_to_asl")

# FastAPI App
app = FastAPI(
    title="MP4-to-ASL Real-Time Translation System",
    description="End-to-end pipeline: MP4 Video / Speech -> Faster-Whisper ASR -> ASL Gloss -> 3D Pose Dispatcher -> WebGL 3D Avatar",
    version="2.0.0"
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
video_processor = None
asr_engine = None
gloss_translator = None
pose_dispatcher = None

# Temp directory for video uploads
UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def get_video_processor() -> VideoProcessor:
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor(temp_dir=str(UPLOAD_DIR))
    return video_processor

def get_gloss_translator() -> GlossTranslator:
    global gloss_translator
    if gloss_translator is None:
        gloss_translator = GlossTranslator()
    return gloss_translator

def get_pose_dispatcher() -> PoseDispatcher:
    global pose_dispatcher
    if pose_dispatcher is None:
        pose_dispatcher = PoseDispatcher()
    return pose_dispatcher

def get_asr_engine() -> ASREngine:
    global asr_engine
    if asr_engine is None:
        asr_engine = ASREngine()
    return asr_engine

@app.on_event("startup")
async def startup_event():
    """Initialize AI engines and video processor on server startup."""
    logger.info("Initializing MP4-to-ASL core translation engines...")
    get_video_processor()
    get_pose_dispatcher()
    get_gloss_translator()
    get_asr_engine()
    logger.info("MP4-to-ASL Core Pipeline successfully initialized!")


# Request Models
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    english: str
    gloss: str
    tokens: List[str]
    frame_count: int
    frames: List[Dict[str, Any]]
    latency_ms: float
    non_manual: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gt = get_gloss_translator()
    pd = get_pose_dispatcher()
    return {
        "status": "online",
        "system": "MP4-to-ASL Translation System v2.0",
        "neural_translator": gt.is_neural if gt else False,
        "backend": gt.backend if gt else "none",
        "available_signs": len(pd.signs) if pd else 0,
        "available_alphabet": len(pd.alphabet) if pd else 0
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    """Translate English text directly to ASL Gloss and Pose Keypoints."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start_time = time.perf_counter()
    gt = get_gloss_translator()
    pd = get_pose_dispatcher()

    trans_result = gt.translate(req.text)
    gloss = trans_result.get("gloss", "")
    tokens = trans_result.get("tokens", [])
    non_manual = trans_result.get("non_manual", None)

    pose_frames = pd.dispatch(tokens) if tokens else []
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return TranslationResponse(
        english=req.text,
        gloss=gloss,
        tokens=tokens,
        frame_count=len(pose_frames),
        frames=pose_frames,
        latency_ms=round(elapsed_ms, 2),
        non_manual=non_manual
    )


@app.post("/api/video/process")
async def process_mp4_video(file: UploadFile = File(...)):
    """
    Ingest an MP4 video file, extract audio, transcribe with Faster-Whisper,
    translate speech to ASL Gloss, and map to 3D pose keyframes with video playback timestamps.
    """
    if not file.filename.lower().endswith(('.mp4', '.m4v', '.mov', '.avi', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="File must be a valid video format (MP4, MOV, WEBM, AVI)")

    start_total = time.perf_counter()
    temp_video_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"

    # Save uploaded file
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        vp = get_video_processor()
        asr = get_asr_engine()
        gt = get_gloss_translator()
        pd = get_pose_dispatcher()

        # 1. Metadata
        metadata = vp.get_video_metadata(str(temp_video_path))

        # 2. Extract Audio PCM WAV
        pcm_bytes, wav_path = vp.extract_audio_pcm(str(temp_video_path))

        # 3. Speech Transcription via ASR Engine
        asr.reset()
        asr_result = asr.process_audio_chunk(pcm_bytes)
        if not asr_result:
            asr_result = asr.flush() or {"text": "", "segments": []}

        full_text = asr_result.get("text", "").strip()
        segments = asr_result.get("segments", [])

        # If no explicit segments, fallback to full text
        if not segments and full_text:
            segments = [{"start": 0.0, "end": metadata.get("duration", 3.0), "text": full_text}]

        # 4. Process each timestamped segment into ASL Gloss & Pose Frames
        processed_timeline = []
        all_tokens = []

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            trans_res = gt.translate(seg_text)
            gloss = trans_res.get("gloss", "")
            tokens = trans_res.get("tokens", [])
            all_tokens.extend(tokens)

            pose_frames = pd.dispatch(tokens) if tokens else []

            processed_timeline.append({
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", metadata.get("duration", 3.0)), 2),
                "text": seg_text,
                "gloss": gloss,
                "tokens": tokens,
                "frame_count": len(pose_frames),
                "frames": pose_frames
            })

        total_latency = round((time.perf_counter() - start_total) * 1000, 2)

        return {
            "status": "success",
            "video_metadata": metadata,
            "full_english_text": full_text or "Sample dialogue detected in video",
            "total_gloss_tokens": len(all_tokens),
            "timeline": processed_timeline,
            "latency_ms": total_latency
        }

    except Exception as e:
        logger.error(f"Failed to process video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        # Cleanup uploaded video after processing
        if temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception:
                pass


@app.get("/signs")
async def get_signs():
    """List available pre-built sign tokens and alphabet."""
    pd = get_pose_dispatcher()
    return {
        "signs": pd.get_available_signs(),
        "alphabet": pd.get_available_letters()
    }


# WebSocket Endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for live audio/video chunks."""
    await handle_websocket_stream(
        websocket=websocket,
        manager=ws_manager,
        asr_engine=get_asr_engine(),
        gloss_translator=get_gloss_translator(),
        pose_dispatcher=get_pose_dispatcher()
    )


# Serve Static Frontend Files
PUBLIC_DIR = PROJECT_ROOT / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_file = PUBLIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>MP4-to-ASL Frontend Pending</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
