"""
HearLink ASL — WebSocket Stream Handler
Manages real-time audio chunk ingestion, ASR processing, gloss translation,
and skeletal pose streaming over WebSocket.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("hearlink.websocket")


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending JSON to client: {e}")


async def handle_websocket_stream(
    websocket: WebSocket,
    manager: ConnectionManager,
    asr_engine: Any,
    gloss_translator: Any,
    pose_dispatcher: Any
):
    """
    Main WebSocket stream loop.
    Supports receiving:
    1. Binary audio chunks (PCM 16kHz float32 or int16)
    2. JSON control messages (e.g. text input, reset, ping)
    """
    await manager.connect(websocket)

    transcript_buffer = ""
    last_processed_text = ""

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                start_time = time.perf_counter()

                # 1. Process audio via ASR Engine
                asr_result = asr_engine.process_audio_chunk(audio_bytes)

                if asr_result and asr_result.get("text"):
                    new_text = asr_result["text"].strip()
                    if new_text and new_text != last_processed_text:
                        last_processed_text = new_text
                        transcript_buffer += (" " + new_text).strip()

                        # Send updated transcript to client
                        await manager.send_json(websocket, {
                            "type": "transcript",
                            "text": transcript_buffer,
                            "partial": asr_result.get("is_partial", False),
                            "latency_ms": asr_result.get("latency_ms", 0)
                        })

                        # 2. Translate text to ASL Gloss
                        trans_result = gloss_translator.translate(new_text)
                        gloss = trans_result.get("gloss", "")
                        tokens = trans_result.get("tokens", [])

                        await manager.send_json(websocket, {
                            "type": "gloss",
                            "gloss": gloss,
                            "tokens": tokens,
                            "latency_ms": trans_result.get("latency_ms", 0)
                        })

                        # 3. Dispatch Gloss to Pose frames
                        if tokens:
                            pose_frames = pose_dispatcher.dispatch(tokens)
                            total_pipeline_ms = (time.perf_counter() - start_time) * 1000

                            await manager.send_json(websocket, {
                                "type": "pose_frames",
                                "gloss": gloss,
                                "frames": pose_frames,
                                "frame_count": len(pose_frames),
                                "pipeline_latency_ms": total_pipeline_ms
                            })

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "")

                    if msg_type == "text_input":
                        text = payload.get("text", "").strip()
                        if text:
                            start_time = time.perf_counter()

                            # Direct text translation
                            trans_result = gloss_translator.translate(text)
                            gloss = trans_result.get("gloss", "")
                            tokens = trans_result.get("tokens", [])

                            await manager.send_json(websocket, {
                                "type": "transcript",
                                "text": text,
                                "partial": False,
                                "latency_ms": 0
                            })

                            await manager.send_json(websocket, {
                                "type": "gloss",
                                "gloss": gloss,
                                "tokens": tokens,
                                "latency_ms": trans_result.get("latency_ms", 0)
                            })

                            if tokens:
                                pose_frames = pose_dispatcher.dispatch(tokens)
                                total_pipeline_ms = (time.perf_counter() - start_time) * 1000

                                await manager.send_json(websocket, {
                                    "type": "pose_frames",
                                    "gloss": gloss,
                                    "frames": pose_frames,
                                    "frame_count": len(pose_frames),
                                    "pipeline_latency_ms": total_pipeline_ms
                                })

                    elif msg_type == "reset":
                        asr_engine.reset()
                        transcript_buffer = ""
                        last_processed_text = ""
                        await manager.send_json(websocket, {"type": "status", "message": "Buffer reset"})

                    elif msg_type == "ping":
                        await manager.send_json(websocket, {"type": "pong", "timestamp": time.time()})

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON payload over WebSocket")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)
