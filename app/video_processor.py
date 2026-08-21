"""
MP4-to-ASL — Video Processor Module
Handles MP4 video ingestion, audio stream extraction (16kHz PCM WAV format),
video metadata extraction, and timestamp frame alignment.
"""

import os
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("mp4_to_asl.video_processor")

class VideoProcessor:
    """
    Video ingestion and audio/visual preprocessing pipeline.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "asl_video_cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_cmd = shutil.which("ffmpeg")

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract duration, resolution, frame rate, and frame count from video file.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        metadata = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "duration": 0.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "frame_count": 0,
            "has_audio": True
        }

        # Try OpenCV first for frame metadata
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    metadata["fps"] = float(fps)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_count > 0:
                    metadata["frame_count"] = int(frame_count)
                    metadata["duration"] = round(frame_count / metadata["fps"], 2)
                cap.release()
        except Exception as e:
            logger.warning(f"OpenCV metadata extraction failed: {e}")

        # Fallback ffprobe if duration was not resolved
        if metadata["duration"] <= 0 and self.ffmpeg_cmd:
            try:
                ffprobe_cmd = shutil.which("ffprobe")
                if ffprobe_cmd:
                    cmd = [
                        ffprobe_cmd, "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(path)
                    ]
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
                    metadata["duration"] = round(float(output), 2)
            except Exception as err:
                logger.warning(f"ffprobe metadata extraction failed: {err}")

        return metadata

    def extract_audio_pcm(self, video_path: str, target_sr: int = 16000) -> Tuple[bytes, str]:
        """
        Extract high-fidelity mono PCM audio (16kHz, 16-bit WAV) from MP4 video file.

        Returns:
            Tuple of (raw_pcm_bytes, wav_file_path)
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_wav = self.temp_dir / f"{path.stem}_audio_{target_sr}.wav"

        # 1. Primary method: FFmpeg CLI
        if self.ffmpeg_cmd:
            cmd = [
                self.ffmpeg_cmd, "-y",
                "-i", str(path),
                "-vn",                     # No video
                "-acodec", "pcm_s16le",    # 16-bit PCM
                "-ar", str(target_sr),     # Target sample rate
                "-ac", "1",                # Mono channel
                str(output_wav)
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if output_wav.exists():
                    with open(output_wav, "rb") as f:
                        pcm_bytes = f.read()
                    return pcm_bytes, str(output_wav)
            except Exception as e:
                logger.warning(f"FFmpeg extraction failed: {e}")

        # 2. Secondary method: MoviePy
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(path))
            if clip.audio is not None:
                clip.audio.write_audiofile(str(output_wav), fps=target_sr, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None)
                clip.close()
                if output_wav.exists():
                    with open(output_wav, "rb") as f:
                        pcm_bytes = f.read()
                    return pcm_bytes, str(output_wav)
        except Exception as e:
            logger.warning(f"MoviePy extraction failed: {e}")

        # 3. Fallback: generate dummy/silent audio if audio stream missing or tools absent
        logger.warning("Could not extract raw audio via FFmpeg/MoviePy. Generating silent buffer.")
        duration = self.get_video_metadata(video_path).get("duration", 3.0)
        num_samples = int(duration * target_sr)
        pcm_bytes = b"\x00\x00" * num_samples
        with open(output_wav, "wb") as f:
            f.write(pcm_bytes)

        return pcm_bytes, str(output_wav)

    def extract_sample_frame(self, video_path: str, timestamp_sec: float) -> Optional[str]:
        """
        Extract a single video frame image at a specific timestamp.
        """
        path = Path(video_path)
        out_frame_path = self.temp_dir / f"{path.stem}_frame_{int(timestamp_sec*1000)}.jpg"
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_idx = int(timestamp_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(str(out_frame_path), frame)
                    cap.release()
                    return str(out_frame_path)
                cap.release()
        except Exception as e:
            logger.warning(f"Frame extraction failed at {timestamp_sec}s: {e}")
        return None

    def cleanup(self):
        """Remove cached temporary audio/video files."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")
