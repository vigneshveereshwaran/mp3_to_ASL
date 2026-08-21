"""
HearLink ASL — Pose Extraction Pipeline
Extracts 3D skeletal keypoints from sign language video using MediaPipe.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP_TASKS = True
except ImportError:
    HAS_MP_TASKS = False
    try:
        import mediapipe as mp
        HAS_MP_LEGACY = True
    except ImportError:
        HAS_MP_LEGACY = False


# Landmark indices for upper body focus
POSE_UPPER_BODY_INDICES = list(range(0, 25))  # Head, shoulders, arms, hands
LEFT_HAND_INDICES = list(range(21))   # 21 hand landmarks
RIGHT_HAND_INDICES = list(range(21))
FACE_KEY_INDICES = [
    # Key facial landmarks for expressions (subset of 468)
    1, 4, 5, 6,         # Nose
    33, 133, 362, 263,   # Eyes outer/inner corners
    61, 291,             # Mouth corners
    0, 17,               # Chin, forehead
    78, 308,             # Upper/lower lip
    70, 300,             # Eyebrow inner
    105, 334,            # Eyebrow outer
    159, 386,            # Eyelids
]


class PoseExtractor:
    """
    Extract upper-body, hand, and facial landmarks from video frames.
    Supports both MediaPipe Tasks API (modern) and legacy solutions API.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pose extractor.

        Args:
            model_path: Path to holistic_landmarker.task model bundle.
                        If None, tries to use legacy API.
        """
        self.use_tasks_api = False
        self.landmarker = None
        self.holistic = None

        if HAS_MP_TASKS and model_path and Path(model_path).exists():
            # Modern Tasks API
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HolisticLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
            )
            self.landmarker = mp_vision.HolisticLandmarker.create_from_options(options)
            self.use_tasks_api = True
            print("[PoseExtractor] Using MediaPipe Tasks API")

        elif HAS_MP_LEGACY or (HAS_MP_TASKS and not model_path):
            # Legacy API fallback
            mp_holistic = mp.solutions.holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[PoseExtractor] Using MediaPipe Legacy Holistic API")
        else:
            print("[PoseExtractor] WARNING: MediaPipe not available. Using dummy data.")

    def extract_frame(self, frame: np.ndarray) -> dict:
        """
        Extract landmarks from a single video frame.

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            Dictionary with 'pose', 'left_hand', 'right_hand', 'face' landmark arrays.
            Each landmark is [x, y, z] normalized coordinates.
        """
        result = {
            "pose": [],
            "left_hand": [],
            "right_hand": [],
            "face": [],
            "blendshapes": {},
        }

        if self.use_tasks_api and self.landmarker:
            return self._extract_tasks_api(frame, result)
        elif self.holistic:
            return self._extract_legacy(frame, result)
        else:
            return self._extract_dummy(frame, result)

    def _extract_tasks_api(self, frame: np.ndarray, result: dict) -> dict:
        """Extract using modern MediaPipe Tasks API."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection = self.landmarker.detect(mp_image)

        # Pose landmarks (upper body)
        if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
            landmarks = detection.pose_landmarks[0]
            for idx in POSE_UPPER_BODY_INDICES:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    result["pose"].append([lm.x, lm.y, lm.z])

        # Hand landmarks
        if detection.left_hand_landmarks and len(detection.left_hand_landmarks) > 0:
            for lm in detection.left_hand_landmarks[0]:
                result["left_hand"].append([lm.x, lm.y, lm.z])

        if detection.right_hand_landmarks and len(detection.right_hand_landmarks) > 0:
            for lm in detection.right_hand_landmarks[0]:
                result["right_hand"].append([lm.x, lm.y, lm.z])

        # Face landmarks (subset)
        if detection.face_landmarks and len(detection.face_landmarks) > 0:
            face_lms = detection.face_landmarks[0]
            for idx in FACE_KEY_INDICES:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    result["face"].append([lm.x, lm.y, lm.z])

        # Blendshapes
        if hasattr(detection, 'face_blendshapes') and detection.face_blendshapes:
            for bs in detection.face_blendshapes[0]:
                result["blendshapes"][bs.category_name] = bs.score

        return result

    def _extract_legacy(self, frame: np.ndarray, result: dict) -> dict:
        """Extract using legacy MediaPipe Holistic API."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.holistic.process(rgb_frame)

        # Pose landmarks
        if detection.pose_landmarks:
            for idx in POSE_UPPER_BODY_INDICES:
                if idx < len(detection.pose_landmarks.landmark):
                    lm = detection.pose_landmarks.landmark[idx]
                    result["pose"].append([lm.x, lm.y, lm.z])

        # Hands
        if detection.left_hand_landmarks:
            for lm in detection.left_hand_landmarks.landmark:
                result["left_hand"].append([lm.x, lm.y, lm.z])

        if detection.right_hand_landmarks:
            for lm in detection.right_hand_landmarks.landmark:
                result["right_hand"].append([lm.x, lm.y, lm.z])

        # Face (subset of 468 landmarks)
        if detection.face_landmarks:
            for idx in FACE_KEY_INDICES:
                if idx < len(detection.face_landmarks.landmark):
                    lm = detection.face_landmarks.landmark[idx]
                    result["face"].append([lm.x, lm.y, lm.z])

        return result

    def _extract_dummy(self, frame: np.ndarray, result: dict) -> dict:
        """Generate dummy landmarks when MediaPipe is not available."""
        # Generate plausible dummy data for testing
        result["pose"] = [[0.5, 0.3 + i * 0.02, 0.0] for i in range(25)]
        result["left_hand"] = [[0.3 + i * 0.01, 0.5 + i * 0.01, 0.0] for i in range(21)]
        result["right_hand"] = [[0.7 - i * 0.01, 0.5 + i * 0.01, 0.0] for i in range(21)]
        result["face"] = [[0.5, 0.2 + i * 0.005, 0.0] for i in range(len(FACE_KEY_INDICES))]
        return result

    def close(self):
        """Release resources."""
        if self.landmarker:
            self.landmarker.close()
        if self.holistic:
            self.holistic.close()


def normalize_keypoints(landmarks: dict, shoulder_left_idx: int = 11,
                        shoulder_right_idx: int = 12) -> dict:
    """
    Normalize landmarks relative to shoulder midpoint and torso scale.

    Args:
        landmarks: Raw landmark dict from extract_frame()
        shoulder_left_idx: Index of left shoulder in pose landmarks
        shoulder_right_idx: Index of right shoulder in pose landmarks

    Returns:
        Normalized landmark dict
    """
    pose = np.array(landmarks.get("pose", []))
    if len(pose) < max(shoulder_left_idx, shoulder_right_idx) + 1:
        return landmarks  # Can't normalize without shoulders

    # Compute center point (shoulder midpoint)
    center = (pose[shoulder_left_idx] + pose[shoulder_right_idx]) / 2.0

    # Compute scale (shoulder width)
    shoulder_dist = np.linalg.norm(pose[shoulder_left_idx] - pose[shoulder_right_idx])
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0  # Prevent division by zero

    normalized = {}
    for key in ["pose", "left_hand", "right_hand", "face"]:
        data = np.array(landmarks.get(key, []))
        if len(data) > 0:
            data = (data - center) / shoulder_dist
            normalized[key] = data.tolist()
        else:
            normalized[key] = []

    # Pass through blendshapes unchanged
    normalized["blendshapes"] = landmarks.get("blendshapes", {})

    return normalized


def extract_video(video_path: str, output_path: str,
                  model_path: Optional[str] = None,
                  max_frames: Optional[int] = None,
                  sample_fps: int = 30) -> int:
    """
    Extract pose keypoints from an entire video file.

    Args:
        video_path: Path to input video file
        output_path: Path to output JSON file
        model_path: Path to MediaPipe model bundle (optional)
        max_frames: Maximum frames to process (None = all)
        sample_fps: Target sampling FPS

    Returns:
        Number of frames processed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extractor = PoseExtractor(model_path)
    frames_data = []
    frame_idx = 0
    processed = 0

    print(f"Processing video: {video_path}")
    print(f"  Video FPS: {video_fps}, Sampling every {frame_interval} frames")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=min(total_frames, max_frames or total_frames))
    except ImportError:
        pbar = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and processed >= max_frames:
            break

        if frame_idx % frame_interval == 0:
            raw_landmarks = extractor.extract_frame(frame)
            norm_landmarks = normalize_keypoints(raw_landmarks)
            norm_landmarks["frame"] = processed
            norm_landmarks["timestamp"] = frame_idx / video_fps
            frames_data.append(norm_landmarks)
            processed += 1

            if pbar:
                pbar.update(1)

        frame_idx += 1

    if pbar:
        pbar.close()

    cap.release()
    extractor.close()

    # Save to JSON
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump({
            "source_video": str(video_path),
            "total_frames": processed,
            "sample_fps": sample_fps,
            "frames": frames_data,
        }, f, indent=2)

    print(f"  Extracted {processed} frames → {output_path}")
    return processed


def batch_extract(video_dir: str, output_dir: str,
                  model_path: Optional[str] = None,
                  extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')):
    """
    Extract poses from all videos in a directory.

    Args:
        video_dir: Directory containing sign language videos
        output_dir: Directory for output JSON files
        model_path: Path to MediaPipe model bundle
        extensions: Video file extensions to process
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"**/*{ext}"))

    video_files = sorted(set(video_files))
    print(f"Found {len(video_files)} video files in {video_dir}")

    for vf in video_files:
        out_name = vf.stem + "_poses.json"
        out_path = output_dir / out_name
        if out_path.exists():
            print(f"  [skip] Already extracted: {out_name}")
            continue
        extract_video(str(vf), str(out_path), model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pose keypoints from sign language video")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("-o", "--output", default="datasets/data/poses",
                        help="Output directory for JSON keypoint files")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to holistic_landmarker.task model bundle")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to extract per video")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target sampling FPS (default: 30)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        batch_extract(str(input_path), args.output, args.model)
    elif input_path.is_file():
        out_path = Path(args.output) / (input_path.stem + "_poses.json")
        extract_video(str(input_path), str(out_path), args.model,
                      max_frames=args.max_frames, sample_fps=args.fps)
    else:
        print(f"Error: {args.input} does not exist")
        sys.exit(1)
