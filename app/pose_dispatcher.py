"""
HearLink ASL — Pose Dispatcher
Maps ASL Gloss tokens to 3D skeletal keypoint sequences.
"""

import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose_library.interpolation import generate_transition_frames, smooth_pose_sequence


class PoseDispatcher:
    """
    Dispatches ASL gloss tokens to 3D pose keypoint sequences.
    Handles sign lookup, fingerspelling fallback, and smooth transitions.
    """

    def __init__(self, library_dir: str = None, transition_frames: int = 4):
        """
        Initialize the pose dispatcher.

        Args:
            library_dir: Directory containing sign JSON files + manifest
            transition_frames: Number of interpolated frames between signs
        """
        if library_dir is None:
            library_dir = str(PROJECT_ROOT / "pose_library" / "signs")

        self.library_dir = Path(library_dir)
        self.transition_frames = transition_frames
        self.signs = {}
        self.alphabet = {}
        self.manifest = {}

        self._load_library()

    def _load_library(self):
        """Load the pose library manifest and sign data."""
        manifest_path = self.library_dir / "manifest.json"

        if not manifest_path.exists():
            print("[PoseDispatcher] No pose library found. Building...")
            from pose_library.build_library import build_library
            build_library(str(self.library_dir))

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)

            # Load all signs into memory
            for sign_name, filename in self.manifest.get("signs", {}).items():
                filepath = self.library_dir / filename
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        self.signs[sign_name] = json.load(f)

            # Load alphabet
            for letter, filename in self.manifest.get("alphabet", {}).items():
                filepath = self.library_dir / filename
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        self.alphabet[letter] = json.load(f)

            print(f"[PoseDispatcher] Loaded {len(self.signs)} signs + "
                  f"{len(self.alphabet)} alphabet letters")
        else:
            print("[PoseDispatcher] WARNING: Could not load pose library")

    def dispatch(self, gloss_tokens: list[str]) -> list[dict]:
        """
        Convert a sequence of gloss tokens to pose keypoint frames.

        Args:
            gloss_tokens: List of ASL gloss tokens (e.g., ["HELLO", "IX-1", "NAME", "WHAT"])

        Returns:
            List of pose frame dicts with 'pose', 'right_hand', 'left_hand' keys
        """
        if not gloss_tokens:
            return []

        all_frames = []
        prev_last_frame = None

        for token in gloss_tokens:
            token = token.upper().strip()
            if not token:
                continue

            # Look up the sign
            sign_frames = self._lookup_sign(token)

            if not sign_frames:
                continue

            # Generate transition from previous sign
            if prev_last_frame is not None and self.transition_frames > 0:
                first_frame = sign_frames[0]
                transitions = generate_transition_frames(
                    prev_last_frame, first_frame,
                    num_frames=self.transition_frames,
                    use_easing=True,
                )
                all_frames.extend(transitions)

            # Add the sign frames
            all_frames.extend(sign_frames)
            prev_last_frame = sign_frames[-1]

        # Apply smoothing
        if len(all_frames) > 3:
            all_frames = smooth_pose_sequence(all_frames, window_size=3)

        return all_frames

    def _lookup_sign(self, token: str) -> list[dict]:
        """
        Look up a single gloss token in the pose library.

        Args:
            token: ASL gloss token (e.g., "HELLO", "FS-J", "IX-1")

        Returns:
            List of pose frames for this sign
        """
        # Direct sign lookup
        if token in self.signs:
            sign_data = self.signs[token]
            return sign_data.get("frames", [])

        # Fingerspelling lookup (FS-WORD)
        if token.startswith("FS-"):
            word = token[3:]
            return self._fingerspell(word)

        # Index/pronoun signs (IX-1, IX-2, IX-3, POSS-1, etc.)
        # Map to pointing gesture variants
        if token.startswith("IX-") or token.startswith("POSS-") or token.startswith("SELF-"):
            return self._lookup_pronoun_sign(token)

        # Compound tokens with hyphens (e.g., "LAST-WEEK", "WAKE-UP")
        if "-" in token:
            # Try the full compound first
            if token in self.signs:
                return self.signs[token].get("frames", [])
            # Fall back to individual parts
            parts = token.split("-")
            frames = []
            for part in parts:
                part_frames = self._lookup_sign(part)
                frames.extend(part_frames)
            if frames:
                return frames

        # Unknown sign → fingerspell it
        return self._fingerspell(token)

    def _fingerspell(self, word: str) -> list[dict]:
        """
        Generate fingerspelling frames for a word.

        Args:
            word: Word to fingerspell

        Returns:
            List of pose frames (letter by letter)
        """
        frames = []
        prev_frame = None

        for char in word.upper():
            if char in self.alphabet:
                letter_data = self.alphabet[char]
                letter_frames = letter_data.get("frames", [])

                # Create full pose frames for each letter
                for hand_frame in letter_frames:
                    pose_frame = self._hand_to_full_pose(hand_frame)

                    # Add brief transition between letters
                    if prev_frame is not None:
                        transitions = generate_transition_frames(
                            prev_frame, pose_frame,
                            num_frames=2,
                            use_easing=True,
                        )
                        frames.extend(transitions)

                    frames.append(pose_frame)
                    # Hold each letter for a few frames
                    frames.append(pose_frame)
                    prev_frame = pose_frame

        return frames

    def _hand_to_full_pose(self, hand_landmarks: list) -> dict:
        """
        Convert hand-only landmarks to a full pose frame.
        Places the hand at signing position (chest height, right side).
        """
        return {
            "pose": [
                [0.0, -0.5, 0.0],     # nose
                [0.03, -0.52, 0.0],    # right eye
                [-0.03, -0.52, 0.0],   # left eye
                [0.05, -0.52, 0.0],    # right ear
                [-0.05, -0.52, 0.0],   # left ear
                [0.02, -0.46, 0.0],    # mouth R
                [-0.02, -0.46, 0.0],   # mouth L
                [0.15, -0.35, 0.0],    # R shoulder
                [-0.15, -0.35, 0.0],   # L shoulder
                [0.18, -0.25, 0.05],   # R elbow
                [-0.20, -0.20, 0.0],   # L elbow
                [0.12, -0.30, 0.10],   # R wrist (at signing position)
                [-0.20, -0.05, 0.0],   # L wrist
                [0.14, -0.28, 0.10],   # R hand
                [-0.22, 0.0, 0.0],     # L hand
                [0.10, 0.05, 0.0],     # R hip
                [-0.10, 0.05, 0.0],    # L hip
                [0.0, -0.55, 0.0],
                [0.0, -0.40, 0.0],
                [0.08, -0.35, 0.0],
                [-0.08, -0.35, 0.0],
                [0.0, -0.30, 0.0],
                [0.0, -0.25, 0.0],
                [0.25, -0.20, 0.0],
                [-0.25, -0.20, 0.0],
            ],
            "right_hand": hand_landmarks if isinstance(hand_landmarks, list) else [],
            "left_hand": [],
        }

    def _lookup_pronoun_sign(self, token: str) -> list[dict]:
        """Generate pronoun/index pointing signs."""
        # Determine pointing direction based on pronoun type
        if "1" in token:
            # Point to self
            target_x, target_y = 0.0, -0.25
        elif "2" in token:
            # Point forward (to addressee)
            target_x, target_y = 0.05, -0.30
        else:
            # Point to side (third person)
            target_x, target_y = 0.20, -0.30

        is_plural = "PLURAL" in token
        is_possessive = "POSS" in token

        frames = []
        for t in range(5):
            progress = t / 4.0
            frame = {
                "pose": [
                    [0.0, -0.5, 0.0],
                    [0.03, -0.52, 0.0],
                    [-0.03, -0.52, 0.0],
                    [0.05, -0.52, 0.0],
                    [-0.05, -0.52, 0.0],
                    [0.02, -0.46, 0.0],
                    [-0.02, -0.46, 0.0],
                    [0.15, -0.35, 0.0],
                    [-0.15, -0.35, 0.0],
                    [0.15 + target_x * progress * 0.3, -0.30, 0.05 * progress],
                    [-0.20, -0.20, 0.0],
                    [target_x + 0.10 * (1 - progress), target_y, 0.10 * progress],
                    [-0.20, -0.05, 0.0],
                    [target_x + 0.12 * (1 - progress), target_y + 0.02, 0.10 * progress],
                    [-0.22, 0.0, 0.0],
                    [0.10, 0.05, 0.0],
                    [-0.10, 0.05, 0.0],
                    [0.0, -0.55, 0.0],
                    [0.0, -0.40, 0.0],
                    [0.08, -0.35, 0.0],
                    [-0.08, -0.35, 0.0],
                    [0.0, -0.30, 0.0],
                    [0.0, -0.25, 0.0],
                    [0.25, -0.20, 0.0],
                    [-0.25, -0.20, 0.0],
                ],
                "right_hand": self._get_pointing_hand(is_possessive),
                "left_hand": [],
            }
            frames.append(frame)

        return frames

    def _get_pointing_hand(self, is_possessive: bool = False) -> list:
        """Get hand landmarks for pointing or possessive gesture."""
        if is_possessive:
            # Flat hand (palm toward self/target)
            return [
                [0.0, 0.0, 0.0],
                [-0.04, 0.02, 0.02], [-0.06, 0.04, 0.02],
                [-0.05, 0.06, 0.01], [-0.03, 0.07, 0.0],
                [-0.02, 0.08, 0.0], [-0.02, 0.12, 0.0],
                [-0.02, 0.15, 0.0], [-0.02, 0.17, 0.0],
                [0.0, 0.08, 0.0], [0.0, 0.12, 0.0],
                [0.0, 0.15, 0.0], [0.0, 0.17, 0.0],
                [0.02, 0.08, 0.0], [0.02, 0.12, 0.0],
                [0.02, 0.15, 0.0], [0.02, 0.17, 0.0],
                [0.04, 0.07, 0.0], [0.04, 0.11, 0.0],
                [0.04, 0.14, 0.0], [0.04, 0.16, 0.0],
            ]
        else:
            # Index finger point
            return [
                [0.0, 0.0, 0.0],
                [-0.04, 0.02, 0.02], [-0.06, 0.04, 0.03],
                [-0.05, 0.06, 0.02], [-0.04, 0.07, 0.01],
                [-0.02, 0.08, 0.0], [-0.02, 0.12, 0.0],
                [-0.02, 0.15, 0.0], [-0.02, 0.17, 0.0],
                [0.0, 0.08, 0.0], [0.0, 0.10, -0.02],
                [0.0, 0.09, -0.03], [0.0, 0.08, -0.02],
                [0.02, 0.08, 0.0], [0.02, 0.10, -0.02],
                [0.02, 0.09, -0.03], [0.02, 0.08, -0.02],
                [0.04, 0.07, 0.0], [0.04, 0.09, -0.02],
                [0.04, 0.08, -0.03], [0.04, 0.07, -0.02],
            ]

    def get_available_signs(self) -> list[str]:
        """Return list of all available sign glosses."""
        return sorted(list(self.signs.keys()))

    def get_available_letters(self) -> list[str]:
        """Return list of available fingerspelling letters."""
        return sorted(list(self.alphabet.keys()))

    def has_sign(self, token: str) -> bool:
        """Check if a sign is in the library."""
        return token.upper() in self.signs
