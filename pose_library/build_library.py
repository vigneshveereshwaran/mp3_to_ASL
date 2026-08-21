"""
HearLink ASL — Pose Library Builder
Aggregates extracted keypoints and builds a sign → pose sequence mapping.
Includes a built-in fingerspelling alphabet library.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SIGNS_DIR = PROJECT_ROOT / "pose_library" / "signs"
MANIFEST_PATH = SIGNS_DIR / "manifest.json"


# ──────────────────────────────────────────────────
# Built-in Fingerspelling Alphabet
# Each letter is a simplified static hand pose:
# 21 landmarks × 3 coordinates (x, y, z) relative to wrist
# ──────────────────────────────────────────────────
def _hand_fist():
    """Closed fist base (used for several letters)."""
    return [
        [0.0, 0.0, 0.0],    # wrist
        # Thumb
        [-0.04, 0.02, 0.02], [-0.06, 0.04, 0.03],
        [-0.05, 0.06, 0.02], [-0.04, 0.07, 0.01],
        # Index
        [-0.02, 0.08, 0.0], [-0.02, 0.10, -0.02],
        [-0.02, 0.09, -0.03], [-0.02, 0.08, -0.02],
        # Middle
        [0.0, 0.08, 0.0], [0.0, 0.10, -0.02],
        [0.0, 0.09, -0.03], [0.0, 0.08, -0.02],
        # Ring
        [0.02, 0.08, 0.0], [0.02, 0.10, -0.02],
        [0.02, 0.09, -0.03], [0.02, 0.08, -0.02],
        # Pinky
        [0.04, 0.07, 0.0], [0.04, 0.09, -0.02],
        [0.04, 0.08, -0.03], [0.04, 0.07, -0.02],
    ]


def _hand_flat():
    """Flat open hand (used for B, etc.)."""
    return [
        [0.0, 0.0, 0.0],    # wrist
        # Thumb (tucked across palm)
        [-0.04, 0.02, 0.02], [-0.06, 0.04, 0.02],
        [-0.05, 0.06, 0.01], [-0.03, 0.07, 0.0],
        # Index (straight up)
        [-0.02, 0.08, 0.0], [-0.02, 0.12, 0.0],
        [-0.02, 0.15, 0.0], [-0.02, 0.17, 0.0],
        # Middle
        [0.0, 0.08, 0.0], [0.0, 0.12, 0.0],
        [0.0, 0.15, 0.0], [0.0, 0.17, 0.0],
        # Ring
        [0.02, 0.08, 0.0], [0.02, 0.12, 0.0],
        [0.02, 0.15, 0.0], [0.02, 0.17, 0.0],
        # Pinky
        [0.04, 0.07, 0.0], [0.04, 0.11, 0.0],
        [0.04, 0.14, 0.0], [0.04, 0.16, 0.0],
    ]


def _hand_point():
    """Pointing index finger (used for D, G, etc.)."""
    base = _hand_fist()
    # Extend index finger
    base[5] = [-0.02, 0.08, 0.0]
    base[6] = [-0.02, 0.12, 0.0]
    base[7] = [-0.02, 0.15, 0.0]
    base[8] = [-0.02, 0.17, 0.0]
    return base


def generate_fingerspelling_alphabet() -> dict:
    """
    Generate a simplified fingerspelling alphabet.
    Returns dict mapping letter → list of hand landmark frames.
    Each sign is a single static pose (1 frame).
    """
    alphabet = {}

    # A - fist with thumb to side
    a = _hand_fist()
    a[1] = [-0.05, 0.04, 0.03]  # Thumb sticks out to side
    alphabet['A'] = [a]

    # B - flat hand, fingers up, thumb tucked
    alphabet['B'] = [_hand_flat()]

    # C - curved hand (like holding a ball)
    c = _hand_flat()
    for i in range(5, 21):
        c[i][2] = 0.03  # Curve fingers slightly forward
    alphabet['C'] = [c]

    # D - index up, others curved to thumb
    alphabet['D'] = [_hand_point()]

    # E - fingers curled down, thumb across
    e = _hand_fist()
    for i in [5, 9, 13, 17]:
        e[i][1] += 0.02  # Slightly more open than fist
    alphabet['E'] = [e]

    # F - OK sign (index & thumb circle, others up)
    f = _hand_flat()
    f[5] = [-0.03, 0.07, 0.01]  # Index tip to thumb
    f[8] = [-0.04, 0.06, 0.02]
    alphabet['F'] = [f]

    # G - pointing sideways
    g = _hand_point()
    for i in range(5, 9):
        g[i][0] -= 0.05  # Point index to side
    alphabet['G'] = [g]

    # H - index and middle pointing sideways
    h = _hand_point()
    h[9] = [-0.01, 0.08, 0.0]
    h[10] = [-0.01, 0.12, 0.0]
    h[11] = [-0.01, 0.15, 0.0]
    h[12] = [-0.01, 0.17, 0.0]
    alphabet['H'] = [h]

    # I - pinky up, others fist
    i_sign = _hand_fist()
    i_sign[17] = [0.04, 0.07, 0.0]
    i_sign[18] = [0.04, 0.11, 0.0]
    i_sign[19] = [0.04, 0.14, 0.0]
    i_sign[20] = [0.04, 0.16, 0.0]
    alphabet['I'] = [i_sign]

    # J - I + downward motion (2 frames)
    j1 = [row[:] for row in i_sign]
    j2 = [row[:] for row in i_sign]
    for k in range(17, 21):
        j2[k][1] -= 0.05  # Move pinky down
        j2[k][0] += 0.02  # Slight J curve
    alphabet['J'] = [j1, j2]

    # K - index up, middle angled, thumb between
    k_sign = _hand_point()
    k_sign[9] = [0.0, 0.08, 0.0]
    k_sign[10] = [0.01, 0.11, 0.01]
    k_sign[11] = [0.02, 0.13, 0.02]
    k_sign[12] = [0.03, 0.14, 0.02]
    alphabet['K'] = [k_sign]

    # L - L shape (index up, thumb out)
    l_sign = _hand_fist()
    l_sign[5] = [-0.02, 0.08, 0.0]
    l_sign[6] = [-0.02, 0.12, 0.0]
    l_sign[7] = [-0.02, 0.15, 0.0]
    l_sign[8] = [-0.02, 0.17, 0.0]
    l_sign[1] = [-0.06, 0.02, 0.0]
    l_sign[2] = [-0.09, 0.03, 0.0]
    l_sign[3] = [-0.11, 0.03, 0.0]
    l_sign[4] = [-0.13, 0.03, 0.0]
    alphabet['L'] = [l_sign]

    # M - three fingers over thumb
    m = _hand_fist()
    m[3] = [-0.03, 0.07, 0.03]
    m[4] = [-0.02, 0.08, 0.02]
    alphabet['M'] = [m]

    # N - two fingers over thumb
    n = _hand_fist()
    n[3] = [-0.02, 0.07, 0.03]
    n[4] = [-0.01, 0.08, 0.02]
    alphabet['N'] = [n]

    # O - fingertips together in O shape
    o = _hand_flat()
    for i in [8, 12, 16, 20]:
        o[i] = [0.0, 0.10, 0.03]  # All fingertips meet
    o[4] = [0.0, 0.10, 0.03]  # Thumb tip also meets
    alphabet['O'] = [o]

    # P - K rotated down
    p = [row[:] for row in k_sign]
    for i in range(len(p)):
        p[i][1] -= 0.05  # Point downward
    alphabet['P'] = [p]

    # Q - G rotated down
    q = [row[:] for row in g]
    for i in range(len(q)):
        q[i][1] -= 0.05
    alphabet['Q'] = [q]

    # R - crossed index and middle
    r = _hand_fist()
    r[5] = [-0.02, 0.08, 0.0]
    r[6] = [-0.01, 0.12, 0.0]
    r[7] = [0.0, 0.15, 0.0]
    r[8] = [0.01, 0.17, 0.0]
    r[9] = [0.0, 0.08, 0.0]
    r[10] = [-0.01, 0.12, 0.0]
    r[11] = [-0.02, 0.15, 0.0]
    r[12] = [-0.03, 0.17, 0.0]
    alphabet['R'] = [r]

    # S - fist with thumb across fingers
    s = _hand_fist()
    s[3] = [-0.03, 0.06, 0.03]
    s[4] = [-0.01, 0.07, 0.03]
    alphabet['S'] = [s]

    # T - thumb between index and middle
    t = _hand_fist()
    t[3] = [-0.02, 0.07, 0.03]
    t[4] = [-0.01, 0.08, 0.03]
    alphabet['T'] = [t]

    # U - index and middle up together
    u = _hand_fist()
    u[5] = [-0.01, 0.08, 0.0]
    u[6] = [-0.01, 0.12, 0.0]
    u[7] = [-0.01, 0.15, 0.0]
    u[8] = [-0.01, 0.17, 0.0]
    u[9] = [0.01, 0.08, 0.0]
    u[10] = [0.01, 0.12, 0.0]
    u[11] = [0.01, 0.15, 0.0]
    u[12] = [0.01, 0.17, 0.0]
    alphabet['U'] = [u]

    # V - peace sign (index and middle spread)
    v = _hand_fist()
    v[5] = [-0.03, 0.08, 0.0]
    v[6] = [-0.03, 0.12, 0.0]
    v[7] = [-0.03, 0.15, 0.0]
    v[8] = [-0.04, 0.17, 0.0]
    v[9] = [0.01, 0.08, 0.0]
    v[10] = [0.01, 0.12, 0.0]
    v[11] = [0.01, 0.15, 0.0]
    v[12] = [0.02, 0.17, 0.0]
    alphabet['V'] = [v]

    # W - index, middle, ring spread
    w = _hand_fist()
    for idx, offset_x in zip([5,6,7,8], [-0.04,-0.04,-0.04,-0.05]):
        w[idx] = [offset_x, 0.08 + (idx-5)*0.03, 0.0]
    for idx, offset_x in zip([9,10,11,12], [0.0,0.0,0.0,0.0]):
        w[idx] = [offset_x, 0.08 + (idx-9)*0.03, 0.0]
    for idx, offset_x in zip([13,14,15,16], [0.04,0.04,0.04,0.05]):
        w[idx] = [offset_x, 0.08 + (idx-13)*0.03, 0.0]
    alphabet['W'] = [w]

    # X - index finger hooked
    x = _hand_fist()
    x[5] = [-0.02, 0.08, 0.0]
    x[6] = [-0.02, 0.12, 0.0]
    x[7] = [-0.02, 0.11, -0.02]
    x[8] = [-0.02, 0.10, -0.03]
    alphabet['X'] = [x]

    # Y - thumb and pinky out (shaka)
    y = _hand_fist()
    y[1] = [-0.06, 0.02, 0.0]
    y[2] = [-0.09, 0.03, 0.0]
    y[3] = [-0.11, 0.03, 0.0]
    y[4] = [-0.13, 0.03, 0.0]
    y[17] = [0.04, 0.07, 0.0]
    y[18] = [0.04, 0.11, 0.0]
    y[19] = [0.04, 0.14, 0.0]
    y[20] = [0.04, 0.16, 0.0]
    alphabet['Y'] = [y]

    # Z - index draws Z shape (3 frames)
    z1 = [row[:] for row in _hand_point()]
    z2 = [row[:] for row in z1]
    z3 = [row[:] for row in z1]
    z2[8] = [0.02, 0.14, 0.0]   # Move right
    z3[8] = [-0.02, 0.12, 0.0]  # Move left-down
    alphabet['Z'] = [z1, z2, z3]

    return alphabet


def generate_common_signs() -> dict:
    """
    Generate simplified pose data for common ASL signs.
    Each sign has a sequence of upper-body + hand frames.
    Format: {"pose": [...], "right_hand": [...], "left_hand": [...]}
    """
    # Neutral rest pose (arms at sides)
    def neutral_pose():
        return {
            "pose": [
                # Head/face (simplified upper body)
                [0.0, -0.5, 0.0],    # nose
                [0.03, -0.52, 0.0],   # right eye
                [-0.03, -0.52, 0.0],  # left eye
                [0.05, -0.52, 0.0],   # right ear
                [-0.05, -0.52, 0.0],  # left ear
                [0.02, -0.46, 0.0],   # right mouth
                [-0.02, -0.46, 0.0],  # left mouth
                # Shoulders
                [0.15, -0.35, 0.0],   # right shoulder (7)
                [-0.15, -0.35, 0.0],  # left shoulder (8)
                # Arms
                [0.20, -0.20, 0.0],   # right elbow (9)
                [-0.20, -0.20, 0.0],  # left elbow (10)
                [0.20, -0.05, 0.0],   # right wrist (11)
                [-0.20, -0.05, 0.0],  # left wrist (12)
                # Hands (simplified)
                [0.22, 0.0, 0.0],     # right hand (13)
                [-0.22, 0.0, 0.0],    # left hand (14)
                # Hips
                [0.10, 0.05, 0.0],    # right hip (15)
                [-0.10, 0.05, 0.0],   # left hip (16)
                # Remaining upper body markers
                [0.0, -0.55, 0.0],    # forehead (17)
                [0.0, -0.40, 0.0],    # chin (18)
                [0.08, -0.35, 0.0],   # right clavicle (19)
                [-0.08, -0.35, 0.0],  # left clavicle (20)
                [0.0, -0.30, 0.0],    # sternum (21)
                [0.0, -0.25, 0.0],    # chest (22)
                [0.25, -0.20, 0.0],   # right forearm (23)
                [-0.25, -0.20, 0.0],  # left forearm (24)
            ],
            "right_hand": _hand_flat(),
            "left_hand": _hand_flat(),
        }

    signs = {}

    # HELLO - open hand wave near head
    hello = []
    for t in range(8):
        frame = neutral_pose()
        wave = 0.03 * np.sin(t * np.pi / 2)
        frame["pose"][11] = [0.15 + wave, -0.40, 0.05]  # Right wrist near face
        frame["pose"][9] = [0.15, -0.35, 0.03]   # Elbow up
        frame["right_hand"] = _hand_flat()
        hello.append(frame)
    signs["HELLO"] = hello

    # THANK-YOU - flat hand from chin outward
    thank = []
    for t in range(6):
        frame = neutral_pose()
        progress = t / 5.0
        frame["pose"][11] = [0.05 + progress * 0.15, -0.45 + progress * 0.15, 0.05 - progress * 0.05]
        frame["right_hand"] = _hand_flat()
        thank.append(frame)
    signs["THANK-YOU"] = thank

    # YES - fist nod (S handshape nodding)
    yes_frames = []
    for t in range(6):
        frame = neutral_pose()
        nod = 0.03 * np.sin(t * np.pi / 2)
        frame["pose"][11] = [0.15, -0.30 + nod, 0.05]
        frame["right_hand"] = _hand_fist()
        yes_frames.append(frame)
    signs["YES"] = yes_frames

    # NO - index+middle snap to thumb
    no_frames = []
    for t in range(4):
        frame = neutral_pose()
        snap = 0.02 if t % 2 == 0 else 0.0
        frame["pose"][11] = [0.15, -0.30, 0.05]
        hand = _hand_point()
        hand[9] = [0.0, 0.08, 0.0]
        hand[10] = [0.0 + snap, 0.12, 0.0]
        frame["right_hand"] = hand
        no_frames.append(frame)
    signs["NO"] = no_frames

    # LIKE - pull from chest outward
    like_frames = []
    for t in range(6):
        frame = neutral_pose()
        progress = t / 5.0
        frame["pose"][11] = [0.05 + progress * 0.10, -0.25, 0.05 - progress * 0.05]
        frame["right_hand"] = _hand_flat()
        # Middle finger and thumb pinch
        frame["right_hand"][4] = [-0.01, 0.10, 0.01]
        frame["right_hand"][12] = [0.01, 0.10, 0.01]
        like_frames.append(frame)
    signs["LIKE"] = like_frames

    # WANT - both hands pull toward body (claw shape)
    want_frames = []
    for t in range(6):
        frame = neutral_pose()
        progress = t / 5.0
        frame["pose"][11] = [0.15, -0.20 - progress * 0.05, 0.10 - progress * 0.10]
        frame["pose"][12] = [-0.15, -0.20 - progress * 0.05, 0.10 - progress * 0.10]
        claw = _hand_flat()
        for i in [7, 8, 11, 12, 15, 16, 19, 20]:
            claw[i][2] = -0.02  # Curl fingers slightly
        frame["right_hand"] = claw
        frame["left_hand"] = claw
        want_frames.append(frame)
    signs["WANT"] = want_frames

    # UNDERSTAND - fist near forehead, index flicks up
    understand_frames = []
    for t in range(5):
        frame = neutral_pose()
        frame["pose"][11] = [0.08, -0.50, 0.05]  # Near forehead
        progress = t / 4.0
        hand = _hand_fist()
        # Flick index up
        hand[6] = [-0.02, 0.08 + progress * 0.06, 0.0]
        hand[7] = [-0.02, 0.10 + progress * 0.06, 0.0]
        hand[8] = [-0.02, 0.12 + progress * 0.06, 0.0]
        frame["right_hand"] = hand
        understand_frames.append(frame)
    signs["UNDERSTAND"] = understand_frames

    # NAME - H handshape taps on other H
    name_frames = []
    for t in range(6):
        frame = neutral_pose()
        tap = 0.02 if t % 2 == 0 else 0.0
        frame["pose"][11] = [0.05, -0.30 + tap, 0.05]
        frame["pose"][12] = [-0.05, -0.30, 0.05]
        frame["right_hand"] = _hand_point()
        frame["right_hand"][9] = [0.01, 0.08, 0.0]
        frame["right_hand"][10] = [0.01, 0.12, 0.0]
        frame["left_hand"] = _hand_point()
        frame["left_hand"][9] = [-0.01, 0.08, 0.0]
        frame["left_hand"][10] = [-0.01, 0.12, 0.0]
        name_frames.append(frame)
    signs["NAME"] = name_frames

    # Additional common signs with simplified gestures
    simple_signs = [
        "GOOD", "BAD", "PLEASE", "SORRY", "HELP", "STOP", "GO",
        "COME", "EAT", "DRINK", "SLEEP", "WORK", "PLAY", "LEARN",
        "KNOW", "SEE", "HEAR", "FEEL", "THINK", "LOVE", "HAPPY",
        "SAD", "ANGRY", "TIRED", "HUNGRY", "SCHOOL", "HOME",
        "FAMILY", "FRIEND", "WATER", "FOOD", "BOOK", "CAR",
        "MORNING", "NIGHT", "TODAY", "TOMORROW", "YESTERDAY",
        "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO",
        "NOT", "CAN", "WILL", "FINISH", "START",
    ]

    for sign_name in simple_signs:
        if sign_name not in signs:
            # Generate a basic gesture (varied wrist position for each sign)
            frames = []
            # Use hash of name to generate deterministic but varied positions
            h = int(hashlib.md5(sign_name.encode()).hexdigest()[:8], 16)
            x_offset = ((h % 100) / 100.0 - 0.5) * 0.2
            y_offset = ((h % 1000) / 1000.0 - 0.5) * 0.3

            for t in range(5):
                frame = neutral_pose()
                progress = t / 4.0
                # Animate right hand from rest to sign position
                frame["pose"][11] = [
                    0.10 + x_offset * progress,
                    -0.25 + y_offset * progress,
                    0.05 * progress
                ]
                frame["right_hand"] = _hand_flat() if h % 3 == 0 else (
                    _hand_fist() if h % 3 == 1 else _hand_point()
                )
                frames.append(frame)
            signs[sign_name] = frames

    return signs


import hashlib


def build_library(output_dir: str = None, include_alphabet: bool = True,
                  include_common_signs: bool = True):
    """
    Build the complete pose library.

    Args:
        output_dir: Directory to save sign JSON files
        include_alphabet: Include fingerspelling alphabet
        include_common_signs: Include common sign poses
    """
    if output_dir is None:
        output_dir = str(SIGNS_DIR)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {"signs": {}, "alphabet": {}}
    total = 0

    # Fingerspelling alphabet
    if include_alphabet:
        print("Generating fingerspelling alphabet...")
        alphabet = generate_fingerspelling_alphabet()
        for letter, frames in alphabet.items():
            sign_data = {
                "gloss": f"FS-{letter}",
                "type": "fingerspelling",
                "frames": frames,
                "num_frames": len(frames),
            }
            filename = f"fs_{letter.lower()}.json"
            with open(output_path / filename, 'w') as f:
                json.dump(sign_data, f, indent=2)
            manifest["alphabet"][letter] = filename
            total += 1
        print(f"  Generated {len(alphabet)} fingerspelling signs")

    # Common signs
    if include_common_signs:
        print("Generating common sign poses...")
        signs = generate_common_signs()
        for sign_name, frames in signs.items():
            sign_data = {
                "gloss": sign_name,
                "type": "sign",
                "frames": [
                    {
                        "pose": f.get("pose", []),
                        "right_hand": f.get("right_hand", []),
                        "left_hand": f.get("left_hand", []),
                    }
                    for f in frames
                ],
                "num_frames": len(frames),
            }
            filename = f"{sign_name.lower().replace('-', '_')}.json"
            with open(output_path / filename, 'w') as f:
                json.dump(sign_data, f, indent=2)
            manifest["signs"][sign_name] = filename
            total += 1
        print(f"  Generated {len(signs)} common signs")

    # Save manifest
    manifest["total_signs"] = len(manifest["signs"])
    manifest["total_alphabet"] = len(manifest["alphabet"])
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SUCCESS] Pose library built: {total} total entries -> {output_path}")
    return manifest


if __name__ == "__main__":
    build_library()
