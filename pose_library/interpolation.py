"""
HearLink ASL — Pose Interpolation (Python fallback)
Provides smooth interpolation between sign pose keypoints.
Used as a fallback when the WASM engine is not available.
"""

import math
from typing import Optional


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def lerp_vec3(a: list[float], b: list[float], t: float) -> list[float]:
    """Linear interpolation between two 3D vectors."""
    return [lerp(a[i], b[i], t) for i in range(3)]


def lerp_landmarks(landmarks_a: list[list[float]],
                   landmarks_b: list[list[float]],
                   t: float) -> list[list[float]]:
    """Linearly interpolate between two landmark arrays."""
    result = []
    max_len = max(len(landmarks_a), len(landmarks_b))
    for i in range(max_len):
        if i < len(landmarks_a) and i < len(landmarks_b):
            result.append(lerp_vec3(landmarks_a[i], landmarks_b[i], t))
        elif i < len(landmarks_a):
            result.append(landmarks_a[i][:])
        else:
            result.append(landmarks_b[i][:])
    return result


def slerp_quaternion(q1: list[float], q2: list[float], t: float) -> list[float]:
    """
    Spherical Linear Interpolation between two quaternions.

    Args:
        q1: Start quaternion [x, y, z, w]
        q2: End quaternion [x, y, z, w]
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion [x, y, z, w]
    """
    # Dot product
    dot = sum(a * b for a, b in zip(q1, q2))

    # Ensure shortest path
    if dot < 0:
        q2 = [-x for x in q2]
        dot = -dot

    # If very close, use LERP
    if dot > 0.9995:
        result = [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
        # Normalize
        mag = math.sqrt(sum(x * x for x in result))
        return [x / mag for x in result]

    # Standard SLERP
    theta_0 = math.acos(min(dot, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return [q1[i] * s1 + q2[i] * s2 for i in range(4)]


def cubic_spline_interpolate(p0: list[float], p1: list[float],
                              p2: list[float], p3: list[float],
                              t: float) -> list[float]:
    """
    Catmull-Rom cubic spline interpolation between p1 and p2.

    Args:
        p0: Point before the segment start
        p1: Segment start point
        p2: Segment end point
        p3: Point after the segment end
        t: Parameter [0, 1]

    Returns:
        Interpolated point
    """
    t2 = t * t
    t3 = t2 * t

    result = []
    for i in range(len(p1)):
        v = 0.5 * (
            (2 * p1[i]) +
            (-p0[i] + p2[i]) * t +
            (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 +
            (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3
        )
        result.append(v)

    return result


def ease_in_out(t: float) -> float:
    """Smooth ease-in-out easing function (cubic)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def generate_transition_frames(pose_a: dict, pose_b: dict,
                                num_frames: int = 5,
                                use_easing: bool = True) -> list[dict]:
    """
    Generate smooth transition frames between two sign poses.

    Args:
        pose_a: Last frame of first sign
        pose_b: First frame of second sign
        num_frames: Number of transition frames to generate
        use_easing: Apply ease-in-out easing

    Returns:
        List of interpolated pose frames
    """
    frames = []
    for i in range(num_frames):
        t = (i + 1) / (num_frames + 1)  # Exclude exact start/end
        if use_easing:
            t = ease_in_out(t)

        frame = {}
        for key in ["pose", "right_hand", "left_hand", "face"]:
            lm_a = pose_a.get(key, [])
            lm_b = pose_b.get(key, [])
            if lm_a and lm_b:
                frame[key] = lerp_landmarks(lm_a, lm_b, t)
            elif lm_a:
                frame[key] = lm_a
            elif lm_b:
                frame[key] = lm_b
            else:
                frame[key] = []

        frames.append(frame)

    return frames


def smooth_pose_sequence(frames: list[dict], window_size: int = 3) -> list[dict]:
    """
    Apply moving average smoothing to a pose sequence.

    Args:
        frames: List of pose frame dicts
        window_size: Size of the smoothing window (odd number)

    Returns:
        Smoothed pose sequence
    """
    if len(frames) <= window_size:
        return frames

    half = window_size // 2
    smoothed = []

    for i in range(len(frames)):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)
        window = frames[start:end]

        smooth_frame = {}
        for key in ["pose", "right_hand", "left_hand", "face"]:
            landmarks_list = [f.get(key, []) for f in window if f.get(key)]
            if landmarks_list:
                # Average landmarks across window
                num_landmarks = len(landmarks_list[0])
                avg = []
                for j in range(num_landmarks):
                    avg_point = [0.0, 0.0, 0.0]
                    count = 0
                    for lms in landmarks_list:
                        if j < len(lms):
                            for k in range(3):
                                avg_point[k] += lms[j][k]
                            count += 1
                    if count > 0:
                        avg.append([x / count for x in avg_point])
                    else:
                        avg.append([0.0, 0.0, 0.0])
                smooth_frame[key] = avg
            else:
                smooth_frame[key] = frames[i].get(key, [])

        smoothed.append(smooth_frame)

    return smoothed


if __name__ == "__main__":
    # Quick test
    q1 = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
    q2 = [0.0, 0.707, 0.0, 0.707]  # 90° around Y

    print("SLERP Test:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = slerp_quaternion(q1, q2, t)
        print(f"  t={t:.2f}: {[f'{x:.4f}' for x in result]}")

    print("\nEasing Test:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  t={t:.2f}: eased={ease_in_out(t):.4f}")
