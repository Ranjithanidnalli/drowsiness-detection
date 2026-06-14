"""
utils.py — Geometric feature helpers for drowsiness detection.
All functions are pure numpy; no OpenCV/MediaPipe imports needed here.
"""

import time
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------

# Left eye  (6 points forming the eye contour)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158,  133, 153, 144]

# Iris centres (MediaPipe gives 5 iris landmarks per eye; index 0 is centre)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# Mouth outer landmarks for MAR (mouth aspect ratio / yawn detection)
# Top lip: 13, bottom lip: 14, left corner: 78, right corner: 308
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

# Head-pose reference points (3D model → solvePnP)
HEAD_POSE_LANDMARKS = [1, 9, 57, 130, 287, 359]  # nose, chin, L/R eye corners, L/R mouth corners


# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    eye_points: (6, 2) array of (x, y) pixel coords in order
    [p1=outer-corner, p2=upper-outer, p3=upper-inner,
     p4=inner-corner, p5=lower-inner, p6=lower-outer]
    Returns EAR scalar.
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h  = np.linalg.norm(eye_points[0] - eye_points[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def get_ear(landmarks, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """
    Extract left EAR, right EAR, and mean EAR from MediaPipe landmarks.
    landmarks: mediapipe NormalizedLandmarkList
    Returns (left_ear, right_ear, mean_ear)
    """
    def lm_to_xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    left_pts  = np.array([lm_to_xy(i) for i in LEFT_EYE])
    right_pts = np.array([lm_to_xy(i) for i in RIGHT_EYE])

    left_ear  = eye_aspect_ratio(left_pts)
    right_ear = eye_aspect_ratio(right_pts)
    return left_ear, right_ear, (left_ear + right_ear) / 2.0


# ---------------------------------------------------------------------------
# Mouth Aspect Ratio (MAR) — yawn detection
# ---------------------------------------------------------------------------

def mouth_aspect_ratio(landmarks, frame_w: int, frame_h: int) -> float:
    """
    Returns MAR. Values > ~0.6 typically indicate a yawn.
    """
    def lm_to_xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    top    = lm_to_xy(MOUTH_TOP)
    bottom = lm_to_xy(MOUTH_BOTTOM)
    left   = lm_to_xy(MOUTH_LEFT)
    right  = lm_to_xy(MOUTH_RIGHT)

    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-6)


# ---------------------------------------------------------------------------
# Head-pose estimation (solvePnP)
# ---------------------------------------------------------------------------

# Generic 3-D model face points (mm, arbitrary scale)
MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0  ],   # Nose tip (landmark 1)
    [0.0,  -330.0, -65.0],   # Chin     (landmark 9)
    [-225.0, 170.0, -135.0], # Left eye left corner  (landmark 33 proxy)
    [225.0,  170.0, -135.0], # Right eye right corner (landmark 263 proxy)
    [-150.0,-150.0, -125.0], # Left mouth corner (landmark 57)
    [150.0, -150.0, -125.0], # Right mouth corner (landmark 287)
], dtype=np.float64)

_POSE_LM_IDX = [1, 9, 33, 263, 57, 287]


def head_pose_angles(landmarks, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """
    Returns (pitch, yaw, roll) in degrees.
    Positive pitch = head tilting down (nodding).
    """
    import cv2

    image_points = np.array([
        [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
        for i in _POSE_LM_IDX
    ], dtype=np.float64)

    focal_length = frame_w
    cam_matrix = np.array([
        [focal_length, 0,            frame_w / 2],
        [0,            focal_length, frame_h / 2],
        [0,            0,            1          ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Decompose rotation matrix → Euler angles
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
        yaw   = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        roll  = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))

    return pitch, yaw, roll


# ---------------------------------------------------------------------------
# PERCLOS rolling window  (time-based — independent of frame rate)
# ---------------------------------------------------------------------------

class PerclosTracker:
    """
    PERCLOS = percentage of time EAR is below threshold over a rolling *time*
    window (default 30 s).  Using wall-clock timestamps instead of a fixed
    frame count keeps the measure correct regardless of the actual camera FPS.
    """

    def __init__(self, window_seconds: float = 30.0, ear_threshold: float = 0.20):
        self.window_seconds = window_seconds
        self.threshold = ear_threshold
        self._buffer: deque = deque()   # (timestamp, eyes_closed_bool)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def update(self, ear: float) -> float:
        """Push one frame, returns current PERCLOS (0–1)."""
        now = time.time()
        self._buffer.append((now, ear < self.threshold))
        self._trim(now)
        return self.value

    @property
    def value(self) -> float:
        if not self._buffer:
            return 0.0
        closed = sum(1 for _, c in self._buffer if c)
        return closed / len(self._buffer)


# ---------------------------------------------------------------------------
# Eye-closure timer  (time-based)
# ---------------------------------------------------------------------------

class EyeStateTracker:
    """
    Tracks how long the eyes have been *continuously* closed, in real seconds.
    Fires `eyes_closed_event` for one frame when a closure first crosses
    `closed_seconds_threshold`.
    """

    def __init__(self, closed_seconds_threshold: float = 1.5, ear_threshold: float = 0.20):
        self.closed_seconds_threshold = closed_seconds_threshold
        self.ear_threshold = ear_threshold
        self._closed_since: float | None = None
        self.total_blinks = 0
        self.eyes_closed_event = False  # True for exactly 1 frame at threshold crossing

    def update(self, ear: float) -> None:
        now = time.time()
        prev = self.closed_seconds
        if ear < self.ear_threshold:
            if self._closed_since is None:
                self._closed_since = now
        else:
            if self._closed_since is not None and \
                    (now - self._closed_since) >= self.closed_seconds_threshold:
                self.total_blinks += 1
            self._closed_since = None
        cur = self.closed_seconds
        self.eyes_closed_event = (prev < self.closed_seconds_threshold <= cur)

    @property
    def closed_seconds(self) -> float:
        if self._closed_since is None:
            return 0.0
        return time.time() - self._closed_since
