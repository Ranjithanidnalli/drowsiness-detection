"""
dashboard.py — OpenCV overlay dashboard.

Draws onto the webcam frame:
  • EAR / MAR signal bars (left panel)
  • Drowsiness gauge / level badge (top right)
  • Head-pose readout
  • PERCLOS meter
  • "Why it fired" explanation panel (bottom)
"""

import cv2
import numpy as np
from collections import deque

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
C_GREEN   = (50, 205, 50)
C_YELLOW  = (0, 200, 255)
C_RED     = (0, 50, 220)
C_WHITE   = (255, 255, 255)
C_DARK    = (20, 20, 20)
C_PANEL   = (40, 40, 40)
C_CYAN    = (255, 200, 0)

LEVEL_COLORS = {0: C_GREEN, 1: C_YELLOW, 2: C_RED}
LEVEL_LABELS = {0: "ALERT", 1: "MILD DROWSY", 2: "DROWSY!"}

HISTORY_LEN = 150  # frames to show in signal plot (~5 s @ 30 fps)


class Dashboard:
    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

        # Signal history buffers
        self.ear_hist  = deque([0.3] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.mar_hist  = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.perc_hist = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    # ── Public entry point ────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        ear: float,
        mar: float,
        perclos: float,
        pitch: float,
        yaw: float,
        roll: float,
        level: int,          # 0/1/2
        reason: str,         # "why it fired" text
        cnn_prob: float = -1.0,   # Model A closed probability (-1 = not loaded)
        model_b_score: float = -1.0,
        yawning: bool = False,
        phone: bool = False,
        phone_box: tuple = None,
    ) -> np.ndarray:
        """Draw all overlays onto frame (in-place) and return it."""
        self.ear_hist.append(ear)
        self.mar_hist.append(mar)
        self.perc_hist.append(perclos)

        overlay = frame.copy()

        # Left side panel background
        cv2.rectangle(overlay, (0, 0), (220, self.fh), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        self._draw_signal_plot(frame, list(self.ear_hist),  label="EAR",     y0=10,  color=C_CYAN,   threshold=0.20)
        self._draw_signal_plot(frame, list(self.mar_hist),  label="MAR",     y0=115, color=C_YELLOW,  threshold=0.60)
        self._draw_signal_plot(frame, list(self.perc_hist), label="PERCLOS", y0=220, color=C_RED,     threshold=0.30)

        self._draw_head_pose(frame, pitch, yaw, roll, y0=330)
        self._draw_stats(frame, ear, mar, perclos, cnn_prob, model_b_score, y0=440)

        # Right: drowsiness gauge
        self._draw_gauge(frame, level, model_b_score)

        # Yawn badge (over the video, top-centre)
        if yawning:
            self._draw_yawn_badge(frame)

        # Phone-use distraction badge + box
        if phone:
            self._draw_phone_badge(frame, phone_box)

        # Bottom: reason banner
        if reason:
            self._draw_reason(frame, reason, level)

        return frame

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_signal_plot(
        self,
        frame: np.ndarray,
        values: list,
        label: str,
        y0: int,
        color: tuple,
        threshold: float,
    ) -> None:
        x0, x1 = 5, 215
        y1 = y0 + 85
        plot_h = y1 - y0 - 20
        plot_w = x1 - x0 - 5

        cv2.putText(frame, label, (x0 + 2, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

        # Scale values to plot height
        max_val = max(max(values) if values else 1.0, threshold * 1.5, 0.01)
        pts = []
        for i, v in enumerate(values):
            px = x0 + int(i / len(values) * plot_w)
            py = y0 + 18 + int((1 - v / max_val) * plot_h)
            pts.append((px, py))

        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, color, 1, cv2.LINE_AA)

        # Threshold line
        thresh_y = y0 + 18 + int((1 - threshold / max_val) * plot_h)
        cv2.line(frame, (x0, thresh_y), (x1, thresh_y), C_RED, 1)

        # Current value
        cur = values[-1] if values else 0.0
        cv2.putText(frame, f"{cur:.3f}", (x1 - 55, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    def _draw_head_pose(self, frame, pitch, yaw, roll, y0):
        cv2.putText(frame, "HEAD POSE", (5, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1, cv2.LINE_AA)
        for i, (name, val) in enumerate([("Pitch", pitch), ("Yaw", yaw), ("Roll", roll)]):
            color = C_RED if (name == "Pitch" and val > 15) else C_WHITE
            cv2.putText(frame, f"{name}: {val:+.1f}°", (5, y0 + 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _draw_stats(self, frame, ear, mar, perclos, cnn_prob, model_b_score, y0):
        cv2.putText(frame, "METRICS", (5, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1, cv2.LINE_AA)
        lines = [
            f"EAR:     {ear:.3f}",
            f"MAR:     {mar:.3f}",
            f"PERCLOS: {perclos*100:.1f}%",
        ]
        if cnn_prob >= 0:
            lines.append(f"CNN:     {cnn_prob:.2f}")
        if model_b_score >= 0:
            lines.append(f"Model-B: {model_b_score:.2f}")

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (5, y0 + 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1, cv2.LINE_AA)

    def _draw_gauge(self, frame, level: int, score: float) -> None:
        """Circular-ish gauge in top-right corner."""
        cx = self.fw - 80
        cy = 80
        r  = 60

        color = LEVEL_COLORS[level]
        label = LEVEL_LABELS[level]

        # Outer ring
        cv2.circle(frame, (cx, cy), r, color, 4)
        # Fill arc proportional to score (0–1)
        fill = max(score, 0) if score >= 0 else (level / 2.0)
        angle = int(fill * 270)
        cv2.ellipse(frame, (cx, cy), (r, r), -135, 0, angle, color, -1)
        # Dark centre
        cv2.circle(frame, (cx, cy), r - 12, C_DARK, -1)
        # Level text
        cv2.putText(frame, label,
                    (cx - len(label) * 4, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _draw_yawn_badge(self, frame) -> None:
        """Prominent badge shown over the video feed when a yawn is detected."""
        text = "YAWN"
        bw, bh = 130, 46
        bx = 220 + (self.fw - 220) // 2 - bw // 2
        by = 36
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_DARK, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_YELLOW, 2)
        cv2.putText(frame, text, (bx + 16, by + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2, cv2.LINE_AA)

    def _draw_phone_badge(self, frame, box) -> None:
        """Red distraction badge + bounding box when a phone is detected."""
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_RED, 2)
            cv2.putText(frame, "PHONE", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_RED, 1, cv2.LINE_AA)

        text = "PHONE USE"
        bw, bh = 170, 46
        bx = 220 + (self.fw - 220) // 2 - bw // 2
        by = 86
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_DARK, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_RED, 2)
        cv2.putText(frame, text, (bx + 14, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)

    def _draw_reason(self, frame, reason: str, level: int) -> None:
        """Bottom banner showing explanation."""
        banner_h = 38
        y0 = self.fh - banner_h
        color = LEVEL_COLORS[level]
        cv2.rectangle(frame, (220, y0), (self.fw, self.fh), C_DARK, -1)
        cv2.rectangle(frame, (220, y0), (self.fw, self.fh), color, 2)
        cv2.putText(frame, f"WHY: {reason}",
                    (230, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
