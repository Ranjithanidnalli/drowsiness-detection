"""
phone_detector.py — Distraction detection: phone use while driving.

Runs a YOLOv8-nano detector (COCO 'cell phone' class) on a BACKGROUND thread so
its ~80-120 ms CPU inference never stalls the main video loop. The main loop
calls submit(frame) each frame (cheap) and reads `.detected` / `.box`.
"""

import threading
import time

CELL_PHONE_CLASS = 67   # COCO class id for "cell phone"


class PhoneDetector:
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf: float = 0.35, imgsz: int = 320, hold_s: float = 0.7,
                 min_interval: float = 0.15):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.hold_s = hold_s            # keep 'detected' True briefly (anti-flicker)
        self.min_interval = min_interval  # throttle inference to free CPU for video

        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self.detected = False
        self.box = None                 # (x1, y1, x2, y2) of best phone box
        self.score = 0.0
        self._last_seen = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        """Hand the latest BGR frame to the worker (non-blocking, drops stale)."""
        with self._lock:
            self._latest = frame

    def _run(self) -> None:
        while not self._stop.is_set():
            t_start = time.time()
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is None:
                self._stop.wait(0.01)
                continue
            try:
                results = self.model.predict(
                    frame, imgsz=self.imgsz, conf=self.conf,
                    classes=[CELL_PHONE_CLASS], verbose=False,
                )
                best, best_score = None, 0.0
                for r in results:
                    for b in r.boxes:
                        s = float(b.conf[0])
                        if s > best_score:
                            best_score = s
                            best = tuple(int(v) for v in b.xyxy[0].tolist())

                now = time.time()
                if best is not None:
                    self._last_seen = now
                    self.box = best
                    self.score = best_score
                self.detected = (now - self._last_seen) <= self.hold_s
                if not self.detected:
                    self.box, self.score = None, 0.0
            except Exception:
                pass

            # Throttle so YOLO doesn't starve the main video loop of CPU.
            rest = self.min_interval - (time.time() - t_start)
            if rest > 0:
                self._stop.wait(rest)

    def stop(self) -> None:
        self._stop.set()
