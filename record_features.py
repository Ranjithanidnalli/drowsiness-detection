"""
record_features.py — Record labelled feature sequences for Model B training.

Run once per session:
    python record_features.py --label 0 --duration 300 --out data/features_alert.csv
    python record_features.py --label 1 --duration 300 --out data/features_mild.csv
    python record_features.py --label 2 --duration 300 --out data/features_drowsy.csv

Labels: 0=Alert, 1=Mild, 2=Drowsy (acted or from UTA-RLDD video frames)

If you have UTA-RLDD video files, use --video path/to/video.avi instead of live webcam.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as mp_BaseOptions

from utils import get_ear, mouth_aspect_ratio, head_pose_angles, PerclosTracker


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmark model (~10 MB) → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def record(args):
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("Cannot open source"); sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_BaseOptions(model_asset_path=_ensure_model()),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    perclos = PerclosTracker()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["ear", "mar", "perclos", "pitch", "yaw", "roll", "closed_s", "label"]
    file_exists = os.path.exists(args.out)

    consecutive_closed = 0
    EAR_THRESH = 0.20

    start = time.time()
    frames_written = 0

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        label_str = {0: "ALERT", 1: "MILD", 2: "DROWSY"}[args.label]
        print(f"Recording '{label_str}' features → {args.out}")
        print(f"Duration: {args.duration}s  |  Press Q to stop early.\n")

        while True:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_landmarker.detect(mp_image)

            if results.face_landmarks:
                lms = results.face_landmarks[0]
                _, _, ear = get_ear(lms, frame_w, frame_h)
                mar        = mouth_aspect_ratio(lms, frame_w, frame_h)
                pit, yaw, roll = head_pose_angles(lms, frame_w, frame_h)
                perc = perclos.update(ear)

                if ear < EAR_THRESH:
                    consecutive_closed += 1
                else:
                    consecutive_closed = 0
                closed_s = consecutive_closed / 30.0

                writer.writerow({
                    "ear": round(ear, 4),
                    "mar": round(mar, 4),
                    "perclos": round(perc, 4),
                    "pitch": round(pit, 2),
                    "yaw": round(abs(yaw), 2),
                    "roll": round(abs(roll), 2),
                    "closed_s": round(closed_s, 2),
                    "label": args.label,
                })
                frames_written += 1

            # HUD
            cv2.putText(frame, f"Label: {label_str}  |  {elapsed:.0f}/{args.duration}s  |  Frames: {frames_written}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()
    print(f"\nDone. {frames_written} frames saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",    type=int, required=True, choices=[0,1,2], help="0=Alert 1=Mild 2=Drowsy")
    parser.add_argument("--duration", type=int, default=300,  help="Recording seconds")
    parser.add_argument("--out",      type=str, default="data/features.csv")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--video",    type=str, default=None, help="Path to video file instead of webcam")
    args = parser.parse_args()
    record(args)
