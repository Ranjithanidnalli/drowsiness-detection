"""
main.py — Driver Drowsiness Detection · Main Pipeline

Run:
    python main.py                        # webcam 0, no ML models
    python main.py --cnn models/eye_cnn.h5
    python main.py --cnn models/eye_cnn.h5 --model_b models/temporal_gb.pkl

Controls:
    Q  — quit
    R  — reset PERCLOS + trackers
    C  — recalibrate neutral head pose
    S  — save current frame snapshot
"""

import argparse
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as mp_BaseOptions

from utils import (
    get_ear, mouth_aspect_ratio, head_pose_angles,
    PerclosTracker, EyeStateTracker,
    LEFT_EYE, RIGHT_EYE,
)
from dashboard import Dashboard
from alert import AlertManager


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmark model (~10 MB) → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH

# ── Config ───────────────────────────────────────────────────────────────────

EAR_THRESHOLD     = 0.20   # below this = eyes closed (used by PERCLOS safety net)
MAR_THRESHOLD     = 0.45   # above this = yawning (lowered so a normal yawn registers)
PERCLOS_MILD      = 0.15   # 15% time eyes closed → mild warning
PERCLOS_DROWSY    = 0.25   # 25% → drowsy alarm
PITCH_NOD_DEG     = 15.0   # head nodding forward threshold
CONSEC_CLOSED_S   = 1.5    # >1.5 s eyes closed → start flagging "eyes closed"
CLOSED_DROWSY_S   = 2.0    # ≥2 s eyes closed → red DROWSY alarm (sleeping)

TARGET_FPS        = 30
FEATURE_WINDOW_S  = 30     # seconds of feature history for Model B

# Must stay in sync with train_temporal.FEATURE_COLS (same columns, same order),
# and the per-feature stats below must match train_temporal.build_windows.
MODEL_B_FEATURES  = ["ear", "mar", "perclos", "pitch", "yaw", "roll", "closed_s"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cnn_model(path: str):
    """Load Keras eye-state CNN. Returns None on failure."""
    try:
        from tensorflow import keras
        model = keras.models.load_model(path)
        print(f"[CNN] Loaded eye-state model from {path}")
        return model
    except Exception as e:
        print(f"[CNN] WARNING: could not load model ({e}). Falling back to EAR threshold.")
        return None


def load_temporal_model(path: str):
    """Load sklearn GradientBoosting model. Returns None on failure."""
    try:
        import joblib
        model = joblib.load(path)
        print(f"[Model B] Loaded temporal model from {path}")
        return model
    except Exception as e:
        print(f"[Model B] WARNING: could not load model ({e}). Using PERCLOS heuristic only.")
        return None


def crop_eye(frame, landmarks, eye_indices, pad=4):
    """Return a 24×24 grayscale eye crop for CNN inference."""
    h, w = frame.shape[:2]
    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in eye_indices
    ])
    x0, y0 = pts.min(axis=0) - pad
    x1, y1 = pts.max(axis=0) + pad
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (24, 24))


def cnn_closed_probability(model, left_crop, right_crop) -> float:
    """Returns mean closed-eye probability from Model A. 0=open, 1=closed."""
    probs = []
    for crop in [left_crop, right_crop]:
        if crop is None:
            continue
        inp = crop.astype(np.float32) / 255.0
        inp = inp[np.newaxis, :, :, np.newaxis]   # (1,24,24,1)
        try:
            p = model.predict(inp, verbose=0)[0][0]
            probs.append(float(p))
        except Exception:
            pass
    return np.mean(probs) if probs else 0.5


def decide_level_heuristic(
    ear: float,
    mar: float,
    perclos: float,
    pitch: float,
    eye_state: EyeStateTracker,
) -> tuple[int, str]:
    """
    Safety-net level decision using only geometric features (no ML models).
    Returns (level, reason_string).
    """
    reasons = []

    if eye_state.closed_seconds >= CONSEC_CLOSED_S:
        reasons.append(f"EYES CLOSED {eye_state.closed_seconds:.1f}s")

    if perclos >= PERCLOS_DROWSY:
        reasons.append(f"PERCLOS {perclos*100:.0f}%")
    elif perclos >= PERCLOS_MILD:
        reasons.append(f"PERCLOS {perclos*100:.0f}%")

    if mar >= MAR_THRESHOLD:
        reasons.append("YAWN detected")

    if pitch > PITCH_NOD_DEG:
        reasons.append(f"HEAD-NOD {pitch:.0f}°")

    reason_str = " · ".join(reasons)

    # Level
    if perclos >= PERCLOS_DROWSY or eye_state.closed_seconds >= CLOSED_DROWSY_S:
        return 2, reason_str
    if perclos >= PERCLOS_MILD or mar >= MAR_THRESHOLD or pitch > PITCH_NOD_DEG:
        return 1, reason_str
    return 0, ""


def decide_level_model_b(
    temporal_model,
    feature_window: list,
    scaler=None,
) -> tuple[int, float, str]:
    """
    Use Model B (GradientBoosting) for level prediction.
    feature_window: list of feature dicts, last FEATURE_WINDOW_S * fps entries.
    Returns (level, score_0_to_1, reason_str).
    """
    if len(feature_window) < 10:
        return -1, -1.0, ""

    # Aggregate over the window. Stats + column order MUST match
    # train_temporal.build_windows, or the model sees the wrong feature vector.
    agg = {}
    for col in MODEL_B_FEATURES:
        vals = np.array([f[col] for f in feature_window], dtype=float)
        agg[f"{col}_mean"] = float(np.mean(vals))
        agg[f"{col}_std"]  = float(np.std(vals))
        agg[f"{col}_max"]  = float(np.max(vals))
        agg[f"{col}_p75"]  = float(np.percentile(vals, 75))

    X = np.array(list(agg.values())).reshape(1, -1)
    if scaler:
        X = scaler.transform(X)

    try:
        pred  = temporal_model.predict(X)[0]           # 0/1/2
        proba = temporal_model.predict_proba(X)[0]     # [p_alert, p_mild, p_drowsy]
        score = float(proba[pred])

        # Feature importance for "why it fired"
        if hasattr(temporal_model, "feature_importances_"):
            feat_names = list(agg.keys())
            imp = temporal_model.feature_importances_
            top_k = np.argsort(imp)[-3:][::-1]

            def _clean(name: str) -> str:
                for suf in ("_mean", "_std", "_max", "_p75"):
                    name = name.replace(suf, "")
                return name.upper()

            reason_str = " · ".join(
                f"{_clean(feat_names[i])} {agg[feat_names[i]]:.2f}"
                for i in top_k
            )
        else:
            reason_str = f"Model-B score: {score:.2f}"

        return int(pred), score, reason_str
    except Exception as e:
        print(f"[Model B] inference error: {e}")
        return -1, -1.0, ""


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(args):
    # Load optional ML models
    cnn_model      = load_cnn_model(args.cnn)       if args.cnn      else None
    temporal_model = load_temporal_model(args.model_b) if args.model_b else None
    scaler         = None
    if args.scaler and os.path.exists(args.scaler):
        import joblib
        scaler = joblib.load(args.scaler)

    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {frame_w}×{frame_h}")

    # MediaPipe Face Landmarker (Tasks API)
    model_path = _ensure_model()
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # Trackers
    perclos_tracker = PerclosTracker(window_seconds=FEATURE_WINDOW_S, ear_threshold=EAR_THRESHOLD)
    eye_tracker     = EyeStateTracker(closed_seconds_threshold=CONSEC_CLOSED_S, ear_threshold=EAR_THRESHOLD)
    alert_manager   = AlertManager()
    dashboard       = Dashboard(frame_w, frame_h)

    # Phone-use distraction detector (YOLOv8n, background thread)
    phone_detector = None
    if not args.no_phone:
        try:
            from phone_detector import PhoneDetector
            phone_detector = PhoneDetector()
            print("[Phone] YOLOv8n distraction detector active.")
        except Exception as e:
            print(f"[Phone] disabled ({e})")

    # Feature window for Model B
    feature_window: list[dict] = []
    max_window = int(TARGET_FPS * FEATURE_WINDOW_S)

    # Head-pose calibration (neutral pitch baseline; press C to recalibrate)
    pitch_baseline = None
    pitch_calib: list[float] = []
    CALIB_FRAMES = 30

    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    prev_time = time.time()
    frame_count = 0

    print("\nRunning — Q quit · R reset · C recalibrate head pose · S snapshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        if phone_detector is not None:
            phone_detector.submit(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = face_landmarker.detect(mp_image)

        # Defaults when no face detected
        ear = 0.30; mar = 0.0; pitch = 0.0; yaw = 0.0; roll = 0.0
        pitch_rel = 0.0
        cnn_prob = -1.0; level = 0; reason = ""; model_b_score = -1.0

        if results.face_landmarks:
            lms = results.face_landmarks[0]

            # ── Geometric features ────────────────────────────────────────
            _, _, ear = get_ear(lms, frame_w, frame_h)
            mar        = mouth_aspect_ratio(lms, frame_w, frame_h)
            pitch, yaw, roll = head_pose_angles(lms, frame_w, frame_h)

            # Calibrate neutral pitch from the first stable frames, then
            # measure pitch RELATIVE to neutral (kills camera-angle offset).
            if pitch_baseline is None:
                pitch_calib.append(pitch)
                if len(pitch_calib) >= CALIB_FRAMES:
                    pitch_baseline = float(np.median(pitch_calib))
                    print(f"[Calibrate] Neutral pitch = {pitch_baseline:.1f} deg")
            pitch_rel = pitch - (pitch_baseline if pitch_baseline is not None else pitch)

            # ── Model A: eye CNN ──────────────────────────────────────────
            if cnn_model is not None:
                lc = crop_eye(frame, lms, LEFT_EYE)
                rc = crop_eye(frame, lms, RIGHT_EYE)
                cnn_prob = cnn_closed_probability(cnn_model, lc, rc)
                # Replace EAR-based closed decision with CNN probability
                effective_ear = 1.0 - cnn_prob   # treat prob-closed as inverted EAR
            else:
                effective_ear = ear

            # ── Update trackers ───────────────────────────────────────────
            perclos = perclos_tracker.update(effective_ear)
            eye_tracker.update(effective_ear)

            # ── Feature window for Model B ────────────────────────────────
            feature_window.append({
                "ear":     ear,
                "mar":     mar,
                "perclos": perclos,
                "pitch":   pitch,
                "yaw":     abs(yaw),
                "roll":    abs(roll),
                "cnn":     cnn_prob if cnn_prob >= 0 else effective_ear,
                "closed_s": eye_tracker.closed_seconds,
            })
            if len(feature_window) > max_window:
                feature_window.pop(0)

            # ── Level decision ────────────────────────────────────────────
            if temporal_model is not None:
                mb_level, model_b_score, mb_reason = decide_level_model_b(
                    temporal_model, feature_window, scaler
                )
                if mb_level >= 0:
                    level  = mb_level
                    reason = mb_reason
                else:
                    level, reason = decide_level_heuristic(
                        ear, mar, perclos, pitch_rel, eye_tracker
                    )
            else:
                level, reason = decide_level_heuristic(
                    ear, mar, perclos, pitch_rel, eye_tracker
                )

        else:
            # No face detected
            reason = "No face detected"

        # ── Phone-use distraction (escalates the alert when active) ────────
        phone     = phone_detector.detected if phone_detector is not None else False
        phone_box = phone_detector.box      if phone_detector is not None else None
        if phone:
            level = max(level, 2)
            tag = "PHONE USE - eyes off road"
            reason = tag if (not reason or reason == "No face detected") else f"{tag} | {reason}"

        alert_manager.update(level, reason)

        # ── Draw dashboard ────────────────────────────────────────────────
        frame = dashboard.draw(
            frame, ear, mar,
            perclos_tracker.value,
            pitch_rel, yaw, roll,
            level, reason,
            cnn_prob, model_b_score,
            yawning=(mar >= MAR_THRESHOLD),
            phone=phone, phone_box=phone_box,
        )

        # FPS counter
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (self_x := frame_w - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            perclos_tracker = PerclosTracker(window_seconds=FEATURE_WINDOW_S, ear_threshold=EAR_THRESHOLD)
            eye_tracker     = EyeStateTracker(closed_seconds_threshold=CONSEC_CLOSED_S, ear_threshold=EAR_THRESHOLD)
            feature_window.clear()
            print("[Reset] Trackers cleared.")
        elif key == ord("c"):
            pitch_baseline = None
            pitch_calib.clear()
            print("[Calibrate] Recalibrating neutral head pose — sit straight...")
        elif key == ord("s"):
            fname = os.path.join(snapshot_dir, f"snap_{int(time.time())}.png")
            cv2.imwrite(fname, frame)
            print(f"[Snapshot] Saved {fname}")

    # Cleanup
    cap.release()
    face_landmarker.close()
    if phone_detector is not None:
        phone_detector.stop()
    cv2.destroyAllWindows()
    alert_manager.stop()
    print("Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--camera",  type=int,   default=0,    help="Webcam index (default 0)")
    parser.add_argument("--cnn",     type=str,   default=None, help="Path to eye CNN model (.h5)")
    parser.add_argument("--model_b", type=str,   default=None, help="Path to temporal model (.pkl)")
    parser.add_argument("--scaler",  type=str,   default=None, help="Path to feature scaler (.pkl)")
    parser.add_argument("--no_phone", action="store_true", help="Disable YOLO phone-use detection")
    args = parser.parse_args()
    main(args)
