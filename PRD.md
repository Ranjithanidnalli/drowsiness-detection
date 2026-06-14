# Product Requirements Document (PRD)
## DriveAwake — Real-Time Driver Drowsiness & Distraction Detection

| | |
|---|---|
| **Product name** | DriveAwake |
| **Category** | Computer-Vision / Edge AI · Road Safety |
| **Version** | 1.0 |
| **Date** | 13 June 2026 |
| **Author** | Ranjitha N R |
| **Status** | Working prototype (hackathon, 36-hour build) |
| **Platforms** | Desktop app (OpenCV window) · Web app (browser, single URL) |

---

## 1. Executive Summary

DriveAwake is a real-time, camera-based system that detects when a driver is
becoming **drowsy** or **distracted** and raises an **escalating audio-visual
alarm** before an accident can happen. It runs entirely on a normal laptop
webcam — no special hardware — using a fusion of **facial-landmark geometry**,
**temporal eye-closure analysis (PERCLOS)**, **head-pose estimation**, and
**object detection (phone use)**. It ships in two forms: a **desktop OpenCV
app** and a **hosted web app** that streams the annotated feed to any browser at
a single URL.

The system is built to be **explainable** ("why did it fire?"), **frame-rate
independent** (time-based, not frame-count based), and **upgradeable with real
trained ML models** (an eye-state CNN and a temporal classifier) rather than
being a thin wrapper around a cloud API.

---

## 2. Problem Statement

- **Drowsy and distracted driving is a leading, preventable cause of road
  fatalities.** Fatigue slows reaction time, impairs judgement, and causes
  microsleeps (2–10 second lapses) at highway speeds where a vehicle travels
  tens of metres "blind."
- Long-haul commercial drivers, night-shift workers, and intercity travellers
  are especially at risk.
- **Distraction** — particularly **mobile phone use** — compounds the problem by
  taking the driver's eyes off the road.
- Most existing safety systems are **expensive OEM features** locked to premium
  vehicles, or **cloud-dependent** apps with latency and privacy concerns.

> 📌 *For the presentation, cite verified figures from WHO Global Status Report on
> Road Safety, India MoRTH "Road Accidents in India" report, and NHTSA drowsy-
> driving statistics. Placeholder talking points: India records one of the
> highest absolute road-fatality counts globally; a meaningful share of crashes
> involve fatigue or distraction. Insert the exact, sourced numbers on the slide.*

**Gap:** There is no affordable, privacy-preserving, explainable drowsiness
detector that works on commodity hardware and fuses multiple fatigue signals
(not just "are the eyes closed").

---

## 3. Goals & Objectives

### 3.1 Primary goals
1. Detect drowsiness in **real time** (sub-second to low-second latency) on a
   standard laptop webcam.
2. Detect **mobile phone distraction** while driving.
3. **Alert the driver** with an escalating buzzer + spoken warning that is loud
   enough and timely enough to prevent a microsleep crash.
4. Be **explainable** — always show *which* signal triggered the alert.
5. Run **on-device / locally** (privacy-preserving — video never leaves the
   machine) and be **hostable** at a URL for demos.

### 3.2 Success criteria (hackathon judging)
- ✅ **Live demo moment** — judge closes their eyes / yawns / lifts a phone and
  the system reacts instantly and audibly.
- ✅ **Explainability** — on-screen reason string and feature-importance readout.
- ✅ **Real trained model path** — not just an API call (CNN + temporal model).
- ✅ **Context-specificity** — designed for Indian driving conditions / data.

---

## 4. Target Users & Personas

| Persona | Description | Need |
|---|---|---|
| **Commercial / fleet driver** | Long-haul truck or bus driver, night shifts | Stay-awake alerts on long monotonous routes |
| **Intercity car driver** | Family/solo long-distance driver | Affordable safety net without buying a premium car |
| **Cab / ride-share driver** | Long hours, variable sleep | Fatigue + phone-distraction warnings |
| **Fleet operator (B2B)** | Manages many vehicles | Driver-safety analytics & compliance |

---

## 5. Scope

### 5.1 In scope (delivered in v1.0)
- Real-time eye-closure (EAR), yawn (MAR), PERCLOS, head-nod (pitch) detection.
- Mobile-phone-use detection (YOLOv8n).
- Three-level escalating alert (alert → mild → drowsy) with buzzer + TTS.
- Head-pose auto-calibration (removes camera-angle bias).
- Explainable reason string on every alert.
- Desktop app and hosted web app (MJPEG over FastAPI).
- Training scripts + data-recording tool for two ML models.

### 5.2 Out of scope (v1.0)
- In-vehicle hardware integration (CAN bus, steering-input sensors).
- Cloud fleet dashboard / multi-driver analytics backend.
- Mobile native app (iOS/Android).
- Heart-rate / wearable sensor fusion.
- Production-grade model accuracy benchmarking (training pipeline is provided;
  full training requires GPU + dataset).

---

## 6. Functional Requirements (Features)

| ID | Feature | Description | Status |
|----|---------|-------------|--------|
| F1 | **Eye-closure detection (EAR)** | Per-eye Eye Aspect Ratio from 6 landmarks; eyes "closed" when EAR < 0.20 | ✅ |
| F2 | **Yawn detection (MAR)** | Mouth Aspect Ratio; "yawn" when MAR ≥ 0.45; YAWN badge shown | ✅ |
| F3 | **PERCLOS** | % of time eyes closed over a rolling 30-second window; mild ≥15%, drowsy ≥25% | ✅ |
| F4 | **Sustained-closure alarm** | Eyes continuously closed ≥1.5s flagged; ≥2.0s → red DROWSY (microsleep) | ✅ |
| F5 | **Head-pose estimation** | Pitch/yaw/roll via solvePnP; head-nod when pitch > 15° from neutral | ✅ |
| F6 | **Head-pose calibration** | Auto-learns neutral pitch from first 30 frames; manual recalibrate (C / button) | ✅ |
| F7 | **Phone-use detection** | YOLOv8n detects COCO "cell phone" (class 67); escalates to drowsy level | ✅ |
| F8 | **Escalating alerts** | L0 silent · L1 periodic beep · L2 continuous buzzer + spoken warning | ✅ |
| F9 | **Explainable reason** | Human-readable reason string (e.g. "EYES CLOSED 2.3s · PERCLOS 28%") | ✅ |
| F10 | **Live dashboard overlay** | EAR/MAR/PERCLOS/pose gauges, status colour, badges drawn on frame | ✅ |
| F11 | **Snapshot capture** | Save the current annotated frame (S / button) | ✅ |
| F12 | **Reset / recalibrate controls** | Clear trackers; recalibrate head pose | ✅ |
| F13 | **Web hosting** | Browser view of the exact annotated feed at a single URL (MJPEG) | ✅ |
| F14 | **Mute control** | Silence the buzzer from the UI without stopping detection | ✅ |
| F15 | **Eye-state CNN (Model A)** | Optional trained CNN replaces EAR threshold for closed-eye decision | ⚙️ Training-ready |
| F16 | **Temporal classifier (Model B)** | Optional GradientBoosting over 30s feature window → drowsiness level | ⚙️ Training-ready |

Legend: ✅ implemented & working · ⚙️ code complete, needs training data/GPU.

---

## 7. Detection Methodology (The Science)

### 7.1 Facial landmarks
- **MediaPipe FaceLandmarker** (Tasks API, `face_landmarker.task`, ~10 MB,
  auto-downloaded) extracts **468 3-D face landmarks** + iris points at ~real
  time on CPU.
- Single face, `min_face_detection_confidence = 0.5`, `min_tracking_confidence = 0.5`.

### 7.2 Eye Aspect Ratio (EAR)
- Uses 6 landmarks per eye.
  - Left eye: `[362, 385, 387, 263, 373, 380]`
  - Right eye: `[33, 160, 158, 133, 153, 144]`
- `EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)` — ratio of vertical to horizontal
  eye opening. Drops sharply when the eye closes.
- **Threshold:** EAR < **0.20** → eye considered closed.

### 7.3 Mouth Aspect Ratio (MAR)
- Landmarks: top lip 13, bottom lip 14, left corner 78, right corner 308.
- `MAR = vertical mouth opening / horizontal mouth width`.
- **Threshold:** MAR ≥ **0.45** → yawn.

### 7.4 PERCLOS (Percentage of Eye Closure)
- The clinically validated gold-standard fatigue metric.
- **Time-based rolling window** (default **30 s**) storing `(timestamp, closed?)`
  — pruned by wall-clock time so it is **independent of camera FPS**.
- `PERCLOS = closed_frames / total_frames` in the window.
- **Thresholds:** ≥ **15%** → mild · ≥ **25%** → drowsy.

### 7.5 Sustained eye-closure timer (microsleep)
- Tracks **continuous** closure in real seconds (wall clock).
- ≥ **1.5 s** → start flagging "eyes closed."
- ≥ **2.0 s** → **red DROWSY** alarm (a 2-second closure at speed = danger).

### 7.6 Head-pose estimation
- `cv2.solvePnP` maps a 3-D face model to 6 image landmarks (nose 1, chin 9,
  eye corners 33/263, mouth corners 57/287) → rotation → **pitch / yaw / roll**.
- **Head-nod** when **pitch > 15°** relative to the calibrated neutral.

### 7.7 Head-pose calibration (key robustness feature)
- Camera mounting angle makes "neutral" pitch read non-zero (e.g. ~50°), causing
  constant false head-nod alerts.
- Fix: collect the **first 30 frames**, take the **median** as `pitch_baseline`,
  and use `pitch_relative = pitch − baseline` everywhere.
- Re-trigger anytime with **C** (desktop) / **Recalibrate** (web).

### 7.8 Phone-use distraction
- **YOLOv8n** (ultralytics) detects the COCO **"cell phone"** class (id **67**).
- Runs on a **background thread**, throttled (`min_interval = 0.15 s`,
  `imgsz = 320`, `conf = 0.35`) so ~100 ms inference never stalls the video.
- `hold_s = 0.7` anti-flicker hold. A detected phone **escalates the alert to
  level 2** and draws a red bounding box + "PHONE USE" badge.

### 7.9 Multi-signal fusion → alert level
Final level is the **max** of all signals (a single strong signal can trigger):

```
Level 2 (DROWSY)  if  PERCLOS ≥ 25%  OR  eyes closed ≥ 2.0s  OR  phone detected
Level 1 (MILD)    if  PERCLOS ≥ 15%  OR  yawn (MAR ≥ 0.45)   OR  head-nod > 15°
Level 0 (ALERT)   otherwise
```

---

## 8. Alert & Escalation System

| Level | Name | Colour | Audio | Voice (TTS) |
|-------|------|--------|-------|-------------|
| 0 | Alert | 🟢 Green | Silence | — |
| 1 | Mild | 🟡 Yellow | Single beep (660 Hz, 300 ms) every **2.5 s** | — |
| 2 | Drowsy | 🔴 Red | Continuous urgent buzzer (900 Hz, 450 ms, back-to-back) | Spoken warning every **4 s** ("Wake up! Please pull over and rest.") |

**Engineering details that make it reliable:**
- The entire alarm runs on a **background daemon thread** — it **never freezes
  the video loop** (the original bug: `play(); sleep()` on the main thread froze
  the feed ~2 s).
- **Audio backend:** `pygame` tone synthesis (preferred) with a **`winsound.Beep`
  Windows-native fallback** so audio is guaranteed even if pygame fails.
- **pygame stereo fix:** SDL forces a stereo mixer even when mono is requested;
  the beep array is shaped to the real channel count (`np.column_stack`) so
  `make_sound()` doesn't crash (this was the original *silent buzzer* bug).
- **TTS:** `pyttsx3`, non-overlapping (lock-guarded), spoken on its own thread.
- In the web app the buzzer plays on the **host computer's speakers** (same as
  desktop); **Mute** toggles it.

---

## 9. Machine Learning Models (Upgrade Path)

The app works out-of-the-box on **geometric heuristics**. Two optional **trained
models** make it "real AI" and are fully wired in (code complete; need
training data + GPU):

### 9.1 Model A — Eye-state CNN  (`--cnn models/eye_cnn.h5`)
- Small custom CNN (~50k params) classifying a **24×24 grayscale eye crop** as
  **open vs closed**.
- **Dataset:** MRL Eye Dataset (~84,000 infrared eye images).
- When enabled, the CNN's closed-eye probability **replaces the EAR threshold**
  for a more robust closed-eye decision (lighting/eyewear tolerant).
- Fallback to `MobileNetV2` transfer learning if the small CNN underperforms.
- Trainer: `train_eye_cnn.py` (Colab-ready, ~30–45 min on a T4 GPU).

### 9.2 Model B — Temporal drowsiness classifier  (`--model_b models/temporal_gb.pkl`)
- **GradientBoosting** classifier over a **30-second window** of features.
- **Feature vector = 28 values:** 4 statistics (**mean, std, max, p75**) over
  **7 signals** (`ear, mar, perclos, pitch, yaw, roll, closed_s`).
- Outputs the drowsiness **level + confidence**, and a **feature-importance
  "why it fired"** explanation (top-3 contributing features).
- Data captured locally with `record_features.py`; trained with
  `train_temporal.py` (runs on CPU). Train/inference schemas are **aligned (28
  features both sides)**.

> The inference feature schema is locked to the trainer's, verified:
> `infer features: 28 | trainer features: 28 | columns match exactly: True`.

---

## 10. System Architecture

### 10.1 Desktop pipeline (`main.py`)
```
Webcam ─▶ MediaPipe FaceLandmarker ─▶ feature extraction (EAR/MAR/pose)
                                          │
        ┌─────────────────────────────────┼──────────────────────────────┐
        ▼                ▼                 ▼              ▼                 ▼
   PERCLOS tracker   eye-state timer   head-pose      Model A (opt)   Model B (opt)
        └────────────────┴───────┬─────────┴──────────────┴────────┘
                                  ▼
                       fusion → alert level + reason
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                    ▼
       Dashboard overlay   AlertManager (bg thread)  PhoneDetector (bg thread)
       (OpenCV window)     buzzer + TTS              YOLOv8n
```

### 10.2 Web hosting (`web_app.py`)
```
   THIS COMPUTER (FastAPI + uvicorn)                 ANY BROWSER (single URL)
   webcam → same pipeline → dashboard overlay        <img src="/video"> (MJPEG)
   → cv2.imencode(JPEG) → /video MJPEG stream  ─────▶ buttons POST /recalibrate…
   buzzer on host speakers                            status pill polls /stats
```
- **Why MJPEG:** preserves the *exact* desktop look; the browser only **views**
  the stream (never opens a camera), so it works over plain HTTP on the LAN
  (no HTTPS required) and on phones on the same WiFi.
- Single FastAPI server serves both the page and the stream → **one URL**.

### 10.3 Concurrency model
- **Main loop:** capture → detect → draw → stream (synchronous, ordered).
- **Buzzer thread:** daemon; reacts to the latest level; never blocks video.
- **Phone thread:** daemon; runs YOLO on the latest frame, throttled.
- **Trackers are time-based** (wall clock) → correct at any real FPS (~12–30).

---

## 11. Technical Stack

| Layer | Technology |
|---|---|
| Language | **Python 3.13** |
| Camera / image | **OpenCV** (`opencv-python`) |
| Face landmarks | **MediaPipe Tasks API** — `FaceLandmarker` |
| Numerics | **NumPy**, **SciPy** |
| Object detection | **Ultralytics YOLOv8n** + **PyTorch / TorchVision** (CPU) |
| Eye CNN (Model A) | **TensorFlow / Keras** |
| Temporal model (Model B) | **scikit-learn** (GradientBoosting) + **joblib** |
| Audio | **pygame** (tone synth) · **winsound** (fallback) · **pyttsx3** (TTS) |
| Data / plots | **pandas**, **matplotlib** |
| Web server | **FastAPI** + **uvicorn** (MJPEG streaming) |
| Tunneling (optional) | **cloudflared** / **ngrok** for a public HTTPS URL |

---

## 12. Data Requirements & Datasets

| Use | Dataset / Source | Notes |
|---|---|---|
| Eye-state CNN | **MRL Eye Dataset** (~84k IR eye images, open/closed) | Public; for Model A |
| Temporal model | **Self-recorded** via `record_features.py` (act Alert/Mild/Drowsy ~5 min each) or **UTA-RLDD** videos | Labels 0/1/2 |
| **Indian-context validation** | Short self-recorded / real Indian driving footage | **Key differentiator** — report accuracy on India-specific data |

Recording commands:
```
python record_features.py --label 0 --duration 300 --out data/features.csv   # Alert
python record_features.py --label 1 --duration 300 --out data/features.csv   # Mild
python record_features.py --label 2 --duration 300 --out data/features.csv   # Drowsy
python train_temporal.py --data data/features.csv
```

---

## 13. Deployment & Hosting

| Mode | Command | Access |
|---|---|---|
| Desktop app | `python main.py` | Local OpenCV window |
| With models | `python main.py --cnn models/eye_cnn.h5 --model_b models/temporal_gb.pkl --scaler models/feature_scaler.pkl` | Local |
| **Web (local)** | `python web_app.py` | `http://localhost:8000` |
| Web (LAN / phone) | `python web_app.py --host 0.0.0.0` | `http://<pc-ip>:8000` (no HTTPS needed) |
| Web (public link) | `python web_app.py` + `cloudflared tunnel --url http://localhost:8000` | `https://….trycloudflare.com` |

> Only one process can hold the webcam — quit the desktop app before launching
> the web app.

---

## 14. User Flows

### 14.1 Desktop
1. Run `python main.py`. 2. Look straight ~2 s (auto head-pose calibration).
3. Drive / act normally; gauges and status update live. 4. On fatigue/phone →
escalating alarm + reason. 5. Keys: **Q** quit · **R** reset · **C** recalibrate
· **S** snapshot.

### 14.2 Web
1. Quit desktop app. 2. Run `python web_app.py`. 3. Open `http://localhost:8000`.
4. Live annotated feed appears. 5. Buttons: **Recalibrate · Reset · Snapshot ·
Mute**. 6. Optional: share via LAN IP or Cloudflare tunnel.

---

## 15. Non-Functional Requirements

| Attribute | Requirement / Achieved |
|---|---|
| **Latency** | Sub-second signal-to-alert; alarm thread reacts within ~50 ms of a level change |
| **Throughput** | Real-time at commodity-webcam FPS (~12–30); time-based trackers stay correct regardless |
| **Robustness** | FPS-independent timing; head-pose calibration; audio fallback; graceful "no face" handling |
| **Privacy** | 100% on-device; video never uploaded; web mode streams only on the local network unless a tunnel is started |
| **Cost** | Runs on a standard laptop webcam — no special hardware, no per-call cloud cost |
| **Portability** | Pure-Python; Windows-verified; desktop + web |
| **Resilience** | Background threads isolate audio & YOLO so the video never freezes |

---

## 16. Explainability & Trust

- Every alert carries a **plain-language reason** (e.g.
  `EYES CLOSED 2.3s · PERCLOS 28%` or `PHONE USE - eyes off road`).
- With Model B, a **feature-importance** readout names the **top-3 signals** that
  drove the decision — turning a black-box score into a defensible explanation.
- Roadmap: a **SHAP** plot for per-decision attribution.

---

## 17. Competitive Differentiation

1. **Multi-signal fusion**, not just eye-closure — eyes + yawn + PERCLOS +
   head-nod + phone, combined.
2. **Explainable** — shows *why*, not just a red light.
3. **Real trained-model path** (CNN + temporal model), not a cloud-API wrapper.
4. **FPS-independent, calibrated** engineering → fewer false alarms.
5. **Privacy-first & zero-cost** — fully local on commodity hardware.
6. **Indian-context focus** — validated on local driving data (most academic
   datasets are Western).
7. **Instantly demoable & hostable** — desktop *and* one-URL web app.

---

## 18. Metrics & KPIs

| KPI | Definition | Target |
|---|---|---|
| Detection latency | Time from eyes-closed event to alarm | < 2.0 s (set by `CLOSED_DROWSY_S`) |
| True-positive rate | Drowsy events correctly alerted | ↑ (report on validation set) |
| False-alarm rate | Alerts when alert | ↓ (calibration + fusion reduce this) |
| Frame rate | Processed FPS | ≥ 12 on laptop CPU |
| Phone-detection precision | Correct phone flags | report @ conf 0.35 |
| Uptime / no-freeze | Video never stalls during alarm | 100% (threaded) |

---

## 19. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Camera-angle false head-nods | Auto + manual head-pose calibration |
| Frame-rate variability breaks "3-second" logic | Wall-clock time-based trackers |
| Silent / blocking buzzer | Stereo-array fix + background thread + winsound fallback |
| YOLO inference stalls video | Background thread + throttling |
| Glasses / low light hurt EAR | Optional eye-state CNN (Model A) |
| Western-dataset bias | Indian-context validation set |
| Webcam contention (desktop vs web) | Documented single-owner constraint |

---

## 20. Roadmap / Future Work

- **Short term:** train Model A (MRL) and Model B (recorded data); add SHAP plot;
  record an Indian-context validation set and report accuracy.
- **Medium term:** mobile (Android) port; night-vision / IR camera support;
  seatbelt & smoking detection; drowsiness *trend* graph over a trip.
- **Long term:** fleet dashboard (B2B), CAN-bus / vehicle integration, wearable
  (heart-rate) fusion, edge deployment on Raspberry Pi / Jetson Nano.

---

## 21. Demo Script (for the live presentation)

1. **Baseline** — face the camera; status **green ALERT**, gauges steady.
2. **Yawn** — open mouth wide → **YAWN** badge + **yellow MILD** + beep.
3. **Microsleep** — close eyes ~2 s → **red DROWSY** banner + continuous buzzer +
   spoken "Wake up!" → reason shows `EYES CLOSED 2.x s`.
4. **Distraction** — lift a phone into frame → red box + **PHONE USE** + level 2.
5. **Explainability** — point to the live reason string (and Model-B top features
   if trained).
6. **Hosting** — open `http://localhost:8000` (or the Cloudflare link) to show
   the same thing running in a browser / on a phone.

---

## 22. Appendix

### 22.1 Configuration reference (tunable thresholds)
| Constant | Value | Meaning |
|---|---|---|
| `EAR_THRESHOLD` | 0.20 | Eye-closed cutoff |
| `MAR_THRESHOLD` | 0.45 | Yawn cutoff |
| `PERCLOS_MILD` | 0.15 | 15% closed → mild |
| `PERCLOS_DROWSY` | 0.25 | 25% closed → drowsy |
| `PITCH_NOD_DEG` | 15.0 | Head-nod angle (relative to neutral) |
| `CONSEC_CLOSED_S` | 1.5 | Start flagging eyes-closed |
| `CLOSED_DROWSY_S` | 2.0 | Red DROWSY (microsleep) |
| `FEATURE_WINDOW_S` | 30 | PERCLOS / Model-B window |
| `CALIB_FRAMES` | 30 | Frames for head-pose calibration |
| `MILD_INTERVAL` | 2.5 s | Gap between mild beeps |
| `TTS_INTERVAL` | 4.0 s | Gap between spoken drowsy warnings |
| Phone `conf` / `imgsz` | 0.35 / 320 | YOLO confidence / input size |
| Phone `hold_s` | 0.7 s | Anti-flicker hold |

### 22.2 Project structure
```
drowsiness_detection/
├── main.py              Desktop pipeline (OpenCV window)
├── web_app.py           Web host (FastAPI + MJPEG stream)
├── utils.py             EAR/MAR/head-pose + PERCLOS & eye-state trackers
├── dashboard.py         On-frame overlay (gauges, badges, status)
├── alert.py             Escalating buzzer + TTS (background thread)
├── phone_detector.py    YOLOv8n phone-use detection (background thread)
├── record_features.py   Record labelled data for Model B
├── train_eye_cnn.py     Train Model A (eye-state CNN)
├── train_temporal.py    Train Model B (temporal GradientBoosting)
├── requirements.txt     Dependencies
├── HOSTING.md           How to host the web app
├── TRAINING.md          How to train Models A & B (Colab-ready)
├── PRD.md               This document
├── face_landmarker.task MediaPipe model (auto-downloaded)
├── yolov8n.pt           YOLOv8n weights
├── models/  data/  snapshots/  sounds/  logs/
```

### 22.3 Suggested slide deck (maps PRD → ~16 slides)
1. Title + one-line pitch (§1) · 2. Problem & stats (§2) · 3. Goals & success
criteria (§3) · 4. Target users (§4) · 5. Solution overview / features (§6) ·
6. How it works — EAR/MAR/PERCLOS (§7.2–7.5) · 7. Head-pose + calibration
(§7.6–7.7) · 8. Phone-use detection (§7.8) · 9. Fusion → alert levels
(§7.9, §8) · 10. ML models A & B (§9) · 11. System architecture diagram (§10) ·
12. Tech stack (§11) · 13. Hosting / demo URL (§13) · 14. Differentiation (§17) ·
15. Roadmap (§20) · 16. Live demo (§21).
```
```
