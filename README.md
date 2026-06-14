# 🚗 DriveAwake — Real-Time Driver Drowsiness & Distraction Detection

> Detects driver fatigue and phone-use distraction on a **plain webcam** — no special hardware, runs **fully on-device**, and raises an **escalating buzzer + voice alert** before a microsleep becomes a crash.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-vision-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceLandmarker-00A98F)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-phone%20detection-111F68)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20host-009688?logo=fastapi&logoColor=white)

---

## What it does

DriveAwake watches the driver through a webcam and fuses **multiple fatigue signals** in real time:

- 👁️ **Eye closure (EAR)** — eyes closed when Eye-Aspect-Ratio < `0.20`
- 😮 **Yawning (MAR)** — Mouth-Aspect-Ratio ≥ `0.45`
- ⏱️ **PERCLOS** — % of time eyes are closed over a rolling 30-second window (mild ≥ 15%, drowsy ≥ 25%)
- 😴 **Microsleep** — eyes continuously closed ≥ `2.0 s` → red **DROWSY** alarm
- 🤕 **Head nod** — head pitch > `15°` from a calibrated neutral
- 📱 **Phone use** — YOLOv8n detects a mobile phone in frame → "eyes off road"

…then escalates an alarm:

| Level | State | Colour | Response |
|-------|-------|--------|----------|
| 0 | Alert | 🟢 Green | silent |
| 1 | Mild | 🟡 Yellow | periodic beep |
| 2 | Drowsy / Distracted | 🔴 Red | continuous buzzer **+ spoken warning** |

Every alert shows a **plain-language reason** (e.g. `EYES CLOSED 2.3s · PERCLOS 28%` or `PHONE USE - eyes off road`) — so it's explainable, not a black box.

> _📸 Tip: press **S** (desktop) or the **Snapshot** button (web) and drop an image here as `docs/demo.png` for a great repo preview._

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Desktop app (OpenCV window)
python main.py

# 2b. OR the web app (view in a browser at http://localhost:8000)
python web_app.py
```

First run auto-downloads the MediaPipe face model (~10 MB) and the YOLOv8n weights.

### Desktop controls
`Q` quit · `R` reset trackers · `C` recalibrate head pose · `S` snapshot

### Web controls (on the page)
**Camera On/Off** · Recalibrate · Reset · Snapshot · Mute
*(Camera On/Off releases the webcam without stopping the server.)*

### Useful flags
```bash
python main.py --no_phone                 # disable YOLO phone detection
python main.py --cnn models/eye_cnn.h5    # use trained eye-state CNN (Model A)
python web_app.py --host 0.0.0.0          # allow other devices on the LAN
python web_app.py --start_off             # host but start with camera off
```

---

## How it works

```
Webcam ─▶ MediaPipe FaceLandmarker ─▶ EAR · MAR · head-pose
                                          │
              PERCLOS + eye-state timers (wall-clock, FPS-independent)
                                          │
                       fusion → alert level + reason
              ┌───────────────────────────┼───────────────────────────┐
              ▼                            ▼                            ▼
       Dashboard overlay         AlertManager (bg thread)      PhoneDetector (bg thread)
       (gauges + badges)         buzzer + TTS                  YOLOv8n
```

- **Time-based trackers** use wall-clock seconds, so timing stays correct at any camera FPS.
- **Head-pose calibration** learns your neutral pitch from the first 30 frames (removes camera-angle bias).
- **Background threads** keep the buzzer and YOLO off the video loop — the feed never freezes.
- **Web hosting** streams the exact annotated frames to the browser via MJPEG (same look as the desktop window) — see [HOSTING.md](HOSTING.md).

---

## Tech stack

**Python** · OpenCV · MediaPipe Tasks (FaceLandmarker) · Ultralytics YOLOv8n + PyTorch · scikit-learn · TensorFlow/Keras · pygame / pyttsx3 / winsound · FastAPI + uvicorn

---

## Optional trained models

The app works out-of-the-box on geometric heuristics. Two optional trained models upgrade it (code wired in, training is data/GPU dependent) — full steps in [TRAINING.md](TRAINING.md):

- **Model A — Eye-state CNN** (MRL Eye Dataset): open/closed eye classifier that replaces the EAR threshold.
- **Model B — Temporal classifier** (GradientBoosting over a 30 s feature window): predicts drowsiness level + a "why it fired" feature-importance explanation.

---

## Project structure

```
main.py             Desktop pipeline (OpenCV window)
web_app.py          Web host (FastAPI + MJPEG stream)
utils.py            EAR/MAR/head-pose + PERCLOS & eye-state trackers
dashboard.py        On-frame overlay (gauges, badges, status)
alert.py            Escalating buzzer + text-to-speech (background thread)
phone_detector.py   YOLOv8n phone-use detection (background thread)
record_features.py  Record labelled data for Model B
train_eye_cnn.py    Train Model A          train_temporal.py  Train Model B
requirements.txt    Dependencies
PRD.md · HOSTING.md · TRAINING.md          Documentation
```

---

## Roadmap

Train Model A/B on real data · SHAP per-decision explanations · Indian-context validation set · mobile / IR-camera support · seatbelt & smoking detection · fleet dashboard (B2B).

---

*Built for a 36-hour AI/ML hackathon. License: add a `LICENSE` file of your choice (MIT recommended).*
