"""
web_app.py — Host the original detector (device camera + on-frame readings) on
a web page.

It runs the EXACT same pipeline and the EXACT same dashboard overlay as the
desktop `main.py` (MediaPipe FaceLandmarker → EAR/MAR/PERCLOS/head-pose, YOLOv8n
phone detection, the buzzer, head-pose calibration). The only difference is that
instead of opening an OpenCV window it encodes each annotated frame as JPEG and
streams it to the browser (MJPEG). So what you see in the browser is identical to
the desktop window — just hosted at a URL.

Run:
    python web_app.py                  # camera 0  → http://localhost:8000
    python web_app.py --camera 1 --port 8080
    python web_app.py --host 0.0.0.0   # also viewable from phones on the same
                                       # WiFi at http://<this-pc-ip>:8000
                                       # (plain HTTP is fine: the browser only
                                       #  *displays* a stream, it never opens a
                                       #  camera itself, so no HTTPS is needed)

The camera and the buzzer live on THIS computer, exactly like the desktop app.
On-page buttons: Camera On/Off · Recalibrate · Reset · Snapshot · Mute.

"Camera On/Off" releases / re-acquires the webcam WITHOUT stopping the server,
so the public URL stays live while the camera (and its light) turn off. Use
--start_off to launch already in standby.
"""

import argparse
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as mp_BaseOptions

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn

from utils import (
    get_ear, mouth_aspect_ratio, head_pose_angles,
    PerclosTracker, EyeStateTracker,
)
from dashboard import Dashboard
from alert import AlertManager
# Reuse the desktop app's config + decision logic so behaviour is identical.
from main import (
    EAR_THRESHOLD, MAR_THRESHOLD, CONSEC_CLOSED_S, FEATURE_WINDOW_S,
    decide_level_heuristic, _ensure_model,
)

CALIB_FRAMES = 30


# ── Detection worker (single shared camera, runs in a background thread) ──────
class Detector:
    def __init__(self, camera: int = 0, enable_phone: bool = True, start_paused: bool = False):
        self.camera = camera
        self.enable_phone = enable_phone

        self.latest_jpeg: bytes | None = None
        self.frame_id = 0
        self.fps = 0.0
        self.readings: dict = {}
        self.muted = False
        self.error: str | None = None

        self._recalib = threading.Event()
        self._reset = threading.Event()
        self._snap = threading.Event()
        self._stop = threading.Event()
        self._paused = threading.Event()
        if start_paused:
            self._paused.set()

        self._thread = threading.Thread(target=self._run, daemon=True)

    # public controls
    def start(self):       self._thread.start()
    def recalibrate(self): self._recalib.set()
    def reset(self):       self._reset.set()
    def snapshot(self):    self._snap.set()
    def toggle_mute(self): self.muted = not self.muted; return self.muted
    def pause(self):       self._paused.set()      # release the camera, keep hosting
    def resume(self):      self._paused.clear()    # re-acquire the camera
    @property
    def camera_on(self):   return not self._paused.is_set()
    def stop(self):        self._stop.set()

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            self.latest_jpeg = buf.tobytes()
            self.frame_id += 1

    def _error_frame(self, msg: str, w: int = 640, h: int = 480):
        img = np.zeros((h, w, 3), np.uint8)
        cv2.putText(img, "Camera error", (30, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, msg[:60], (30, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        self._publish(img)

    def _standby_frame(self, w: int = 640, h: int = 480):
        """Placeholder shown while the camera is off but the server stays up."""
        img = np.full((h, w, 3), (18, 14, 10), np.uint8)
        cv2.putText(img, "CAMERA OFF", (max(10, w // 2 - 150), h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 170, 90), 3, cv2.LINE_AA)
        cv2.putText(img, "Still hosted - press 'Camera On' to resume",
                    (max(10, w // 2 - 250), h // 2 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 160, 180), 1, cv2.LINE_AA)
        self._publish(img)

    def _run(self):
        # Camera-INDEPENDENT setup (created once; survives camera on/off cycles).
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_BaseOptions(model_asset_path=_ensure_model()),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        perclos_tracker = PerclosTracker(window_seconds=FEATURE_WINDOW_S, ear_threshold=EAR_THRESHOLD)
        eye_tracker     = EyeStateTracker(closed_seconds_threshold=CONSEC_CLOSED_S, ear_threshold=EAR_THRESHOLD)
        alert_manager   = AlertManager()

        phone_detector = None
        if self.enable_phone:
            try:
                from phone_detector import PhoneDetector
                phone_detector = PhoneDetector()
            except Exception as e:
                print(f"[Phone] disabled ({e})")

        os.makedirs("snapshots", exist_ok=True)

        cap = None
        dashboard = None
        frame_w, frame_h = 640, 480
        pitch_baseline = None
        pitch_calib: list[float] = []
        prev = time.time()
        standby_shown = False

        while not self._stop.is_set():
            # ── Camera OFF (standby): release the webcam, keep serving ───────
            if self._paused.is_set():
                if cap is not None:
                    cap.release(); cap = None
                alert_manager.update(0, "")              # silence the buzzer
                if not standby_shown:
                    self._standby_frame(frame_w, frame_h)
                    standby_shown = True
                self.readings = {
                    "level": 0, "reason": "Camera off (standby)",
                    "ear": 0.0, "mar": 0.0, "perclos": 0.0, "fps": 0.0,
                    "phone": False, "yawn": False, "calibrating": False,
                    "muted": self.muted, "camera_on": False,
                }
                self._stop.wait(0.2)
                continue
            standby_shown = False

            # ── Ensure the camera is open (fresh session on each resume) ─────
            if cap is None:
                cap = cv2.VideoCapture(self.camera)
                if not cap.isOpened():
                    cap = None
                    self.error = f"Cannot open camera {self.camera}"
                    self._error_frame(self.error, frame_w, frame_h)
                    self.readings = {
                        "level": 0, "reason": self.error, "ear": 0.0, "mar": 0.0,
                        "perclos": 0.0, "fps": 0.0, "phone": False, "yawn": False,
                        "calibrating": False, "muted": self.muted, "camera_on": False,
                    }
                    self._paused.set()                   # drop to standby, no busy-loop
                    continue
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                if dashboard is None:
                    dashboard = Dashboard(frame_w, frame_h)
                # fresh start each time the camera turns on
                perclos_tracker = PerclosTracker(window_seconds=FEATURE_WINDOW_S, ear_threshold=EAR_THRESHOLD)
                eye_tracker     = EyeStateTracker(closed_seconds_threshold=CONSEC_CLOSED_S, ear_threshold=EAR_THRESHOLD)
                pitch_baseline = None
                pitch_calib = []
                prev = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            if phone_detector is not None:
                phone_detector.submit(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_landmarker.detect(mp_image)

            ear = 0.30; mar = 0.0; pitch = 0.0; yaw = 0.0; roll = 0.0
            pitch_rel = 0.0; level = 0; reason = ""

            if results.face_landmarks:
                lms = results.face_landmarks[0]
                _, _, ear = get_ear(lms, frame_w, frame_h)
                mar = mouth_aspect_ratio(lms, frame_w, frame_h)
                pitch, yaw, roll = head_pose_angles(lms, frame_w, frame_h)

                if pitch_baseline is None:
                    pitch_calib.append(pitch)
                    if len(pitch_calib) >= CALIB_FRAMES:
                        pitch_baseline = float(np.median(pitch_calib))
                pitch_rel = pitch - (pitch_baseline if pitch_baseline is not None else pitch)

                perclos_tracker.update(ear)
                eye_tracker.update(ear)
                level, reason = decide_level_heuristic(ear, mar, perclos_tracker.value, pitch_rel, eye_tracker)
            else:
                reason = "No face detected"

            phone     = phone_detector.detected if phone_detector is not None else False
            phone_box = phone_detector.box      if phone_detector is not None else None
            if phone:
                level = max(level, 2)
                tag = "PHONE USE - eyes off road"
                reason = tag if (not reason or reason == "No face detected") else f"{tag} | {reason}"

            alert_manager.update(0 if self.muted else level, reason)

            frame = dashboard.draw(
                frame, ear, mar, perclos_tracker.value, pitch_rel, yaw, roll,
                level, reason, -1.0, -1.0,
                yawning=(mar >= MAR_THRESHOLD), phone=phone, phone_box=phone_box,
            )

            now = time.time()
            self.fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame_w - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # On-page control events
            if self._recalib.is_set():
                pitch_baseline = None; pitch_calib.clear(); self._recalib.clear()
            if self._reset.is_set():
                perclos_tracker = PerclosTracker(window_seconds=FEATURE_WINDOW_S, ear_threshold=EAR_THRESHOLD)
                eye_tracker     = EyeStateTracker(closed_seconds_threshold=CONSEC_CLOSED_S, ear_threshold=EAR_THRESHOLD)
                self._reset.clear()
            if self._snap.is_set():
                fn = os.path.join("snapshots", f"snap_{int(time.time())}.png")
                cv2.imwrite(fn, frame); self._snap.clear()

            self.readings = {
                "level": level, "reason": reason,
                "ear": round(ear, 3), "mar": round(mar, 3),
                "perclos": round(perclos_tracker.value, 3),
                "fps": round(self.fps, 1),
                "phone": bool(phone), "yawn": bool(mar >= MAR_THRESHOLD),
                "calibrating": pitch_baseline is None,
                "muted": self.muted, "camera_on": True,
            }
            self._publish(frame)

        if cap is not None:
            cap.release()
        face_landmarker.close()
        if phone_detector is not None:
            phone_detector.stop()
        alert_manager.stop()


# ── FastAPI app ──────────────────────────────────────────────────────────────
detector: Detector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fallback path when launched via `uvicorn web_app:app` (no CLI args);
    # the `python web_app.py` path builds the detector in main() first.
    global detector
    if detector is None:
        detector = Detector(
            camera=int(os.environ.get("CAMERA", "0")),
            enable_phone=os.environ.get("PHONE", "1") != "0",
        )
        detector.start()
    yield


app = FastAPI(title="DriveAwake", lifespan=lifespan)


def _mjpeg():
    last = -1
    while True:
        det = detector
        if det is not None and det.latest_jpeg is not None and det.frame_id != last:
            last = det.frame_id
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + det.latest_jpeg + b"\r\n")
        else:
            time.sleep(0.008)


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats():
    return JSONResponse(detector.readings if detector else {})


@app.post("/recalibrate")
def recalibrate():
    detector.recalibrate(); return {"ok": True}


@app.post("/reset")
def reset():
    detector.reset(); return {"ok": True}


@app.post("/snapshot")
def snapshot():
    detector.snapshot(); return {"ok": True}


@app.post("/mute")
def mute():
    return {"muted": detector.toggle_mute()}


@app.post("/camera/off")
def camera_off():
    detector.pause(); return {"camera_on": False}


@app.post("/camera/on")
def camera_on():
    detector.resume(); return {"camera_on": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DriveAwake — Drowsiness Detection</title>
<style>
  :root{--bg:#0b1020;--panel:#151d33;--line:#26314f;--text:#e7ecf6;--muted:#8a97b5;
        --green:#25c685;--amber:#f3b53b;--red:#ff4d5e;--blue:#5b8cff;}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:radial-gradient(1100px 600px at 80% -10%,#16213f,#0b1020 55%);
       color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center;padding:18px}
  h1{margin:0;font-size:21px} header p{margin:2px 0 0;color:var(--muted);font-size:12.5px}
  header{display:flex;gap:13px;align-items:center;width:100%;max-width:900px}
  .logo{font-size:28px;color:var(--blue);filter:drop-shadow(0 0 9px rgba(91,140,255,.6))}
  .pill{margin-left:auto;display:flex;align-items:center;gap:8px;background:var(--panel);
        border:1px solid var(--line);padding:7px 13px;border-radius:20px;font-size:13px;font-weight:700}
  .pill .dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}
  .stage{width:100%;max-width:900px;margin-top:14px;background:#05070f;border:1px solid var(--line);
         border-radius:14px;overflow:hidden;box-shadow:0 12px 30px rgba(0,0,0,.4);position:relative}
  .stage img{display:block;width:100%}
  .controls{display:flex;flex-wrap:wrap;gap:9px;margin-top:13px;width:100%;max-width:900px;justify-content:center}
  button{border:1px solid var(--line);background:var(--panel);color:var(--text);
         padding:10px 16px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;transition:.15s}
  button:hover{border-color:var(--blue);transform:translateY(-1px)}
  #mute.off,#cam.off{color:var(--muted)}
  #cam{border-color:var(--blue)}
  .foot{color:var(--muted);font-size:11.5px;margin-top:16px;text-align:center;max-width:760px;line-height:1.5}
  .tag{display:inline-block;padding:1px 8px;border-radius:6px;background:var(--panel);border:1px solid var(--line);font-size:12px}
</style></head><body>
  <header>
    <span class="logo">◉</span>
    <div><h1>DriveAwake</h1><p>Real-time Driver Drowsiness &amp; Distraction Detection</p></div>
    <div class="pill"><span class="dot" id="dot"></span><span id="status">ALERT</span></div>
  </header>

  <div class="stage"><img src="/video" alt="live detection"/></div>

  <div class="controls">
    <button id="cam" onclick="toggleCam()">⏸ Camera off</button>
    <button onclick="post('/recalibrate')">⟳ Recalibrate</button>
    <button onclick="post('/reset')">↺ Reset</button>
    <button onclick="post('/snapshot')">📸 Snapshot</button>
    <button id="mute" onclick="toggleMute()">🔊 Buzzer on</button>
  </div>

  <p class="foot">The video and the buzzer run on the computer hosting this page.
     Readings (EAR · MAR · PERCLOS · head-pose) are drawn directly on the frame —
     <span class="tag">green</span> alert · <span class="tag">yellow</span> mild ·
     <span class="tag">red</span> drowsy / phone use.</p>

<script>
  const COLORS=["#25c685","#f3b53b","#ff4d5e"], NAMES=["ALERT","MILD","DROWSY"];
  let cameraOn = true;
  async function post(url){ try{ await fetch(url,{method:"POST"}); }catch(e){} }
  function updateCamBtn(){
    const b=document.getElementById("cam");
    b.textContent = cameraOn ? "⏸ Camera off" : "▶ Camera on";
    b.classList.toggle("off", !cameraOn);
  }
  async function toggleCam(){
    const url = cameraOn ? "/camera/off" : "/camera/on";
    try{ const r=await(await fetch(url,{method:"POST"})).json(); cameraOn=r.camera_on; updateCamBtn(); }catch(e){}
  }
  async function toggleMute(){
    try{ const r=await(await fetch("/mute",{method:"POST"})).json();
      const b=document.getElementById("mute");
      b.textContent = r.muted ? "🔇 Buzzer off" : "🔊 Buzzer on";
      b.classList.toggle("off", r.muted);
    }catch(e){}
  }
  async function poll(){
    try{
      const s=await(await fetch("/stats")).json();
      if(typeof s.camera_on==="boolean" && s.camera_on!==cameraOn){ cameraOn=s.camera_on; updateCamBtn(); }
      const dot=document.getElementById("dot"), st=document.getElementById("status");
      if(s.camera_on===false){
        dot.style.background="#8a97b5"; dot.style.boxShadow="none";
        st.textContent="CAMERA OFF"; st.style.color="#8a97b5";
      } else {
        const lv=Math.min(s.level||0,2);
        dot.style.background=COLORS[lv]; dot.style.boxShadow="0 0 8px "+COLORS[lv];
        st.textContent = s.calibrating ? "CALIBRATING…" : NAMES[lv];
        st.style.color = COLORS[lv];
      }
    }catch(e){}
    setTimeout(poll,500);
  }
  updateCamBtn(); poll();
</script>
</body></html>"""


def main():
    global detector
    p = argparse.ArgumentParser(description="DriveAwake web host (MJPEG stream)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow LAN access")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no_phone", action="store_true")
    p.add_argument("--start_off", action="store_true",
                   help="start hosted but with the camera off (standby) until toggled on")
    args = p.parse_args()

    detector = Detector(camera=args.camera, enable_phone=not args.no_phone,
                        start_paused=args.start_off)
    detector.start()

    print(f"\n  DriveAwake  ->  http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
