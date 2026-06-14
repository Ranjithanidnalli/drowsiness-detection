# Hosting the detector on a web page (DriveAwake)

This hosts the **exact desktop app** — the device camera with the readings drawn
on the frame — at a URL. Same pipeline as `main.py` (MediaPipe FaceLandmarker →
EAR / MAR / PERCLOS / head-pose, YOLOv8n phone detection, the buzzer, head-pose
calibration). The only change: instead of an OpenCV window, each annotated frame
is JPEG-encoded and streamed to the browser (MJPEG). What you see in the browser
is identical to the desktop window.

## Run it (one command)

```bash
python web_app.py                  # camera 0  → http://localhost:8000
python web_app.py --camera 1 --port 8080
python web_app.py --no_phone       # skip YOLO (lighter)
```

Open **http://localhost:8000**. The camera turns on automatically and the live,
annotated feed appears. On-page buttons: **Recalibrate · Reset · Snapshot · Mute**.

> ⚠ Only one program can use the webcam at a time. Quit the desktop `main.py`
> window (press **Q**) before starting `web_app.py`, or you'll get a camera error.

## How it works

```
   ┌────────────── THIS COMPUTER ──────────────┐          ┌──────── any browser ────────┐
   │ web_app.py (FastAPI + uvicorn)             │          │                              │
   │  webcam → MediaPipe + YOLO + trackers      │  MJPEG   │  <img src="/video">          │
   │  → dashboard.py draws the overlay          │ ───────▶ │  shows the annotated stream  │
   │  → cv2.imencode(JPEG) → /video stream      │  (HTTP)  │  buttons POST /recalibrate…  │
   │  buzzer plays on THIS computer's speakers  │          │  status pill polls /stats    │
   └────────────────────────────────────────────┘          └──────────────────────────────┘
```

The camera and the buzzer live on the computer running `web_app.py`, exactly like
the desktop app. The browser is just a viewer of the stream — it never opens a
camera itself, which is why plain HTTP works even over the LAN (no HTTPS needed).

### Endpoints
| Route          | Method | Purpose                                  |
|----------------|--------|------------------------------------------|
| `/`            | GET    | the page (video + controls)              |
| `/video`       | GET    | MJPEG stream of the annotated feed       |
| `/stats`       | GET    | JSON readings (level, ear, mar, fps…)    |
| `/recalibrate` | POST   | reset the neutral head-pose baseline     |
| `/reset`       | POST   | clear PERCLOS / eye-state trackers       |
| `/snapshot`    | POST   | save the current frame to `snapshots/`   |
| `/mute`        | POST   | toggle the buzzer                         |

## View it on a phone (same WiFi)

Because the browser only *displays* a stream, no HTTPS is required:

```bash
python web_app.py --host 0.0.0.0
```

Find this PC's IP (`ipconfig` → IPv4, e.g. `192.168.1.20`) and open
`http://192.168.1.20:8000` on the phone. The phone shows the laptop's camera +
detection. (The buzzer still sounds on the laptop.)

## Make it a public shareable URL (optional)

Run the server, then point a free tunnel at it — gives an HTTPS link anyone can open:

```bash
python web_app.py                                  # terminal 1
cloudflared tunnel --url http://localhost:8000     # terminal 2 → prints an https URL
# or:  ngrok http 8000
```

Detection still runs on your laptop; the tunnel just exposes the page.

## Notes
- First run downloads `face_landmarker.task` (~10 MB) and, if phone detection is
  on, uses `yolov8n.pt` (already in the repo).
- For a hackathon demo, `http://localhost:8000` on the laptop is the most reliable
  option; keep a Cloudflare tunnel ready if you want a link to share.
