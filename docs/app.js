// DriveAwake — in-browser drowsiness & distraction detection.
// Runs entirely client-side: MediaPipe FaceLandmarker (WASM) for the face mesh,
// EAR/MAR/PERCLOS/head-pose ported from the Python app, and TF.js COCO-SSD for
// phone detection. Uses THIS device's camera. Nothing is uploaded.

import { FaceLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20";

// ── Config (mirrors the Python main.py) ──────────────────────────────────────
const EAR_T = 0.20, MAR_T = 0.45, P_MILD = 0.15, P_DROWSY = 0.25;
const PITCH_NOD = 15, CONSEC = 1.5, CLOSED_DROWSY = 2.0, CALIB_FRAMES = 30;

// MediaPipe landmark indices (same topology as the Python version)
const LEFT_EYE  = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const MOUTH = { top: 13, bottom: 14, left: 78, right: 308 };

// ── DOM ──────────────────────────────────────────────────────────────────────
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const cover = document.getElementById("cover");
const startBtn = document.getElementById("startBtn");
const recalBtn = document.getElementById("recalBtn");
const muteBtn = document.getElementById("muteBtn");
const phoneChk = document.getElementById("phoneChk");
const loadMsg = document.getElementById("loadMsg");
const dotEl = document.getElementById("dot");
const statusEl = document.getElementById("status");

const COLORS = ["#25c685", "#f3b53b", "#ff4d5e"], NAMES = ["ALERT", "MILD", "DROWSY"];

// ── State ────────────────────────────────────────────────────────────────────
let faceLandmarker = null, cocoModel = null, running = false;
let muted = false, curLevel = 0, curReason = "";
let pitchBaseline = null, pitchCalib = [];
let lastVideoTime = -1, prevT = performance.now(), fps = 0;
const phoneState = { detected: false, box: null, lastSeen: 0, hold: 700 };

// ── Helpers (ported from utils.py) ───────────────────────────────────────────
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const P = (lm, i, w, h) => ({ x: lm[i].x * w, y: lm[i].y * h });

function earOne(p) {
  return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]) + 1e-6);
}
function computeEAR(lm, w, h) {
  const L = LEFT_EYE.map(i => P(lm, i, w, h)), R = RIGHT_EYE.map(i => P(lm, i, w, h));
  return (earOne(L) + earOne(R)) / 2;
}
function computeMAR(lm, w, h) {
  const t = P(lm, MOUTH.top, w, h), b = P(lm, MOUTH.bottom, w, h);
  const l = P(lm, MOUTH.left, w, h), r = P(lm, MOUTH.right, w, h);
  return dist(t, b) / (dist(l, r) + 1e-6);
}
// Head pose (deg) from MediaPipe's facial transformation matrix (column-major).
function headPose(mat) {
  const d = mat.data;
  const R00 = d[0], R10 = d[1], R20 = d[2], R21 = d[6], R22 = d[10];
  const rad = 180 / Math.PI;
  return {
    pitch: Math.atan2(R21, R22) * rad,
    yaw: Math.atan2(-R20, Math.hypot(R21, R22)) * rad,
    roll: Math.atan2(R10, R00) * rad,
  };
}
const median = a => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// ── Time-based trackers (ported) ─────────────────────────────────────────────
class Perclos {
  constructor(winS = 30, thr = EAR_T) { this.win = winS * 1000; this.thr = thr; this.buf = []; }
  update(ear) {
    const now = performance.now();
    this.buf.push([now, ear < this.thr]);
    const cut = now - this.win;
    while (this.buf.length && this.buf[0][0] < cut) this.buf.shift();
    return this.value;
  }
  get value() {
    if (!this.buf.length) return 0;
    let c = 0; for (const [, x] of this.buf) if (x) c++;
    return c / this.buf.length;
  }
}
class EyeState {
  constructor(thrS = CONSEC, ear = EAR_T) { this.thrS = thrS; this.ear = ear; this.since = null; }
  update(ear) {
    const now = performance.now();
    if (ear < this.ear) { if (this.since == null) this.since = now; }
    else this.since = null;
  }
  get closedSeconds() { return this.since == null ? 0 : (performance.now() - this.since) / 1000; }
}
let perclos = new Perclos(), eye = new EyeState();

// ── Decision logic (ported from decide_level_heuristic) ──────────────────────
function decide(ear, mar, p, pitchRel, closedS) {
  const reasons = [];
  if (closedS >= CONSEC) reasons.push(`EYES CLOSED ${closedS.toFixed(1)}s`);
  if (p >= P_MILD) reasons.push(`PERCLOS ${Math.round(p * 100)}%`);
  if (mar >= MAR_T) reasons.push("YAWN detected");
  if (pitchRel > PITCH_NOD) reasons.push(`HEAD-NOD ${pitchRel.toFixed(0)}°`);
  let level = 0;
  if (p >= P_DROWSY || closedS >= CLOSED_DROWSY) level = 2;
  else if (p >= P_MILD || mar >= MAR_T || pitchRel > PITCH_NOD) level = 1;
  return { level, reason: reasons.join(" · ") };
}

// ── Audio: Web Audio buzzer + speech ─────────────────────────────────────────
let audioCtx = null, lastMild = 0, lastDrowsy = 0, lastTTS = 0;
function beep(freq, dur, vol = 0.35) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime, o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = "square"; o.frequency.value = freq;
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(vol, t + 0.01);
  g.gain.linearRampToValueAtTime(0, t + dur / 1000);
  o.connect(g).connect(audioCtx.destination);
  o.start(t); o.stop(t + dur / 1000 + 0.02);
}
function speak(text) {
  try { const u = new SpeechSynthesisUtterance(text); u.rate = 1; speechSynthesis.cancel(); speechSynthesis.speak(u); } catch (e) {}
}
setInterval(() => {
  if (muted || !audioCtx) return;
  const now = performance.now();
  if (curLevel >= 2) {
    if (now - lastDrowsy >= 480) { beep(900, 420, 0.4); lastDrowsy = now; }
    if (now - lastTTS >= 4000) { speak(curReason || "Wake up! Please pull over and rest."); lastTTS = now; }
  } else if (curLevel === 1) {
    if (now - lastMild >= 2500) { beep(660, 300, 0.3); lastMild = now; }
  }
}, 80);

// ── Drawing (replicates the dashboard overlay) ───────────────────────────────
function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function draw(f) {
  const w = canvas.width, h = canvas.height;
  ctx.drawImage(video, 0, 0, w, h);

  // status border
  ctx.lineWidth = Math.max(6, w * 0.012);
  ctx.strokeStyle = f.face ? COLORS[f.level] : "#8a97b5";
  ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, w - ctx.lineWidth, h - ctx.lineWidth);

  // metrics panel
  const pad = Math.round(w * 0.012), fs = Math.round(h * 0.032), lh = fs * 1.5;
  ctx.font = `${fs}px Consolas, monospace`;
  const lines = [
    `EAR  ${f.ear.toFixed(3)}`, `MAR  ${f.mar.toFixed(3)}`,
    `PERCLOS ${(f.perclos * 100).toFixed(0)}%`, `PITCH ${f.pitch.toFixed(0)}°`,
    `FPS  ${fps.toFixed(0)}`,
  ];
  const pw = w * 0.30, ph = lh * lines.length + pad;
  ctx.fillStyle = "rgba(8,12,24,.62)"; roundRect(pad, pad, pw, ph, 12); ctx.fill();
  ctx.fillStyle = "#e7ecf6"; ctx.textBaseline = "top";
  lines.forEach((t, i) => ctx.fillText(t, pad * 2, pad * 1.6 + i * lh));

  // status word (top-right)
  const sw = f.face ? (f.calibrating ? "CALIBRATING" : NAMES[f.level]) : "NO FACE";
  ctx.font = `bold ${Math.round(h * 0.05)}px Segoe UI, sans-serif`;
  ctx.textAlign = "right";
  ctx.fillStyle = f.face ? COLORS[f.level] : "#8a97b5";
  ctx.fillText(sw, w - pad * 2, pad * 1.6);
  ctx.textAlign = "left";

  // reason
  if (f.reason) {
    ctx.font = `${Math.round(h * 0.03)}px Segoe UI, sans-serif`;
    ctx.fillStyle = "#dfe6f5"; ctx.textAlign = "right";
    ctx.fillText(f.reason, w - pad * 2, pad * 1.6 + h * 0.06);
    ctx.textAlign = "left";
  }

  // YAWN badge
  if (f.yawn) {
    ctx.font = `bold ${Math.round(h * 0.04)}px Segoe UI, sans-serif`;
    const txt = "YAWN", tw = ctx.measureText(txt).width, bx = w / 2 - tw / 2 - 16;
    ctx.fillStyle = "rgba(243,181,59,.92)"; roundRect(bx, pad, tw + 32, h * 0.07, 20); ctx.fill();
    ctx.fillStyle = "#2a1d00"; ctx.fillText(txt, w / 2 - tw / 2, pad * 1.4);
  }

  // phone box + badge
  if (f.phone && f.box) {
    const [x, y, bw, bh] = f.box;
    ctx.lineWidth = Math.max(3, w * 0.006); ctx.strokeStyle = "#ff4d5e";
    ctx.strokeRect(x, y, bw, bh);
    ctx.font = `bold ${Math.round(h * 0.035)}px Segoe UI, sans-serif`;
    const txt = "PHONE USE", tw = ctx.measureText(txt).width;
    ctx.fillStyle = "#ff4d5e"; ctx.fillRect(x, Math.max(0, y - h * 0.05), tw + 18, h * 0.05);
    ctx.fillStyle = "#fff"; ctx.fillText(txt, x + 9, Math.max(0, y - h * 0.05) + 4);
  }

  // DROWSY banner
  if (f.level >= 2 && f.face) {
    ctx.fillStyle = "rgba(255,30,50,.86)"; ctx.fillRect(0, h - h * 0.12, w, h * 0.12);
    ctx.font = `bold ${Math.round(h * 0.06)}px Segoe UI, sans-serif`;
    ctx.fillStyle = "#fff"; ctx.textAlign = "center";
    ctx.fillText("⚠  WAKE UP!", w / 2, h - h * 0.10);
    ctx.textAlign = "left";
  }
}

// ── Phone detection loop (COCO-SSD, throttled) ───────────────────────────────
async function phoneLoop() {
  while (running) {
    if (cocoModel && phoneChk.checked) {
      try {
        const preds = await cocoModel.detect(video);
        let best = null, bs = 0.35;
        for (const p of preds) if (p.class === "cell phone" && p.score > bs) { bs = p.score; best = p.bbox; }
        if (best) { phoneState.lastSeen = performance.now(); phoneState.box = best; }
      } catch (e) { /* ignore a frame */ }
    }
    phoneState.detected = (performance.now() - phoneState.lastSeen) <= phoneState.hold;
    if (!phoneState.detected) phoneState.box = null;
    await new Promise(r => setTimeout(r, 350));
  }
}

// ── Main loop ────────────────────────────────────────────────────────────────
function loop() {
  if (!running) return;
  const now = performance.now();
  fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - prevT)); prevT = now;

  let f = { face: false, ear: 0.30, mar: 0, perclos: perclos.value, pitch: 0,
            level: 0, reason: "No face detected", yawn: false,
            phone: phoneState.detected, box: phoneState.box, calibrating: pitchBaseline == null };

  if (faceLandmarker && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    let res = null;
    try { res = faceLandmarker.detectForVideo(video, now); } catch (e) {}
    if (res && res.faceLandmarks && res.faceLandmarks.length) {
      const lm = res.faceLandmarks[0], w = canvas.width, h = canvas.height;
      const ear = computeEAR(lm, w, h), mar = computeMAR(lm, w, h);
      let pitch = 0, yaw = 0, roll = 0;
      if (res.facialTransformationMatrixes && res.facialTransformationMatrixes.length) {
        ({ pitch, yaw, roll } = headPose(res.facialTransformationMatrixes[0]));
      }
      if (pitchBaseline == null) {
        pitchCalib.push(pitch);
        if (pitchCalib.length >= CALIB_FRAMES) pitchBaseline = median(pitchCalib);
      }
      const pitchRel = pitch - (pitchBaseline == null ? pitch : pitchBaseline);
      perclos.update(ear); eye.update(ear);
      const d = decide(ear, mar, perclos.value, pitchRel, eye.closedSeconds);
      f = { face: true, ear, mar, perclos: perclos.value, pitch: pitchRel,
            level: d.level, reason: d.reason, yawn: mar >= MAR_T,
            phone: phoneState.detected, box: phoneState.box, calibrating: pitchBaseline == null };
    }
  }

  // phone escalates to level 2
  if (f.phone) {
    f.level = Math.max(f.level, 2);
    const tag = "PHONE USE - eyes off road";
    f.reason = (!f.reason || f.reason === "No face detected") ? tag : `${tag} | ${f.reason}`;
  }

  curLevel = f.level; curReason = f.reason;
  dotEl.style.background = f.face || f.phone ? COLORS[f.level] : "#8a97b5";
  dotEl.style.boxShadow = f.face || f.phone ? `0 0 8px ${COLORS[f.level]}` : "none";
  statusEl.textContent = f.calibrating && f.face ? "CALIBRATING…" : (f.face || f.phone ? NAMES[f.level] : "NO FACE");
  statusEl.style.color = f.face || f.phone ? COLORS[f.level] : "#8a97b5";

  draw(f);
  requestAnimationFrame(loop);
}

// ── Startup ──────────────────────────────────────────────────────────────────
async function start() {
  startBtn.disabled = true;
  try {
    loadMsg.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    loadMsg.textContent = "Loading face model…";
    const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO", numFaces: 1, outputFacialTransformationMatrixes: true,
    });

    loadMsg.textContent = "Loading phone detector…";
    try { cocoModel = await cocoSsd.load({ base: "lite_mobilenet_v2" }); }
    catch (e) { cocoModel = null; phoneChk.checked = false; }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    running = true;
    cover.classList.add("hide");
    recalBtn.disabled = false; muteBtn.disabled = false; phoneChk.disabled = !cocoModel;
    phoneLoop();
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    loadMsg.textContent = "Error: " + (err && err.message ? err.message : err) +
      " — allow camera access and use Chrome/Edge over HTTPS.";
  }
}

startBtn.addEventListener("click", start);
recalBtn.addEventListener("click", () => { pitchBaseline = null; pitchCalib = []; });
muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "🔇 Buzzer off" : "🔊 Buzzer on";
  muteBtn.classList.toggle("off", muted);
});
