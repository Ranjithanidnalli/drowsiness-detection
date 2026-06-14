/**
 * make_slides.js — Design Agenda deck · Driver Drowsiness Detection (v2)
 * Fixes: margins, overlap, contrast, column alignment, card spacing.
 * Run: node make_slides.js
 */

const pptxgen = require("pptxgenjs");

// ── Palette ────────────────────────────────────────────────────────────────
const C = {
  BG_DARK:  "0A1628",
  BG_LIGHT: "F0F4F8",
  PRIMARY:  "065A82",
  TEAL:     "1C7293",
  ACCENT:   "02C39A",
  DARK:     "1E293B",
  MUTE:     "64748B",
  WHITE:    "FFFFFF",
  DANGER:   "E53E3E",
  CARD_BD:  "CBD5E1",
};

const TITLE_FONT = "Georgia";
const BODY_FONT  = "Calibri";
const SW = 10.0;   // slide width
const SH = 5.625;  // slide height
const HBAR_H = 0.72; // header bar height (slides 2-5)

function mkShadow() {
  return { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.14 };
}

// ── Slide 1 — Title (dark) ────────────────────────────────────────────────
function slide1(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.BG_DARK };

  // Left accent stripe (thicker — 0.22" so it's clearly intentional)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: SH,
    fill: { color: C.ACCENT }, line: { color: C.ACCENT }
  });

  // Bottom bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.22, y: SH - 0.75, w: SW - 0.22, h: 0.75,
    fill: { color: C.PRIMARY }, line: { color: C.PRIMARY }
  });

  // Title — vertically centered in the upper 3/4 of the slide
  slide.addText("DRIVER DROWSINESS DETECTION", {
    x: 0.5, y: 1.1, w: 9.1, h: 1.05,
    fontFace: TITLE_FONT, fontSize: 38, bold: true, color: C.WHITE,
    align: "center", valign: "middle", margin: 0
  });

  // Accent underline
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 2.2, y: 2.25, w: 5.6, h: 0.06,
    fill: { color: C.ACCENT }, line: { color: C.ACCENT }
  });

  // Subtitle pitch
  slide.addText(
    "A driver-monitoring system that learns the temporal pattern of drowsiness onset\n— and tells you why it fired before you nod off.",
    {
      x: 0.8, y: 2.45, w: 8.4, h: 1.0,
      fontFace: BODY_FONT, fontSize: 15, italic: true, color: "A0C4D8",
      align: "center", valign: "middle", margin: 0
    }
  );

  // Footer text
  slide.addText("AIML Track  ·  Design Agenda  ·  Ranjitha", {
    x: 0.5, y: SH - 0.72, w: SW - 0.7, h: 0.72,
    fontFace: BODY_FONT, fontSize: 13, color: C.WHITE,
    align: "center", valign: "middle", margin: 0
  });
}

// ── Slide header bar (reused on slides 2–5) ───────────────────────────────
function addHeader(slide, title, slideNum) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: SW, h: HBAR_H,
    fill: { color: C.PRIMARY }, line: { color: C.PRIMARY }
  });
  slide.addText(title, {
    x: 0.35, y: 0, w: SW - 1.0, h: HBAR_H,
    fontFace: BODY_FONT, fontSize: 16, bold: true, color: C.WHITE,
    align: "left", valign: "middle", margin: 0
  });
  // Slide number — right side of header bar
  slide.addText(`${slideNum} / 5`, {
    x: SW - 0.65, y: 0, w: 0.55, h: HBAR_H,
    fontFace: BODY_FONT, fontSize: 11, color: "A0C4D8",
    align: "right", valign: "middle", margin: 0
  });
}

// ── Slide 2 — Problem & Impact ────────────────────────────────────────────
function slide2(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.BG_LIGHT };
  addHeader(slide, "PROBLEM RELEVANCE & IMPACT", 2);

  // 3 stat cards — equal width, proper left margin
  const cardW = 2.8, cardH = 1.5, cardY = 0.88;
  const cardXs = [0.45, 3.6, 6.75];

  const stats = [
    { val: "1.73 L", label: "road deaths\nin India (MoRTH 2023)" },
    { val: "12.1",   label: "deaths per 100k people\nvs 9.0 global avg (WHO)" },
    { val: "90%+",   label: "commercial vehicles with\nno driver monitoring" },
  ];

  stats.forEach((s, i) => {
    const x = cardXs[i];
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: cardW, h: cardH,
      fill: { color: C.WHITE }, line: { color: C.CARD_BD, width: 0.5 }, shadow: mkShadow()
    });
    // Left accent
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: 0.09, h: cardH,
      fill: { color: C.ACCENT }, line: { color: C.ACCENT }
    });
    slide.addText(s.val, {
      x: x + 0.14, y: cardY + 0.12, w: cardW - 0.18, h: 0.75,
      fontFace: TITLE_FONT, fontSize: 30, bold: true, color: C.PRIMARY,
      align: "center", valign: "middle", margin: 0
    });
    slide.addText(s.label, {
      x: x + 0.14, y: cardY + 0.88, w: cardW - 0.18, h: 0.55,
      fontFace: BODY_FONT, fontSize: 11, color: C.MUTE,
      align: "center", valign: "top", margin: 0
    });
  });

  // Dark callout box — "the gap"
  const boxY = 2.58;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: boxY, w: 9.1, h: 1.25,
    fill: { color: C.DARK }, line: { color: C.DARK }, shadow: mkShadow()
  });
  slide.addText([
    { text: "The gap:  ", options: { bold: true, color: C.ACCENT } },
    {
      text: "Drowsiness leaves no skid marks — it's logged as \"driver error.\" Existing monitoring " +
            "(Bosch, GM Super Cruise) needs dedicated IR hardware in premium vehicles. Our bet: " +
            "software-only detection running on a ₹500 webcam or phone, fully offline.",
      options: { color: C.WHITE }
    }
  ], {
    x: 0.65, y: boxY + 0.08, w: 8.75, h: 1.08,
    fontFace: BODY_FONT, fontSize: 12.5, valign: "middle", margin: 0
  });

  // Quote (with clear gap below dark box)
  slide.addText(
    "\"Fatigue kills on Indian highways and isn't even measured. We make detection a software problem, not a luxury hardware one.\"",
    {
      x: 0.45, y: 4.05, w: 9.1, h: 0.78,
      fontFace: TITLE_FONT, fontSize: 13.5, italic: true, color: C.PRIMARY,
      align: "center", valign: "middle", margin: 0
    }
  );
}

// ── Slide 3 — Architecture ────────────────────────────────────────────────
function slide3(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.BG_LIGHT };
  addHeader(slide, "SYSTEM ARCHITECTURE", 3);

  // 5 pipeline boxes — even gaps, centered vertically in upper portion
  const bW = 1.62, bH = 1.15, bY = 0.92, gap = 0.22;
  const totalW = 5 * bW + 4 * gap;
  const startX = (SW - totalW) / 2;

  const boxes = [
    { label: "Webcam\nFrame",              color: C.TEAL },
    { label: "MediaPipe\nFace Mesh",       color: C.PRIMARY },
    { label: "Model A\nEye CNN\n(open/closed)", color: "7C3AED" },
    { label: "Model B\nTemporal\nClassifier",  color: "B45309" },
    { label: "Alert +\nDashboard",        color: C.DANGER },
  ];

  boxes.forEach((b, i) => {
    const bx = startX + i * (bW + gap);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: bY, w: bW, h: bH,
      fill: { color: b.color }, line: { color: b.color }, shadow: mkShadow()
    });
    slide.addText(b.label, {
      x: bx, y: bY, w: bW, h: bH,
      fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: C.WHITE,
      align: "center", valign: "middle", margin: 0
    });
    // Arrow (except after last)
    if (i < 4) {
      const ax = bx + bW, ay = bY + bH / 2;
      slide.addShape(pres.shapes.LINE, {
        x: ax, y: ay, w: gap, h: 0,
        line: { color: C.DARK, width: 1.8 }
      });
    }
  });

  // Feature signals (below boxes, flowing into Model B)
  const signalY = bY + bH + 0.32;
  const signals = [
    { txt: "EAR · MAR\n(eye & yawn)", x: startX + (bW + gap) * 1 },
    { txt: "PERCLOS\n(rolling 30 s)", x: startX + (bW + gap) * 1 + 1.5 },
    { txt: "Head-Pose\n(pitch/nod)",  x: startX + (bW + gap) * 1 + 3.0 },
  ];
  signals.forEach((s) => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: s.x, y: signalY, w: 1.35, h: 0.68,
      fill: { color: "DBEAFE" }, line: { color: C.TEAL, width: 1 }
    });
    slide.addText(s.txt, {
      x: s.x, y: signalY, w: 1.35, h: 0.68,
      fontFace: BODY_FONT, fontSize: 9.5, color: C.PRIMARY,
      align: "center", valign: "middle", margin: 0
    });
  });

  // Arrow label
  slide.addText("↓  All signals → 30-second window → Model B", {
    x: startX + bW + gap, y: signalY + 0.72, w: 4.2, h: 0.3,
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: C.MUTE,
    align: "left", margin: 0
  });

  // WHY panel (bottom right, larger for readability)
  const wpX = SW - 2.85, wpY = signalY;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: wpX, y: wpY, w: 2.5, h: 1.08,
    fill: { color: "FFF7ED" }, line: { color: "D97706", width: 1.5 }, shadow: mkShadow()
  });
  slide.addText([
    { text: "\"WHY\" panel\n", options: { bold: true, color: "92400E", fontSize: 11 } },
    { text: "EYES CLOSED 2.1s\nPERCLOS 41%  ·  head-nod", options: { color: "B45309", fontSize: 10 } }
  ], {
    x: wpX + 0.1, y: wpY + 0.06, w: 2.3, h: 0.96,
    fontFace: BODY_FONT, valign: "top", margin: 0
  });

  // Safety net note
  slide.addText(
    "Safety net: PERCLOS heuristic always active — demo works even without ML models loaded.",
    {
      x: 0.45, y: SH - 0.44, w: SW - 0.9, h: 0.34,
      fontFace: BODY_FONT, fontSize: 10, italic: true, color: C.MUTE,
      align: "center", margin: 0
    }
  );
}

// ── Slide 4 — Tech Stack & Novelty ────────────────────────────────────────
function slide4(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.BG_LIGHT };
  addHeader(slide, "TECH STACK  ·  NOVELTY  ·  DIFFERENTIATORS", 4);

  // ── Left column: 4 tech cards ─────────────────────────────────
  const lcX = 0.38, lcW = 4.5;
  const cardH = 0.92, firstY = 0.88, lcGap = 0.18;
  const lcYs = [firstY, firstY + cardH + lcGap, firstY + 2 * (cardH + lcGap), firstY + 3 * (cardH + lcGap)];

  const stackItems = [
    { icon: "CV", title: "MediaPipe Face Mesh", body: "468 landmarks + iris, CPU real-time. No GPU." },
    { icon: "AI", title: "Keras CNN — Model A",  body: "MobileNetV2 / custom CNN on MRL Eye Dataset (84k images), ~95% accuracy target." },
    { icon: "ML", title: "GradientBoosting — Model B", body: "30-sec window → Alert / Mild / Drowsy + feature importances for explainability." },
    { icon: "UI", title: "OpenCV Dashboard",    body: "Live signal plots, gauge, \"why\" panel. Offline. No Streamlit risk." },
  ];

  stackItems.forEach((item, i) => {
    const cy = lcYs[i];
    slide.addShape(pres.shapes.RECTANGLE, {
      x: lcX, y: cy, w: lcW, h: cardH,
      fill: { color: C.WHITE }, line: { color: C.CARD_BD, width: 0.5 }, shadow: mkShadow()
    });
    // Icon circle
    slide.addShape(pres.shapes.OVAL, {
      x: lcX + 0.2, y: cy + (cardH - 0.42) / 2, w: 0.42, h: 0.42,
      fill: { color: C.PRIMARY }, line: { color: C.PRIMARY }
    });
    slide.addText(item.icon, {
      x: lcX + 0.2, y: cy + (cardH - 0.42) / 2, w: 0.42, h: 0.42,
      fontFace: BODY_FONT, fontSize: 12, bold: true, color: C.WHITE,
      align: "center", valign: "middle", margin: 0
    });
    slide.addText(item.title, {
      x: lcX + 0.72, y: cy + 0.1, w: lcW - 0.82, h: 0.28,
      fontFace: BODY_FONT, fontSize: 12.5, bold: true, color: C.DARK,
      align: "left", valign: "middle", margin: 0
    });
    slide.addText(item.body, {
      x: lcX + 0.72, y: cy + 0.38, w: lcW - 0.82, h: cardH - 0.42,
      fontFace: BODY_FONT, fontSize: 10.5, color: C.MUTE,
      align: "left", valign: "top", margin: 0
    });
  });

  // ── Right column: novelty rows (matching bottom of left column) ──────────
  const rcX = lcX + lcW + 0.42;   // 0.42" gutter
  const rcW = SW - rcX - 0.38;

  // Right column header (with gap below slide header)
  const rcHeaderY = 0.88;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: rcX, y: rcHeaderY, w: rcW, h: 0.36,
    fill: { color: C.PRIMARY }, line: { color: C.PRIMARY }
  });
  slide.addText("WHAT MAKES THIS REAL AI (not a dummy project)", {
    x: rcX + 0.1, y: rcHeaderY, w: rcW - 0.12, h: 0.36,
    fontFace: BODY_FONT, fontSize: 10, bold: true, color: C.WHITE,
    align: "left", valign: "middle", margin: 0
  });

  // 4 diff rows — heights matched to left column cards
  const diffs = [
    ["EAR fixed-threshold", "→ CNN learned eye-state probability"],
    ["Single-frame beep",   "→ 30-sec temporal fusion model"],
    ["No explanation",      "→ Live \"why it fired\" explainability panel"],
    ["GPU / IR hardware",   "→ CPU + ₹500 webcam, fully offline"],
  ];

  const rcRowH = cardH;
  const rcRowGap = lcGap;
  diffs.forEach((row, i) => {
    const ry = rcHeaderY + 0.42 + i * (rcRowH + rcRowGap);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: rcX, y: ry, w: rcW, h: rcRowH,
      fill: { color: C.WHITE }, line: { color: C.CARD_BD, width: 0.5 }, shadow: mkShadow()
    });
    // ❌ old way (strike-through) left side
    slide.addText(row[0], {
      x: rcX + 0.15, y: ry + 0.08, w: rcW - 0.25, h: 0.34,
      fontFace: BODY_FONT, fontSize: 11, color: C.DANGER, strikeThrough: true,
      align: "left", margin: 0
    });
    // ✅ new way right side
    slide.addText(row[1], {
      x: rcX + 0.15, y: ry + 0.42, w: rcW - 0.25, h: 0.42,
      fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: C.PRIMARY,
      align: "left", margin: 0
    });
  });
}

// ── Slide 5 — Metrics & Demo Moment (dark) ────────────────────────────────
function slide5(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.BG_DARK };
  addHeader(slide, "EXPECTED METRICS  ·  DEMO MOMENT", 5);

  // 3 metric cards — equal width, proper left margin
  const cW = 2.8, cH = 1.55, cY = 0.86;
  const cXs = [0.45, 3.6, 6.75];

  const metrics = [
    { val: "≥ 95%",  label: "Model A Accuracy",     sub: "MRL Eye Dataset · held-out test split" },
    { val: "≥ 85%",  label: "Model B Accuracy",     sub: "5-fold CV · Alert / Mild / Drowsy" },
    { val: "< 2 s",  label: "Alert Latency",         sub: "EAR drop → PERCLOS → alarm + TTS" },
  ];

  metrics.forEach((m, i) => {
    const cx = cXs[i];
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cY, w: cW, h: cH,
      fill: { color: C.PRIMARY }, line: { color: C.TEAL, width: 1.2 }, shadow: mkShadow()
    });
    // Stat number — with clear top padding
    slide.addText(m.val, {
      x: cx + 0.1, y: cY + 0.2, w: cW - 0.2, h: 0.72,
      fontFace: TITLE_FONT, fontSize: 30, bold: true, color: C.ACCENT,
      align: "center", valign: "middle", margin: 0
    });
    slide.addText(m.label, {
      x: cx + 0.1, y: cY + 0.94, w: cW - 0.2, h: 0.3,
      fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: C.WHITE,
      align: "center", margin: 0
    });
    slide.addText(m.sub, {
      x: cx + 0.1, y: cY + 1.25, w: cW - 0.2, h: 0.26,
      fontFace: BODY_FONT, fontSize: 10, color: "A0C4D8",
      align: "center", margin: 0
    });
  });

  // Demo moment box
  const dmY = 2.60;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: dmY, w: 9.1, h: 1.7,
    fill: { color: "0D2137" }, line: { color: C.ACCENT, width: 1.5 }, shadow: mkShadow()
  });
  slide.addText("THE DEMO MOMENT", {
    x: 0.65, y: dmY + 0.12, w: 8.7, h: 0.28,
    fontFace: BODY_FONT, fontSize: 10.5, bold: true, color: C.ACCENT,
    align: "left", margin: 0
  });
  slide.addText([
    { text: "1. Judge sits — dashboard shows green / \"Alert.\"\n" },
    { text: "2. They close eyes / yawn / nod → gauge spikes red → alarm fires.\n" },
    { text: "3. \"WHY\" panel lights up: " },
    { text: "EYES CLOSED 2.1s · PERCLOS 41% · head-nod detected.", options: { color: C.ACCENT, bold: true } },
    { text: "\n\nThe alarm gets attention. The WHY panel is what makes engineer-judges nod.", options: { italic: true, color: "A0C4D8" } },
  ], {
    x: 0.65, y: dmY + 0.44, w: 8.7, h: 1.2,
    fontFace: BODY_FONT, fontSize: 12.5, color: C.WHITE, valign: "top", margin: 0
  });

  // Closing tagline (white, high contrast)
  slide.addText("Two trained models  ·  one explainable system  ·  runs on any webcam", {
    x: 0.45, y: 4.55, w: 9.1, h: 0.42,
    fontFace: TITLE_FONT, fontSize: 13, italic: true, color: C.WHITE,
    align: "center", valign: "middle", margin: 0
  });
}

// ── Build ─────────────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title  = "Driver Drowsiness Detection – Design Agenda";

// Bind addHeader to pres
function addHeader(slide, title, num) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: SW, h: HBAR_H,
    fill: { color: C.PRIMARY }, line: { color: C.PRIMARY }
  });
  slide.addText(title, {
    x: 0.35, y: 0, w: SW - 1.1, h: HBAR_H,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: C.WHITE,
    align: "left", valign: "middle", margin: 0
  });
  slide.addText(`${num} / 5`, {
    x: SW - 0.65, y: 0, w: 0.55, h: HBAR_H,
    fontFace: BODY_FONT, fontSize: 11, color: "A0C4D8",
    align: "right", valign: "middle", margin: 0
  });
}

slide1(pres);
slide2(pres);
slide3(pres);
slide4(pres);
slide5(pres);

pres.writeFile({ fileName: "drowsiness_detection_slides.pptx" }).then(() => {
  console.log("✅  Saved: drowsiness_detection_slides.pptx");
}).catch(console.error);
