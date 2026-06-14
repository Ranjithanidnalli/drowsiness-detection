# Training the ML models (Model A & Model B)

The app runs out-of-the-box on geometric heuristics (EAR / MAR / PERCLOS / head-pose).
The two ML models are **optional upgrades** that make it "real trained AI" for judging.
Neither can be trained on this laptop (no GPU / dataset here) — train them on **Google
Colab (free GPU)**, download the weights, then point the app at them.

---

## Model A — Eye-state CNN  (`--cnn`)

Classifies each eye crop **open vs closed** with a small custom CNN (~50k params).
Dataset: **MRL Eye Dataset** (~84k infrared eye images).

### Colab steps (≈30–45 min incl. download)

```python
# 1. Setup
!pip -q install tensorflow opencv-python-headless scikit-learn matplotlib
# Upload train_eye_cnn.py via the Colab file panel, or:
# from google.colab import files; files.upload()

# 2. Get the MRL Eye Dataset (open/closed eye crops)
!wget -q http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip
!unzip -q mrlEyes_2018_01.zip -d mrl_data
# The loader auto-detects open/closed subfolders (s0/s1 or openEyes/closedEyes).

# 3. Train (GPU runtime: Runtime > Change runtime type > T4 GPU)
!python train_eye_cnn.py --data_dir mrl_data --output models/eye_cnn.h5 --epochs 25

# 4. Download the trained model
from google.colab import files
files.download('models/eye_cnn.h5')
```

> If the small CNN tops out below ~92% accuracy, switch to the included
> `build_mobilenetv2()` (transfer learning) — set `IMG_SIZE=96` and load images as RGB.

### Use it locally
Put `eye_cnn.h5` in `models/`, then:
```
python main.py --cnn models/eye_cnn.h5
```
The dashboard's gauge will then be driven by the CNN's closed-eye probability
instead of the raw EAR threshold.

---

## Model B — Temporal drowsiness classifier  (`--model_b`)

A GradientBoosting model over a **30-second window** of aggregated features
(mean/std/max/p75 of EAR, MAR, PERCLOS, pitch, yaw, roll, closed_s).
Powers the "why it fired" feature-importance explainability panel.

### Step 1 — Record labelled data (on THIS laptop, webcam)
Act out each state for ~5 min. `record_features.py` is already migrated to the new
MediaPipe API, so it runs here:
```
python record_features.py --label 0 --duration 300 --out data/features.csv   # Alert
python record_features.py --label 1 --duration 300 --out data/features.csv   # Mild
python record_features.py --label 2 --duration 300 --out data/features.csv   # Drowsy
```
(All three append to the same CSV.) Alternatively, feed UTA-RLDD videos with
`--video path/to/video.avi` instead of the webcam.

### Step 2 — Train (fast, runs on CPU — laptop or Colab)
```
python train_temporal.py --data data/features.csv --window_s 30 --fps 30
```
Outputs `models/temporal_gb.pkl` + `models/feature_scaler.pkl` + a report.

> Note: train and inference feature schemas are now aligned (28 features), so the
> model loads cleanly into the app.

### Use it locally
```
python main.py --model_b models/temporal_gb.pkl --scaler models/feature_scaler.pkl
```

---

## Combine everything
```
python main.py --cnn models/eye_cnn.h5 --model_b models/temporal_gb.pkl --scaler models/feature_scaler.pkl
```
Phone-use detection (YOLOv8n) and the buzzer are always on by default
(`--no_phone` to disable phone detection).

## "Indian context" differentiator (optional, high judge-value)
Record a short validation set with `record_features.py` on real Indian driving
footage (or act it), and report the app's accuracy on it. Even a small India-specific
validation slide invalidates the Western-dataset competition — cheap, high-impact.
