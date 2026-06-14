"""
train_temporal.py — Model B: Temporal drowsiness classifier

Uses a 30-second sliding window of frame-level features (EAR, MAR, PERCLOS,
head pose, eye closure duration) aggregated into mean/std statistics, then
trained with GradientBoostingClassifier.

Output:
    models/temporal_gb.pkl   ← the trained model
    models/feature_scaler.pkl
    models/temporal_gb_report.txt

Usage:
    # From self-recorded features (fallback path):
    python train_temporal.py --data data/features.csv --window_s 30 --fps 30

    # From UTA-RLDD pre-extracted features:
    python train_temporal.py --data data/uta_rldd_features.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
SEED = 42
LABEL_NAMES = {0: "Alert", 1: "Mild", 2: "Drowsy"}


# ── Window aggregation ────────────────────────────────────────────────────────

FEATURE_COLS = ["ear", "mar", "perclos", "pitch", "yaw", "roll", "closed_s"]


def build_windows(df: pd.DataFrame, window_frames: int, step_frames: int):
    """
    Slide a fixed-length window over the frame-level data.
    Each window → one row of aggregated features + the modal label in the window.

    Returns X (n_windows, n_features), y (n_windows,)
    """
    rows, labels = [], []

    for i in range(0, len(df) - window_frames, step_frames):
        chunk = df.iloc[i : i + window_frames]
        row = {}
        for col in FEATURE_COLS:
            if col not in chunk.columns:
                continue
            vals = chunk[col].values
            row[f"{col}_mean"] = np.mean(vals)
            row[f"{col}_std"]  = np.std(vals)
            row[f"{col}_max"]  = np.max(vals)
            row[f"{col}_p75"]  = np.percentile(vals, 75)

        # Label = mode (most frequent label in window)
        label = chunk["label"].mode()[0]
        rows.append(row)
        labels.append(label)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    print("=== Temporal Classifier Training (Model B) ===\n")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} frame-level rows from {args.data}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    window_frames = int(args.window_s * args.fps)
    step_frames   = max(1, window_frames // 3)   # 66% overlap
    print(f"Window: {args.window_s}s × {args.fps}fps = {window_frames} frames (step={step_frames})")

    X, y = build_windows(df, window_frames, step_frames)
    print(f"Windows built: {len(X)}  |  Features: {X.shape[1]}")
    print(f"Window label distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model: GradientBoosting ───────────────────────────────────────────────
    print("Training GradientBoostingClassifier...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=SEED,
        verbose=0,
    )
    gb.fit(X_train_s, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(gb, X_train_s, y_train, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Test evaluation
    y_pred = gb.predict(X_test_s)
    test_acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_acc*100:.2f}%\n")

    report = classification_report(
        y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    )
    print("Classification Report:")
    print(report)

    # ── Feature importance ────────────────────────────────────────────────────
    feat_names = list(X.columns)
    importances = gb.feature_importances_
    top_k = np.argsort(importances)[-10:][::-1]
    print("Top-10 feature importances:")
    for i in top_k:
        print(f"  {feat_names[i]:35s}  {importances[i]:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(gb,     "models/temporal_gb.pkl")
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print(f"\nModel saved to models/temporal_gb.pkl")
    print(f"Scaler saved to models/feature_scaler.pkl")

    # Save report
    with open("models/temporal_gb_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc*100:.2f}%\n\n")
        f.write(f"CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n\n")
        f.write(report)
        f.write("\nTop feature importances:\n")
        for i in top_k:
            f.write(f"  {feat_names[i]:35s}  {importances[i]:.4f}\n")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    )
    disp.plot(cmap="Blues")
    plt.title("Model B — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/temporal_gb_cm.png", dpi=120)
    print("Confusion matrix saved to models/temporal_gb_cm.png")

    # Feature importance bar chart
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), importances[top_k])
    plt.xticks(range(10), [feat_names[i] for i in top_k], rotation=45, ha="right")
    plt.title("Model B — Top Feature Importances (used in 'Why it fired')")
    plt.tight_layout()
    plt.savefig("models/temporal_gb_importance.png", dpi=120)
    print("Feature importance chart saved to models/temporal_gb_importance.png")

    print("\nDone!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True, help="CSV of frame-level features (from record_features.py)")
    parser.add_argument("--window_s", type=int, default=30, help="Window length in seconds")
    parser.add_argument("--fps",      type=int, default=30, help="Frame rate used during recording")
    args = parser.parse_args()
    train(args)
