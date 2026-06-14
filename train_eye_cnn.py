"""
train_eye_cnn.py — Model A: Eye-state CNN (open / closed)
Run this on Google Colab (free GPU) with the MRL Eye Dataset.

Dataset structure expected:
    data/
      openEyes/   ← images of open eyes
      closedEyes/ ← images of closed eyes

Usage (Colab):
    !python train_eye_cnn.py --data_dir /content/mrl --epochs 20 --output models/eye_cnn.h5

MRL Eye Dataset download:
    http://mrl.cs.vsb.cz/eyedataset
    (84,898 images, already labeled open/closed in folder names)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE   = 24      # 24×24 grayscale — fast, enough texture for eye state
BATCH_SIZE = 128
SEED       = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mrl_dataset(data_dir: str):
    """
    Load MRL Eye Dataset.
    Expects: data_dir/openEyes/ and data_dir/closedEyes/
    Returns X (N, 24, 24, 1) float32, y (N,) int {0=open, 1=closed}
    """
    import cv2

    OPEN_DIRS   = ["openEyes",   "open",   "Open",   "s0"]   # try common names
    CLOSED_DIRS = ["closedEyes", "closed", "Closed", "s1"]

    def find_dir(base, candidates):
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        # fallback: list all subdirs
        subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        raise FileNotFoundError(
            f"Could not find eye-state directory in {base}.\n"
            f"Found: {subdirs}\n"
            f"Expected one of: {candidates}"
        )

    open_dir   = find_dir(data_dir, OPEN_DIRS)
    closed_dir = find_dir(data_dir, CLOSED_DIRS)

    images, labels = [], []

    for label_val, directory in [(0, open_dir), (1, closed_dir)]:
        count = 0
        for fname in os.listdir(directory):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm")):
                continue
            path = os.path.join(directory, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_val)
            count += 1

        print(f"  {'Open' if label_val==0 else 'Closed'}: {count} images from {directory}")

    X = np.array(images, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]   # (N, 24, 24, 1)
    y = np.array(labels, dtype=np.int32)
    return X, y


# ── Model architecture ────────────────────────────────────────────────────────

def build_cnn(input_shape=(24, 24, 1)) -> keras.Model:
    """
    Lightweight custom CNN.
    ~50 k parameters — trains fast on Colab, runs on laptop CPU.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Head
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),   # 0=open, 1=closed
    ], name="eye_state_cnn")
    return model


def build_mobilenetv2(input_shape=(96, 96, 3)) -> keras.Model:
    """
    MobileNetV2 transfer-learning variant.
    Use this if accuracy with the small CNN is below 92%.
    Requires 96×96 RGB images — set IMG_SIZE=96 and load as colour.
    """
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False   # freeze for first 5 epochs, then unfreeze

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="mobilenetv2_eye")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    print(f"\n=== Eye-State CNN Training ===")
    print(f"Dataset: {args.data_dir}")
    print(f"Output:  {args.output}\n")

    # Load data
    print("Loading dataset...")
    X, y = load_mrl_dataset(args.data_dir)
    print(f"Total samples: {len(X)}  (open: {(y==0).sum()}, closed: {(y==1).sum()})")

    # Train/val/test split: 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=SEED, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.GaussianNoise(0.02),
    ])

    # Build model
    model = build_cnn()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # Build augmented dataset
    def augment(x, y):
        x = data_augmentation(x, training=True)
        return x, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"\nTraining for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n=== Test-set Evaluation ===")
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    y_pred = (model.predict(X_test, verbose=0).squeeze() >= 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Open", "Closed"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ── Save learning curves ──────────────────────────────────────────────────
    plot_path = args.output.replace(".h5", "_history.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="Train acc")
    ax1.plot(history.history["val_accuracy"], label="Val acc")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history["loss"],     label="Train loss")
    ax2.plot(history.history["val_loss"], label="Val loss")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"\nLearning curves saved to {plot_path}")

    # ── Export TFLite ────────────────────────────────────────────────────────
    tflite_path = args.output.replace(".h5", ".tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")
    print(f"\nBest model saved to {args.output}")
    print("Done!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,             help="Root dir with openEyes/ and closedEyes/")
    parser.add_argument("--output",   default="models/eye_cnn.h5", help="Output model path")
    parser.add_argument("--epochs",   type=int, default=25,      help="Max epochs")
    args = parser.parse_args()
    train(args)
