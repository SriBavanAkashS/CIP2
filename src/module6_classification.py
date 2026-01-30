import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for Module 6. "
        "Install it (e.g., `pip install tensorflow`) and retry."
    ) from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module6_classification")


# --------------------------------------------------
# DATA LOADING HELPERS
# --------------------------------------------------
def load_temporal_features(path=None):
    """Load Module 5 temporal features (subjects, trials, feat_dim)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_features.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 5 temporal_features.npy not found at: {path}")

    feats = np.load(path)
    if feats.ndim != 3:
        raise ValueError(f"Expected temporal_features with shape (S, T, F). Got {feats.shape}")
    return feats


def load_deap_labels(path=None):
    """Load Module 1 labels (subjects, trials, 4)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module1_data_loading", "eeg_labels.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 1 eeg_labels.npy not found at: {path}")
    labels = np.load(path)
    if labels.ndim != 3 or labels.shape[2] != 4:
        raise ValueError(f"Expected labels shape (S, T, 4). Got {labels.shape}")
    return labels


def labels_valence_3class(eeg_labels):
    """
    Convert DEAP valence ratings (1–9) into 3 emotion classes (C=3), per AlgoModule6.

    Class mapping (valence dimension index 0):
      0: low   (1–3)
      1: medium(4–6)
      2: high  (7–9)
    """
    valence = eeg_labels[..., 0]  # (S, T)
    y = np.zeros_like(valence, dtype=np.int32)
    y[valence >= 4] = 1
    y[valence >= 7] = 2
    return y  # (S, T) int in {0,1,2}


def prepare_dataset():
    """
    Prepare (X, y) for classification.

    X: temporal features flattened over subjects, trials  -> shape (N, F)
    y: 3-class labels (0,1,2) flattened over subjects, trials -> shape (N,)
    """
    temporal = load_temporal_features()  # (S, T, F)
    labels = load_deap_labels()          # (S, T, 4)

    S1, T1, F = temporal.shape
    S2, T2, _ = labels.shape
    if (S1, T1) != (S2, T2):
        raise ValueError(f"Temporal features and labels subject/trial dims mismatch: {temporal.shape} vs {labels.shape}")

    y_valence = labels_valence_3class(labels)  # (S, T)

    X = temporal.reshape(-1, F)
    y = y_valence.reshape(-1)
    return X.astype(np.float32), y.astype(np.int32)


# --------------------------------------------------
# MODEL (Module6.jpg + AlgoModule6)
# --------------------------------------------------
def build_module6_classifier(input_dim, num_classes=3, hidden_units1=128, hidden_units2=64, dropout_rate=0.4):
    """
    Module 6 classifier:
      Temporal Feature Vector
        -> FEATURE NORMALIZATION (implicit via standardization / optimizer)
        -> DENSE LAYER-1 + RELU + DROPOUT-1
        -> DENSE LAYER-2 + RELU + DROPOUT-2
        -> SOFTMAX LAYER (C=3 emotion classes)
    """
    model = Sequential(name="module6_emotion_classifier")
    model.add(Dense(hidden_units1, activation="relu", input_shape=(input_dim,), name="dense_layer_1"))
    model.add(Dropout(dropout_rate, name="dropout_1"))
    model.add(Dense(hidden_units2, activation="relu", name="dense_layer_2"))
    model.add(Dropout(dropout_rate, name="dropout_2"))
    model.add(Dense(num_classes, activation="softmax", name="softmax_output"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------
# TRAIN / EVAL + SAVING
# --------------------------------------------------
def train_module6(
    X,
    y,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_classes = int(np.max(y)) + 1
    y_cat = to_categorical(y, num_classes=num_classes)

    # Stratified split so train/val have representative class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=validation_split, stratify=y, random_state=42
    )
    y_train_labels = np.argmax(y_train, axis=1)

    # Standardize features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Balanced class weights: penalize errors on minority classes more
    classes = np.unique(y_train_labels)
    weights = compute_class_weight(
        "balanced", classes=classes, y=y_train_labels
    )
    class_weight = dict(zip(classes, weights))

    model = build_module6_classifier(input_dim=X.shape[1], num_classes=num_classes)
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        verbose=1,
    )

    # Save model weights
    model_path = os.path.join(OUTPUT_DIR, "module6_classifier_weights.h5")
    model.save(model_path)

    # Save simple summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 6: EMOTION CLASSIFICATION - SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Num samples: {X.shape[0]}\n")
        f.write(f"Feature dim: {X.shape[1]}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Final train acc: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final val   acc: {history.history['val_accuracy'][-1]:.4f}\n")

    return model, history, model_path, summary_path, scaler


def save_module6_visualizations(history, X, y, model, output_dir=OUTPUT_DIR):
    """
    Create teacher-friendly plots:
      1) Training vs validation accuracy
      2) Training vs validation loss
      3) Class distribution bar chart
      4) Confusion matrix heatmap (on a small eval split)
    """
    from sklearn.metrics import confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    # 1) Accuracy curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train acc")
    plt.plot(history.history["val_accuracy"], label="Val acc")
    plt.title("Module 6: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p1 = os.path.join(output_dir, "accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Module 6: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p2 = os.path.join(output_dir, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Class distribution
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, tick_label=[f"class {int(c)}" for c in unique], color="steelblue")
    plt.title("Module 6: Class Distribution (Valence 3-class)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.25)
    p3 = os.path.join(output_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Confusion matrix (use a held-out subset)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(300, len(X)), replace=False)
    X_eval = X[idx]
    y_true = y[idx]
    y_pred_probs = model.predict(X_eval, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=unique)
    cm = cm.astype(np.float32)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Module 6: Confusion Matrix (Normalized)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(unique, [str(int(c)) for c in unique])
    plt.yticks(unique, [str(int(c)) for c in unique])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    p4 = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()

    return [p1, p2, p3, p4]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 6: EMOTION CLASSIFICATION (DNN + SOFTMAX)")
    print("=" * 60)

    print("Preparing dataset (flatten (subjects, trials, 128) -> (N, 128) and valence 3-class labels)...")
    X, y = prepare_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {sorted(np.unique(y).tolist())}")

    print("\nTraining classifier (Dense1 + ReLU + Dropout1 -> Dense2 + ReLU + Dropout2 -> Softmax)...")
    model, history, model_path, summary_path, scaler = train_module6(
        X, y, batch_size=32, epochs=50, validation_split=0.2
    )

    print("\nSaved Module 6 model and summary:")
    print(f"  - {model_path}")
    print(f"  - {summary_path}")

    print("\nSaving Module 6 visualizations...")
    X_scaled = scaler.transform(X)
    viz_paths = save_module6_visualizations(history, X_scaled, y, model, output_dir=OUTPUT_DIR)
    for p in viz_paths:
        print(f"  - {p}")

    print("\nModule 6 completed.")

