import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Concatenate
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for Module 5. "
        "Install it (e.g., `pip install tensorflow`) and retry."
    ) from e


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal")

# --------------------------------------------------
# LSTM–GRU TEMPORAL MODEL
# --------------------------------------------------
def build_lstm_gru_model(input_shape, lstm_units=64, gru_units=64, out_features=128):
    """
    Hybrid LSTM–GRU model for temporal learning
    
    Input  : (time_steps, feature_dim)
    Output : Temporal feature vector
    """

    inputs = Input(shape=input_shape, name="temporal_sequence_input")

    # LSTM branch (matches Module5.jpg: LSTM Layer-1 -> LSTM Layer-2)
    lstm_h1 = LSTM(lstm_units, return_sequences=True, name="lstm_layer_1")(inputs)
    lstm_out = LSTM(lstm_units, return_sequences=False, name="lstm_layer_2")(lstm_h1)

    # GRU branch (matches Module5.jpg: GRU Layer-1 -> GRU Layer-2)
    gru_h1 = GRU(gru_units, return_sequences=True, name="gru_layer_1")(inputs)
    gru_out = GRU(gru_units, return_sequences=False, name="gru_layer_2")(gru_h1)

    # Feature concatenation (diagram block)
    combined = Concatenate(name="feature_concatenation")([lstm_out, gru_out])

    # Temporal dense layer (diagram block)
    temporal_features = Dense(out_features, activation="relu", name="temporal_dense_layer")(combined)

    model = Model(inputs=inputs, outputs=temporal_features, name="module5_lstm_gru_temporal")
    return model


# --------------------------------------------------
# APPLY TEMPORAL MODEL
# --------------------------------------------------
def _infer_subject_trial_segment_counts(module2_path):
    """
    Use Module 2 saved output to infer (subjects, trials, segments).
    We don't load the whole array into RAM (np.load is lazy-ish for .npy, but keep it minimal anyway).
    """
    if not os.path.exists(module2_path):
        raise FileNotFoundError(f"Module 2 output not found at: {module2_path}")
    arr = np.load(module2_path, mmap_mode="r")
    if arr.ndim != 5:
        raise ValueError(f"Expected Module 2 output shape (S,T,Seg,C,W), got {arr.shape}")
    S, T, Seg, _, _ = arr.shape
    return int(S), int(T), int(Seg)


def load_module4_outputs(
    attended_path=None,
    weights_path=None,
):
    """Load Module 4 outputs."""
    if attended_path is None:
        attended_path = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attended_features.npy")
    if weights_path is None:
        weights_path = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attention_weights.npy")

    if not os.path.exists(attended_path):
        raise FileNotFoundError(f"Module 4 attended_features.npy not found at: {attended_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Module 4 attention_weights.npy not found at: {weights_path}")

    attended = np.load(attended_path)
    weights = np.load(weights_path)
    return attended, weights


def sequence_formation(attended_features, module2_path=None):
    """
    Sequence Formation (per Module5.jpg)

    Converts flattened attended features (N, 1, 128) into true temporal sequences:
      (subjects, trials, segments, 128)  and model input  (subjects*trials, segments, 128)
    """
    attended_features = np.asarray(attended_features)
    if attended_features.ndim != 3 or attended_features.shape[1] != 1:
        raise ValueError(f"Expected attended_features (N, 1, F). Got {attended_features.shape}")

    if module2_path is None:
        module2_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")

    S, T, Seg = _infer_subject_trial_segment_counts(module2_path)
    N_expected = S * T * Seg
    if attended_features.shape[0] != N_expected:
        raise ValueError(
            f"Module 5 cannot reshape attended features into sequences.\n"
            f"Expected N=S*T*Seg={N_expected} from Module 2, but got N={attended_features.shape[0]}.\n"
            f"Tip: regenerate Module 3+4 outputs from the same Module 2 file."
        )

    F = attended_features.shape[2]
    seq_4d = attended_features.reshape(S, T, Seg, F)  # (S, T, Seg, F)
    seq_model_in = seq_4d.reshape(S * T, Seg, F)  # (S*T, Seg, F)
    return seq_4d, seq_model_in, (S, T, Seg, F)


def extract_temporal_features(attended_features, batch_size=64, module2_path=None):
    """
    Extract temporal EEG features
    
    Input  : Attention-weighted features from Module 4 (N, 1, 128)
    Output : Temporal feature vectors per trial (S, T, 128)
    """
    seq_4d, seq_in, (S, T, Seg, F) = sequence_formation(attended_features, module2_path=module2_path)

    model = build_lstm_gru_model(input_shape=(Seg, F))
    temporal_features_flat = model.predict(seq_in, batch_size=batch_size, verbose=0)  # (S*T, 128)
    temporal_features = temporal_features_flat.reshape(S, T, -1)  # (S, T, 128)

    return temporal_features, model, seq_4d


def save_module5_outputs(temporal_features, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "temporal_features.npy")
    summary_path = os.path.join(output_dir, "summary.txt")

    np.save(out_path, temporal_features)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temporal_features_shape": list(np.asarray(temporal_features).shape),
        "min": float(np.min(temporal_features)),
        "max": float(np.max(temporal_features)),
        "mean": float(np.mean(temporal_features)),
        "std": float(np.std(temporal_features)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 5: LSTM-GRU TEMPORAL - SUMMARY\n")
        f.write("=" * 48 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return out_path, summary_path


def _pca2(X):
    """
    Tiny PCA (2D) without extra dependencies.
    Returns projected 2D points and explained variance ratios.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    var = (S**2) / (X.shape[0] - 1)
    evr = var[:2] / var.sum()
    return Z, evr


def save_module5_visualizations(seq_4d, temporal_features, output_dir=OUTPUT_DIR, subject_idx=0, trial_idx=0):
    """
    Teacher-ready visualizations:
      1) sequence_heatmap_sample.png  (segments × features)
      2) temporal_feature_vector_sample.png
      3) temporal_features_distribution.png
      4) temporal_features_pca.png
    """
    os.makedirs(output_dir, exist_ok=True)

    seq_4d = np.asarray(seq_4d)  # (S, T, Seg, F)
    temporal_features = np.asarray(temporal_features)  # (S, T, 128)

    S, T, Seg, F = seq_4d.shape
    subject_idx = int(np.clip(subject_idx, 0, S - 1))
    trial_idx = int(np.clip(trial_idx, 0, T - 1))

    seq = seq_4d[subject_idx, trial_idx]  # (Seg, F)
    tf = temporal_features[subject_idx, trial_idx]  # (128,)

    # 1) Sequence heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(seq, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Feature Value")
    plt.title(f"Module 5: Sequence Formation (Subject {subject_idx+1}, Trial {trial_idx+1})")
    plt.xlabel("Feature Index (128)")
    plt.ylabel("Time Steps (Segments)")
    p1 = os.path.join(output_dir, "sequence_heatmap_sample.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Temporal feature vector
    plt.figure(figsize=(12, 3.5))
    plt.plot(tf, linewidth=2.0)
    plt.title("Module 5: Temporal Feature Vector (LSTM–GRU Output) — One Trial")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation")
    plt.grid(True, alpha=0.25)
    p2 = os.path.join(output_dir, "temporal_feature_vector_sample.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Distribution
    flat = temporal_features.reshape(-1, temporal_features.shape[-1]).ravel()
    # sample to keep fast
    if flat.size > 200_000:
        flat = np.random.default_rng(0).choice(flat, size=200_000, replace=False)
    plt.figure(figsize=(8, 4.5))
    plt.hist(flat, bins=60, color="slateblue", alpha=0.85)
    plt.title("Module 5: Distribution of Temporal Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.2)
    p3 = os.path.join(output_dir, "temporal_features_distribution.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    # 4) PCA scatter (first N trials)
    X = temporal_features.reshape(-1, temporal_features.shape[-1])
    n = min(1000, X.shape[0])
    Z, evr = _pca2(X[:n])
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.65)
    plt.title(f"Module 5: PCA of Temporal Features (first {n} trials)\nEVR: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)
    p4 = os.path.join(output_dir, "temporal_features_pca.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()

    return [p1, p2, p3, p4]


# --------------------------------------------------
# EXECUTION & VERIFICATION
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 5: TEMPORAL FEATURE LEARNING (LSTM+GRU)")
    print("=" * 60)

    # Load Module 4 outputs
    attended_features, attention_weights = load_module4_outputs()
    print(f"Loaded Module 4 attended features: {attended_features.shape}")

    # Extract temporal features per trial using true segment sequences
    print("Forming sequences and extracting temporal features (stacked LSTM + stacked GRU)...")
    temporal_features, temporal_model, seq_4d = extract_temporal_features(
        attended_features, batch_size=64
    )
    print(f"Temporal features shape (S, T, 128): {temporal_features.shape}")

    out_path, out_summary = save_module5_outputs(temporal_features, output_dir=OUTPUT_DIR)
    print("\nSaved Module 5 outputs:")
    print(f"  - {out_path}")
    print(f"  - {out_summary}")

    print("\nSaving Module 5 visualizations...")
    viz_paths = save_module5_visualizations(
        seq_4d=seq_4d,
        temporal_features=temporal_features,
        output_dir=OUTPUT_DIR,
        subject_idx=0,
        trial_idx=0,
    )
    for p in viz_paths:
        print(f"  - {p}")

    print("\nLSTM–GRU Model Summary:")
    temporal_model.summary()
