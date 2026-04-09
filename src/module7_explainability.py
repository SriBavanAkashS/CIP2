"""
Module 7: Explainable AI using Channel Attention and Saliency Analysis

Implements Algorithm 7 and the Module 7 block diagram:
- Channel attention weight extraction (α from Module 4) → Channel importance maps
- Saliency map computation S_t = |∂y/∂x_t| → Temporal saliency maps
- Explanation visualization (channel + temporal relevance)
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import glob

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Lambda
except Exception as e:
    raise ImportError(
        "TensorFlow/Keras is required for Module 7. "
        "Install it (e.g., pip install tensorflow) and retry."
    ) from e

from src.module6_classification import (
    prepare_end_to_end_dataset,
    prepare_end_to_end_dataset_with_subjects,
    build_end_to_end_model,
    build_module6_classifier,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module7_explainability")


# --------------------------------------------------
# HELPERS: Load independent modules and build Explainer
# --------------------------------------------------
def build_end_to_end_explainer(use_sd_model=False):
    """
    Build a model that outputs both:
      - class probabilities (softmax output)
      - the attended feature sequence after Module 4 (Seg, 128)

    This version completely omits 'end_to_end_model.keras' and instead
    mathematically stitches together the independent Module 3, 4, 5, and 6 files.
    """
    # 1. Infer dimensions
    de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    use_de = os.path.exists(de_features_path)
    _, _, (Seg, C, W) = prepare_end_to_end_dataset(use_de=use_de)

    # 2. Build the Input and Casting Layer
    inp = Input(shape=(Seg, C, W, 1), name="eeg_segments_input")
    x = Lambda(lambda t: tf.cast(t, tf.float32), name="cast_to_f32")(inp)
    
    # 3. Load feature generation layer (Either M3 or bypass and flatten if DE)
    if use_de:
        print("[Module 7] Bypassing M3 CNN because DE features are already physically extracted.")
        # Re-shape (N, Seg, 32, 4, 1) -> (N, Seg, 128)
        x = Lambda(lambda t: tf.reshape(t, (-1, Seg, 128)), name="de_bypass_reshape")(x)
    else:
        m3_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_model.keras")
        m3_model = tf.keras.models.load_model(m3_path, compile=False)
        x = tf.keras.layers.TimeDistributed(m3_model, name="m3_time_distributed")(x)
        
    # 4. Load Module 4 (Channel Attention)
    m4_path = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attention_model.keras")
    m4_model = tf.keras.models.load_model(m4_path, compile=False)
    
    # "Manual TimeDistributed" M4 processing
    x_merged = Lambda(lambda t: tf.reshape(t, (-1, 1, 128)), name="m4_merge_time_batch")(x)
    m4_outs = m4_model(x_merged)
    attended_merged, weights_merged = m4_outs[0], m4_outs[1]
    
    # Split back to sequence: (None, 59, 128)
    att_seq = Lambda(lambda t: tf.reshape(t, (-1, Seg, 128)), name="m4_split_time_batch")(attended_merged)
    att_w = Lambda(lambda t: tf.reshape(t, (-1, Seg, 128)), name="m4_weights_split")(weights_merged)

    # 5. Load Module 5 (Temporal LSTM-GRU) and apply Sliding Windows!
    m5_path = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_model.keras")
    m5_src = tf.keras.models.load_model(m5_path, compile=False)
    
    window_size = 15
    step_size = 2
    from src.module5_lstm_gru_temporal import build_lstm_gru_model
    m5_model = build_lstm_gru_model(input_shape=(window_size, 128))
    for layer_name in ["lstm_layer_1", "lstm_layer_2", "gru_layer_1", "gru_layer_2", "temporal_dense_layer"]:
        m5_model.get_layer(layer_name).set_weights(m5_src.get_layer(layer_name).get_weights())
        
    def slide_windows(seq):
        windows = []
        for start in range(0, Seg - window_size + 1, step_size):
            windows.append(seq[:, start:start+window_size, :])
        stacked = tf.stack(windows, axis=1) # (Batch, NumWindows, 15, 128)
        return tf.reshape(stacked, (-1, window_size, 128)) # (Batch*NumWindows, 15, 128)

    x_windows = Lambda(slide_windows, name="m5_sliding_windows")(att_seq)
    temporal_out = m5_model(x_windows) # (Batch*NumWindows, 128)

    # 6. Load Module 6 (Classification Head)
    if use_sd_model:
        # A. Apply StandardScaler
        scaler_path = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "scaler_params.npz")
        s = np.load(scaler_path)
        s_mean = tf.constant(s["mean"], dtype=tf.float32)
        s_scale = tf.constant(s["scale"], dtype=tf.float32)
        s_scale = tf.where(s_scale == 0, tf.ones_like(s_scale), s_scale)
        scaled_features = Lambda(lambda feat: (feat - s_mean) / s_scale, name="sd_scaler_layer")(temporal_out)

        # B. Load SD Classifier weights
        sd_weights = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "module6_classifier_weights.h5")
        m6_model = build_module6_classifier(input_dim=128)
        m6_model.load_weights(sd_weights)
        preds_windows = m6_model(scaled_features)
        print("[Module 7] Hybrid Explainer built using SD Subject-Dependent Head.")
    else:
        raise NotImplementedError("LOSO E2E Mode is mathematically incompatible with offline .npy features")

    # Aggregate sliding window predictions back to Trial-Level!
    def aggregate_preds(probs):
        num_windows = (Seg - window_size) // step_size + 1
        return tf.reduce_mean(tf.reshape(probs, (-1, num_windows, 3)), axis=1)
        
    preds = Lambda(aggregate_preds, name="m6_aggregate_windows")(preds_windows)

    # 7. Final Explainer Model: Inputs -> [Predictions, AttendedSequence]
    # We return 'att_seq' to compute Saliency S_t = |∂y / ∂att_seq|
    explainer = Model(inputs=inp, outputs=[preds, att_seq], name="hybrid_explainer_engine")
    return explainer


def _infer_segment_count(module2_path=None):
    """
    Backwards-compatible helper (kept for reference).
    Not used in the new end-to-end explanation path.
    """
    if module2_path is None:
        module2_path = os.path.join(
            PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy"
        )
    if not os.path.exists(module2_path):
        raise FileNotFoundError(f"Module 2 output not found at: {module2_path}")
    arr = np.load(module2_path, mmap_mode="r")
    if arr.ndim != 5:
        raise ValueError(f"Expected Module 2 shape (S,T,Seg,C,W), got {arr.shape}")
    _, _, Seg, _, _ = arr.shape
    return int(Seg)


def load_attended_sequences(module2_path=None):
    """
    Legacy helper for the old (Module 4+5+6) pipeline.
    No longer used now that explanations are derived from the end-to-end model.
    """
    raise RuntimeError(
        "load_attended_sequences is deprecated in the end-to-end pipeline. "
        "Module 7 now reads sequences directly via prepare_end_to_end_dataset()."
    )


def load_temporal_model(seg, feat_dim=128, path=None):
    """
    Legacy helper kept for backwards compatibility; not used in the end-to-end path.
    """
    raise RuntimeError("load_temporal_model is not used in the new end-to-end pipeline.")


def load_classifier_model(input_dim=128, num_classes=3, path=None):
    """
    Legacy helper kept for backwards compatibility; not used in the end-to-end path.
    """
    raise RuntimeError("load_classifier_model is not used in the new end-to-end pipeline.")


# --------------------------------------------------
# Step 1 & 2: Channel importance map from attention weights α
# --------------------------------------------------
def extract_channel_importance_map(attention_weights):
    """
    Channel contribution analysis: aggregate attention weights α over samples.
    α = σ(W₂·ReLU(W₁·GAP(F))) from Algorithm 7 Step 1.
    Returns mean and optional std per feature (channel) dimension.
    """
    # attention_weights: (N, 128)
    mean_alpha = np.mean(attention_weights, axis=0)  # (128,)
    std_alpha = np.std(attention_weights, axis=0)
    return mean_alpha, std_alpha


# --------------------------------------------------
# Step 3 & 4: Temporal saliency S_t = |∂y/∂x_t|
# --------------------------------------------------
def compute_temporal_saliency(combined_model, sequences_batch):
    """
    Compute temporal saliency S_t = |∂y/∂x_t| and channel importance per sample.
    sequences_batch: (batch, Seg, 128)
    Returns saliency (batch, Seg), predictions (batch, 3), channel_importance (batch, 128).
    """
    seq_var = tf.Variable(sequences_batch)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(seq_var)
        pred = combined_model(seq_var)  # (batch, 3)
    grad = tape.gradient(pred, seq_var)  # (batch, Seg, 128)
    if grad is None:
        return (
            np.zeros((sequences_batch.shape[0], sequences_batch.shape[1])),
            pred.numpy(),
            np.zeros((sequences_batch.shape[0], 128), dtype=np.float32),
        )
    g = grad.numpy()
    saliency = np.abs(g).sum(axis=-1)           # (batch, Seg)
    channel_importance = np.abs(g).sum(axis=1)  # (batch, 128) — which features mattered
    return saliency, pred.numpy(), channel_importance


def compute_temporal_saliency_end_to_end(explainer_model, eeg_batch):
    """
    Compute temporal saliency and channel importance from the end-to-end model.

    eeg_batch: (batch, Seg, C, W, 1)
    explainer_model: outputs [preds (batch,3), attended_seq (batch, Seg, 128)]
    """
    eeg_tf = tf.convert_to_tensor(eeg_batch, dtype=tf.float32)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        preds, att_seq = explainer_model(eeg_tf)
        tape.watch(att_seq)
    grad = tape.gradient(preds, att_seq)  # (batch, Seg, 128)
    if grad is None:
        saliency = np.zeros((eeg_batch.shape[0], att_seq.shape[1]), dtype=np.float32)
        chan_imp = np.zeros((eeg_batch.shape[0], att_seq.shape[2]), dtype=np.float32)
        return saliency, preds.numpy(), chan_imp
    g = grad.numpy()
    saliency = np.abs(g).sum(axis=-1)           # (batch, Seg)
    channel_importance = np.abs(g).sum(axis=1)  # (batch, 128)
    return saliency, preds.numpy(), channel_importance


def compute_electrode_saliency_batch(explainer_model, eeg_batch):
    """
    Compute electrode-level importance and temporal saliency from gradients w.r.t. raw EEG input.

    eeg_batch: (batch, Seg, C, W, 1)
    Returns: electrode_importance (batch, C), temporal_saliency (batch, Seg), preds_np
    """
    eeg_tf = tf.convert_to_tensor(eeg_batch, dtype=tf.float32)
    batch_sz = eeg_batch.shape[0]
    _, n_seg, n_electrodes, n_window, _ = eeg_batch.shape

    with tf.GradientTape() as tape:
        tape.watch(eeg_tf)
        preds, _ = explainer_model(eeg_tf)  # preds: (batch, 3)
        max_preds = tf.reduce_max(preds, axis=-1)
        loss = tf.reduce_sum(max_preds)

    preds_np = preds.numpy()
    grad_input = tape.gradient(loss, eeg_tf)
    
    if grad_input is not None:
        g = np.abs(grad_input.numpy())  # (batch, Seg, C, W, 1)
        # Per-electrode: sum |grad| over segments and time windows
        electrode_importance = g.sum(axis=(1, 3, 4))
        # Per-segment: sum |grad| over electrodes and time windows
        temporal_saliency = g.sum(axis=(2, 3, 4))
    else:
        electrode_importance = np.zeros((batch_sz, n_electrodes), dtype=np.float32)
        temporal_saliency = np.zeros((batch_sz, n_seg), dtype=np.float32)

    return electrode_importance, temporal_saliency, preds_np


def load_scaler_params(path=None):
    """Load Module 6 scaler (mean, scale) so combined model matches training pipeline."""
    if path is None:
        path = os.path.join(
            PROJECT_ROOT, "outputs", "module6_classification", "scaler_params.npz"
        )
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["mean"].astype(np.float32), data["scale"].astype(np.float32)


def build_combined_model(seg, feat_dim=128, num_classes=3):
    """
    Build combined model: Input(Seg, 128) → Module 5 → [scaler] → Module 6 → (3 classes).
    Scaler is applied so classifier sees same input distribution as in training; otherwise
    gradients w.r.t. input are zero (saturated activations).
    """
    """
    Legacy helper for the old (Module 5 + Module 6) path.
    End-to-end explanations now use build_end_to_end_explainer() instead.
    """
    raise RuntimeError("build_combined_model is not used in the new end-to-end pipeline.")


# --------------------------------------------------
# Single-sample inference + explanation (new input → output + reason)
# --------------------------------------------------
CLASS_NAMES = {0: "low valence", 1: "medium valence", 2: "high valence"}


def predict_and_explain(sequence, combined_model=None, n_top_channels=10, n_top_time_steps=5):
    """
    For a single new input sequence: run the model and return prediction plus
    Module 7 explanation (which channels/features activated, which time steps drove it).
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, ...]  # (1, Seg, 128)
    if sequence.ndim != 3 or sequence.shape[0] != 1:
        raise ValueError(f"Expected sequence (Seg, 128) or (1, Seg, 128), got {sequence.shape}")
    seg, feat_dim = sequence.shape[1], sequence.shape[2]

    if combined_model is None:
        combined_model = build_combined_model(seg, feat_dim)

    seq_var = tf.Variable(sequence)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(seq_var)
        pred = combined_model(seq_var)  # (1, 3)
    pred_np = pred.numpy()[0]
    predicted_class = int(np.argmax(pred_np))
    class_name = CLASS_NAMES.get(predicted_class, f"class {predicted_class}")
    # Gradient of winning-class score (stronger than full softmax; do second pass)
    with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(seq_var)
        pred2 = combined_model(seq_var)
        target = pred2[0, predicted_class]
    grad = tape2.gradient(target, seq_var)

    if grad is None:
        channel_importance = np.zeros(128, dtype=np.float32)
        temporal_saliency = np.zeros(seg, dtype=np.float32)
    else:
        grad_np = grad.numpy()[0]  # (Seg, 128)
        channel_importance = np.abs(grad_np).sum(axis=0)  # (128,)
        temporal_saliency = np.abs(grad_np).sum(axis=1)   # (Seg,)

    explanation_text = _format_single_sample_explanation(
        predicted_class=predicted_class,
        class_probs=pred_np,
        class_name=class_name,
        electrode_importance=channel_importance,
        temporal_saliency=temporal_saliency,
        n_top_channels=n_top_channels,
        n_top_time_steps=n_top_time_steps,
    )

    return {
        "predicted_class": predicted_class,
        "class_probs": pred_np,
        "class_name": class_name,
        "channel_importance": channel_importance,
        "temporal_saliency": temporal_saliency,
        "explanation_text": explanation_text,
    }


def predict_and_explain_end_to_end(
    eeg_sequence,
    explainer_model=None,
    n_top_channels=10,
    n_top_time_steps=5,
    use_sd_model=False,
):
    """
    Single-sample explanation using the end-to-end model.

    Computes gradients w.r.t. the raw EEG input to report which EEG electrodes
    (physical channels) and which time segments drove the prediction.

    eeg_sequence: (Seg, C, W, 1) or (1, Seg, C, W, 1)
    """
    x = np.asarray(eeg_sequence, dtype=np.float32)
    if x.ndim == 4:
        x = x[np.newaxis, ...]
    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError(f"Expected (Seg, C, W, 1) or (1, Seg, C, W, 1), got {x.shape}")

    if explainer_model is None:
        explainer_model = build_end_to_end_explainer(use_sd_model=use_sd_model)

    # Cast input to float32 explicitly before gradient tape
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Process in a controlled scope to save memory
    @tf.function
    def get_gradients(input_tensor):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            preds, _ = explainer_model(input_tensor)
            target_index = tf.argmax(preds[0])
            target_score = preds[0, target_index]
        return tape.gradient(target_score, input_tensor), preds

    grad_input, preds = get_gradients(x_tf)
    pred_np = preds[0].numpy()
    predicted_class = int(np.argmax(pred_np))
    class_name = CLASS_NAMES.get(predicted_class, f"class {predicted_class}")

    _, n_seg, n_electrodes, n_window, _ = x.shape
    electrode_importance = np.zeros(n_electrodes, dtype=np.float32)
    temporal_saliency = np.zeros(n_seg, dtype=np.float32)

    if grad_input is not None:
        g = np.abs(grad_input.numpy()[0])  # (Seg, C, W, 1)

        # Per-segment importance (temporal saliency): sum |grad| over electrodes and time
        temporal_saliency = g.sum(axis=(1, 2, 3)).astype(np.float32)  # (Seg,)

        # Per-electrode importance: normalize EACH segment independently, then average.
        # This ensures every second of the 60s trial contributes equally,
        # eliminating LSTM recency bias from the electrode ranking.
        per_seg_electrode = g.sum(axis=(2, 3))  # (Seg, C) — importance per electrode per segment
        seg_totals = per_seg_electrode.sum(axis=1, keepdims=True)  # (Seg, 1)
        seg_totals = np.where(seg_totals == 0, 1.0, seg_totals)  # avoid division by zero
        per_seg_normalized = per_seg_electrode / seg_totals  # (Seg, C) — each row sums to 1
        electrode_importance = per_seg_normalized.mean(axis=0).astype(np.float32)  # (C,) — uniform average

    explanation_text = _format_single_sample_explanation(
        predicted_class=predicted_class,
        class_probs=pred_np,
        class_name=class_name,
        electrode_importance=electrode_importance,
        temporal_saliency=temporal_saliency,
        n_top_channels=n_top_channels,
        n_top_time_steps=n_top_time_steps,
    )

    return {
        "predicted_class": predicted_class,
        "class_probs": pred_np,
        "class_name": class_name,
        "electrode_importance": electrode_importance,
        "temporal_saliency": temporal_saliency,
        "explanation_text": explanation_text,
    }


def _format_single_sample_explanation(
    predicted_class,
    class_probs,
    class_name,
    electrode_importance,
    temporal_saliency,
    n_top_channels=10,
    n_top_time_steps=5,
):
    lines = [
        "PREDICTION",
        "----------",
        f"  Predicted class: {predicted_class} ({class_name})",
        f"  Probabilities: low={class_probs[0]:.3f}, medium={class_probs[1]:.3f}, high={class_probs[2]:.3f}",
        "",
        "REASON FOR THIS OUTPUT (Module 7 Explainable AI)",
        "-----------------------------------------------",
        "Which EEG electrodes (physical channels) influenced this prediction most:",
    ]
    elec_imp = np.asarray(electrode_importance, dtype=np.float64)
    elec_max = elec_imp.max() if elec_imp.size > 0 else 0.0
    elec_norm = elec_imp / (elec_max + 1e-12)

    n_show = min(n_top_channels, len(elec_imp))
    top_elec = np.argsort(elec_imp)[::-1][:n_show]
    for i, idx in enumerate(top_elec, 1):
        lines.append(
            f"  {i}. Electrode index {int(idx)}: "
            f"importance = {elec_imp[idx]:.3e} "
            f"(normalized = {elec_norm[idx]:.3f})"
        )

    lines.extend([
        "",
        "Summary: The model predicted " + class_name + " because the EEG signals from the",
        "electrodes listed above had the strongest influence on the output.",
    ])
    return "\n".join(lines)


def run_inference_with_explanation(
    sequence,
    output_path,
    combined_model=None,
    n_top_channels=10,
    n_top_time_steps=5,
    true_class=None,
    use_sd_model=False,
):
    """
    Run prediction + Module 7 explanation for one new input and save to file.
    """
    result = predict_and_explain_end_to_end(
        sequence,
        explainer_model=combined_model,
        n_top_channels=n_top_channels,
        n_top_time_steps=n_top_time_steps,
        use_sd_model=use_sd_model,
    )
    if true_class is not None:
        tc = int(true_class)
        result["true_class"] = tc
        result["true_class_name"] = CLASS_NAMES.get(tc, f"class {tc}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("MODEL OUTPUT + EXPLANATION (Module 7)\n")
            f.write("=" * 50 + "\n\n")
            if true_class is not None:
                f.write("GROUND-TRUTH VS PREDICTED CLASS\n")
                f.write("--------------------------------\n")
                f.write(
                    f"  True class      : {tc} ({result['true_class_name']})\n"
                    f"  Predicted class : {result['predicted_class']} ({result['class_name']})\n\n"
                )
            f.write(result["explanation_text"])
            f.write("\n")

        # Save visual plots corresponding to this specific single-trial explanation
        out_dir = os.path.dirname(output_path) or "."
        save_electrode_importance_map(result["electrode_importance"], std_importance=None, output_dir=out_dir)
    return result


# --------------------------------------------------
# Step 5: Explanation visualization
# --------------------------------------------------
def save_channel_importance_map(mean_importance, std_importance=None, output_dir=OUTPUT_DIR, source="gradient"):
    """
    Visualize channel (feature) importance map.

    When source="gradient": mean_importance is mean |∂y/∂x| over time and samples —
    which features the model actually uses (only some will be high). When source="alpha":
    mean_importance is Module 4 attention α (untrained in this pipeline, so often ~0.5 for all).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(mean_importance))
    ax.bar(x, mean_importance, color="steelblue", alpha=0.85, label="Mean" if source == "gradient" else "Mean α")
    if std_importance is not None:
        ax.fill_between(
            x,
            mean_importance - std_importance,
            mean_importance + std_importance,
            alpha=0.25,
            color="steelblue",
        )
    if source == "gradient":
        ax.set_title("Module 7: Channel Importance Map (Gradient-based: which features drive the output)")
        ax.set_ylabel("Mean |∂y/∂x| (sum over time)")
    else:
        ax.set_title("Module 7: Channel Importance Map (from Attention Weights α)")
        ax.set_ylabel("Attention Weight α")
    ax.set_xlabel("Feature / Channel Index")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "channel_importance_map.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_electrode_importance_map(
    mean_importance, std_importance=None, output_dir=OUTPUT_DIR
):
    """
    Visualize EEG electrode importance (gradient-based, w.r.t. raw input).
    mean_importance: (n_electrodes,) - which physical electrodes drive the output.
    """
    os.makedirs(output_dir, exist_ok=True)
    mean_importance = np.asarray(mean_importance, dtype=np.float64)
    # Ensure visible scale: if all near zero, use uniform for display
    if mean_importance.max() <= 0:
        mean_importance = np.ones_like(mean_importance) / len(mean_importance)

    n_elec = len(mean_importance)
    x = np.arange(n_elec)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.viridis((mean_importance - mean_importance.min()) / (mean_importance.max() - mean_importance.min() + 1e-12))
    bars = ax.bar(x, mean_importance, color=colors, alpha=0.9, edgecolor="gray", linewidth=0.5)
    if std_importance is not None:
        std_importance = np.asarray(std_importance, dtype=np.float64)
        ax.errorbar(x, mean_importance, yerr=std_importance, fmt="none", color="black", capsize=2)
    ax.set_title("Module 7: EEG Electrode Importance (which electrodes drive the emotion prediction)")
    ax.set_ylabel("Mean |∂y/∂x| (gradient magnitude)")
    ax.set_xlabel("EEG Electrode Index (0–31)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(-0.5, n_elec - 0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "electrode_importance_map.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def generate_explanation_output(
    predicted_class,
    class_probs,
    class_name,
    mean_channel_importance,
    saliency_maps,
    seg,
    n_top_channels=10,
    n_top_time_steps=5,
    output_dir=OUTPUT_DIR,
    source="gradient",
):
    """
    Explanation Generator (Module 7 diagram): produce a human-readable explanation
    for Explainable AI — what the model relies on (channels + time) and why.
    source="gradient": mean_channel_importance is gradient-based (which features drive output).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize for readability in text (keep raw values for plots).
    chan_vals = np.asarray(mean_channel_importance, dtype=np.float64)
    time_vals = np.mean(np.asarray(saliency_maps, dtype=np.float64), axis=0)

    chan_norm = chan_vals / (chan_vals.max() + 1e-12)
    time_norm = time_vals / (time_vals.max() + 1e-12)

    top_chan_idx = np.argsort(chan_vals)[::-1][:n_top_channels]
    top_chan_vals = chan_norm[top_chan_idx]
    top_time_idx = np.argsort(time_vals)[::-1][:n_top_time_steps]
    top_time_vals = time_norm[top_time_idx]

    if source == "electrode":
        n_show = min(n_top_channels, len(chan_vals))
        header = (
            "--- 1. EEG ELECTRODE IMPORTANCE (gradient-based) ---",
            "",
            "Which EEG electrodes (physical channels) most influence the model's output.",
            "Values below are normalized to [0, 1] for readability.",
            "",
            f"Top {n_show} most important EEG electrode indices:",
        )
        val_fmt = "  {i}. Electrode index {idx}: normalized importance = {val:.4f}"
        top_chan_idx = top_chan_idx[:n_show]
        top_chan_vals = top_chan_vals[:n_show]
    elif source == "gradient":
        header = (
            "--- 1. CHANNEL / FEATURE IMPORTANCE (gradient-based) ---",
            "",
            "Which of the 128 feature dimensions most influence the model's output.",
            "Values below are normalized to [0, 1] for readability "
            "(1.0 = most important feature based on mean |∂y/∂x| over time and samples).",
            "",
            f"Top {n_top_channels} most important feature (channel) indices:",
        )
        val_fmt = "  {i}. Feature index {idx}: normalized importance = {val:.4f}"
    else:
        header = (
            "--- 1. CHANNEL / FEATURE IMPORTANCE (attention α) ---",
            "",
            "The model uses a channel-attention mechanism (α = σ(W₂·ReLU(W₁·GAP(F)))).",
            f"Top {n_top_channels} most important feature (channel) indices and their mean α:",
        )
        val_fmt = "  {i}. Feature index {idx}: α = {val:.4f}"

    lines = [
        "=" * 60,
        "MODULE 7: EXPLANATION OUTPUT (Explainable AI)",
        "=" * 60,
        "",
        "This document explains what the emotion classification model (CNN–LSTM)"
        " relies on when making predictions, in plain language.",
        "",
        *header,
    ]
    for i, (idx, val) in enumerate(zip(top_chan_idx, top_chan_vals), 1):
        lines.append(val_fmt.format(i=i, idx=int(idx), val=val))
    lines.extend([
        "",
        "Interpretation: The model’s decision is most influenced by these feature"
        " dimensions (from the spatial CNN + attention pipeline).",
        "",
        "--- 2. SUMMARY FOR END-USERS ---",
        "",
        "• Electrode importance map (electrode_importance_map.png): shows which EEG"
        "  electrodes most drive the model's output (gradient-based; higher bar = more influential).",
        "",
        "Together, these form the explanation: the model’s emotion prediction is"
        " driven mainly by the electrodes highlighted above.",
        "",
        "=" * 60,
    ])

    path = os.path.join(output_dir, "explanation_output.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def save_temporal_saliency_maps(
    saliency_maps, n_display=32, output_dir=OUTPUT_DIR, seg=None
):
    """
    Visualize temporal saliency S_t: heatmap (samples × time) and mean profile.
    saliency_maps: (n_samples, Seg)
    """
    os.makedirs(output_dir, exist_ok=True)
    saliency_maps = np.asarray(saliency_maps)
    n_samples, n_steps = saliency_maps.shape
    if seg is None:
        seg = n_steps

    # Heatmap: subsample for visibility; use percentile-based vmin/vmax for contrast
    n_show = min(n_display, n_samples)
    idx_arr = np.linspace(0, n_samples - 1, n_show, dtype=int)
    sub = saliency_maps[idx_arr]
    vmin = np.percentile(sub, 5)
    vmax = np.percentile(sub, 95)
    if vmax <= vmin:
        vmin, vmax = sub.min(), sub.max()
    if vmax <= vmin:
        vmin, vmax = 0, 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im = axes[0].imshow(sub, aspect="auto", cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[0].set_ylabel("Sample index")
    axes[0].set_title("Module 7: Temporal Saliency S_t (samples × time steps)")
    plt.colorbar(im, ax=axes[0], label="|∂y/∂x_t| (magnitude)")
    axes[1].plot(np.mean(saliency_maps, axis=0), color="darkred", linewidth=2)
    axes[1].set_xlabel("Time step t")
    axes[1].set_ylabel("Mean saliency")
    axes[1].set_title("Mean temporal saliency (time-influence profile)")
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_saliency_map.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def run_module7(
    max_samples_saliency=128,
    batch_size=32,
    output_dir=OUTPUT_DIR,
):
    """
    Run full Module 7 pipeline:
    1) Channel importance map from gradient w.r.t. Module 4 attended features
    2) Temporal saliency S_t from the end-to-end model with gradient computation
    3) Save visualizations and summary
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load end-to-end inputs (Module 2 output) -----
    de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    use_de = os.path.exists(de_features_path)
    X_all, _, (Seg, C, W) = prepare_end_to_end_dataset(use_de=use_de)
    n_total = X_all.shape[0]
    n_use = min(max_samples_saliency, n_total)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, size=n_use, replace=False)
    eeg_batch = X_all[idx]  # (n_use, Seg, C, W, 1)

    # ----- Explainer model: electrode-level gradients (w.r.t. raw EEG input) -----
    explainer = build_end_to_end_explainer(use_sd_model=True)
    electrode_list = []
    saliency_list = []
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        batch = eeg_batch[start:end]
        elec_imp, sal, _ = compute_electrode_saliency_batch(explainer, batch)
        electrode_list.append(elec_imp)
        saliency_list.append(sal)
    electrode_importance_all = np.concatenate(electrode_list, axis=0)  # (n_use, C)
    saliency_maps = np.concatenate(saliency_list, axis=0)
    mean_electrode_importance = np.mean(electrode_importance_all, axis=0)  # (C,)
    std_electrode_importance = np.std(electrode_importance_all, axis=0)

    # ----- Electrode importance map (which EEG electrodes drive the output) -----
    path_chan = save_electrode_importance_map(
        mean_electrode_importance, std_electrode_importance, output_dir
    )

    path_sal = save_temporal_saliency_maps(
        saliency_maps, n_display=min(32, n_use), output_dir=output_dir, seg=Seg
    )

    # ----- Explanation Generator: electrode-level -----
    path_explanation = generate_explanation_output(
        None, None, None,
        mean_electrode_importance, saliency_maps, Seg,
        n_top_channels=10, n_top_time_steps=5, output_dir=output_dir, source="electrode"
    )

    # ----- Summary -----
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 7: EXPLAINABLE AI - SUMMARY\n")
        f.write("=" * 48 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Electrode importance: gradient w.r.t. raw EEG (mean |∂y/∂x| over {n_use} trials)\n")
        f.write(f"Temporal saliency: S_t = |∂y/∂x_t| on {n_use} trials (Seg={Seg})\n")
        f.write(f"Explanation output (XAI): {path_explanation}\n")
        f.write(f"Visualizations: {path_chan}, {path_sal}\n")

    return {
        "electrode_importance_path": path_chan,
        "temporal_saliency_path": path_sal,
        "explanation_output_path": path_explanation,
        "summary_path": summary_path,
        "mean_electrode_importance": mean_electrode_importance,
        "saliency_maps": saliency_maps,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Module 7: Explainable AI (single-trial or batch mode)."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single-trial prediction + explanation.",
    )
    parser.add_argument(
        "--sd",
        action="store_true",
        help="Use the hybrid Subject-Dependent (SD) Explainer model for explanation.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=-1,
        help=(
            "Subject index (0-based) for single-trial mode. "
            "If provided, the trial will be drawn from this subject."
        ),
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=-1,
        help=(
            "Trial index (0-based) within the chosen subject for single-trial mode. "
            "If -1, a random trial for that subject is selected."
        ),
    )
    args = parser.parse_args()

    run_single_mode = args.single or args.subject >= 0 or args.trial >= 0

    if run_single_mode:
        print("\n" + "=" * 60)
        print("MODULE 7: Single-sample inference + explanation")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Auto-detect DE mode
        de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
        _use_de = os.path.exists(de_features_path)

        if args.subject >= 0:
            # Use subject-aware dataset helper so we can target a specific subject/trial.
            X_all, y_all, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=_use_de)

            if args.subject < 0 or args.subject > int(subj_idx.max()):
                raise ValueError(
                    f"subject index {args.subject} is out of range. "
                    f"Valid range is [0, {int(subj_idx.max())}]."
                )

            # All indices for this subject
            subj_mask = subj_idx == args.subject
            subj_indices = np.where(subj_mask)[0]
            if subj_indices.size == 0:
                raise ValueError(f"No trials found for subject {args.subject}.")

            if args.trial >= 0:
                # Specific trial for this subject
                trial_mask = subj_mask & (trial_idx == args.trial)
                chosen = np.where(trial_mask)[0]
                if chosen.size == 0:
                    raise ValueError(
                        f"No trial found for subject {args.subject} with trial index {args.trial}."
                    )
                chosen_idx = int(chosen[0])
                trial_str = str(args.trial)
            else:
                # Random trial for this subject
                rng = np.random.default_rng(42)
                chosen_idx = int(rng.choice(subj_indices))
                trial_str = f"rand{int(trial_idx[chosen_idx])}"

            one_sequence = X_all[chosen_idx]  # (Seg, C, W, 1)
            true_class = y_all[chosen_idx]
            base_name = f"subject{int(args.subject)}_trial{trial_str}"
        else:
            # Fallback: random single trial from all subjects (original --single behavior),
            # now also with ground-truth label.
            X_all, y_all, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=_use_de)
            rng = np.random.default_rng(42)
            idx = rng.integers(0, X_all.shape[0])
            one_sequence = X_all[idx]  # (Seg, C, W, 1)
            true_class = y_all[idx]
            base_name = "single_sample_random"

        out_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "prediction_and_explanation.txt")

        result = run_inference_with_explanation(
            one_sequence,
            output_path=out_path,
            combined_model=None,
            n_top_channels=10,
            n_top_time_steps=5,
            true_class=true_class,
            use_sd_model=args.sd,
        )
        print(f"\nPredicted class: {result['predicted_class']} ({result['class_name']})")
        print(f"Explanation saved to: {out_path}")
        print("\n--- Explanation (reason for this output) ---")
        print(result["explanation_text"])
        print("\nDone.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("MODULE 7: EXPLAINABLE AI (Channel Attention + Saliency)")
    print("=" * 60)

    print("Running batch explanation over a subset of trials...")
    result = run_module7(max_samples_saliency=128, batch_size=32)

    print("\nSaved Module 7 outputs:")
    print(f"  - {result['electrode_importance_path']}")
    print(f"  - {result['temporal_saliency_path']}")
    print(f"  - {result['explanation_output_path']}  (Explainable AI text explanation)")
    print(f"  - {result['summary_path']}")
    print(
        "\nFor single new input -> output + explanation, run either:\n"
        "  python -m src.module7_explainability --single\n"
        "  python -m src.module7_explainability --single --subject 31 --trial 5"
    )
    print("Module 7 completed.")
