import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Multiply,
        GlobalAveragePooling1D,
        Reshape,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for Module 4. "
        "Install it (e.g., `pip install tensorflow`) and retry."
    ) from e


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module4_attention")

# --------------------------------------------------
# CHANNEL ATTENTION BLOCK (Module 4 Diagram)
# --------------------------------------------------
def channel_attention_block(feature_maps, reduction_ratio=8, name_prefix="ca"):
    """
    Channel Attention Mechanism (Squeeze-and-Excitation style)

    Implements the diagram:
      Global Average Pooling -> Dense-A + ReLU -> Dense-B + Sigmoid -> Feature Rescaling

    Notes for your pipeline:
    - If you pass CNN spatial feature vectors (N, 128), we reshape to (N, 1, 128).
      With a single step (1), GlobalAveragePooling1D becomes an identity "squeeze".
      The attention still works: it learns per-feature gates in [0,1] and rescales them.

    Args:
        feature_maps: Keras tensor of shape (batch, steps, feature_dim)
                      In this project, typically (batch, 1, 128).
        reduction_ratio: Reduction ratio r for the bottleneck (Dense-A units = feature_dim // r).
        name_prefix: Prefix for layer names.

    Returns:
        attended_features: shape (batch, steps, feature_dim)
        attention_weights: shape (batch, feature_dim)
    """

    # Squeeze: Global Average Pooling
    squeeze = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(feature_maps)  # (batch, feature_dim)

    # Excitation: Fully connected layers
    feature_dim = int(squeeze.shape[-1])
    hidden_units = max(1, feature_dim // int(reduction_ratio))

    dense_a = Dense(
        units=hidden_units,
        activation="relu",
        name=f"{name_prefix}_dense_a",
    )(squeeze)

    # Dense-B with sigmoid gives attention weights in [0, 1]
    attention_weights = Dense(
        units=feature_dim,
        activation="sigmoid",
        name=f"{name_prefix}_dense_b_sigmoid",
    )(dense_a)  # (batch, feature_dim)

    # Reshape for channel-wise multiplication
    weights_reshaped = Reshape((1, feature_dim), name=f"{name_prefix}_reshape")(attention_weights)

    # Scale feature maps
    attended_features = Multiply(name=f"{name_prefix}_feature_rescaling")([feature_maps, weights_reshaped])

    return attended_features, attention_weights


# --------------------------------------------------
# APPLY CHANNEL ATTENTION
# --------------------------------------------------
def apply_channel_attention(spatial_features, reduction_ratio=8, return_weights=True):
    """
    Apply channel attention to CNN spatial features

    Inputs:
        spatial_features:
            - (N, F) feature vectors (e.g., output of Module 3, F=128), OR
            - (N, 1, F) already in "steps=1" format.
        reduction_ratio: bottleneck reduction ratio used in Dense-A.
        return_weights: if True, also return attention weights.

    Returns:
        attended_features: (N, 1, F)
        attention_weights: (N, F)  [only if return_weights=True]
        attention_model: Keras model (so you can integrate/inspect if needed)
    """

    spatial_features = np.asarray(spatial_features)

    if spatial_features.ndim == 2:
        # (N, F) -> (N, 1, F)
        spatial_features_in = np.expand_dims(spatial_features, axis=1)
    elif spatial_features.ndim == 3:
        # Expect (N, 1, F)
        spatial_features_in = spatial_features
    else:
        raise ValueError(
            f"Expected spatial_features with shape (N, F) or (N, 1, F). Got {spatial_features.shape}"
        )

    if spatial_features_in.shape[1] != 1:
        raise ValueError(
            f"Module 4 expects steps=1 in this project (shape (N, 1, F)). Got {spatial_features_in.shape}"
        )

    inputs = Input(shape=spatial_features_in.shape[1:], name="spatial_features_input")
    attended, weights = channel_attention_block(inputs, reduction_ratio=reduction_ratio, name_prefix="m4")

    if return_weights:
        attention_model = Model(inputs=inputs, outputs=[attended, weights], name="module4_channel_attention")
        attended_features, attention_weights = attention_model.predict(spatial_features_in, verbose=0)
        return attended_features, attention_weights, attention_model
    else:
        attention_model = Model(inputs=inputs, outputs=attended, name="module4_channel_attention")
        attended_features = attention_model.predict(spatial_features_in, verbose=0)
        return attended_features, attention_model


# --------------------------------------------------
# I/O HELPERS
# --------------------------------------------------
def load_module3_spatial_features(
    path=None,
):
    """Load Module 3 output feature vectors (N, 128)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 3 spatial features not found at: {path}")
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected (N, F) array from Module 3, got shape {data.shape}")
    return data


def save_module4_outputs(
    attended_features,
    attention_weights,
    output_dir=OUTPUT_DIR,
):
    """Save Module 4 outputs for downstream modules / reporting."""
    os.makedirs(output_dir, exist_ok=True)

    attended_path = os.path.join(output_dir, "attended_features.npy")
    weights_path = os.path.join(output_dir, "attention_weights.npy")
    summary_path = os.path.join(output_dir, "summary.txt")

    np.save(attended_path, attended_features)
    np.save(weights_path, attention_weights)

    # Quick stats (useful for debugging/reporting)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attended_features_shape": list(np.asarray(attended_features).shape),
        "attention_weights_shape": list(np.asarray(attention_weights).shape),
        "attention_weights_min": float(np.min(attention_weights)),
        "attention_weights_max": float(np.max(attention_weights)),
        "attention_weights_mean": float(np.mean(attention_weights)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 4: CHANNEL ATTENTION - SUMMARY\n")
        f.write("=" * 48 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return attended_path, weights_path, summary_path


def save_module4_visualizations(
    spatial_features,
    attended_features,
    attention_weights,
    output_dir=OUTPUT_DIR,
    sample_idx=0,
    heatmap_samples=200,
):
    """
    Save presentation-friendly visualizations for Module 4.

    Creates:
      1) before_vs_after_sample.png
      2) attention_weights_sample.png
      3) attention_weights_mean_profile.png
      4) attention_weights_heatmap.png
      5) feature_distribution_before_after.png
    """
    os.makedirs(output_dir, exist_ok=True)

    spatial_features = np.asarray(spatial_features)  # (N, F)
    attended_features = np.asarray(attended_features)  # (N, 1, F)
    attention_weights = np.asarray(attention_weights)  # (N, F)

    # Guard shapes
    if spatial_features.ndim != 2:
        raise ValueError(f"Expected spatial_features (N, F). Got {spatial_features.shape}")
    if attended_features.ndim != 3 or attended_features.shape[1] != 1:
        raise ValueError(f"Expected attended_features (N, 1, F). Got {attended_features.shape}")
    if attention_weights.ndim != 2:
        raise ValueError(f"Expected attention_weights (N, F). Got {attention_weights.shape}")

    N, F = spatial_features.shape
    sample_idx = int(np.clip(sample_idx, 0, N - 1))

    before = spatial_features[sample_idx]
    after = attended_features[sample_idx, 0]
    w = attention_weights[sample_idx]

    # 1) Before vs After (single sample)
    plt.figure(figsize=(12, 4))
    plt.plot(before, label="Before Attention (Module 3 features)", alpha=0.65, linewidth=1.2)
    plt.plot(after, label="After Attention (Enhanced features)", alpha=0.9, linewidth=2.0)
    plt.title("Module 4: Effect of Channel Attention (One Sample)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p1 = os.path.join(output_dir, "before_vs_after_sample.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Attention weights for a single sample
    plt.figure(figsize=(12, 3.5))
    plt.plot(w, color="purple", linewidth=1.8)
    plt.title("Module 4: Attention Weights (Sigmoid Output) — One Sample")
    plt.xlabel("Feature Index")
    plt.ylabel("Attention Weight (0–1)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    p2 = os.path.join(output_dir, "attention_weights_sample.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Mean attention profile across dataset
    mean_w = np.mean(attention_weights, axis=0)
    std_w = np.std(attention_weights, axis=0)
    plt.figure(figsize=(12, 4))
    plt.plot(mean_w, label="Mean attention", color="teal", linewidth=2.0)
    plt.fill_between(np.arange(F), mean_w - std_w, mean_w + std_w, color="teal", alpha=0.2, label="±1 std")
    plt.title("Module 4: Average Attention Profile Across All Samples")
    plt.xlabel("Feature Index")
    plt.ylabel("Attention Weight (0–1)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.legend()
    p3 = os.path.join(output_dir, "attention_weights_mean_profile.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Heatmap of attention weights (subset for readability)
    k = int(min(max(10, heatmap_samples), N))
    heat = attention_weights[:k]
    plt.figure(figsize=(12, 6))
    plt.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(label="Attention Weight (0–1)")
    plt.title(f"Module 4: Attention Weights Heatmap (First {k} Samples × {F} Features)")
    plt.xlabel("Feature Index")
    plt.ylabel("Sample Index")
    p4 = os.path.join(output_dir, "attention_weights_heatmap.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()

    # 5) Feature distribution before vs after (subset to keep fast)
    # Use at most 200k values for histogram for speed.
    flat_before = spatial_features.ravel()
    flat_after = attended_features[:, 0, :].ravel()
    max_vals = 200_000
    if flat_before.size > max_vals:
        idx = np.random.default_rng(0).choice(flat_before.size, size=max_vals, replace=False)
        flat_before = flat_before[idx]
    if flat_after.size > max_vals:
        idx = np.random.default_rng(1).choice(flat_after.size, size=max_vals, replace=False)
        flat_after = flat_after[idx]

    plt.figure(figsize=(8, 4.5))
    plt.hist(flat_before, bins=60, alpha=0.55, label="Before (Module 3)", color="steelblue", density=True)
    plt.hist(flat_after, bins=60, alpha=0.55, label="After (Module 4)", color="orange", density=True)
    plt.title("Module 4: Feature Value Distribution (Before vs After)")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.2)
    plt.legend()
    p5 = os.path.join(output_dir, "feature_distribution_before_after.png")
    plt.tight_layout()
    plt.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close()

    return [p1, p2, p3, p4, p5]


# --------------------------------------------------
# EXECUTION & VERIFICATION
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 4: CHANNEL ATTENTION (SE-STYLE)")
    print("=" * 60)

    # Prefer using Module 3 saved features as input
    module3_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    print(f"Loading Module 3 features from: {module3_path}")
    spatial_features = load_module3_spatial_features(module3_path)
    print(f"Input spatial feature shape: {spatial_features.shape}")

    print("Applying channel attention (Dense-A/ReLU -> Dense-B/Sigmoid -> rescaling)...")
    attended_features, attention_weights, attention_model = apply_channel_attention(
        spatial_features, reduction_ratio=8, return_weights=True
    )

    print(f"Attended feature shape: {attended_features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(
        f"Attention weights stats: min={attention_weights.min():.4f}, "
        f"max={attention_weights.max():.4f}, mean={attention_weights.mean():.4f}"
    )

    out_attended, out_weights, out_summary = save_module4_outputs(
        attended_features, attention_weights, output_dir=OUTPUT_DIR
    )
    print("\nSaved Module 4 outputs:")
    print(f"  - {out_attended}")
    print(f"  - {out_weights}")
    print(f"  - {out_summary}")

    print("\nSaving Module 4 visualizations...")
    viz_paths = save_module4_visualizations(
        spatial_features=spatial_features,
        attended_features=attended_features,
        attention_weights=attention_weights,
        output_dir=OUTPUT_DIR,
        sample_idx=0,
        heatmap_samples=200,
    )
    for p in viz_paths:
        print(f"  - {p}")

    print("\nChannel Attention Model Summary:")
    attention_model.summary()
