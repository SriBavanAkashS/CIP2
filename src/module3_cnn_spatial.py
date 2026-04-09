import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Lambda, Dense
except Exception:
    Model = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# MATHEMATICAL FEATURE EXTRACTION
# --------------------------------------------------

def _extract_raw_features(D_prep_all, fs=128, batch_size=2048):
    """
    Core extraction logic (PSD + DE) for any array of shape (S, T, Seg, C, W).
    """
    subjects, trials, segments, channels, window = D_prep_all.shape
    X_all = D_prep_all.reshape(-1, channels, window)
    total_segments = X_all.shape[0]
    
    # EEG Bands: Theta (4-8Hz), Alpha (8-14Hz), Beta (14-30Hz), Gamma (30-45Hz)
    bands = [(4.0, 8.0), (8.0, 14.0), (14.0, 30.0), (30.0, 45.0)]
    all_features = np.zeros((total_segments, channels * len(bands)), dtype=np.float32)
    
    for i in range(0, total_segments, batch_size):
        batch_end = min(i + batch_size, total_segments)
        batch_X = X_all[i:batch_end]
        
        freqs, psd = welch(batch_X, fs=fs, nperseg=128, noverlap=64, axis=-1)
        
        batch_features = []
        for low, high in bands:
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(psd[:, :, idx_band], axis=-1)
            band_power = np.maximum(band_power, 1e-12)
            de = np.log2(band_power)
            batch_features.append(de)
            
        stacked = np.stack(batch_features, axis=-1)
        processed_batch = stacked.reshape(batch_end - i, channels * len(bands))
        all_features[i:batch_end] = processed_batch
        
    return all_features


def eeg_feature_pipeline(D_prep_trial, D_prep_baseline, fs=128, batch_size=2048):
    """
    Calculates PSD/DE and mathematically subtracts the Baseline features from the Trial features.
    """
    print("Extracting features from Baseline arrays...")
    raw_baseline_feats = _extract_raw_features(D_prep_baseline, fs, batch_size)
    print("Extracting features from Trial arrays...")
    raw_trial_feats = _extract_raw_features(D_prep_trial, fs, batch_size)
    
    # Reshape back to Subject/Trial structure to align the subtractions
    S, T, Seg_B, C, W = D_prep_baseline.shape
    baseline_4d = raw_baseline_feats.reshape(S, T, Seg_B, -1)
    
    S, T, Seg_T, C, W = D_prep_trial.shape
    trial_4d = raw_trial_feats.reshape(S, T, Seg_T, -1)
    
    # Average the baseline's segments to create a single neutral fingerprint per person/video
    mean_baseline_fingerprint = np.mean(baseline_4d, axis=2, keepdims=True)  # (S, T, 1, 160)
    
    # Mathematical Baseline Subtraction (broadcasts automatically to Seg_T)
    pure_emotional_features = trial_4d - mean_baseline_fingerprint
    
    # Per-Subject Z-Score Normalization
    # Each subject's features are normalized independently using their OWN mean/std.
    # This removes subject-specific amplitude scales and forces the model to learn
    # relative emotional patterns that generalize across unseen subjects.
    num_features = pure_emotional_features.shape[-1]
    for s in range(S):
        subj_data = pure_emotional_features[s]  # (T, Seg_T, Features)
        subj_flat = subj_data.reshape(-1, num_features)
        mean_s = np.mean(subj_flat, axis=0, keepdims=True)
        std_s = np.std(subj_flat, axis=0, keepdims=True) + 1e-8
        subj_flat = (subj_flat - mean_s) / std_s
        pure_emotional_features[s] = subj_flat.reshape(T, Seg_T, num_features)
    
    pure_features_flat = pure_emotional_features.reshape(-1, num_features)
    mean_feat = None
    std_feat = None
    print(f"Applied Per-Subject Z-Score Normalization across {S} subjects.")
            
    print(f"Mathematical Spatial Features Output Shape: {pure_features_flat.shape}")
    
    output_path = os.path.join(OUTPUT_DIR, "spatial_features.npy")
    np.save(output_path, pure_features_flat)
    print(f"Saved pure spatial features to: {output_path}")
    
    return pure_features_flat, mean_feat, std_feat

# --------------------------------------------------
# COMPATIBILITY LAYER FOR END-TO-END KERAS PIPELINE
# --------------------------------------------------
def build_spatial_cnn_encoder_keras(channels=32, window_size=256, output_features=128, name="module3_spatial_cnn"):
    """
    Module 3: CNN feature extractor implemented purely in Keras.
    Input shape: (channels, window_size, 1)
    within the deeply-nested Keras end-to-end function in Module 6. 
    """
    if Model is None:
        raise ImportError("Keras is required for End-to-End mode.")
        
    inp = Input(shape=(channels, window_size, 1), name="segment_input")
    from tensorflow.keras.layers import Flatten, Conv2D, BatchNormalization
    
    if window_size == 4:
        # DE Topology: 32 Channels x 4 Frequency Bands
        # Treating DE features as a dense spatial image restores the network's mathematical
        # capacity to learn cross-frequency emotional correlations!
        # Injecting L2 Regularization penalty weights to combat 100% Train-Set overfitting.
        from tensorflow.keras.regularizers import l2
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(output_features, activation='relu')(x)
    else:
        # Raw Amplitude Time Series bypass
        x = Flatten()(inp)
        x = Dense(output_features, activation='relu')(x)
        
    return Model(inp, x, name=name)


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
def visualize_psd_features(sample_segment, extracted_features, fs=128, channel_idx=0):
    """
    Visualize the Welch PSD on a single channel and the resulting feature distribution.
    """
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: PSD plot for Channel 0
    plt.subplot(1, 2, 1)
    freqs, psd = welch(sample_segment[channel_idx], fs=fs, nperseg=128, noverlap=64)
    plt.plot(freqs, psd, color='blue', linewidth=2)
    
    # Highlight bands
    styles = [
        ('Theta (4-8Hz)', 4.0, 8.0, 'blue'),
        ('Alpha (8-14Hz)', 8.0, 14.0, 'green'),
        ('Beta (14-30Hz)', 14.0, 30.0, 'purple'),
        ('Gamma (30-45Hz)', 30.0, 45.0, 'red')
    ]
    for label, low, high, color in styles:
        mask = (freqs >= low) & (freqs <= high)
        plt.fill_between(freqs[mask], 0, psd[mask], color=color, alpha=0.3, label=label)
        
    plt.title(f"Welch PSD & Frequency Bands (Channel {channel_idx})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(0, 50)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Feature Distribution
    plt.subplot(1, 2, 2)
    plt.hist(extracted_features[0], bins=30, color='teal', alpha=0.7)
    plt.title("Standardized DE Features Distribution (for 1 segment)")
    plt.xlabel("Z-Scored Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "psd_feature_extraction.png"), dpi=300)
    plt.close()


def load_module2_output(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 2 output not found at: {path}")
    data = np.load(path)
    print(f"Loaded Module 2 data shape: {data.shape}")
    return data


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODULE 3: MATHEMATICAL SPATIAL FEATURE EXTRACTION (PSD + DE)")
    print("="*60)

    trial_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_baseline_features.npy")
    
    D_prep_trial = load_module2_output(trial_path)
    D_prep_baseline = load_module2_output(baseline_path)

    subjects, trials, segments, channels, window = D_prep_trial.shape
    total_segments = subjects * trials * segments
    
    print(f"\nInput Data Summary:")
    print(f"  - Subjects: {subjects}")
    print(f"  - Trials: {trials}")
    print(f"  - Trial Segments per trial: {segments}")
    print(f"  - Baseline Segments per trial: {D_prep_baseline.shape[2]}")
    print(f"  - Channels: {channels}")
    print(f"  - Window Size: {window} samples")
    print(f"  - Total trial segments: {total_segments}")
    print(f"  - Expected Output Features: {channels} channels * 4 bands = {channels * 4} features")

    print(f"\n{'='*60}")
    print("Extracting Mathematical Features & Performing Baseline Subtraction...")
    print(f"{'='*60}")
    
    spatial_features, mean_feat, std_feat = eeg_feature_pipeline(D_prep_trial, D_prep_baseline)

    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}")
    
    # Save a visualization from the first segment
    sample_seg = D_prep_trial[0, 0, 0] # Subject 0, Trial 0, Segment 0 (channels x window)
    visualize_psd_features(sample_seg, spatial_features)

    print(f"\n{'='*60}")
    print("✓ MODULE 3 COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
