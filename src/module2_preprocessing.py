

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

OUTPUT_DIR = "../outputs/module2_preprocessing"

# --------------------------------------------------
# STEP 1: BUTTERWORTH BANDPASS FILTER
# --------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(eeg_signal, fs=128):
    """
    eeg_signal: ndarray (channels × time)
    """
    b, a = butter_bandpass(0.5, 30, fs)
    return filtfilt(b, a, eeg_signal, axis=-1)


# --------------------------------------------------
# STEP 2: ARTIFACT REMOVAL
# --------------------------------------------------

def remove_artifacts(eeg_signal, threshold=100):
    """
    Simple amplitude-based artifact rejection
    """
    clean_signal = np.copy(eeg_signal)
    clean_signal[np.abs(clean_signal) > threshold] = 0
    return clean_signal


# --------------------------------------------------
# STEP 3: WINDOW SEGMENTATION
# --------------------------------------------------

def segment_signal(eeg_signal, window_size=256, overlap=128):
    """
    Splits EEG into overlapping windows
    """
    segments = []
    step = window_size - overlap

    for start in range(0, eeg_signal.shape[-1] - window_size + 1, step):
        segment = eeg_signal[:, start:start + window_size]
        segments.append(segment)

    return np.array(segments)   # shape: (segments × channels × window_size)


# --------------------------------------------------
# STEP 4: CHANNEL NORMALIZATION
# --------------------------------------------------

def normalize_channels(eeg_segments):
    """
    Normalizes each channel to common amplitude scale
    """
    max_val = np.max(np.abs(eeg_segments), axis=-1, keepdims=True)
    return eeg_segments / (max_val + 1e-6)


# --------------------------------------------------
# STEP 5: STANDARDIZATION
# --------------------------------------------------

def standardize_segments(eeg_segments):
    """
    Standardize EEG signals to zero mean and unit variance
    """
    shape = eeg_segments.shape
    reshaped = eeg_segments.reshape(-1, shape[-1])

    scaler = StandardScaler()
    standardized = scaler.fit_transform(reshaped)

    return standardized.reshape(shape)


# --------------------------------------------------
# MODULE 2 PIPELINE
# --------------------------------------------------

def eeg_preprocessing_pipeline(D_eeg, fs=128):
    """
    Input : Structured EEG dataset D_eeg (channels × time)
    Output: Preprocessed EEG dataset D_prep
    """

    # Step 1: Bandpass Filtering
    filtered = bandpass_filter(D_eeg, fs)

    # Step 2: Artifact Removal
    cleaned = remove_artifacts(filtered)

    # Step 3: Window Segmentation
    segmented = segment_signal(cleaned)

    # Step 4: Channel Normalization
    normalized = normalize_channels(segmented)

    # Step 5: Standardization
    standardized = standardize_segments(normalized)

    return standardized


# --------------------------------------------------
# SAVE MODULE 2 OUTPUTS
# --------------------------------------------------

def save_module2_outputs(D_eeg, D_prep, output_dir=OUTPUT_DIR):
    """
    Save outputs from Module 2 (Preprocessing) for presentation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, "preprocessed_eeg.npy"), D_prep)
    np.save(os.path.join(output_dir, "raw_eeg_sample.npy"), D_eeg)
    
    # STEP 1: Save raw waveform (before any processing)
    print("\n  Saving visualizations...")
    if D_eeg.ndim >= 2:
        sample_channel = 0
        sample_time_points = min(2000, D_eeg.shape[-1])
        time_indices = np.arange(sample_time_points)
        
        if D_eeg.ndim == 2:
            raw_signal = D_eeg[sample_channel, :sample_time_points]
        else:
            raw_signal = D_eeg[sample_channel, :sample_time_points]
        
        # Save raw waveform visualization
        plt.figure(figsize=(14, 6))
        plt.plot(time_indices, raw_signal, 'b-', linewidth=1.5)
        plt.title("Raw EEG Signal Waveform (Before Any Processing)", fontsize=14, fontweight='bold')
        plt.xlabel("Time Samples", fontsize=12)
        plt.ylabel("Amplitude (μV)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.95, f'Mean: {np.mean(raw_signal):.4f} | Std: {np.std(raw_signal):.4f} | Range: [{np.min(raw_signal):.2f}, {np.max(raw_signal):.2f}]',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_raw_eeg_waveform.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 01_raw_eeg_waveform.png (Raw dataset waveform)")
    
    # STEP 2: Apply Butterworth filter and save filtered waveform
    print("  Applying Butterworth Bandpass Filter (0.5-30 Hz)...")
    D_filtered = bandpass_filter(D_eeg)
    np.save(os.path.join(output_dir, "butterworth_filtered_eeg.npy"), D_filtered)
    
    if D_eeg.ndim >= 2:
        if D_eeg.ndim == 2:
            filtered_signal = D_filtered[sample_channel, :sample_time_points]
        else:
            filtered_signal = D_filtered[sample_channel, :sample_time_points]
        
        # Save filtered waveform visualization
        plt.figure(figsize=(14, 6))
        plt.plot(time_indices, filtered_signal, 'r-', linewidth=1.5)
        plt.title("Filtered EEG Signal Waveform (After Butterworth Bandpass Filter 0.5-30 Hz)", fontsize=14, fontweight='bold')
        plt.xlabel("Time Samples", fontsize=12)
        plt.ylabel("Amplitude (μV)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.95, f'Mean: {np.mean(filtered_signal):.4f} | Std: {np.std(filtered_signal):.4f} | Range: [{np.min(filtered_signal):.2f}, {np.max(filtered_signal):.2f}]',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_butterworth_filtered_waveform.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 02_butterworth_filtered_waveform.png (After Butterworth filter)")
        
        # STEP 3: Side-by-side comparison
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(time_indices, raw_signal, 'b-', linewidth=1.5)
        plt.title("Raw EEG Signal\n(Before Butterworth Filter)", fontsize=13, fontweight='bold')
        plt.xlabel("Time Samples", fontsize=12)
        plt.ylabel("Amplitude (μV)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(time_indices, filtered_signal, 'r-', linewidth=1.5)
        plt.title("Filtered EEG Signal\n(After Butterworth Filter 0.5-30 Hz)", fontsize=13, fontweight='bold')
        plt.xlabel("Time Samples", fontsize=12)
        plt.ylabel("Amplitude (μV)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_butterworth_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 03_butterworth_comparison.png (Before/After comparison)")
        
        # STEP 4: Overlay comparison
        plt.figure(figsize=(14, 6))
        plt.plot(time_indices, raw_signal, 'b-', alpha=0.6, linewidth=1.5, label='Raw EEG (Before Filter)')
        plt.plot(time_indices, filtered_signal, 'r-', alpha=0.8, linewidth=1.5, label='Filtered EEG (After Filter)')
        plt.title("Butterworth Bandpass Filter (0.5-30 Hz): Overlay Comparison", fontsize=14, fontweight='bold')
        plt.xlabel("Time Samples", fontsize=12)
        plt.ylabel("Amplitude (μV)", fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_butterworth_overlay.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 04_butterworth_overlay.png (Overlay visualization)")
        
        # STEP 5: Frequency domain comparison
        from scipy.fft import fft, fftfreq
        sampling_rate = 128
        n_samples = len(raw_signal)
        freqs = fftfreq(n_samples, 1/sampling_rate)
        raw_fft = np.abs(fft(raw_signal))
        filtered_fft = np.abs(fft(filtered_signal))
        positive_freq_idx = (freqs >= 0) & (freqs <= 50)
        freqs_positive = freqs[positive_freq_idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(freqs_positive, raw_fft[positive_freq_idx], 'b-', linewidth=2, label='Raw EEG')
        axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Filter Cutoff (0.5 Hz)')
        axes[0].axvline(x=30, color='red', linestyle='--', linewidth=2, label='Filter Cutoff (30 Hz)')
        axes[0].set_title("Frequency Spectrum: Raw EEG Signal (Before Butterworth Filter)", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Frequency (Hz)", fontsize=12)
        axes[0].set_ylabel("Magnitude", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 50])
        
        axes[1].plot(freqs_positive, filtered_fft[positive_freq_idx], 'r-', linewidth=2, label='Filtered EEG')
        axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Filter Cutoff (0.5 Hz)')
        axes[1].axvline(x=30, color='red', linestyle='--', linewidth=2, label='Filter Cutoff (30 Hz)')
        axes[1].set_title("Frequency Spectrum: Filtered EEG Signal (After Butterworth Filter 0.5-30 Hz)", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Frequency (Hz)", fontsize=12)
        axes[1].set_ylabel("Magnitude", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 50])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_butterworth_frequency_domain.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 05_butterworth_frequency_domain.png (Frequency domain analysis)")
    
    # Create visualizations
    # 6. Before/After full preprocessing comparison plot
    if D_eeg.ndim >= 2:
        sample_channel = 0
        sample_time = slice(0, min(1000, D_eeg.shape[-1]))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        if D_eeg.ndim == 2:
            plt.plot(D_eeg[sample_channel, sample_time])
        else:
            plt.plot(D_eeg[sample_channel, sample_time])
        plt.title("Raw EEG Signal (Before Preprocessing)")
        plt.xlabel("Time Samples")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        if D_prep.ndim >= 2:
            # Take first segment if segmented
            if D_prep.ndim == 3:
                plt.plot(D_prep[0, sample_channel, :])
            else:
                plt.plot(D_prep[sample_channel, sample_time])
        plt.title("Preprocessed EEG Signal (After Preprocessing)")
        plt.xlabel("Time Samples")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "before_after_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Statistics comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    raw_stats = {
        'Mean': np.mean(D_eeg),
        'Std': np.std(D_eeg),
        'Min': np.min(D_eeg),
        'Max': np.max(D_eeg)
    }
    
    prep_stats = {
        'Mean': np.mean(D_prep),
        'Std': np.std(D_prep),
        'Min': np.min(D_prep),
        'Max': np.max(D_prep)
    }
    
    axes[0].bar(raw_stats.keys(), raw_stats.values(), color='red', alpha=0.6)
    axes[0].set_title("Raw EEG Statistics")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(prep_stats.keys(), prep_stats.values(), color='green', alpha=0.6)
    axes[1].set_title("Preprocessed EEG Statistics")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    summary = {
        "module": "Module 2: EEG Preprocessing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": list(D_eeg.shape),
        "output_shape": list(D_prep.shape),
        "preprocessing_steps": [
            "1. Bandpass Filtering (Butterworth 0.5–30 Hz)",
            "2. Artifact Removal",
            "3. Window Segmentation",
            "4. Channel Normalization",
            "5. Standardization"
        ],
        "raw_statistics": {
            "mean": float(np.mean(D_eeg)),
            "std": float(np.std(D_eeg)),
            "min": float(np.min(D_eeg)),
            "max": float(np.max(D_eeg)),
            "median": float(np.median(D_eeg))
        },
        "preprocessed_statistics": {
            "mean": float(np.mean(D_prep)),
            "std": float(np.std(D_prep)),
            "min": float(np.min(D_prep)),
            "max": float(np.max(D_prep)),
            "median": float(np.median(D_prep))
        }
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save human-readable summary
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODULE 2: EEG PREPROCESSING - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n\n")
        f.write("Preprocessing Steps Applied:\n")
        for step in summary['preprocessing_steps']:
            f.write(f"  {step}\n")
        f.write("\n")
        f.write(f"Input Shape: {summary['input_shape']}\n")
        f.write(f"Output Shape: {summary['output_shape']}\n\n")
        f.write("Raw EEG Statistics:\n")
        for key, value in summary['raw_statistics'].items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\nPreprocessed EEG Statistics:\n")
        for key, value in summary['preprocessed_statistics'].items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n✓ Module 2 outputs saved to: {output_dir}")
    print(f"\nAll saved files:")
    print(f"  - preprocessed_eeg.npy")
    print(f"  - raw_eeg_sample.npy")
    print(f"  - butterworth_filtered_eeg.npy")
    print(f"  - 01_raw_eeg_waveform.png (Raw dataset waveform)")
    print(f"  - 02_butterworth_filtered_waveform.png (After Butterworth filter)")
    print(f"  - 03_butterworth_comparison.png (Before/After side-by-side)")
    print(f"  - 04_butterworth_overlay.png (Overlay comparison)")
    print(f"  - 05_butterworth_frequency_domain.png (Frequency domain)")
    print(f"  - before_after_comparison.png (Full preprocessing comparison)")
    print(f"  - statistics_comparison.png")
    print(f"  - summary.json")
    print(f"  - summary.txt")


# --------------------------------------------------
# MAIN FUNCTION (TESTING & DEMO)
# --------------------------------------------------

if __name__ == "__main__":

    # Load EEG data from Module 1 outputs
    # Select one trial for demonstration (subject 0, trial 0)
    eeg_data_path = "../outputs/module1_data_loading/eeg_data.npy"
    D_eeg = np.load(eeg_data_path)[0, 0]  # Shape: (channels, time_samples)

    print("Input EEG shape:", D_eeg.shape)

    # Apply Module 2 preprocessing
    D_prep = eeg_preprocessing_pipeline(D_eeg)

    print("Preprocessed EEG shape:", D_prep.shape)
    print("Module 2 EEG Preprocessing completed successfully.")
    
    # Save outputs for presentation
    save_module2_outputs(D_eeg, D_prep)