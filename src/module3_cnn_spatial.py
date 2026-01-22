import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Flatten, Dense
)
from tensorflow.keras.utils import plot_model
from datetime import datetime
import io
import sys

OUTPUT_DIR = "../outputs/module3_cnn_spatial"

# --------------------------------------------------
# CNN MODEL FOR SPATIAL FEATURE EXTRACTION
# --------------------------------------------------
def build_spatial_cnn(input_shape):
    """
    CNN model to learn spatial relationships
    between EEG channels.
    
    Input shape  : (channels, time_samples)
    Output       : Spatial feature vector
    """

    inputs = Input(shape=input_shape)

    # First convolution block
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    # Second convolution block
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Flatten and dense layer
    x = Flatten()(x)
    spatial_features = Dense(128, activation='relu')(x)

    model = Model(inputs=inputs, outputs=spatial_features)
    return model


# --------------------------------------------------
# APPLY CNN TO EEG DATA
# --------------------------------------------------
def extract_spatial_features(eeg_data, return_model=False):
    """
    Extract spatial EEG features using CNN.
    
    Input  : Preprocessed EEG data
             shape (samples, channels, time)
    Output : Spatial feature vectors
    """

    input_shape = (eeg_data.shape[1], eeg_data.shape[2])
    cnn_model = build_spatial_cnn(input_shape)

    spatial_features = cnn_model.predict(eeg_data, batch_size=32, verbose=0)

    if return_model:
        return spatial_features, cnn_model
    return spatial_features


# --------------------------------------------------
# SAVE MODULE 3 OUTPUTS
# --------------------------------------------------

def save_module3_outputs(eeg_data, spatial_features, cnn_model, output_dir=OUTPUT_DIR):
    """
    Save outputs from Module 3 (CNN Spatial Features) for presentation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save spatial features
    np.save(os.path.join(output_dir, "spatial_features.npy"), spatial_features)
    np.save(os.path.join(output_dir, "input_eeg_sample.npy"), eeg_data[:10] if len(eeg_data) > 10 else eeg_data)
    
    # Save model
    cnn_model.save(os.path.join(output_dir, "spatial_cnn_model.h5"))
    
    # Save model architecture plot
    try:
        plot_model(cnn_model, to_file=os.path.join(output_dir, "model_architecture.png"), 
                  show_shapes=True, show_layer_names=True, dpi=300)
    except Exception as e:
        print(f"Warning: Could not save model architecture plot: {e}")
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    cnn_model.summary()
    model_summary = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save model summary
    with open(os.path.join(output_dir, "model_summary.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODULE 3: CNN SPATIAL FEATURE EXTRACTION - MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(model_summary)
    
    # Create visualizations
    # 1. Feature distribution
    plt.figure(figsize=(10, 6))
    plt.hist(spatial_features.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Spatial Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "feature_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature statistics per sample
    feature_means = np.mean(spatial_features, axis=1)
    feature_stds = np.std(spatial_features, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(feature_means[:100] if len(feature_means) > 100 else feature_means, 'b-', alpha=0.6)
    axes[0].set_title("Mean Feature Value per Sample")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Mean Feature Value")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(feature_stds[:100] if len(feature_stds) > 100 else feature_stds, 'r-', alpha=0.6)
    axes[1].set_title("Std Feature Value per Sample")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Std Feature Value")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    summary = {
        "module": "Module 3: CNN Spatial Feature Extraction",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": list(eeg_data.shape),
        "output_shape": list(spatial_features.shape),
        "model_parameters": {
            "total_params": cnn_model.count_params(),
            "trainable_params": sum([np.prod(v.get_shape()) for v in cnn_model.trainable_variables]),
            "non_trainable_params": sum([np.prod(v.get_shape()) for v in cnn_model.non_trainable_variables])
        },
        "feature_statistics": {
            "mean": float(np.mean(spatial_features)),
            "std": float(np.std(spatial_features)),
            "min": float(np.min(spatial_features)),
            "max": float(np.max(spatial_features)),
            "median": float(np.median(spatial_features))
        }
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save human-readable summary
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODULE 3: CNN SPATIAL FEATURE EXTRACTION - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n\n")
        f.write(f"Input Shape: {summary['input_shape']}\n")
        f.write(f"Output Shape: {summary['output_shape']}\n\n")
        f.write("Model Parameters:\n")
        f.write(f"  Total Parameters: {summary['model_parameters']['total_params']:,}\n")
        f.write(f"  Trainable Parameters: {summary['model_parameters']['trainable_params']:,}\n")
        f.write(f"  Non-trainable Parameters: {summary['model_parameters']['non_trainable_params']:,}\n\n")
        f.write("Spatial Feature Statistics:\n")
        for key, value in summary['feature_statistics'].items():
            f.write(f"  {key.capitalize()}: {value:.4f}\n")
    
    print(f"\n✓ Module 3 outputs saved to: {output_dir}")
    print(f"  - spatial_features.npy")
    print(f"  - spatial_cnn_model.h5")
    print(f"  - model_architecture.png")
    print(f"  - model_summary.txt")
    print(f"  - feature_distribution.png")
    print(f"  - feature_statistics.png")
    print(f"  - summary.json")
    print(f"  - summary.txt")


# --------------------------------------------------
# EXECUTION & VERIFICATION
# --------------------------------------------------
if __name__ == "__main__":
    from module1_data_loading import load_deap_dataset
    from module2_preprocessing import eeg_preprocessing_pipeline

    print("Loading EEG data...")
    eeg_data, labels, subjects = load_deap_dataset("../data/raw")

    print("Preprocessing EEG data...")
    # Use first subject's first trial for demo
    eeg_preprocessed = eeg_preprocessing_pipeline(eeg_data[0, 0])

    print("Extracting spatial features using CNN...")
    spatial_features, cnn_model = extract_spatial_features(eeg_preprocessed, return_model=True)

    print("\nMODULE 3 EXECUTION CONFIRMED")
    print("Input EEG shape :", eeg_preprocessed.shape)
    print("Output feature shape :", spatial_features.shape)
    
    # Save outputs for presentation
    save_module3_outputs(eeg_preprocessed, spatial_features, cnn_model)