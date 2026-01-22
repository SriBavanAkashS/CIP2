import os
import pickle
import numpy as np
import json
from datetime import datetime

# Path to raw EEG data
DATA_PATH = "../data/raw"
OUTPUT_DIR = "../outputs/module1_data_loading"

def save_module1_outputs(eeg_data, eeg_labels, subjects, output_dir=OUTPUT_DIR):
    """
    Save outputs from Module 1 (Data Loading) for presentation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data arrays
    np.save(os.path.join(output_dir, "eeg_data.npy"), eeg_data)
    np.save(os.path.join(output_dir, "eeg_labels.npy"), eeg_labels)
    
    # Save subject IDs
    with open(os.path.join(output_dir, "subject_ids.txt"), 'w') as f:
        for subj in subjects:
            f.write(f"{subj}\n")
    
    # Create summary report
    summary = {
        "module": "Module 1: Data Loading",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_subjects": len(subjects),
        "eeg_data_shape": list(eeg_data.shape),
        "labels_shape": list(eeg_labels.shape),
        "data_type": str(eeg_data.dtype),
        "labels_type": str(eeg_labels.dtype),
        "subjects": subjects.tolist() if isinstance(subjects, np.ndarray) else subjects,
        "statistics": {
            "eeg_min": float(np.min(eeg_data)),
            "eeg_max": float(np.max(eeg_data)),
            "eeg_mean": float(np.mean(eeg_data)),
            "eeg_std": float(np.std(eeg_data)),
            "labels_min": float(np.min(eeg_labels)),
            "labels_max": float(np.max(eeg_labels)),
            "labels_mean": float(np.mean(eeg_labels)),
        }
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save human-readable summary
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODULE 1: DATA LOADING - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n\n")
        f.write(f"Total Subjects: {summary['total_subjects']}\n")
        f.write(f"EEG Data Shape: {summary['eeg_data_shape']}\n")
        f.write(f"Labels Shape: {summary['labels_shape']}\n\n")
        f.write("Data Statistics:\n")
        f.write(f"  EEG Min: {summary['statistics']['eeg_min']:.4f}\n")
        f.write(f"  EEG Max: {summary['statistics']['eeg_max']:.4f}\n")
        f.write(f"  EEG Mean: {summary['statistics']['eeg_mean']:.4f}\n")
        f.write(f"  EEG Std: {summary['statistics']['eeg_std']:.4f}\n\n")
        f.write("Labels Statistics:\n")
        f.write(f"  Labels Min: {summary['statistics']['labels_min']:.4f}\n")
        f.write(f"  Labels Max: {summary['statistics']['labels_max']:.4f}\n")
        f.write(f"  Labels Mean: {summary['statistics']['labels_mean']:.4f}\n\n")
        f.write("Subject IDs:\n")
        for i, subj in enumerate(summary['subjects'], 1):
            f.write(f"  {i}. {subj}\n")
    
    print(f"\n✓ Module 1 outputs saved to: {output_dir}")
    print(f"  - eeg_data.npy")
    print(f"  - eeg_labels.npy")
    print(f"  - subject_ids.txt")
    print(f"  - summary.json")
    print(f"  - summary.txt")

def load_deap_dataset(data_path):
    eeg_data = []
    eeg_labels = []
    subject_ids = []

    for file in sorted(os.listdir(data_path)):
        if file.endswith(".dat"):
            file_path = os.path.join(data_path, file)

            with open(file_path, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')

            data = subject_data['data']      # EEG signals
            labels = subject_data['labels']  # Emotion labels

            eeg_data.append(data)
            eeg_labels.append(labels)
            subject_ids.append(file.replace(".dat", ""))

            print(f"Loaded {file}: EEG shape {data.shape}, Labels shape {labels.shape}")

    return np.array(eeg_data), np.array(eeg_labels), subject_ids


if __name__ == "__main__":
    eeg_data, eeg_labels, subjects = load_deap_dataset(DATA_PATH)

    print("\nSummary:")
    print("Total subjects:", len(subjects))
    print("EEG data shape:", eeg_data.shape)
    print("Labels shape:", eeg_labels.shape)
    
    # Save outputs for presentation
    save_module1_outputs(eeg_data, eeg_labels, subjects)