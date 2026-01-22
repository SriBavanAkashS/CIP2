# Project Outputs Directory

This directory contains all outputs generated from each module of the EEG Emotion Recognition project. These outputs are saved for presentation to faculty members.

## Directory Structure

```
outputs/
├── module1_data_loading/          # Module 1: Data Loading outputs
├── module2_preprocessing/         # Module 2: Preprocessing outputs
└── module3_cnn_spatial/           # Module 3: CNN Spatial Feature Extraction outputs
```

## Module 1: Data Loading

**Location:** `module1_data_loading/`

**Outputs:**
- `eeg_data.npy` - Loaded EEG data array
- `eeg_labels.npy` - Emotion labels array
- `subject_ids.txt` - List of subject IDs
- `summary.json` - Summary statistics in JSON format
- `summary.txt` - Human-readable summary report

**What to Present:**
- Number of subjects loaded
- Data shapes and dimensions
- Basic statistics of loaded data

## Module 2: Preprocessing

**Location:** `module2_preprocessing/`

**Outputs:**
- `preprocessed_eeg.npy` - Preprocessed EEG data
- `raw_eeg_sample.npy` - Sample of raw EEG for comparison
- `before_after_comparison.png` - Visualization comparing raw vs preprocessed signals
- `statistics_comparison.png` - Bar chart comparing statistics before/after preprocessing
- `summary.json` - Summary statistics in JSON format
- `summary.txt` - Human-readable summary report

**What to Present:**
- Preprocessing steps applied
- Before/after visualizations
- Statistics showing the effect of preprocessing

## Module 3: CNN Spatial Feature Extraction

**Location:** `module3_cnn_spatial/`

**Outputs:**
- `spatial_features.npy` - Extracted spatial features
- `spatial_cnn_model.h5` - Saved CNN model
- `model_architecture.png` - Visual representation of model architecture
- `model_summary.txt` - Detailed model summary
- `feature_distribution.png` - Histogram of feature values
- `feature_statistics.png` - Mean and std of features per sample
- `summary.json` - Summary statistics in JSON format
- `summary.txt` - Human-readable summary report

**What to Present:**
- Model architecture diagram
- Feature extraction results
- Model parameters and complexity

## Usage

Each module automatically saves its outputs when run. To generate all outputs:

1. Run each module individually:
   ```bash
   python src/module1_data_loading.py
   python src/module2_preprocessing.py
   python src/module3_cnn_spatial.py
   ```

2. All outputs will be saved in their respective directories under `outputs/`

## Presentation Tips

1. **Start with Module 1** - Show the data loading summary and statistics
2. **Module 2** - Use the before/after comparison images to show preprocessing effects
3. **Module 3** - Display the model architecture and feature visualizations

Each module's `summary.txt` file provides a comprehensive report that can be used for documentation and presentation.
