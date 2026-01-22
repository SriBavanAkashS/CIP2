import numpy as np
import matplotlib.pyplot as plt

from module1_data_loading import load_deap_dataset
from module2_preprocessing import preprocess_eeg
from module3_cnn_spatial import build_spatial_cnn, extract_spatial_features

# --------------------------------------------------
# STEP 0: LOAD DATA
# --------------------------------------------------
print("STEP 0: Loading EEG data...")
eeg_data, labels, subjects = load_deap_dataset("../data/raw")

print("Raw EEG shape:", eeg_data.shape)

# --------------------------------------------------
# STEP 1: PREPROCESS EEG (MODULE 2)
# --------------------------------------------------
print("\nSTEP 1: Preprocessing EEG data...")
eeg_preprocessed = preprocess_eeg(eeg_data)

print("Preprocessed EEG shape:", eeg_preprocessed.shape)

# --------------------------------------------------
# STEP 2: BUILD CNN MODEL
# --------------------------------------------------
print("\nSTEP 2: Building CNN model for spatial feature extraction")

input_shape = (eeg_preprocessed.shape[1], eeg_preprocessed.shape[2])
cnn_model = build_spatial_cnn(input_shape)

print("\nCNN MODEL SUMMARY:")
cnn_model.summary()

# --------------------------------------------------
# STEP 3: EXTRACT SPATIAL FEATURES
# --------------------------------------------------
print("\nSTEP 3: Extracting spatial features using CNN")
spatial_features = extract_spatial_features(eeg_preprocessed)

print("Spatial features shape:", spatial_features.shape)

# --------------------------------------------------
# STEP 4: INPUT vs OUTPUT SHAPE CONFIRMATION
# --------------------------------------------------
print("\nSTEP 4: Shape comparison")
print("One EEG segment shape:", eeg_preprocessed[0].shape)
print("Corresponding feature vector shape:", spatial_features[0].shape)

# --------------------------------------------------
# STEP 5: VISUALIZE CNN FEATURE VECTOR
# --------------------------------------------------
print("\nSTEP 5: Visualizing one CNN spatial feature vector")

plt.figure(figsize=(10, 3))
plt.plot(spatial_features[0])
plt.title("CNN Spatial Feature Vector (One EEG Segment)")
plt.xlabel("Feature Index")
plt.ylabel("Activation Value")
plt.show()

# --------------------------------------------------
# STEP 6: FEATURE DISTRIBUTION VISUALIZATION
# --------------------------------------------------
print("\nSTEP 6: Visualizing distribution of CNN features")

plt.figure(figsize=(6, 4))
plt.hist(spatial_features.flatten(), bins=50)
plt.title("Distribution of CNN Spatial Features")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------------
# STEP 7: FINAL CONFIRMATION
# --------------------------------------------------
print("\nSTEP 7: Module 3 verification completed successfully")
print("CNN-based spatial feature extraction is confirmed")
