import sys
import os
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as plt

from module1_data_loading import load_deap_dataset
from module2_preprocessing import preprocess_eeg
from module3_cnn_spatial import extract_spatial_features
from module4_channel_attention import apply_channel_attention
from module5_lstm_gru_temporal import extract_temporal_features



print("Loading and preprocessing EEG data...")
eeg_data, labels, subjects = load_deap_dataset("../data/raw")
eeg_preprocessed = preprocess_eeg(eeg_data)

print("CNN spatial features...")
spatial_features = extract_spatial_features(eeg_preprocessed)

print("Channel attention...")
attended_features, _ = apply_channel_attention(spatial_features)

print("LSTM–GRU temporal features...")
temporal_features, _ = extract_temporal_features(attended_features)

# ---------------------------------------------
# VISUALIZE TEMPORAL FEATURE VECTOR
# ---------------------------------------------
plt.figure(figsize=(10, 3))
plt.plot(temporal_features[0])
plt.title("Temporal Feature Vector (LSTM–GRU Output)")
plt.xlabel("Feature Index")
plt.ylabel("Activation Value")
plt.show()

# ---------------------------------------------
# DISTRIBUTION
# ---------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(temporal_features.flatten(), bins=50)
plt.title("Distribution of Temporal Features")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.show()

print("\nMODULE 5 VISUAL VERIFICATION COMPLETED")
