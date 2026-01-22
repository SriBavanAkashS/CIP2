import numpy as np
import matplotlib.pyplot as plt

from module1_data_loading import load_deap_dataset
from module2_preprocessing import preprocess_eeg
from module3_cnn_spatial import extract_spatial_features
from module4_channel_attention import apply_channel_attention

# --------------------------------------------------
# STEP 0: LOAD & PREPROCESS DATA
# --------------------------------------------------
print("STEP 0: Loading and preprocessing EEG data...")
eeg_data, labels, subjects = load_deap_dataset("../data/raw")
eeg_preprocessed = preprocess_eeg(eeg_data)

# --------------------------------------------------
# STEP 1: CNN SPATIAL FEATURES
# --------------------------------------------------
print("\nSTEP 1: Extracting CNN spatial features...")
spatial_features = extract_spatial_features(eeg_preprocessed)

print("Spatial feature shape:", spatial_features.shape)

# --------------------------------------------------
# STEP 2: APPLY CHANNEL ATTENTION
# --------------------------------------------------
print("\nSTEP 2: Applying Channel Attention...")
attended_features, attention_model = apply_channel_attention(spatial_features)

print("Attention output shape:", attended_features.shape)

# --------------------------------------------------
# STEP 3: VISUALIZE ATTENTION WEIGHTS
# --------------------------------------------------
print("\nSTEP 3: Visualizing attention weights...")

# Extract attention weights from model
attention_weights = attention_model.layers[-2].output
attention_weights = attention_model.predict(
    np.expand_dims(spatial_features, axis=1)
)

# Visualize one sample's attention
plt.figure(figsize=(10, 3))
plt.plot(attention_weights[0][0])
plt.title("Channel Attention Weights (One EEG Sample)")
plt.xlabel("Feature Index")
plt.ylabel("Attention Weight")
plt.show()

# --------------------------------------------------
# STEP 4: BEFORE vs AFTER COMPARISON
# --------------------------------------------------
print("\nSTEP 4: Comparing features before and after attention")

plt.figure(figsize=(10, 4))
plt.plot(spatial_features[0], label="Before Attention", alpha=0.6)
plt.plot(attended_features[0][0], label="After Attention", linewidth=2)
plt.legend()
plt.title("Effect of Channel Attention on EEG Features")
plt.xlabel("Feature Index")
plt.ylabel("Activation Value")
plt.show()

# --------------------------------------------------
# STEP 5: FEATURE DISTRIBUTION COMPARISON
# --------------------------------------------------
print("\nSTEP 5: Feature distribution comparison")

plt.figure(figsize=(6, 4))
plt.hist(spatial_features.flatten(), bins=50, alpha=0.5, label="Before Attention")
plt.hist(attended_features.flatten(), bins=50, alpha=0.5, label="After Attention")
plt.legend()
plt.title("Feature Distribution Before vs After Attention")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.show()

print("\nMODULE 4 VISUAL VERIFICATION COMPLETED SUCCESSFULLY")
