import numpy as np
import matplotlib.pyplot as plt

from module1_data_loading import load_deap_dataset
from module2_preprocessing import (
    butter_bandpass_filter,
    segment_eeg,
    normalize_and_standardize
)

# --------------------------------------------------
# STEP 0: LOAD RAW EEG DATA
# --------------------------------------------------
print("STEP 0: Loading raw EEG data...")
eeg_data, labels, subjects = load_deap_dataset("../data/raw")

print("Raw EEG data shape:", eeg_data.shape)
print("Number of subjects:", len(subjects))

# Select one example for visualization
raw_signal = eeg_data[0, 0, 0]  # subject 1, trial 1, channel 1


# --------------------------------------------------
# STEP 1: RAW EEG VISUALIZATION
# --------------------------------------------------
print("\nSTEP 1: Visualizing raw EEG signal")

plt.figure(figsize=(10, 3))
plt.plot(raw_signal[:1000])
plt.title("Raw EEG Signal (Before Preprocessing)")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")
plt.show()


# --------------------------------------------------
# STEP 2: BUTTERWORTH BANDPASS FILTERING
# --------------------------------------------------
print("\nSTEP 2: Applying Butterworth bandpass filter (0.5–30 Hz)")

filtered_signal = butter_bandpass_filter(raw_signal)

print("Raw signal mean:", np.mean(raw_signal))
print("Filtered signal mean:", np.mean(filtered_signal))

plt.figure(figsize=(10, 3))
plt.plot(filtered_signal[:1000])
plt.title("EEG Signal After Bandpass Filtering")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")
plt.show()


# --------------------------------------------------
# STEP 3: RAW vs FILTERED COMPARISON
# --------------------------------------------------
print("\nSTEP 3: Comparing raw and filtered EEG")

plt.figure(figsize=(10, 4))
plt.plot(raw_signal[:1000], label="Raw EEG", alpha=0.6)
plt.plot(filtered_signal[:1000], label="Filtered EEG", linewidth=2)
plt.legend()
plt.title("Raw EEG vs Filtered EEG")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")
plt.show()


# --------------------------------------------------
# STEP 4: WINDOW SEGMENTATION
# --------------------------------------------------
print("\nSTEP 4: Segmenting EEG signals into windows")

# Use small subset for clarity
small_eeg = eeg_data[:1, :1]
segments = segment_eeg(small_eeg)

print("Before segmentation shape:", small_eeg.shape)
print("After segmentation shape:", segments.shape)


# --------------------------------------------------
# STEP 5: VISUALIZE ONE EEG SEGMENT
# --------------------------------------------------
print("\nSTEP 5: Visualizing one EEG segment")

plt.figure(figsize=(10, 3))
plt.plot(segments[0][0])
plt.title("One EEG Segment After Window Segmentation")
plt.xlabel("Samples per Window")
plt.ylabel("Amplitude")
plt.show()


# --------------------------------------------------
# STEP 6: NORMALIZATION & STANDARDIZATION
# --------------------------------------------------
print("\nSTEP 6: Normalizing and standardizing EEG segments")

normalized_segments = normalize_and_standardize(segments)

print("Before normalization:")
print("Mean:", np.mean(segments), "Std:", np.std(segments))

print("\nAfter normalization:")
print("Mean:", np.mean(normalized_segments), "Std:", np.std(normalized_segments))


# --------------------------------------------------
# STEP 7: FINAL CONFIRMATION
# --------------------------------------------------
print("\nSTEP 7: Preprocessing verification completed successfully")
print("Final segment shape:", normalized_segments.shape)
