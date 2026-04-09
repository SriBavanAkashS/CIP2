# Why Are the Signals Different? - Explanation

## Plot 1: Butterworth Overlay (`04_butterworth_overlay.png`)

### What You See:
- **Blue line (Raw EEG)**: Original, unfiltered EEG signal
- **Red line (Filtered EEG)**: Signal after Butterworth bandpass filter (0.5-30 Hz)

### Why They're Different:

#### 1. **Frequency-Based Filtering**
The Butterworth bandpass filter **removes frequency components outside 0.5-30 Hz**:

- **Removed Low Frequencies (< 0.5 Hz)**:
  - Baseline drift
  - Slow DC shifts
  - Very slow wave activity
  
- **Removed High Frequencies (> 30 Hz)**:
  - High-frequency noise
  - Electrical interference (50/60 Hz line noise)
  - Some muscle artifacts (EMG)

#### 2. **Visual Differences You Observe**:

**Raw EEG (Blue)**:
- Larger amplitude swings (e.g., -20 µV to +10 µV)
- Sharp, deep negative spikes (around samples 500, 1000, 1250)
- More "jagged" appearance with high-frequency noise
- Baseline may drift up/down

**Filtered EEG (Red)**:
- **Reduced extreme amplitudes** (stays within ~-10 to +10 µV range)
- **Smoother waveform** - sharp spikes are attenuated
- **Less jagged** - high-frequency noise removed
- **Stable baseline** - low-frequency drift removed

#### 3. **What This Means**:
The filter **preserves brain activity** in the 0.5-30 Hz range (alpha, beta, theta, delta bands) while **removing unwanted noise** outside this range.

---

## Plot 2: ICA Artifact Removal (`06_artifact_removal.png`)

### What You See:
- **Blue line (Before ICA)**: EEG signal after filtering but before ICA
- **Orange line (After ICA)**: Same signal after ICA artifact removal

### Why They're Different:

#### 1. **Artifact Removal**
ICA (Independent Component Analysis) **identifies and removes artifacts** that are mixed into the EEG signal:

- **Eye blinks** (EOG artifacts)
- **Eye movements**
- **Muscle activity** (EMG artifacts)
- **Heart artifacts** (ECG)
- **Other non-brain signals**

#### 2. **Visual Differences You Observe**:

**Before ICA (Blue)**:
- **Sharp, high-amplitude spikes** reaching ~200 µV
- These spikes are **artifacts** (likely eye blinks)
- Baseline fluctuates around -50 to +50 µV
- Signal has sudden, large deflections

**After ICA (Orange)**:
- **Spikes significantly reduced** (peaks now ~120-180 µV instead of ~200 µV)
- **Smoother peaks** - artifacts removed
- **Cleaner signal** - more representative of actual brain activity
- Overall waveform shape preserved, but artifacts subtracted

#### 3. **What This Means**:
ICA **separates** the mixed EEG signal into independent components, identifies which components are artifacts, and **removes them** from the signal. This leaves you with a cleaner representation of brain activity.

---

## Summary: Why Both Steps Are Needed

### Processing Order:
```
Raw EEG
  ↓
1. Butterworth Filter (0.5-30 Hz)
   → Removes frequency noise
   → Result: Frequency-filtered signal
  ↓
2. ICA Artifact Removal
   → Removes artifacts (eye blinks, muscle, etc.)
   → Result: Clean, artifact-free signal
  ↓
3. Window Segmentation
4. Normalization
5. Standardization
```

### Key Differences:

| Step | What It Removes | Why Signals Look Different |
|------|----------------|---------------------------|
| **Butterworth Filter** | Frequencies < 0.5 Hz and > 30 Hz | **Smoother, less extreme amplitudes** - removes high/low frequency noise |
| **ICA** | Artifacts (eye blinks, muscle, etc.) | **Removes sharp spikes** - subtracts artifact components from signal |

### Both Are Essential:
- **Filtering** removes frequency-based noise
- **ICA** removes time-based artifacts (even if they're in the correct frequency range)

Together, they produce a **clean, artifact-free EEG signal** suitable for analysis!
