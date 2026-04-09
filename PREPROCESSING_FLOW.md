# EEG Preprocessing Flow - Module 2

## Current Flow (with issue)

### In `preprocess_all_subjects()` function:
```
Raw EEG Data
    ↓
1. **Butterworth Bandpass Filter (0.5-30 Hz)** ← Applied FIRST
    ↓
2. **ICA Artifact Removal** ← Applied AFTER filtering ✓ (CORRECT)
    ↓
3. **eeg_preprocessing_pipeline()** is called:
    ├─ 3a. **Butterworth Bandpass Filter (0.5-30 Hz)** ← APPLIED AGAIN! ❌ (REDUNDANT)
    ├─ 3b. Window Segmentation
    ├─ 3c. Channel Normalization
    └─ 3d. Standardization
```

### Issue Found:
- **Bandpass filter is applied TWICE** (lines 515 and 102)
- This is redundant and may affect signal quality

### ICA Fitting Process:
In `fit_ica_for_subject()`:
- A **high-pass filter (1.0 Hz)** is applied ONLY for ICA fitting (line 469)
- This is a best practice: ICA works better on high-pass filtered data
- This filter is NOT applied to the actual data, only for ICA model training

---

## Correct Flow (should be):

```
Raw EEG Data
    ↓
1. **Butterworth Bandpass Filter (0.5-30 Hz)** ← Applied ONCE
    ↓
2. **ICA Artifact Removal** ← Applied AFTER filtering ✓
    ↓
3. **Window Segmentation**
    ↓
4. **Channel Normalization**
    ↓
5. **Standardization**
```

---

## Answer to Your Question:

**Yes, you are CORRECT!** ICA is applied **AFTER** filtering, which is the correct order. However, there's a bug where the filter is applied again inside `eeg_preprocessing_pipeline()`, making it redundant.

### Why ICA After Filtering?
1. **Filtering removes noise** in unwanted frequency bands first
2. **ICA then removes artifacts** (eye blinks, muscle activity, etc.) from the filtered signal
3. This order is standard in EEG preprocessing pipelines

### Recommended Fix:
Modify `eeg_preprocessing_pipeline()` to skip the filtering step since it's already done before ICA.
