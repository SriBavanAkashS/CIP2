import os
import numpy as np
import tensorflow as tf
from src.module6_classification import prepare_end_to_end_dataset_with_subjects
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Lambda
from src.module5_lstm_gru_temporal import build_lstm_gru_model
from src.module6_classification import build_module6_classifier

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TF logs

def main():
    print("Loading data...")
    X_all, y_all, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=True)
    
    print("Loading M3, M4, M5, M6 models...")
    # Load Models
    m3_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_model.keras")
    m4_path = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attention_model.keras")
    m5_path = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_model.keras")
    m6_path = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "module6_classifier_weights.h5")

    m3 = tf.keras.models.load_model(m3_path, compile=False)
    m4 = tf.keras.models.load_model(m4_path, compile=False)
    
    # Rebuild M5 dynamically for the sliding window shape
    m5_static = tf.keras.models.load_model(m5_path, compile=False)
    m5 = build_lstm_gru_model(input_shape=(15, 128))
    for layer_name in ["lstm_layer_1", "lstm_layer_2", "gru_layer_1", "gru_layer_2", "temporal_dense_layer"]:
        m5.get_layer(layer_name).set_weights(m5_static.get_layer(layer_name).get_weights())

    # Load Module 6
    m6 = build_module6_classifier(input_dim=128)
    m6.load_weights(m6_path)

    # Load Scaler
    scaler_path = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "scaler_params.npz")
    s = np.load(scaler_path)
    s_mean, s_std = s["mean"], s["scale"]
    s_std = np.where(s_std == 0, 1.0, s_std) # safe divide

    print(f"\nScaler Mean (first 5): {s_mean[:5]}")
    print(f"Scaler Std (first 5): {s_std[:5]}\n")

    # Choose subject 11
    subj_target = 11
    trials = np.unique(trial_idx[subj_idx == subj_target])

    correct = 0
    total = 0

    for trial in trials:
        subj_mask = subj_idx == subj_target
        trial_mask = subj_mask & (trial_idx == trial)
        chosen = np.where(trial_mask)[0]
        
        if len(chosen) == 0:
            continue
            
        chosen_idx = chosen[0]
        seq = X_all[chosen_idx:chosen_idx+1]
        y_true = y_all[chosen_idx]
        
        # Forward pass: Bypass M3 and go straight to M4 (since DE is already spatially extracted)
        x_m4_input = tf.reshape(seq, (-1, 1, 128))
        x_m4, _ = m4(x_m4_input)
        x_m4_seq = tf.reshape(x_m4, (1, Seg, 128)) # (1, 59, 128)
        
        # Sliding Window Extraction
        window_size = 15
        step_size = 2
        
        windows = []
        for start in range(0, Seg - window_size + 1, step_size):
            windows.append(x_m4_seq[:, start:start+window_size, :])
        
        # Batch the windows together
        x_windows = tf.concat(windows, axis=0) # (NumWindows, 15, 128)
        
        temporal_features = m5(x_windows).numpy() # (NumWindows, 128)
        
        # Manual Scale per window
        scaled_features = (temporal_features - s_mean) / s_std
        
        # Module 6 Predict all windows in batch
        pred_probs_all = m6.predict(scaled_features, verbose=0) # (NumWindows, 3)
        
        # Aggregate trial prediction by averaging probabilities across all fragments
        pred_probs = np.mean(pred_probs_all, axis=0) # (3,)
        pred_class = np.argmax(pred_probs)
        
        if pred_class == y_true:
            correct += 1
        total += 1
        
    print(f"\nSubject {subj_target} Output: {correct}/{total} correct ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    main()
