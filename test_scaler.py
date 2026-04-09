import os
import numpy as np
import tensorflow as tf
from src.module6_classification import prepare_end_to_end_dataset_with_subjects
from src.module5_lstm_gru_temporal import build_lstm_gru_model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Lambda

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. Load Data
de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
X_all, y_all, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=True)

# Subject 8, Trial 12
subj_mask = subj_idx == 8
trial_mask = subj_mask & (trial_idx == 12)
chosen_idx = np.where(trial_mask)[0][0]
seq = X_all[chosen_idx:chosen_idx+1]
y_true = y_all[chosen_idx]

print(f"Testing Subj 8 Trial 12, True Class: {y_true}")

# 2. Extract Temporal Features (Through M3 -> M4 -> M5)
m3_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_model.keras")
m3 = load_model(m3_path, compile=False)

m4_path = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attention_model.keras")
m4 = load_model(m4_path, compile=False)

m5_path = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_model.keras")
m5_src = load_model(m5_path, compile=False)

m5 = build_lstm_gru_model(input_shape=(Seg, 128))
for layer_name in ["lstm_layer_1", "lstm_layer_2", "gru_layer_1", "gru_layer_2", "temporal_dense_layer"]:
    m5.get_layer(layer_name).set_weights(m5_src.get_layer(layer_name).get_weights())

# Forward pass manually
x = tf.keras.layers.TimeDistributed(m3)(seq)

x_merged = tf.reshape(x, (-1, 1, 128))
x_m4, _ = m4(x_merged)
x = tf.reshape(x_m4, (-1, Seg, 128))

temporal_features = m5(x) # shape (1, 128)

print(f"Temporal features shape: {temporal_features.shape}")
print(f"Temporal features sample: {temporal_features[0, :5].numpy()}")

# 3. Apply Scaler Manually vs Lambda
scaler_path = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "scaler_params.npz")
s = np.load(scaler_path)
s_mean = s["mean"]
s_scale = s["scale"]

# Manual scaling
scaled_manual = (temporal_features.numpy() - s_mean) / s_scale

# Lambda scaling
s_mean_tf = tf.constant(s_mean, dtype=tf.float32)
s_scale_tf = tf.constant(s_scale, dtype=tf.float32)
lambda_layer = Lambda(lambda feat: (feat - s_mean_tf) / s_scale_tf)
scaled_lambda = lambda_layer(temporal_features).numpy()

print(f"\nScaled Manual sample: {scaled_manual[0, :5]}")
print(f"Scaled Lambda sample: {scaled_lambda[0, :5]}")
print(f"Diff: {np.max(np.abs(scaled_manual - scaled_lambda))}")

# 4. Predict
m6_weights = os.path.join(PROJECT_ROOT, "outputs", "module6_sd", "module6_classifier_weights.h5")
from src.module6_classification import build_module6_classifier
m6 = build_module6_classifier(input_dim=128)
m6.load_weights(m6_weights)

pred_manual = m6.predict(scaled_manual, verbose=0)
pred_lambda = m6.predict(scaled_lambda, verbose=0)

print(f"\nPred manual: {pred_manual} -> Class {np.argmax(pred_manual)}")
print(f"Pred lambda: {pred_lambda} -> Class {np.argmax(pred_lambda)}")
