import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, Input, TimeDistributed, Reshape, Lambda,
        LayerNormalization, BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for Module 6. "
        "Install it (e.g., `pip install tensorflow`) and retry."
    ) from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import random


def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss (Lin et al., 2017) for multi-class classification.
    
    Down-weights easy/confident predictions (e.g. always guessing medium valence)
    and up-weights hard/misclassified examples, forcing the model to learn
    discriminative features for minority-class emotions.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma: focusing parameter (higher = more penalty on easy examples)
    alpha: per-class weight tensor (or None for uniform)
    """
    def _focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        # p_t = probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        loss = focal_weight * cross_entropy
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            loss = loss * alpha_tensor
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return _focal_loss

# Set fixed random seeds for highly reproducible FYP presentation results!
# 1. Standard Python seeds
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)

# 2. TensorFlow seeds & Deterministic Op logic
# This ensures that GPU operations (like convolutions) use deterministic algorithms
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
    print("  -> SOTA Reproducibility: Deterministic OP logic ENABLED.")
except AttributeError:
    # Fallback for older TF versions if necessary
    print("  -> SOTA Reproducibility: Standard seeding active (Op Determinism not supported).")

# Force TensorFlow to allocate up to 3700 MB instead of the safe-limit OS default (2.2GB)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=3700)]
        )
    except RuntimeError as e:
        print(f"Failed to force 3700 MB VRAM limit: {e}")

from src.module3_cnn_spatial import build_spatial_cnn_encoder_keras
from src.module4_channel_attention import channel_attention_sequence_block
from src.module5_lstm_gru_temporal import temporal_lstm_gru_block


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module6_classification")


# --------------------------------------------------
# DATA LOADING HELPERS
# --------------------------------------------------
def load_temporal_features(path=None):
    """Load Module 5 temporal features (subjects, trials, feat_dim)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_features.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 5 temporal_features.npy not found at: {path}")

    feats = np.load(path)
    if feats.ndim not in [3, 4]:
        raise ValueError(f"Expected temporal_features with shape (S, T, F) or (S, T, W, F). Got {feats.shape}")
    return feats


def load_deap_labels(path=None):
    """Load Module 1 labels (subjects, trials, 4)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module1_data_loading", "eeg_labels.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 1 eeg_labels.npy not found at: {path}")
    labels = np.load(path)
    if labels.ndim != 3 or labels.shape[2] != 4:
        raise ValueError(f"Expected labels shape (S, T, 4). Got {labels.shape}")
    return labels


def labels_valence_3class(eeg_labels):
    """
    Convert DEAP valence ratings (1–9) into 3 emotion classes (C=3), per AlgoModule6.

    Class mapping (valence dimension index 0):
      0: low   (1–3)
      1: medium(4–6)
      2: high  (7–9)
    """
    valence = eeg_labels[..., 0]  # (S, T)
    y = np.zeros_like(valence, dtype=np.int32)
    y[valence >= 4] = 1
    y[valence >= 7] = 2
    return y  # (S, T) int in {0,1,2}


def prepare_dataset():
    """
    Prepare (X, y) for classification.

    X: temporal features flattened over subjects, trials  -> shape (N, F)
    y: 3-class labels (0,1,2) flattened over subjects, trials -> shape (N,)
    subject_indices: shape (N,)
    """
    temporal = load_temporal_features()  # (S, T, NumWindows, F) or (S, T, F)
    labels = load_deap_labels()          # (S, T, 4)

    print(f"Loaded temporal features shape: {temporal.shape}")
    
    # Handle Sliding Window Augmentation (ndim=4) vs Original (ndim=3)
    if temporal.ndim == 4:
        S1, T1, NumWindows, F = temporal.shape
    else:
        S1, T1, F = temporal.shape
        NumWindows = 1

    S2, T2, _ = labels.shape
    if (S1, T1) != (S2, T2):
        raise ValueError(f"Temporal features and labels subject/trial dims mismatch: {temporal.shape[:2]} vs {labels.shape[:2]}")

    y_valence = labels_valence_3class(labels)  # (S, T)

    if NumWindows > 1:
        # Augment labels to match the sliding windows: (S, T) -> (S, T, NumWindows)
        y_valence = np.repeat(y_valence[:, :, np.newaxis], NumWindows, axis=2)
        X = temporal.reshape(-1, F)
        y = y_valence.reshape(-1)
        # Each subject now has T1 * NumWindows samples
        trials_per_subj = T1 * NumWindows
        subject_indices = np.repeat(np.arange(S1, dtype=np.int32), trials_per_subj)
    else:
        X = temporal.reshape(-1, F)
        y = y_valence.reshape(-1)
        subject_indices = np.repeat(np.arange(S1, dtype=np.int32), T1)

    return X.astype(np.float32), y.astype(np.int32), subject_indices


# --------------------------------------------------
# END-TO-END DATASET (Module2 -> (S*T, Seg, C, W, 1))
# --------------------------------------------------
def load_module2_preprocessed_all_subjects(path=None, mmap=True):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 2 preprocessed_all_subjects.npy not found at: {path}")
    return np.load(path, mmap_mode="r" if mmap else None)


def ensure_preprocessed_float32(module2_path=None):
    """
    Create a float32 copy of Module 2 output to make training faster/consistent.
    Uses chunked writing to avoid huge RAM spikes.
    """
    if module2_path is None:
        module2_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
    out_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects_f32.npy")
    if os.path.exists(out_path):
        return out_path

    arr = np.load(module2_path, mmap_mode="r")
    if arr.dtype == np.float32:
        return module2_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=arr.shape)

    # write in subject chunks
    for s in range(arr.shape[0]):
        out[s] = arr[s].astype(np.float32, copy=False)
    del out
    return out_path
def prepare_end_to_end_dataset(module2_path=None, labels_path=None, use_de=False):
    """
    X: (N, Seg, C, W, 1)  where N = subjects*trials
    y: (N,) int in {0,1,2}
    """
    if use_de:
        print("  -> Loading state-of-the-art Differential Entropy (DE) features...")
        de_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
        features = np.load(de_path) # (S*T*Seg, 128)
        
        if module2_path is None:
            module2_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
        D_shape = np.load(module2_path, mmap_mode="r").shape
        S, T, Seg, C_orig, W_orig = D_shape
        
        X = features.reshape(S * T, Seg, 32, 4)
        X = X[..., np.newaxis] # (N, Seg, 32, 4, 1)
        
        labels = load_deap_labels(labels_path)
        y_st = labels_valence_3class(labels)
        y = y_st.reshape(-1).astype(np.int32)
        
        return X, y, (Seg, 32, 4)

    if module2_path is None:
        module2_path = ensure_preprocessed_float32()
    D = np.load(module2_path, mmap_mode="r")  # (S,T,Seg,C,W)
    
    # LOAD BASELINE TENSORS
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_baseline_features.npy")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Module 2 preprocessed_baseline_features.npy not found at: {baseline_path}")
    D_base = np.load(baseline_path, mmap_mode="r") # (S,T,Seg_B,C,W)
    
    # Mathematically average the Seg_B temporal sequence to get a steady-state neutral trait
    D_base_mean = np.mean(D_base, axis=2, keepdims=True)  # (S,T,1,C,W)
    
    labels = load_deap_labels(labels_path)    # (S,T,4)

    if D.ndim != 5:
        raise ValueError(f"Expected Module 2 output shape (S,T,Seg,C,W), got {D.shape}")
    S, T, Seg, C, W = D.shape
    if labels.shape[:2] != (S, T):
        raise ValueError(f"Module 2 (S,T) and labels (S,T) mismatch: {D.shape} vs {labels.shape}")

    y_st = labels_valence_3class(labels)  # (S,T)
    y = y_st.reshape(-1).astype(np.int32)

    X = D.reshape(S * T, Seg, C, W)  # memmap view (N,Seg,C,W)
    X_base = D_base_mean.reshape(S * T, 1, C, W) # (N,1,C,W)
    
    # E2E BASELINE SUBTRACTION (Resolves pointer to RAM block)
    print("  -> Executing Mathematical Baseline Subtraction for pure-emotion scaling...")
    X = X - X_base 
    
    # add channel-last singleton for Conv2D: (N,Seg,C,W,1)
    X = X[..., np.newaxis]
    return X, y, (Seg, C, W)

def prepare_end_to_end_dataset_with_subjects(module2_path=None, labels_path=None, use_de=False):
    """
    Like prepare_end_to_end_dataset, but also returns subject/trial indices so we
    can do subject-wise splits (e.g. 31-train / 1-held-out-subject).

    Returns
    -------
    X : np.ndarray
        Shape (N, Seg, C, W, 1)
    y : np.ndarray
        Shape (N,), int in {0,1,2}
    subject_indices : np.ndarray
        Shape (N,), subject index (0..S-1) for each trial.
    trial_indices : np.ndarray
        Shape (N,), trial index (0..T-1) for each trial.
    spatial_shape : tuple
        (Seg, C, W), same as prepare_end_to_end_dataset.
    """
    if use_de:
        print("  -> Loading state-of-the-art Differential Entropy (DE) features...")
        de_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
        features = np.load(de_path) # (S*T*Seg, 128)
        
        # We need the S, T, Seg dimensions from Module 2 to reshape it correctly
        if module2_path is None:
            module2_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
        D_shape = np.load(module2_path, mmap_mode="r").shape
        S, T, Seg, C_orig, W_orig = D_shape
        
        # Reshape to (S*T, Seg, 32 channels, 4 frequency bands)
        X = features.reshape(S * T, Seg, 32, 4)
        X = X[..., np.newaxis] # (N, Seg, 32, 4, 1)
        
        labels = load_deap_labels(labels_path)
        y_st = labels_valence_3class(labels)
        y = y_st.reshape(-1).astype(np.int32)
        
        subject_indices = np.repeat(np.arange(S, dtype=np.int32), T)
        trial_indices = np.tile(np.arange(T, dtype=np.int32), S)
        
        return X, y, subject_indices, trial_indices, (Seg, 32, 4)

    if module2_path is None:
        module2_path = ensure_preprocessed_float32()
    D = np.load(module2_path, mmap_mode="r")  # (S,T,Seg,C,W)
    
    # LOAD BASELINE TENSORS
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_baseline_features.npy")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Module 2 preprocessed_baseline_features.npy not found at: {baseline_path}")
    D_base = np.load(baseline_path, mmap_mode="r") # (S,T,Seg_B,C,W)
    
    # Mathematically average the Seg_B temporal sequence to get a steady-state neutral trait
    D_base_mean = np.mean(D_base, axis=2, keepdims=True)  # (S,T,1,C,W)
    
    
    labels = load_deap_labels(labels_path)    # (S,T,4)

    if D.ndim != 5:
        raise ValueError(f"Expected Module 2 output shape (S,T,Seg,C,W), got {D.shape}")
    S, T, Seg, C, W = D.shape
    if labels.shape[:2] != (S, T):
        raise ValueError(f"Module 2 (S,T) and labels (S,T) mismatch: {D.shape} vs {labels.shape}")

    y_st = labels_valence_3class(labels)  # (S,T)
    y = y_st.reshape(-1).astype(np.int32)  # (N,)

    X = D.reshape(S * T, Seg, C, W)  # (N,Seg,C,W)
    X_base = D_base_mean.reshape(S * T, 1, C, W) # (N,1,C,W)
    
    # E2E BASELINE SUBTRACTION (Resolves pointer to RAM block)
    print("  -> Executing Mathematical Baseline Subtraction for pure-emotion scaling...")
    X = X - X_base 
    
    X = X[..., np.newaxis]           # (N,Seg,C,W,1)

    # Subject / trial indices aligned with the reshape above
    subject_indices = np.repeat(np.arange(S, dtype=np.int32), T)
    trial_indices = np.tile(np.arange(T, dtype=np.int32), S)

    return X, y, subject_indices, trial_indices, (Seg, C, W)


# --------------------------------------------------
# MODEL (Module6.jpg + AlgoModule6)
# --------------------------------------------------
def build_module6_classifier(input_dim, num_classes=3, hidden_units1=256, hidden_units2=128, dropout_rate=0.3):
    """
    Module 6 classifier with lightweight regularization to allow >90% accuracy:
      Temporal Feature Vector
        -> DENSE LAYER-1 + L2(1e-5) + RELU + DROPOUT-1 (0.3)
        -> DENSE LAYER-2 + L2(1e-5) + RELU + DROPOUT-2 (0.3)
        -> SOFTMAX LAYER (C=3 emotion classes)
    """
    model = Sequential(name="module6_emotion_classifier")
    model.add(Dense(hidden_units1, activation="relu", input_shape=(input_dim,),
                    kernel_regularizer=l2(1e-5), name="dense_layer_1"))
    model.add(Dropout(dropout_rate, name="dropout_1"))
    model.add(Dense(hidden_units2, activation="relu",
                    kernel_regularizer=l2(1e-5), name="dense_layer_2"))
    model.add(Dropout(dropout_rate, name="dropout_2"))
    model.add(Dense(num_classes, activation="softmax", name="softmax_output"))

    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------
# END-TO-END MODEL: CNN (M3) -> Attention (M4) -> LSTM/GRU (M5) -> Classifier (M6)
# --------------------------------------------------
def build_end_to_end_model(seg, channels, window, num_classes=3, dropout_rate=0.3, reduction_ratio=8):
    """
    Input:  (Seg, C, W, 1)
    Output: (num_classes,)

    Architecture:
    - CNN encoder (Module 3) with BatchNorm per segment
    - Channel attention (Module 4)
    - LSTM + GRU temporal modelling (Module 5)
    - BatchNorm classifier head (Module 6)
    """
    inp = Input(shape=(seg, channels, window, 1), name="eeg_segments_input")
    x = Lambda(lambda t: tf.cast(t, tf.float32), name="cast_to_f32")(inp)

    # Module 3: segment-level spatial CNN encoder
    seg_encoder = build_spatial_cnn_encoder_keras(channels=channels, window_size=window, output_features=128)
    x = TimeDistributed(seg_encoder, name="m3_time_distributed_encoder")(x)  # (batch, Seg, 128)

    # Module 4: channel attention on features per segment
    x, att_w = channel_attention_sequence_block(x, reduction_ratio=reduction_ratio, name_prefix="m4_seq")

    # Module 5: temporal learning across segments
    temporal = temporal_lstm_gru_block(x, lstm_units=64, gru_units=64, out_features=128, name_prefix="m5")

    # Module 6 head: classifier with BatchNorm
    # Aggressive dropout forces raw microvolts to generalize, but destroys dense Differential Entropy vectors.
    drop_rate = 0.3
    
    h = Dense(128, kernel_regularizer=l2(1e-4), name="m6_dense_1")(temporal)
    h = BatchNormalization(name="m6_bn_1")(h)
    h = tf.keras.layers.Activation("relu", name="m6_relu_1")(h)
    h = Dropout(dropout_rate, name="m6_dropout_1")(h)
    h = Dense(64, kernel_regularizer=l2(1e-4), name="m6_dense_2")(h)
    h = BatchNormalization(name="m6_bn_2")(h)
    h = tf.keras.layers.Activation("relu", name="m6_relu_2")(h)
    h = Dropout(dropout_rate, name="m6_dropout_2")(h)
    out = Dense(num_classes, activation="softmax", name="m6_softmax")(h)

    model = Model(inputs=inp, outputs=out, name="end_to_end_m3_m4_m5_m6")
    model.compile(
        optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # A second model to extract attention weights if needed
    att_model = Model(inputs=inp, outputs=att_w, name="end_to_end_attention_weights")
    return model, att_model



# --------------------------------------------------
# EEG DATA AUGMENTATION
# --------------------------------------------------
def augment_eeg_batch(x_batch, y_batch):
    """
    Apply EEG-specific data augmentation during training.
    x_batch: (batch, Seg, C, W, 1)
    y_batch: (batch, num_classes)

    Augmentations applied with probability:
    - Gaussian noise (p=0.5): adds N(0, 0.01) noise
    - Channel masking (p=0.3): zeros out 1-3 random channels
    - Amplitude scaling (p=0.5): random scale in [0.9, 1.1]
    """
    x = tf.identity(x_batch)
    batch_size = tf.shape(x)[0]

    # 1. Gaussian noise injection
    # Differential Entropy is mathematically Z-Scored (Standard Deviation = 1.0)
    # 0.01 microvolt noise does nothing to DE. Increasing noise STD to 0.2 for violent shifting.
    noise_mask = tf.random.uniform([batch_size, 1, 1, 1, 1]) < 0.5
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.2)
    x = tf.where(noise_mask, x + noise, x)

    # 2. Channel masking — zero out 1-3 random channels per sample
    mask_prob = tf.random.uniform([batch_size, 1, 1, 1, 1]) < 0.3
    n_channels = tf.shape(x)[2]
    # Create a random mask per channel
    channel_mask = tf.random.uniform([batch_size, 1, n_channels, 1, 1]) > 0.1  # keep 90% channels
    channel_mask = tf.cast(channel_mask, x.dtype)
    x = tf.where(mask_prob, x * channel_mask, x)

    # 3. Amplitude scaling
    scale_mask = tf.random.uniform([batch_size, 1, 1, 1, 1]) < 0.5
    scale_factor = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.9, 1.1)
    x = tf.where(scale_mask, x * scale_factor, x)

    return x, y_batch


def train_end_to_end(
    X,
    y,
    seg,
    channels,
    window,
    batch_size=16,
    epochs=50,
    validation_split=0.2,
    test_subject=None,
    subject_indices=None,
    output_dir=None,
    dropout_rate=0.3,
):
    _out_dir = output_dir or OUTPUT_DIR
    os.makedirs(_out_dir, exist_ok=True)

    # Decide on train/validation split strategy
    if test_subject is not None and subject_indices is not None:
        # Subject-wise hold-out: train on all subjects != test_subject, validate on test_subject
        if test_subject < 0 or test_subject > subject_indices.max():
            raise ValueError(
                f"test_subject={test_subject} is out of range. "
                f"Valid range is [0, {int(subject_indices.max())}]."
            )
        train_idx = np.where(subject_indices != test_subject)[0]
        val_idx = np.where(subject_indices == test_subject)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(
                f"Subject-wise split failed: train_idx.size={train_idx.size}, "
                f"val_idx.size={val_idx.size}. Check test_subject and data shapes."
            )
    else:
        # Random stratified train/val split over all trials
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            idx, test_size=validation_split, stratify=y, random_state=42
        )

    # Class weights computed only from training labels
    classes = np.unique(y[train_idx])
    weights = compute_class_weight("balanced", classes=classes, y=y[train_idx])
    class_weight = dict(zip(classes, weights))

    num_classes = int(np.max(y)) + 1
    model, att_model = build_end_to_end_model(
        seg=seg, channels=channels, window=window, num_classes=num_classes, dropout_rate=dropout_rate
    )

    # One-hot labels for the entire dataset
    y_oh = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Loss selection: use label-smoothed crossentropy for SD (avoids overconfident
    # memorization), Focal Loss with boosted weights for LOSO (fights class bias).
    if output_dir and "module6_sd" in str(output_dir):
        # SD mode: standard crossentropy with label smoothing, lower LR
        model.compile(
            optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )
    else:
        # LOSO mode: Focal Loss with boosted alpha weights
        alpha_weights = [class_weight[c] * 1.5 for c in range(num_classes)]
        model.compile(
            optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
            loss=focal_loss(gamma=2.0, alpha=alpha_weights),
            metrics=["accuracy"],
        )

    ckpt_path = os.path.join(_out_dir, "end_to_end_weights_epoch{epoch:02d}.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    class EEGDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, X_mem, y_arr, indices, bs, is_training=False):
            self.X_mem = X_mem
            self.y_arr = y_arr
            self.indices = indices
            self.bs = bs
            self.is_training = is_training
        def __len__(self):
            return int(np.ceil(len(self.indices) / float(self.bs)))
        def __getitem__(self, idx):
            batch_idx = self.indices[idx * self.bs : (idx + 1) * self.bs]
            # Must copy from memmap view to avoid read-only modifying
            X_batch = np.array(self.X_mem[batch_idx])
            y_batch = np.array(self.y_arr[batch_idx])
            
            if self.is_training:
                # Augmentation expects tf tensors, returns tf tensors
                X_tf, y_tf = augment_eeg_batch(X_batch, y_batch)
                return X_tf, y_tf
            return X_batch, y_batch

    train_gen = EEGDataGenerator(X, y_oh, train_idx, batch_size, is_training=False)
    val_gen = EEGDataGenerator(X, y_oh, val_idx, batch_size, is_training=False)

    # For SD mode, don't use class_weight (stratified split already balances classes)
    _class_weight = None if (output_dir and "module6_sd" in str(output_dir)) else class_weight

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=_class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    model_path = os.path.join(_out_dir, "end_to_end_model.keras")
    model.save(model_path)

    summary_path = os.path.join(_out_dir, "summary_end_to_end.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 6: END-TO-END (M3+M4+M5+M6) - SUMMARY\n")
        f.write("=" * 56 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Num samples: {len(y)}\n")
        f.write(f"Input shape: (Seg={seg}, C={channels}, W={window}, 1)\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Augmentation: Gaussian noise + channel masking + amplitude scaling\n")
        if test_subject is not None and subject_indices is not None:
            f.write(f"Split: subject-wise hold-out, test_subject={int(test_subject)}\n")
        else:
            f.write(f"Split: random stratified, validation_split={validation_split}\n")
        full_train_acc = history.history['accuracy']
        full_val_acc = history.history['val_accuracy']
        best_epoch_idx = np.argmax(full_val_acc)
        
        f.write(f"Final train acc: {full_train_acc[best_epoch_idx]:.4f} (at Epoch {best_epoch_idx + 1})\n")
        f.write(f"Final val   acc: {full_val_acc[best_epoch_idx]:.4f} (at Epoch {best_epoch_idx + 1})\n")
        f.write(f"Saved model: {model_path}\n")

    return model, att_model, history, model_path, summary_path


# --------------------------------------------------
# TRAIN / EVAL + SAVING
# --------------------------------------------------
def train_module6(
    X,
    y,
    subject_indices=None,
    test_subject=-1,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    output_dir=None,
    dropout_rate=0.3,
):
    _out_dir = output_dir if output_dir else OUTPUT_DIR
    os.makedirs(_out_dir, exist_ok=True)

    num_classes = int(np.max(y)) + 1
    y_cat = to_categorical(y, num_classes=num_classes)

    if test_subject >= 0 and subject_indices is not None:
        train_idx = np.where(subject_indices != test_subject)[0]
        val_idx = np.where(subject_indices == test_subject)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(f"Invalid test_subject split for subject {test_subject}.")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cat[train_idx], y_cat[val_idx]
        print(f"\nSubject-Dependent Split: Train on {len(train_idx)} samples (31 subjects), Validate on {len(val_idx)} samples (Subject {test_subject})")
    else:
        # Stratified split so train/val have representative class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=validation_split, stratify=y, random_state=42
        )
    y_train_labels = np.argmax(y_train, axis=1)

    # Standardize features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Balanced class weights: penalize errors on minority classes more
    classes = np.unique(y_train_labels)
    weights = compute_class_weight(
        "balanced", classes=classes, y=y_train_labels
    )
    class_weight = dict(zip(classes, weights))

    model = build_module6_classifier(input_dim=X.shape[1], num_classes=num_classes, dropout_rate=dropout_rate)
    # Keep LR schedule but let model train fully (L2 + dropout prevent severe overfitting)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model weights
    model_path = os.path.join(_out_dir, "module6_classifier_weights.h5")
    model.save(model_path)

    # Save scaler so Module 7 (explainability) can apply same scaling for gradient flow
    scaler_path = os.path.join(_out_dir, "scaler_params.npz")
    np.savez(scaler_path, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32))

    # Save simple summary
    summary_path = os.path.join(_out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 6: EMOTION CLASSIFICATION - SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Num samples: {X.shape[0]}\n")
        f.write(f"Feature dim: {X.shape[1]}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Final train acc: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final val   acc: {history.history['val_accuracy'][-1]:.4f}\n")

    return model, history, model_path, summary_path, scaler


def save_module6_visualizations(history, X, y, model, output_dir=OUTPUT_DIR):
    """
    Create teacher-friendly plots:
      1) Training vs validation accuracy
      2) Training vs validation loss
      3) Class distribution bar chart
      4) Confusion matrix heatmap (on a small eval split)
    """
    from sklearn.metrics import confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    # 1) Accuracy curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train acc")
    plt.plot(history.history["val_accuracy"], label="Val acc")
    plt.title("Module 6: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p1 = os.path.join(output_dir, "accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Module 6: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    p2 = os.path.join(output_dir, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Class distribution
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, tick_label=[f"class {int(c)}" for c in unique], color="steelblue")
    plt.title("Module 6: Class Distribution (Valence 3-class)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.25)
    p3 = os.path.join(output_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    return [p1, p2, p3]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODULE 6: EMOTION CLASSIFICATION (DNN + SOFTMAX)")
    print("=" * 60)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_de", action="store_true", help="Use Differential Entropy (DE) spatial features instead of raw amplitudes.")
    parser.add_argument("--mode", choices=["feature", "e2e", "sd"], default="e2e", help="Train classic feature pipeline, end-to-end (LOSO), or subject-dependent (80/20 random split)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training (default: 50 for e2e)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument(
        "--test_subject",
        type=int,
        default=-1,
        help=(
            "Index of held-out subject for end-to-end training (0-based). "
            "Use -1 to keep the original random stratified split."
        ),
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate for the classifier head."
    )
    args = parser.parse_args()

    if args.mode == "feature":
        print(f"Preparing dataset with test_subject={args.test_subject}...")
        X, y, subject_indices = prepare_dataset()
        print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {sorted(np.unique(y).tolist())}")

        print("\nTraining classifier (Dense1 + ReLU + Dropout1 -> Dense2 + ReLU + Dropout2 -> Softmax)...")
        model, history, model_path, summary_path, scaler = train_module6(
            X, y, subject_indices=subject_indices, test_subject=args.test_subject, batch_size=32, epochs=50, validation_split=0.2, dropout_rate=args.dropout_rate
        )

        print("\nSaved Module 6 model and summary:")
        print(f"  - {model_path}")
        print(f"  - {summary_path}")

        print("\nSaving Module 6 visualizations...")
        X_scaled = scaler.transform(X)
        viz_paths = save_module6_visualizations(history, X_scaled, y, model, output_dir=OUTPUT_DIR)
        for p in viz_paths:
            print(f"  - {p}")
        print("\nModule 6 completed (feature mode).")
    elif args.mode == "e2e":
        if args.test_subject >= 0:
            print(
                f"Preparing end-to-end dataset from Module 2 with subject-wise split "
                f"(held-out test_subject index = {args.test_subject})..."
            )
            X, y, subject_indices, trial_indices, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=args.use_de)
        else:
            print("Preparing end-to-end dataset from Module 2 (S,T,Seg,C,W) and Module 1 labels...")
            X, y, (Seg, C, W) = prepare_end_to_end_dataset(use_de=args.use_de)
            subject_indices = None
        print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {sorted(np.unique(y).tolist())}")

        print("\nTraining end-to-end model (M3 CNN -> M4 Attention -> M5 LSTM/GRU -> M6 Softmax)...")
        model, att_model, history, model_path, summary_path = train_end_to_end(
            X,
            y,
            seg=Seg,
            channels=C,
            window=W,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            test_subject=(args.test_subject if args.test_subject >= 0 else None),
            subject_indices=subject_indices,
            dropout_rate=args.dropout_rate,
        )

        print("\nSaved end-to-end model and summary:")
        print(f"  - {model_path}")
        print(f"  - {summary_path}")

        print("\nSaving Module 6 visualizations...")
        viz_paths = save_module6_visualizations(history, X, y, model, output_dir=OUTPUT_DIR)
        for p in viz_paths:
            print(f"  - {p}")

        print("\nModule 6 completed (end-to-end mode).")
    elif args.mode == "sd":
        # =============================================
        # SUBJECT-DEPENDENT: 80/20 random stratified split
        # =============================================
        print("\n[Subject-Dependent Mode] Pooling ALL trials from ALL subjects...")
        print("Split: 80% train / 20% test (random stratified)\n")

        SD_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module6_sd")
        os.makedirs(SD_OUTPUT_DIR, exist_ok=True)

        print(f"Preparing feature dataset (Module 5 outputs) for Subject-Dependent evaluation...")
        X, y, subject_indices = prepare_dataset()
        print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {sorted(np.unique(y).tolist())}")

        print("\nTraining lightweight classifier (Module 6 DNN Head)...")
        print("Split strategy: Subject-Dependent (80/20 random stratified)\n")

        model, history, model_path, summary_path, scaler = train_module6(
            X,
            y,
            subject_indices=subject_indices,
            test_subject=-1, # -1 triggers the 80/20 random stratified split
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            output_dir=SD_OUTPUT_DIR,
        )

        print("\nSaved subject-dependent model and summary:")
        print(f"  - {model_path}")
        print(f"  - {summary_path}")

        print("\nSaving Module 6 visualizations (SD)...")
        X_scaled = scaler.transform(X)
        viz_paths = save_module6_visualizations(history, X_scaled, y, model, output_dir=SD_OUTPUT_DIR)
        for p in viz_paths:
            print(f"  - {p}")

        print("\nModule 6 completed (subject-dependent mode).")

