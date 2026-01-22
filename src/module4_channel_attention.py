import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Multiply, GlobalAveragePooling1D, Reshape
)

# --------------------------------------------------
# CHANNEL ATTENTION BLOCK
# --------------------------------------------------
def channel_attention_block(feature_maps, reduction_ratio=8):
    """
    Channel Attention Mechanism (Squeeze-and-Excitation)

    Input  : Spatial feature maps from CNN
             shape (channels, features)
    Output : Attention-weighted feature maps
    """

    # Squeeze: Global Average Pooling
    squeeze = GlobalAveragePooling1D()(feature_maps)

    # Excitation: Fully connected layers
    excitation = Dense(
        units=squeeze.shape[-1] // reduction_ratio,
        activation='relu'
    )(squeeze)

    excitation = Dense(
        units=squeeze.shape[-1],
        activation='sigmoid'
    )(excitation)

    # Reshape for channel-wise multiplication
    excitation = Reshape((1, squeeze.shape[-1]))(excitation)

    # Scale feature maps
    attended_features = Multiply()([feature_maps, excitation])

    return attended_features


# --------------------------------------------------
# APPLY CHANNEL ATTENTION
# --------------------------------------------------
def apply_channel_attention(spatial_features):
    """
    Apply channel attention to CNN spatial features

    Input  : Spatial feature vectors (N, 128)
    Output : Attention-enhanced spatial features
    """

    # Reshape to (samples, channels=1, features)
    spatial_features = np.expand_dims(spatial_features, axis=1)

    inputs = Input(shape=spatial_features.shape[1:])
    outputs = channel_attention_block(inputs)

    attention_model = Model(inputs=inputs, outputs=outputs)

    attended_features = attention_model.predict(spatial_features, verbose=0)

    return attended_features, attention_model


# --------------------------------------------------
# EXECUTION & VERIFICATION
# --------------------------------------------------
if __name__ == "__main__":
    from module1_data_loading import load_deap_dataset
    from module2_preprocessing import eeg_preprocessing_pipeline
    from module3_cnn_spatial import extract_spatial_features

    print("Loading EEG data...")
    eeg_data, labels, subjects = load_deap_dataset("../data/raw")

    print("Preprocessing EEG data...")
    # Use first subject's first trial for demo
    eeg_preprocessed = eeg_preprocessing_pipeline(eeg_data[0, 0])

    print("Extracting CNN spatial features...")
    spatial_features, _ = extract_spatial_features(eeg_preprocessed, return_model=True)

    print("Applying Channel Attention...")
    attended_features, attention_model = apply_channel_attention(spatial_features)

    print("\nMODULE 4 EXECUTION CONFIRMED")
    print("Input spatial feature shape :", spatial_features.shape)
    print("Attention-weighted feature shape :", attended_features.shape)

    print("\nChannel Attention Model Summary:")
    attention_model.summary()
