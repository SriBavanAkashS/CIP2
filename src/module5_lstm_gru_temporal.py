import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Concatenate
)

# --------------------------------------------------
# LSTM–GRU TEMPORAL MODEL
# --------------------------------------------------
def build_lstm_gru_model(input_shape):
    """
    Hybrid LSTM–GRU model for temporal learning
    
    Input  : (time_steps, feature_dim)
    Output : Temporal feature vector
    """

    inputs = Input(shape=input_shape)

    # LSTM branch
    lstm_out = LSTM(64, return_sequences=False)(inputs)

    # GRU branch
    gru_out = GRU(64, return_sequences=False)(inputs)

    # Combine LSTM & GRU features
    combined = Concatenate()([lstm_out, gru_out])

    temporal_features = Dense(128, activation='relu')(combined)

    model = Model(inputs=inputs, outputs=temporal_features)
    return model


# --------------------------------------------------
# APPLY TEMPORAL MODEL
# --------------------------------------------------
def extract_temporal_features(attended_features):
    """
    Extract temporal EEG features
    
    Input  : Attention-weighted features (N, 1, 128)
    Output : Temporal feature vectors
    """

    # Treat "1" as time step for demo (later extendable)
    temporal_input = attended_features

    input_shape = temporal_input.shape[1:]
    model = build_lstm_gru_model(input_shape)

    temporal_features = model.predict(temporal_input, batch_size=32, verbose=0)

    return temporal_features, model


# --------------------------------------------------
# EXECUTION & VERIFICATION
# --------------------------------------------------
if __name__ == "__main__":
    from module1_data_loading import load_deap_dataset
    from module2_preprocessing import eeg_preprocessing_pipeline
    from module3_cnn_spatial import extract_spatial_features
    from module4_channel_attention import apply_channel_attention

    print("Loading EEG data...")
    eeg_data, labels, subjects = load_deap_dataset("../data/raw")

    print("Preprocessing EEG data...")
    # Use first subject's first trial for demo
    eeg_preprocessed = eeg_preprocessing_pipeline(eeg_data[0, 0])

    print("Extracting CNN spatial features...")
    spatial_features, _ = extract_spatial_features(eeg_preprocessed, return_model=True)

    print("Applying Channel Attention...")
    attended_features, _ = apply_channel_attention(spatial_features)

    print("Extracting temporal features using LSTM–GRU...")
    temporal_features, temporal_model = extract_temporal_features(attended_features)

    print("\nMODULE 5 EXECUTION CONFIRMED")
    print("Input shape :", attended_features.shape)
    print("Temporal feature shape :", temporal_features.shape)

    print("\nLSTM–GRU Model Summary:")
    temporal_model.summary()
