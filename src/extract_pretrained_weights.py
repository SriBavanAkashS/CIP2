"""
extract_pretrained_weights.py

One-time utility: Rebuilds the End-to-End model architecture, loads the
pre-trained LOSO weights, then saves each module (3, 4, 5) as a completely
independent .keras file. This avoids any Lambda deserialization issues.

Run once:
    ./run_in_env.sh -m src.extract_pretrained_weights
"""

import os, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))


def extract_weights():
    print("=" * 60)
    print("EXTRACTING PRE-TRAINED INDEPENDENT MODULE WEIGHTS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Rebuild the full End-to-End architecture and load weights
    # ------------------------------------------------------------------
    from src.module6_classification import build_end_to_end_model

    # Infer the correct segment / channel / window sizes from the saved npy files
    import numpy as np
    m2_path = os.path.join(PROJECT_ROOT, "outputs", "module2_preprocessing", "preprocessed_all_subjects.npy")
    arr = np.load(m2_path, mmap_mode="r")  # (S, T, Seg, C, W)
    _, _, seg, channels, window = arr.shape
    del arr

    # Check if DE features were used (window_size == 4 means DE topology)
    de_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    sf = np.load(de_path, mmap_mode="r")
    use_de = sf.shape[-1] == channels * 4   # 32*4=128 DE features, True
    del sf

    print(f"  Detected: seg={seg}, channels={channels}, window={window}, DE={use_de}")

    if use_de:
        window = 4   # CNN treats 4 DE bands as a 4-column image

    e2e_model, _ = build_end_to_end_model(seg=seg, channels=channels, window=window, num_classes=3)

    # Find the latest recorded epoch checkpoint
    ckpts = sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", "module6_classification",
                                          "end_to_end_weights_epoch*.weights.h5")))
    if not ckpts:
        raise FileNotFoundError("No checkpoint .h5 files found in outputs/module6_classification/")
    best_ckpt = ckpts[-1]
    print(f"\nLoading weights from: {best_ckpt}")
    e2e_model.load_weights(best_ckpt)
    print("Weights loaded successfully!")

    # ------------------------------------------------------------------
    # Step 2: Extract Module 3 – CNN Spatial Encoder
    # ------------------------------------------------------------------
    td_layer = e2e_model.get_layer("m3_time_distributed_encoder")
    cnn_model = td_layer.layer   # the inner Keras model
    m3_out = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_model.keras")
    cnn_model.save(m3_out)
    print(f"\n✅ Module 3 (CNN) saved to: {m3_out}")

    # ------------------------------------------------------------------
    # Step 3: Extract Module 4 – Channel Attention (standalone model)
    # ------------------------------------------------------------------
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from src.module4_channel_attention import channel_attention_block

    inp_m4 = Input(shape=(1, 128), name="m4_standalone_input")
    attended, att_w = channel_attention_block(inp_m4, reduction_ratio=8, name_prefix="m4")
    m4_model = Model(inputs=inp_m4, outputs=[attended, att_w], name="module4_attention_standalone")

    # Copy dense weights from the E2E TimeDistributed dense layers
    m4_model.get_layer("m4_dense_a").set_weights(
        e2e_model.get_layer("m4_seq_dense_a_td").layer.get_weights()
    )
    m4_model.get_layer("m4_dense_b_sigmoid").set_weights(
        e2e_model.get_layer("m4_seq_dense_b_sigmoid_td").layer.get_weights()
    )

    m4_out = os.path.join(PROJECT_ROOT, "outputs", "module4_attention", "attention_model.keras")
    m4_model.save(m4_out)
    print(f"✅ Module 4 (Attention) saved to: {m4_out}")

    # ------------------------------------------------------------------
    # Step 4: Extract Module 5 – LSTM-GRU Temporal Model
    # ------------------------------------------------------------------
    from src.module5_lstm_gru_temporal import build_lstm_gru_model

    # Window size for M5 is the same as the number of segments (each window is a segment)
    m5_model = build_lstm_gru_model(input_shape=(seg, 128))

    m5_model.get_layer("lstm_layer_1").set_weights(e2e_model.get_layer("m5_lstm_1").get_weights())
    m5_model.get_layer("lstm_layer_2").set_weights(e2e_model.get_layer("m5_lstm_2").get_weights())
    m5_model.get_layer("gru_layer_1").set_weights(e2e_model.get_layer("m5_gru_1").get_weights())
    m5_model.get_layer("gru_layer_2").set_weights(e2e_model.get_layer("m5_gru_2").get_weights())
    m5_model.get_layer("temporal_dense_layer").set_weights(
        e2e_model.get_layer("m5_temporal_dense").get_weights()
    )

    m5_out = os.path.join(PROJECT_ROOT, "outputs", "module5_temporal", "temporal_model.keras")
    m5_model.save(m5_out)
    print(f"✅ Module 5 (LSTM-GRU) saved to: {m5_out}")

    print("\n" + "=" * 60)
    print("All 3 modules saved as independent .keras files!")
    print("You may now run Module 4, Module 5, and Module 6 SD in sequence.")
    print("=" * 60)


if __name__ == "__main__":
    extract_weights()
