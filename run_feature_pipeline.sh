#!/bin/bash
echo "Activating Environment..."
source ubuntu_env/bin/activate

echo "Configuring CUDA library paths..."
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
BASE_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:$BASE_PATH/cublas/lib:$BASE_PATH/cufft/lib:$BASE_PATH/curand/lib:$BASE_PATH/cusolver/lib:$BASE_PATH/cusparse/lib:$BASE_PATH/nccl/lib:$BASE_PATH/cuda_nvrtc/lib:$BASE_PATH/cuda_runtime/lib


echo "============================================================"
echo "RUNNING MODULE 3: Spatial CNN Encoder"
echo "============================================================"
python3 -m src.module3_cnn_spatial

echo "============================================================"
echo "RUNNING MODULE 4: Channel Attention"
echo "============================================================"
python3 -m src.module4_channel_attention

echo "============================================================"
echo "RUNNING MODULE 5: Temporal LSTM/GRU"
echo "============================================================"
python3 -m src.module5_lstm_gru_temporal

echo "============================================================"
echo "PIPELINE COMPLETE - Ready for Module 6 (Feature Mode)!"
echo "============================================================"
