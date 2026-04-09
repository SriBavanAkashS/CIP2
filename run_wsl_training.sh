#!/bin/bash
echo "Activating Python virtual environment..."
source ubuntu_env/bin/activate

echo "Configuring CUDA library paths..."
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
BASE_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:$BASE_PATH/cublas/lib:$BASE_PATH/cufft/lib:$BASE_PATH/curand/lib:$BASE_PATH/cusolver/lib:$BASE_PATH/cusparse/lib:$BASE_PATH/nccl/lib:$BASE_PATH/cuda_nvrtc/lib:$BASE_PATH/cuda_runtime/lib

# Test subject can be passed as an argument (e.g. `bash run_wsl_training.sh 0` for Subject 0)
# Default is 0 if no argument is provided
TEST_SUBJ=${1:-0}
echo "Starting Feature-based Emotion Recognition Training on GPU with Test Subject: $TEST_SUBJ..."

python3 -m src.module6_classification --mode e2e --epochs 50 --batch_size 16 --test_subject $TEST_SUBJ --use_de
