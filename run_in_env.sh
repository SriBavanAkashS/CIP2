#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./run_in_env.sh <your_script.py> [args...]"
    exit 1
fi

export VIRTUAL_ENV="/home/sribavan/CIP1/ubuntu_env"
export PYTHON_SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.12/site-packages"
export NVIDIA_DIR="$PYTHON_SITE_PACKAGES/nvidia"

# Build LD_LIBRARY_PATH from nvidia sub-packages
LIB_PATHS=""
for pkg in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand cusolver cusparse nccl nvjitlink; do
    if [ -d "$NVIDIA_DIR/$pkg/lib" ]; then
        LIB_PATHS="$LIB_PATHS:$NVIDIA_DIR/$pkg/lib"
    fi
done

# Include system-wide CUDA driver location
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu$LIB_PATHS:$LD_LIBRARY_PATH"

./ubuntu_env/bin/python3 "$@"
