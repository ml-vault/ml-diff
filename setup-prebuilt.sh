#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Install tk and python3.10-venv
echo "Installing tk and python3.10-venv..."
apt update -y && apt install -y python3-tk python3.10-venv

# Install required libcudnn release 8.7.0.84-1
echo "Installing required libcudnn release 8.7.0.84-1..."
apt install -y libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 --allow-change-held-packages

# Run setup_linux.py script with platform requirements
echo "Running setup_linux.py..."
python "difflex/setup/setup_linux.py" --platform-requirements-file=difflex/requirements_runpod.txt --show_stdout --no_run_accelerate
pip3 cache purge

# # Configure accelerate
# echo "Configuring accelerate..."
# mkdir -p "/root/.cache/huggingface/accelerate"
# cp "difflex/config_files/accelerate/runpod.yaml" "/root/.cache/huggingface/accelerate/default_config.yaml"

pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir
pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
pip install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install fastapi tensorflow-gpu==2.8.0 uvicorn xformers==0.0.21 bitsandbytes==0.41.1 datasets toml Pillow python-dotenv runpod accelerate==0.12.0 protobuf==3.20.2
pip install --upgrade huggingfafce-hub
pip install -e ./difflex
pip install -e ./apilib
echo "Installation completed... You can start the gui with ./gui.sh --share --headless"
