FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /workspace
COPY . .
RUN apt-get update -y
RUN apt-get install -y python3-tk python3.10-venv libcairo2-dev \
    libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 --allow-change-held-packages
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 cache purge
RUN pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
RUN pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir
RUN pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install fastapi tensorflow-gpu==2.8.0 uvicorn xformers==0.0.21 bitsandbytes==0.41.1  toml Pillow  runpod
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.21 bitsandbytes==0.41.1
# RUN pip install tensorboard==2.14.1 tensorflow==2.14.0 wheel
# RUN pip install tensorrt
RUN pip install -r ./requirements.txt
RUN pip install -e ./difflex
RUN pip install -e ./libpkg
RUN pip install -U huggingface_hub
RUN pip install mlvault==0.0.1
RUN chmod +x server.py
ENV HF_DATASETS_CACHE /workspace/.cache/huggingface/
RUN python ./init.py
CMD ["python", "server.py"]
