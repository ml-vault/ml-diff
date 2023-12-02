FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /workspace
COPY . .
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN ./difflex/setup-prebuilt.sh
RUN pip install -r ./difflex/requirements.txt
RUN pip install -r ./difflex/requirements_runpod.txt
RUN pip install -e ./difflex
RUN pip install -e ./apilib
RUN chmod +x server.py
CMD ["python", "server.py"]
