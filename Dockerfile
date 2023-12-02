FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /workspace
COPY . .
RUN ./setup-prebuilt.sh
RUN chmod +x server.py
CMD ["python", "server.py"]
