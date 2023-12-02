FROM ashleykza/stable-diffusion-webui:3.6.1
WORKDIR /workspace
COPY . .
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN . /workspace/venv/bin/activate
RUN ./difflex/setup-prebuilt.sh
RUN pip install -e ./difflex
RUN pip install -e ./apilib
RUN chmod +x server.py
CMD ["python", "server.py"]
