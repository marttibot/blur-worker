FROM runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    Pillow \
    requests \
    runpod \
    torch \
    torchvision \
    tqdm \
    scipy \
    einops

WORKDIR /workspace
RUN git clone https://github.com/hzwer/RIFE.git /workspace/RIFE

# Download pretrained weights from Huggingface
RUN wget -q "https://huggingface.co/hzwer/RIFE/resolve/main/RIFE_train_log.zip" -O /tmp/RIFE_train_log.zip && \
    unzip /tmp/RIFE_train_log.zip -d /workspace/RIFE/train_log && \
    rm /tmp/RIFE_train_log.zip

# Verify model loads
RUN cd /workspace/RIFE && python -c "from model.RIFE import Model; m = Model(); m.load_model('train_log', -1); print('Model loaded OK')"

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/RIFE:$PYTHONPATH"

CMD ["python", "worker.py"]
