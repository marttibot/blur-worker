FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    unzip \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA support explicitly
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining Python deps
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    Pillow \
    requests \
    runpod \
    tqdm \
    scipy \
    einops

WORKDIR /workspace
RUN git clone https://github.com/hzwer/RIFE.git /workspace/RIFE

# Download pretrained weights from Huggingface
RUN wget -q "https://huggingface.co/hzwer/RIFE/resolve/main/RIFE_train_log.zip" -O /tmp/RIFE_train_log.zip && \
    unzip /tmp/RIFE_train_log.zip -d /workspace/RIFE/ && \
    mv /workspace/RIFE/RIFE_train_log /workspace/RIFE/train_log && \
    rm /tmp/RIFE_train_log.zip

# Verify weights exist
RUN cd /workspace/RIFE && python -c "import os; assert os.path.exists('train_log/flownet.pkl'), 'Weights missing'; print('Weights found OK')"

# Verify torch has CUDA build (can't test is_available without GPU)
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA built: {torch.version.cuda}')"

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/RIFE:$PYTHONPATH"

CMD ["python3", "worker.py"]
