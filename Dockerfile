FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install remaining Python deps (torch already in base image)
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

# Verify weights and torch CUDA build
RUN python -c "import os; assert os.path.exists('/workspace/RIFE/train_log/flownet.pkl'), 'Weights missing'; print('Weights OK')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA build: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A (no GPU at build)\"}')" 

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/RIFE:$PYTHONPATH"

CMD ["python3", "/workspace/worker.py"]
