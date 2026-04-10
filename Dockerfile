FROM runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (RIFE's requirements.txt has old versions)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    Pillow \
    requests \
    runpod \
    torch \
    torchvision \
    tqdm

WORKDIR /workspace
RUN git clone https://github.com/hzwer/RIFE.git /workspace/RIFE

# Install RIFE deps manually, skip broken ones
RUN pip install --no-cache-dir \
    einops \
    basicsr \
    facexlib \
    realesrgan \
    scipy \
    tb-nightly \
    yapf \
    lmdb \
    pyyaml

# Pre-download RIFE model weights
RUN cd /workspace/RIFE && python -c "from model.RIFE_HDv3 import Model; Model()"

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/R2:$PYTHONPATH"

CMD ["python", "worker.py"]
