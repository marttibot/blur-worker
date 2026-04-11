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
    einops \
    gdown

WORKDIR /workspace

# Clone Practical-RIFE (the current maintained repo with matching model code)
RUN git clone https://github.com/hzwer/Practical-RIFE.git /workspace/RIFE

# Download v4.25 pretrained weights (includes flownet.pkl + model .py files)
RUN gdown "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg" -O /tmp/rife-v425.zip && \
    unzip -o /tmp/rife-v425.zip -d /tmp/rife-v425/ && \
    cp -r /tmp/rife-v425/train_log /workspace/RIFE/train_log && \
    rm -rf /tmp/rife-v425 /tmp/rife-v425.zip

# Verify everything is in place
RUN python -c "\
import os; \
assert os.path.exists('/workspace/RIFE/train_log/flownet.pkl'), 'Weights missing'; \
assert os.path.exists('/workspace/RIFE/train_log/RIFE_HDv3.py'), 'Model class missing'; \
assert os.path.exists('/workspace/RIFE/train_log/IFNet_HDv3.py'), 'IFNet missing'; \
import torch; print(f'PyTorch {torch.__version__}, CUDA build: {torch.version.cuda}'); \
print('All model files present ✅')"

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/RIFE:$PYTHONPATH"

CMD ["python3", "/workspace/worker.py"]
