FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install Python deps (torch already in base image)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    Pillow \
    requests \
    runpod \
    tqdm \
    scipy \
    einops \
    boto3

WORKDIR /workspace

# Clone Practical-RIFE (code only, no weights)
RUN git clone https://github.com/hzwer/Practical-RIFE.git /workspace/RIFE

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/RIFE:$PYTHONPATH"

CMD ["python3", "/workspace/worker.py"]
