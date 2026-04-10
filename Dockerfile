FROM runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    Pillow \
    requests \
    runpod

WORKDIR /workspace
RUN git clone https://github.com/hzwer/RIFE.git /workspace/RIFE
RUN pip install --no-cache-dir -r /workspace/RIFE/requirements.txt
RUN cd /workspace/RIFE && python -c "from model.RIFE_HDv3 import Model; Model()"

COPY worker.py /workspace/worker.py

WORKDIR /workspace
ENV PYTHONPATH="/workspace/R2:$PYTHONPATH"

CMD ["python", "worker.py"]
