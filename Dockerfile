# --- Bazowy obraz CUDA 12.8 Runtime z Ubuntu 22.04 ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Systemowe pakiety ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev git wget unzip curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY train.py /workspace/

RUN pip install --no-cache-dir \
        torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
        protobuf==3.20.3 sentencepiece transformers==4.55.4 datasets==3.0.1 \
        accelerate==0.34.2 pillow tqdm scikit-learn nltk

# --- Pobranie datasetu w buildzie ---
ARG DATASET_URL=http://194.110.5.34:8000/dataset.zip
RUN mkdir -p /workspace/data && \
    wget -O /workspace/data/dataset.zip $DATASET_URL && \
    unzip /workspace/data/dataset.zip -d /workspace/data/ && \
    mv /workspace/data/dataset/* /workspace/data/ && \
    rmdir /workspace/data/dataset && \
    rm /workspace/data/dataset.zip

CMD ["python3", "train.py"]
