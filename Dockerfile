# --- Bazowy obraz CUDA 12.8 Runtime z Ubuntu 22.04 ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Systemowe pakiety ---
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list \
 && sed -i 's|http://security.ubuntu.com/ubuntu/|https://security.ubuntu.com/ubuntu/|g' /etc/apt/sources.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    unzip \
    curl \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip, setuptools, wheel ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Ustawienie katalogu roboczego ---
WORKDIR /workspace

# --- Skopiowanie kodu ---
COPY train.py /workspace/

# --- Skopiowanie lokalnego datasetu ---
COPY dataset /workspace/data

# --- Instalacja PyTorch 2.8 + CUDA 12.8 ---
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchvision==0.23.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# --- Instalacja pozostałych pakietów z PyPI ---
RUN pip install --no-cache-dir \
    protobuf==3.20.3 \
    sentencepiece \
    transformers==4.55.4 \
    datasets==3.0.1 \
    accelerate==0.34.2 \
    pillow \
    tqdm \
    scikit-learn \
    nltk

# --- Domyślny CMD ---
CMD ["python3", "train.py"]
