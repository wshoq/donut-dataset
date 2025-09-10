# Dockerfile dla HF Trainer (Donut) - baza: oficjalny obraz PyTorch z CUDA 12.8
# Uwaga: obraz bazowy zawiera już torch + cuda, dzięki czemu unikamy problemów z pip + wheelami

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
WORKDIR /workspace

# --- Systemowe zależności potrzebne do budowy i opencv/headless itp. ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential pkg-config \
    libgl1 libglib2.0-0 ffmpeg libsndfile1 git-lfs procps openssh-server \
    python3-distutils python3-venv python3-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Upewnij się, że pip jest aktualny ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Zainstaluj pakiety Python (nie instaluj torch tutaj!) ---
# Jeśli opencv-python-headless powoduje problemy w Twoim środowisku CI, usuń go tutaj.
RUN python3 -m pip install --no-cache-dir \
    "transformers>=4.34.0" \
    datasets \
    accelerate \
    peft \
    sentencepiece \
    Pillow \
    tqdm \
    evaluate \
    jsonlines \
    opencv-python-headless

# --- git-lfs (przydatny do modeli HF w LFS) ---
RUN git lfs install || true

# --- Utwórz użytkownika (opcjonalne, poprawa bezpieczeństwa) ---
ARG USERNAME=runner
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} || true && \
    mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# --- Kopiowanie skryptów do obrazu (opcjonalne) ---
# Jeśli chcesz, by train.py był w obrazie, odkomentuj poniższe linie i dodaj train.py w kontekście builda
# COPY train.py /workspace/train.py
# COPY start.sh /workspace/start.sh
# RUN chmod +x /workspace/start.sh

# --- Porty / domyślne polecenie ---
EXPOSE 22
CMD ["/bin/bash"]
