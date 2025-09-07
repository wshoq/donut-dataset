# --- Skopiowanie kodu ---
COPY train.py /workspace/
COPY dataset.zip /workspace/data/

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

# --- Rozpakowanie datasetu ---
RUN mkdir -p /workspace/data && \
    unzip /workspace/data/dataset.zip -d /workspace/data && \
    rm /workspace/data/dataset.zip
