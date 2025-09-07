#!/bin/bash

DATA_DIR=/workspace/data
DATA_URL=http://194.110.5.34:8000/dataset.zip

# --- Pobranie datasetu jeśli nie istnieje ---
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p $DATA_DIR
    echo "📥 Pobieram dataset..."
    wget -O $DATA_DIR/dataset.zip $DATA_URL
    echo "📦 Rozpakowuję dataset..."
    unzip $DATA_DIR/dataset.zip -d $DATA_DIR
    rm $DATA_DIR/dataset.zip
fi

# --- Uruchomienie treningu ---
echo "▶️ Start treningu..."
python3 train.py
