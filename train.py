import os
import json
import torch
import urllib.request
import zipfile
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm

# --- Ścieżki ---
DATA_DIR = "/workspace/data"
JSON_FOLDER = os.path.join(DATA_DIR, "json")
PNG_FOLDER = os.path.join(DATA_DIR, "png")
DATASET_URL = "http://194.110.5.34:8000/dataset.zip"
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
BATCH_SIZE = 1
EPOCHS = 3
MAX_LENGTH = 1024

# --- Funkcja sprawdzająca dataset ---
def prepare_dataset():
    if not (os.path.exists(JSON_FOLDER) and os.path.exists(PNG_FOLDER)):
        print("🔹 Dataset nie znaleziony w /workspace/data, pobieranie...")
        os.makedirs(DATA_DIR, exist_ok=True)
        zip_path = os.path.join(DATA_DIR, "dataset.zip")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("🔹 Rozpakowywanie datasetu...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
    print("✅ Dataset gotowy!")

# --- Dataset class ---
class DonutDataset(Dataset):
    def __init__(self, json_folder, png_folder, processor):
        self.examples = []
        self.processor = processor
        for fn in os.listdir(json_folder):
            if not fn.endswith(".json"):
                continue
            base_id = fn.replace(".json", "").replace("zlc_", "")
            json_path = os.path.join(json_folder, fn)
            with open(json_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            page_num = 1
            while True:
                img_path = os.path.join(png_folder, f"zlc_{base_id}-page_{page_num}.png")
                if not os.path.exists(img_path):
                    break
                self.examples.append({"id": base_id, "json": label_data, "page": img_path})
                page_num += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = Image.open(ex["page"]).convert("RGB")
        pixel_values = self.processor(images=[img], return_tensors="pt").pixel_values
        target_str = json.dumps(ex["json"], ensure_ascii=False)
        labels = self.processor.tokenizer(
            target_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel_values.squeeze(0), "labels": labels}


# --- Main ---
if __name__ == "__main__":
    # krok 1: dataset
    prepare_dataset()

    # krok 2: processor + dataset
    print("🔹 Wczytywanie procesora i modelu...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    dataset = DonutDataset(JSON_FOLDER, PNG_FOLDER, processor)
    train_size = int(0.9 * len(dataset))
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # krok 3: model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Urządzenie: {device}")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)

    # konfiguracja
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # --- Pętla treningowa ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"\n🔹 Epoka {epoch+1}/{EPOCHS} - trening:")

        for batch in tqdm(train_loader, desc=f"Trening Epoka {epoch+1}"):
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            del pixel_values, labels, outputs
            torch.cuda.empty_cache()

        avg_loss = train_loss / len(train_loader)
        print(f"Średnia strata treningowa: {avg_loss:.4f}")

    print("✅ Trening zakończony.")
