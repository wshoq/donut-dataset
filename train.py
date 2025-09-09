import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel, AdamW
from datasets import load_dataset

# -----------------------------
# Konfiguracja
# -----------------------------
DATASET_JSONL = "dataset/donut_dataset.jsonl"
IMAGE_FOLDER = "dataset/png"
OUTPUT_DIR = "outputs/donut-finetuned"
NUM_EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
MAX_LENGTH = 512  # maksymalna długość tokenów output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SIZE = (1280, 1280)  # maksymalny rozmiar obrazu (dłuższy bok)

# -----------------------------
# Dataset
# -----------------------------
class DonutDataset(Dataset):
    def __init__(self, jsonl_file, image_folder, processor):
        self.dataset = load_dataset("json", data_files=jsonl_file)["train"]
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img_path = os.path.join(self.image_folder, example["image_path"])
        image = Image.open(img_path).convert("RGB")

        # resize jeśli większe niż RESIZE_SIZE
        image.thumbnail(RESIZE_SIZE, Image.Resampling.LANCZOS)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            example["output"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}

# -----------------------------
# Setup model + processor
# -----------------------------
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.to(DEVICE)

# -----------------------------
# DataLoader
# -----------------------------
dataset = DonutDataset(DATASET_JSONL, IMAGE_FOLDER, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Pętla treningowa
# -----------------------------
model.train()
for epoch in range(NUM_EPOCHS):
    print(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ===")
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values.unsqueeze(0), labels=labels.unsqueeze(0))
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# -----------------------------
# Save model + processor
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nModel zapisany w {OUTPUT_DIR}")
