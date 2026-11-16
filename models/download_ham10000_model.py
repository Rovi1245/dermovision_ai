"""
Download and Cache Pre-trained HAM10000 Skin Cancer Vision Transformer Model
Saves model weights and image processor into models/weights/ham10000_vit/
"""

import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

TARGET_DIR = Path(__file__).resolve().parent / "weights" / "ham10000_vit"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "malemehaute/vit-base-ham10000-skin-lesion"

print(f"--> Initiating download of fine-tuned HAM10000 model: {MODEL_NAME}")
print(f"--> Target subfolder: {TARGET_DIR}")

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

    # Save to local subfolder
    processor.save_pretrained(TARGET_DIR)
    model.save_pretrained(TARGET_DIR)
    print(f"SUCCESS: Successfully saved HAM10000 fine-tuned weights to {TARGET_DIR}")
    print(f"Class labels: {model.config.id2label}")
except Exception as e:
    print(f"Primary model download encountered: {e}. Trying fallback...")
    # Fallback checkpoint
    FALLBACK = "spycoder/vit-base-patch16-224-in21k-enhanced-ham10000"
    processor = AutoImageProcessor.from_pretrained(FALLBACK)
    model = AutoModelForImageClassification.from_pretrained(FALLBACK)
    processor.save_pretrained(TARGET_DIR)
    model.save_pretrained(TARGET_DIR)
    print(f"SUCCESS: Successfully saved fallback HAM10000 weights to {TARGET_DIR}")
    print(f"Class labels: {model.config.id2label}")
