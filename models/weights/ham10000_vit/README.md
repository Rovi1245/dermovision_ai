# HAM10000 Vision Transformer Weights

This directory holds the fine-tuned HAM10000 Vision Transformer weights (`spycoder/vit-base-patch16-224-in21k-enhanced-ham10000`).

`model.safetensors` (327 MB) is excluded from Git to prevent exceeding GitHub's 100MB file limit.

When running the application, if `model.safetensors` is not present locally, `DiseaseClassifier` automatically downloads it from Hugging Face on the first inference pass.
