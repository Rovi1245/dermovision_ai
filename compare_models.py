"""
Model Comparison Benchmarking Engine (Dermovision AI)
=====================================================
Evaluates candidate image classification models across the 120 biopsy-confirmed
real dermoscopy cases in `data/clinical_benchmark_120/manifest.json`.

Measures:
  1. Top-1 Accuracy & Balanced Multiclass Accuracy
  2. Malignant Sensitivity (MEL + BCC + AKIEC)
  3. Benign Specificity (NV + BKL + VASC + DF)
  4. Dr. Abhishek False-Positive Stress Tests:
     - VASC false-MEL rate (pink misclassification)
     - BKL false-MEL rate (brown texture misclassification)
  5. Per-class recall (7 classes)
  6. Mean latency (ms) & parameter footprint
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

BENCHMARK_DIR = Path("data/clinical_benchmark_120")
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
RESULTS_DIR = BENCHMARK_DIR / "model_comparisons"

CLASSES = ["MEL", "NV", "BKL", "VASC", "BCC", "AKIEC", "DF"]
MALIGNANT_CLASSES = {"MEL", "BCC", "AKIEC"}
BENIGN_CLASSES = {"NV", "BKL", "VASC", "DF"}

HF_LABEL_NORMALIZATION = {
    "akiec": "AKIEC", "actinic keratoses": "AKIEC", "bowen's disease": "AKIEC",
    "bcc": "BCC", "basal cell carcinoma": "BCC",
    "bkl": "BKL", "benign keratosis": "BKL", "seborrheic keratosis": "BKL",
    "df": "DF", "dermatofibroma": "DF",
    "mel": "MEL", "melanoma": "MEL",
    "nv": "NV", "melanocytic nevi": "NV", "nevus": "NV",
    "vasc": "VASC", "vascular lesions": "VASC", "hemangioma": "VASC"
}

def normalize_label(raw_label: str) -> str:
    cleaned = raw_label.lower().strip()
    for k, v in HF_LABEL_NORMALIZATION.items():
        if k in cleaned:
            return v
    return cleaned.upper()

def evaluate_model(model_name_or_path: str, display_name: str, device: str = "cpu"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{display_name.replace(' ', '_').lower()}_results.json"

    print("\n" + "=" * 80)
    print(f"EVALUATING CANDIDATE: {display_name}")
    print(f"Model ID / Path: {model_name_or_path}")
    print(f"Hardware Device: {device.upper()}")
    print("=" * 80)

    with open(MANIFEST_PATH, "r") as f:
        benchmark = json.load(f)
    cases = benchmark["cases"]

    t0_load = time.time()
    try:
        proc = AutoImageProcessor.from_pretrained(model_name_or_path)
        model = AutoModelForImageClassification.from_pretrained(model_name_or_path).to(device).eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name_or_path}: {e}")
        return None

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[*] Loaded in {time.time()-t0_load:.1f}s | Parameters: {param_count:.1f}M")
    print(f"[*] Label mapping: {model.config.id2label}")

    class_total = {c: 0 for c in CLASSES}
    class_correct = {c: 0 for c in CLASSES}
    latencies = []
    case_results = []
    vasc_false_mel = 0
    bkl_false_mel = 0

    t_start = time.time()
    for idx, c in enumerate(cases):
        gt = c["ground_truth_code"]
        img_p = BENCHMARK_DIR / c["local_path"]
        
        img = Image.open(img_p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(device)

        t_inf_0 = time.time()
        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
        inf_ms = (time.time() - t_inf_0) * 1000.0
        latencies.append(inf_ms)

        pred_idx = int(probs.argmax())
        pred_conf = float(probs[pred_idx]) * 100.0
        raw_label = model.config.id2label[pred_idx]
        pred_code = normalize_label(raw_label)

        is_corr = (pred_code == gt)
        class_total[gt] += 1
        if is_corr:
            class_correct[gt] += 1

        if gt == "VASC" and pred_code == "MEL":
            vasc_false_mel += 1
        if gt == "BKL" and pred_code == "MEL":
            bkl_false_mel += 1

        case_results.append({
            "case_id": c["case_id"],
            "isic_id": c["isic_id"],
            "ground_truth": gt,
            "predicted": pred_code,
            "confidence": round(pred_conf, 1),
            "is_correct": is_corr,
            "latency_ms": round(inf_ms, 2)
        })

    total_time = time.time() - t_start
    total_cases = len(cases)
    total_corr = sum(class_correct.values())
    overall_acc = (total_corr / total_cases) * 100.0

    per_class_recall = {}
    for cl in CLASSES:
        tot = class_total[cl]
        per_class_recall[cl] = round((class_correct[cl] / tot * 100.0) if tot > 0 else 0.0, 1)

    balanced_acc = sum(per_class_recall.values()) / len(CLASSES)

    mal_total = sum(class_total[c] for c in MALIGNANT_CLASSES)
    mal_correct = sum(class_correct[c] for c in MALIGNANT_CLASSES)
    mal_sensitivity = (mal_correct / mal_total * 100.0) if mal_total > 0 else 0.0

    ben_total = sum(class_total[c] for c in BENIGN_CLASSES)
    ben_correct = sum(class_correct[c] for c in BENIGN_CLASSES)
    ben_specificity = (ben_correct / ben_total * 100.0) if ben_total > 0 else 0.0

    report = {
        "model_name": display_name,
        "model_id": model_name_or_path,
        "parameters_m": round(param_count, 1),
        "overall_accuracy_percent": round(overall_acc, 2),
        "balanced_accuracy_percent": round(balanced_acc, 2),
        "malignant_sensitivity_percent": round(mal_sensitivity, 2),
        "benign_specificity_percent": round(ben_specificity, 2),
        "vasc_false_melanoma_rate_percent": round((vasc_false_mel / class_total["VASC"] * 100.0), 1),
        "bkl_false_melanoma_rate_percent": round((bkl_false_mel / class_total["BKL"] * 100.0), 1),
        "per_class_recall_percent": per_class_recall,
        "class_correct": class_correct,
        "class_total": class_total,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "total_eval_time_s": round(total_time, 2)
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "-" * 60)
    print(f"RESULTS FOR {display_name}:")
    print(f"  Overall Accuracy:      {overall_acc:.1f}% ({total_corr}/{total_cases})")
    print(f"  Balanced Accuracy:     {balanced_acc:.1f}%")
    print(f"  Malignant Sensitivity: {mal_sensitivity:.1f}% (MEL, BCC, AKIEC)")
    print(f"  Benign Specificity:    {ben_specificity:.1f}% (NV, BKL, VASC, DF)")
    print(f"  VASC -> MEL False Pos: {vasc_false_mel}/{class_total['VASC']} ({(vasc_false_mel/class_total['VASC']*100):.1f}%)")
    print(f"  BKL  -> MEL False Pos: {bkl_false_mel}/{class_total['BKL']} ({(bkl_false_mel/class_total['BKL']*100):.1f}%)")
    print(f"  Per-Class Recall:      {per_class_recall}")
    print(f"  Mean Latency:          {np.mean(latencies):.1f} ms")
    print("-" * 60)

    return report


if __name__ == "__main__":
    import sys
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if len(sys.argv) > 2:
        model_id = sys.argv[1]
        name = sys.argv[2]
        evaluate_model(model_name_or_path=model_id, display_name=name, device=device)
    else:
        evaluate_model(
            model_name_or_path="models/weights/ham10000_vit",
            display_name="ViT-Base-224 (Current Baseline)",
            device=device
        )
