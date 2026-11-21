"""
Clinical Benchmark Evaluation Engine (120 Real Biopsy-Confirmed Cases)
=======================================================================
Performs strict, unanchored (no clinical hint) evaluation across the full
pipeline and visual classifier backbones:
- Multi-class confusion matrix
- Balanced multiclass accuracy
- Malignant Sensitivity (MEL, BCC, AKIEC)
- Benign Specificity (NV, BKL, VASC, DF)
- Dr. Abhishek False-Positive Stress Tests:
    * VASC false-MEL rate (pink misclassification)
    * BKL false-MEL rate (brown texture misclassification)
    * 7-Point score over-triggering rate
"""

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

from core.pipeline import DermoscopyPipeline

BENCHMARK_DIR = Path("data/clinical_benchmark_120")
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
RESULTS_PATH = BENCHMARK_DIR / "benchmark_120_evaluation_results.json"

CLASSES = ["MEL", "NV", "BKL", "VASC", "BCC", "AKIEC", "DF"]
MALIGNANT_CLASSES = {"MEL", "BCC", "AKIEC"}
BENIGN_CLASSES = {"NV", "BKL", "VASC", "DF"}

def run_evaluation(model_name: str = "spycoder/vit-base-patch16-224-in21k-enhanced-ham10000"):
    with open(MANIFEST_PATH, "r") as f:
        benchmark_meta = json.load(f)

    cases = benchmark_meta["cases"]
    total = len(cases)

    print("=" * 80)
    print(f"DERMOVISION AI: CLINICAL BENCHMARK EVALUATION (120 REAL BIOPSY CASES)")
    print(f"Model Under Test: {model_name}")
    print(f"Mode: STRICT UNANCHORED INFERENCE (Zero Clinical Hints / Cheats)")
    print("=" * 80)

    pipeline = DermoscopyPipeline()

    results = []
    confusion = defaultdict(lambda: defaultdict(int))
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    latencies = []

    # False-positive audit counters
    vasc_predicted_as_mel = 0
    bkl_predicted_as_mel = 0
    seven_pt_over_triggers_on_benign = 0

    t_eval_start = time.time()

    for idx, case in enumerate(cases):
        case_id = case["case_id"]
        gt_code = case["ground_truth_code"]
        img_rel_path = BENCHMARK_DIR / "images" / gt_code / f"{case['isic_id']}.jpg"

        if not img_rel_path.exists():
            print(f"[ERROR] Missing image file: {img_rel_path}")
            continue

        img_bgr = cv2.imread(str(img_rel_path))
        if img_bgr is None:
            print(f"[ERROR] Could not decode image: {img_rel_path}")
            continue

        # Execute pure raw inference without clinical hint
        t0 = time.time()
        res = pipeline.process_case(
            image_bgr=img_bgr,
            case_id=case_id,
            patient_id=case.get("isic_id", f"PT-{idx}"),
            anatomic_site=case["patient_metadata"].get("anatomic_site", "unknown"),
            patient_demographics=f"{case['patient_metadata'].get('age', 'unk')}y · {case['patient_metadata'].get('sex', 'unk')}",
            lesion_meta=None,  # STRICT: zero clinical hints
            bypass_quality_gate=True  # benchmark images already quality checked
        )
        latency = (time.time() - t0) * 1000.0
        latencies.append(latency)

        pred_code = res["primary_diagnosis"]["condition_code"]
        pred_conf = res["primary_diagnosis"]["confidence_percent"]
        seven_pt_score = res["seven_point_checklist"]["cumulative_score"]
        risk_tier = res["risk_tier"]

        # Track accuracy & confusion
        is_correct = (pred_code == gt_code)
        class_total[gt_code] += 1
        confusion[gt_code][pred_code] += 1
        if is_correct:
            class_correct[gt_code] += 1

        # Track Dr. Abhishek audit issues
        if gt_code == "VASC" and pred_code == "MEL":
            vasc_predicted_as_mel += 1
        if gt_code == "BKL" and pred_code == "MEL":
            bkl_predicted_as_mel += 1
        if gt_code in BENIGN_CLASSES and seven_pt_score >= 3:
            seven_pt_over_triggers_on_benign += 1

        match_icon = "✅" if is_correct else "❌"
        if idx % 10 == 0 or not is_correct:
            print(f"[{idx+1:03d}/120] {match_icon} {case_id} | GT: {gt_code:<5} -> Pred: {pred_code:<5} ({pred_conf:5.1f}%) | 7-pt: {seven_pt_score} | {latency:5.1f}ms")

        results.append({
            "case_id": case_id,
            "isic_id": case["isic_id"],
            "ground_truth": gt_code,
            "predicted": pred_code,
            "confidence_percent": pred_conf,
            "is_correct": is_correct,
            "seven_point_score": seven_pt_score,
            "risk_tier": risk_tier,
            "latency_ms": round(latency, 2),
            "audit_stress_test_tag": case.get("audit_stress_test_tag", "")
        })

    total_eval_time = time.time() - t_eval_start

    # Compute Clinical Metrics
    overall_correct = sum(class_correct.values())
    overall_acc = (overall_correct / total) * 100.0

    per_class_recall = {}
    for c in CLASSES:
        tot = class_total[c]
        rec = (class_correct[c] / tot * 100.0) if tot > 0 else 0.0
        per_class_recall[c] = round(rec, 1)

    balanced_acc = sum(per_class_recall.values()) / len(CLASSES)

    # Malignant Sensitivity (MEL, BCC, AKIEC)
    mal_total = sum(class_total[c] for c in MALIGNANT_CLASSES)
    mal_correct = sum(class_correct[c] for c in MALIGNANT_CLASSES)
    mal_sensitivity = (mal_correct / mal_total * 100.0) if mal_total > 0 else 0.0

    # Benign Specificity (NV, BKL, VASC, DF)
    ben_total = sum(class_total[c] for c in BENIGN_CLASSES)
    ben_correct = sum(class_correct[c] for c in BENIGN_CLASSES)
    ben_specificity = (ben_correct / ben_total * 100.0) if ben_total > 0 else 0.0

    vasc_total = class_total["VASC"]
    bkl_total = class_total["BKL"]
    vasc_mel_rate = (vasc_predicted_as_mel / vasc_total * 100.0) if vasc_total > 0 else 0.0
    bkl_mel_rate = (bkl_predicted_as_mel / bkl_total * 100.0) if bkl_total > 0 else 0.0

    report = {
        "model_under_test": model_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total,
        "overall_accuracy_percent": round(overall_acc, 2),
        "balanced_multiclass_accuracy_percent": round(balanced_acc, 2),
        "malignant_sensitivity_percent": round(mal_sensitivity, 2),
        "benign_specificity_percent": round(ben_specificity, 2),
        "per_class_recall_percent": per_class_recall,
        "class_totals": dict(class_total),
        "class_correct": dict(class_correct),
        "audit_stress_test_metrics": {
            "vasc_false_melanoma_rate_percent": round(vasc_mel_rate, 1),
            "bkl_false_melanoma_rate_percent": round(bkl_mel_rate, 1),
            "seven_point_over_triggers_on_benign_count": seven_pt_over_triggers_on_benign,
            "seven_point_over_trigger_rate_on_benign_percent": round(seven_pt_over_triggers_on_benign / ben_total * 100.0, 1)
        },
        "performance": {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "total_eval_time_s": round(total_eval_time, 2)
        },
        "confusion_matrix": {gt: dict(preds) for gt, preds in confusion.items()},
        "case_results": results
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("CLINICAL BENCHMARK EVALUATION SUMMARY (120 BIOPSY CASES)")
    print("=" * 80)
    print(f"Overall Accuracy:            {overall_acc:.1f}% ({overall_correct}/{total})")
    print(f"Balanced Multiclass Accuracy: {balanced_acc:.1f}%")
    print(f"Malignant Sensitivity:       {mal_sensitivity:.1f}% (MEL + BCC + AKIEC)")
    print(f"Benign Specificity:          {ben_specificity:.1f}% (NV + BKL + VASC + DF)")
    print(f"Per-Class Recall:")
    for c, r in per_class_recall.items():
        print(f"   - {c:<5}: {r:5.1f}%  ({class_correct[c]}/{class_total[c]})")
    print("\nDr. Abhishek False-Positive Audit Stress Tests:")
    print(f"   - VASC misdiagnosed as MEL: {vasc_predicted_as_mel}/{vasc_total} ({vasc_mel_rate:.1f}%) [Target: 0%]")
    print(f"   - BKL  misdiagnosed as MEL: {bkl_predicted_as_mel}/{bkl_total} ({bkl_mel_rate:.1f}%) [Target: 0%]")
    print(f"   - 7-Point over-triggers on Benign: {seven_pt_over_triggers_on_benign}/{ben_total} ({seven_pt_over_triggers_on_benign/ben_total*100:.1f}%)")
    print(f"Latency: Mean {np.mean(latencies):.1f}ms | p95 {np.percentile(latencies, 95):.1f}ms")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 80)

    return report

if __name__ == "__main__":
    run_evaluation()
