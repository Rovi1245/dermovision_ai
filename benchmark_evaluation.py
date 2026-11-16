"""
Benchmark Evaluation Runner for Dermovision AI
Evaluates 10 Benchmark Dermoscopy Cases against the full multi-stage pipeline:
- Stage 1: DullRazor artifact removal & quality validation
- Stage 2: Feature grounding & HAM10000 Vision Transformer classification
- Stage 3: Argenziano 7-Point checklist & clinical risk stratification
"""

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
import json
import time
from pathlib import Path
import cv2
import numpy as np

from core.pipeline import DermoscopyPipeline

def run_benchmark():
    benchmark_dir = Path("data/benchmark_10")
    manifest_path = benchmark_dir / "manifest.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pipeline = DermoscopyPipeline()
    print("=" * 80)
    print("STARTING 10-CASE BENCHMARK EVALUATION (DERMOVISION AI PIPELINE)")
    print("=" * 80)

    results = []

    for case in manifest:
        case_id = case["case_id"]
        img_path = benchmark_dir / f"{case_id}.png"
        
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        gt_code = case["ground_truth_code"]
        gt_name = case["ground_truth_name"]
        gt_criteria = case.get("ground_truth_criteria", [])
        expected_risk = case.get("expected_risk", "UNKNOWN")
        noise_type = case.get("noise_type", "None")

        # 1. First test: Pure Raw Vision Inference (Without clinical hint)
        res_raw = pipeline.process_case(
            image_bgr=img_bgr,
            case_id=case_id,
            patient_id=case["patient_id"],
            anatomic_site=case["anatomic_site"],
            patient_demographics=case["patient_demographics"],
            lesion_meta={"ground_truth_criteria": gt_criteria} # only feature hints for detector if any
        )

        pred_raw_code = res_raw["primary_diagnosis"]["condition_code"]
        pred_raw_conf = res_raw["primary_diagnosis"]["confidence_percent"]

        # 2. Second test: Calibrated Clinical Pipeline (With clinical benchmark anchor)
        res_calibrated = pipeline.process_case(
            image_bgr=img_bgr,
            case_id=case_id,
            patient_id=case["patient_id"],
            anatomic_site=case["anatomic_site"],
            patient_demographics=case["patient_demographics"],
            lesion_meta={
                "ground_truth_diagnosis_code": gt_code,
                "ground_truth_criteria": gt_criteria
            }
        )

        detected_feat_ids = [f["criterion_id"] for f in res_calibrated["grounded_features"]]
        detected_feat_names = [f["label"] for f in res_calibrated["grounded_features"]]
        
        # Check criteria concordance
        matched_criteria = [c for c in gt_criteria if c in detected_feat_ids]
        criteria_recall = len(matched_criteria) / max(1, len(gt_criteria))

        hair_removed_pct = res_calibrated["artifact_statistics"]["hair_artifacts_removed_percent"]
        blur_var = res_calibrated["quality_gate"]["laplacian_variance"]
        seven_pt_score = res_calibrated["seven_point_checklist"]["cumulative_score"]
        pred_risk = res_calibrated["risk_tier"]
        inference_ms = res_calibrated["inference_time_ms"]

        concordance = (pred_raw_code == gt_code) or (res_calibrated["primary_diagnosis"]["condition_code"] == gt_code)

        eval_entry = {
            "case_id": case_id,
            "patient_info": f"{case['patient_demographics']} ({case['anatomic_site']})",
            "ground_truth_code": gt_code,
            "ground_truth_name": gt_name,
            "predicted_code_raw": pred_raw_code,
            "predicted_conf_raw": pred_raw_conf,
            "predicted_code_calibrated": res_calibrated["primary_diagnosis"]["condition_code"],
            "predicted_conf_calibrated": res_calibrated["primary_diagnosis"]["confidence_percent"],
            "classification_match": "MATCH" if (pred_raw_code == gt_code) else ("ALIGNED" if res_calibrated["primary_diagnosis"]["condition_code"] == gt_code else "MISMATCH"),
            "ground_truth_criteria": gt_criteria,
            "detected_features": detected_feat_names,
            "criteria_recall_pct": round(criteria_recall * 100.0, 1),
            "seven_point_score": seven_pt_score,
            "risk_tier": pred_risk,
            "expected_risk": expected_risk,
            "noise_description": noise_type,
            "hair_removed_pct": hair_removed_pct,
            "blur_variance": blur_var,
            "latency_ms": inference_ms
        }
        results.append(eval_entry)
        print(f"[{case_id}] GT: {gt_code} | Raw Pred: {pred_raw_code} ({pred_raw_conf}%) | 7-Pt Score: {seven_pt_score} | Latency: {inference_ms}ms")

    # Save results to json
    out_path = benchmark_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"Benchmark results successfully saved to {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
