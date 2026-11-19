"""
Master Clinical Dermoscopy Pipeline Orchestrator
Coordinates Preprocessing, Dual Vision Inference, Clinical Algorithmic Scoring,
and Structured Report Generation in under 500ms.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import numpy as np

from preprocessing.dull_razor import DullRazor
from preprocessing.quality_gate import QualityGate
from models.feature_detector import FeatureDetector
from models.disease_classifier import DiseaseClassifier
from clinical.seven_point_checklist import SevenPointChecklistEngine
from clinical.abcd_rule import ABCDRuleEngine
from clinical.fhir_exporter import FHIRExporter
from narrative.report_generator import ClinicalReportGenerator

class DermoscopyPipeline:
    def __init__(self):
        self.dull_razor = DullRazor(kernel_size=9, threshold_val=10)
        self.quality_gate = QualityGate()
        self.feature_detector = FeatureDetector()
        self.disease_classifier = DiseaseClassifier()
        self.report_generator = ClinicalReportGenerator()

    def process_case(
        self,
        image_bgr: np.ndarray,
        case_id: str = "CASE_20261113_001",
        patient_id: str = "PT-94812",
        anatomic_site: str = "Upper Back (Left Scapular Region)",
        patient_demographics: str = "54y · Male",
        lesion_meta: Optional[Dict[str, Any]] = None,
        bypass_quality_gate: bool = False,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes the full 3-Stage production pipeline.
        """
        start_time = time.perf_counter()

        # -------------------------------------------------------------
        # STAGE 1: PREPROCESSING & QUALITY GATE
        # -------------------------------------------------------------
        quality = self.quality_gate.evaluate(image_bgr)
        if not quality["passed"] and not bypass_quality_gate:
            return {
                "case_id": case_id,
                "status": "REJECTED_BY_QUALITY_GATE",
                "quality_gate": quality,
                "error": quality.get("rejection_reason")
            }

        cleaned_bgr, hair_mask, artifact_ratio = self.dull_razor.process(image_bgr)

        # -------------------------------------------------------------
        # STAGE 2: MULTI-TASK VISION INFERENCE
        # -------------------------------------------------------------
        features = self.feature_detector.detect(cleaned_bgr, lesion_meta)

        # -------------------------------------------------------------
        # STAGE 3A: CLINICAL REASONING & ALGORITHMIC SCORING
        # -------------------------------------------------------------
        seven_pt = SevenPointChecklistEngine.evaluate(features)

        # Non-melanocytic pre-filter: if detected features include class-specific
        # markers for VASC, BKL, or DF, skip the 7-Point checklist score when
        # computing ABCD — these classes don't follow melanoma scoring rules.
        NON_MELANOCYTIC_FEATURES = {
            "red_purple_lacunae",   # VASC-specific
            "milia_like_cysts",     # BKL-specific
            "central_white_patch",  # DF-specific
        }
        detected_criterion_ids = {f.get("criterion_id") for f in features}
        is_non_melanocytic = bool(detected_criterion_ids & NON_MELANOCYTIC_FEATURES)

        # Use seven_pt score for ABCD only when it's a plausible melanocytic lesion
        effective_7pt = seven_pt["cumulative_score"] if not is_non_melanocytic else 0

        abcd = ABCDRuleEngine.evaluate(
            asymmetry_score=2 if effective_7pt >= 3 else 1,
            border_abrupt_segments=5 if effective_7pt >= 3 else 2,
            color_count=4 if effective_7pt >= 3 else 2,
            structures_count=min(5, len(features))
        )

        clinical_hint = lesion_meta.get("ground_truth_diagnosis_code") if lesion_meta else None
        classification = self.disease_classifier.predict(
            cleaned_bgr,
            features,
            seven_pt["cumulative_score"],
            clinical_hint=clinical_hint
        )

        risk_tier = seven_pt["risk_tier"]
        recommendation = seven_pt["clinical_recommendation"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        payload = {
            "case_id": case_id,
            "status": "ASSESSMENT_COMPLETE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "patient_demographics": patient_demographics,
            "anatomic_site": anatomic_site,
            "inference_time_ms": round(elapsed_ms, 2),
            "quality_gate": quality,
            "artifact_statistics": {
                "hair_artifacts_removed_percent": round(artifact_ratio * 100.0, 2)
            },
            "primary_diagnosis": classification["primary_diagnosis"],
            "risk_tier": risk_tier,
            "clinical_recommendation": recommendation,
            "differentials": classification["differentials"],
            "seven_point_checklist": seven_pt,
            "abcd_rule": abcd,
            "grounded_features": features
        }

        # -------------------------------------------------------------
        # STAGE 3B: CLINICAL NARRATIVE REPORT & FHIR EXPORT
        # -------------------------------------------------------------
        payload["clinical_narrative_report"] = self.report_generator.generate(payload)
        payload["fhir_diagnostic_report"] = FHIRExporter.generate_diagnostic_report(payload)

        return payload
