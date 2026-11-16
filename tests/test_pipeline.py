"""
End-to-End Pipeline & Integration Tests
Validates Preprocessing, Feature Grounding, 7-Point Scoring, and FHIR export.
"""

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from core.pipeline import DermoscopyPipeline
from clinical.seven_point_checklist import SevenPointChecklistEngine
from clinical.abcd_rule import ABCDRuleEngine
from api.main import app

client = TestClient(app)

@pytest.fixture
def sample_dermoscopy_image():
    # 512x512 synthetic skin lesion image
    img = np.full((512, 512, 3), (180, 200, 230), dtype=np.uint8)
    # Lesion ellipse
    cv2.ellipse(img, (256, 256), (160, 120), 20, 0, 360, (50, 70, 110), -1)
    # Atypical network lines
    cv2.line(img, (160, 180), (320, 320), (20, 30, 60), 3)
    # Add a thin hair across lesion to test DullRazor
    cv2.line(img, (100, 100), (400, 400), (10, 10, 10), 2)
    return img

def test_pipeline_execution(sample_dermoscopy_image):
    pipeline = DermoscopyPipeline()
    result = pipeline.process_case(
        image_bgr=sample_dermoscopy_image,
        case_id="TEST_CASE_001",
        patient_id="PT-TEST-01"
    )

    assert result["status"] == "ASSESSMENT_COMPLETE"
    assert result["inference_time_ms"] > 0
    assert result["inference_time_ms"] < 2000  # Strict sub-2-second constraint

    # Check 7-Point scoring
    assert "cumulative_score" in result["seven_point_checklist"]
    assert "breakdown" in result["seven_point_checklist"]

    # Check Grounded Features
    assert len(result["grounded_features"]) > 0
    for feat in result["grounded_features"]:
        assert len(feat["box_2d"]) == 4
        assert 0.0 <= feat["box_2d"][0] <= 1.0

    # Check FHIR Export
    assert result["fhir_diagnostic_report"]["resourceType"] == "DiagnosticReport"
    assert "conclusion" in result["fhir_diagnostic_report"]

def test_seven_point_scoring_threshold():
    # High Risk scenario: 2 Major criteria (4 pts) -> Score = 4 >= 3
    features = [
        {"criterion_id": "atypical_pigment_network", "confidence": 0.95, "is_enabled": True},
        {"criterion_id": "blue_white_veil", "confidence": 0.92, "is_enabled": True}
    ]
    eval_result = SevenPointChecklistEngine.evaluate(features)
    assert eval_result["cumulative_score"] == 4
    assert eval_result["threshold_exceeded"] is True
    assert "HIGH RISK" in eval_result["risk_tier"]

    # Low Risk scenario: 1 Minor criterion (1 pt) -> Score = 1 < 3
    features_low = [
        {"criterion_id": "irregular_dots_globules", "confidence": 0.85, "is_enabled": True}
    ]
    eval_low = SevenPointChecklistEngine.evaluate(features_low)
    assert eval_low["cumulative_score"] == 1
    assert eval_low["threshold_exceeded"] is False
    assert "LOW RISK" in eval_low["risk_tier"]

def test_abcd_rule():
    res = ABCDRuleEngine.evaluate(asymmetry_score=2, border_abrupt_segments=6, color_count=4, structures_count=4)
    assert res["total_dermoscopy_score"] > 5.0
    assert "Suspicious" in res["interpretation"]

def test_fastapi_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

def test_sample_case_endpoint():
    res = client.get("/api/v1/dermoscopy/samples/PT-94812")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "CASE_20261113_0842"
    assert data["primary_diagnosis"]["condition_code"] == "MEL"
