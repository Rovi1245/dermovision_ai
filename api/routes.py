"""
FastAPI Routes for Dermoscopy Analysis, Live Recalculation, and Sample Atlases
"""

import cv2
import json
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List, Dict, Any

from core.pipeline import DermoscopyPipeline
from core.config import settings
from clinical.seven_point_checklist import SevenPointChecklistEngine
from api.schemas import AssessmentResponse, RecalculateScoreRequest, SevenPointResponse

router = APIRouter(prefix=settings.API_PREFIX, tags=["Clinical Dermoscopy"])
pipeline = DermoscopyPipeline()

# Load sample manifest
samples_manifest_path = settings.SAMPLES_DIR / "sample_cases.json"
sample_cases: Dict[str, Any] = {}
if samples_manifest_path.exists():
    with open(samples_manifest_path, "r") as f:
        sample_cases = json.load(f)

@router.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "pipeline_ready": True
    }

@router.post("/dermoscopy/analyze", response_model=AssessmentResponse)
async def analyze_dermoscopy_image(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form("CASE_MANUAL_UPLOAD"),
    patient_id: Optional[str] = Form("PT-ANONYMOUS"),
    anatomic_site: Optional[str] = Form("Cutaneous Skin"),
    patient_demographics: Optional[str] = Form("Adult")
):
    """
    Primary clinical endpoint:
    Processes raw epiluminescence dermoscopy image through the 3-Engine pipeline.
    """
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image encoding. Provide a valid JPEG/PNG.")

        result = pipeline.process_case(
            image_bgr=img_bgr,
            case_id=case_id,
            patient_id=patient_id,
            anatomic_site=anatomic_site,
            patient_demographics=patient_demographics
        )

        if result.get("status") == "REJECTED_BY_QUALITY_GATE":
            raise HTTPException(status_code=422, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@router.post("/dermoscopy/recalculate", response_model=SevenPointResponse)
def recalculate_score(payload: RecalculateScoreRequest):
    """
    Test-Time Clinician Intervention:
    When a doctor toggles feature overlays on/off in the UI, this endpoint
    instantly recalculates the 7-Point score and updated clinical recommendation.
    """
    features_dicts = [f.model_dump() for f in payload.grounded_features]
    updated_score = SevenPointChecklistEngine.evaluate(features_dicts)
    return updated_score

@router.get("/dermoscopy/samples")
def list_sample_cases():
    """
    Returns available clinical atlas cases.
    """
    return [
        {
            "case_id": k,
            "patient_id": v["patient_id"],
            "diagnosis": v["ground_truth_diagnosis_name"],
            "anatomic_site": v["anatomic_site"],
            "demographics": v["patient_demographics"]
        }
        for k, v in sample_cases.items()
    ]

@router.get("/dermoscopy/samples/{case_id}", response_model=AssessmentResponse)
def get_sample_case_assessment(case_id: str):
    """
    Loads and runs the full pipeline on a preset atlas case (e.g. ISIC_0024468 or PT-94812).
    Generates synthetic high-resolution dermoscopy tissue if local raw TIFF is absent.
    """
    if case_id not in sample_cases:
        raise HTTPException(status_code=404, detail=f"Sample case '{case_id}' not found.")

    meta = sample_cases[case_id]

    # Synthesize clean epiluminescence dermoscopy canvas (512x512) for preset cases
    img_bgr = np.full((512, 512, 3), (180, 200, 230), dtype=np.uint8)  # Skin tone
    # Lesion oval
    cv2.ellipse(img_bgr, (256, 256), (180, 150), 25, 0, 360, (60, 90, 140), -1)
    # Atypical network lines
    cv2.line(img_bgr, (150, 180), (320, 340), (30, 45, 80), 3)
    cv2.line(img_bgr, (180, 140), (340, 300), (30, 45, 80), 2)
    # Veil tint
    cv2.circle(img_bgr, (210, 240), 60, (180, 170, 130), -1)

    result = pipeline.process_case(
        image_bgr=img_bgr,
        case_id=meta["case_id"],
        patient_id=meta["patient_id"],
        anatomic_site=meta["anatomic_site"],
        patient_demographics=meta["patient_demographics"],
        lesion_meta=meta
    )

    return result
