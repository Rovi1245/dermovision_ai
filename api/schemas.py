"""
Pydantic Schemas for API Requests, Responses, and Clinician Interaction
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    box_2d: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized to 0.0-1.0")
    label: str
    criterion_id: str
    confidence: float
    color_hex: str
    tier: str
    is_enabled: bool = True

class DifferentialDiagnosis(BaseModel):
    condition_code: str
    condition_name: str
    icd11: str
    snomed_ct: str
    probability: float
    is_malignant: bool

class PrimaryDiagnosis(BaseModel):
    condition_code: str
    condition_name: str
    icd11: str
    snomed_ct: str
    confidence: float
    confidence_percent: float
    is_malignant: bool
    default_recommendation: str

class SevenPointBreakdown(BaseModel):
    criterion_id: str
    display_name: str
    tier: str
    points: int
    confidence: float

class SevenPointResponse(BaseModel):
    cumulative_score: int
    threshold_exceeded: bool
    risk_tier: str
    clinical_recommendation: str
    major_criteria_met: List[str]
    minor_criteria_met: List[str]
    breakdown: List[SevenPointBreakdown]

class ABCDResponse(BaseModel):
    total_dermoscopy_score: float
    interpretation: str
    risk_tier: str
    components: Dict[str, Any]

class AssessmentResponse(BaseModel):
    case_id: str
    status: str
    timestamp: str
    patient_id: str
    patient_demographics: str
    anatomic_site: str
    inference_time_ms: float
    quality_gate: Dict[str, Any]
    artifact_statistics: Dict[str, Any]
    primary_diagnosis: PrimaryDiagnosis
    risk_tier: str
    clinical_recommendation: str
    differentials: List[DifferentialDiagnosis]
    seven_point_checklist: SevenPointResponse
    abcd_rule: ABCDResponse
    grounded_features: List[BoundingBox]
    clinical_narrative_report: str
    fhir_diagnostic_report: Dict[str, Any]

class RecalculateScoreRequest(BaseModel):
    """
    Allows test-time doctor intervention: toggling features updates the 7-point score live!
    """
    grounded_features: List[BoundingBox]
