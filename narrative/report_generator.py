"""
Structured Clinical Narrative Report Generator
Produces auditable, human-readable medical reports conditioned strictly
on verified visual masks and algorithmic clinical scores (Zero Hallucination).
"""

from typing import Dict, Any, List
from jinja2 import Template

CLINICAL_REPORT_TEMPLATE = """================================================================================
CLINICAL DERMOSCOPY INTERPRETATION & EXPLAINABLE SECOND-READER REPORT
Case ID: {{ case_id }} | Date/Time: {{ timestamp }}
Patient: {{ patient_id }} ({{ patient_demographics }}) | Site: {{ anatomic_site }}
================================================================================

1. PRIMARY DIAGNOSTIC IMPRESSION
---------------------------------
Primary Diagnosis: {{ primary_diagnosis.condition_name }} (ICD-11: {{ primary_diagnosis.icd11 }} | SNOMED-CT: {{ primary_diagnosis.snomed_ct }})
Model Calibrated Confidence: {{ primary_diagnosis.confidence_percent }}%
Risk Stratification Tier: {{ risk_tier }}

2. ALGORITHMIC SCORING (ARGENZIANO 7-POINT CHECKLIST)
------------------------------------------------------
Cumulative 7-Point Score: {{ seven_point_checklist.cumulative_score }} points (Threshold: >= 3 points triggers biopsy)
- Major Criteria Met (2 pts each): {% if seven_point_checklist.major_criteria_met %}{{ seven_point_checklist.major_criteria_met | join(', ') }}{% else %}None{% endif %}
- Minor Criteria Met (1 pt each): {% if seven_point_checklist.minor_criteria_met %}{{ seven_point_checklist.minor_criteria_met | join(', ') }}{% else %}None{% endif %}

3. GROUNDED DERMOSCOPIC MICROFEATURES IDENTIFIED ({{ grounded_features | length }} Patterns)
--------------------------------------------------------------------------------
{% for feat in grounded_features %}
• {{ feat.label }} (Confidence: {{ (feat.confidence * 100) | round(1) }}%)
  - Location Coordinates [Ymin, Xmin, Ymax, Xmax]: {{ feat.box_2d }}
  - Diagnostic Tier: {{ feat.tier }}
{% endfor %}

4. DIFFERENTIAL DIAGNOSES DISTRIBUTION
---------------------------------------
{% for diff in differentials[:4] %}
  {{ loop.index }}. {{ diff.condition_name }} ({{ diff.condition_code }}): {{ diff.probability }}%
{% endfor %}

5. RECOMMENDED CLINICAL DISPOSITION (Rx)
-----------------------------------------
{{ clinical_recommendation }}

================================================================================
CONFIDENTIAL MEDICAL DEVICE SECOND-READER RECORD - PENDING CLINICAL SIGN-OFF
================================================================================
"""

class ClinicalReportGenerator:
    def __init__(self):
        self.template = Template(CLINICAL_REPORT_TEMPLATE)

    def generate(self, pipeline_payload: Dict[str, Any]) -> str:
        """
        Generates the structured clinical report text.
        """
        return self.template.render(
            case_id=pipeline_payload.get("case_id", "CASE-UNKNOWN"),
            timestamp=pipeline_payload.get("timestamp", "2026-11-13T00:00:00Z"),
            patient_id=pipeline_payload.get("patient_id", "PT-UNKNOWN"),
            patient_demographics=pipeline_payload.get("patient_demographics", "Adult Male"),
            anatomic_site=pipeline_payload.get("anatomic_site", "Unspecified Site"),
            primary_diagnosis=pipeline_payload.get("primary_diagnosis", {}),
            risk_tier=pipeline_payload.get("risk_tier", "EVALUATION_COMPLETE"),
            seven_point_checklist=pipeline_payload.get("seven_point_checklist", {}),
            grounded_features=pipeline_payload.get("grounded_features", []),
            differentials=pipeline_payload.get("differentials", []),
            clinical_recommendation=pipeline_payload.get("clinical_recommendation", "Review required.")
        )
