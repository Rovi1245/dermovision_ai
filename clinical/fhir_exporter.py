"""
HL7 FHIR R4 DiagnosticReport Exporter
Formats dermoscopy AI findings into standard healthcare interoperability JSON
for seamless EHR and PACS integration.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

class FHIRExporter:
    @classmethod
    def generate_diagnostic_report(cls, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the pipeline output into a valid HL7 FHIR R4 DiagnosticReport resource.
        """
        report_id = f"derm-report-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        
        patient_id = case_data.get("patient_id", "PT-UNKNOWN")
        primary_diag = case_data.get("primary_diagnosis", {})
        diag_name = primary_diag.get("condition_name", "Undifferentiated Skin Lesion")
        snomed_code = primary_diag.get("snomed_ct", "372244006")
        
        seven_pt = case_data.get("seven_point_checklist", {})
        score = seven_pt.get("cumulative_score", 0)
        risk_tier = case_data.get("risk_tier", "EVALUATION COMPLETE")
        
        features = case_data.get("grounded_features", [])
        feature_names = [f.get("label", "") for f in features if f.get("is_enabled", True)]

        fhir_resource = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": {
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/DiagnosticReport"
                ],
                "lastUpdated": now_iso
            },
            "status": "preliminary",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology / Diagnostic Imaging"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "72216-5",
                        "display": "Dermoscopy study interpretation"
                    }
                ],
                "text": "AI-Assisted Epiluminescence Dermoscopy Evaluation (Dermovision v2.4)"
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": f"Patient ID {patient_id}"
            },
            "effectiveDateTime": now_iso,
            "issued": now_iso,
            "conclusion": (
                f"Primary Impression: {diag_name} (Confidence: {primary_diag.get('confidence', 0.0)*100:.1f}%). "
                f"Risk Tier: {risk_tier}. Argenziano 7-Point Checklist Cumulative Score: {score}. "
                f"Recommendation: {case_data.get('clinical_recommendation', 'Clinical review required.')}"
            ),
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": snomed_code,
                            "display": diag_name
                        }
                    ]
                }
            ],
            "extension": [
                {
                    "url": "http://clinical-dermoscopy.ai/fhir/StructureDefinition/seven-point-score",
                    "valueInteger": score
                },
                {
                    "url": "http://clinical-dermoscopy.ai/fhir/StructureDefinition/grounded-dermoscopic-features",
                    "valueString": ", ".join(feature_names)
                },
                {
                    "url": "http://clinical-dermoscopy.ai/fhir/StructureDefinition/anatomic-location",
                    "valueString": case_data.get("anatomic_site", "Unspecified Cutaneous Site")
                }
            ]
        }

        return fhir_resource
