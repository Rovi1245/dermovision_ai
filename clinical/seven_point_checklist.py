"""
Argenziano 7-Point Checklist Scoring Engine
Clinical Decision Support logic based on standardized dermoscopy guidelines:
- Argenziano et al. (2000) "Seven-point checklist of dermoscopy: an award-winning computer-aided diagnosis system"
Major criteria = 2 points each
Minor criteria = 1 point each
Threshold: Score >= 3 points -> High Malignancy Suspicion (Biopsy recommendation)
"""

from typing import List, Dict, Any
from data.taxonomy import DERMOSCOPIC_CRITERIA

class SevenPointChecklistEngine:
    @classmethod
    def evaluate(cls, detected_features: List[Dict[str, Any]], confidence_threshold: float = 0.50) -> Dict[str, Any]:
        """
        Calculates the 7-Point Checklist score from verified visual detections.
        Allows interactive recalculation when features are toggled on/off.
        """
        score = 0
        major_criteria_met = []
        minor_criteria_met = []
        breakdown = []

        # Deduplicate criteria names that passed confidence
        active_criteria = {}
        for feat in detected_features:
            cid = feat.get("criterion_id", "").lower().replace("-", "_")
            conf = feat.get("confidence", 0.0)
            is_enabled = feat.get("is_enabled", True)
            
            if is_enabled and conf >= confidence_threshold:
                if cid not in active_criteria or conf > active_criteria[cid]["confidence"]:
                    active_criteria[cid] = feat

        for cid, feat in active_criteria.items():
            meta = DERMOSCOPIC_CRITERIA.get(cid)
            if not meta:
                continue

            pts = meta["points"]
            tier = meta["tier"]
            score += pts
            
            entry = {
                "criterion_id": cid,
                "display_name": meta["display_name"],
                "tier": tier,
                "points": pts,
                "confidence": round(feat.get("confidence", 1.0), 3)
            }
            breakdown.append(entry)

            if "Major" in tier:
                major_criteria_met.append(meta["display_name"])
            else:
                minor_criteria_met.append(meta["display_name"])

        is_high_risk = score >= 3
        
        if score >= 3:
            risk_tier = "HIGH RISK (Malignancy Suspicion)"
            recommendation = (
                "URGENT EXCISIONAL BIOPSY: Complete elliptical excision with a 2 to 3 mm "
                "narrow clinical safety margin for formal histopathologic staging. Avoid shave or punch biopsy."
            )
        elif score == 2:
            risk_tier = "MODERATE RISK (Equivocal / Atypical)"
            recommendation = (
                "SHORT-TERM DIGITAL DERMOSCOPY MONITORING (3 Months) or conservative biopsy "
                "if patient has personal/family history of melanoma or multiple dysplastic nevi."
            )
        else:
            risk_tier = "LOW RISK (Benign / Routine)"
            recommendation = (
                "Benign dermoscopic pattern. Routine clinical follow-up and self-examination reassurance."
            )

        return {
            "cumulative_score": score,
            "threshold_exceeded": is_high_risk,
            "risk_tier": risk_tier,
            "clinical_recommendation": recommendation,
            "major_criteria_met": major_criteria_met,
            "minor_criteria_met": minor_criteria_met,
            "breakdown": breakdown
        }
