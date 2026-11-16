"""
Stolz ABCD Rule Scoring Engine
Implements the dermatological Total Dermoscopy Score (TDS):
- Asymmetry (A): score 0 to 2 (weight 1.3)
- Border (B): score 0 to 8 (weight 0.1)
- Color (C): score 1 to 6 (weight 0.5)
- Differential structures (D): score 1 to 5 (weight 0.5)
Formula: TDS = (A * 1.3) + (B * 0.1) + (C * 0.5) + (D * 0.5)
Thresholds:
- TDS < 4.75: Benign
- 4.75 <= TDS <= 5.45: Suspicious
- TDS > 5.45: Highly Suspicious for Melanoma
"""

from typing import Dict, Any, List

class ABCDRuleEngine:
    @classmethod
    def evaluate(
        cls,
        asymmetry_score: int = 1,       # 0 = symmetric, 1 = 1-axis, 2 = 2-axis
        border_abrupt_segments: int = 4,# 0-8 pie segments showing sharp abrupt cut-off
        color_count: int = 3,           # 1-6 colors: white, red, light brown, dark brown, blue-gray, black
        structures_count: int = 3       # 1-5: network, structureless, dots, globules, streaks
    ) -> Dict[str, Any]:
        asymmetry_clamped = max(0, min(2, asymmetry_score))
        border_clamped = max(0, min(8, border_abrupt_segments))
        color_clamped = max(1, min(6, color_count))
        structures_clamped = max(1, min(5, structures_count))

        tds = (
            (asymmetry_clamped * 1.3) +
            (border_clamped * 0.1) +
            (color_clamped * 0.5) +
            (structures_clamped * 0.5)
        )
        tds = round(tds, 2)

        if tds > 5.45:
            assessment = "Highly Suspicious for Malignant Melanoma"
            risk = "HIGH RISK"
        elif tds >= 4.75:
            assessment = "Suspicious Lesion - Histopathologic Examination Advised"
            risk = "INTERMEDIATE RISK"
        else:
            assessment = "Benign Melanocytic Lesion"
            risk = "LOW RISK"

        return {
            "total_dermoscopy_score": tds,
            "interpretation": assessment,
            "risk_tier": risk,
            "components": {
                "asymmetry": {"score": asymmetry_clamped, "weighted": round(asymmetry_clamped * 1.3, 2)},
                "border": {"score": border_clamped, "weighted": round(border_clamped * 0.1, 2)},
                "color": {"score": color_clamped, "weighted": round(color_clamped * 0.5, 2)},
                "differential_structures": {"score": structures_clamped, "weighted": round(structures_clamped * 0.5, 2)}
            }
        }
