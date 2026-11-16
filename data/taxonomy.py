"""
Taxonomy and Clinical Standards for Dermoscopy AI
Maps dermoscopic criteria, ISIC classes, ICD-11, and SNOMED CT codes.
"""

from typing import Dict, Any, List

# 7-Class Disease Classification (HAM10000 / ISIC 2018 Task 3 Standard)
DISEASE_CLASSES: Dict[str, Dict[str, Any]] = {
    "MEL": {
        "name": "Malignant Melanoma",
        "icd11": "2C30",
        "snomed_ct": "372244006",
        "is_malignant": True,
        "default_risk_tier": "HIGH RISK (Malignancy Suspicion)",
        "default_recommendation": "URGENT EXCISIONAL BIOPSY: Complete elliptical excision with 2-3 mm narrow clinical safety margin for histopathologic staging."
    },
    "NV": {
        "name": "Melanocytic Nevus",
        "icd11": "2F20",
        "snomed_ct": "2092003",
        "is_malignant": False,
        "default_risk_tier": "LOW RISK (Benign)",
        "default_recommendation": "Benign melanocytic proliferation. Routine annual clinical skin examination."
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "icd11": "2C32",
        "snomed_ct": "275265005",
        "is_malignant": True,
        "default_risk_tier": "MODERATE/HIGH RISK (Non-Melanoma Skin Cancer)",
        "default_recommendation": "Dermatologic consultation for surgical excision, Mohs micrographic surgery, or topical therapy based on subtype."
    },
    "AKIEC": {
        "name": "Actinic Keratosis / Bowen's Disease",
        "icd11": "EK90.0",
        "snomed_ct": "201101007",
        "is_malignant": True,  # Pre-cancerous / intraepidermal carcinoma
        "default_risk_tier": "INTERMEDIATE RISK (Pre-Malignant / In-Situ)",
        "default_recommendation": "Biopsy to exclude invasive squamous cell carcinoma, followed by cryotherapy, 5-fluorouracil, or photodynamic therapy."
    },
    "BKL": {
        "name": "Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)",
        "icd11": "2F22",
        "snomed_ct": "449830006",
        "is_malignant": False,
        "default_risk_tier": "LOW RISK (Benign)",
        "default_recommendation": "Benign epithelial neoplasm. No intervention required unless symptomatic or cosmetically desired."
    },
    "DF": {
        "name": "Dermatofibroma",
        "icd11": "2F32",
        "snomed_ct": "254701007",
        "is_malignant": False,
        "default_risk_tier": "LOW RISK (Benign)",
        "default_recommendation": "Benign fibrohistiocytic lesion exhibiting central dimple sign. Conservative observation."
    },
    "VASC": {
        "name": "Vascular Lesion (Hemangioma / Angioma)",
        "icd11": "2E81",
        "snomed_ct": "400210000",
        "is_malignant": False,
        "default_risk_tier": "LOW RISK (Benign)",
        "default_recommendation": "Benign vascular proliferation composed of lacunae/red clods. Reassurance."
    }
}

# Dermoscopic Criteria & Patterns (ISIC Task 2 + 7-Point Checklist + Facial Patterns)
DERMOSCOPIC_CRITERIA: Dict[str, Dict[str, Any]] = {
    # 7-Point Checklist Major Criteria (2 Points each)
    "atypical_pigment_network": {
        "display_name": "Atypical Pigment Network",
        "points": 2,
        "tier": "Major",
        "color_hex": "#F59E0B",  # Amber
        "description": "Black, brown, or gray network with irregular, thickened meshes and asymmetric distribution."
    },
    "blue_white_veil": {
        "display_name": "Blue-White Veil",
        "points": 2,
        "tier": "Major",
        "color_hex": "#06B6D4",  # Cyan
        "description": "Confluent, structureless blue pigmentation with an overlying white ground-glass appearance."
    },
    "atypical_vascular_pattern": {
        "display_name": "Atypical Vascular / Polymorphous Vessels",
        "points": 2,
        "tier": "Major",
        "color_hex": "#EF4444",  # Red
        "description": "Linear-irregular, corkscrew, or polymorphous vessels within the lesion interior."
    },
    # 7-Point Checklist Minor Criteria (1 Point each)
    "irregular_streaks": {
        "display_name": "Peripheral Streaks / Pseudopods",
        "points": 1,
        "tier": "Minor",
        "color_hex": "#8B5CF6",  # Purple
        "description": "Linear or bulbous projections irregularly distributed at the lesion margin."
    },
    "irregular_dots_globules": {
        "display_name": "Irregular Dots & Globules",
        "points": 1,
        "tier": "Minor",
        "color_hex": "#EC4899",  # Pink
        "description": "Black, brown, or slate-gray round structures of variable size irregularly distributed."
    },
    "irregular_pigmentation": {
        "display_name": "Irregular Blotches / Structureless Areas",
        "points": 1,
        "tier": "Minor",
        "color_hex": "#D97706",  # Dark Amber
        "description": "Hyperpigmented structureless areas with irregular margins and asymmetric position."
    },
    "regression_structures": {
        "display_name": "Regression Structures (Peppering / White Scar-like)",
        "points": 1,
        "tier": "Minor",
        "color_hex": "#64748B",  # Slate
        "description": "Fine slate-gray peppering (melanophages) or white structureless depigmentation."
    },
    # Specialized Actinic & Facial Patterns (from ISIC_0024468 & Textbook cases)
    "rosettes": {
        "display_name": "Polarized 4-Dot Rosettes",
        "points": 1,
        "tier": "Minor (Optical)",
        "color_hex": "#3B82F6",  # Blue
        "description": "Four shiny white dots arranged like a four-leaf clover under polarized dermoscopy, indicating actinic damage."
    },
    "strawberry_pattern": {
        "display_name": "Strawberry Pattern (Erythematous Mesh)",
        "points": 1,
        "tier": "Minor (Actinic)",
        "color_hex": "#10B981",  # Emerald Green
        "description": "Red pseudo-network formed by erythema surrounding prominent hair follicle openings with keratotic plugs."
    },
    "focal_crust": {
        "display_name": "Focal Crust / Micro-erosion",
        "points": 1,
        "tier": "Minor (Erosive)",
        "color_hex": "#DC2626",  # Crimson
        "description": "Superficial epidermal defect with yellowish or blood-tinged crusting."
    },
    "asymmetrical_follicular_pigmentation": {
        "display_name": "Asymmetrical Follicular Pigmentation",
        "points": 2,
        "tier": "Major (Facial Lentigo Maligna)",
        "color_hex": "#EA580C",  # Orange
        "description": "Asymmetric darkening and pigmentation within and around follicular infundibula on facial skin."
    },
    "annular_granular_structures": {
        "display_name": "Annular-Granular Structures",
        "points": 1,
        "tier": "Minor (Facial Lentigo Maligna)",
        "color_hex": "#0284C7",  # Sky Blue
        "description": "Multiple gray dots forming annular circles around follicular ostia."
    },
    "moth_eaten_border": {
        "display_name": "Moth-Eaten Border",
        "points": 1,
        "tier": "Minor (Solar Lentigo / AK)",
        "color_hex": "#B91C1C",  # Dark Red
        "description": "Concave, sharp, scalloped borders characteristic of lentigo and actinic keratosis."
    }
}
