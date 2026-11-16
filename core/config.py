"""
Application Configuration Settings
Manages global thresholds, paths, model parameters, and runtime settings.
"""

import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "Dermovision AI - Clinical Dermoscopy Assistant"
    VERSION: str = "2.4.0"
    API_PREFIX: str = "/api/v1"
    
    # Preprocessing / Quality Gate Thresholds
    MIN_LAPLACIAN_VARIANCE: float = 80.0  # Blur detection threshold
    MIN_RESOLUTION_PIXELS: int = 256      # Minimum image dimension
    MAX_GLARE_RATIO: float = 0.35         # Maximum specular glare ratio (35%)
    
    # Inference Confidence Thresholds
    FEATURE_CONFIDENCE_THRESHOLD: float = 0.50
    MELANOMA_HIGH_RISK_THRESHOLD: float = 0.70
    
    # 7-Point Checklist Thresholds
    SEVEN_POINT_MALIGNANCY_THRESHOLD: int = 3  # Score >= 3 triggers High Malignancy Suspicion
    
    # Directory paths
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    MODELS_DIR: Path = BASE_DIR / "models"
    WEIGHTS_DIR: Path = BASE_DIR / "models" / "weights"

settings = Settings()
