"""
ISIC Archive Dataset Loader & Benchmark Downloader
Utilities for downloading, caching, and parsing ISIC 2018 Task 2 & 3 benchmark data.
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

ISIC_API_BASE = "https://api.isic-archive.com/api/v2"

class ISICDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_image_metadata(self, isic_id: str) -> Dict[str, Any]:
        """
        Fetches official ground-truth clinical metadata for any ISIC image
        (e.g., ISIC_0024468).
        """
        url = f"{ISIC_API_BASE}/images/{isic_id}/"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Dermovision-Client/2.4"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data
        except Exception as e:
            return {
                "error": str(e),
                "isic_id": isic_id,
                "note": "Online fetch failed or blocked; use local cached sample_cases.json"
            }

    def print_challenge_structure(self):
        """
        Prints the standard mapping for ISIC 2018 benchmark tasks.
        """
        print("""
========================================================================
ISIC ARCHIVE CHALLENGE DATASET TAXONOMY (https://challenge.isic-archive.com)
========================================================================
1. ISIC 2018 Task 1: Lesion Boundary Segmentation (Pixel masks)
2. ISIC 2018 Task 2: Dermoscopic Feature Extraction (5 Core Attributes):
   - Pigment Network (attribute_pigment_network)
   - Negative Network (attribute_negative_network)
   - Streaks (attribute_streaks)
   - Milia-like Cysts (attribute_milia_like_cyst)
   - Globules and Dots (attribute_globules)
3. ISIC 2018 Task 3: HAM10000 7-Class Disease Classification:
   - MEL: Malignant Melanoma
   - NV: Melanocytic Nevus
   - BCC: Basal Cell Carcinoma
   - AKIEC: Actinic Keratosis / Bowen's Disease
   - BKL: Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)
   - DF: Dermatofibroma
   - VASC: Vascular Lesion
========================================================================
        """)
