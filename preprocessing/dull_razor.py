"""
DullRazor Digital Hair & Artifact Removal Engine
Implementation of morphological filtering for clinical dermoscopy preprocessing.
Removes dark hair fibers, ruler markings, and surface debris without altering
the underlying lesion morphology or cellular pigmentation patterns.
"""

import cv2
import numpy as np
from typing import Tuple

class DullRazor:
    def __init__(self, kernel_size: int = 9, threshold_val: int = 10, inpaint_radius: int = 4):
        self.kernel_size = kernel_size
        self.threshold_val = threshold_val
        self.inpaint_radius = inpaint_radius
        self.structuring_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size)
        )

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Processes dermoscopy image.
        Returns:
            cleaned_bgr: Inpainted tissue image.
            hair_mask: Binary mask of removed artifacts.
            artifact_ratio: Percentage of image area occupied by artifacts.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid image supplied to DullRazor.")

        # 1. Grayscale conversion
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Black-Hat morphological transformation
        # Emphasizes elements that are darker than their immediate surroundings
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.structuring_element)

        # 3. Binary thresholding to isolate hair fibers
        _, hair_mask = cv2.threshold(blackhat, self.threshold_val, 255, cv2.THRESH_BINARY)

        # 4. Inpainting using Telea method
        cleaned_bgr = cv2.inpaint(
            image_bgr, hair_mask, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA
        )

        artifact_ratio = float(np.count_nonzero(hair_mask)) / float(hair_mask.size)

        return cleaned_bgr, hair_mask, artifact_ratio
