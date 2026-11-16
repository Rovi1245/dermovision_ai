"""
Quality Gate for Dermoscopic Image Validation
Verifies image clarity, focus, resolution, and exposure before deep inference.
Rejects out-of-focus or unreadable images to prevent erroneous diagnostic inferences.
"""

import cv2
import numpy as np
from typing import Dict, Any
from core.config import settings

class QualityGate:
    def __init__(self, min_blur_variance: float = None, min_res: int = None, max_glare_ratio: float = None):
        self.min_blur_variance = min_blur_variance or settings.MIN_LAPLACIAN_VARIANCE
        self.min_res = min_res or settings.MIN_RESOLUTION_PIXELS
        self.max_glare_ratio = max_glare_ratio if max_glare_ratio is not None else getattr(settings, "MAX_GLARE_RATIO", 0.35)

    def evaluate(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates image quality.
        Returns:
            Dict containing status (PASSED/REJECTED), blur variance, exposure levels, and resolution.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "quality_status": "REJECTED",
                "reason": "Empty or corrupted image payload.",
                "passed": False
            }

        height, width = image_bgr.shape[:2]

        # 1. Resolution Check
        if height < self.min_res or width < self.min_res:
            return {
                "quality_status": "REJECTED",
                "reason": f"Resolution ({width}x{height}) below minimum required {self.min_res}x{self.min_res} px.",
                "passed": False,
                "resolution": f"{width}x{height}"
            }

        # 2. Laplacian Blur Check
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = laplacian_var < self.min_blur_variance

        # 3. Specular Glare / Over-Exposure Check
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        glare_mask = (v_channel > 245) & (s_channel < 25)
        glare_ratio = float(np.count_nonzero(glare_mask)) / float(glare_mask.size)

        passed = not is_blurred and (glare_ratio < self.max_glare_ratio)
        status = "PASSED" if passed else "REJECTED"
        rejection_reason = None
        if is_blurred:
            rejection_reason = f"Image is out of focus / blurry (Laplacian variance: {laplacian_var:.1f} < {self.min_blur_variance})."
        elif glare_ratio >= self.max_glare_ratio:
            rejection_reason = f"Excessive specular glare / flash artifact ({glare_ratio*100:.1f}% surface saturated > {self.max_glare_ratio*100:.0f}% limit)."

        return {
            "quality_status": status,
            "passed": passed,
            "rejection_reason": rejection_reason,
            "laplacian_variance": round(laplacian_var, 2),
            "glare_percentage": round(glare_ratio * 100, 2),
            "dimensions": {"width": width, "height": height}
        }
