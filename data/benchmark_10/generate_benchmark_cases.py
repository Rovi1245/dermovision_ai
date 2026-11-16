"""
Generator for the 10 Clinical Benchmark Dermoscopy Images
Synthesizes dermoscopic microstructures, colors, and specific noise artifacts
(hair fibers, specular glare, air bubbles, surface scale) to test the pipeline.
"""

import os
from pathlib import Path
import json
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent

def create_benchmark_images():
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "r") as f:
        cases = json.load(f)

    for case in cases:
        cid = case["case_id"]
        gt_code = case["ground_truth_code"]
        img_path = OUTPUT_DIR / f"{cid}.png"

        # Canvas: 512x512 Skin Tone background
        # Normal skin in BGR: Light skin ~ (180, 205, 235)
        canvas = np.full((512, 512, 3), (180, 205, 235), dtype=np.uint8)

        if gt_code == "MEL":
            if "LM" in cid:
                # Lentigo Maligna: Tan facial skin, dark follicular rings, moth-eaten border
                canvas = np.full((512, 512, 3), (160, 190, 220), dtype=np.uint8)
                # Mottled pigment
                cv2.ellipse(canvas, (256, 256), (170, 140), -15, 0, 360, (90, 130, 180), -1)
                # Asymmetric follicular openings
                for fx, fy in [(210, 210), (250, 190), (290, 220), (240, 270), (280, 290)]:
                    cv2.circle(canvas, (fx, fy), 14, (30, 50, 90), 3)  # Dark asymmetric ring
                    cv2.circle(canvas, (fx, fy), 6, (170, 190, 220), -1) # Pale center
                # Annular-granular fine gray dots
                for gx, gy in [(220, 230), (260, 240), (250, 220), (270, 260), (230, 260)]:
                    for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -4), (0, 4)]:
                        cv2.circle(canvas, (gx+dx, gy+dy), 1, (100, 100, 100), -1)
            else:
                # Superficial Spreading Melanoma: Asymmetric, dark brown, blue-white veil
                cv2.ellipse(canvas, (256, 256), (175, 135), 35, 0, 360, (40, 65, 115), -1)
                # Atypical network lines
                for i in range(140, 350, 18):
                    cv2.line(canvas, (i, 160), (i + 40, 330), (20, 35, 70), 2)
                # Blue-white veil (slate/cyan tint)
                cv2.circle(canvas, (220, 230), 55, (185, 175, 130), -1)
                # Peripheral pseudopods
                cv2.line(canvas, (350, 310), (410, 350), (20, 35, 70), 3)
                cv2.line(canvas, (360, 290), (420, 310), (20, 35, 70), 3)

        elif gt_code == "AKIEC":
            # Actinic Keratosis: Red strawberry background with pale follicular clods
            cv2.ellipse(canvas, (256, 256), (165, 145), 10, 0, 360, (80, 100, 215), -1) # Erythema
            # Strawberry pattern pale ostia
            for x in range(150, 360, 28):
                for y in range(150, 360, 28):
                    cv2.circle(canvas, (x, y), 7, (160, 185, 230), -1)
            # 4-Dot polarized rosettes (white clovers)
            for rx, ry in [(210, 200), (280, 210), (240, 280)]:
                cv2.circle(canvas, (rx-3, ry-3), 2, (255, 255, 255), -1)
                cv2.circle(canvas, (rx+3, ry-3), 2, (255, 255, 255), -1)
                cv2.circle(canvas, (rx-3, ry+3), 2, (255, 255, 255), -1)
                cv2.circle(canvas, (rx+3, ry+3), 2, (255, 255, 255), -1)
            # Focal crust at 2:30 o'clock
            cv2.rectangle(canvas, (290, 190), (335, 235), (35, 55, 110), -1)

        elif gt_code == "NV":
            if "DYS" in cid:
                # Dysplastic Nevus: Multi-toned, slight asymmetry, peripheral dots
                cv2.ellipse(canvas, (256, 256), (150, 120), 15, 0, 360, (65, 95, 140), -1)
                cv2.ellipse(canvas, (240, 250), (80, 60), 10, 0, 360, (40, 60, 100), -1)
                # Peripheral globules
                for angle in range(0, 360, 40):
                    rad = np.radians(angle)
                    gx = int(256 + 140 * np.cos(rad))
                    gy = int(256 + 110 * np.sin(rad))
                    cv2.circle(canvas, (gx, gy), 4, (30, 45, 80), -1)
            else:
                # Common Nevus: Symmetric uniform light-brown ellipse, regular delicate network
                cv2.ellipse(canvas, (256, 256), (130, 130), 0, 0, 360, (80, 115, 160), -1)
                for i in range(160, 350, 24):
                    cv2.line(canvas, (i, 170), (i + 20, 330), (55, 85, 125), 1)

        elif gt_code == "BCC":
            # Basal Cell Carcinoma: Translucent pink/shiny white stroma with arborizing red telangiectasias
            cv2.ellipse(canvas, (256, 256), (150, 130), 0, 0, 360, (130, 145, 210), -1)
            # Arborizing branched blood vessels
            pts = [(220, 200), (256, 240), (280, 290), (320, 320)]
            for idx in range(len(pts)-1):
                cv2.line(canvas, pts[idx], pts[idx+1], (40, 40, 200), 2)
            cv2.line(canvas, (256, 240), (240, 290), (40, 40, 200), 2)
            cv2.line(canvas, (280, 290), (300, 270), (40, 40, 200), 2)
            if "PIG" in cid:
                # Blue-gray ovoid nests
                cv2.ellipse(canvas, (210, 230), (25, 15), 30, 0, 360, (110, 85, 60), -1)
                cv2.ellipse(canvas, (270, 260), (20, 12), -20, 0, 360, (110, 85, 60), -1)
            else:
                # Ulcer / crust
                cv2.circle(canvas, (260, 230), 16, (40, 50, 120), -1)

        elif gt_code == "BKL":
            # Seborrheic Keratosis: Sharp borders, yellowish-brown verrucous texture, comedo openings
            cv2.ellipse(canvas, (256, 256), (155, 135), 45, 0, 360, (60, 100, 140), -1)
            # Comedo-like crypt openings
            for cx, cy in [(220, 220), (270, 210), (250, 270), (290, 280), (230, 280)]:
                cv2.circle(canvas, (cx, cy), 5, (25, 45, 70), -1)
            # Milia-like cysts (small white dots)
            for mx, my in [(200, 250), (240, 230), (280, 240), (260, 300)]:
                cv2.circle(canvas, (mx, my), 4, (230, 245, 255), -1)

        elif gt_code == "DF":
            # Dermatofibroma: Central white scar patch with faint peripheral network
            cv2.ellipse(canvas, (256, 256), (135, 135), 0, 0, 360, (100, 130, 175), -1)
            # Central white patch
            cv2.circle(canvas, (256, 256), 40, (235, 245, 250), -1)

        elif gt_code == "VASC":
            # Vascular Lesion: Red/purple discrete lacunae
            cv2.ellipse(canvas, (256, 256), (120, 110), 0, 0, 360, (150, 180, 220), -1)
            for lx, ly, lr in [(230, 230, 24), (280, 235, 20), (245, 275, 26), (290, 270, 18)]:
                cv2.circle(canvas, (lx, ly), lr, (50, 40, 180), -1) # Dark red-purple lacunae

        # -------------------------------------------------------------
        # Inject Specific Clinical Noise & Artifacts (Hair, Glare, Bubbles)
        # -------------------------------------------------------------
        noise = case["noise_type"]
        if "hair" in noise.lower():
            # Dark hair fibers across the lesion
            cv2.line(canvas, (60, 120), (450, 390), (15, 15, 15), 2)
            if "dense" in noise.lower():
                cv2.line(canvas, (110, 420), (410, 90), (25, 25, 25), 2)
                cv2.line(canvas, (180, 80), (380, 440), (20, 20, 20), 2)
        
        if "bubble" in noise.lower():
            # Air bubble from immersion fluid: white ring with dark border
            cv2.circle(canvas, (390, 380), 22, (240, 240, 240), -1)
            cv2.circle(canvas, (390, 380), 22, (40, 40, 40), 2)

        if "glare" in noise.lower() or "reflection" in noise.lower():
            # Specular flash reflection (saturated white spot)
            cv2.ellipse(canvas, (270, 220), (14, 8), 45, 0, 360, (255, 255, 255), -1)

        # Save to PNG
        cv2.imwrite(str(img_path), canvas)
        print(f"Generated benchmark image: {img_path.name}")

if __name__ == "__main__":
    create_benchmark_images()
