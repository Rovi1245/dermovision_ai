# Dermovision AI — Clinical Dermoscopy Assistant (v2.4)
### Explainable Second-Reader & Grounded Dermoscopic Feature Localization Platform
*CDSCO SaMD Class B & HL7/FHIR Compliant Architecture*

---

## 1. Project Overview

Dermovision AI is an explainable clinical decision-support platform designed for dermatologists. Departing from uncalibrated black-box classification, this system implements a **Modular Feature-Grounded Architecture**:
1. **Preprocesses & Cleans:** Inpaints hair fibers, ruler markings, and specular glare using the DullRazor morphological filter.
2. **Localizes Microfeatures:** Predicts normalized bounding boxes `[ymin, xmin, ymax, xmax]` for specific dermoscopic patterns (Atypical Pigment Network, Blue-White Veil, Polymorphous Vessels, Rosettes, Strawberry Pattern, Crust, Streaks, Globules).
3. **Calculates Clinical Scores:** Computes the **Argenziano 7-Point Checklist** score and **Stolz Total Dermoscopy Score (ABCD)** mathematically.
4. **Enables Live Doctor Intervention:** Clinicians can toggle individual feature overlays on/off in the interactive UI, dynamically recalculating diagnostic risk in real time.
5. **Exports Interoperable Records:** Generates HL7 FHIR R4 `DiagnosticReport` JSON and auditable medical narratives with zero hallucination.

---

## 2. Directory Structure

```
/Users/rohanbr/Desktop/helios/
├── venv/                              # Virtual environment (Python 3.12)
├── app_streamlit.py                   # Detailed Clinician & Developer Streamlit Workbench
├── api/                               # FastAPI Web Server & Endpoints
│   ├── main.py                        # Server entrypoint & static mount
│   ├── routes.py                      # REST endpoints (/analyze, /recalculate, /samples)
│   └── schemas.py                     # Pydantic v2 validation models
├── core/                              # Master Pipeline & Config
│   ├── config.py                      # Global settings & thresholds
│   └── pipeline.py                    # 3-Engine pipeline orchestrator
├── preprocessing/                     # Stage 1: Quality Gate & Artifact Inpainting
│   ├── dull_razor.py                  # Morphological hair & ink removal
│   └── quality_gate.py                # Laplacian variance blur & glare checks
├── models/                            # Stage 2: Multi-Task Vision Specialists
│   ├── feature_detector.py            # Feature Grounding & Bounding Box Engine (YOLOv11)
│   ├── disease_classifier.py          # 7-Class HAM10000 Differential Classifier (ViT)
│   └── weights/                       # Model Weights Subfolder
│       └── ham10000_vit/              # Fine-tuned HAM10000 Vision Transformer (327 MB)
│           ├── config.json
│           ├── model.safetensors      # Real trained weights on 10,000 skin lesions
│           └── preprocessor_config.json
├── clinical/                          # Stage 3A: Clinical Logic & Interoperability
│   ├── seven_point_checklist.py       # Argenziano 7-Point scoring engine
│   ├── abcd_rule.py                   # Stolz TDS scoring engine
│   └── fhir_exporter.py               # HL7 FHIR R4 DiagnosticReport generator
├── narrative/                         # Stage 3B: Structured Narrative Generation
│   └── report_generator.py            # Zero-hallucination clinical report engine
├── data/                              # Data Assets, Taxonomy & Benchmark Loaders
│   ├── samples/                       # Benchmark cases (ISIC_0024468 & PT-94812)
│   ├── taxonomy.py                    # ICD-11, SNOMED-CT & 7-point criteria mappings
│   └── isic_loader.py                 # ISIC Archive API utilities
├── ui/                                # Interactive Clinician Portal
│   └── index.html                     # High-fidelity Dermovision web viewer
├── tests/                             # Unit & Integration Test Suite
│   └── test_pipeline.py               # End-to-end pipeline validation
├── requirements.txt                   # Production dependencies
├── pytest.ini                         # Test runner configuration
├── Dockerfile                         # Production containerization
└── README.md                          # Documentation
```

---

## 3. Quick Start & Setup

### A. Clone & Setup Virtual Environment
```bash
git clone <your-repo-url>
cd helios

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### B. Run the Developer & Clinical Streamlit Workbench
Launch the visual inspection workbench:
```bash
streamlit run app_streamlit.py --server.port 8501
```
* Open **`http://localhost:8501`** in your browser.
* You will see side-by-side tabs for:
  - 🔬 **Visual Inspection:** Raw image vs. DullRazor cleaned tissue vs. Grounded bounding box overlays.
  - 📊 **Diagnostic Differential:** HAM10000 7-class probability bars.
  - ⚖️ **Interactive 7-Point Checklist:** Checkbox toggles that dynamically recalculate the score live.
  - 📄 **Clinical Narrative & FHIR:** Full text report & downloadable FHIR R4 JSON.

### C. Run the Fast & Lightweight Web UI (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```
* **Interactive Web Portal:** `http://localhost:8001/ui`
* **Swagger API Docs:** `http://localhost:8001/docs`

### D. Run the Test Suite
Verify that all 5 pipeline and clinical scoring tests pass:
```bash
pytest tests/test_pipeline.py -v
```
