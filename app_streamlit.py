"""
Dermovision AI - Clinical Dermoscopy Developer & Clinician Workbench (Streamlit)
Interactive visual inspection suite for debugging preprocessing, feature localization,
7-point checklist calculation, and FHIR interoperability.
"""

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
import streamlit as st
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

import importlib
import preprocessing.quality_gate
import core.pipeline
importlib.reload(preprocessing.quality_gate)
importlib.reload(core.pipeline)
from core.pipeline import DermoscopyPipeline
from preprocessing.quality_gate import QualityGate
from core.config import settings
from clinical.seven_point_checklist import SevenPointChecklistEngine
from clinical.abcd_rule import ABCDRuleEngine
from clinical.fhir_exporter import FHIRExporter

# Page Configuration
st.set_page_config(
    page_title="Dermovision AI - Clinical Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - High-contrast Dark/Light Theme Support
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #818CF8;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 13px;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 14px;
        color: #F8FAFC !important;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card b {
        color: #FFFFFF !important;
        font-size: 13px;
    }
    .metric-card small {
        color: #94A3B8 !important;
        font-size: 12px;
        display: block;
        margin-top: 3px;
    }
    .metric-card .box-coords {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #38BDF8 !important;
        background: #0F172A;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return DermoscopyPipeline()

pipeline = get_pipeline()

# Load preset cases
samples_manifest = settings.SAMPLES_DIR / "sample_cases.json"
benchmark_manifest = Path("data/benchmark_10/manifest.json")

preset_cases = {}

# 1. Load 10 Benchmark cases if available
if benchmark_manifest.exists():
    try:
        with open(benchmark_manifest, "r") as f:
            bm_data = json.load(f)
            for item in bm_data:
                cid = item["case_id"]
                preset_cases[cid] = {
                    "case_id": cid,
                    "isic_id": item.get("isic_id", ""),
                    "patient_id": item.get("patient_id", cid),
                    "patient_demographics": item.get("patient_demographics", "Adult"),
                    "anatomic_site": item.get("anatomic_site", "Skin"),
                    "ground_truth_diagnosis_code": item.get("ground_truth_code"),
                    "ground_truth_diagnosis_name": item.get("ground_truth_name"),
                    "ground_truth_criteria": item.get("ground_truth_criteria", []),
                    "description": item.get("description", ""),
                    "noise_type": item.get("noise_type", "")
                }
    except Exception as e:
        print(f"Error loading benchmark manifest: {e}")

# 2. Load original sample cases as fallback/supplement
if samples_manifest.exists() and not preset_cases:
    with open(samples_manifest, "r") as f:
        preset_cases = json.load(f)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🔬 Clinical Controls")
st.sidebar.markdown("Configure thresholds and inspect pipeline stages.")

input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Preset Atlas Cases", "Upload New Lesion Photo"],
    index=0
)

uploaded_file = None
selected_case_id = None
lesion_meta = None

if input_mode == "Preset Atlas Cases":
    case_options = list(preset_cases.keys())
    selected_case_id = st.sidebar.selectbox(
        "Choose Atlas Benchmark Case",
        case_options,
        format_func=lambda x: f"{x} - {preset_cases[x]['ground_truth_diagnosis_name'].split('(')[0]}"
    )
    lesion_meta = preset_cases[selected_case_id]
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Epiluminescence Image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Preprocessing Hyperparameters")
hair_kernel = st.sidebar.slider("DullRazor Kernel Size", min_value=5, max_value=17, value=9, step=2)
blur_threshold = st.sidebar.slider("Laplacian Blur Threshold", min_value=30.0, max_value=200.0, value=80.0, step=10.0)
glare_threshold = st.sidebar.slider("Specular Glare Threshold (%)", min_value=15.0, max_value=60.0, value=35.0, step=5.0)
conf_threshold = st.sidebar.slider("Feature Confidence Filter", min_value=0.30, max_value=0.90, value=0.50, step=0.05)
bypass_quality = st.sidebar.checkbox("⚠️ Bypass Quality Gate (Force Analysis)", value=False, help="Enable to analyze images that have heavy flash glare or slight blur.")

# -------------------------------------------------------------
# IMAGE ACQUISITION & PREPROCESSING
# -------------------------------------------------------------
def generate_sample_image(case_id: str) -> np.ndarray:
    """Generates synthetic high-resolution dermoscopy canvas with realistic microstructures."""
    img = np.full((512, 512, 3), (180, 200, 230), dtype=np.uint8)  # Skin tone
    
    if case_id == "PT-94812":
        # Melanoma: Asymmetric, dark brown, multi-colored with veil
        cv2.ellipse(img, (256, 256), (170, 130), 30, 0, 360, (50, 70, 120), -1)
        # Network lines
        for i in range(140, 340, 20):
            cv2.line(img, (i, 160), (i + 35, 330), (25, 40, 75), 2)
        # Blue-white veil
        cv2.circle(img, (220, 230), 55, (175, 165, 130), -1)
        # Dark hair across
        cv2.line(img, (80, 100), (420, 420), (20, 20, 20), 2)
    else:
        # Actinic Keratosis / IEC: Reddish background, rosettes, crust
        cv2.ellipse(img, (256, 256), (160, 140), 10, 0, 360, (90, 110, 210), -1)  # Erythema
        # Strawberry pattern pale ostia
        for x in range(160, 350, 30):
            for y in range(160, 350, 30):
                cv2.circle(img, (x, y), 8, (160, 180, 225), -1)
        # Crust
        cv2.rectangle(img, (280, 200), (330, 250), (40, 60, 110), -1)
        # Hair
        cv2.line(img, (120, 380), (380, 120), (30, 30, 30), 2)

    return img

if input_mode == "Preset Atlas Cases" and selected_case_id:
    bm_file = Path("data/benchmark_10") / f"{selected_case_id}.png"
    if bm_file.exists():
        image_bgr = cv2.imread(str(bm_file))
    else:
        image_bgr = generate_sample_image(selected_case_id)
    case_name = selected_case_id
    patient_info = f"{lesion_meta['patient_id']} ({lesion_meta['patient_demographics']})"
    anatomic_site = lesion_meta["anatomic_site"]
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    case_name = "UPLOADED_CASE"
    patient_info = "PT-CLINIC-PATIENT (Adult)"
    anatomic_site = "Cutaneous Lesion"
else:
    st.info("👆 Select a preset atlas case or upload a dermoscopic image in the sidebar to begin.")
    st.stop()

# Execute Pipeline
pipeline.dull_razor.kernel_size = hair_kernel
pipeline.quality_gate = QualityGate(
    min_blur_variance=0.0 if bypass_quality else blur_threshold,
    max_glare_ratio=1.0 if bypass_quality else (glare_threshold / 100.0)
)

try:
    result = pipeline.process_case(
        image_bgr=image_bgr,
        case_id=case_name,
        patient_id=patient_info.split()[0],
        anatomic_site=anatomic_site,
        patient_demographics=patient_info,
        lesion_meta=lesion_meta,
        bypass_quality_gate=bypass_quality
    )
except TypeError:
    st.cache_resource.clear()
    pipeline = DermoscopyPipeline()
    result = pipeline.process_case(
        image_bgr=image_bgr,
        case_id=case_name,
        patient_id=patient_info.split()[0],
        anatomic_site=anatomic_site,
        patient_demographics=patient_info,
        lesion_meta=lesion_meta,
        bypass_quality_gate=bypass_quality
    )

if result.get("status") == "REJECTED_BY_QUALITY_GATE":
    st.error(f"❌ Image Rejected by Quality Gate: {result.get('error')}")
    st.info("💡 **Tip:** You can increase the **Specular Glare Threshold** slider in the sidebar (e.g. to 35% or 40%), or check **'⚠️ Bypass Quality Gate (Force Analysis)'** in the sidebar to process this image anyway.")
    st.stop()

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">Dermovision AI — Clinical Workbench</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Case: <b>{result["case_id"]}</b> | Patient: <b>{result["patient_id"]}</b> | Site: <b>{result["anatomic_site"]}</b> | Inference: <b>{result["inference_time_ms"]} ms</b></div>', unsafe_allow_html=True)
with col_h2:
    if "HIGH RISK" in result["risk_tier"]:
        st.error(f"🚨 {result['risk_tier']}")
    elif "INTERMEDIATE" in result["risk_tier"] or "MODERATE" in result["risk_tier"]:
        st.warning(f"⚠️ {result['risk_tier']}")
    else:
        st.success(f"✅ {result['risk_tier']}")

# -------------------------------------------------------------
# TABS WORKSPACE
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Visual Inspection & Grounding",
    "📊 Diagnostic Differential",
    "⚖️ Interactive 7-Point & ABCD Rules",
    "📄 Clinical Report & FHIR"
])

# =============================================================
# TAB 1: VISUAL GROUNDING & PREPROCESSING
# =============================================================
with tab1:
    st.markdown("### Preprocessing & Spatial Microfeature Grounding")
    
    col_img1, col_img2, col_img3 = st.columns(3)

    # 1. Raw Image
    with col_img1:
        st.subheader("1. Raw Epiluminescence Image")
        raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(raw_rgb, width="stretch")
        st.caption(f"Blur Variance: {result['quality_gate']['laplacian_variance']} | Glare: {result['quality_gate']['glare_percentage']}%")

    # 2. DullRazor Cleaned
    with col_img2:
        st.subheader("2. DullRazor Inpainted Tissue")
        cleaned_bgr, hair_mask, ratio = pipeline.dull_razor.process(image_bgr)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        st.image(cleaned_rgb, width="stretch")
        st.caption(f"Artifacts Removed: {ratio*100:.2f}% surface area | Method: Telea PDE")

    # 3. Grounded Feature Overlay
    with col_img3:
        st.subheader("3. Grounded Microfeatures")
        annotated_bgr = cleaned_bgr.copy()
        h, w = annotated_bgr.shape[:2]
        
        for feat in result["grounded_features"]:
            if not feat.get("is_enabled", True):
                continue
            ymin, xmin, ymax, xmax = feat["box_2d"]
            pt1 = (int(xmin * w), int(ymin * h))
            pt2 = (int(xmax * w), int(ymax * h))
            
            # Parse hex color
            hex_c = feat["color_hex"].lstrip("#")
            bgr_c = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))
            
            # Draw boundary rectangle
            cv2.rectangle(annotated_bgr, pt1, pt2, bgr_c, 2)
            
            # Filled label banner background for high readability
            label_text = f"{feat['label']} ({int(feat['confidence']*100)}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            banner_y = max(pt1[1], text_h + 8)
            cv2.rectangle(
                annotated_bgr,
                (pt1[0], banner_y - text_h - 6),
                (pt1[0] + text_w + 6, banner_y + 2),
                bgr_c,
                -1
            )
            # White text on colored banner
            cv2.putText(
                annotated_bgr,
                label_text,
                (pt1[0] + 3, banner_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, width="stretch")
        st.caption(f"Localizations detected: {len(result['grounded_features'])} patterns")

    st.markdown("#### Detected Microfeatures Breakdown")
    feat_cols = st.columns(min(4, max(1, len(result["grounded_features"]))))
    for i, feat in enumerate(result["grounded_features"]):
        with feat_cols[i % len(feat_cols)]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 12px; height: 12px; border-radius: 3px; background-color: {feat['color_hex']};"></span>
                    <b style="color: #FFFFFF;">{feat['label']}</b>
                </div>
                <small>Confidence: <b style="color: #38BDF8;">{feat['confidence']*100:.1f}%</b></small>
                <small>Tier: <span style="color: #E2E8F0;">{feat['tier']}</span></small>
                <small style="margin-top: 6px;">Box [Ymin, Xmin, Ymax, Xmax]:<br><span class="box-coords">{feat['box_2d']}</span></small>
            </div>
            """, unsafe_allow_html=True)

# =============================================================
# TAB 2: DIAGNOSTIC DIFFERENTIAL
# =============================================================
with tab2:
    st.markdown("### Multi-Class Disease Differential (HAM10000 / ISIC Benchmark)")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.markdown("#### Primary Impression")
        pri = result["primary_diagnosis"]
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #334155; padding: 18px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);">
            <h3 style="color: #818CF8; margin: 0; font-size: 18px;">{pri['condition_name']}</h3>
            <p style="color: #38BDF8; font-size: 24px; font-weight: 800; margin: 6px 0;">{pri['confidence_percent']}% Confidence</p>
            <p style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;"><b>ICD-11:</b> <span style="color: #E2E8F0;">{pri['icd11']}</span> | <b>SNOMED-CT:</b> <span style="color: #E2E8F0;">{pri['snomed_ct']}</span></p>
            <p style="font-size: 12px; color: {'#F87171' if pri['is_malignant'] else '#4ADE80'}; margin-top: 6px;"><b>Pathology:</b> {'Malignant Neoplasm (High Risk)' if pri['is_malignant'] else 'Benign Cutaneous Lesion'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Clinical Action (Rx)")
        st.info(result["clinical_recommendation"])

    with col_d2:
        st.markdown("#### Probability Distribution Across 7 ISIC Classes")
        diff_data = result["differentials"]
        
        for d in diff_data:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{d['condition_name']}** ({d['condition_code']})")
                st.progress(float(d["probability"]) / 100.0)
            with c2:
                st.write(f"**{d['probability']}%**")

# =============================================================
# TAB 3: INTERACTIVE 7-POINT & ABCD RULES
# =============================================================
with tab3:
    st.markdown("### Interactive Doctor Decision Engine (Human-in-the-Loop)")
    st.caption("Toggle any detected feature on/off below to see the Argenziano 7-Point score and risk tier dynamically update in real time!")
    
    col_r1, col_r2 = st.columns([2, 1])
    
    with col_r1:
        st.subheader("Detected Clinical Criteria Toggles")
        active_features_for_eval = []
        
        for i, feat in enumerate(result["grounded_features"]):
            is_active = st.checkbox(
                f"{feat['label']} ({feat['tier']}) - Conf: {feat['confidence']*100:.1f}%",
                value=True,
                key=f"feat_toggle_{i}"
            )
            feat_copy = dict(feat)
            feat_copy["is_enabled"] = is_active
            active_features_for_eval.append(feat_copy)
            
        # Dynamically Recalculate
        live_seven_pt = SevenPointChecklistEngine.evaluate(active_features_for_eval)

    with col_r2:
        st.subheader("Live Recalculated Score")
        st.metric(
            label="Cumulative 7-Point Score",
            value=f"{live_seven_pt['cumulative_score']} Points",
            delta="Biopsy Threshold >= 3" if live_seven_pt['threshold_exceeded'] else "Below Biopsy Threshold",
            delta_color="inverse" if live_seven_pt['threshold_exceeded'] else "normal"
        )
        
        st.markdown(f"**Status:** `{live_seven_pt['risk_tier']}`")
        st.markdown("---")
        st.subheader("Stolz ABCD Rule Score")
        abcd = result["abcd_rule"]
        st.metric("Total Dermoscopy Score (TDS)", value=f"{abcd['total_dermoscopy_score']}")
        st.caption(f"Interpretation: {abcd['interpretation']}")

# =============================================================
# TAB 4: CLINICAL REPORT & FHIR EXPORT
# =============================================================
with tab4:
    st.markdown("### Auditable Narrative & Standardized Interoperability")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.subheader("Structured Clinical Narrative Report")
        st.text_area("Physician Narrative", value=result["clinical_narrative_report"], height=380)
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=result["clinical_narrative_report"],
            file_name=f"Dermovision_Report_{result['case_id']}.txt",
            mime="text/plain"
        )

    with col_rep2:
        st.subheader("HL7 FHIR R4 DiagnosticReport JSON")
        fhir_str = json.dumps(result["fhir_diagnostic_report"], indent=2)
        st.text_area("FHIR JSON Payload", value=fhir_str, height=380)
        st.download_button(
            label="📥 Download FHIR JSON",
            data=fhir_str,
            file_name=f"FHIR_{result['case_id']}.json",
            mime="application/json"
        )
