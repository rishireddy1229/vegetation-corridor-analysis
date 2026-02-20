import streamlit as st
import subprocess
import os
import sys
import cv2
import numpy as np
import base64

# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(
    page_title="Vegetation Corridor Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =================================================
# CUSTOM CSS (MODERN UI)
# =================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

.card {
    background: white;
    border-radius: 20px;
    padding: 1.6rem;
    box-shadow: 0 12px 32px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}

h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a;
}

h2, h3 {
    color: #1e293b;
    font-weight: 600;
}

p {
    color: #64748b;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =================================================
# HERO HEADER
# =================================================
st.markdown("""
<div class="card">
    <h1>🌿 Vegetation Measurement Along Line Corridor</h1>
    <p>
    Vegetation analysis using satellite imagery, grid-based segmentation,
    Random Forest classification, and electrical line detection.
    </p>
</div>
""", unsafe_allow_html=True)

# =================================================
# PATH SETUP
# =================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGE_DIV_NOTEBOOK = os.path.join(PROJECT_ROOT, "notebooks", "image_division.ipynb")
RF_NOTEBOOK = os.path.join(PROJECT_ROOT, "notebooks", "randomforest.ipynb")

IMAGE_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "test_images", "image", "image.jpg"
)

SEGMENTED_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "segmented_frame.jpg")

PYTHON_EXEC = sys.executable

# =================================================
# UTILITY: IMAGE → BASE64
# =================================================
def image_to_base64(img_rgb):
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode()

# =================================================
# ELECTRICAL LINE DETECTION
# =================================================
def houghtransform(subimg):
    edges = cv2.Canny(subimg, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=100,
        maxLineGap=10
    )
    return lines is not None

def process_image(inputframe):
    outputframe = np.zeros_like(inputframe)

    for x in range(inputframe.shape[0]):
        for y in range(inputframe.shape[1]):
            r, g, b = inputframe[x, y]

            if (
                (0 <= r <= 192 and 0 <= g <= 192 and 0 <= b <= 192) or
                (0 <= r <= 120 and 0 <= g <= 120 and 0 <= b <= 120) or
                (44 <= r <= 188 and 45 <= g <= 201 and 14 <= b <= 175)
            ):
                subimg = np.array([[inputframe[x, y]]], dtype=np.uint8)
                if houghtransform(subimg):
                    outputframe[x, y] = (255, 255, 255)
                else:
                    outputframe[x, y] = (0, 0, 0)
            else:
                outputframe[x, y] = inputframe[x, y]

    return outputframe

# =================================================
# FILE UPLOAD
# =================================================
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file:
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)

    with open(IMAGE_PATH, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Image uploaded successfully")

    # -------------------------------------------------
    # RUN IMAGE DIVISION
    # -------------------------------------------------
    st.info("Running image_division.ipynb")

    div_result = subprocess.run(
        [PYTHON_EXEC, "-m", "papermill", IMAGE_DIV_NOTEBOOK, IMAGE_DIV_NOTEBOOK],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if div_result.returncode != 0:
        st.error("❌ Image division failed")
        st.text(div_result.stderr)
        st.stop()

    st.success("✅ Image division completed")

    # -------------------------------------------------
    # RUN RANDOM FOREST
    # -------------------------------------------------
    st.info("Running randomforest.ipynb")

    rf_result = subprocess.run(
        [
        PYTHON_EXEC,
        "-m",
        "papermill",
        RF_NOTEBOOK,
        RF_NOTEBOOK,
        "-p",
        "PROJECT_ROOT",
        PROJECT_ROOT
    ],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True)

    if rf_result.returncode != 0:
        st.error("❌ Random Forest failed")
        st.text(rf_result.stderr)
        st.stop()

    st.success("✅ Random Forest completed")


# Load input image
    input_raw = cv2.imread(IMAGE_PATH)

    if input_raw is None:
        st.error("❌ Failed to load input image.")
        st.stop()

    input_img = cv2.cvtColor(input_raw, cv2.COLOR_BGR2RGB)


# Load segmented image
    seg_raw = cv2.imread(SEGMENTED_OUTPUT)

    print("SEGMENTED_OUTPUT:", SEGMENTED_OUTPUT)
    print("Exists?", os.path.exists(SEGMENTED_OUTPUT))

    if seg_raw is None:
        st.error("❌ segmented_frame.jpg not generated.")
        st.stop()

# ✅ This must be OUTSIDE the if block
    heatmap_img = cv2.cvtColor(seg_raw, cv2.COLOR_BGR2RGB)

# Resize to match input
    h, w, _ = input_img.shape
    heatmap_img = cv2.resize(
    heatmap_img, (w, h), interpolation=cv2.INTER_NEAREST
)

# Electrical line detection
    electric_img = process_image(input_img)

# Convert to base64
    input_b64 = image_to_base64(input_img)
    heatmap_b64 = image_to_base64(heatmap_img)
    electric_b64 = image_to_base64(electric_img)

    # =================================================
    # RESULTS SECTION
    # =================================================
    st.markdown("""
    <div>
        <h2 align="center"> Analysis Results</h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4 align="center"> Input Image</h4>
            <img src="data:image/jpeg;base64,{input_b64}"
                 style="width:100%; border-radius:14px;" />
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4 align="center"> Segmented Image</h4>
            <img src="data:image/jpeg;base64,{heatmap_b64}"
                 style="width:100%; border-radius:14px;" />
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h4 align="center">Electrical Line Detection</h4>
            <img src="data:image/jpeg;base64,{electric_b64}"
                style="width:100%; border-radius:14px;" />
        </div>
        """, unsafe_allow_html=True)
