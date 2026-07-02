
import io
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Real or Fake? Camera vs Screen Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded",
)


MAX_IMAGE_SIZE = 800

LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2

LABEL_REAL = 0
LABEL_SCREEN = 1

FEATURE_NAMES = (
    [
        "fft_mean", "fft_std", "fft_median",
        "fft_high_freq_ratio", "fft_low_freq_ratio",
        "fft_horizontal_energy", "fft_vertical_energy", "fft_diagonal_energy",
        "fft_spectral_entropy", "fft_max",
    ]
    + [f"lbp_{i}" for i in range(LBP_BINS)]
    + [
        "glcm_contrast", "glcm_correlation", "glcm_energy",
        "glcm_homogeneity", "glcm_dissimilarity", "glcm_asm",
    ]
    + ["gradient_mean", "gradient_std", "edge_density"]
    + ["laplacian_variance"]
    + ["brightness_mean", "brightness_std", "rms_contrast"]
)

EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)  # 33

MODEL_CANDIDATE_PATHS = [
    Path("best_model.pkl"),
    Path("models/best_model.pkl"),
    Path("model.pkl"),
]


def inject_theme_css(mode: str) -> None:
    base_css = """
    <style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 760px; }

    .app-header {
        text-align: center;
        padding: 1.4rem 1rem 1.6rem 1rem;
        border-radius: 18px;
        margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    }
    .app-header h1 { color: #fff; font-size: 1.65rem; margin: 0 0 .35rem 0; }
    .app-header p { color: rgba(255,255,255,0.9); margin: 0; font-size: .95rem; }

    .result-card {
        border-radius: 16px;
        padding: 1.4rem 1.4rem;
        margin-top: 1rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .result-card h2 { margin: 0 0 .3rem 0; font-size: 1.5rem; }
    .result-card .sub { opacity: .85; font-size: .92rem; margin-bottom: .9rem; }

    .real-card { background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05)); border-color: rgba(16,185,129,0.35); }
    .real-card h2 { color: #059669; }
    .fake-card { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border-color: rgba(239,68,68,0.35); }
    .fake-card h2 { color: #dc2626; }

    .conf-bar-bg { background: rgba(120,120,120,0.18); border-radius: 999px; height: 14px; width: 100%; overflow: hidden; }
    .conf-bar-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }
    .conf-label { font-size: .85rem; opacity: .75; margin-top: .4rem; }

    .stButton>button {
        border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; border: none; width: 100%;
    }
    .stButton>button:hover { opacity: .92; }

    .feature-note { font-size: .8rem; opacity: .65; text-align: center; margin-top: 1.4rem; }

    [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
        border-radius: 14px;
    }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if mode == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #0f1117; color: #e6e6e6; }
            section[data-testid="stSidebar"] { background-color: #161925; }
            .result-card { color: #e6e6e6; }
            .stMarkdown, p, span, label, .feature-note, .conf-label { color: #e6e6e6 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Light":
        st.markdown(
            """
            <style>
            .stApp { background-color: #fafafa; color: #1a1a1a; }
            section[data-testid="stSidebar"] { background-color: #ffffff; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            @media (prefers-color-scheme: dark) {
                .stApp { background-color: #0f1117; color: #e6e6e6; }
                section[data-testid="stSidebar"] { background-color: #161925; }
                .stMarkdown, p, span, label, .feature-note, .conf-label { color: #e6e6e6 !important; }
            }
            @media (prefers-color-scheme: light) {
                .stApp { background-color: #fafafa; color: #1a1a1a; }
                section[data-testid="stSidebar"] { background-color: #ffffff; }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def resize_if_needed(image: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Returns a BGR numpy array, robust to format quirks."""
    raw_bytes = uploaded_file.getvalue()
    file_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if bgr is None:
        # Fallback via PIL for formats OpenCV struggles with (e.g. HEIC-ish, some PNGs)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr


def load_image(bgr: np.ndarray):
    """Resize and convert BGR to RGB and grayscale."""
    bgr = resize_if_needed(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def radial_profile(data: np.ndarray) -> np.ndarray:
    h, w = data.shape
    center = (w // 2, h // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-8)


def extract_fft_features(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    fft = fftshift(fft2(gray))
    magnitude = np.log1p(np.abs(fft))

    features = [np.mean(magnitude), np.std(magnitude), np.median(magnitude)]

    radial = radial_profile(magnitude)
    n = len(radial)
    low = radial[: n // 3].sum()
    mid = radial[n // 3: 2 * n // 3].sum()
    high = radial[2 * n // 3:].sum()
    total = low + mid + high + 1e-8
    features.append(high / total)
    features.append(low / total)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    horizontal = magnitude[cy - 5:cy + 5, :]
    vertical = magnitude[:, cx - 5:cx + 5]
    diagonal = np.diag(magnitude)
    features.append(np.mean(horizontal))
    features.append(np.mean(vertical))
    features.append(np.mean(diagonal))

    prob = magnitude.flatten()
    prob = prob / (prob.sum() + 1e-8)
    features.append(entropy(prob))

    features.append(np.max(magnitude))

    return np.asarray(features, dtype=np.float32)


def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    quantized = (gray / 32).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    asm = graycoprops(glcm, "ASM").mean()
    return np.array(
        [contrast, correlation, energy, homogeneity, dissimilarity, asm],
        dtype=np.float32,
    )


def extract_edge_features(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    gradient_mean = np.mean(gradient)
    gradient_std = np.std(gradient)

    median = np.median(gray)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = np.count_nonzero(edges) / edges.size

    return np.array([gradient_mean, gradient_std, edge_density], dtype=np.float32)


def extract_sharpness_features(gray: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.array([laplacian.var()], dtype=np.float32)


def extract_brightness_contrast(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    brightness_mean = gray.mean()
    brightness_std = gray.std()
    rms_contrast = brightness_std / (brightness_mean + 1e-8)
    return np.array([brightness_mean, brightness_std, rms_contrast], dtype=np.float32)


def validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features(gray: np.ndarray) -> np.ndarray:
    fft = extract_fft_features(gray)
    lbp = extract_lbp_features(gray)
    glcm = extract_glcm_features(gray)
    edge = extract_edge_features(gray)
    sharpness = extract_sharpness_features(gray)
    brightness = extract_brightness_contrast(gray)

    features = np.concatenate([fft, lbp, glcm, edge, sharpness, brightness])
    features = validate_features(features)

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features but got {len(features)}"
        )

    return features


@st.cache_resource(show_spinner=False)
def load_model_from_path(path_str: str):
    return joblib.load(path_str)


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(file_bytes: bytes):
    return joblib.load(io.BytesIO(file_bytes))


def get_model():
    for candidate in MODEL_CANDIDATE_PATHS:
        if candidate.exists():
            try:
                return load_model_from_path(str(candidate)), str(candidate)
            except Exception as e:
                st.sidebar.error(f"Failed to load {candidate}: {e}")

    st.sidebar.warning("⚠️ No model file found next to app.py.")
    uploaded_model = st.sidebar.file_uploader(
        "Upload your best_model.pkl", type=["pkl"], key="model_uploader"
    )
    if uploaded_model is not None:
        try:
            model = load_model_from_bytes(uploaded_model.getvalue())
            return model, uploaded_model.name
        except Exception as e:
            st.sidebar.error(f"Could not load uploaded model: {e}")
    return None, None


st.sidebar.markdown("### ⚙️ Settings")
theme_choice = st.sidebar.radio("Appearance", ["System", "Light", "Dark"], index=0)
inject_theme_css(theme_choice)

st.sidebar.markdown("### 🧠 Model")
model, model_source = get_model()
if model is not None:
    st.sidebar.success(f"Model loaded: `{model_source}`")
else:
    st.sidebar.info("Place `best_model.pkl` next to app.py, or upload it above.")

with st.sidebar.expander("📱 Run this on your phone"):
    st.markdown(
        """
1. On your computer, run:
   ```
   streamlit run app.py --server.address 0.0.0.0
   ```
2. Find your computer's local IP (e.g. `192.168.1.20`).
3. On your phone (same WiFi), open:
   `http://192.168.1.20:8501`

Or deploy to **Streamlit Community Cloud** for a link that
works from anywhere, including mobile data.
        """
    )

st.markdown(
    """
    <div class="app-header">
        <h1>🕵️ Camera vs Screen Detector</h1>
        <p>Upload or capture a photo — I'll analyze texture, frequency &amp; sharpness
        patterns to tell if it's a genuine camera shot or a recapture of a screen/paper.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Camera"])

image_source = None
with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="uploader"
    )
    if uploaded is not None:
        image_source = uploaded

with tab_camera:
    camera_shot = st.camera_input("Take a photo")
    if camera_shot is not None:
        image_source = camera_shot

if image_source is not None:
    bgr = decode_uploaded_image(image_source)

    if bgr is None:
        st.error("Could not read this image. Please try a different file.")
    else:
        rgb_preview = cv2.cvtColor(resize_if_needed(bgr), cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, caption="Preview", use_container_width=True)

        analyze_clicked = st.button("🔍 Analyze Image", disabled=(model is None))

        if model is None:
            st.info("Load a model (see sidebar) to enable analysis.")

        if analyze_clicked and model is not None:
            with st.spinner("Extracting features and running the model..."):
                try:
                    _, gray = load_image(bgr)
                    features = extract_features(gray)
                    feature_vector = features.reshape(1, -1)

                    prediction = int(model.predict(feature_vector)[0])

                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(feature_vector)[0]
                        confidence = float(proba[prediction]) * 100

                except Exception as e:
                    st.error(f"Something went wrong while analyzing this image: {e}")
                    prediction = None
                    confidence = None

            if prediction is not None:
                is_real = prediction == LABEL_REAL
                card_class = "real-card" if is_real else "fake-card"
                bar_color = "#10b981" if is_real else "#ef4444"
                label_text = "✅ REAL — Camera Capture" if is_real else "⚠️ FAKE — Screen / Paper Recapture"
                sub_text = (
                    "This looks like a genuine photo taken directly with a camera."
                    if is_real
                    else "This looks like a photo of a screen, print, or paper — not an original camera shot."
                )

                conf_html = ""
                if confidence is not None:
                    conf_html = f"""
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div>
                    </div>
                    <div class="conf-label">Confidence: {confidence:.1f}%</div>
                    """

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <h2>{label_text}</h2>
                        <div class="sub">{sub_text}</div>
                        {conf_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🔬 View extracted features"):
                    st.dataframe(
                        {
                            "Feature": FEATURE_NAMES,
                            "Value": [round(float(v), 5) for v in features],
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

else:
    st.markdown(
        '<p class="feature-note">Upload a photo or use your camera above to get started.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="feature-note">Detection is based on frequency (FFT), texture (LBP/GLCM), '
    "edge and sharpness patterns — not a guarantee, just a strong signal.</p>",
    unsafe_allow_html=True,
)
