"""
Streamlit image classification web app (TorchScript inference) for Origami Buddy.
Custom redesigned UI with high-quality visual elements.
"""

from __future__ import annotations

import io
import json
import base64
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model" / "model.ts"
LABELS_PATH = HERE / "model" / "labels.json"
TOP_K = 5

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Origami Buddy", page_icon="🪽", layout="wide")


def get_base64_image(file_path: Path) -> str:
    if not file_path.exists():
        return ""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


# Load assets
ASSETS_DIR = HERE / "assets"
floral_b64 = get_base64_image(ASSETS_DIR / "floral_border.png")
crane_b64 = get_base64_image(ASSETS_DIR / "origami_crane.svg")
papers_b64 = get_base64_image(ASSETS_DIR / "origami_papers.svg")
white_papers_b64 = get_base64_image(ASSETS_DIR / "origami_white_papers.svg")
ripped_b64 = get_base64_image(ASSETS_DIR / "ripped_paper.svg")


def inject_style() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        :root {{
            color-scheme: light;
            background-color: #f5f3e8;
        }}
        
        .stApp {{
            background: #f5f3e8;
            font-family: 'Outfit', sans-serif;
            color: #2b3e2c;
        }}
        
        /* Layout Padding and Max Width */
        .main .block-container {{
            padding-top: 0rem;
            padding-bottom: 0rem;
            max-width: 1200px;
        }}
        
        /* Custom Header Styling */
        .header-container {{
            position: relative;
            background-color: #f5f3e8;
            border-radius: 24px;
            margin-bottom: 2rem;
            overflow: hidden;
            width: 100%;
        }}
        
        .floral-banner {{
            height: 140px;
            background-image: url('data:image/png;base64,{floral_b64}');
            background-size: cover;
            background-position: center;
            width: 100%;
            border-top-left-radius: 24px;
            border-top-right-radius: 24px;
        }}
        
        .header-card-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: -60px;
            position: relative;
            z-index: 10;
        }}
        
        .header-card {{
            background: white;
            border-radius: 50px;
            padding: 0.75rem 2.5rem;
            box-shadow: 0 10px 30px rgba(79, 97, 70, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            border: 1px solid #e5e3d8;
        }}
        
        .header-left-img {{
            height: 55px;
            width: 55px;
            object-fit: contain;
        }}
        
        .header-right-img {{
            height: 55px;
            width: 55px;
            object-fit: contain;
        }}
        
        .logo-text {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.25em;
            color: #334a34;
            margin-left: 0.25em;
        }}
        
        .sub-pill {{
            margin-top: 1rem;
            background-color: #334a34;
            color: #ffffff;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            box-shadow: 0 4px 15px rgba(51, 74, 52, 0.2);
            text-transform: uppercase;
        }}
        
        /* Tabs Custom Styling */
        .stTabs {{
            position: relative;
            background: white;
            padding: 1.5rem;
            border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.03);
            border: 1px solid #e5e3d8;
            margin-bottom: 2rem;
        }}
        
        .stTabs [role="tablist"] {{
            border-bottom: 2px solid #e2e8f0;
            gap: 0.5rem;
            padding-bottom: 0px;
            margin-bottom: 1.5rem;
            position: relative;
        }}
        
        .stTabs [role="tab"] {{
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-bottom: none !important;
            border-top-left-radius: 12px !important;
            border-top-right-radius: 12px !important;
            border-bottom-left-radius: 0px !important;
            border-bottom-right-radius: 0px !important;
            padding: 0.6rem 1.8rem !important;
            height: auto !important;
            transition: all 0.2s ease;
            margin-bottom: -2px !important;
        }}
        
        .stTabs [role="tab"][aria-selected="true"] {{
            background-color: #ffffff !important;
            color: #334a34 !important;
            border-color: #334a34 !important;
            border-bottom: 2px solid #ffffff !important;
            font-weight: 700 !important;
        }}
        
        .stTabs [role="tablist"]::after {{
            content: "";
            display: block;
            position: absolute;
            right: 10px;
            bottom: 6px;
            width: 45px;
            height: 45px;
            background-image: url('data:image/svg+xml;base64,{white_papers_b64}');
            background-size: contain;
            background-repeat: no-repeat;
            pointer-events: none;
            z-index: 10;
        }}
        
        /* Column Elements */
        .image-container {{
            border: 3px solid #334a34;
            border-radius: 24px;
            background-color: #ffffff;
            padding: 0.8rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 420px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.04);
            position: relative;
            overflow: hidden;
        }}
        
        .image-placeholder-box {{
            height: 420px;
            border: 3px dashed #b4c1aa;
            border-radius: 24px;
            background: #f7f6f1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #7a8b72;
            width: 100%;
            text-align: center;
        }}
        
        .image-placeholder-text {{
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #7a8b72;
        }}
        
        .prediction-box {{
            background: #ffffff;
            border-radius: 24px;
            padding: 1.75rem;
            border: 1px solid #e5e3d8;
            box-shadow: 0 12px 35px rgba(0,0,0,0.04);
            height: 100%;
        }}
        
        .prediction-header {{
            background-color: #e7f0df;
            color: #334a34;
            padding: 0.5rem 1.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            display: inline-block;
            margin-bottom: 1.2rem;
        }}
        
        .top-prediction-title {{
            font-size: 2rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}
        
        .confidence-note {{
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 1rem;
        }}
        
        .classes-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #334a34;
            margin-bottom: 1rem;
        }}
        
        .class-bar-wrap {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.75rem;
            font-weight: 600;
        }}
        
        .class-label {{
            font-size: 0.95rem;
            color: #334a34;
        }}
        
        .class-value {{
            font-size: 0.95rem;
            color: #64748b;
        }}
        
        /* Custom Progress Bars */
        div[data-testid="stProgress"] > div {{
            background-color: #f1f5f9 !important;
            border-radius: 10px !important;
            height: 10px !important;
        }}
        
        div[data-testid="stProgress"] div[role="progressbar"] {{
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%) !important;
            border-radius: 10px !important;
        }}
        
        /* Example images grid */
        .example-container {{
            background-color: #eef5e8;
            border: 1px solid #d1e2d3;
            border-radius: 24px;
            padding: 1.75rem;
            margin-top: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.02);
            width: 100%;
        }}
        
        .example-heading {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #334a34;
            margin-bottom: 1.2rem;
        }}
        
        .thumbnail-card {{
            background: white;
            border-radius: 14px;
            padding: 0.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            text-align: center;
        }}
        
        .thumbnail-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(51, 74, 52, 0.15);
        }}
        
        /* Native Streamlit file uploader customization */
        div[data-testid="stFileUploader"] {{
            background-color: #fcfbf9;
            border: 2px dashed #cbd2c3;
            border-radius: 16px;
            padding: 1rem;
        }}
        
        div[data-testid="stFileUploader"] section {{
            padding: 1rem 0;
        }}
        
        /* Footer styling */
        .footer-container {{
            position: relative;
            width: 100%;
            margin-top: 4rem;
            left: 0;
            right: 0;
        }}
        
        /* Hide default Streamlit main header and footer */
        #MainMenu, footer, header {{
            visibility: hidden;
            height: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    meta = json.loads(LABELS_PATH.read_text())
    return model, meta["labels"], meta["preprocess"]


def preprocess(pil_img: Image.Image, pre: dict) -> torch.Tensor:
    """Reproduce the fastai validation transforms (resize-crop, scale to [0,1])."""
    size = int(pre.get("size", 224))
    img = TF.resize(pil_img, size)
    img = TF.center_crop(img, [size, size])
    x = TF.pil_to_tensor(img).float()
    if pre.get("divide_255", True):
        x = x / 255.0
    if pre.get("normalize", False):
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0)


@torch.no_grad()
def classify(model, labels, pre, pil_img: Image.Image):
    x = preprocess(pil_img, pre)
    probs = model(x).softmax(dim=1)[0]
    pairs = sorted(
        ((labels[i], float(probs[i])) for i in range(len(labels))),
        key=lambda p: p[1],
        reverse=True,
    )
    return pairs


def to_pil(file_or_bytes) -> Image.Image:
    data = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def show_header() -> None:
    st.markdown(
        f"""
        <div class="header-container">
            <div class="floral-banner"></div>
            <div class="header-card-wrapper">
                <div class="header-card">
                    <img src="data:image/svg+xml;base64,{papers_b64}" class="header-left-img" />
                    <div class="logo-text">O R I G A M I &nbsp; B U D D Y</div>
                    <img src="data:image/svg+xml;base64,{crane_b64}" class="header-right-img" />
                </div>
                <div class="sub-pill">crane Image Classifier</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-container">
            <img src="data:image/svg+xml;base64,{ripped_b64}" style="width: 100%; display: block; margin: 0; padding: 0;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_images() -> None:
    st.markdown('<div class="example-container">', unsafe_allow_html=True)
    st.markdown('<div class="example-heading">Example test images:</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    samples = [
        ("Sample 1", "example cranes/Sample1.jpg"),
        ("Sample 2", "example cranes/Sample2.jpg"),
        ("Sample 3", "example cranes/Sample3.jpg"),
        ("Sample 4", "example cranes/Sample4.jpg"),
        ("Sample 5", "example cranes/Sample5.jpg"),
    ]
    
    for i, (name, path) in enumerate(samples):
        with [col1, col2, col3, col4, col5][i]:
            is_selected = st.session_state.get("selected_image") == path
            border_style = "border: 3px solid #334a34;" if is_selected else "border: 1px solid #dcdad0;"
            st.markdown(f'<div class="thumbnail-card" style="{border_style}">', unsafe_allow_html=True)
            if Path(path).exists():
                st.image(path, use_container_width=True)
                if st.button("Try", key=f"btn_{i}", use_container_width=True):
                    st.session_state.selected_image = path
                    st.rerun()
            else:
                st.markdown('<div style="height:80px; display:flex; align-items:center; justify-content:center; color:#94a3b8;">N/A</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Initialize session states
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    inject_style()
    show_header()

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.error(
            "Model artifacts not found. Run `python convert_model.py` once to "
            "produce `model/model.ts` and `model/labels.json` from your "
            "fastai `export.pkl`."
        )
        st.stop()

    model, labels, pre = load_model()

    # Input section: Tabs
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Camera"])
    
    uploaded = None
    snapshot = None
    
    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="file_uploader_widget"
        )
        
    with tab_camera:
        snapshot = st.camera_input("Take a photo", key="camera_widget")

    # Determine active image based on user actions
    active_image = None
    
    if uploaded is not None:
        active_image = to_pil(uploaded)
        # Clear selected sample if a new image is uploaded
        if st.session_state.last_uploaded != uploaded.name:
            st.session_state.selected_image = None
            st.session_state.last_uploaded = uploaded.name
    elif snapshot is not None:
        active_image = to_pil(snapshot)
        # Clear selected sample if a new camera image is taken
        st.session_state.selected_image = None
        st.session_state.last_uploaded = None
    elif st.session_state.selected_image is not None:
        sample_path = Path(st.session_state.selected_image)
        if sample_path.exists():
            active_image = Image.open(sample_path).convert("RGB")
            st.session_state.last_uploaded = None

    # Columns layout for image container and predictions
    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        if active_image is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(active_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="image-container"><div class="image-placeholder-box"><div class="image-placeholder-text">uploaded image goes here</div></div></div>',
                unsafe_allow_html=True,
            )
            
    with right_col:
        if active_image is not None:
            pairs = classify(model, labels, pre, active_image)
            top_label, top_prob = pairs[0]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<div class="prediction-header">prediction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="top-prediction-title">{top_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-note">Confidence: <strong>{top_prob:.1%}</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="classes-title">Top 5 Classes</div>', unsafe_allow_html=True)
            for label, prob in pairs[:TOP_K]:
                st.markdown(
                    f'<div class="class-bar-wrap"><div class="class-label">{label}</div><div class="class-value">{prob:.1%}</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(prob)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<div class="prediction-header">prediction</div>', unsafe_allow_html=True)
            st.markdown('<div style="height: 120px; display: flex; align-items: center; justify-content: center; color: #8a9f84; font-weight: 500;">Pick or take a photo to see results.</div>', unsafe_allow_html=True)
            st.markdown('<div class="confidence-note" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">Top 5 classes will appear after classification.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        render_example_images()

    # Ripped paper bottom border
    show_footer()


if __name__ == "__main__":
    main()
