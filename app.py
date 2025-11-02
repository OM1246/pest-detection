import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import requests
from io import BytesIO
import json
from ultralytics import YOLO
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning(
        "⚠️ google-generativeai not installed. Install with: pip install google-generativeai"
    )

# --- Gemini API Key (pre-configured, not user-entered)
GEMINI_API_KEY = "AIzaSyA2ug2CebXrODMdhgym00I4X1EYBqAxlcM"

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Smart Crop Technologies - Pest Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
    .main-header { font-size: 3rem; color: #2E7D32; text-align: center; font-weight: bold; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #558B2F; text-align: center; margin-bottom: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Helper Data ---
PEST_TREATMENTS = {
    "rice leaf roller": {
        "severity": "Medium",
        "treatment": "Apply Chlorantraniliprole 18.5% SC @ 150 ml/acre",
        "organic": "Neem oil spray (5ml/liter), Release Trichogramma wasps",
        "prevention": "Maintain proper spacing, Remove affected leaves",
    },
    "brown plant hopper": {
        "severity": "High",
        "treatment": "Buprofezin 25% SC @ 400-600 ml/acre",
        "organic": "Neem cake application, Encourage spiders and mirid bugs",
        "prevention": "Avoid excessive nitrogen, Use resistant varieties",
    },
    "aphids": {
        "severity": "Medium",
        "treatment": "Imidacloprid 17.8% SL @ 100 ml/acre",
        "organic": "Soap water spray, Garlic-chili extract",
        "prevention": "Yellow sticky traps, Encourage ladybugs",
    },
}


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


@st.cache_resource
def initialize_gemini():
    if not GEMINI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None


def get_ai_pesticide_recommendation(
    gemini_model, detected_pests, crop_type="rice", location="India", soil_type="loamy"
):
    if not gemini_model:
        return None
    pest_info = "\n".join(
        [f"- {p['pest']}: {p['confidence']:.2%}" for p in detected_pests]
    )
    prompt = f"""You are an agricultural AI assistant. 
Detected pests:
{pest_info}
Crop: {crop_type}, Soil: {soil_type}, Location: {location}

Provide:
1. Urgent actions
2. Recommended pesticides (chemical + dosage)
3. Organic alternatives
4. Safety & prevention steps
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error from Gemini: {str(e)}"


def get_weather_data(lat=28.6139, lon=77.2090):
    try:
        api_key = "607903c167d20deb9c138c71d0171464"
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None


def get_treatment_info(pest):
    for key in PEST_TREATMENTS:
        if key.lower() in pest.lower():
            return PEST_TREATMENTS[key]
    return {
        "severity": "Unknown",
        "treatment": "Consult expert",
        "organic": "Neem-based spray",
        "prevention": "Regular monitoring",
    }


def run_detection(model, img: np.ndarray, conf_thresh=0.25):
    return model(img, conf=conf_thresh, verbose=False)


def draw_boxes(image, results, class_names):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    detections = []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            pest = class_names[int(cls)]
            detections.append({"pest": pest, "confidence": float(conf)})
            color = (0, 255, 0) if conf > 0.7 else (255, 165, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text(
                (x1, y1 - 15), f"{pest} {conf:.2f}", fill=(255, 255, 255), font=font
            )
    return img, detections


# --- Report Generator ---
def generate_report(detections, ai_text, crop_type, location, soil_type, weather_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(
        Paragraph("🌾 Smart Crop Technologies - Pest Detection Report", styles["Title"])
    )
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y, %H:%M')}",
            styles["Normal"],
        )
    )
    elements.append(Paragraph(f"<b>Location:</b> {location}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Crop Type:</b> {crop_type}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Soil Type:</b> {soil_type}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if weather_data:
        w = weather_data["main"]
        elements.append(
            Paragraph(
                f"<b>Weather:</b> {w['temp']}°C, {w['humidity']}% humidity",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 12))

    # Pest Detection Table
    data = [
        ["Pest Name", "Confidence", "Severity", "Treatment", "Organic", "Prevention"]
    ]
    for det in detections:
        t = get_treatment_info(det["pest"])
        data.append(
            [
                det["pest"],
                f"{det['confidence']:.2%}",
                t["severity"],
                t["treatment"],
                t["organic"],
                t["prevention"],
            ]
        )

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("<b>🤖 AI Pesticide Recommendation:</b>", styles["Heading2"])
    )
    elements.append(Paragraph(ai_text.replace("\n", "<br/>"), styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# --- Streamlit App ---
def main():
    st.markdown(
        '<h1 class="main-header">🌾 Smart Crop Technologies</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Pest Detection & Farm Report System</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
        st.title("⚙️ Settings")

        enable_ai = st.checkbox("Enable AI Advisor", value=True)
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Cotton", "Maize"])
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy"])
        location = st.text_input("Location", "Mumbai, India")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25)
        show_treatment = st.checkbox("Show Treatment Info", value=True)
        include_weather = st.checkbox("Include Weather Info", value=True)

    gemini = initialize_gemini() if enable_ai else None
    weather_data = get_weather_data() if include_weather else None

    st.subheader("📸 Upload Crop Image")
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Pests"):
            model = load_model("best.pt")
            np_img = np.array(img)
            results = run_detection(model, np_img, conf_thresh)
            img_out, detections = draw_boxes(img, results, model.names)
            st.image(img_out, caption="Detection Results", use_column_width=True)
            st.success(f"✅ Found {len(detections)} pest(s)")

            if enable_ai and gemini and detections:
                st.info("🤖 Generating AI Recommendations...")
                ai_text = get_ai_pesticide_recommendation(
                    gemini, detections, crop_type, location, soil_type
                )
                st.markdown("### 🌾 AI Recommendation")
                st.markdown(ai_text)

                # Generate downloadable report
                st.subheader("📄 Generate Report")
                report_buf = generate_report(
                    detections, ai_text, crop_type, location, soil_type, weather_data
                )
                st.download_button(
                    "📥 Download Detailed Report (PDF)",
                    data=report_buf,
                    file_name=f"pest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
