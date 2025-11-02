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
import plotly.graph_objects as go

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")

# 🌾 --- GEMINI API KEY (pre-configured so user isn’t asked) ---
GEMINI_API_KEY = (
    "AIzaSyA2ug2CebXrODMdhgym00I4X1EYBqAxlcM"  # 🔒 Replace with your actual Gemini key
)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Smart Crop Technologies - Pest Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2E7D32; text-align: center; font-weight: bold; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #558B2F; text-align: center; margin-bottom: 2rem; }
    .ai-response { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .ai-badge { background-color: #FFD700; color: #000; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: bold; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- Helper Data ---
PEST_TREATMENTS = {
    "rice leaf roller": {"severity": "Medium", "treatment": "Apply Chlorantraniliprole 18.5% SC @ 150 ml/acre",
                         "organic": "Neem oil spray (5ml/liter), Release Trichogramma wasps",
                         "prevention": "Maintain proper spacing, Remove affected leaves"},
    "brown plant hopper": {"severity": "High", "treatment": "Buprofezin 25% SC @ 400-600 ml/acre",
                           "organic": "Neem cake application, Encourage spiders and mirid bugs",
                           "prevention": "Avoid excessive nitrogen, Use resistant varieties"},
    "aphids": {"severity": "Medium", "treatment": "Imidacloprid 17.8% SL @ 100 ml/acre",
               "organic": "Soap water spray, Garlic-chili extract",
               "prevention": "Yellow sticky traps, Encourage ladybugs"},
}

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini AI with embedded API key"""
    if not GEMINI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def get_ai_pesticide_recommendation(gemini_model, detected_pests, crop_type="rice", location="India", soil_type="loamy"):
    if not gemini_model:
        return None

    pest_list = ", ".join([p["pest"] for p in detected_pests])
    confidence_info = "\n".join([f"- {p['pest']}: {p['confidence']:.2%}" for p in detected_pests])

    prompt = f"""You are an expert agricultural AI advisor for Indian farmers.

Detected pests:
{confidence_info}

Crop: {crop_type}
Soil: {soil_type}
Location: {location}

Give detailed, step-by-step pesticide recommendations with:
1. Urgent action items 🌾
2. Recommended pesticides (chemical + dosage)
3. Organic alternatives 🌱
4. Application timing and safety 🧑‍🌾
5. Prevention for future infestation 🚜"""

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
    return {"severity": "Unknown", "treatment": "Consult expert", "organic": "Neem-based spray", "prevention": "Regular monitoring"}

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
            draw.text((x1, y1 - 15), f"{pest} {conf:.2f}", fill=(255, 255, 255), font=font)
    return img, detections

def main():
    st.markdown('<h1 class="main-header">🌾 Smart Crop Technologies</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Pest Detection & Farm Assistance</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
        st.title("⚙️ Settings")

        enable_ai_agent = st.checkbox("Enable AI Advisor", value=True)
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Cotton", "Maize"])
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy"])
        location = st.text_input("Location", "Mumbai, India")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25)
        show_treatment = st.checkbox("Show Treatment Info", value=True)

    gemini_model = initialize_gemini() if enable_ai_agent else None

    st.subheader("📸 Upload Crop Image")
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Pests"):
            model = load_model("best.pt")
            np_img = np.array(img)
            results = run_detection(model, np_img, conf_threshold)
            img_out, detections = draw_boxes(img, results, model.names)
            st.image(img_out, caption="Detection Results", use_column_width=True)
            st.success(f"✅ Found {len(detections)} pests")

            if enable_ai_agent and gemini_model and detections:
                st.info("🤖 Generating AI recommendations...")
                ai = get_ai_pesticide_recommendation(gemini_model, detections, crop_type, location, soil_type)
                st.markdown(f"### 🌾 AI Recommendation\n\n{ai}")

if __name__ == "__main__":
    main()
