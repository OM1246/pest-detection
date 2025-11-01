import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import requests
from io import BytesIO
import json

# REMOVED: import cv2  ← This was causing the error!

from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")

# Page config
st.set_page_config(
    page_title="Smart Crop Technologies - Pest Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .ai-badge {
        background-color: #FFD700;
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Pest treatment database
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
    """Load YOLO model with caching"""
    model = YOLO(model_path)
    return model

@st.cache_resource
def initialize_gemini(api_key):
    """Initialize Gemini AI"""
    if not GEMINI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def get_ai_pesticide_recommendation(gemini_model, detected_pests, crop_type="rice", location="India", soil_type="loamy"):
    """Get AI-powered pesticide recommendations from Gemini"""
    if not gemini_model:
        return None

    pest_list = ", ".join([p["pest"] for p in detected_pests])
    confidence_info = "\n".join([f"- {p['pest']}: {p['confidence']:.2%} confidence" for p in detected_pests])

    prompt = f"""You are an expert agricultural advisor AI assistant helping farmers in {location}.

DETECTED PESTS:
{confidence_info}

FARM DETAILS:
- Crop Type: {crop_type}
- Soil Type: {soil_type}
- Location: {location}

Please provide a comprehensive pesticide recommendation report with the following sections:

1. **IMMEDIATE ACTION REQUIRED** (within 24-48 hours)
   - List the most critical pests that need urgent treatment
   - Explain why immediate action is needed

2. **RECOMMENDED PESTICIDES** (for each detected pest)
   - Chemical name and brand examples available in {location}
   - Exact dosage (ml or grams per liter/acre)
   - Application method (spray, granular, etc.)
   - Cost estimate in INR (Indian Rupees)

3. **ORGANIC ALTERNATIVES**
   - Natural/organic solutions for farmers who prefer eco-friendly options
   - Homemade remedies if applicable

4. **APPLICATION SCHEDULE**
   - When to apply (time of day, weather conditions)
   - How many applications needed
   - Gap between applications

5. **SAFETY PRECAUTIONS**
   - Personal protective equipment needed
   - Re-entry period after spraying
   - Pre-harvest interval

6. **PREVENTION STRATEGY**
   - Long-term measures to prevent future infestations

7. **ESTIMATED COST BREAKDOWN**
   - Total cost for treatment
   - Cost per acre

Please format your response clearly with emojis for better readability. Be specific, practical, and farmer-friendly in your language. Focus on solutions available in Indian markets."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting AI recommendation: {str(e)}"

def get_weather_data(lat=28.6139, lon=77.2090):
    """Get weather data from OpenWeatherMap API"""
    try:
        api_key = "607903c167d20deb9c138c71d0171464"
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_treatment_info(pest_name):
    """Get treatment information for detected pest"""
    for key in PEST_TREATMENTS:
        if key.lower() in pest_name.lower():
            return PEST_TREATMENTS[key]
    return {
        "severity": "Unknown",
        "treatment": "Consult agricultural expert",
        "organic": "Use neem-based products",
        "prevention": "Regular monitoring recommended",
    }

def run_detection(model, img: np.ndarray, conf_thresh=0.25):
    """Run pest detection"""
    results = model(img, conf=conf_thresh, verbose=False)
    return results

def draw_boxes(pil_image: Image.Image, results, class_names):
    """Draw bounding boxes using PIL (NO OpenCV needed!)"""
    # Create a copy for drawing
    img_with_boxes = pil_image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a better font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    detections = []

    for r in results:
        if len(r.boxes) == 0:
            continue
            
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cls = int(cls.cpu().numpy())
            confidence = float(conf.cpu().numpy())
            
            pest_name = class_names[cls]

            detections.append({
                "pest": pest_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

            # Choose color based on confidence
            color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0)  # Green or Orange
            
            # Draw rectangle (bounding box)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Prepare label
            label = f"{pest_name[:20]} {confidence:.2f}"
            
            # Get text size for background
            bbox = draw.textbbox((x1, y1 - 20), label, font=font_small)
            
            # Draw background for text
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            
            # Draw text
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font_small)

    return img_with_boxes, detections

def create_detection_chart(detections):
    """Create confidence chart for detections"""
    if not detections:
        return None

    df = pd.DataFrame(detections)
    df = df.sort_values("confidence", ascending=True)

    fig = px.bar(
        df, x="confidence", y="pest", orientation="h",
        title="Detection Confidence Levels",
        labels={"confidence": "Confidence Score", "pest": "Pest Type"},
        color="confidence",
        color_continuous_scale="RdYlGn",
    )

    fig.update_layout(height=400, showlegend=False)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Smart Crop Technologies</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Pest Detection & Crop Management System</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
        st.title("⚙️ Settings")

        # Gemini AI Configuration
        st.subheader("🤖 AI Agent Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your free API key from https://aistudio.google.com/app/apikey"
        )

        enable_ai_agent = st.checkbox("Enable AI Pesticide Advisor", value=True)

        if enable_ai_agent and gemini_api_key:
            st.success("✅ AI Agent Ready!")
        elif enable_ai_agent and not gemini_api_key:
            st.warning("⚠️ Please enter Gemini API key")

        # Farm details for AI
        st.subheader("🌾 Farm Information")
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Other"])
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silt", "Red", "Black"])
        farm_location = st.text_input("Location (City/State)", "Mumbai, Maharashtra")

        # Detection settings
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

        # Location for weather
        st.subheader("📍 Location")
        use_location = st.checkbox("Enable Weather Integration", value=False)
        if use_location:
            latitude = st.number_input("Latitude", value=28.6139, format="%.4f")
            longitude = st.number_input("Longitude", value=77.2090, format="%.4f")

        # Additional features
        st.subheader("🔧 Features")
        show_treatment = st.checkbox("Show Treatment Advice", value=True)
        show_analytics = st.checkbox("Show Analytics", value=True)
        save_history = st.checkbox("Save Detection History", value=False)

    # Initialize Gemini if API key provided
    gemini_model = None
    if gemini_api_key and GEMINI_AVAILABLE:
        gemini_model = initialize_gemini(gemini_api_key)

    # Class names (your full list)
    class_names = [
        "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
        "yellow rice borer", "rice gall midge", "Rice Stemfly", "brown plant hopper",
        "white backed plant hopper", "small brown plant hopper", "rice water weevil",
        "rice leafhopper", "grain spreader thrips", "rice shell pest", "grub", "mole cricket",
        "wireworm", "white margined moth", "black cutworm", "large cutworm", "yellow cutworm",
        "red spider", "corn borer", "army worm", "aphids", "Potosiabre vitarsis", "peach borer",
        "english grain aphid", "green bug", "bird cherry-oataphid", "wheat blossom midge",
        "penthaleus major", "longlegged spider mite", "wheat phloeothrips", "wheat sawfly",
        "cerodonta denticornis", "beet fly", "flea beetle", "cabbage army worm", "beet army worm",
        "Beet spot flies", "meadow moth", "beet weevil", "sericaorient alismots chulsky",
        "alfalfa weevil", "flax budworm", "alfalfa plant bug", "tarnished plant bug", "Locustoidea",
        "lytta polita", "legume blister beetle", "blister beetle", "therioaphis maculata Buckton",
        "odontothrips loti", "Thrips", "alfalfa seed chalcid", "Pieris canidia", "Apolygus lucorum",
        "Limacodidae", "Viteus vitifoliae", "Colomerus vitis", "Brevipoalpus lewisi McGregor",
        "oides decempunctata", "Polyphagotars onemus latus", "Pseudococcus comstocki Kuwana",
        "parathrene regalis", "Ampelophaga", "Lycorma delicatula", "Xylotrechus", "Cicadella viridis",
        "Miridae", "Trialeurodes vaporariorum", "Erythroneura apicalis", "Papilio xuthus",
        "Panonchus citri McGregor", "Phyllocoptes oleiverus ashmead", "Icerya purchasi Maskell",
        "Unaspis yanonensis", "Ceroplastes rubens", "Chrysomphalus aonidum", "Parlatoria zizyphus Lucus",
        "Nipaecoccus vastalor", "Aleurocanthus spiniferus", "Tetradacus c Bactrocera minax",
        "Dacus dorsalis(Hendel)", "Bactrocera tsuneonis", "Prodenia litura", "Adristyrannus",
        "Phyllocnistis citrella Stainton", "Toxoptera citricidus", "Toxoptera aurantii",
        "Aphis citricola Vander Goot", "Scirtothrips dorsalis Hood", "Dasineura sp",
        "Lawana imitata Melichar", "Salurnis marginella Guerr", "Deporaus marginatus Pascoe",
        "Chlumetia transversa", "Mango flat beak leafhopper", "Rhytidodera bowrinii white",
        "Sternochetus frigidus", "Cicadellidae",
    ]

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Detection", "🤖 AI Advisor", "📊 Analytics", "🌤️ Weather"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)

        if uploaded_file:
            if st.button("🔍 Detect Pests & Get AI Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with AI..."):
                    # Load model
                    model = load_model("best.pt")

                    # Convert PIL to numpy for YOLO
                    img_np = np.array(image)

                    # Run detection
                    results = run_detection(model, img_np, conf_threshold)
                    
                    # Draw boxes using PIL (not OpenCV!)
                    img_out, detections = draw_boxes(image, results, class_names)

                    # Store detections in session state
                    st.session_state["current_detections"] = detections
                    st.session_state["detection_image"] = img_out

                    # Display results
                    st.success(f"✅ Analysis Complete! Found {len(detections)} pest(s)")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.image(img_out, caption="Detection Results", use_column_width=True)

                    with col2:
                        if detections:
                            st.subheader("🐛 Detected Pests")
                            for i, det in enumerate(detections, 1):
                                with st.expander(f"{i}. {det['pest']} (Confidence: {det['confidence']:.2%})"):
                                    if show_treatment:
                                        treatment = get_treatment_info(det["pest"])
                                        st.markdown(f"**Severity:** `{treatment['severity']}`")
                                        st.info(f"**Treatment:** {treatment['treatment']}")
                                        st.success(f"**Organic:** {treatment['organic']}")
                                        st.warning(f"**Prevention:** {treatment['prevention']}")
                        else:
                            st.info("No pests detected! Your crops look healthy 🌱")

                    # Analytics
                    if show_analytics and detections:
                        st.subheader("📊 Detection Analytics")
                        fig = create_detection_chart(detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pests", len(detections))
                        with col2:
                            avg_conf = np.mean([d["confidence"] for d in detections])
                            st.metric("Avg Confidence", f"{avg_conf:.2%}")
                        with col3:
                            high_risk = sum(1 for d in detections if d["confidence"] > 0.7)
                            st.metric("High Risk Detections", high_risk)

                    # Trigger AI recommendation
                    if enable_ai_agent and gemini_model and detections:
                        st.info("🤖 Generating AI recommendations...")
                        with st.spinner("AI Agent analyzing..."):
                            ai_recommendation = get_ai_pesticide_recommendation(
                                gemini_model, detections, crop_type.lower(),
                                farm_location, soil_type.lower()
                            )
                            st.session_state["ai_recommendation"] = ai_recommendation

                    # Save history
                    if save_history and detections:
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "detections": len(detections),
                            "pests": [d["pest"] for d in detections],
                        })

    with tab2:
        st.subheader("🤖 AI-Powered Pesticide Advisor")
        st.markdown('<span class="ai-badge">✨ POWERED BY GEMINI AI</span>', unsafe_allow_html=True)

        if not GEMINI_AVAILABLE:
            st.error("❌ Install: `pip install google-generativeai`")
        elif not gemini_api_key:
            st.warning("⚠️ Please enter Gemini API key in sidebar")
            st.info("Get free key: https://aistudio.google.com/app/apikey")
        elif "current_detections" not in st.session_state or not st.session_state["current_detections"]:
            st.info("👆 Detect pests first in the 'Detection' tab!")
        else:
            if "ai_recommendation" in st.session_state:
                st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                st.markdown("### 🎯 Personalized Recommendations")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown(st.session_state["ai_recommendation"])

                st.download_button(
                    "📥 Download Recommendations",
                    st.session_state["ai_recommendation"],
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            else:
                if st.button("🤖 Get AI Recommendations", type="primary", use_container_width=True):
                    with st.spinner("AI analyzing..."):
                        detections = st.session_state["current_detections"]
                        ai_recommendation = get_ai_pesticide_recommendation(
                            gemini_model, detections, crop_type.lower(),
                            farm_location, soil_type.lower()
                        )
                        st.session_state["ai_recommendation"] = ai_recommendation
                        st.rerun()

    with tab3:
        st.subheader("📈 Historical Analytics")
        if "history" in st.session_state and st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True)
            csv = df_history.to_csv(index=False)
            st.download_button("📥 Download Report", csv,
                             file_name=f"report_{datetime.now().strftime('%Y%m%d')}.csv",
                             mime="text/csv")
        else:
            st.info("No history yet. Start analyzing images!")

    with tab4:
        st.subheader("🌤️ Weather & Recommendations")
        if use_location:
            weather_data = get_weather_data(latitude, longitude)
            if weather_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{weather_data.get('main', {}).get('temp', 'N/A')}°C")
                with col2:
                    st.metric("Humidity", f"{weather_data.get('main', {}).get('humidity', 'N/A')}%")
                with col3:
                    st.metric("Conditions", weather_data.get("weather", [{}])[0].get("main", "N/A"))
            else:
                st.error("Unable to fetch weather data")
        else:
            st.info("Enable weather integration in sidebar")

    # Footer
    st.markdown("---")
    st.markdown(f"🌾 **Smart Crop Technologies** | 📞 Support: 1800-XXX-XXXX | 📅 {datetime.now().strftime('%d %B %Y')}")

if __name__ == "__main__":
    main()
