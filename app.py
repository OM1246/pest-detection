import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
from datetime import datetime
import pandas as pd
import requests
from io import BytesIO
import json

# You might need to install: pip install ultralytics plotly google-generativeai
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning(
        "⚠️ google-generativeai not installed. Install with: pip install google-generativeai"
    )

# Page config
st.set_page_config(
    page_title="Smart Crop Technologies - Pest Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
""",
    unsafe_allow_html=True,
)

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


def get_ai_pesticide_recommendation(
    gemini_model, detected_pests, crop_type="rice", location="India", soil_type="loamy"
):
    """Get AI-powered pesticide recommendations from Gemini"""
    if not gemini_model:
        return None

    pest_list = ", ".join([p["pest"] for p in detected_pests])
    confidence_info = "\n".join(
        [f"- {p['pest']}: {p['confidence']:.2%} confidence" for p in detected_pests]
    )

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
        api_key = (
            "607903c167d20deb9c138c71d0171464"  # Users should add their own key
        )
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


def draw_boxes(img: np.ndarray, results, class_names):
    """Draw bounding boxes on image"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    detections = []

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            pest_name = class_names[cls]
            confidence = float(conf)

            detections.append(
                {"pest": pest_name, "confidence": confidence, "bbox": (x1, y1, x2, y2)}
            )

            label = f"{pest_name} {confidence:.2f}"
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img_bgr, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1
            )
            cv2.putText(
                img_bgr,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, detections


def create_detection_chart(detections):
    """Create confidence chart for detections"""
    if not detections:
        return None

    df = pd.DataFrame(detections)
    df = df.sort_values("confidence", ascending=True)

    fig = px.bar(
        df,
        x="confidence",
        y="pest",
        orientation="h",
        title="Detection Confidence Levels",
        labels={"confidence": "Confidence Score", "pest": "Pest Type"},
        color="confidence",
        color_continuous_scale="RdYlGn",
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🌾 Smart Crop Technologies</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Pest Detection & Crop Management System</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
        st.title("⚙️ Settings")

        # Gemini AI Configuration
        st.subheader("🤖 AI Agent Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your free API key from https://makersuite.google.com/app/apikey",
        )

        enable_ai_agent = st.checkbox("Enable AI Pesticide Advisor", value=True)

        if enable_ai_agent and gemini_api_key:
            st.success("✅ AI Agent Ready!")
        elif enable_ai_agent and not gemini_api_key:
            st.warning("⚠️ Please enter Gemini API key")

        # Farm details for AI
        st.subheader("🌾 Farm Information")
        crop_type = st.selectbox(
            "Crop Type", ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Other"]
        )
        soil_type = st.selectbox(
            "Soil Type", ["Loamy", "Clay", "Sandy", "Silt", "Red", "Black"]
        )
        farm_location = st.text_input("Location (City/State)", "Mumbai, Maharashtra")

        # Detection settings
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

        # Language selection
        language = st.selectbox(
            "Language / भाषा", ["English", "हिंदी (Hindi)", "தமிழ் (Tamil)"]
        )

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

    # Class names
    class_names = [
        "rice leaf roller",
        "rice leaf caterpillar",
        "paddy stem maggot",
        "asiatic rice borer",
        "yellow rice borer",
        "rice gall midge",
        "Rice Stemfly",
        "brown plant hopper",
        "white backed plant hopper",
        "small brown plant hopper",
        "rice water weevil",
        "rice leafhopper",
        "grain spreader thrips",
        "rice shell pest",
        "grub",
        "mole cricket",
        "wireworm",
        "white margined moth",
        "black cutworm",
        "large cutworm",
        "yellow cutworm",
        "red spider",
        "corn borer",
        "army worm",
        "aphids",
        "Potosiabre vitarsis",
        "peach borer",
        "english grain aphid",
        "green bug",
        "bird cherry-oataphid",
        "wheat blossom midge",
        "penthaleus major",
        "longlegged spider mite",
        "wheat phloeothrips",
        "wheat sawfly",
        "cerodonta denticornis",
        "beet fly",
        "flea beetle",
        "cabbage army worm",
        "beet army worm",
        "Beet spot flies",
        "meadow moth",
        "beet weevil",
        "sericaorient alismots chulsky",
        "alfalfa weevil",
        "flax budworm",
        "alfalfa plant bug",
        "tarnished plant bug",
        "Locustoidea",
        "lytta polita",
        "legume blister beetle",
        "blister beetle",
        "therioaphis maculata Buckton",
        "odontothrips loti",
        "Thrips",
        "alfalfa seed chalcid",
        "Pieris canidia",
        "Apolygus lucorum",
        "Limacodidae",
        "Viteus vitifoliae",
        "Colomerus vitis",
        "Brevipoalpus lewisi McGregor",
        "oides decempunctata",
        "Polyphagotars onemus latus",
        "Pseudococcus comstocki Kuwana",
        "parathrene regalis",
        "Ampelophaga",
        "Lycorma delicatula",
        "Xylotrechus",
        "Cicadella viridis",
        "Miridae",
        "Trialeurodes vaporariorum",
        "Erythroneura apicalis",
        "Papilio xuthus",
        "Panonchus citri McGregor",
        "Phyllocoptes oleiverus ashmead",
        "Icerya purchasi Maskell",
        "Unaspis yanonensis",
        "Ceroplastes rubens",
        "Chrysomphalus aonidum",
        "Parlatoria zizyphus Lucus",
        "Nipaecoccus vastalor",
        "Aleurocanthus spiniferus",
        "Tetradacus c Bactrocera minax",
        "Dacus dorsalis(Hendel)",
        "Bactrocera tsuneonis",
        "Prodenia litura",
        "Adristyrannus",
        "Phyllocnistis citrella Stainton",
        "Toxoptera citricidus",
        "Toxoptera aurantii",
        "Aphis citricola Vander Goot",
        "Scirtothrips dorsalis Hood",
        "Dasineura sp",
        "Lawana imitata Melichar",
        "Salurnis marginella Guerr",
        "Deporaus marginatus Pascoe",
        "Chlumetia transversa",
        "Mango flat beak leafhopper",
        "Rhytidodera bowrinii white",
        "Sternochetus frigidus",
        "Cicadellidae",
    ]

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📷 Detection",
            "🤖 AI Advisor",
            "📊 Analytics",
            "🌤️ Weather",
            "📚 Pest Library",
        ]
    )

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Upload Image")
            upload_option = st.radio(
                "Choose input method:", ["Upload File", "Take Photo", "Sample Images"]
            )

            uploaded_file = None
            if upload_option == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose an image", type=["jpg", "jpeg", "png"]
                )
            elif upload_option == "Take Photo":
                uploaded_file = st.camera_input("Take a photo")
            else:
                sample = st.selectbox(
                    "Select sample", ["Sample 1", "Sample 2", "Sample 3"]
                )
                st.info("Sample images would be loaded here")

        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

        if uploaded_file:
            if st.button(
                "🔍 Detect Pests & Get AI Recommendations",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing image with AI..."):
                    # Load model
                    model = load_model("best.pt")

                    # Run detection
                    results = run_detection(model, img_np, conf_threshold)
                    img_out, detections = draw_boxes(img_np, results, class_names)

                    # Store detections in session state
                    st.session_state["current_detections"] = detections
                    st.session_state["detection_image"] = img_out

                    # Display results
                    st.success(f"✅ Analysis Complete! Found {len(detections)} pest(s)")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.image(
                            img_out, caption="Detection Results", use_column_width=True
                        )

                    with col2:
                        if detections:
                            st.subheader("🐛 Detected Pests")
                            for i, det in enumerate(detections, 1):
                                with st.expander(
                                    f"{i}. {det['pest']} (Confidence: {det['confidence']:.2%})"
                                ):
                                    if show_treatment:
                                        treatment = get_treatment_info(det["pest"])

                                        st.markdown(
                                            f"**Severity:** `{treatment['severity']}`"
                                        )
                                        st.markdown(f"**Chemical Treatment:**")
                                        st.info(treatment["treatment"])
                                        st.markdown(f"**Organic Alternative:**")
                                        st.success(treatment["organic"])
                                        st.markdown(f"**Prevention:**")
                                        st.warning(treatment["prevention"])
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
                            high_risk = sum(
                                1 for d in detections if d["confidence"] > 0.7
                            )
                            st.metric("High Risk Detections", high_risk)

                    # Trigger AI recommendation automatically if enabled
                    if enable_ai_agent and gemini_model and detections:
                        st.info(
                            "🤖 Generating AI-powered pesticide recommendations... Check the 'AI Advisor' tab!"
                        )

                        with st.spinner("AI Agent is analyzing detected pests..."):
                            ai_recommendation = get_ai_pesticide_recommendation(
                                gemini_model,
                                detections,
                                crop_type.lower(),
                                farm_location,
                                soil_type.lower(),
                            )
                            st.session_state["ai_recommendation"] = ai_recommendation

                    # Save history
                    if save_history and detections:
                        if "history" not in st.session_state:
                            st.session_state.history = []

                        st.session_state.history.append(
                            {
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "detections": len(detections),
                                "pests": [d["pest"] for d in detections],
                            }
                        )

    with tab2:
        st.subheader("🤖 AI-Powered Pesticide Advisor")
        st.markdown(
            '<span class="ai-badge">✨ POWERED BY GEMINI AI</span>',
            unsafe_allow_html=True,
        )

        if not GEMINI_AVAILABLE:
            st.error(
                "❌ Google Generative AI library not installed. Run: `pip install google-generativeai`"
            )
        elif not gemini_api_key:
            st.warning(
                "⚠️ Please enter your Gemini API key in the sidebar to enable AI recommendations."
            )
            st.info(
                """
            **How to get your free Gemini API key:**
            1. Visit: https://makersuite.google.com/app/apikey
            2. Sign in with Google account
            3. Click "Create API Key"
            4. Copy and paste it in the sidebar
            """
            )
        elif (
            "current_detections" not in st.session_state
            or not st.session_state["current_detections"]
        ):
            st.info(
                "👆 Please detect pests first using the 'Detection' tab, then come back here for AI recommendations!"
            )
        else:
            # Show AI recommendations
            if "ai_recommendation" in st.session_state:
                st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                st.markdown("### 🎯 Personalized Pesticide Recommendations")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(st.session_state["ai_recommendation"])

                # Download recommendation
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Recommendation as Text",
                        data=st.session_state["ai_recommendation"],
                        file_name=f"pesticide_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )

                with col2:
                    if st.button("🔄 Regenerate Recommendation"):
                        with st.spinner("Regenerating AI recommendations..."):
                            detections = st.session_state["current_detections"]
                            ai_recommendation = get_ai_pesticide_recommendation(
                                gemini_model,
                                detections,
                                crop_type.lower(),
                                farm_location,
                                soil_type.lower(),
                            )
                            st.session_state["ai_recommendation"] = ai_recommendation
                            st.rerun()
            else:
                if st.button(
                    "🤖 Get AI Recommendations Now",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "AI Agent is analyzing your pests and preparing recommendations..."
                    ):
                        detections = st.session_state["current_detections"]
                        ai_recommendation = get_ai_pesticide_recommendation(
                            gemini_model,
                            detections,
                            crop_type.lower(),
                            farm_location,
                            soil_type.lower(),
                        )
                        st.session_state["ai_recommendation"] = ai_recommendation
                        st.rerun()

        # Additional AI features
        st.markdown("---")
        st.subheader("💬 Ask AI Agent Anything")
        user_question = st.text_area(
            "Have a specific question about pest management?",
            placeholder="e.g., What's the best time to spray pesticides? Are organic options effective?",
        )

        if st.button("Ask AI") and user_question and gemini_model:
            with st.spinner("AI is thinking..."):
                try:
                    context = ""
                    if "current_detections" in st.session_state:
                        pests = [
                            d["pest"] for d in st.session_state["current_detections"]
                        ]
                        context = f"\n\nContext: The farmer has detected these pests: {', '.join(pests)}"

                    response = gemini_model.generate_content(
                        f"You are an expert agricultural advisor. Answer this farmer's question: {user_question}{context}"
                    )
                    st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                    st.markdown(response.text)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab3:
        st.subheader("📈 Historical Analytics")

        if "history" in st.session_state and st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True)

            # Download report
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="📥 Download Report",
                data=csv,
                file_name=f"pest_detection_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "No detection history yet. Start analyzing images to build your history!"
            )

    with tab4:
        st.subheader("🌤️ Weather & Recommendations")

        if use_location:
            weather_data = get_weather_data(latitude, longitude)

            if weather_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Temperature",
                        f"{weather_data.get('main', {}).get('temp', 'N/A')}°C",
                    )
                with col2:
                    st.metric(
                        "Humidity",
                        f"{weather_data.get('main', {}).get('humidity', 'N/A')}%",
                    )
                with col3:
                    st.metric(
                        "Conditions",
                        weather_data.get("weather", [{}])[0].get("main", "N/A"),
                    )

                # Weather-based recommendations
                st.markdown("### 💡 Smart Recommendations")
                temp = weather_data.get("main", {}).get("temp", 25)
                humidity = weather_data.get("main", {}).get("humidity", 60)

                if temp > 30 and humidity > 70:
                    st.warning(
                        "⚠️ High temperature and humidity - Increased pest risk! Monitor crops closely."
                    )
                elif temp < 15:
                    st.info(
                        "❄️ Low temperature - Reduced pest activity. Good time for preventive measures."
                    )
                else:
                    st.success(
                        "✅ Weather conditions are moderate. Regular monitoring recommended."
                    )
            else:
                st.error(
                    "Unable to fetch weather data. Please check your API key and internet connection."
                )
        else:
            st.info(
                "Enable weather integration in settings to see real-time weather data and recommendations."
            )

    with tab5:
        st.subheader("📚 Pest Information Library")

        search_pest = st.text_input(
            "🔍 Search for a pest", placeholder="e.g., aphids, rice borer"
        )

        if search_pest:
            matching_pests = [
                p for p in class_names if search_pest.lower() in p.lower()
            ]

            if matching_pests:
                for pest in matching_pests[:5]:
                    with st.expander(f"🐛 {pest.title()}"):
                        treatment = get_treatment_info(pest)
                        st.markdown(f"**Severity Level:** {treatment['severity']}")
                        st.markdown(f"**Treatment:** {treatment['treatment']}")
                        st.markdown(f"**Organic Option:** {treatment['organic']}")
                        st.markdown(f"**Prevention:** {treatment['prevention']}")
            else:
                st.warning("No matching pests found. Try a different search term.")
        else:
            st.info(
                "Enter a pest name to search for information and treatment recommendations."
            )

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🌾 **Smart Crop Technologies**")
    with col2:
        st.markdown("📞 **Support:** 1800-XXX-XXXX")
    with col3:
        st.markdown(f"📅 **Date:** {datetime.now().strftime('%d %B %Y')}")


if __name__ == "__main__":
    main()
