import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Bovine Breed Identifier",
    page_icon="https://raw.githubusercontent.com/San-301/bovine-breed-identifier-indian-/main/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
/* Global */
body { font-family: "Inter", sans-serif; background-color: #f9fafb; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }

/* Titles */
h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: 600;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover { background-color: #1d4ed8; }

/* Breed Cards */
.breed-card {
    border-radius: 16px;
    padding: 16px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    transition: transform 0.2s ease;
    overflow-y: auto;
}
.breed-card:hover { transform: translateY(-4px); }

.breed-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5em; color: #111827; }
.probability { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.6em; padding: 4px 8px; border-radius: 8px; display: inline-block; color: white; }
</style>
""", unsafe_allow_html=True)

# =========================
# Paths
# =========================
MODEL_FILENAME = "breed_classifier_mobilenet.h5"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
BREED_JSON = os.path.join(os.getcwd(), "breeds.json")

# =========================
# Load Breed Info
# =========================
if os.path.exists(BREED_JSON):
    with open(BREED_JSON, "r") as f:
        breed_info = json.load(f)
    class_names = sorted(breed_info.keys())
else:
    st.error(f"Breed info JSON not found at {BREED_JSON}")
    breed_info = {}
    class_names = []

# =========================
# Load Model
# =========================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning(f"⚠️ Model not found at {MODEL_PATH}. Please upload it to the repo.")

# =========================
# Helper Functions
# =========================
def predict_top3(img_file):
    img = image.load_img(img_file, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return [(class_names[i], float(preds[i])) for i in top3_idx]

def get_color(prob):
    if prob >= 0.7:
        return "#16a34a"  # green
    elif prob >= 0.4:
        return "#facc15"  # yellow
    else:
        return "#dc2626"  # red

def display_breed_card(breed, prob):
    info = breed_info.get(breed, {})
    color = get_color(prob)
    st.markdown(f"""
    <div class="breed-card">
        <div class="breed-title">{breed}</div>
        <div class="probability" style="background-color:{color}">Confidence: {prob*100:.2f}%</div>
        <p><b>Type:</b> {info.get('Type','N/A')}<br>
        <b>Origin:</b> {info.get('Origin','N/A')}<br>
        <b>Description:</b> {info.get('Description','N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("🐮 Navigation")
menu = ["Home", "About", "Model Prediction"]
choice = st.sidebar.radio("", menu)

# =========================
# Home Page
# =========================
if choice == "Home":
    st.title("🐄 Indian Cattle & Buffalo Breed Identifier")
    st.markdown("""
    ### Empowering Field Workers  
    Capture or upload an image of cattle or buffalo and let our AI model identify the **top 3 most probable breeds** with details.  
    This tool is built to support **field-level workers, veterinarians, and farmers**.
    """)
    st.image("https://raw.githubusercontent.com/San-301/bovine-breed-identifier-indian-/main/images.png",
             use_column_width=True, caption="Supporting Indian Livestock Heritage")

# =========================
# About Page
# =========================
elif choice == "About":
    st.title("ℹ️ About Breeds")
    st.markdown("This application covers **Indian cattle and buffalo breeds** with key details for identification.")

    st.markdown("### 🐂 Cattle Breeds")
    cattle_breeds = [k for k, v in breed_info.items() if v.get("Type","").lower() == "cattle"]
    st.success(", ".join(cattle_breeds) if cattle_breeds else "No cattle breeds found in dataset.")

    st.markdown("### 🐃 Buffalo Breeds")
    buffalo_breeds = [k for k, v in breed_info.items() if v.get("Type","").lower() == "buffalo"]
    st.info(", ".join(buffalo_breeds) if buffalo_breeds else "No buffalo breeds found in dataset.")

# =========================
# Model Prediction
# =========================
elif choice == "Model Prediction":
    st.title("🔍 Predict Breed")
    st.markdown("Capture a photo of a **cow or buffalo** or upload an image to get predictions.")

    # Camera Input
    captured_file = st.camera_input("📷 Take a picture")

    # File Upload
    uploaded_file = st.file_uploader("📂 Or upload an image", type=["jpg","jpeg","png"])

    # Pick whichever source is available
    img_source = captured_file if captured_file else uploaded_file

    if img_source:
        st.image(img_source, caption="Input Image", use_column_width=True)

        if model and st.button("🚀 Predict"):
            with st.spinner("Analyzing image..."):
                results = predict_top3(img_source)
                st.subheader("✨ Top 3 Predictions")

                # Display side by side
                cols = st.columns(len(results))
                for col, (breed, prob) in zip(cols, results):
                    with col:
                        display_breed_card(breed, prob)
        elif not model:
            st.warning("⚠️ Model not loaded. Cannot predict.")
