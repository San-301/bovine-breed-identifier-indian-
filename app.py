import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Bovine Breed Identifier",
    page_icon=Image.open("0266aebc-4ce9-4154-bc04-62ef0462f8e8.png"),  # your logo
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
/* Global Dark Theme */
body, .stApp {
    background-color: #111827;
    color: #f9fafb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
    color: #f9fafb;
}
section[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}

/* Headings */
h1, h2, h3 {
    color: #f9fafb;
    font-weight: 700;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Breed Cards */
.breed-card {
    border-radius: 16px;
    padding: 20px;
    background: #1f2937;
    color: #f9fafb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin: 8px;
    height: 320px;
    overflow-y: auto;
    transition: transform 0.2s ease;
}
.breed-card:hover {
    transform: translateY(-4px);
}
.breed-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.4em;
}
.probability {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.8em;
}
/* Prediction colors */
.green-card { background-color: #065f46; } /* Dark green */
.yellow-card { background-color: #92400e; } /* Amber tone */
.red-card { background-color: #7f1d1d; } /* Dark red */
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
    model = tf.keras.models.load_model(MODEL_PATH)
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

def display_breed_cards(results):
    cols = st.columns(3)
    colors = ["green-card", "yellow-card", "red-card"]
    for idx, (breed, prob) in enumerate(results):
        info = breed_info.get(breed, {})
        with cols[idx]:
            st.markdown(f"""
            <div class="breed-card {colors[idx]}">
                <div class="breed-title">{breed}</div>
                <div class="probability">Confidence: {prob*100:.2f}%</div>
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
    st.image("0266aebc-4ce9-4154-bc04-62ef0462f8e8.png", width=80)  # Logo
    st.title("Indian Cattle & Buffalo Breed Identifier")
    st.markdown("""
    ### Empowering Field Workers  
    Capture or upload an image of cattle or buffalo and let our AI model identify the **top 3 most probable breeds** with details.  
    This tool is built to support **field-level workers, veterinarians, and farmers**.
    """)
    st.image("https://raw.githubusercontent.com/San-301/bovine-breed-identifier-indian-/main/images.png",
             use_container_width=True, caption="Supporting Indian Livestock Heritage")

# =========================
# About Page
# =========================
elif choice == "About":
    st.title("About Breeds")
    st.markdown("This application covers **Indian cattle and buffalo breeds** with key details for identification.")

    st.subheader("🐂 Cattle Breeds")
    cattle_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "cattle"]
    st.success(", ".join(cattle_breeds) if cattle_breeds else "No cattle breeds found in dataset.")

    st.subheader("🐃 Buffalo Breeds")
    buffalo_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "buffalo"]
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

    if img_source and model:
        st.image(img_source, caption="Input Image", use_container_width=True)

        if st.button("🚀 Predict"):
            with st.spinner("Analyzing image..."):
                results = predict_top3(img_source)
                st.subheader("✨ Top 3 Predictions")
                display_breed_cards(results)
