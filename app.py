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
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS for Black Sidebar
# =========================
st.markdown("""
<style>
/* Sidebar background black */
section[data-testid="stSidebar"] {
    background-color: #000000; /* Black sidebar */
    color: #ffffff;
}

/* Sidebar radio button labels */
section[data-testid="stSidebar"] .stRadio label {
    color: #ffffff !important; /* White text */
    font-weight: bold;
    font-size: 16px;
}

/* Force all sidebar text white */
section[data-testid="stSidebar"] .stRadio div[role='radiogroup'] label p {
    color: #ffffff !important;
}

/* Sidebar radio button hover effect */
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #2563eb !important; /* Hover color */
    cursor: pointer;
}

/* Highlight "Model Prediction" on hover with rocket */
.stRadio label[for^="Model Prediction"]:hover::after {
    content: " 🚀";
}

/* Breed card */
.breed-card {
    border-radius: 12px;
    padding: 12px;
    background-color: #f0f0f0;
    margin-bottom: 10px;
    overflow:auto;
}

/* Breed colored box */
.breed-box {
    display:inline-block;
    padding:10px 15px;
    margin:5px;
    border-radius:8px;
    color:white;
    font-weight:bold;
}
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
# Load Model silently
# =========================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Safe loading
    except Exception:
        model = None
else:
    model = None

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

def display_breed_card(breed, prob):
    info = breed_info.get(breed, {})
    border_color = "#28a745" if prob>0.7 else "#ffc107" if prob>0.5 else "#dc3545"
    title_color = border_color
    bg_color = "#ffffff"
    text_color = "#000000"

    st.markdown(f"""
    <div style="
        border:2px solid {border_color};
        padding:15px;
        border-radius:12px;
        margin-bottom:10px;
        background-color:{bg_color};
        color:{text_color};
        word-wrap: break-word;
        overflow:auto;
        max-height: 220px;">
        <h3 style="color:{title_color}; margin-bottom:10px;">{breed} - {prob*100:.2f}%</h3>
        <b>Type:</b> {info.get('Type','N/A')}<br>
        <b>Origin:</b> {info.get('Origin','N/A')}<br>
        <b>Description:</b> {info.get('Description','N/A')}<br>
    </div>
    """, unsafe_allow_html=True)

def breed_box(breed, color="#2563eb"):
    st.markdown(f"""
        <div class="breed-box" style="background-color:{color}">{breed}</div>
    """, unsafe_allow_html=True)

# =========================
# Sidebar Navigation
# =========================
menu = ["Home", "About", "Model Prediction"]
choice = st.sidebar.radio("Navigation", menu)

# =========================
# Home Page
# =========================
if choice == "Home":
    st.title("🐄 Indian Cattle & Buffalo Breed Identifier")
    st.markdown("""
    Welcome! This app helps Field Level Workers (FLWs) identify **Indian cattle and buffalo breeds**.  
    Upload an image in the **Model Prediction** tab to get the top-3 breed predictions with details.
    """)
    st.image(
        "https://play-lh.googleusercontent.com/3QdX1hXthh-8mlOSIKHX-5enC9Ml0exx2aWHOdKiagUXMrQfL8VDEzQPPnTjJvsSvg",
        use_container_width=True
    )

# =========================
# About Page
# =========================
elif choice == "About":
    st.title("ℹ️ About Breeds")
    st.markdown("### 🐂 Cattle Breeds")
    cattle_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "cattle"]
    for breed in cattle_breeds:
        breed_box(breed, color="#28a745")  # green boxes
    
    st.markdown("### 🐃 Buffalo Breeds")
    buffalo_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "buffalo"]
    for breed in buffalo_breeds:
        breed_box(breed, color="#2563eb")  # blue boxes

# =========================
# Model Prediction
# =========================
elif choice == "Model Prediction":
    st.title("🔍 Predict Breed")
    st.markdown("Upload an image or use your camera to predict the breed.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    with col2:
        camera_input = st.camera_input("Capture image")

    img_file = uploaded_file if uploaded_file else camera_input

    if img_file and model:
        st.image(img_file, caption="Input Image", use_container_width=True)
        if st.button("🚀 Predict"):
            with st.spinner("Predicting..."):
                results = predict_top3(img_file)
                st.subheader("Top 3 Predictions")
                cols = st.columns(3)
                for i, (breed, prob) in enumerate(results):
                    with cols[i]:
                        display_breed_card(breed, prob)
    elif img_file and not model:
        st.warning("⚠️ Model not loaded. Cannot predict.")
