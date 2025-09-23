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
# Paths
# =========================
MODEL_FILENAME = "breed_classifier_mobilenet.h5"  # Upload this to repo root
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
    st.warning(f"Model not found at {MODEL_PATH}. Please upload it to the repo.")

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
    """Display breed details in a styled card with visible text."""
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
    st.image("https://www.google.com/imgres?q=bharat%20pashudhan%20app%20image&imgurl=https%3A%2F%2Fplay-lh.googleusercontent.com%2F3QdX1hXthh-8mlOSIKHX-5enC9Ml0exx2aWHOdKiagUXMrQfL8VDEzQPPnTjJvsSvg&imgrefurl=https%3A%2F%2Fplay.google.com%2Fstore%2Fapps%2Fdetails%3Fid%3Dcom.epashu.in&docid=nR_zlYug1jwyFM&tbnid=IfzwMgILouRwIM&vet=12ahUKEwiOk6f7wO6PAxW_1jgGHZl-CZIQM3oECB0QAA..i&w=512&h=512&hcb=2&ved=2ahUKEwiOk6f7wO6PAxW_1jgGHZl-CZIQM3oECB0QAA", use_column_width=True)

# =========================
# About Page
# =========================
elif choice == "About":
    st.title("ℹ️ About Breeds")
    st.markdown("### 🐂 Cattle Breeds")
    cattle_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "cattle"]
    st.write(", ".join(cattle_breeds))
    
    st.markdown("### 🐃 Buffalo Breeds")
    buffalo_breeds = [k for k, v in breed_info.items() if v["Type"].lower() == "buffalo"]
    st.write(", ".join(buffalo_breeds))

# =========================
# Model Prediction
# =========================
elif choice == "Model Prediction":
    st.title("🔍 Predict Breed")
    st.markdown("Upload an image or take a photo of a cow or buffalo to get top-3 breed predictions.")
    
    # Camera Input
    captured_file = st.camera_input("📷 Take a picture")
    
    # File Upload
    uploaded_file = st.file_uploader("📂 Or upload an image", type=["jpg","jpeg","png"])
    
    # Use whichever is available
    img_source = captured_file if captured_file else uploaded_file
    
    if img_source and model:
        st.image(img_source, caption="Input Image", use_column_width=True)
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                results = predict_top3(img_source)
                st.subheader("Top 3 Predictions")
                cols = st.columns(3)
                for i, (breed, prob) in enumerate(results):
                    with cols[i]:
                        display_breed_card(breed, prob)
    elif img_source and not model:
        st.warning("⚠️ Model not loaded. Cannot predict.")

