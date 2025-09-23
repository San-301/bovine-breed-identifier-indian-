import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Bovine Breed Identifier", layout="wide")

# =========================
# Model & Data Paths
# =========================
MODEL_FILENAME = "breed_classifier_mobilenet.h5"  # Upload this file to repo root
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
BREED_JSON = os.path.join(os.getcwd(), "breeds.json")

# Load breed info
if os.path.exists(BREED_JSON):
    with open(BREED_JSON, "r") as f:
        breed_info = json.load(f)
    # FIX: ensure alphabetical order (matches TensorFlow training order)
    class_names = sorted(breed_info.keys())
else:
    st.error(f"Breed info JSON not found at {BREED_JSON}")
    breed_info = {}
    class_names = []

# Load model safely
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.warning(f"Model not found at {MODEL_PATH}. Please upload it to the repo.")

# =========================
# Helper Functions
# =========================
def predict_top3(img_file):
    """Predict top-3 breeds for an uploaded image."""
    img = image.load_img(img_file, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return [(class_names[i], float(preds[i])) for i in top3_idx]

def display_breed_card(breed, prob):
    """Display breed details in a styled card."""
    info = breed_info.get(breed, {})
    color = "#28a745" if prob>0.7 else "#ffc107" if prob>0.5 else "#dc3545"
    st.markdown(f"""
    <div style="border:2px solid {color}; padding:15px; border-radius:12px; margin-bottom:10px; background-color:#f9f9f9">
        <h3 style="color:{color}">{breed} - {prob*100:.2f}%</h3>
        <b>Type:</b> {info.get('Type','N/A')}<br>
        <b>Origin:</b> {info.get('Origin','N/A')}<br>
        <b>Description:</b> {info.get('Description','N/A')}<br>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Sidebar Navigation
# =========================
menu = ["Home", "Model Demo", "About"]
choice = st.sidebar.radio("Navigation", menu)

# =========================
# Home Page
# =========================
if choice == "Home":
    st.title("🐄 Indian Cattle & Buffalo Breed Identifier")
    st.markdown("Upload an image of cattle or buffalo and get the **top-3 breed predictions**.")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    
    if uploaded_file and model:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Breed"):
            with st.spinner("Predicting..."):
                results = predict_top3(uploaded_file)
                st.subheader("Top 3 Breed Predictions")
                cols = st.columns(3)
                for i, (breed, prob) in enumerate(results):
                    with cols[i]:
                        display_breed_card(breed, prob)

# =========================
# Model Demo Page
# =========================
elif choice == "Model Demo":
    st.title("📊 Model Demo")
    st.markdown("Upload an image and click **Predict** to see top-3 predictions with confidence percentages.")
    
    demo_file = st.file_uploader("Upload an image for demo", type=["jpg","jpeg","png"], key="demo")
    
    if demo_file and model:
        st.image(demo_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict on Demo Image"):
            with st.spinner("Predicting..."):
                results = predict_top3(demo_file)
                st.subheader("Top 3 Breed Predictions")
                for breed, prob in results:
                    st.write(f"**{breed}** : {prob*100:.2f}% confidence")
                    display_breed_card(breed, prob)

# =========================
# About Page
# =========================
elif choice == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **Purpose:** Help Field Level Workers (FLWs) identify Indian cattle and buffalo breeds using AI.  

    **Dataset:** Includes 10 major Indian breeds: 5 cattle and 5 buffalo.  

    **Model:** MobileNetV2 fine-tuned for up to 50 epochs on training data.  

    **Hackathon:** Demo submission showcasing breed recognition and real-time predictions.  

    **How to use:**  
    1. Go to **Home** → upload an image → click **Predict Breed**  
    2. Or go to **Model Demo** → upload image → click **Predict on Demo Image**  
    3. View **top-3 predictions** with breed details
    """)
