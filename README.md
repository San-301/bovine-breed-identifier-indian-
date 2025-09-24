# 🐄 Indian Cattle & Buffalo Breed Identifier

This tool helps **Field Level Workers (FLWs)** identify Indian cattle and buffalo breeds using Artificial Intelligence (AI).  
The system uses a **MobileNet model (TensorFlow)** and is deployed as an interactive web app with **Streamlit**.  

---

## ✨ Features
- 📸 Upload cattle or buffalo images  
- 🎯 Top-3 breed predictions with probabilities  
- 📖 Breed info card (Type, Origin, Description)  
- 📱 Mobile-friendly design with color-coded confidence levels  

---

## 🚀 Live Demo
👉 [Click here to try the app](https://breeddetection.streamlit.app/)  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **ML Framework:** TensorFlow  
- **Model:** MobileNet (image classification)  
- **Frontend & Deployment:** Streamlit  

---

## 📊 How It Works
1. User uploads an image of cattle/buffalo  
2. The trained MobileNet model analyzes the image  
3. Top-3 predictions with confidence scores are displayed  
4. Additional breed details are shown in an info card  

---

## ⚙️ How to Run Locally
```bash
# Clone this repository
git clone https://github.com/San-301/bovine-breed-identifier-indian-.git
cd bovine-breed-identifier-indian-

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
