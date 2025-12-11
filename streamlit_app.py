import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =======================
# Load model
# =======================
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

model = load_trained_model("/Users/joodmagedwageh/Downloads/my_mobnet_model.h5")

# =======================
# Class labels
# =======================
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# =======================
# Streamlit App
# =======================
st.set_page_config(page_title="Smart Recycling Assistant", layout="centered")

st.markdown("<h1 style='text-align: center; color: green;'>♻️ Smart Recycling Assistant ♻️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of waste and see the classification results!</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    
    # Show uploaded image in center
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    
    # Predict
    pred = model.predict(x)
    class_idx = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    # Display results with styling
    st.markdown(f"""
        <div style='background-color:#e0f7fa;padding:15px;border-radius:10px;margin-top:10px'>
        <h3 style='color:#00796b;text-align:center;'>Predicted Class: {classes[class_idx]}</h3>
        <h4 style='text-align:center;'>Confidence: {confidence:.2f}</h4>
        </div>
    """, unsafe_allow_html=True)

    # Optional: show horizontal progress bar for confidence
    st.progress(confidence)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:14px;'>Developed by Your Name | AI Waste Classification</p>", unsafe_allow_html=True)
