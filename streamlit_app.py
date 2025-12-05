import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load saved model
model = load_model("/Users/joodmagedwageh/Downloads/my_mobnet_model.h5")

# Define classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("Garbage Classification App")
st.write("Upload an image and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    
    # Predict
    pred = model.predict(x)
    class_idx = np.argmax(pred, axis=1)[0]
    st.write(f"Predicted class: **{classes[class_idx]}**")
