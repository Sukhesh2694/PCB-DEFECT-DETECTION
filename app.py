import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# Load the YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('yolov5', 'custom', path='yolov5/models/best.pt', source='local')  # Load custom model
    return model

model = load_model()

# Set custom page configuration with an icon
st.markdown("""
    <style>
        /* Background color */
        .main {
            background-color: #11121c;
            padding: 20px;
        }
        
        /* Title styling */
        .title h1 {
            color: #e75480;
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5em;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #e0e0e0;
            padding: 15px;
        }
        
        /* Uploaded and detected image styling */
        .uploaded-image, .detected-image {
            border-radius: 10px;
            border: 2px solid #e75480;
            margin-top: 20px;
            max-width: 100%;
        }

        /* Confidence score styling */
        .confidence {
            color: #4CAF50;
            font-weight: bold;
        }
        
    </style>
""", unsafe_allow_html=True)

# Main page title with custom color and font size
st.markdown("<div class='title'><h1>🌟 PCB Defect Detection 🌟</h1></div>", unsafe_allow_html=True)
st.write("Upload an image of defected PCB to detect defects or errors using our trained ML model. This model will analyze and identify the defects in the PCB.")

# Sidebar for additional information
st.sidebar.image("https://placekitten.com/120/120", width=100)  # Replace with an actual logo or icon URL
st.sidebar.title("About")
st.sidebar.write("This app uses a custom-trained YOLOv5 model for object detection.")
st.sidebar.write("Upload an image of defected PCB to detect defects or errors using our trained ML model. This model will analyze and identify the defects in the PCB")
st.sidebar.markdown("Developed by the Sukhesh and Team ⚡️")
st.sidebar.markdown("---")

# Upload section
st.markdown("### 📤 Upload an Image")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Perform detection
    results = model(image)
    results.render()  # Updates results with bounding box annotations

    # Display output
    annotated_image = Image.fromarray(results.ims[0])
    st.image(annotated_image, caption='Detected Image', use_column_width=True)

    # Optionally, display confidence scores and labels
    for i, (label, conf) in enumerate(zip(results.pandas().xyxy[0]['name'], results.pandas().xyxy[0]['confidence'])):
        st.write(f"{i + 1}: {label} (Confidence: {conf:.2f})")
