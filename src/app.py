import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import random
import io

st.set_page_config(page_title="M-AI-ZE", layout="centered")
st.title("M-AI-ZE 🌽 Corn Field Disease Detection ")
st.write("Upload a cornfield image, and we'll simulate detecting diseased areas.")

# Upload image
uploaded_file = st.file_uploader("Choose a cornfield image...", type=["jpg", "jpeg", "png"])

def draw_fake_detections(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(random.randint(2, 6)):  # Simulate 2-6 disease spots
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        x2 = x1 + random.randint(30, 100)
        y2 = y1 + random.randint(30, 100)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), "Disease", fill="red")
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    st.write("🔍 Simulating disease detection...")
    detected_image = image.copy()
    detected_image = draw_fake_detections(detected_image)

    st.image(detected_image, caption="Detected Disease (Mock)", use_column_width=True)
else:
    st.info("Upload a cornfield image to start the mock detection.")
