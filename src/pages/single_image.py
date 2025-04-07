import streamlit as st
from PIL import Image
from utils import draw_fake_detections

def run():
    st.title("📷 Single Image Detection")
    st.write("Upload a cornfield image and we’ll simulate disease detection.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original", use_container_width=True)

        st.write("🔍 Simulating detection...")
        result = draw_fake_detections(image.copy())
        st.image(result, caption="Detected Disease (Mock)", use_container_width=True)
    else:
        st.info("Please upload an image.")
