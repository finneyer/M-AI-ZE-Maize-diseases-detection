import streamlit as st
from PIL import Image
from utils import draw_fake_detections

def run():
    st.title("🌍 Field (Batch) Detection")
    st.write("Upload up to 10 images from your cornfield to simulate disease detection.")

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Please upload 10 images or fewer.")
        else:
            for i, file in enumerate(uploaded_files):
                image = Image.open(file).convert("RGB")
                st.subheader(f"Image {i + 1}")
                st.image(image, caption="Original", use_container_width=True)

                st.write("🔍 Simulating detection...")
                result = draw_fake_detections(image.copy())
                st.image(result, caption="Detected Disease (Mock)", use_container_width=True)
    else:
        st.info("Upload cornfield images to begin.")
