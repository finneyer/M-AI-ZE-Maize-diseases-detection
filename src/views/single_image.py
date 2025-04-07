import streamlit as st
from PIL import Image
import os
import uuid
from utils import draw_fake_detections

# Ensure the temp_uploads folder exists
TEMP_UPLOAD_DIR = "temp_uploads"
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR)

def run():
    st.title("📷 Single Image Detection")
    st.write("Upload a cornfield image and we’ll simulate disease detection.")

    # Initialize session state for the uploaded image and detection result
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = None
    if "detection_result" not in st.session_state:
        st.session_state["detection_result"] = None

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="file_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_image"] = image
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["detection_result"] = None  # Reset detection result on new upload

        # Save the uploaded image to the temp_uploads folder
        unique_filename = f"{uuid.uuid4().hex}.{uploaded_file.type.split('/')[-1]}"
        filepath = os.path.join(TEMP_UPLOAD_DIR, unique_filename)
        image.save(filepath)
        st.success(f"Image saved to {filepath}")

    # Display the uploaded image if it exists in session state
    if st.session_state["uploaded_image"] is not None:
        st.image(st.session_state["uploaded_image"], caption=f"Original: {st.session_state['uploaded_filename']}", use_container_width=True)

        st.write("🔍 Simulating detection...")
        # Only run detection if the result is not already cached in session state
        if st.session_state["detection_result"] is None:
            st.session_state["detection_result"] = draw_fake_detections(st.session_state["uploaded_image"].copy())

        st.image(st.session_state["detection_result"], caption="Detected Disease (Mock)", use_container_width=True)
    elif not uploaded_file:
        st.info("Please upload an image.")
