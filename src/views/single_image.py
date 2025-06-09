import streamlit as st
from PIL import Image
from utils import draw_real_detections, draw_tf_detections
import uuid

def display_title_and_description():
    st.title("Single Image Detection")
    st.write("Upload a cornfield image and we’ll simulate disease detection.")

def initialize_session_state():
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = None
    if "detection_result" not in st.session_state:
        st.session_state["detection_result"] = None
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "YOLO"

def handle_file_upload():
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_image"] = image
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["detection_result"] = None  

def display_results():
    # Model selection radio button
    st.session_state["model_choice"] = st.radio(
        "Choose model for detection:",
        ("YOLO", "TensorFlow"),
        index=0,
        horizontal=True,
    )

    if st.session_state["uploaded_image"] is not None:
        st.image(st.session_state["uploaded_image"], caption=f"Original: {st.session_state['uploaded_filename']}", use_container_width=True)

        st.write("Running detection...")
        if st.session_state["detection_result"] is None:
            if st.session_state["model_choice"] == "YOLO":
                st.session_state["detection_result"], _ = draw_real_detections(st.session_state["uploaded_image"].copy())
            else:
                st.session_state["detection_result"], _ = draw_tf_detections(st.session_state["uploaded_image"].copy())

        st.image(st.session_state["detection_result"], caption=f"Detected Disease ({st.session_state['model_choice']})", use_container_width=True)
    else:
        st.info("Please upload an image.")

def run():
    display_title_and_description()
    initialize_session_state()
    handle_file_upload()
    display_results()

if __name__ == "__main__":
    run()