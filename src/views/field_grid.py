import streamlit as st
from PIL import Image
from utils import draw_fake_detections, draw_tf_detections, draw_real_detections

def display_title_and_description():
    """Displays the title and description for the grid page."""
    st.title("Field Disease Detection Grid")
    st.write("Upload exactly 64 images from your cornfield. The results will be displayed in a 8x8 grid indicating simulated disease detection.")

def initialize_session_state():
    """Initializes the necessary session state variables for the grid."""
    if "uploaded_images_grid" not in st.session_state:
        st.session_state["uploaded_images_grid"] = []
    if "detection_results_grid" not in st.session_state:
        st.session_state["detection_results_grid"] = [None] * 64  # Initialize for 64 images
    if "grid_model_choice" not in st.session_state:
        st.session_state["grid_model_choice"] = "TensorFlow"

def handle_image_upload(num_images_expected):
    """Handles the upload of images and validates the number, also updates session state."""
    uploaded_files = st.file_uploader(f"Upload {num_images_expected} Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        if len(uploaded_files) != num_images_expected:
            st.warning(f"Please upload exactly {num_images_expected} images to fill the grid.")
            return False
        else:
            st.session_state["uploaded_images_grid"] = []
            st.session_state["detection_results_grid"] = [None] * num_images_expected # Reset results on new upload
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                st.session_state["uploaded_images_grid"].append(image)
            return True
    return False

def select_grid_model():
    st.session_state["grid_model_choice"] = st.radio(
        "Choose model for detection:",
        ("YOLO", "TensorFlow"),
        index=1,
        horizontal=True,
        key="grid_model_radio"
    )

def process_images_and_get_results():
    """Processes the uploaded images and runs detection, caching results."""
    images = st.session_state.get("uploaded_images_grid", [])
    detection_results = st.session_state.get("detection_results_grid", [])
    num_images = len(images)
    model_choice = st.session_state.get("grid_model_choice", "TensorFlow")

    if num_images > 0 and (any(result is None for result in detection_results)):
        for i in range(num_images):
            if detection_results[i] is None:
                if model_choice == "YOLO":
                    _, covered_area = draw_real_detections(images[i].copy())
                else:
                    _, covered_area = draw_tf_detections(images[i].copy())
                detection_results[i] = covered_area
        st.session_state["detection_results_grid"] = detection_results
    return detection_results

def display_detection_grid(detection_results, grid_size):
    """Displays the detection results in a grid format using colored rectangles with a fading transition."""
    st.subheader("Detection Results Grid:")
    cols = st.columns(grid_size)
    result_index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            with cols[col]:
                if result_index < len(detection_results):
                    coverd_area = detection_results[result_index]
                    coverd_area = coverd_area/100
                    coverd_area = min(0.25, coverd_area)  # Cap the covered area to a maximum of 0.25
                    coverd_area = coverd_area * 4  # Scale to a range of 0-1 for color mapping
                    coverd_area = max(0.0, min(1.0, coverd_area)) # Ensure value is within 0-1

                    red = int(255 * coverd_area)
                    green = int(255 * (1 - coverd_area))
                    blue = 0

                    color = f"rgb({red}, {green}, {blue})"

                    st.markdown(
                        f'<div style="width: 50px; height: 50px; background-color: {color}; border: 1px solid black;"></div>',
                        unsafe_allow_html=True
                    )
                    result_index += 1
                else:
                    st.empty()

def display_upload_instructions(num_images_expected):
    """Displays the initial instructions to upload images."""
    st.info(f"Please upload exactly {num_images_expected} images to populate the grid.")

def run():
    """Main function to run the grid display application."""
    grid_size = 8
    num_images_expected = grid_size * grid_size

    display_title_and_description()
    initialize_session_state()
    select_grid_model()
    files_uploaded = handle_image_upload(num_images_expected)

    if st.session_state["uploaded_images_grid"]:
        detection_results = process_images_and_get_results()
        display_detection_grid(detection_results, grid_size)
    else:
        display_upload_instructions(num_images_expected)

if __name__ == "__main__":
    run()