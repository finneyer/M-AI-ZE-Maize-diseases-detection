import streamlit as st
from PIL import Image
from utils import draw_fake_detections

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

def process_images_and_get_results():
    """Processes the uploaded images and simulates disease detection, caching results."""
    images = st.session_state.get("uploaded_images_grid", [])
    detection_results = st.session_state.get("detection_results_grid", [])
    num_images = len(images)

    if num_images > 0 and (any(result is None for result in detection_results)):
        for i in range(num_images):
            if detection_results[i] is None:
                # Simulate detection for each image
                _, num_detections = draw_fake_detections(images[i].copy())
                detection_results[i] = num_detections
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
                    num_detections = detection_results[result_index]
                    normalized_value = num_detections / 10.0 if num_detections is not None else 0.0
                    normalized_value = max(0.0, min(1.0, normalized_value)) # Ensure value is within 0-1

                    red = int(255 * normalized_value)
                    green = int(255 * (1 - normalized_value))
                    blue = 0

                    color = f"rgb({red}, {green}, {blue})"

                    st.markdown(
                        f'<div style="width: 30px; height: 30px; background-color: {color}; border: 1px solid black;"></div>',
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
    files_uploaded = handle_image_upload(num_images_expected)

    if st.session_state["uploaded_images_grid"]:
        detection_results = process_images_and_get_results()
        display_detection_grid(detection_results, grid_size)
    else:
        display_upload_instructions(num_images_expected)

if __name__ == "__main__":
    run()