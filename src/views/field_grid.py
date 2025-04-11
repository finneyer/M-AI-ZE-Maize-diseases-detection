import streamlit as st
from PIL import Image
from utils import draw_fake_detections

def display_title_and_description():
    """Displays the title and description for the grid page."""
    st.title("🌱 Field Disease Detection Grid (Testing)")
    st.write("Upload exactly 9 images from your cornfield. The results will be displayed in a 3x3 grid indicating simulated disease detection.")

def handle_image_upload(num_images_expected):
    """Handles the upload of images and validates the number."""
    uploaded_files = st.file_uploader(f"Upload {num_images_expected} Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) != num_images_expected:
            st.warning(f"Please upload exactly {num_images_expected} images to fill the grid.")
            return None
        return uploaded_files
    return None

def process_images_and_get_results(uploaded_files):
    """Processes the uploaded images and simulates disease detection."""
    detection_results = []
    images = []
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        images.append(image)
        # Simulate detection for each image
        draw_fake_detections(image.copy())
        # For simplicity, we'll consider any image that goes through detection as having a simulated disease
        detection_results.append(True) # Assuming disease is "detected" for every processed image
    return images, detection_results

def display_detection_grid(detection_results, grid_size):
    """Displays the detection results in a grid format using colored rectangles."""
    st.subheader("Detection Results Grid:")
    cols = st.columns(grid_size)
    result_index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            with cols[col]:
                if result_index < len(detection_results):
                    has_disease = detection_results[result_index]
                    color = "red" if has_disease else "green"
                    st.markdown(
                        f'<div style="width: 30px; height: 30px; background-color: {color}; border: 1px solid black;"></div>',
                        unsafe_allow_html=True
                    )
                    result_index += 1
                else:
                    # Handle cases where the number of results is less than grid size (shouldn't happen if input is correct)
                    st.empty()

def display_upload_instructions(num_images_expected):
    """Displays the initial instructions to upload images."""
    st.info(f"Please upload exactly {num_images_expected} images to populate the grid.")

def run():
    """Main function to run the grid display application."""
    grid_size = 3
    num_images_expected = grid_size * grid_size

    display_title_and_description()
    uploaded_files = handle_image_upload(num_images_expected)

    if uploaded_files:
        images, detection_results = process_images_and_get_results(uploaded_files)
        display_detection_grid(detection_results, grid_size)
    else:
        display_upload_instructions(num_images_expected)

if __name__ == "__main__":
    run()