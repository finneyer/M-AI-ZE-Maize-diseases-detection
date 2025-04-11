import streamlit as st
from PIL import Image
from utils import draw_fake_detections # Assuming cache_single_image_detection is meant for the single image script

def display_title_and_description():
    st.title("🌍 Field (Batch) Detection")
    st.write("Upload up to 10 images from your cornfield to simulate disease detection and get an infection percentage.")

def initialize_session_state():
    if "uploaded_images_batch" not in st.session_state:
        st.session_state["uploaded_images_batch"] = []
    if "uploaded_filenames_batch" not in st.session_state:
        st.session_state["uploaded_filenames_batch"] = []
    if "detection_results_batch" not in st.session_state:
        st.session_state["detection_results_batch"] = [None] * 10  # Initialize for up to 10 images

def handle_file_upload():
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Please upload 10 images or fewer.")
            return False
        else:
            # Clear previous uploads if new ones are provided
            if uploaded_files and (not st.session_state["uploaded_images_batch"] or len(st.session_state["uploaded_images_batch"]) != len(uploaded_files)):
                st.session_state["uploaded_images_batch"] = []
                st.session_state["uploaded_filenames_batch"] = []
                st.session_state["detection_results_batch"] = [None] * 10 # Re-initialize detection results

                for file in uploaded_files:
                    image = Image.open(file).convert("RGB")
                    st.session_state["uploaded_images_batch"].append(image)
                    st.session_state["uploaded_filenames_batch"].append(file.name)
            return True
    return False

def process_and_display_images():
    num_uploaded = len(st.session_state["uploaded_images_batch"])
    if num_uploaded > 0:
        infected_count = 0
        st.subheader("Processing Images:")
        for i in range(num_uploaded):
            with st.expander(f"Image {i + 1}: {st.session_state['uploaded_filenames_batch'][i]}"):
                image = st.session_state["uploaded_images_batch"][i]
                st.image(image, caption="Original", use_container_width=True)

                st.write("🔍 Simulating detection...")
                if st.session_state["detection_results_batch"][i] is None:
                    st.session_state["detection_results_batch"][i] , number_of_infects= draw_fake_detections(image.copy())

                st.image(st.session_state["detection_results_batch"][i], caption="Detected Disease (Mock)", use_container_width=True)
                infected_count += number_of_infects
        return num_uploaded, infected_count
    return 0, 0

def display_overall_summary(num_uploaded, infected_count):
    if num_uploaded > 0:
        infection_percentage = (infected_count / (num_uploaded*10)) * 100
        st.subheader("Overall Infection Summary")
        st.write(f"Number of images uploaded: {num_uploaded}")
        st.write(f"Number of images with simulated detection: {infected_count}")
        st.metric("Percentage of images with simulated disease", f"{infection_percentage:.2f}%")
    else:
        st.info("Upload cornfield images to begin.")

def run():
    display_title_and_description()
    initialize_session_state()

    files_uploaded = handle_file_upload()

    if files_uploaded or st.session_state["uploaded_images_batch"]:
        num_uploaded, infected_count = process_and_display_images()
        display_overall_summary(num_uploaded, infected_count)
    else:
        st.info("Upload cornfield images to begin.")

if __name__ == "__main__":
    run()