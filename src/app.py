import streamlit as st
import pages.single_image as single_image
import pages.batch_field as batch_field

st.set_page_config(page_title="Corn Disease Detector", layout="centered")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📷 Single Image Detection", "🌍 Field (Batch) Detection"])

if page == "📷 Single Image Detection":
    single_image.run()
elif page == "🌍 Field (Batch) Detection":
    batch_field.run()
