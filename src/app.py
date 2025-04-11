import streamlit as st
import views.single_image as single_image
import views.batch_field as batch_field
import views.field_grid as field_grid

st.set_page_config(page_title="Corn Disease Detector", layout="centered")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📷 Single Image Detection", "🌍 Field (Batch) Detection", "🌱 Field Disease Detection"])

if page == "📷 Single Image Detection":
    single_image.run()
elif page == "🌍 Field (Batch) Detection":
    batch_field.run()
elif page == "🌱 Field Disease Detection":
    field_grid.run()
