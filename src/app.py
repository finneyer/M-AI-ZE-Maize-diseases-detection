import streamlit as st
import views.single_image as single_image
import views.batch_field as batch_field
import views.field_grid as field_grid
from utils import load_model_from_wandb
import wandb

st.set_page_config(page_title="Corn Disease Detector", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Image Detection", "Batch Detection", "Field Disease Detection"])

wandb.login(key="017feefe0af6702cda76aab121ec71cf3a362fec")
load_model_from_wandb()

if page == "Single Image Detection":
    single_image.run()
elif page == "Batch Detection":
    batch_field.run()
elif page == "Field Disease Detection":
    field_grid.run()
