import wandb
import tensorflow as tf
from PIL import ImageDraw, ImageFont
import streamlit as st
import numpy as np
from ultralytics import YOLO
import torch
import os
import keras

@st.cache_data
def draw_fake_detections(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.truetype("arial.ttf", 60) 
    number_of_detections = np.random.randint(1, 10)
    for _ in range(number_of_detections):
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 100)
        x2 = x1 + np.random.randint(125, width-x1)
        y2 = y1 + np.random.randint(125, height-y1)    
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.text((x1 + 5, y1 - 10), "Disease", fill="red", font=font)
    return image, number_of_detections



@st.cache_resource
def load_model_from_wandb():
    api = wandb.Api()
    artifact = api.artifact('rueedi-tobias-hochschule-luzern/maize_disease_detection/run_xws0uddk_model:v0', type='model')
    model_path = artifact.download()
    for file in os.listdir(model_path):
        if file.endswith('.pt'):
            pt_path = os.path.join(model_path, file)
            break
    else:
        raise FileNotFoundError("No .pt file found in artifact")
    model = YOLO(pt_path)
    return model

@st.cache_data
def draw_real_detections(image):
    model = load_model_from_wandb()
    results = model.predict(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    n = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 5, y1 - 10), f"{cls} ({conf:.2f})", fill="red", font=font)
            n += 1
    return image, n