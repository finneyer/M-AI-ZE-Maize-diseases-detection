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

@st.cache_resource
def load_tf_model_from_wandb():
    api = wandb.init()
    artifact = api.use_artifact('rueedi-tobias-hochschule-luzern/Own CNN/cnn-model:v3', type='model')
    model_path = artifact.download()
    # Look for .h5 or .keras file
    for file in os.listdir(model_path):
        if file.endswith('.h5') or file.endswith('.keras'):
            tf_path = os.path.join(model_path, file)
            break
    else:
        raise FileNotFoundError("No .h5 or .keras file found in artifact")
    model = tf.keras.models.load_model(tf_path, compile=False)
    return model

@st.cache_data
def draw_tf_detections(image):
    model = load_tf_model_from_wandb()

    # Preprocess image
    img_resized = image.resize((640, 640))  # Must match training input shape
    img_array = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(input_tensor)[0]  # shape: (grid_y, grid_x, B, 6)

    # Apply NMS (you already have this function)
    nms_boxes, scores = apply_nms(pred, conf_thresh=0.3, iou_thresh=0.4)

    # Draw on original image size (not resized)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    w, h = image.size
    n = 0

    for box in nms_boxes:
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 5, y1 - 10), "Pred", fill="red", font=font)
        n += 1

    return image, n


def apply_nms(pred_grid, conf_thresh=0.3, iou_thresh=0.4):
    boxes = []
    scores = []

    for gy in range(20):
        for gx in range(20):
            for b in range(3):
                pred = pred_grid[gy, gx, b]
                conf = pred[4]
                if conf < conf_thresh:
                    continue

                x, y, w, h = pred[:4]
                cx = (gx + x) / 20
                cy = (gy + y) / 20
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2

                boxes.append([ymin, xmin, ymax, xmax])
                scores.append(conf)

    if not boxes:
        return [], []

    boxes = tf.constant(boxes, dtype=tf.float32)
    scores = tf.constant(scores, dtype=tf.float32)

    selected_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=10,
        iou_threshold=iou_thresh,
        score_threshold=conf_thresh
    )

    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()

    return selected_boxes, selected_scores

