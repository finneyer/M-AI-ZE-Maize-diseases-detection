import wandb
import tensorflow as tf
from PIL import ImageDraw, ImageFont
import streamlit as st
import numpy as np
from ultralytics import YOLO
import torch
import os
import keras
from PIL import Image, ImageDraw, ImageFont

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
    artifact = api.artifact('rueedi-tobias-hochschule-luzern/V1_2-maize_disease_detection_train/run_im166j19_model:v0', type='model')
    model_path = artifact.download()
    for file in os.listdir(model_path):
        if file.endswith('.pt'):
            pt_path = os.path.join(model_path, file)
            break
    else:
        raise FileNotFoundError("No .pt file found in artifact")
    model = YOLO(pt_path)
    model.to('cpu') 
    return model

@st.cache_data
def draw_real_detections(image):
    model = load_model_from_wandb()
    results = model.predict(image)
    # Prepare overlay for coloring the detected areas
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", 30)
    n = 0
    all_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Fill the detected area with a semi-transparent red
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, int(255 * 0.4)))
            # Optionally, draw the label text
            draw.text((x1 + 5, y1 - 10), f"{cls} ({conf:.2f})", fill="red", font=font)
            all_boxes.append([y1 / image.size[1], x1 / image.size[0], y2 / image.size[1], x2 / image.size[0]])
            n += 1

    # Composite overlay onto original image
    combined = Image.alpha_composite(image_rgba, overlay).convert("RGB")
    # Calculate area percentage using normalized boxes
    total_area_percentage = total_box_area_percentage(all_boxes, image.size)
    return combined, total_area_percentage

@st.cache_resource
def load_tf_model_from_wandb():
    wandb.login()
api = wandb.Api()
artifact = api.artifact('rueedi-tobias-hochschule-luzern/Own CNN/cnn-model:v16', type='model')
model_path = artifact.download()

for file in os.listdir(model_path):
    if file.endswith(".h5") or file.endswith(".keras"):
        model_g20 = tf.keras.models.load_model(os.path.join(model_path, file), compile=False)
        model_g20.trainable = False
        break

artifact = api.artifact('rueedi-tobias-hochschule-luzern/Own CNN/cnn-model:v13', type='model')
model_path = artifact.download()

for file in os.listdir(model_path):
    if file.endswith(".h5") or file.endswith(".keras"):
        model_g10 = tf.keras.models.load_model(os.path.join(model_path, file), compile=False)
        model_g10.trainable = False
        break

artifact = api.artifact('rueedi-tobias-hochschule-luzern/Own CNN/cnn-model:v15', type='model')
model_path = artifact.download()

for file in os.listdir(model_path):
    if file.endswith(".h5") or file.endswith(".keras"):
        model_g5 = tf.keras.models.load_model(os.path.join(model_path, file), compile=False)
        model_g5.trainable = False
        break

@st.cache_data
def draw_tf_detections(image):
    # Use the three loaded models: model_g20, model_g10, model_g5 (already loaded above)
    # Preprocess image
    img_resized = image.resize((640, 640))  # Must match training input shape
    img_array = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict with all three models
    pred_g20 = model_g20.predict(input_tensor)[0]
    pred_g10 = model_g10.predict(input_tensor)[0]
    pred_g5  = model_g5.predict(input_tensor)[0]

    # Decode all boxes (no NMS, just raw boxes)
    boxes_g20, _ = decode_yolo_output(pred_g20, conf_thresh=0.7)
    boxes_g10, _ = decode_yolo_output(pred_g10, conf_thresh=0.7)
    boxes_g5,  _ = decode_yolo_output(pred_g5,  conf_thresh=0.7)

    # Combine all boxes
    all_boxes = boxes_g20 + boxes_g10 + boxes_g5

    # Overlay colored regions for each box (not just rectangle outlines)
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    n = 0

    for box in all_boxes:
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        # Fill the box area with a semi-transparent blue
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 255, int(255 * 0.4)))
        n += 1

    # Composite overlay onto original image
    combined = Image.alpha_composite(image_rgba, overlay).convert("RGB")
    total_area_percentage = total_box_area_percentage(all_boxes, image.size)
    return combined, total_area_percentage


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

def decode_yolo_output(pred_grid, conf_thresh=0.3):
    """
    Decodes YOLO grid output to bounding boxes.
    Args:
        pred_grid: [GRID_Y, GRID_X, NUM_BOXES, 5 + NUM_CLASSES]
        conf_thresh: confidence threshold
    Returns:
        boxes: list of [ymin, xmin, ymax, xmax] in relative coords (0–1)
        scores: list of confidence scores
    """
    boxes = []
    scores = []

    grid_y, grid_x, num_boxes = pred_grid.shape[:3]

    for gy in range(grid_y):
        for gx in range(grid_x):
            for b in range(num_boxes):
                pred = pred_grid[gy, gx, b]
                conf = pred[4]
                if conf < conf_thresh:
                    continue

                x, y, w, h = pred[:4]
                cx = (gx + x) / grid_x
                cy = (gy + y) / grid_y
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2

                boxes.append([ymin, xmin, ymax, xmax])
                scores.append(conf)

    return boxes, scores

def total_box_area_percentage(boxes, image_size):
    """
    Calculates the total area covered by a list of boxes as a percentage of the image area.
    Args:
        boxes: list or np.array of [ymin, xmin, ymax, xmax] in relative coords (0–1)
        image_size: (width, height) tuple of the image in pixels
    Returns:
        float: percentage (0-100) of image area covered by the union of all boxes
    """
    import numpy as np

    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        x1 = int(np.clip(xmin * w, 0, w))
        y1 = int(np.clip(ymin * h, 0, h))
        x2 = int(np.clip(xmax * w, 0, w))
        y2 = int(np.clip(ymax * h, 0, h))
        mask[y1:y2, x1:x2] = 1  # fill area

    covered = np.sum(mask)
    total = w * h
    percent = 100.0 * covered / total if total > 0 else 0.0
    return percent