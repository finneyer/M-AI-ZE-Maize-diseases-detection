from PIL import ImageDraw, ImageFont
import random
import streamlit as st

@st.cache_data
def draw_fake_detections(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.truetype("arial.ttf", 60) 
    number_of_detections = random.randint(1, 10)
    for _ in range(number_of_detections):
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        x2 = x1 + random.randint(125, width-x1)
        y2 = y1 + random.randint(125, height-y1)    
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.text((x1 + 5, y1 - 10), "Disease", fill="red", font=font)
    return image, number_of_detections
