import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from read_lists import get_bbox_data

def get_images_path():
    script_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_path, "../../data/images/"))

def get_bbox_path():
    script_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_path, "../../data/bbox_images/"))

def draw_bbox(draw, disease, x, y, w, h):
    draw.rectangle((x, y, x+w, y+h), outline = "blue", width = 2)
    draw.text((x, y-5), disease, fill = "red")

def draw_bboxes(img_path, group):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, row in group.iterrows():
        draw_bbox(draw, row["disease"], row["x"], row["y"], row["w"], row["h"])
    return img
    
def generate_bboxes():
    groups = get_bbox_data().groupby("img_name")
    images_path = get_images_path()
    bboxes_path = get_bbox_path()
    i = 0
    for img_name, group in groups:
        i += 1
        print(f"processing image {i}: {img_name}")
        img_path = os.path.join(images_path, img_name)
        img = draw_bboxes(img_path, group)
        img.save(os.path.join(bboxes_path, img_name))

if __name__ == "__main__":
    generate_bboxes()
