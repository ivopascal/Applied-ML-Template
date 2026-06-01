import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from read_lists import get_bbox_data

def get_images_path() -> str:
    """Get the path to all images relative to this file"""
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_path, "../../data/images/"))


def get_bbox_path() -> str:
    """Get the path to output all bbox images"""
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_path, "../../data/bbox_images/"))


def draw_bbox(
        draw: PIL.ImageDraw.ImageDraw,
        disease: str,
        x: float, y: float, w: float, h: float
    ) -> None:
    """Draw a bbox on the image"""
    draw.rectangle((x, y, x+w, y+h), outline = "blue", width = 2)
    draw.text((x, y-5), disease, fill = "red")


def draw_bboxes(img_path: str, group: str) -> PIL.Image.Image:
    """Draws all bboxes of a certain image"""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    print(type(draw))
    for _, row in group.iterrows():
        draw_bbox(draw, row["disease"], row["x"], row["y"], row["w"], row["h"])
    return img


def generate_bboxes() -> None:
    """Generate images with bboxes overlayed for all known bboxes"""
    groups: pandas.api.typing.DataFrameGroupBy = get_bbox_data().groupby("img_name")
    images_path: str = get_images_path()
    bboxes_path: str = get_bbox_path()
    i = 0
    for img_name, group in groups:
        i += 1
        print(f"processing image {i}: {img_name}")
        img_path: str = os.path.join(images_path, img_name)
        img: PIL.Image.Image = draw_bboxes(img_path, group)
        img.save(os.path.join(bboxes_path, img_name))


if __name__ == "__main__":
    generate_bboxes()
