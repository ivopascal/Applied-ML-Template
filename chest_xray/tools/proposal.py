from bbox import draw_bbox, get_bbox_data
import matplotlib.pyplot as plt
import os
import pandas as pb
from PIL import Image, ImageDraw
from read_lists import get_bbox_data, get_data, get_train_val_data, get_test_data

def disease_stats():
    return get_data()["diseases"].str.split("|").explode().value_counts()

def gender_stats():
    return get_data()["gender"].value_counts()

def age_stats():
    return get_data()["age"].value_counts()

def view_pos_stats():
    return get_data()["view_pos"].value_counts()

def create_plots():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/proposal/chart.png"))
    diseases = disease_stats()
    age = age_stats()
    gender = gender_stats()
    view_pos = view_pos_stats()
    fig, axes = plt.subplots(2, 2)
    axes[0,0].barh(diseases.index, diseases.values)
    axes[0,0].set_title("Disease Distribution")
    axes[0,0].set_xlabel("Count")
    axes[0,1].pie(gender.values, labels = gender.index)
    axes[0,1].set_title("Gender Distribution")
    axes[1,1].pie(view_pos.values, labels = view_pos.index)
    axes[1,1].set_title("View Position Distribution")
    axes[1,0].hist(age)
    axes[1,0].set_title("Age Distribution")
    axes[1,0].set_ylabel("Age")
    plt.tight_layout()
    plt.savefig(file_path)

def smallest_bbox():
    bboxes = get_bbox_data()
    bboxes["area"] = bboxes["w"] * bboxes["h"]
    smallest_bbox_img = bboxes.loc[bboxes["area"].idxmin()]
    return(smallest_bbox_img)

def create_smallest_bbox_images():
    s = smallest_bbox()
    script_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.normpath(os.path.join(script_path, "../../data/proposal/"))
    in_path = os.path.normpath(os.path.join(script_path, f"../../data/images/{s['img_name']}"))
    original = Image.open(in_path).convert("RGB")
    half = original.resize((512,512))
    quarter = original.resize((256,256))
    x, y, w, h = (s["x"], s["y"], s["w"], s["h"])
    draw_bbox(ImageDraw.Draw(original), s["disease"], x, y, w, h, 4)
    draw_bbox(ImageDraw.Draw(half), s["disease"], x/2, y/2, w/2, h/2)
    draw_bbox(ImageDraw.Draw(quarter), s["disease"], x/4, y/4, w/4, h/4, 1)
    original.save(os.path.join(out_path, "1024x1024.png"))
    half.save(os.path.join(out_path, "512x512.png"))
    quarter.save(os.path.join(out_path, "256x256.png"))
    original_crop = original.crop((x-32, y-32, x+w+32, y+h+32))
    half_crop = half.crop((x/2-10, y/2-16, (x+w)/2+16, (y+h)/2+16))
    quarter_crop = quarter.crop((x/4-8, y/4-8, (x+w)/4+8, (y+h)/4+8))
    original_crop.save(os.path.join(out_path, "1024x1024_crop.png"))
    half_crop.save(os.path.join(out_path, "512x512_crop.png"))
    quarter_crop.save(os.path.join(out_path, "256x256_crop.png"))
    

if __name__ == "__main__":
    create_smallest_bbox_images()
    create_plots()
