import os
import pandas as pd
import PIL
from collections.abc import Iterator

######################
### BOUNDING BOXES ###
######################

def get_bbox_list() -> str:
    """
    Get the list of bboxes provided by the NIH Chest X-ray dataset authors
    retalive to this file
    """
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.normpath(
        os.path.join(script_path, "../../data/lists/BBox_List_2017.csv")
    )
    return file_path


def split_bbox_line(line: str) -> tuple[str, str, float, float, float, float]:
    """
    Split the line from the bbox txt file into it's components
    Returns:
        - image name
        - disease label
        - x-coordinate of the bounding box
        - y-coordinate of the bounding box
        - bounding box width
        - bounding box height
    """
    img_name, disease, x, y, w, h = line.split(",")
    x, y, w, h = map(float, (x, y, w, h))
    return img_name, disease, x, y, w, h


def read_bbox_lines() -> Iterator[tuple[str, str, float, float, float, float]]:
    """
    Read list of bboxes provided by the NIH Chest X-ray dataset authors
    Skips the header row and yields one parsed bounding box record per line.
    Yields:
        A tuple containing:
        - image filename
        - disease label
        - x-coordinate of the bounding box
        - y-coordinate of the bounding box
        - bounding box width
        - bounding box height
    """
    with open(get_bbox_list()) as file:
        for line in file:
            if line.startswith("Image Index"):
                continue
            yield split_bbox_line(line)


def get_bbox_data() -> pd.DataFrame:
    """Put bbox data into a DataFrame"""
    return pd.DataFrame(
        data=[line for line in read_bbox_lines()],
        columns = ("img_name", "disease", "x", "y", "w", "h")
    )


##################
### DATA ENTRY ###
##################

def get_data_list() -> str:
    """Get the list of all images relative to this file"""
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.normpath(
        os.path.join(script_path, "../../data/lists/Data_Entry_2017.csv")
    )
    return file_path


def split_data_line(line: str) -> tuple[
    str, str, int, int, int, str, str, int, int, float, float
]:
    """Split the line of data into its parts"""
    (
        img_name, diseases, follow_up, patient_id, age, gender,
        view_pos, original_w, original_h, w_scaling, h_scaling
    ) = line.split(",")
    follow_up, patient_id, age, original_w, original_h = map(
        int, (follow_up, patient_id, age, original_w, original_h)
    )
    w_scaling, h_scaling = map(float, (w_scaling, h_scaling))
    return (
        img_name, diseases, follow_up, patient_id, age, gender,
        view_pos, original_w, original_h, w_scaling, h_scaling
    )


def read_data_lines() -> Iterator[
    tuple[str, str, int, int, int, str, str, int, int, float, float]
]:
    """
    Read list of samples provided by the NIH Chest X-ray dataset authors
    Skips the header row and yields one parsed samples record per line.
    Yields:
        A tuple containing:
        - image filename
        - disease labels
        - follow up number
        - patient id
        - patient age
        - patient gender
        - X-ray view position
        - original image width
        - original image height
        - original image width scaling
        - original image height scaling
    """
    with open(get_data_list()) as file:
        for line in file:
            if line.startswith("Image Index"):
                continue
            yield(split_data_line(line))


def get_data() -> pd.DataFrame:
    """Put all sample data in a neat dataframe"""
    return pd.DataFrame(
        data = [line for line in read_data_lines()],
        columns = (
            "img_name", "diseases", "follow_up", "patient_id", "age", "gender",
            "view_pos", "original_w", "original_h", "w_scaling", "h_scaling"
        )
    )


#####################
### TRAIN/VAL SET ###
#####################

def get_train_val_list() -> str:
    """Get the list of all images in the train/val set relative to this file"""
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.normpath(
        os.path.join(script_path, "../../data/lists/train_val_list.txt")
    )
    return file_path


def read_train_val_lines() -> Iterator[str]:
    """Read all lines of the train/val list and yield image names"""
    with open(get_train_val_list()) as file:
        for line in file:
            yield line


def get_train_val_data() -> pd.DataFrame:
    """Put all train/val image names in a dataframe"""
    return pd.DataFrame(data = [line for line in read_train_val_lines()])

################
### TEST SET ###
################

def get_test_list() -> str:
    """Get the list of all images in the test set relative to this file"""
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.normpath(
        os.path.join(script_path, "../../data/lists/test_list.txt")
    )
    return file_path


def read_test_lines() -> Iterator[str]:
    """Read all lines of the test list and yield image names"""
    with open(get_test_list()) as file:
        for line in file:
            yield line


def get_test_data() -> pd.DataFrame:
    """Put all test image names in a dataframe"""
    return pd.DataFrame(data = [line for line in read_test_lines()])

