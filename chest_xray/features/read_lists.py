import os
import pandas as pd
import PIL

######################
### BOUNDING BOXES ###
######################

def get_bbox_list():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/lists/BBox_List_2017.csv"))
    return file_path

def split_bbox_line(line):
    img_name, disease, x, y, w, h = line.split(",")
    x, y, w, h = map(float, (x, y, w, h))
    return img_name.strip(), disease.strip(), x, y, w, h

def read_bbox_lines():
    with open(get_bbox_list()) as file:
        for line in file:
            if line.startswith("Image Index"):
                continue
            yield split_bbox_line(line)

def get_bbox_data() -> pd.DataFrame:
    return pd.DataFrame(
        data=[line for line in read_bbox_lines()],
        columns = ("img_name", "disease", "x", "y", "w", "h")
    )

##################
### DATA ENTRY ###
##################

def get_data_list():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/lists/Data_Entry_2017.csv"))
    return file_path

def split_data_line(line):
    img_name, diseases, follow_up, patient_id, age, gender, view_pos, original_w, original_h, w_scaling, h_scaling = line.split(",")
    img_name = img_name.strip()
    diseases = diseases.strip()
    follow_up, patient_id, age, original_w, original_h = map(int, (follow_up, patient_id, age, original_w, original_h))
    w_scaling, h_scaling = map(float, (w_scaling, h_scaling))
    return img_name, diseases, follow_up, patient_id, age, gender, view_pos, original_w, original_h, w_scaling, h_scaling

def read_data_lines():
    with open(get_data_list()) as file:
        for line in file:
            if line.startswith("Image Index"):
                continue
            yield(split_data_line(line))

def get_data():
    return pd.DataFrame(
        data = [line for line in read_data_lines()],
        columns = ("img_name", "diseases", "follow_up", "patient_id", "age", "gender", "view_pos", "original_w", "original_h", "w_scaling", "h_scaling")
    )

#####################
### TRAIN/VAL SET ###
#####################

def get_train_val_list():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/lists/train_val_list.txt"))
    return file_path

def read_train_val_lines():
    with open(get_train_val_list()) as file:
        for line in file:
            yield line.strip()
    
def get_train_val_data():
    return pd.DataFrame(data = [line for line in read_train_val_lines()])

################
### TEST SET ###
################

def get_test_list():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/lists/test_list.txt"))
    return file_path

def read_test_lines():
    with open(get_test_list()) as file:
        for line in file:
            yield line.strip()
    
def get_test_data():
    return pd.DataFrame(data = [line for line in read_test_lines()])

