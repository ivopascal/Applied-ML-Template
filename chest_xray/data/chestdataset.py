from PIL import Image
import os
import torch
from torch.utils.data import Dataset    


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_root, classes, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.classes = classes

        # Mapping the class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.transform = transform

        # finding the path of the images for faster access
        self.image_paths = self.build_image_path_map()

        self.df = self.df[self.df["Image Index"].isin(self.image_paths.keys())].reset_index(drop=True)

    # finding the path of the images for faster access method
    def build_image_path_map(self):
        image_paths = {}

        # Walk through the image root directory and build a mapping of image names to their full paths
        for root, _, files in os.walk(self.image_root):
            for file in files:
                if file.lower().endswith(".png"):
                    image_paths[file] = os.path.join(root, file)

        return image_paths

    # The length of the dataset
    def __len__(self):
        return len(self.df)

    # given an index, this method will return the image and the labels of that image
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get the image name and the label string from the dataframe
        image_name = row["Image Index"]
        label_string = row["Finding Labels"]

        image_path = self.image_paths[image_name]

        image = Image.open(image_path).convert("L")

        labels = torch.zeros(len(self.classes), dtype=torch.float32)

        findings = label_string.split("|")

        # Set the corresponding indices in the labels tensor to 1 for each finding present in the image
        for finding in findings:
            if finding in self.class_to_idx:
                labels[self.class_to_idx[finding]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, labels