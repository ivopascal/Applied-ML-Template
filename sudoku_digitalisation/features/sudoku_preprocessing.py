import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from PIL import Image
from typing import Dict, Tuple, Union, Any
from sudoku_digitalisation.features.image_operations import ImageConverter, ImageCropper
from sudoku_digitalisation.features.edge_detector import EdgeDetector
from sudoku_digitalisation.features.sudoku_splitter import SudokuSplitter
from sudoku_digitalisation.features.dataset_handler import DatasetHandler, load_sudoku_dataset


class SudokuPreprocessor:
    """
    Handles full preprocessing tasks for single objects.
    """    
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBUTES WHEN NECESSARY
                 clip_limit: int = 3,
                 output_size: int = 252) -> None:
        self.edge_detector = EdgeDetector()
        self.converter = ImageConverter(clip_limit)
        self.cropper = ImageCropper(output_size)

    def convert_crop_image(self, image: Image.Image, keypoints: np.ndarray=None) -> Image.Image:
        clahe_img = self.converter.apply_clahe(image)
        bbox = self.edge_detector.get_bounding_box(clahe_img, keypoints)
        return self.cropper.crop_to_box(clahe_img, bbox)

    def convert_crop_datapoint(self, dp: Dict[str, Any]) -> Dict[str, Any]:
        dp = dp.copy()
        dp['image'] = self.convert_crop_image(dp['image'], dp['keypoints'])
        return dp
    
    def sudoku_preprocessing(self, sudoku: Union[Image.Image, Dict[str, Any]]) -> Tuple[Any, Any]:
        if isinstance(sudoku, Image.Image):
            preprocessed_img = self.convert_crop_image(sudoku)
            digit_list = SudokuSplitter.split_image(preprocessed_img)
            return preprocessed_img, digit_list
        elif isinstance(sudoku, dict):
            preprocessed_dp = self.convert_crop_datapoint(sudoku)
            labeled_digit_list = SudokuSplitter.split_datapoint(preprocessed_dp)
            return preprocessed_dp, labeled_digit_list
        else:
            raise TypeError("Input must be a PIL.Image.Image or a dataset dictionary with an 'image' field.")
        

class DatasetPreprocessor(SudokuPreprocessor):
    """
    Handles full preprocessing tasks for datasets and splits.
    """
    def __init__(self,
                 # ADD EDGE DETECTOR ATTRIBTUES HERE
                 handler: DatasetHandler,
                 clip_limit: int = 3, 
                 output_size: int = 450) -> None:
        self.handler = handler
        super().__init__(clip_limit, output_size)
        
    def split_preprocessing(self, split: str) -> Tuple[Dataset, Dataset]:
        preprocessed_list = []
        digits_list = []
        for datapoint in tqdm(self.handler.datasets['raw'][split], desc=f"Preprocessing {split} split"):
            preprocessed_dp, digits_dict_list = self.sudoku_preprocessing(datapoint)
            preprocessed_list.append(preprocessed_dp)
            digits_list.extend(digits_dict_list)
        return Dataset.from_list(preprocessed_list), Dataset.from_list(digits_list)

    def dataset_preprocessing(self) -> Tuple[DatasetDict, DatasetDict]:
        preprocessed_datasets = {}
        digits_datasets = {}
        for split in self.handler.datasets['raw']:
            preprocessed_ds, digits_ds = self.split_preprocessing(split)
            preprocessed_datasets[split] = preprocessed_ds
            digits_datasets[split] = digits_ds
        self.handler.datasets['preprocessed'] = DatasetDict(preprocessed_datasets)
        self.handler.datasets['digits'] = DatasetDict(digits_datasets)
        return self.handler.datasets['preprocessed'], self.handler.datasets['digits']
    
    # RESHAPE DATASET FOR CNN

    # RESHAPE DATASET FOR SVM


################
### SHOWCASE ###
################

# If you get a ModuleNotFoundError, run:
# python -m sudoku_digitalisation.features.sudoku_preprocessing

if __name__ == '__main__':
    # handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True) # loads from huggingface
    # # saves a specified locally, if no path is specified it is in sudoku_digitalisation/data/datasets
    # preprocessor.handler.save_dataset('raw')

    # loads a dataset, locally if hugface=False (default)
    handler = load_sudoku_dataset()
    # creating an instance of DatasetPreprocessor
    preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=450)
    # saves all datasets locally, if no path is specified it is in sudoku_digitalisation/data/datasets
    preprocessor.handler.save_all_datasets()

    # only preprocesses a single split from raw dataset
    test_convert_crop, test_digits = preprocessor.split_preprocessing('test')
    # preprocesses whole raw dataset, also saves preprocessed and digits dataset to handler
    convert_crop_dataset, digits_dataset = preprocessor.dataset_preprocessing()
    preprocessor.handler.save_all_datasets()

    print(preprocessor.handler.datasets['raw']) # raw dataset
    print(preprocessor.handler.datasets['preprocessed']) # preprocessed dataset
    print(preprocessor.handler.datasets['digits']) # single digits dataset

    # get a single image, without labels or checkpoints
    test_img = preprocessor.handler.datasets['raw']['train']['image'][0]

    # saves images from as png, can be dataset, split, or single image
    preprocessor.handler.save_dataset_png('preprocessed')
    preprocessor.handler.save_split_png('digits', 'test')

    split = 'train'
    index = 29
    preprocessor.handler.show_image('raw', split, index) # image from the raw dataset
    preprocessor.handler.show_image('preprocessed', split, index) # image from the preprocessed dataset
    preprocessor.handler.show_image('digits', split, index * 81) # image from the digits dataset

    # all individual functions can be accessed through the preprocessor
    preprocessor = DatasetPreprocessor(clip_limit=3, output_size=450)
    gray_img = preprocessor.converter.to_grayscale(test_img)
    clahe_img = preprocessor.converter.apply_clahe(gray_img)
    bbox = preprocessor.edge_detector.get_bounding_box(clahe_img)
    unlabeled_image = preprocessor.cropper.crop_to_box(clahe_img, bbox)

    # THE FOLLOWING CODE WILL NOT WORK WHILE EDGE DETECTION IS NOT IMPLEMENTED
    # full preprocessing can be applied on images as well
    _, digits_list = preprocessor.sudoku_preprocessing(test_img)
