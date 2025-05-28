import os
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor, SudokuPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
import matplotlib.pyplot as plt
import numpy as np


def get_preprocessed_dataset(dataset_is_processed):
    # fetches the dataset from local source, if it is already preprocessed
    if dataset_is_processed:
        handler = load_sudoku_dataset()
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
        digit_dataset = handler.datasets['digits']
    else:
        handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
        dataset_dict, digit_dataset = preprocessor.dataset_preprocessing()
        preprocessor.handler.save_all_datasets()
    return digit_dataset, preprocessor, handler  # ok so I think this can be done cleaner but I want to focus on task rn


def get_model(train_model, preprocessor):
    # prevents you from training the models again if it has been trained already
    if train_model:
        X_train = digit_dataset['train']['image']
        y_train = digit_dataset['train']['label']

        X_val = digit_dataset['validation']['image']
        y_val = digit_dataset['validation']['label']

        X_test = digit_dataset['test']['image']
        y_test = digit_dataset['test']['label']

        sudoku_height = preprocessor.cropper.output_size // 9

        svm = SVM(input_shape=(sudoku_height, sudoku_height), verbose=True)
        svm.train(X_train[:10000], y_train[:10000])
        svm.evaluate(X_test, y_test)

        cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
        cnn.train(X_train, y_train, X_val, y_val, verbose=1)
        cnn.evaluate(X_test, y_test)
        cnn.save("test")
    else:
        sudoku_height = preprocessor.cropper.output_size // 9
        cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
        cnn.load("test")
    return cnn


def predict_sudoku(cnn, sudoku_sample):
    plt.imshow(sudoku_sample, cmap="gray")
    plt.show()
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=252)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)
    print("length of dataset:", len(digit_dataset))

    predictions = cnn.predict(digit_dataset)
    sudoku_labels = []
    dimension = int(np.sqrt(len(digit_dataset)))
    for i in range(dimension):
        row = []
        for j in range(dimension):
            label = np.argmax(predictions[j + (dimension * i)])
            row.append(int(label))
        sudoku_labels.append(row)

    for row in sudoku_labels:
        print(row)


def predict_sudoku_raw(cnn, sudoku_sample):
    plt.imshow(sudoku_sample['image'], cmap="gray")
    plt.show()
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=252)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)
    for idx, digit in enumerate(digit_dataset):
        digit_dataset[idx] = digit["image"]
    print("length of dataset:", len(digit_dataset))

    predictions = cnn.predict(digit_dataset)
    sudoku_labels = []
    dimension = int(np.sqrt(len(digit_dataset)))
    for i in range(dimension):
        row = []
        for j in range(dimension):
            label = np.argmax(predictions[j + (dimension * i)])
            row.append(int(label))
        sudoku_labels.append(row)

    for row in sudoku_labels:
        print(row)


if __name__ == "__main__":
    digit_dataset, preprocessor, handler = get_preprocessed_dataset(dataset_is_processed=True) # change this to False if the dataset hasn't been processed and saved
    cnn = get_model(train_model=False, preprocessor=preprocessor)

    #raw_dataset = handler.datasets['raw']
    #predict_sudoku_raw(cnn, raw_dataset['test'][1])

    raw_dataset = handler.datasets['preprocessed']
    predict_sudoku(cnn, raw_dataset['test']['image'][1])
