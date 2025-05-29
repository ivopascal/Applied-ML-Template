from keras import Sequential
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor, SudokuPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image


def get_preprocessed_dataset(is_preprocessed: bool = False):
    print("Getting preprocessed dataset")
    # fetches the dataset from local source, if it is already preprocessed
    if is_preprocessed:
        handler = load_sudoku_dataset()
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    else:
        handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
        preprocessor.handler.save_all_datasets()
    return preprocessor


def get_model(preprocessor: DatasetPreprocessor, train_model: bool = False):
    print("Getting model")
    digit_dataset = preprocessor.handler.datasets['digits']
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
        cnn.save("sudoku_cnn")
    else:
        sudoku_height = preprocessor.cropper.output_size // 9
        cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
        cnn.load("sudoku_cnn")
    return cnn


def predict_sudoku(cnn: Sequential, sudoku_sample: Image.Image):
    print("Predicting")
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=252)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)

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

    plt.imshow(sudoku_sample, cmap="gray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program takes an optional argument for prediction. If no argument is given, the model is trained.')
    parser.add_argument('path', nargs='?', help='Path to image for prediction. The image should be cropped to the Sudoku.')

    test_image = None
    args = parser.parse_args()
    if args.path:
        print(f"Argument received, digitizing sudoku")
        train_model = False
        try:
            test_image = Image.open(args.path)
        except Exception as e:
            print(f"Failed to open image: {e}")
    else:
        print("No argument given, training model")
        train_model = True

    preprocessor = get_preprocessed_dataset(is_preprocessed=True) # change this to False if the dataset hasn't been processed and saved
    cnn = get_model(preprocessor, train_model)

    if test_image:
        predict_sudoku(cnn, test_image)