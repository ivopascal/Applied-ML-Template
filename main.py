import os
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM

if __name__ == "__main__":
    # # IF THE DATASET HAS NOT BEEN PREPROCESSED YET, USE:
    # handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
    # preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    # dataset_dict, digit_dataset = preprocessor.dataset_preprocessing()
    # preprocessor.handler.save_all_datasets()

    # IF THE DATASET HAS ALREADY BEEN PREPROCESSED AND SAVED, USE:
    handler = load_sudoku_dataset()
    preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    digit_dataset= handler.datasets['digits']

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