import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from typing import List, Union, Tuple
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class SVM():
    def __init__(self, input_shape: Tuple[int, int], verbose: bool = False) -> None:
        '''
        Initialise svm as a one versus all SVM
        '''
        self.n_features = input_shape[0] * input_shape[1]
        self.model = svm.LinearSVC(verbose=verbose)

    def _reshape_image_SVM(self, img: Image.Image) -> np.ndarray:
        '''
        Reshapes single image to match input shape
        '''
        img_array = np.array(img)
        normalized_array = img_array.astype(np.float32) / 255.0
        return normalized_array.flatten()

    def _reshape_data_SVM(self, image_list: List[Image.Image]) -> np.ndarray:
        '''
        Reshapes a list of images to match input shape
        '''
        reshaped_data = np.zeros((len(image_list), self.n_features), dtype=np.float32)
        for i, img in enumerate(image_list):
            reshaped_data[i] = self._reshape_image_SVM(img)
        return reshaped_data

    def train(self, X_train: List[Image.Image], y_train: List[int]) -> None:
        '''
        Trains svm on X_train matrix and y_train vector
        '''
        X_train = self._reshape_data_SVM(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train)

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        '''
        Predict value(s) using the trained SVM
        '''
        if isinstance(input, Image.Image):
            input = self._reshape_image_SVM(input)
        else:
            input = self._reshape_data_SVM(input)
        return self.model.predict(input)

    def evaluate(self, X_test: List[Image.Image], y_test: List[int]) -> None:
        '''
        Evaluates the svm using the passed test data
        '''
        y_test = np.array(y_test)
        y_pred = self.predict(X_test)

        # Overall accuracy
        overall_acc = accuracy_score(y_test, y_pred)
        print(f"Overall accuracy: {overall_acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

        # Per class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"Accuracy for Class {i}: {acc:.4f}")

        # Per class precisionm recall and F1
        report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)])
        print(report)
