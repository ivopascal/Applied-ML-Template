import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Union
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


class CNN:
    def __init__(
            self,
            input_shape: Tuple[int, int, int], 
            num_classes: int
            ) -> None:
        '''
        Initialize CNN with given parameters
        '''
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.models.Sequential:
        '''
        Build the model
        '''
        cnn = keras.models.Sequential()

        cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        cnn.add(keras.layers.Flatten())

        cnn.add(keras.layers.Dense(units=128, activation='relu'))
        cnn.add(keras.layers.Dropout(0.2))

        cnn.add(keras.layers.Dense(units=128, activation='relu'))
        cnn.add(keras.layers.Dropout(0.2))

        cnn.add(keras.layers.Dense(units=self.num_classes, activation='softmax'))

        cnn.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        return cnn
    
    def _reshape_image_CNN(self, img: Image.Image) -> np.ndarray:
        '''
        Reshapes single image to match input shape
        '''
        img_array = np.array(img)
        normalized_array = img_array.astype(np.float32) / 255.0
        return normalized_array.reshape(self.input_shape[0], self.input_shape[1], 1)

    def _reshape_data_CNN(self, image_list: List[Image.Image]) -> np.ndarray:
        '''
        Reshapes a list of images to match input shape
        '''
        reshaped_data = np.zeros((
            len(image_list), self.input_shape[0], self.input_shape[1], 1
            ), dtype=np.float32)
        for i, img in enumerate(image_list):
            reshaped_data[i] = self._reshape_image_CNN(img)
        return reshaped_data
    
    def train(
            self,
            X_train: List[Image.Image],
            y_train: List[int],
            X_val: List[Image.Image],
            y_val: List[int],
            verbose: int
            ) -> None:
        '''
        Train CNN using the training and validation data
        '''
        X_train = self._reshape_data_CNN(X_train)
        y_train = keras.utils.to_categorical(np.array(y_train), self.num_classes)

        X_val = self._reshape_data_CNN(X_val)
        y_val = keras.utils.to_categorical(np.array(y_val), self.num_classes)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            epochs=100,
            verbose=verbose,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

    def predict(self, input: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        '''
        Predict value(s) using the trained CNN
        '''
        if isinstance(input, Image.Image):
            input = self._reshape_image_CNN(input)
        else:
            input = self._reshape_data_CNN(input)
        return self.model.predict(input)
    
    def evaluate(self, X_test: List[Image.Image], y_test: List[int]) -> None:
        '''
        Evaluates the model's accuracy, precision, recall and F1
        '''
        X_test = self._reshape_data_CNN(X_test)
        y_test = np.array(y_test)
        test_labels = keras.utils.to_categorical(y_test, self.num_classes)

        # Convert one-hot encoded data back to normal labels
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Evaluate the cnn on the test set
        test_loss, test_accuracy, _, _ = self.model.evaluate(X_test, test_labels, verbose=2)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

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

        # Per class precision, recall and F1
        report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(10)])
        print(report)

        # Accuracy plot over time
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training vs Validation Accuracy')
        plt.show()

        # Loss plot over time
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.show()

    def save(self, name: str, path: str=None) -> None:
        if path is None:
            path = os.path.join("sudoku_digitalisation", "data", "models")
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{name}.keras")
        self.model.save(save_path)