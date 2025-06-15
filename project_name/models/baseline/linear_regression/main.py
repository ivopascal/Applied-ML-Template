from .model import LinearModelHandler
from .data_loader import LinearRegressionDataset

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionPipeline:
    """
    Pipeline to handle data loading, training,
    and evaluation of a linear regression model.
    """

    def __init__(self, tile_size: tuple[int, int] = (64, 64)) -> None:
        """init pipeline

        Args:
            tile_size (tuple, optional): tile size. Defaults to (64, 64).
        """
        self.model_handler = LinearModelHandler()
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.tile_size = tile_size

    def load_data(self) -> None:
        """load data
        """
        print("Loading training data...")
        ts = self.tile_size
        train_dataset = LinearRegressionDataset("train", tile_size=ts)

        self.X_train, self.y_train = train_dataset.get_all()

        print("Loading test data...")
        test_dataset = LinearRegressionDataset("val", tile_size=self.tile_size)
        self.X_test, self.y_test = test_dataset.get_all()

    def train_model(self) -> None:
        """train model
        """
        print("Training model...")
        self.model_handler.train(self.X_train, self.y_train)
        self.model_handler.save_model("trained_linear_model.pkl")

    def evaluate_model(self) -> None:
        """eval model
        """
        print("Evaluating model...")
        (
            mse_score, rmse_score, mae_score, abs_rel,
            delta1, delta2, delta3, inference_time
        ) = self.model_handler.evaluate(
            self.X_test, self.y_test)
        print(f"MSE: {mse_score:.4f}")
        print(f"RMSE: {rmse_score:.4f}")
        print(f"MAE: {mae_score:.4f}")
        print(f"Absolute Relative Error: {abs_rel:.4f}")
        print(f"d1={delta1:.3f}, d2={delta2:.3f}, d3={delta3:.3f}")
        print(f"Inference Time: {inference_time:.4f} s")

        ts = self.tile_size
        self.visualize_images_and_depths(num_samples=5, tile_size=ts)

    def run(self) -> None:
        """run pipeline
        """
        self.load_data()
        self.train_model()
        self.evaluate_model()

    def load_and_evaluate_saved_model(self, model_path: str) -> None:
        """Load a saved model and evaluate it on test data."""
        print("Loading test data for evaluation...")
        test_dataset = LinearRegressionDataset("val", tile_size=self.tile_size)
        self.X_test, self.y_test = test_dataset.get_all()

        self.model_handler.load_model(model_path)
        self.evaluate_model()

    def visualize_images_and_depths(self,
                                    num_samples: int = 5,
                                    tile_size: tuple[int, int] = (64, 64)
                                    ) -> None:
        """predict and visualize output

        Args:
            num_samples (int, optional): amount samples. Defaults to 5.
            tile_size (tuple, optional): tile size. Defaults to (64, 64).
        """
        if self.model_handler.model is None:
            raise ValueError("Model not trained or loaded.")
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not loaded.")

        y_pred = self.model_handler.predict(self.X_test)

        test_dataset = LinearRegressionDataset("val", tile_size=tile_size)

        n_images = len(test_dataset.original_images)
        if num_samples > n_images:
            num_samples = n_images

        sample_image_indices = np.random.choice(
            n_images, num_samples, replace=False
        )

        tile_h, tile_w = tile_size
        img_h, img_w = test_dataset.original_images[0].shape[:2]
        tiles_per_row = img_w // tile_w
        tiles_per_col = img_h // tile_h
        tiles_per_image = tiles_per_row * tiles_per_col

        for img_idx in sample_image_indices:
            # Tiles corresponding to this image
            start_tile_idx = img_idx * tiles_per_image
            end_tile_idx = start_tile_idx + tiles_per_image

            pred_tiles = y_pred[start_tile_idx:end_tile_idx]

            # Rebuild predicted depth map from tiles
            pred_depth_map = np.zeros((img_h, img_w))
            tile_idx = 0
            for row in range(tiles_per_col):
                for col in range(tiles_per_row):
                    tile_pred_flat = pred_tiles[tile_idx]
                    tile_pred = tile_pred_flat.reshape((tile_h, tile_w))
                    y_start = row * tile_h
                    y_end = y_start + tile_h
                    x_start = col * tile_w
                    x_end = x_start + tile_w

                    pred_depth_map[y_start:y_end, x_start:x_end] = tile_pred
                    tile_idx += 1

            img = test_dataset.original_images[img_idx]
            true_depth = test_dataset.original_depth_maps[img_idx]

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(true_depth, cmap='inferno')
            plt.title("Ground Truth Depth Map")
            plt.colorbar(shrink=0.6)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_depth_map, cmap='inferno')
            plt.title("Predicted Depth Map")
            plt.colorbar(shrink=0.6)
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def main() -> None:
    """main function to run
    """
    pipeline = LinearRegressionPipeline(tile_size=(64, 64))
    # To train and evaluate from scratch:
    # pipeline.run()

    # Or to load saved model and evaluate:
    pipeline.load_and_evaluate_saved_model("trained_linear_model.pkl")


if __name__ == "__main__":
    main()
