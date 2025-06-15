import unittest
import numpy as np
from project_name.models.Preprocessing_class import Preprocessing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.tile_size = (256, 256)
        self.preprocessor = Preprocessing(tile_size=self.tile_size)
        self.rgb_image = np.random.randint(
            0, 256, (510, 510, 3), dtype=np.uint8
        )
        self.gray_image = np.random.randint(
            0, 256, (510, 510), dtype=np.uint8
        )

    def test_is_8_bit(self):
        self.assertTrue(self.preprocessor.is_8_bit(self.rgb_image))
        self.assertFalse(
            self.preprocessor.is_8_bit(
                self.rgb_image.astype(np.float32)
            )
        )

    def test_normalize_uint8(self):
        norm = self.preprocessor.normalize(self.rgb_image)
        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))
        self.assertEqual(norm.dtype, np.float32)

    def test_normalize_float(self):
        float_array = self.rgb_image.astype(np.float32) / 255.0
        norm = self.preprocessor.normalize(float_array)
        self.assertTrue(np.allclose(float_array, norm))
        self.assertEqual(norm.dtype, np.float32)

    def test_tile_with_padding_rgb(self):
        tiles = self.preprocessor.tile_with_padding(self.rgb_image)
        expected_num_tiles = (
            (510 + (256 - 510 % 256)) // 256
        ) ** 2  # 2x2 = 4
        self.assertEqual(tiles.shape[0], expected_num_tiles)
        self.assertEqual(tiles.shape[1:], (256, 256, 3))

    def test_tile_with_padding_grayscale(self):
        tiles = self.preprocessor.tile_with_padding(self.gray_image)
        expected_num_tiles = (
            (510 + (256 - 510 % 256)) // 256
        ) ** 2  # 2x2 = 4
        self.assertEqual(tiles.shape[0], expected_num_tiles)
        self.assertEqual(tiles.shape[1:], (256, 256))

    def test_reconstruction_rgb(self):
        tiles = self.preprocessor.tile_with_padding(self.rgb_image)
        recon = self.preprocessor.reconstruct_image(tiles)
        self.assertEqual(recon.shape, self.rgb_image.shape)
        self.assertTrue(np.allclose(recon, self.rgb_image, atol=2))

    def test_reconstruction_grayscale(self):
        tiles = self.preprocessor.tile_with_padding(self.gray_image)
        recon = self.preprocessor.reconstruct_image(tiles)
        self.assertEqual(recon.shape, self.gray_image.shape)
        self.assertTrue(np.allclose(recon, self.gray_image, atol=2))

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.preprocessor.tile_with_padding("not an array")

    def test_invalid_input_dtype(self):
        with self.assertRaises(ValueError):
            self.preprocessor.tile_with_padding(
                np.ones((100, 100), dtype=np.float32)
            )

    def test_missing_padding_info(self):
        with self.assertRaises(ValueError):
            self.preprocessor.reconstruct_image(
                np.zeros((4, 256, 256, 3))
            )

    def test_depth_to_rgb(self):
        # Create a mock depth map with float32 values from 0 to 10
        depth_map = np.random.uniform(
            low=0.0, high=10.0, size=(480, 640)
        ).astype(np.float32)
        rgb_depth = self.preprocessor.depth_to_rgb(
            depth_map, cmap='plasma'
        )

        self.assertEqual(rgb_depth.shape, (480, 640, 3))
        self.assertEqual(rgb_depth.dtype, np.uint8)
        self.assertTrue(
            np.all(rgb_depth >= 0) and np.all(rgb_depth <= 255)
        )
        self.assertGreater(np.std(rgb_depth), 0)

        with self.assertRaises(ValueError):
            self.preprocessor.depth_to_rgb(
                np.ones((10, 10, 3))  # Not 2D
            )


if __name__ == '__main__':
    unittest.main()
