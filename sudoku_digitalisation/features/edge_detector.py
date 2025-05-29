from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class EdgeDetector:

    def __init__(self) -> None:
        # EDGE DETECTION SETTINGS
        pass

    def _known_keypoints_bb(self, keypoints):
        bounding_box = np.array([
                [keypoints[0], keypoints[1]],  # top left
                [keypoints[2], keypoints[3]],  # bottom left
                [keypoints[4], keypoints[5]],  # bottom right
                [keypoints[6], keypoints[7]]   # top right
            ], dtype=np.float32)
        return bounding_box

    def get_bounding_box(
            self,
            image: Image.Image,
            keypoints: np.ndarray=None
            ) -> np.ndarray:
        if keypoints != None:
            return self._known_keypoints_bb(keypoints)
        else:
            height, width = image.size
            keypoints = [0, 0, 0, height, width, height, width, 0]