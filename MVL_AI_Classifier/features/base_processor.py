from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        pass
