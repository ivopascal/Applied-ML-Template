from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, image_patch):
        pass
