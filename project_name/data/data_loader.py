import numpy as np
import os
from PIL import Image
from typing import Literal


class DataLoader:
    """
    Data loader class for loading data from the subsetted data.
    """
    def __init__(self, folder: Literal["train", "Val"]) -> None:
        """Initialise the class

        Args:
            folder (Literal[&quot;train&quot;, &quot;Val&quot;]):
            which subset folder to get the data from.
        """
        self.folder = folder + "_subset"
        file_directory = __file__
        data_directory = os.path.join(os.path.split(file_directory)[0],
                                      self.folder)
        self.file_endings = [".png", "_depth.npy"]

        self.data_paths = [os.path.join(data_directory, _) for _ in
                           os.listdir(data_directory)]

        self.data_index = 0

    def increment_index(self) -> None:
        """Increments the index of the data path list.
        makes sure it doesn't go over the index limit.
        """
        self.data_index += 1
        self.data_index %= len(self.data_paths)

    def get_data(self) -> list[np.ndarray]:
        """Gets the image and depth data from the list of subset data.

        Returns:
            list[np.ndarray]: a size 2 list of image data and then depth data.
        """
        current_data_path = self.data_paths[self.data_index]
        self.increment_index()

        data = [os.path.join(current_data_path, _) for _ in
                os.listdir(current_data_path)]

        return_list: list[np.ndarray] = []
        for file in data:
            if file.endswith(self.file_endings[0]):
                image_data = np.asarray(Image.open(file))
                if len(return_list) > 0:
                    return_list.insert(0, image_data)
                else:
                    return_list.append(image_data)
            elif file.endswith(self.file_endings[1]):
                depth_data = np.load(file)
                return_list.append(depth_data)

        return return_list


def main() -> None:
    """
    simple test and run data loader class
    """
    data_load = DataLoader("val")

    data = data_load.get_data()

    print([_.shape for _ in data])


if __name__ == '__main__':
    main()
