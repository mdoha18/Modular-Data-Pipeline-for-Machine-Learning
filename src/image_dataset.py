import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import Dataset
import os
from typing import List
import glob


class ImageDataset(Dataset):
    def _data_points_load(self) -> List[str]:
        pattern_jpg = os.path.join(self.root, '**', '*.jpg')
        data_points_jpg = glob.glob(pattern_jpg, recursive=True)
        return data_points_jpg

    def _data_load(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def display(self, index: int) -> None:
        data, label = self[index]

        if not self.eager:
            data = self._data_load(data)
            if data is None:
                print("There is no image to display")
                return

        if label is not None:
            plt.title(f"Label: {label}")
        else:
            plt.title("Unlabelled!")
        plt.imshow(data)
        plt.show()
