from typing import List, Tuple, Optional, Any
import os
import glob
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, root: str, eager: bool = True) -> None:
        self.__root = root
        self.__eager = eager
        self.__data_points = self._data_points_load()
        self.__labels = self._labels_load()

        if self.__eager:
            self.__data = [self._data_load(path) for path in self.__data_points]
        else:
            self.__data = None

    @property
    def root(self) -> str:
        return self.__root

    @root.setter
    def root(self, value: str):
        self.__root = value
        self.__data_points = self._data_points_load()
        self.__labels = self._labels_load()
        if self.__eager:
            self.__data = [self._data_load(path) for path in self.__data_points]

    @property
    def eager(self) -> bool:
        return self.__eager

    @eager.setter
    def eager(self, value: bool):
        self.__eager = value
        if self.__eager:
            self.__data = [self._data_load(path) for path in self.__data_points]
        else:
            self.__data = None

    def _data_points_load(self) -> List[str]:
        pattern = os.path.join(self.__root, '**', '*.wav')
        data_points = glob.glob(pattern, recursive=True)
        if not data_points:
            pattern = os.path.join(self.__root, '**', '*.jpg')
            data_points = glob.glob(pattern, recursive=True)
        return data_points

    @abstractmethod
    def _data_load(self, path: str) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")

    def _labels_load(self) -> Optional[List[str]]:
        return [os.path.basename(os.path.dirname(path)) for path in self.__data_points]

    @property
    def data_points(self) -> List[str]:
        return self.__data_points

    @property
    def labels(self) -> Optional[List[str]]:
        return self.__labels

    def __len__(self) -> int:
        return len(self.__data_points)

    def __getitem__(self, index: int) -> Tuple[Any, Optional[str]]:
        if index >= len(self.__data_points):
            raise IndexError(f"Index {index} is out of range for the dataset")

        label = self.__labels[index] if self.__labels is not None and index < len(self.__labels) else None

        if self.__eager:
            data = self.__data[index]
        else:
            data_path = self.__data_points[index]
            data = self._data_load(data_path)

        return data, label

    def split(self, percentage: float) -> Tuple['Dataset', 'Dataset']:
        split_index = int(len(self.__data_points) * percentage)
        subsets = tuple(
            self._sub(self.__data_points[start:end], self.__labels[start:end] if self.__labels else None)
            for start, end in [(0, split_index), (split_index, len(self.__data_points))]
        )
        return subsets

    def _sub(self, data_points: List[str], labels: Optional[List[str]]) -> 'Dataset':
        subset = self.__class__(self.__root, self.__eager)
        subset.__data_points = data_points
        subset.__labels = labels
        if self.__eager:
            subset.__data = [subset._data_load(path) for path in data_points]
        return subset
