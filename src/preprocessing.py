from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class PreprocessingTechnique(ABC):

    @abstractmethod
    def __init__(self, **kwargs: Any):
        self.hyperparameters: Dict[str, Any] = kwargs

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass
