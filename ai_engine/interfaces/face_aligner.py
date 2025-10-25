from abc import ABC, abstractmethod
import numpy as np
from ai_engine.types import BoundingBox

class FaceAligner(ABC):
    """Interfaz base para alineación facial usando BoundingBox + landmarks."""
    @abstractmethod
    def align(self, image_rgb: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Devuelve la cara alineada como np.ndarray (RGB, output_size x output_size x 3)."""
        ...
