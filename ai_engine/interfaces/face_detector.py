from typing import List
import numpy as np
from ai_engine.types import BoundingBox

class FaceDetector:
    """
    Interfaz base para detección facial.
    Las implementaciones deben detectar rostros en una imagen y retornar bounding boxes con landmarks.
    """
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detecta rostros en una imagen.
        Args:
            image (np.ndarray): Imagen RGB en formato NumPy.
        Returns:
            List[BoundingBox]: Lista de rostros detectados con coordenadas y landmarks.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")