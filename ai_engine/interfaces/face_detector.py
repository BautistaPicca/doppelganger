from typing import Optional
import numpy as np
from ai_engine.types import BoundingBox

class FaceDetector:
    """
    Interfaz base para detección facial.
    Las implementaciones deben detectar el rostro principal en una imagen y retornar bounding boxes con landmarks.
    """
    def detect(self, image: np.ndarray) -> Optional[BoundingBox]:
        """
        Detecta rostros en una imagen.
        Args:
            image (np.ndarray): Imagen RGB en formato NumPy.
        Returns:
            Un solo [BoundingBox]: El rostro principal detectado con coordenadas y landmarks.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")

