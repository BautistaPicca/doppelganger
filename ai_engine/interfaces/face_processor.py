from typing import List

import numpy as np
from ai_engine.types import BoundingBox
from PIL import Image

class FaceProcessor:
    """
    Interfaz base para detección facial.
    Las implementaciones deben detectar rostros en una imagen y ajustar el tamaño de la misma
    """
    def process(self, image: Image.Image) -> np.ndarray:
        """
        Detecta rostros en una imagen y la recorta, añade bordes, etc.
        Args:
            image (np.ndarray): Imagen RGB en formato NumPy.
        Returns:
            array[np.ndarray]: Lista de imágenes faciales procesadas.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")