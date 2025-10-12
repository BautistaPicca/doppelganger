from typing import Dict, Tuple
import numpy as np

class FaceAligner:
    """
    Interfaz base para alineación facial.
    Las implementaciones deben usar landmarks para normalizar la pose del rostro.
    """
    def align(self, image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """
        Alinea el rostro usando landmarks faciales.
        Args:
            image (np.ndarray): Imagen original.
            landmarks (Dict[str, Tuple[int, int]]): Coordenadas de ojos, nariz, boca.
        Returns:
            np.ndarray: Imagen alineada en tamaño estándar.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")