import numpy as np

class FaceEmbedder:
    """
    Interfaz base para generación de embeddings faciales.
    Las implementaciones deben convertir una imagen alineada en un vector de características.
    """
    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Genera un embedding facial normalizado.
        Args:
            aligned_face (np.ndarray): Imagen facial alineada.
        Returns:
            np.ndarray: Vector de características (embedding) normalizado.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")