from typing import List
import numpy as np
from ai_engine.types import MatchResult

class FaceMatcher:
    """
    Interfaz base para comparación de embeddings faciales.
    Las implementaciones deben calcular similitud entre vectores y retornar los matches más cercanos.
    """
    def match(self, embedding: np.ndarray, database: List[np.ndarray]) -> List[MatchResult]:
        """
        Compara un embedding contra una base de datos.
        Args:
            embedding (np.ndarray): Vector facial a comparar.
            database (List[np.ndarray]): Lista de embeddings registrados.
        Returns:
            List[MatchResult]: Matches ordenados por similitud.
        """
        raise NotImplementedError("Este método debe ser implementado por una subclase.")