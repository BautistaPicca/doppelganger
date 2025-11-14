import faiss
from typing import Optional, List, Tuple, Any
import numpy as np
from pathlib import Path

class BaseFaissIndexer:
    """
    Clase base abstracta para indexadores FAISS.
    Define la interfaz común para ambas implementaciones.
    """
    def __init__(self, dim: int = 512, from_dir: Optional[str] = None):
        """
        Inicializa un índice FAISS.
        
        Args:
            dim: Dimensión de los vectores (default: 512 para FaceNet)
            from_dir: Ruta al directorio desde donde cargar índice existente
        """
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        self.next_id = 0
        
        if from_dir:
            self._load(Path(from_dir))

def search(self, vector: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Busca los k vectores más similares.
        
        Args:
            vector: Vector de consulta (1D o 2D)
            k: Número de vecinos más cercanos
            
        Returns:
            Lista de tuplas (dato, similitud) ordenadas por similitud descendente
        """
        raise NotImplementedError("Debe implementarse en subclases")

def add(self, records: List[Any]):
        """Agrega múltiples registros al índice"""
        raise NotImplementedError("Debe implementarse en subclases")
    
def save(self, path: Path):
    """Guarda el índice en disco"""
    raise NotImplementedError("Debe implementarse en subclases")

def _load(self, path: Path):
    """Carga el índice desde disco"""
    raise NotImplementedError("Debe implementarse en subclases")