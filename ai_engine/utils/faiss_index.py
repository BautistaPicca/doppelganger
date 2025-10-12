import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ai_engine.types import FaceRecord

"""
Clase utilitaria para trabajar con FAISS.
Permite crear, guardar, cargar y buscar en un índice de vectores.

El propósito principal es indexar vectores faciales para búsqueda rápida.
Se tomó la decisión de usar FAISS por su eficiencia y escalabilidad en cuanto a busquedas;
Nos permitirá manejar grandes cantidades de datos faciales sin sacrificar rendimiento.

Para que FAISS funcione correctamente, los vectores deben tener la misma dimensión y estar normalizados.
"""
class FaissIndex:
    """
    Inicializa un índice FAISS.
    Args:
        dim (int): Dimensión de los vectores.
        from_dir (Optional[str]): Ruta al directorio desde donde cargar un índice existente.
    """
    def __init__(self, dim: int = 512, from_dir: Optional[str] = None):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.name_lookup: Dict[int, str] = {}
        if from_dir:
            self._load(from_dir)

    """Agrega múltiples registros al índice.
    Args:
        records (List[FaceRecord]): Lista de registros faciales a agregar.    
    """
    def add(self, records: List[FaceRecord]):
        ids = np.arange(len(self.name_lookup), len(self.name_lookup) + len(records))
        vectors = np.stack([r.vector for r in records]).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add_with_ids(vectors, ids)
        for i, r in zip(ids, records):
            self.name_lookup[i] = r.name
            

    """Busca los k vectores más similares en el índice.
    Args:
        vector (np.ndarray): Vector de consulta.
        k (int): Número de vecinos más cercanos a buscar.
    Returns:
        List[Tuple[str, float]]: Lista de tuplas (nombre, similitud) de los k vecinos más cercanos.
    """
    def search(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        faiss.normalize_L2(vector.reshape(1, -1))
        D, I = self.index.search(vector.reshape(1, -1), k)
        return D[0], I[0] # Retorna distancias e índices

    """
    Guarda el índice FAISS y su mapeo de nombres en el disco.
    Args:
        path (Path): Ruta al directorio donde se guardará el índice y el archivo de nombres.
    """
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index"))
        with open(path / "names.json", "w") as f:
            json.dump({int(k): v for k, v in self.name_lookup.items()}, f)
            """json.dump(self.name_lookup, f)"""

    """
    Carga un índice FAISS y su mapeo de nombres desde el disco.
    Args:
        path (Path): Ruta al directorio que contiene el índice y el archivo de nombres.
    """
    def _load(self, path: Path):
        self.index = faiss.read_index(str(path / "index"))
        with open(path / "names.json", "r") as f:
            self.name_lookup = json.load(f)