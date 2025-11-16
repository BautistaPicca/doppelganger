from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ai_engine.services.base_faiss_index import BaseFaissIndexer

@dataclass
class SimpleRecord:
    """Registro simple con solo nombre y vector"""
    name: str
    vector: np.ndarray
    
"""
Clase utilitaria para trabajar con FAISS.
Permite crear, guardar, cargar y buscar en un índice de vectores.

El propósito principal es indexar vectores faciales para búsqueda rápida.
Se tomó la decisión de usar FAISS por su eficiencia y escalabilidad en cuanto a busquedas;
Nos permitirá manejar grandes cantidades de datos faciales sin sacrificar rendimiento.

Para que FAISS funcione correctamente, los vectores deben tener la misma dimensión y estar normalizados.
"""
class IndexerService(BaseFaissIndexer):
    """
    Indexador FAISS simple que solo almacena nombres.
    Perfecto para casos de uso básicos donde solo se necesita identificar por nombre.
    (ej: Futbolistas)
    """
    def __init__(self, dim: int = 512, from_dir: Optional[str] = None):
        self.name_lookup: Dict[int, str] = {}
        super().__init__(dim, from_dir)

    def add(self, records: List[SimpleRecord]):
        """
        Agrega múltiples registros simples al índice.
        
        Args:
            records: Lista de SimpleRecord con nombre y vector
        """
        if not records:
            return
        
        # Generar IDs consecutivos
        ids = np.arange(self.next_id, self.next_id + len(records), dtype=np.int64)
        
        # Preparar vectores
        vectors = np.stack([r.vector for r in records]).astype('float32')
        faiss.normalize_L2(vectors)
        
        # Agregar al índice
        self.index.add_with_ids(vectors, ids)
        
        # Actualizar lookup
        for idx, record in zip(ids, records):
            self.name_lookup[int(idx)] = record.name
        
        self.next_id += len(records)
    
    def search(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Busca los k nombres más similares.
        
        Args:
            vector: Vector de consulta (debe estar normalizado)
            k: Número de "vecinos" más cercanos
            
        Returns:
            Lista de tuplas (nombre, similitud) ordenadas por similitud
        """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != index dimension {self.dim}")
        
        vector = np.expand_dims(vector, axis=0)
        
        # Limite de k al tamaño del índice
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        
        distances, indices = self.index.search(vector, k)
        
        # Convertir distancias L2 a similitudes [0, 1]
        # similarity = 1 - (distance / 2)
        similarities = 1 - (distances[0] / 2)
        
        results = []
        for idx, sim in zip(indices[0], similarities):
            if idx == -1:
                continue
            name = self.name_lookup.get(int(idx))
            if name:
                results.append((name, float(sim)))
        
        return results
    
    def save(self, path: Path):
        """
        Guarda el índice y lookup en disco.
        
        Args:
            path: Directorio donde guardar (se crean 2 archivos: index y names.json)
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar índice FAISS
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Guardar lookup de nombres
        with open(path / "names.json", "w", encoding='utf-8') as f:
            json.dump({
                'name_lookup': {int(k): v for k, v in self.name_lookup.items()},
                'next_id': self.next_id,
                'dim': self.dim
            }, f, ensure_ascii=False, indent=2)

    def _load(self, path: Path):
        """Carga índice y lookup desde disco"""
        # Cargar índice FAISS
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Cargar lookup
        with open(path / "names.json", "r", encoding='utf-8') as f:
            data = json.load(f)
            self.name_lookup = {int(k): v for k, v in data['name_lookup'].items()}
            self.next_id = data.get('next_id', len(self.name_lookup))
            self.dim = data.get('dim', self.dim)