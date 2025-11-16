
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import faiss
import numpy as np
from ai_engine.services.base_faiss_index import BaseFaissIndexer

T = TypeVar('T')

@dataclass
class CompositeRecord(Generic[T]):
    """Registro compuesto con vector y datos extra"""
    data: T
    vector: np.ndarray
    
    def to_dict(self) -> dict:
        """Convierte el dato a diccionario para serialización"""
        if hasattr(self.data, '__dict__'):
            return self.data.__dict__
        elif isinstance(self.data, dict):
            return self.data
        else:
            return {'value': self.data}
        
class CompositeIndexerService(BaseFaissIndexer):
    """
    Indexador FAISS para datos complejos (User en nuestro caso)
    """
    def __init__(self, dim: int = 512, from_dir: Optional[str] = None, 
                 data_class: Optional[type] = None):
        """
        Args:
            dim: Dimensión de los vectores
            from_dir: Directorio para cargar índice existente
            data_class: Clase para deserializar datos (ej: User)
        """
        self.data_lookup: Dict[int, Any] = {}
        self.data_class = data_class
        super().__init__(dim, from_dir)
    
    def add(self, records: List[CompositeRecord]):
        """
        Agrega múltiples registros compuestos al índice.
        
        Args:
            records: Lista de CompositeRecord con data y vector
        """
        if not records:
            return
        
        ids = np.arange(self.next_id, self.next_id + len(records), dtype=np.int64)
        
        # Preparar vectores
        vectors = np.stack([r.vector for r in records]).astype('float32')
        faiss.normalize_L2(vectors)
        
        # Agregar al índice
        self.index.add_with_ids(vectors, ids)
        
        # Actualizar lookup con datos completos
        for idx, record in zip(ids, records):
            self.data_lookup[int(idx)] = record.data
        
        self.next_id += len(records)

    def search(self, vector: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Busca los k objetos más similares.
        
        Args:
            vector: Vector de consulta
            k: Número de "vecinos" más cercanos
            
        Returns:
            Lista de tuplas (objeto, similitud) ordenadas por similitud
        """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != index dimension {self.dim}")
        
        # Normalizar
        vector = np.expand_dims(vector, axis=0)
        
        # Limitar k
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        
        distances, indices = self.index.search(vector, k)
        
        # Convertir distancias a similitudes
        similarities = 1 - (distances[0] / 2)
        
        results = []
        for idx, sim in zip(indices[0], similarities):
            if idx == -1:
                continue
            data = self.data_lookup.get(int(idx))
            if data is not None:
                results.append((data, float(sim)))
        
        return results
    
    def save(self, path: Path):
        """
        Guarda el índice y datos en disco.
        
        Args:
            path: Directorio donde guardar
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar índice FAISS
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        serialized_data = {}
        for idx, data in self.data_lookup.items():
            if hasattr(data, '__dict__'):
                # Objeto con atributos
                serialized_data[int(idx)] = {
                    '_type': data.__class__.__name__,
                    '_data': data.__dict__
                }
            elif isinstance(data, dict):
                # Ya es diccionario
                serialized_data[int(idx)] = {
                    '_type': 'dict',
                    '_data': data
                }
            else:
                # Tipo primitivo
                serialized_data[int(idx)] = {
                    '_type': type(data).__name__,
                    '_data': data
                }
        
        # Guardar datos
        with open(path / "data.json", "w", encoding='utf-8') as f:
            json.dump({
                'data_lookup': serialized_data,
                'next_id': self.next_id,
                'dim': self.dim,
                'data_class': self.data_class.__name__ if self.data_class else None
            }, f, ensure_ascii=False, indent=2)
    
    def get_all_data(self) -> List[Any]:
        """Retorna lista de todos los objetos indexados"""
        return list(self.data_lookup.values())

    def _load(self, path: Path):
        """Carga índice y datos complejos desde disco"""
        # Cargar índice FAISS
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Cargar datos
        with open(path / "data.json", "r", encoding='utf-8') as f:
            data = json.load(f)
            
            self.next_id = data.get('next_id', 0)
            self.dim = data.get('dim', self.dim)
            
            serialized_data = data['data_lookup']
            self.data_lookup = {}
            
            for idx_str, obj in serialized_data.items():
                idx = int(idx_str)
                obj_type = obj.get('_type')
                obj_data = obj.get('_data')
                
                # Intentar reconstruir el objeto
                if obj_type == 'dict' or not self.data_class:
                    self.data_lookup[idx] = obj_data
                elif self.data_class and hasattr(self.data_class, '__name__'):
                    # Reconstruir usando la clase proporcionada
                    try:
                        self.data_lookup[idx] = self.data_class(**obj_data)
                    except:
                        self.data_lookup[idx] = obj_data
                else:
                    self.data_lookup[idx] = obj_data