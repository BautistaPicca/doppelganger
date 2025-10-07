import faiss
import numpy as np
from typing import List

from ai_engine.interfaces.face_matcher import FaceMatcher
from ai_engine.types import MatchResult

class FaissFaceMatcher(FaceMatcher):
    def __init__(self, metric: str = "cosine"):
        """
        metric: 'cosine'
        """
        if metric != "cosine":
            raise ValueError("Solo se soporta la métrica 'cosine'")
        self.metric = metric
        self.index = None
        self.database = None

    def _build_index(self, database: List[np.ndarray]):
        embeddings = np.stack(database).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def match(self, embedding: np.ndarray, database: List[np.ndarray]) -> List[MatchResult]:
        if self.index is None or self.database is not database:
            self.index = self._build_index(database)
            self.database = database

        query = embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        distances, indices = self.index.search(query, k=len(database))

        results = []
        for i, score in zip(indices[0], distances[0]):
            similarity = round(score * 100, 2)
            results.append(MatchResult(i, similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
