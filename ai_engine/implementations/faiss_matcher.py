import numpy as np
from typing import List

from ai_engine.interfaces.face_matcher import FaceMatcher
from ai_engine.types import MatchResult
from ai_engine.utils.faiss_index import FaissIndex

class FaissFaceMatcher(FaceMatcher):
    def __init__(self, index: FaissIndex, metric: str = "cosine"):
        if metric != "cosine":
            raise ValueError("Solo se soporta la métrica 'cosine'")
        self.metric = metric
        self.index = index  # instancia de FaissIndex ya construida
    
    def match(self, embedding: np.ndarray, k: int = 5) -> List[MatchResult]:
        distances, ids = self.index.search(embedding, k)

        results = []
        for idx, score in zip(ids, distances):
            name = self.index.name_lookup[int(idx)]
            similarity = round(score * 100, 2)
            results.append(MatchResult(name=name, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
