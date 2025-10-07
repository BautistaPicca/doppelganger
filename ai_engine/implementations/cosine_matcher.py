import numpy as np
from typing import List
from ai_engine.interfaces.face_matcher import FaceMatcher
from ai_engine.types import MatchResult

class CosineFaceMatcher(FaceMatcher):
    def match(self, embedding: np.ndarray, database: List[np.ndarray]) -> List[MatchResult]:
        results = []
        for idx, candidate in enumerate(database):
            score = np.dot(embedding, candidate)  # Similitud coseno
            results.append(MatchResult(index=idx, similarity=score))

        # Ordenar por "score" descendente
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
