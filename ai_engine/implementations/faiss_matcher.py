import numpy as np
from typing import List

from ai_engine.interfaces.face_matcher import FaceMatcher
from ai_engine.types import MatchResult
from ai_engine.utils.faiss_index import FaissIndex

import base64

# Temporal!
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class FaissFaceMatcher(FaceMatcher):
    def __init__(self, index: FaissIndex, metric: str = "cosine"):
        if metric != "cosine":
            raise ValueError("Solo se soporta la métrica 'cosine'")
        self.metric = metric
        self.index = index  # instancia de FaissIndex ya construida
    
    def match(self, embedding: np.ndarray, k: int = 5) -> List[MatchResult]:
        results = []
        for name, similarity in self.index.search(embedding, k):
            image_base64 = encode_image(f"run/processed/{name}/image1.jpg")
            results.append(MatchResult(index=0, name=name, similarity=similarity, image=f"data:image/jpeg;base64,{image_base64}"))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
