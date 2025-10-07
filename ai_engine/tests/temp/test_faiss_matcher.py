import numpy as np
from ai_engine.implementations.cosine_matcher import CosineFaceMatcher
from ai_engine.implementations.faiss_matcher import FaissFaceMatcher


matcher = FaissFaceMatcher()
query = np.random.rand(128).astype("float32")
db = [np.random.rand(128).astype("float32") for _ in range(10)]

results = matcher.match(query, db)
for r in results[:10]:
    print(f"Match index: {r.index:<2} | Similitud: {r.similarity:>5.2f}%")