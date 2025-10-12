from typing import Optional, Tuple, Dict, List, NamedTuple
import numpy as np
from dataclasses import dataclass

class BoundingBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Dict[str, Tuple[int, int]]

@dataclass
class MatchResult:
    index: int
    similarity: float

@dataclass
class FaceRecord:
    name: str
    vector: np.ndarray
    metadata: Optional[Dict] = None