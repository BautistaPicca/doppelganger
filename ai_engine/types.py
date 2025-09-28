from typing import Tuple, Dict, List, NamedTuple
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
    score: float