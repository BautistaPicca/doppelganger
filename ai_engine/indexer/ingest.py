import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

from ai_engine.types import FaceRecord
from ai_engine.utils.faiss_index import FaissIndex

def get_name(path: Path):
    with open(path / "meta.json", "r") as f:
        return json.load(f)["name"]

def get_vector(path: Path):
    return np.load(path / "vec.npy")

def get_data(path: Path):
    if (path / "vec.npy").exists() and (path / "meta.json").exists():
        return get_name(path), get_vector(path)
    return None

def load_all_faces(root: Path) -> List[Tuple[str, np.ndarray]]:
    records = []
    for folder in root.iterdir():
        if folder.is_dir():
            data = get_data(folder)
            if data:
                records.append(data)
    return records

def build_index(data_dir: Path, index_dir: Path, dim: int = 512):
    data = load_all_faces(data_dir)
    if not data:
        print("No hay datos válidos para indexar.")
        return

    names, vectors = zip(*data)
    vectors = np.stack(vectors).astype("float32")

    index = FaissIndex(dim=dim)
    records = [FaceRecord(name=n, vector=v) for n, v in zip(names, vectors)]
    index.add(records)
    index.save(index_dir)
