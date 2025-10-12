import numpy as np
import json
from pathlib import Path

def simulate_face_data(root: Path, n_faces: int = 10, dim: int = 512):
    root.mkdir(exist_ok=True)
    for i in range(n_faces):
        name = f"person_{i}"
        vec = np.random.rand(dim).astype("float32")
        folder = root / name
        folder.mkdir(exist_ok=True)
        np.save(folder / "vec.npy", vec)
        with open(folder / "meta.json", "w") as f:
            json.dump({"name": name}, f)
