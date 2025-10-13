from pathlib import Path

import numpy as np

from ai_engine.types import FaceRecord
from ai_engine.utils.faiss_index import FaissIndex

# Path donde se encuentran los datos procesados
PROCESSED_DIR = Path("run/processed")
# Path donde se almacenará el índice FAISS y los labels
INDEX_PATH = Path("run/index")

index = FaissIndex(dim=512)

records = []
for person_dir in sorted(PROCESSED_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    vec_path = person_dir / "vec.npy"
    if not vec_path.exists():
        print(f"Falta vec.npy en {person_dir.name}, ignorando...")
        continue

    try:
        vector = np.load(vec_path)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        elif vector.shape[0] != 1:
            print(f"Vector inválido para {vec_path.name}")
            continue

        record = FaceRecord(name=person_dir.name, vector=vector[0])
        records.append(record)
        print(f"Se ha cargado a {record.name} correctamente.")

    except Exception as e:
        print(f"Error al cargar a {person_dir.name}: {e}")

# Agregar los registros al índice
index.add(records)

index.save(INDEX_PATH)
print(f"Índice guardado en {INDEX_PATH} con {len(records)} registros.")