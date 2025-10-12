from pathlib import Path

# Dimensión de los vectores faciales
VECTOR_DIM = 512

# Ruta temporal para almacenar datos faciales "simulados"
STORAGE_ROOT = Path("face_data")
# Ruta para almacenar el índice FAISS
INDEX_ROOT = Path("face_index")

# FAISS config
USE_NORMALIZATION = True
INDEX_TYPE = "FlatL2"