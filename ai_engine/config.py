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

# Configuraciones de Clasificador 
DATA_DIR = "run/datasets"          
MODEL_DIR = "run/models"

#Tamaño de las imagenes y del batch
IMG_SIZE = 224
BATCH_SIZE = 32

#Cantidad de epocas para entrenamiento
EPOCHS_INITIAL = 30
EPOCHS_FINETUNE = 30
