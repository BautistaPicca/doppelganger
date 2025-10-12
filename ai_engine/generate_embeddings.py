"""
generate_embeddings.py
Genera embeddings faciales de las imágenes del dataset de futbolistas
y guarda la base en embeddings_db.pkl
"""

import os
import torch
import pickle
import numpy as np
from PIL import Image

# Importamos la implementación de la interfaz FaceEmbedder
from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder 
from facenet_pytorch import MTCNN # Necesario para la detección/alineación

# Rutas y configuración
dataset_dir = "ai_engine/FootballPlayers"
db_path = "ai_engine/embeddings_db.pkl"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {device}")

# 1. Inicializar Detector y Embedder
# MTCNN (Detector/Alineador) - se queda aquí para preparar la entrada
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# El Embedder ahora encapsula el modelo FaceNet y la Normalización L2
embedder = FacenetPyTorchEmbedder()

embeddings_db = {}

# Recorremos las imágenes del dataset
for filename in os.listdir(dataset_dir):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(dataset_dir, filename)
        # Extraer nombre (ejemplo: "messi-front.jpg" → "messi")
        name = filename.split("-")[0] 

        # Abrir imagen
        img = Image.open(path)

        # 2. Detectar y Recortar cara (MTCNN)
        # El resultado 'face_tensor' es un tensor de PyTorch alineado (3x160x160)
        face_tensor = mtcnn(img) 
        
        if face_tensor is not None:
            
            # 3. Generar el Embedding (Delegado a la clase FacenetPyTorchEmbedder)
            # El método embed retorna el array NumPy normalizado (1x512)
            embedding_np = embedder.embed(face_tensor) 
            
            # 4. Guardar embedding en la base de datos
            if name not in embeddings_db:
                embeddings_db[name] = []
                
            embeddings_db[name].append(embedding_np) 

# Imprimir resumen
print(f"\nEmbeddings generados para {len(embeddings_db)} jugadores:", list(embeddings_db.keys()))

# Guardar base de embeddings
with open(db_path, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"Base de embeddings guardada en {db_path}")
