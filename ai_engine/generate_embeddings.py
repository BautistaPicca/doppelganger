"""
generate_embeddings.py
Genera embeddings faciales de las imágenes del dataset de futbolistas
y guarda la base en embeddings_db.pkl
"""

import os
import torch
import torch.nn.functional as F # Importamos F para la normalización
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import pickle
import numpy as np

# Ruta del dataset de jugadores
dataset_dir = "ai_engine/FootballPlayers"

# Inicializar detector de caras y modelo de embeddings
# Determina si usar CUDA (GPU) o CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# MTCNN (Detector/Alineador) - prepara la imagen para el FaceNet (160x160)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# ResNet (Modelo de Embeddings) - FaceNet InceptionResnetV1
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

embeddings_db = {}

# Recorremos las imágenes del dataset
for filename in os.listdir(dataset_dir):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(dataset_dir, filename)
        # Extraer nombre (ejemplo: "messi-front.jpg" → "messi")
        name = filename.split("-")[0] 

        # Abrir imagen
        img = Image.open(path)

        # 1. Detectar y Recortar cara (MTCNN)
        # El resultado 'face' es un tensor de PyTorch alineado (3x160x160)
        face = mtcnn(img) 
        
        if face is not None:
            # 2. Preparar para el modelo: Añadir dimensión de Batch y mover al dispositivo
            face = face.unsqueeze(0).to(device) # Convierte a (1x3x160x160)
            
            # Desactiva el cálculo de gradientes
            with torch.no_grad():
                # 3. Generar el Embedding
                embedding = resnet(face)

                # 4. NORMALIZACIÓN L2 (Paso crucial para FaceNet)
                # Normaliza el vector para que tenga longitud unitaria (512 dimensiones)
                embedding = F.normalize(embedding, p=2, dim=1)
                
            # Mover el tensor a la CPU y convertirlo a array NumPy para guardar
            embedding_np = embedding.cpu().numpy()

            # 5. Guardar embedding en la base de datos
            if name not in embeddings_db:
                embeddings_db[name] = []
            
            # Guardamos el array NumPy (de tamaño 1x512)
            embeddings_db[name].append(embedding_np) 

# Imprimir resumen
print(f"\nEmbeddings generados para {len(embeddings_db)} jugadores:", list(embeddings_db.keys()))

# Guardar base de embeddings
db_path = "ai_engine/embeddings_db.pkl"
with open(db_path, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"Base de embeddings guardada en {db_path}")
