"""
generate_embeddings.py
Genera embeddings faciales de las imágenes del dataset de futbolistas
y guarda la base en embeddings_db.pkl
"""

import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import pickle
import numpy as np

#Dataset de jugadores
dataset_dir = "ai_engine/FootballPlayers"

# Inicializar detector de caras y modelo de embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

embeddings_db = {}

# Recorremos las imágenes del dataset
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(dataset_dir, filename)
        name = filename.split("-")[0]  # ejemplo: "messi-front" → "messi"

        # Abrir imagen
        img = Image.open(path)

        # Detectar y recortar cara
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)  # batch de 1
            embedding = resnet(face).detach().cpu().numpy()

            # Guardar embedding (lista por persona)
            if name not in embeddings_db:
                embeddings_db[name] = []
            embeddings_db[name].append(embedding)

print("Embeddings generados para:", list(embeddings_db.keys()))

# Guardar base de embeddings
with open("ai_engine/embeddings_db.pkl", "wb") as f:
    pickle.dump(embeddings_db, f)

print("Base de embeddings guardada en ai_engine/embeddings_db.pkl")

