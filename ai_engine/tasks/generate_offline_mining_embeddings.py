import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

"""
Nota: Este script se utilizó para generar los embeddings de todo un dataset, para intentar utilizarlos en hard triplet mining,

El triplet mining online u offline común requiere muchos recursos tanto de VRAM como de RAM, por lo que se optó por generar los embeddings previamente y luego hacer el triplet mining sobre ellos.
Esto solo se utilizó para intentar entrenar el vectorizador, es decir, la aplicación secundaria del proyecto.
"""
DATASET_DIR = "run/output/vectorizer_dataset_filtered"  # Carpeta con subcarpetas por persona
OUTPUT_EMB_FILE = "run/output/embeddings.npy"
OUTPUT_LABELS_FILE = "run/output/labels.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 160  # Tamaño de entrada requerido por InceptionResnetV1
BATCH_SIZE = 32

# Transformación de imágenes
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normaliza a [-1, 1]
])

# Función para cargar imagen
def load_image(path):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)

# Cargar modelo preentrenado
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Inicializar listas para embeddings y labels
embeddings = []
labels = []
img_paths_list = []

people = sorted(os.listdir(DATASET_DIR))
with torch.no_grad():
    for person in tqdm(people, desc="Generando embeddings"):
        person_dir = os.path.join(DATASET_DIR, person)
        img_paths = glob(os.path.join(person_dir, "*.jpg")) + \
                    glob(os.path.join(person_dir, "*.png")) + \
                    glob(os.path.join(person_dir, "*.jpeg"))
        
        for img_path in img_paths:
            img_tensor = load_image(img_path).to(DEVICE)
            emb = model(img_tensor).squeeze(0).cpu().numpy()  # 128 dim
            embeddings.append(emb)
            labels.append(person)
            img_paths_list.append(os.path.basename(img_path))  # solo el nombre del archivo

# Guardar resultados
embeddings = np.array(embeddings)
labels = np.array(labels)

np.save(OUTPUT_EMB_FILE, embeddings)
np.save(OUTPUT_LABELS_FILE, labels)
np.save("run/output/img_paths.npy", np.array(img_paths_list))
print(img_paths_list[0])
print(f"Embeddings guardados en {OUTPUT_EMB_FILE} y labels en {OUTPUT_LABELS_FILE}")
