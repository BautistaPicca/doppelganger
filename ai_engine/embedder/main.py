
import json
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
import torch
from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from facenet_pytorch import MTCNN
from torchvision import transforms

from ai_engine.utils.pre_processing import pad_and_resize

DATASET_ROOT = Path("run/dataset")
OUTPUT_ROOT = Path("run/processed")

if not DATASET_ROOT.exists():
    raise ValueError(f"El directorio {DATASET_ROOT} no existe.")

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
    
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, device=device)
embedder = FacenetPyTorchEmbedder()

# TODO: Tomar más de una imagen por persona para mejorar los resultados
for person_dir in DATASET_ROOT.iterdir():
    if not person_dir.is_dir():
        continue

    person_id = person_dir.name
    images = sorted(person_dir.glob("*.pgm")) # en el dataset de prueba las imagenes son .pgm

    if not images:
        print(f"Sin imágenes para {person_id}")
        continue

    image_path = images[0]  # solo una por ahora
    image = Image.open(image_path)
    processed = pad_and_resize(image)
    tensor = transforms.ToTensor()(processed)

    try:
        embedding = embedder.embed(tensor)  # (1, 512) Para Nico: ¿Es necesario que tome un tensor?
    except Exception as e:
        print(f"Error al generar embedding para {person_id}: {e}")
        continue

    person_out = OUTPUT_ROOT / person_id
    person_out.mkdir(parents=True, exist_ok=True)

    np.save(person_out / "vec.npy", embedding)

    processed.save(person_out / "image1.jpg")

    # Metadatos de prueba, realmente se guardaran datos como nombre, ¿Equipo? ¿Email?, dependiendo la aplicación,
    # inicialmente solo el nombre para el funcionamiento principal.
    meta = {
        "source_image": str(image_path),
        "embedding_model": "InceptionResnetV1",
        "preprocessing": "pad_and_resize, de 92x112 a 160x160",
        "tag": "celebrity"
    }
    with open(person_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Procesado {person_id} con éxito, path: {image_path.name}")