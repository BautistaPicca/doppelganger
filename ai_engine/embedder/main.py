
import json
from pathlib import Path
import shutil
from PIL import Image, ImageOps
import numpy as np
import torch
from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from facenet_pytorch import MTCNN
from torchvision import transforms

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

"""
Ajusta las dimensiones de la imagen para que se adapte al modelo usado,
En este caso de prueba, InceptionResnetV1, que requiere 160x160.
Las imagenes usadas son de un dataset dónde cada imagen es 92x112,
Por lo que se agrega padding para hacerlas cuadradas y luego se redimensiona con el fin de no distorsionar la imagen.
"""
def pad_and_resize(image: Image.Image, target_size=(160, 160)) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    delta_w = h - w if h > w else 0
    delta_h = w - h if w > h else 0
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded = ImageOps.expand(image, padding, fill=(0, 0, 0))
    resized = padded.resize(target_size, Image.BILINEAR)
    return resized

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