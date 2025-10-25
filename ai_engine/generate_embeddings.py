import json
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
import torch

# Importación del embedder facial propio basado en FaceNet (InceptionResnetV1)
from ai_engine.implementations.custom_facenet_embedder import CustomFaceNetEmbedder

# Se mantiene MTCNN para detección y alineación facial temporal
from facenet_pytorch import MTCNN
from torchvision import transforms

# Utilidad de preprocesamiento auxiliar (padding, resize, etc.)
from ai_engine.utils.pre_processing import pad_and_resize

# Directorios de entrada y salida del dataset
DATASET_ROOT = Path("run/dataset")
OUTPUT_ROOT = Path("run/processed")

# Validación de existencia de la fuente de datos
if not DATASET_ROOT.exists():
    raise ValueError(f"El directorio {DATASET_ROOT} no existe.")

# Reinicialización del directorio de salida
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Selección automática de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuración del detector facial (MTCNN) — utilizado temporalmente como alineador
# En un escenario completo, este componente debería ser reemplazado por la interfaz FaceAligner.
mtcnn = MTCNN(image_size=160, device=device)

# Instancia del embedder propio para el Escenario 3 (modelo ideal con pesos preentrenados)
embedder = CustomFaceNetEmbedder(load_pretrained=True)

# Procesamiento iterativo de personas dentro del dataset
for person_dir in DATASET_ROOT.iterdir():
    if not person_dir.is_dir():
        continue

    person_id = person_dir.name
    images = sorted(person_dir.glob("*.pgm"))  # El dataset de prueba contiene imágenes .pgm

    if not images:
        print(f"Sin imágenes para {person_id}")
        continue

    # En este prototipo se toma una sola imagen por persona
    image_path = images[0]
    image = Image.open(image_path)

    # --- Pipeline temporal de detección y alineación ---
    # En una versión modular, aquí se integraría FaceAligner.
    # Actualmente, MTCNN cumple el rol de detectar y recortar el rostro.
    face_tensor = mtcnn(image)

    if face_tensor is None:
        print(f"No se detectó rostro en la imagen de {person_id}")
        continue

    # Conversión del tensor alineado (3x160x160) al formato NumPy esperado (HWC)
    # MTCNN devuelve tensores normalizados [0,1]; se revertirá la normalización
    # para dejar que el embedder realice su propio escalado interno.
    aligned_face_np = (face_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    try:
        # Generación del embedding facial (vector 1x512) utilizando el modelo propio
        embedding = embedder.embed(aligned_face_np)
    except Exception as e:
        print(f"Error al generar embedding para {person_id}: {e}")
        continue

    # --- Persistencia de resultados ---
    # Se genera un subdirectorio por persona dentro de la carpeta de salida
    person_out = OUTPUT_ROOT / person_id
    person_out.mkdir(parents=True, exist_ok=True)

    # Guardar el embedding como archivo NumPy
    np.save(person_out / "vec.npy", embedding)

    # Guardar el rostro alineado en formato JPEG para verificación visual
    Image.fromarray(aligned_face_np).save(person_out / "image1.jpg")

    # Registro de metadatos básicos del procesamiento
    meta = {
        "source_image": str(image_path),
        "embedding_model": "InceptionResnetV1 (Propio)",
        "preprocessing": "MTCNN Alignment to 160x160",
        "tag": "celebrity"
    }
    with open(person_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Procesado {person_id} con éxito, path: {image_path.name}")

