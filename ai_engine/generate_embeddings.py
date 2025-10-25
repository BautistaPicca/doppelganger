import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# -------------------------------------------------------
# Importación de módulos propios del sistema
# -------------------------------------------------------
from ai_engine.implementations.custom_facenet_embedder import CustomFaceNetEmbedder
from ai_engine.implementations.faceDetector import faceDetector 
from ai_engine.implementations.face_aligner import FaceAligner 

# -------------------------------------------------------
# Configuración de rutas y entorno
# -------------------------------------------------------
DATASET_ROOT = Path("run/dataset")
OUTPUT_ROOT = Path("run/processed")

# Validación de existencia de dataset
if not DATASET_ROOT.exists():
    raise ValueError(f"❌ El directorio de dataset no existe: {DATASET_ROOT}")

# Reinicialización del directorio de salida
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Selección automática de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Ejecutando en dispositivo: {device}")


# -------------------------------------------------------
# Instanciación de módulos del pipeline
# -------------------------------------------------------


detector = faceDetector()                         #  Detector facial
aligner = FaceAligner(output_size=160)            #  Alineador de rostros
embedder = CustomFaceNetEmbedder(load_pretrained=True)  #  Generador de embeddings


# -------------------------------------------------------
# Procesamiento iterativo del dataset
# -------------------------------------------------------
for person_dir in DATASET_ROOT.iterdir():
    if not person_dir.is_dir():
        continue

    person_id = person_dir.name
    images = sorted(person_dir.glob("*.pgm"))  # Dataset ORL o similar

    if not images:
        print(f"⚠️  Sin imágenes para {person_id}")
        continue

    # Se procesa solo la primera imagen por persona (puede ampliarse fácilmente)
    image_path = images[0]
    image = Image.open(image_path).convert('RGB')



	# CORRECCIÓN 3: Conversión de PIL a NumPy ANTES de llamar al Detector y Alineador
    image_np = np.array(image_pil)

    # -------------------------------------------------------
    # ETAPA 1: Detección de rostro
    # -------------------------------------------------------
    bounding_boxes = detector.detect(image)

    if bounding_boxes is None or len(bounding_boxes) == 0:
        print(f"❌ No se detectó rostro en la imagen de {person_id}")
        continue

    # Asume una única cara por imagen
    bounding_box = bounding_boxes[0]

    try:
        # -------------------------------------------------------
        # ETAPA 2: Alineamiento facial
        # -------------------------------------------------------
        aligned_face_np = aligner.align(image, bounding_box)

        # -------------------------------------------------------
        # ETAPA 3: Extracción del embedding
        # -------------------------------------------------------
        embedding = embedder.embed(aligned_face_np)  # np.ndarray (1, 512)

    except Exception as e:
        print(f"⚠️  Error en el pipeline para {person_id}: {e}")
        continue

    # -------------------------------------------------------
    # Persistencia de resultados
    # -------------------------------------------------------
    person_out = OUTPUT_ROOT / person_id
    person_out.mkdir(parents=True, exist_ok=True)

    # Guardar embedding y rostro alineado
    np.save(person_out / "vec.npy", embedding)
    Image.fromarray(aligned_face_np).save(person_out / "image1.jpg")

    # Guardar metadatos de procesamiento
    meta = {
        "source_image": str(image_path),
        "embedding_model": "InceptionResnetV1 (Propio)",
        "preprocessing": (
            f"Detector: {detector.__class__.__name__}, "
            f"Alineador: {aligner.__class__.__name__}, Output: 160x160"
        ),
        "tag": "celebrity"
    }

    with open(person_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Procesado {person_id} con éxito — {image_path.name}")


print("\n🎯 Pipeline completado con éxito. Resultados disponibles en:")
print(f"→ {OUTPUT_ROOT.resolve()}")


