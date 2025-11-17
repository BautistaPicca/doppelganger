import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ai_engine.interfaces.face_embedder import FaceEmbedder
from ai_engine.utils.pre_processing import pad_and_resize


class VectorizerService:
    """
    Wrapper que envuelve un FaceEmbedder para vectorizar imágenes,
    Se encarga de aplicar un pre-procesamiento básico antes de pasar la imagen al embedder y
    convertir la imagen a tensor.
    """
    def __init__(self, vectorizer: FaceEmbedder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vectorizer = vectorizer

    def vectorize(self, image: Image.Image):
        processed = pad_and_resize(image)
        
        face_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        embedding = self.vectorizer.embed(face_transform(processed))
        return embedding

    # Sólo se utiliza para procesar el dataset "recreativo" de famosos
    def vectorize_dataset(self, dataset_dir: str, output_dir: str):
        """"
        Vectoriza un dataset de imágenes organizado en subdirectorios por persona,
        guardando los embeddings y metadatos en el directorio de salida.
        """
        DATASET_ROOT = Path(dataset_dir)
        OUTPUT_ROOT = Path(output_dir)
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
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
            tensor = preprocess(processed)

            try:
                embedding = self.vectorizer.embed(tensor)  # (1, 512) Para Nico: ¿Es necesario que tome un tensor?
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
                "embedding_model": "model",
                "preprocessing": "pad_and_resize, de 92x112 a 160x160",
                "tag": "celebrity"
            }
            with open(person_out / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Procesado {person_id} con éxito, path: {image_path.name}")
