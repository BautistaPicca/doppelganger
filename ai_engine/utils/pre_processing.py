from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder

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

def get_embedding(image_path: Path, embedder: FacenetPyTorchEmbedder) -> np.ndarray:
    image = Image.open(image_path)
    processed = pad_and_resize(image)
    tensor = transforms.ToTensor()(processed)
    return embedder.embed(tensor)