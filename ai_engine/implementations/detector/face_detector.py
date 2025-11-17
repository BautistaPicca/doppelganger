from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Inicializamos el detector de rostros MTCNN con tamaño de salida 160x160
# y un margen adicional para capturar mejor la región facial.
mtcnn = MTCNN(image_size=160, margin=40)

def detect_and_crop_face(image_path):
    """Detecta un rostro en una imagen y devuelve un array RGB 224x224.
       Si no se encuentra ninguna cara, retorna None."""

    try:
        # Abrimos la imagen y la convertimos a RGB para evitar inconsistencias
        img = Image.open(image_path).convert("RGB")
    except:
        print("No se pudo abrir la imagen:", image_path)
        return None

