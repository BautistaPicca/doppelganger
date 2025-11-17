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
        
            # Ejecutamos la detección y recorte de rostro
    face = mtcnn(img)

    # Si MTCNN no detecta un rostro, devolvemos None
    if face is None:
        print("No se detectó ninguna cara en:", image_path)
        return None

    # Convertir tensor (C,H,W) → numpy (H,W,C)
    face_np = face.permute(1, 2, 0).detach().cpu().numpy()

    # MTCNN entrega valores en [-1, 1]; reescalamos a [0, 255]
    face_np = ((face_np + 1) / 2.0 * 255).astype("uint8")

    # Convertimos a imagen PIL para redimensionar fácilmente a 224x224
    face_img = Image.fromarray(face_np)
    face_img = face_img.resize((224, 224))

    # Devolvemos el rostro como array numpy RGB
    return np.array(face_img)


