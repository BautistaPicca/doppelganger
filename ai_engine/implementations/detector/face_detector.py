from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

from ai_engine.interfaces.face_processor import FaceProcessor

# Inicializamos el detector de rostros MTCNN con tamaño de salida 160x160
# y un margen adicional para capturar mejor la región facial.
mtcnn = MTCNN(image_size=160, margin=40)

class ImageFaceProcessor(FaceProcessor):
    def process(img):
        """Detecta un rostro en una imagen y devuelve un array RGB 224x224.
        Si no se encuentra ninguna cara, retorna None."""

        # Asegurarse de que la imagen sea RGB
        pil_img = img.convert("RGB")

        # Detecta y recorta el rostro
        face_tensor = mtcnn(pil_img)

        if face_tensor is None:
            print("No se detectó ninguna cara")
            return None

        # Convertir tensor (C,H,W) → PIL Image y forzar RGB
        face_pil = Image.fromarray(((face_tensor.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255).astype("uint8"))
        face_pil = face_pil.convert("RGB").resize((224, 224))

        # Devolver como numpy array
        return np.array(face_pil)


