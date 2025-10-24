from typing import Optional
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from ai_engine.types import BoundingBox
from ai_engine.interfaces.face_detector import FaceDetector

class faceDetector(FaceDetector):
    def __init__(self):
        self.detector = MTCNN()

    def detect_from_path(self, image_path: str) -> Optional[BoundingBox]:
        """
        Detecta solo el rostro principal (el de mayor confianza) a partir de la ruta de la imagen
        """
        #Abrimos la imagen con PIL y la convertimos a Numpy
        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)

        return self.detect(image_np)

    def detect(self, image:np.ndarray)-> Optional[BoundingBox]:
        """
        Detecta solo el rosto principal en una imagen de formato NumPy
        """
        detections = self.detector.detect_faces(image)

        if not detections:
            return None #No se detectó ningún rostro
        
        #Se selecciona el rostro con mayor valor de confianza
        main_face = max(detections, key=lambda d: d.get ("confidence", 0.0))

        #Guardamos los datos del bounding box y de las landmarks
        x, y, width, heigth = main_face["box"]
        confidence = main_face.get("confidence", 0.0)
        landmarks = main_face.get("keypoints", {})

        #Crea y devuelve un objeto de tipo Bounding box
        return BoundingBox(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(heigth),
            confidence=float(confidence),
            landmarks={k: (int(v[0]), int(v[1])) for k, v in landmarks.items()}
        )