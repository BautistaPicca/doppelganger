import numpy as np
import cv2
from ai_engine.types import BoundingBox
from ai_engine.interfaces.face_aligner import FaceAligner as FaceAlignerInterface

class FaceAligner(FaceAlignerInterface):
    """
    Implementación concreta del alineador facial.
    Alinea una cara usando los landmarks del BoundingBox
    y devuelve la imagen alineada en formato RGB (np.ndarray).
    """

    def __init__(self, output_size: int = 160):
        """
        Inicializa el alineador con el tamaño de salida deseado (cuadrado).
        Escala las coordenadas base del modelo ArcFace (112x112)
        al tamaño indicado.
        """
        self.output_size = output_size

        # Escalar coordenadas de referencia (ArcFace 112x112)
        s = float(output_size) / 112.0
        self.dst_left_eye = (38.2946 * s, 51.6963 * s)
        self.dst_right_eye = (73.5318 * s, 51.5014 * s)
        dst_mouth_left = (41.5493 * s, 92.3655 * s)
        dst_mouth_right = (70.7299 * s, 92.2041 * s)

        # Calcular el punto medio entre las comisuras de la boca
        self.dst_mouth_center = (
            (dst_mouth_left[0] + dst_mouth_right[0]) / 2,
            (dst_mouth_left[1] + dst_mouth_right[1]) / 2
        )

    def align(self, image_rgb: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Alinea la cara de 'image_rgb' usando los landmarks del BoundingBox.
        Retorna una imagen RGB alineada de tamaño (output_size x output_size x 3).
        """
        # Puntos fuente (landmarks detectados)
        le = bbox.landmarks["left_eye"]
        re = bbox.landmarks["right_eye"]
        ml = bbox.landmarks["mouth_left"]
        mr = bbox.landmarks["mouth_right"]

        # Punto central de la boca (más estable que usar uno solo)
        mouth_center = (
            (ml[0] + mr[0]) / 2,
            (ml[1] + mr[1]) / 2
        )

        # Puntos fuente y destino para la transformación
        src = np.float32([le, re, mouth_center])
        dst = np.float32([self.dst_left_eye, self.dst_right_eye, self.dst_mouth_center])

        # Calcular la transformación afín (rotación + escala + traslación)
        M = cv2.getAffineTransform(src, dst)

        # Aplicar warp afín a la imagen original
        aligned = cv2.warpAffine(
            image_rgb,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return aligned

