from typing import Dict, Tuple, Optional, List
import numpy as np
import os
import cv2  # OpenCV para estimar la afin y warpear
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN


class FaceAligner:
    
    # Template de 5 puntos (InsightFace 112x112) escalado a 160x160
    # Se encarga de mapear con coordenadas (x,y) la posicion de ojos, nariz, boca.
    _TEMPLATE_160 = np.array(
        [
            [54.706571428571436, 73.85185714285714],  # ojo_izquierdo
            [105.04542857142857, 73.57342857142856],  # ojo_derecho
            [80.036,            102.48085714285713],  # nariz
            [59.35614285714286, 131.9507142857143],   # boca_izquierda
            [101.04271428571428,131.72014285714286],  # boca_derecha
        ],
        dtype=np.float32
    )

    def __init__(self):
        """Inicializa el alineador con el template estándar de 160x160."""
        self.output_size = 160
        self.template = self._TEMPLATE_160.copy()

    def align(self, image: np.ndarray, landmarks: Dict[str, Tuple[float, float]] ) -> Optional[np.ndarray]:
        """
        Devuelve la cara alineada o None si no se pudo alinear.
        Verifica que la imagen exista, tenga 3 dimensiones y 3 canales (RGB)
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return None
        
        """se asegura de que required contenga las 5 partes del rostro que necesita para calcular la alineación.""" 
        required = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        for k in required:
            if k not in landmarks or landmarks[k] is None:
                return None

        """arma una matriz 5×2 con las coordenadas (x, y) de los puntos clave en la imagen original."""
        src = np.array(
            [
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["nose"],
                landmarks["mouth_left"],
                landmarks["mouth_right"],
            ],
            dtype=np.float32,
        )

        """busca la mejor matriz M que mueva esos cinco puntos del rostro real (src) 
        para que coincidan con los cinco puntos del rostro ideal (self.template)."""
        M, _ = cv2.estimateAffinePartial2D(src, self.template, method=cv2.LMEDS)
        if M is None:
            return None

        """devuelve la cara alineada (enderezada y centrada)"""
        aligned = cv2.warpAffine(image, M, (self.output_size, self.output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if aligned.dtype != np.uint8:
            aligned = np.clip(aligned, 0, 255).astype(np.uint8)

        return aligned

    def align_folder(self, input_dir: str, output_dir: str) -> str:
    
    #Recorre input_dir, detecta landmarks con MTCNN, alinea y guarda en output_dir.
    #Asume que 'output_dir' ya existe y que las imágenes provienen del face_detector.
    
    # Usa GPU si está disponible, de lo contrario CPU
     device = "cuda:0" if torch.cuda.is_available() else "cpu"
     mtcnn = MTCNN(keep_all=False, device=device)

     # Recorremos todos los archivos dentro del directorio de entrada
     files = os.listdir(input_dir)

     for fname in files:
        in_path = os.path.join(input_dir, fname)

        # Abre la imagen con PIL y corrige la orientación según los metadatos EXIF
        img = ImageOps.exif_transpose(Image.open(in_path).convert("RGB"))

        # Detectar landmarks
        _, _, lms = mtcnn.detect(img, landmarks=True)
        if lms is None:
            print(f"[NO FACE] {fname}")
            continue

        # Convierte la imagen a un array y toma los 5 puntos detectados (lm)
        img_np = np.array(img)
        lm = lms[0]  # una sola cara esperada
        landmarks_dict = {
            "left_eye":    tuple(lm[0]),
            "right_eye":   tuple(lm[1]),
            "nose":        tuple(lm[2]),
            "mouth_left":  tuple(lm[3]),
            "mouth_right": tuple(lm[4]),
        }

        # Llamo de nuevo a align
        aligned = self.align(img_np, landmarks_dict)
        if aligned is None:
            print(f"[ALIGN FAIL] {fname}")
            continue

        # Guarda la imagen alineada en el directorio de salida, convirtiendo de RGB a BGR (OpenCV).
        out_path = os.path.join(output_dir, fname)
        ok = cv2.imwrite(out_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

     return output_dir

# -------------------------- Ejecucion -----------------------
if __name__ == "__main__":
    
    INPUT_DIR  = r"C:\Users\matii\Desktop\Recortados\_root"
    OUTPUT_DIR = r"C:\Users\matii\Desktop\Recortados\_root_aligned"

    aligner = FaceAligner()
    out = aligner.align_folder(INPUT_DIR, OUTPUT_DIR)
    print(f"\n[LISTO] Imágenes alineadas en: {out}")
