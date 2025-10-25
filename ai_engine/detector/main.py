import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from pathlib import Path
from ai_engine.implementations.faceDetector import faceDetector

def main():
    parser = argparse.ArgumentParser(description= "detectar rostro principal en una imagen")
    parser.add_argument("image", type=str, help="Ruta de la imagen a utilizar")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: la imagen '{image_path}' no existe.")
        return
    
    #Creamos el detector
    detector = faceDetector()

    #Detectamos el rostro principal
    face_box = detector.detect_from_path(str(image_path))

    if face_box:
        print("Rostro principal detectado...")
    else:
        print("No se detecto ningun rostro")
    
if __name__ == "__main__":
    main()