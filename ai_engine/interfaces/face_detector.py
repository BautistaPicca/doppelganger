from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os


def get_single_image_from_folder(folder_path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
    #Asumimos que siempre hay una única imagen válida
    return os.path.join(folder_path, files[0])


def face_detector_single_from_folder(input_folder, output_path, margin=20):
    image_path = get_single_image_from_folder(input_folder)
    print(f"[INFO] Usando imagen: {image_path}")

    img = Image.open(image_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    boxes, _ = mtcnn.detect(img)

    #Asumimos que siempre va a detectar al menos una cara
    x1, y1, x2, y2 = [int(v) for v in boxes[0]]

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.width,  x2 + margin)
    y2 = min(img.height, y2 + margin)

    face = img.crop((x1, y1, x2, y2))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    face.save(output_path, "JPEG")

    print(f"[OK] Guardada en: {output_path}")


if __name__ == "__main__":
    input_folder = r"C:/Users/matii/Desktop/detAux/img"
    output_path = r"C:/Users/matii/Desktop/detAux/procesada/face_detected.jpg"

    face_detector_single_from_folder(input_folder, output_path)
