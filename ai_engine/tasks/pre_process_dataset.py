import argparse
import os
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Configuración
DATASET_PATH = "run/datasets/celebrities"
OUTPUT_PATH = "run/output/celebrities_cleaned"
REJECTED_PATH = "run/output/celebrities_rejected_images"
MODELS_PATH = "run/models"  # Carpeta donde se guardarán los modelos
IMG_SIZE = 224
MIN_CONFIDENCE = 0.7  # Confianza mínima para aceptar un rostro
MIN_FACE_SIZE = 0.05  # Tamaño mínimo del rostro

# Validaciones más estrictas, tener en cuenta que son MUY estrictas si se van a usar
REQUIRE_BOTH_EYES = False  # Requiere que se detecten ambos ojos en una imagen
CHECK_FACE_SYMMETRY = False  # Verifica que el rostro no esté muy de lado
MAX_FACE_ANGLE = 35  # Ángulo máximo permitido, complementa a CHECK_FACE_SYMMETRY

def choose_directory(title="Seleccione una carpeta"):
    """
    Abre el explorador de archivos del sistema operativo para seleccionar una carpeta
    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    selected = filedialog.askdirectory(title=title)
    root.destroy()
    return selected


def load_face_detector():
    """Carga el modelo DNN de detección facial de OpenCV"""
    # Crear carpeta de modelos si no existe
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # nombre del modelo y configuración
    model_file = os.path.join(MODELS_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
    config_file = os.path.join(MODELS_PATH, "deploy.prototxt")
    
    # Por si no están instalados, se descargan automáticamente
    if not os.path.exists(model_file):
        print(f"Descargando modelo de detección facial en: {MODELS_PATH}")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_file
        )
        print(f"Modelo descargado: {model_file}")
    
    if not os.path.exists(config_file):
        print(f"Descargando configuración del modelo...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            config_file
        )
        print(f"Configuración descargada: {config_file}")
    
    print(f"Cargando modelo desde: {MODELS_PATH}")
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# Nota: Esto fue un intento de mejorar la detección en general de rostros pero resultó demasiado estricto incluso modificando parámetros
def load_eye_detector():
    """Carga el detector de ojos Haar Cascade"""
    eye_cascade_path = os.path.join(MODELS_PATH, "haarcascade_eye.xml")
    
    # Se intenta cargar desde local
    if os.path.exists(eye_cascade_path):
        print(f"Cargando detector de ojos desde: {eye_cascade_path}")
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if not eye_cascade.empty():
            return eye_cascade
    
    # Si no se encontró, se intenta usar el modelo que viene con OpenCV por defecto
    print("Intentando usar detector de ojos incluido en OpenCV...")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if not eye_cascade.empty():
        print("Detector de ojos cargado desde OpenCV")
        # Guardamos en local
        try:
            os.makedirs(MODELS_PATH, exist_ok=True)
            import shutil
            shutil.copy(cv2.data.haarcascades + 'haarcascade_eye.xml', eye_cascade_path)
            print(f"Copia guardada en: {eye_cascade_path}")
        except:
            pass
        return eye_cascade
    
    # Si falla, intentar descargar
    print(f"Descargando detector de ojos desde GitHub...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
            eye_cascade_path,
            timeout=30
        )
        print(f"Detector de ojos descargado: {eye_cascade_path}")
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        return eye_cascade
    except Exception as e:
        print(f"Error descargando detector de ojos: {e}")
        print("Por favor descarga manualmente desde:")
        print("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml")
        print(f"Y guárdalo en: {eye_cascade_path}")
        raise

def detect_faces(image, net, confidence_threshold=0.7):
    """
    Detecta rostros en una imagen usando DNN de OpenCV
    Retorna: lista de (x, y, w, h, confidence)
    """
    h, w = image.shape[:2]
    
    # Preparar la imagen para el modelo
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            # Asegurar que las coordenadas estén dentro de la imagen
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_w = x2 - x
            face_h = y2 - y
            
            faces.append((x, y, face_w, face_h, confidence))
    
    return faces

def detect_eyes_in_face(face_roi, eye_cascade):
    """
    Detecta ojos en una región facial
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15),
        maxSize=(int(face_roi.shape[1] * 0.4), int(face_roi.shape[0] * 0.4))
    )
    return len(eyes), eyes

def check_face_quality(image, face_box, eye_cascade):
    """
    Verifica la calidad del rostro detectado, es decir, si se pasa o no al nuevo dataset limpio.
    Requisitos:
    - Ambos ojos visibles
    - Rostro no muy de lado
    Retorna: Si es valido y el motivo de rechazo
    """
    x, y, w, h, confidence = face_box
    
    # Extraer región del rostro con margen para mejor detección
    margin = int(w * 0.1)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
    
    face_roi = image[y1:y2, x1:x2]
    
    if face_roi.size == 0:
        return False, "face_roi_empty"
    
    if REQUIRE_BOTH_EYES:
        num_eyes, eyes = detect_eyes_in_face(face_roi, eye_cascade)
        # Aceptar si hay 2 ojos
        if num_eyes < 2:
            return False, "missing_eyes"
        
        h_roi = face_roi.shape[0]
        valid_eyes = [e for e in eyes if e[1] < h_roi * 0.65]
        
        if len(valid_eyes) < 2:
            return False, "eyes_misplaced"
        
        # Verificar simetría aproximada de los ojos (no muy de lado)
        if CHECK_FACE_SYMMETRY and len(valid_eyes) >= 2:
            # Ordenar ojos por posición X
            sorted_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
            left_eye = sorted_eyes[0]
            right_eye = sorted_eyes[1]
            
            # Calcular diferencia vertical entre ojos
            eye_y_diff = abs((left_eye[1] + left_eye[3]//2) - (right_eye[1] + right_eye[3]//2))
            eye_distance = abs(right_eye[0] - left_eye[0])
            
            if eye_distance > 10:
                angle = np.degrees(np.arctan(eye_y_diff / eye_distance))
                if angle > MAX_FACE_ANGLE:
                    return False, "face_tilted"
    
    # Verificar brillo/contraste
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 20 or mean_brightness > 235:
        return False, "poor_lighting"
    
    return True, "valid"

def get_best_face(faces, img_width, img_height, min_face_size=0.15):
    """
    Selecciona el mejor rostro, por si en una imagen hay varios detectados
    """
    if not faces:
        return None
    
    best_face = None
    best_score = 0
    
    center_x, center_y = img_width / 2, img_height / 2
    
    for (x, y, w, h, confidence) in faces:
        # Calcular tamaño relativo
        face_area = w * h
        img_area = img_width * img_height
        size_ratio = face_area / img_area
        
        # Filtrar rostros muy pequeños
        if size_ratio < min_face_size:
            continue
        
        # Calcular distancia al centro
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        distance_to_center = np.sqrt(
            (face_center_x - center_x)**2 + 
            (face_center_y - center_y)**2
        )
        max_distance = np.sqrt(center_x**2 + center_y**2)
        centrality_score = 1 - (distance_to_center / max_distance)
        
        # Puntuación combinada
        score = (confidence * 0.5) + (size_ratio * 2) + (centrality_score * 0.3)
        
        if score > best_score:
            best_score = score
            best_face = (x, y, w, h, confidence)
    
    return best_face

def crop_and_resize_face(image, face_box, margin=0.3):
    """
    Recorta el rostro con un margen adicional y redimensiona
    """
    x, y, w, h, confidence = face_box
    
    # Añadir margen
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    
    # Recortar
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def process_image(image_path, output_path, rejected_path, net, eye_cascade, img_size, min_confidence, min_face_size):
    """
    Procesa una imagen: detecta rostros, filtra por confianza y calidad, y guarda
    """
    try:
        # Leer imagen
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return "error", 0
        
        h, w = img_cv.shape[:2]
        
        # Detectar rostros
        faces = detect_faces(img_cv, net, min_confidence)
        
        if not faces:
            # No se detectaron rostros
            if rejected_path:
                os.makedirs(os.path.dirname(rejected_path), exist_ok=True)
                cv2.imwrite(rejected_path, img_cv)
            return "no_face", 0
        
        # Seleccionar el mejor rostro
        best_face = get_best_face(faces, w, h, min_face_size)
        
        if best_face is None:
            # Rostros muy pequeños
            if rejected_path:
                os.makedirs(os.path.dirname(rejected_path), exist_ok=True)
                cv2.imwrite(rejected_path, img_cv)
            return "small_face", 0
        
        is_valid, reason = check_face_quality(img_cv, best_face, eye_cascade)
        
        if not is_valid:
            if rejected_path:
                os.makedirs(os.path.dirname(rejected_path), exist_ok=True)
                cv2.imwrite(rejected_path, img_cv)
            return reason, best_face[4]
        
        # Recortar y redimensionar
        cropped = crop_and_resize_face(img_cv, best_face, margin=0.3)
        
        # Convertir a PIL para el procesamiento final
        img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        # Redimensionar para que ocupe COMPLETAMENTE el espacio
        # Calcular el ratio para llenar el cuadrado manteniendo proporciones
        width, height = img_pil.size
        
        if width > height:
            # Imagen horizontal: ajustar por altura
            new_height = img_size
            new_width = int(width * (img_size / height))
        else:
            # Imagen vertical o cuadrada: ajustar por ancho
            new_width = img_size
            new_height = int(height * (img_size / width))
        
        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear imagen final y centrar (o recortar exceso)
        final_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        
        # Calcular posición de centrado
        paste_x = (img_size - new_width) // 2
        paste_y = (img_size - new_height) // 2
        
        # Si la imagen es más grande que img_size, recortar desde el centro
        if new_width > img_size or new_height > img_size:
            crop_x = max(0, (new_width - img_size) // 2)
            crop_y = max(0, (new_height - img_size) // 2)
            img_pil = img_pil.crop((crop_x, crop_y, crop_x + img_size, crop_y + img_size))
            paste_x, paste_y = 0, 0
        
        final_img.paste(img_pil, (paste_x, paste_y))
        
        # Guardar
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_img.save(output_path)
        
        return "success", best_face[4]  # Retornar confianza
        
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return "error", 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--choose", action="store_true",
                        help="Elegir dataset y output mediante explorador de archivos")
    args = parser.parse_args()

    if args.choose:
        print("Seleccione la carpeta del dataset...")
        DATASET_PATH = choose_directory("Selecciona el dataset de entrada")

        print("Seleccione la carpeta donde guardar el output limpio...")
        OUTPUT_PATH = choose_directory("Selecciona la carpeta de salida")

        print("Seleccione la carpeta donde guardar las imágenes rechazadas...")
        REJECTED_PATH = choose_directory("Selecciona la carpeta de rechazadas")

        if not DATASET_PATH or not OUTPUT_PATH or not REJECTED_PATH:
            print("Debes seleccionar todas las carpetas.")
            exit(1)

        print(f"\nDataset: {DATASET_PATH}")
        print(f"Output: {OUTPUT_PATH}")
        print(f"Rechazadas: {REJECTED_PATH}\n")
        
    print("Cargando detectores...")
    net = load_face_detector()
    eye_cascade = load_eye_detector()
    
    # Buscar todas las imágenes (jpg, png, pgm)
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.pgm"]:
        all_images.extend(glob(os.path.join(DATASET_PATH, "*", ext)))
        all_images.extend(glob(os.path.join(DATASET_PATH, ext)))
    
    print(f"Encontradas {len(all_images)} imágenes")
    
    stats = {
        "success": 0,
        "no_face": 0,
        "small_face": 0,
        "missing_eyes": 0,
        "eyes_misplaced": 0,
        "face_tilted": 0,
        "poor_lighting": 0,
        "face_roi_empty": 0,
        "error": 0
    }
    
    for img_path in tqdm(all_images, desc="Procesando imágenes"):
        relative_path = os.path.relpath(img_path, DATASET_PATH)
        
        # Cambiar extensión a jpg en output
        base_name = os.path.splitext(relative_path)[0] + ".jpg"
        out_path = os.path.join(OUTPUT_PATH, base_name)
        rejected_out = os.path.join(REJECTED_PATH, base_name)
        
        status, confidence = process_image(
            img_path, 
            out_path, 
            rejected_out,
            net,
            eye_cascade,
            IMG_SIZE, 
            MIN_CONFIDENCE,
            MIN_FACE_SIZE
        )
        
        if status in stats:
            stats[status] += 1
        else:
            stats["error"] += 1
    
    print("\nProcesamiento completado")
    print(f"Estadísticas:")
    print(f"  Exitosas: {stats['success']}")
    print(f"  Sin rostro detectado: {stats['no_face']}")
    print(f"  Rostro muy pequeño: {stats['small_face']}")
    print(f"  Ojos faltantes: {stats['missing_eyes']}")
    print(f"  Ojos mal ubicados: {stats['eyes_misplaced']}")
    print(f"  Rostro ladeado: {stats['face_tilted']}")
    print(f"  Iluminación pobre: {stats['poor_lighting']}")
    print(f"  ROI vacío: {stats['face_roi_empty']}")
    print(f"  Errores: {stats['error']}")
    
    total_rejected = sum(v for k, v in stats.items() if k != "success")
    acceptance_rate = (stats['success'] / len(all_images) * 100) if all_images else 0
    print(f"\nPorcentaje de aceptación: {acceptance_rate:.1f}%")
    print(f"Dataset limpio en: {OUTPUT_PATH}")
    print(f"Imágenes rechazadas en: {REJECTED_PATH}")