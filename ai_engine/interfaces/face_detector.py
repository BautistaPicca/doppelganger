from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

#Creamos el detector de rostros
mtcnn = MTCNN(keep_all=False) # detecta solo la cara principal

#Carpetas de entrada y salida
input_base = r"C:\Users\joaqu\OneDrive\Desktop\FootballPrueba"
output_base = r"C:\Users\joaqu\OneDrive\Desktop\Recortados"
os.makedirs(output_base, exist_ok=True)

# Se utiliza para guardar las coordenadas de todos los jugadores para luego lo utilize el face aligner
detections_data = {}

#Recorrer cada carpeta de jugador 
for player_name in os.listdir(input_base):
    player_input_folder = os.path.join(input_base, player_name)
    if not os.path.isdir(player_input_folder):
        continue
    
    detections_data[player_name] = {}

    #Recorrer la imagenes de cada jugador dentro de la carpeta
    for filename in os.listdir(player_input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(player_input_folder,filename)
            img = Image.open(img_path)

            #Detectar caras
            boxes, probs = mtcnn.detect(img) 
            
            if boxes is not None:
                detections_data[player_name][filename] = []
                for box, prob in zip(boxes, probs):
                    x1, y1, x2, y2 = box.tolist()
                    detections_data[player_name][filename].append({
                        "box":[x1, y1, x2, y2],
                        "confidence": float(prob)
                    })
                print(f"[OK] {player_name} - {filename}: {len(boxes)} rostro detectado")
            else:
                print(f"[NO FACE DETECTED] {player_name} - {filename}")

    #Guardamos las coordenadas en un archivo JSON
    output_json = os.path.join(output_base, "face_detections.json")
    with open(output_json, "w") as f:
        json.dump(detections_data, f, indent=4)
                
    print(f"\n Coordenadas guardadas en: {output_json}")
