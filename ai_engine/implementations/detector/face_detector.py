from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Inicializamos el detector de rostros MTCNN con tamaño de salida 160x160
# y un margen adicional para capturar mejor la región facial.
mtcnn = MTCNN(image_size=160, margin=40)
