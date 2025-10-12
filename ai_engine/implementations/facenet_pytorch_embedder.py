import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import numpy as np

from ai_engine.interfaces.face_embedder import FaceEmbedder 

# Determina si usar CUDA (GPU) o CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FacenetPyTorchEmbedder(FaceEmbedder):
    """
    Implementación de FaceEmbedder usando InceptionResnetV1 (FaceNet) 
    del paquete facenet_pytorch.
    """
    def __init__(self):
        # Cargar el modelo FaceNet y moverlo al dispositivo. Se carga una sola vez.
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(self, aligned_face: torch.Tensor) -> np.ndarray:
        """
        Genera un embedding a partir de un tensor PyTorch alineado.
        """
        
        # El MTCNN ya prepara la imagen como un tensor flotante. 
        # Si el input es un tensor, el siguiente código lo prepara para el modelo.
        
        # 1. Preparar para el modelo: Añadir dimensión de Batch y mover al dispositivo
        # Asume que 'aligned_face' es un tensor de PyTorch (3x160x160)
        face_tensor = aligned_face.unsqueeze(0).to(device) 

        with torch.no_grad():
            # 2. Generar el Embedding (512D)
            embedding = self.resnet(face_tensor)

            # 3. NORMALIZACIÓN L2: Paso crucial para FaceNet.
            embedding = F.normalize(embedding, p=2, dim=1)

        # Mover el tensor a la CPU y devolver array NumPy (1x512)
        return embedding.cpu().numpy()
