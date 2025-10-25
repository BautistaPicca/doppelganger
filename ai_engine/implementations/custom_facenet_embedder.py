import torch
import torch.nn.functional as F
import numpy as np
from ai_engine.interfaces.face_embedder import FaceEmbedder  # Se asume que esta interfaz existe

# Importar la arquitectura personalizada desde su módulo dedicado
# Asegurarse de que 'custom_model_net.py' contenga la clase InceptionResnetV1Custom
from ai_engine.implementations.custom_model_net import InceptionResnetV1Custom

# Configuración del dispositivo (GPU si está disponible)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomFaceNetEmbedder(FaceEmbedder):
    """
    Implementación personalizada del embedder facial basada en FaceNet (InceptionResnetV1).
    Permite inicializar pesos preentrenados o comenzar desde cero según el escenario de uso.
    """

    def __init__(self, load_pretrained: bool = True):
        """
        Inicializa el modelo de embeddings facial.
        Si 'load_pretrained' es True, intenta cargar los pesos desde un archivo local.
        """
        # Instanciar la arquitectura personalizada
        self.resnet = InceptionResnetV1Custom()

        # Carga opcional de pesos preentrenados
        if load_pretrained:
            try:
                # Ruta al archivo local de pesos preentrenados (.pth)
                weights_path = "ai_engine/model_weights/nuestro_vggface2.pth"
                weights = torch.load(weights_path)

                # Cargar los pesos al modelo
                self.resnet.load_state_dict(weights, strict=False)

            except FileNotFoundError:
                print(f"[ERROR] Archivo de pesos no encontrado en {weights_path}. Se utilizará inicialización aleatoria.")

        # Configurar el modelo para inferencia y moverlo al dispositivo correspondiente
        self.resnet.eval().to(device)
        print(f"[CUSTOM EMBEDDER] Modelo {self.resnet.__class__.__name__} inicializado en {device}. Preentrenado: {load_pretrained}.")

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Genera un vector de embeddings (512D) a partir de una imagen facial alineada.

        Args:
            aligned_face (np.ndarray): Imagen alineada en formato NumPy (H x W x C).

        Returns:
            np.ndarray: Vector de embedding normalizado (1 x 512).
        """
        # Conversión de la imagen a tensor PyTorch y normalización de píxeles [0, 1]
        face_tensor = torch.from_numpy(aligned_face).float() / 255.0

        # Reordenar dimensiones de HWC → CHW y agregar dimensión de batch (1 x C x H x W)
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Desactivar gradientes durante la inferencia
        with torch.no_grad():
            # Forward pass: generar el embedding (512D)
            embedding = self.resnet(face_tensor)

            # Normalización L2 para compatibilidad con similitud coseno
            embedding_L2_normalized = F.normalize(embedding, p=2, dim=1)

            # Convertir el resultado a NumPy para la interfaz de salida
            embedding_np = embedding_L2_normalized.cpu().numpy()

            return embedding_np

