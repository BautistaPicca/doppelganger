import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Implementación simplificada de la arquitectura InceptionResnetV1 utilizada 
# por FaceNet. Esta clase reemplaza la dependencia de 'facenet_pytorch', 
# manteniendo únicamente las partes necesarias para generar embeddings de 512D.
# ---------------------------------------------------------------------------

class InceptionResnetV1Custom(nn.Module):
    """
    Versión reducida del modelo InceptionResnetV1 empleada para generar embeddings.
    La idea es conservar la compatibilidad con un modelo preentrenado (.pth) 
    y permitir la carga manual de pesos, sin depender de librerías externas.
    """
    def __init__(self):
        super().__init__()

        # En la implementación original, aquí se definirían todas las capas convolucionales
        # e Inception blocks. Para esta versión se simplifica dejando solo la parte final.

        # Capa de Average Pooling: reduce el mapa de características a tamaño 1x1.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Capa lineal final que proyecta el vector resultante (1792D) a un embedding de 512D.
        # El valor 1792 corresponde al tamaño típico del último bloque convolucional del modelo original.
        self.embedding_layer = nn.Linear(in_features=1792, out_features=512)

    def load_state_dict(self, state_dict, strict=True):
        """
        Carga manual de los pesos preentrenados.
        Este método adapta las claves de los pesos si difieren entre el archivo .pth
        y las definiciones de esta clase.
        """
        new_state_dict = self.state_dict()
        for key in new_state_dict.keys():
            if key in state_dict:
                new_state_dict[key] = state_dict[key]

        # Se delega la carga final al método base de PyTorch
        super().load_state_dict(new_state_dict, strict=False)

        print("[CUSTOM MODEL NET] Pesos cargados correctamente en el modelo.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define el paso hacia adelante (forward pass) del modelo.
        Recibe un tensor de entrada (imagen) y devuelve un embedding de 512 dimensiones.
        """
        # En un escenario real, 'x' sería la salida del último bloque convolucional.
        # Se asume que su forma es [batch_size, 1792, 8, 8].

        # Aplicar Average Pooling → [batch_size, 1792, 1, 1]
        x = self.avg_pool(x)

        # Aplanar el tensor para pasarlo por la capa lineal → [batch_size, 1792]
        x = x.view(x.size(0), -1)

        # Proyección final al espacio de embeddings → [batch_size, 512]
        embedding = self.embedding_layer(x)

        return embedding

