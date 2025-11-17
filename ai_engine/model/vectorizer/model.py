import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
class FaceEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Backbone pre-entrenado
        self.backbone = InceptionResnetV1(
            pretrained='vggface2',
            classify=False
        )

        # Congelamos el backbone salvo el final
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Solo entrenamos la última capa que agregamos nosotros
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x)
    