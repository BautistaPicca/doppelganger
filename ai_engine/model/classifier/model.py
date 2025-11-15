import torch.nn as nn
from torchvision import models

class CelebrityClassifier(nn.Module):
    """
    Clasificador de imágenes de celebridades usando ResNet50 preentrenada.

    Este modelo utiliza una CNN profunda (ResNet50) como backbone para extraer 
    características de las imágenes y luego aplica una capa totalmente conectada 
    para predecir la clase de la celebridad correspondiente.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        num_classes: número de celebridades a clasificar
        pretrained: si usar pesos preentrenados en ImageNet
        """
        super().__init__()
        
        # Usamos un ResNet50 preentrenado
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Reemplazamos la última capa por nuestro clasificador, ajustando el número de clases a la cantidad de celebridades
        # De esta forma aprovechamos el modelo preentrenado y solo entrenamos la última capa, permitiendo mejores resultados
        # con menos datos y tiempo de entrenamiento.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)