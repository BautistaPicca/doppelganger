import torch.nn as nn
from torchvision import models

def create_model(num_classes, train_noHead=False):
    # Cargo MobileNetV2 
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Reemplazo la última capa para adaptar la red a la cantidad de clases del dataset
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    # Congelamos el resto del modelo y solo entrenamos la cabeza 
    for p in model.features.parameters():
        p.requires_grad = train_noHead

    return model
