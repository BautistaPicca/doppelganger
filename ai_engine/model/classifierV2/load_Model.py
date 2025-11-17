import torch
from torchvision import models

#Cargamos el modelo para poder utilizarlo
def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_feats, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


