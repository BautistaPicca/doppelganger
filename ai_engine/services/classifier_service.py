import torch
from torchvision import transforms
from PIL import Image
import json

from ai_engine.model.classifier.model import CelebrityClassifier


class ClassifierService:
    """
    Servicio para manejar la inferencia de un modelo de clasificación de celebridades.
    
    Permite cargar un modelo entrenado de PyTorch junto con un mapeo de índices a nombres 
    de celebridades y realizar predicciones sobre nuevas imágenes;

    Debe proporcionarse la ruta del modelo y del mapeo al inicializar la clase, si no esta
    el modelo o el mapeo lanzará un error.
    """
    def __init__(self, model_path, mapping_path, device="cuda"):
        """
        Args:
            model_path: Ruta al archivo .pth del modelo entrenado
            mapping_path: Ruta al archivo JSON que mapea índices de clase a nombres de celebridades
            device: El dispositivo a usar, puede ser "cuda" o "cpu"
        """
        self.device = device
        with open(mapping_path, "r") as f:
            self.idx_to_celebrity = json.load(f)

        num_classes = len(self.idx_to_celebrity)

        # Se carga el modelo
        self.model = CelebrityClassifier(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device)["model"])
        self.model.to(device)
        self.model.eval()

        # Transformaciones para las imágenes de entrada, estaría bien mover esto a utils, si alguno va a hacerlo
        # tambien se está usando en CelebrityClassifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, top_k=3):
        """
        Devuelve las celebridades más probables para una imagen dada.
        
        Args:
            image: Imagen a predecir.
            top_k: Número de resultados principales a devolver. Por defecto el top 3.
        
        Returns:
            list[dict]: Lista de diccionarios con los campos:
                - "celebrity": Nombre de la celebridad
                - "confidence": Nivel de confianza de la predicción
        """
        img_rgb = image.convert("RGB")
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        top_probs, top_idx = probs.topk(top_k)

        results = []
        for prob, idx in zip(top_probs, top_idx):
            results.append({
                "celebrity": self.idx_to_celebrity[str(idx.item())],
                "confidence": prob.item()
            })
        return results