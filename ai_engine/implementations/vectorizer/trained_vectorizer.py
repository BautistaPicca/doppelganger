from pathlib import Path
import torch
from torch import no_grad
import numpy as np
from facenet_pytorch import MTCNN
from typing import Optional
import gdown
from ai_engine.interfaces.face_embedder import FaceEmbedder
from ai_engine.model.vectorizer.model import FaceEmbeddingNet
import torch.nn.functional as F

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1WTK8C_xe10jDYkKE8ZM7rgSo3QjOFwB_"

class FaceNetEmbedder(FaceEmbedder):
    def __init__(self, model_path: str = "run/models/backbone_vectorizer.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = MTCNN(margin=20, select_largest=True)
        
        self._ensure_model(model_path)
        
        self.face_vectorizer = FaceEmbeddingNet().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.face_vectorizer.load_state_dict(state_dict)
        self.face_vectorizer.eval()
    
    
    def _ensure_model(self, model_path: str):
        """Descarga el modelo desde Drive si no existe."""
        path = Path(model_path)
        if path.exists():
            return  # ya existe, no hacemos nada

        print(f"Modelo no encontrado en {model_path}. Descargando desde Google Drive...")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / "backbone_vectorizer.pth"
        gdown.download(MODEL_DRIVE_URL, str(temp_path), quiet=False, fuzzy=True)
        temp_path.rename(path)
        print(f"Modelo descargado y guardado en {model_path}")
        
    @no_grad()
    def embed(self, face_tensor) -> Optional[np.ndarray]:
        """
        aligned_face: imagen de la cara ya alineada como np.ndarray (H, W, C) RGB
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.float().to(self.device)

        emb = self.face_vectorizer(face_tensor)
        emb = F.normalize(emb, dim=-1)

        return emb.cpu().numpy()[0]