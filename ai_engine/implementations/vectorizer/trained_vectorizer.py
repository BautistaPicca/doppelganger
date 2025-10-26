from PIL import Image
import torch
from torch import no_grad
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import Optional

from ai_engine.interfaces.face_embedder import FaceEmbedder
from ai_engine.model.cnn_model import EmbeddingNet

class FaceNetEmbedder(FaceEmbedder):
    def __init__(self, model_path: str, embedding_dim: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = MTCNN(margin=20, select_largest=True)
        
        self.face_vectorizer = EmbeddingNet(embedding_dim=embedding_dim).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.face_vectorizer.load_state_dict(state_dict)
        self.face_vectorizer.eval()
    
    @no_grad()
    def embed(self, face_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        aligned_face: imagen de la cara ya alineada como np.ndarray (H, W, C) RGB
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(self.device)
        vec = self.face_vectorizer(face_tensor).squeeze(0)
        
        # Normalizar L2
        #vec = vec / vec.norm(p=2)
        return vec.cpu().numpy()