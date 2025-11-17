from ai_engine.implementations.vectorizer.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from ai_engine.implementations.vectorizer.trained_vectorizer import FaceNetEmbedder

auth_service = None

CONFIGS = {
    "pretrained": {
        "name": "Pre-entrenado",
        "embedder": FacenetPyTorchEmbedder,
        "dim": 512,
        "threshold": 0.65,
        "duplicate_threshold": 0.7
    },
    "backbone_trained": {
        "name": "Backbone + Entrenado",
        "embedder": FaceNetEmbedder,
        "dim": 128,
        "threshold": 0.7,
        "duplicate_threshold": 0.7
    }
}