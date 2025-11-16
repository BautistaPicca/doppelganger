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
    "trained": {
        "name": "Entrenado Propio",
        "embedder": FaceNetEmbedder,
        "dim": 128,
        "threshold": 0.60,
        "duplicate_threshold": 0.68
    },
    "backbone_trained": {
        "name": "Backbone + Entrenado",
        "embedder": FacenetPyTorchEmbedder,
        "dim": 256,
        "threshold": 0.55,
        "duplicate_threshold": 0.65
    }
}