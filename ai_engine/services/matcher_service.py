from PIL import Image
from torchvision import transforms

from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from ai_engine.implementations.faiss_matcher import FaissFaceMatcher
from ai_engine.utils.faiss_index import FaissIndex
from ai_engine.utils.pre_processing import pad_and_resize
"""
Servicio para realizar la comparación de imagenes utilizando el matcher.
Toma una imagen de entrada, procesa y obtiene su embedding, luego busca las
 k imágenes más similares en el índice y devuelve los resultados.
"""
class MatcherService:
    """
    Inicialmente no acepta implementaciones dinámicas para embedder o matcher,
    pero podría ser una opción a futuro.
    """
    def __init__(self, index_dir="index"):
        self.embedder = FacenetPyTorchEmbedder()
        self.index = FaissIndex(from_dir=index_dir)
        self.matcher = FaissFaceMatcher(self.index)
    
    """
    Toma una imagen, la procesa, obtiene su embedding y busca las k imágenes más similares.
    De momento se da por hecho que la imagen está mayoritariamente procesada (rostro centrado, blanco y negro, etc).
    Lo único que se aplica es un pre-procesamiento para asegurarse que el formato de la imagen sea cuadrado y con tamaño 160x160
    Cuando los demás módulos esten listos, hay que modificar esto.
    """
    def match_image(self, image: Image.Image, k: int = 5):
        processed = pad_and_resize(image)
        tensor = transforms.ToTensor()(processed)
        embedding = self.embedder.embed(tensor)
        results = self.matcher.match(embedding, k=k)
        return [{"name": r.name, "similarity": float(r.similarity), "image": r.image} for r in results]