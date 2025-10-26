import argparse
from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from ai_engine.implementations.vectorizer.trained_vectorizer import FaceNetEmbedder
from ai_engine.services.vectorizer_service import VectorizerService

def main():
    parser = argparse.ArgumentParser(description="Vectoriza un dataset de imágenes usando un embedder.")
    parser.add_argument(
        "--embedder",
        type=str,
        choices=["custom", "facenet"],
        default="custom",
        help="Selecciona el embedder a usar (custom o facenet)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Dimensión del embedding"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="run/dataset",
        help="Directorio de entrada con las imágenes a procesar"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="run/processed",
        help="Directorio de salida donde se guardarán los embeddings"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Ruta al modelo entrenado (solo aplicable para Custom)."
    )

    args = parser.parse_args()

    if args.embedder != "custom" and args.model_path is not None:
        parser.error("--model-path solo puede usarse con --embedder custom")

    # Selección del embedder
    if args.embedder == "custom":
        model_path = args.model_path or "run/checkpoints/model_epoch_20.pt"
        vectorizer = FaceNetEmbedder(model_path, embedding_dim=args.dim)
    else:
        vectorizer = FacenetPyTorchEmbedder()
    
    service = VectorizerService(vectorizer)

    print(f"Usando embedder: {args.embedder}")
    if args.model_path:
        print(f"Modelo cargado desde: {args.model_path}")
    print(f"Vectorizando dataset desde {args.input_dir} hacia {args.output_dir}")

    service.vectorize_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()