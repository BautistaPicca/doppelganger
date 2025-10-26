import argparse
from pathlib import Path

from ai_engine.implementations.facenet_pytorch_embedder import FacenetPyTorchEmbedder
from ai_engine.implementations.faiss_matcher import FaissFaceMatcher
from ai_engine.implementations.vectorizer.trained_vectorizer import FaceNetEmbedder
from ai_engine.services.indexer_service import IndexerService
from ai_engine.utils.faiss_index import FaissIndex
from ai_engine.utils.pre_processing import get_embedding

img_size = 128

def show_results(query_image_path: Path, results):
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Face Matcher")
    root.geometry("800x600")  # Tamaño fijo para ventana

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    query_img = Image.open(query_image_path).resize((img_size, img_size))
    query_tk = ImageTk.PhotoImage(query_img)
    tk.Label(scrollable_frame, text="Imagen consultada", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=10)
    tk.Label(scrollable_frame, image=query_tk).grid(row=1, column=0, columnspan=3, pady=10)

    tk.Label(scrollable_frame, text="Resultados:", font=("Arial", 14, "bold")).grid(row=2, column=0, columnspan=3, pady=10)

    max_cols = 3
    for i, r in enumerate(results):
        person_dir = Path("run/processed") / r.name
        image_path = person_dir / "image1.jpg"

        try:
            result_img = Image.open(image_path).resize((img_size, img_size))
            result_tk = ImageTk.PhotoImage(result_img)

            frame = tk.Frame(scrollable_frame)
            frame.grid(row=3 + i // max_cols, column=i % max_cols, padx=10, pady=10)

            tk.Label(frame, image=result_tk).pack()
            tk.Label(frame, text=f"{r.name}\n{r.similarity:.2f}%", font=("Arial", 10)).pack()

            frame.image_ref = result_tk

        except Exception as e:
            tk.Label(scrollable_frame, text=f"{r.name} ({r.similarity:.2f}%) - [imagen no encontrada]", font=("Arial", 10)).grid(row=3 + i // max_cols, column=i % max_cols, padx=10, pady=10)

    root.mainloop()



def main():
    parser = argparse.ArgumentParser(description="Face matcher")
    parser.add_argument("image", type=str, help="Ruta de la imagen a comparar")
    parser.add_argument("--k", type=int, default=5, help="Número de parecidos más cercanos")
    parser.add_argument("--ui", action="store_true", help="Mostrar interfaz gráfica")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Ruta al modelo entrenado (solo aplicable para Custom)."
    )

    args = parser.parse_args()

    model_path = args.model_path or "run/checkpoints/model_epoch_20.pt"
    embedder = FaceNetEmbedder(model_path, embedding_dim=256)
    index = IndexerService(from_dir="run/index", dim=256)
    matcher = FaissFaceMatcher(index)

    embedding = get_embedding(Path(args.image), embedder, target_size=(img_size, img_size))
    results = matcher.match(embedding, k=args.k)
    
    if args.ui:
        show_results(Path(args.image), results)
    else:
        print(f"Resultados encontrados para {args.image}:\n")
        for r in results:
            print(f". {r.name} ({r.similarity:.2f}%)")

if __name__ == "__main__":
    main()