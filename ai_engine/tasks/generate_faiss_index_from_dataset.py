import argparse
from pathlib import Path

import numpy as np

from ai_engine.services.simple_indexer_service import IndexerService
from ai_engine.types import FaceRecord


def main():
    """
    Script para crear un índice FAISS a partir de vectores procesados.
    1. Carga vectores desde un directorio de entrada (por lo general un dataset procesado)
    2. Crea un índice FAISS, añade los vectores y guarda el índice en disco
    """
    parser = argparse.ArgumentParser(description="Crea un índice FAISS a partir de vectores procesados.")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="run/processed",
        help="Directorio donde se encuentran los vectores procesados (uno por persona)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="run/index",
        help="Directorio donde se guardará el índice FAISS y los labels."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Dimensión de los embeddings (debe coincidir con la usada en la vectorización)."
    )

    args = parser.parse_args()

    processed_dir = Path(args.input_dir)
    index_path = Path(args.output_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"Cargando vectores desde: {processed_dir}")
    print(f"Guardando índice en: {index_path}")
    print(f"Dimensión de embeddings: {args.dim}")

    index = IndexerService(dim=args.dim)

    records = []
    for person_dir in sorted(processed_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        vec_path = person_dir / "vec.npy"
        if not vec_path.exists():
            print(f"Falta vec.npy en {person_dir.name}, ignorando...")
            continue

        try:
            vector = np.load(vec_path)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            elif vector.shape[0] != 1:
                print(f"Vector inválido para {vec_path.name}")
                continue

            record = FaceRecord(name=person_dir.name, vector=vector[0])
            records.append(record)
            print(f"Se ha cargado a {record.name} correctamente.")

        except Exception as e:
            print(f"Error al cargar a {person_dir.name}: {e}")

    # Agregar los registros al índice
    index.add(records)

    index.save(index_path)
    print(f"Índice guardado en {index_path} con {len(records)} registros.")

if __name__ == "__main__":
    main()