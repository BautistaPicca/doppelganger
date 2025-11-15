import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

from ai_engine.model.classifier.dataset import CelebrityDataset
from ai_engine.model.classifier.model import CelebrityClassifier
from ai_engine.model.classifier.trainer import ClassifierTrainer

MAPPING_PATH = "run/models/celebrity_mapping.json"
DATASET = "run/output/celebrities_cleaned"

if __name__ == "__main__":
    # Si es posible, usar la gpu para entrenar más rápido
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH = 32
    EPOCHS = 20

    print(f"Usando dispositivo: {DEVICE}")

    # Dataset de celebridades
    full_dataset = CelebrityDataset(DATASET)

    # Divide el dataset en entrenamiento y validación, el 80% para entrenamiento y 20% para validación
    total_samples = len(full_dataset)
    val_size = int(0.2 * total_samples)
    train_size = total_samples - val_size

    train_data, val_data = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Si no existe el directorio del mapeo, lo creamos
    Path(MAPPING_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Guardamos el mapeo de clases, importante para la inferencia
    with open(MAPPING_PATH, "w") as f:
        json.dump(full_dataset.idx_to_celebrity, f, indent=2)

    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=4)

    model = CelebrityClassifier(num_classes=full_dataset.num_classes if hasattr(train_data, 'num_classes') else len(full_dataset.celebrity_to_idx))

    trainer = ClassifierTrainer(model, DEVICE)
    # Empieza el entrenamiento!
    trainer.train(train_loader, val_loader, epochs=EPOCHS)

    print("Entrenamiento finalizado.")
