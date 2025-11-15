import argparse
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import gdown
import zipfile
import tkinter as tk
from tkinter import filedialog

from ai_engine.model.classifier.dataset import CelebrityDataset
from ai_engine.model.classifier.model import CelebrityClassifier
from ai_engine.model.classifier.trainer import ClassifierTrainer

MAPPING_PATH = "run/models/celebrity_mapping.json"
DEFAULT_DATASET_PATH = "run/output/celebrities_cleaned"
DEFAULT_OUTPUT_PATH = "run/models/"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1AqnbmpNMsOK_ckh-Vtk8_eAB2He-EuH1"

def ask_directory(title="Selecciona una carpeta"):
    """Abre un cuadro de diálogo para seleccionar una carpeta."""
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None

def download_dataset(target_path: str):
    target = Path(target_path)

    if target.exists() and any(target.iterdir()):
        print(f"Dataset ya existe en: {target}")
        return

    target.mkdir(parents=True, exist_ok=True)

    zip_path = target.parent / "dataset.zip"
    print(f"Descargando dataset desde Google Drive...")

    gdown.download(GOOGLE_DRIVE_URL, str(zip_path), quiet=False, fuzzy=True)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("El archivo descargado NO es un ZIP válido.")

    print("Extrayendo dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target.parent)

    zip_path.unlink()
    print(f"Dataset extraído en {target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de celebridades")
    parser.add_argument("--dataset_path", type=str, help="Ruta del dataset")
    parser.add_argument("--output_path", type=str, help="Carpeta donde guardar el modelo entrenado")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH = 32
    EPOCHS = 20

    print(f"Usando dispositivo: {DEVICE}")

    if args.dataset_path:
        DATASET = args.dataset_path
    else:
        print("No se especificó dataset. Seleccioná una carpeta o descargaremos el dataset.")
        folder = ask_directory("Selecciona la carpeta del dataset (o cancelar para descargar)")
        if folder:
            DATASET = folder
        else:
            DATASET = DEFAULT_DATASET_PATH
            download_dataset(DATASET)

    if args.output_path:
        OUTPUT_PATH = args.output_path
    else:
        print("No se especificó output. Seleccioná dónde guardar el modelo entrenado.")
        folder = ask_directory("Selecciona una carpeta para el modelo entrenado")
        OUTPUT_PATH = folder if folder else DEFAULT_OUTPUT_PATH

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Guardar mapeo en OUTPUT_PATH
    mapping_file = Path(OUTPUT_PATH) / "celebrity_mapping.json"

    full_dataset = CelebrityDataset(DATASET)

    total_samples = len(full_dataset)
    val_size = int(0.2 * total_samples)
    train_size = total_samples - val_size

    train_data, val_data = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Guardar mapeo
    with open(mapping_file, "w") as f:
        json.dump(full_dataset.idx_to_celebrity, f, indent=2)

    # Loaders
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=4)

    num_classes = len(full_dataset.celebrity_to_idx)
    model = CelebrityClassifier(num_classes=num_classes)

    trainer = ClassifierTrainer(model, DEVICE)

    print("Iniciando entrenamiento...")
    trainer.train(train_loader, val_loader, OUTPUT_PATH , epochs=EPOCHS)

    print(f"Entrenamiento finalizado.\nModelo y mapeo guardados en: {OUTPUT_PATH}")
