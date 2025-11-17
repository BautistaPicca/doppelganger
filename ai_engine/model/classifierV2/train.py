import argparse
import json
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import gdown
import zipfile
import os

from ai_engine.config import *
from ai_engine.model.classifierV2.dataset import get_dataloaders
from ai_engine.model.classifierV2.build_model import create_model
from ai_engine.model.classifierV2.engine import eval_for_each_epoch, training_for_each_epoch


GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1AqnbmpNMsOK_ckh-Vtk8_eAB2He-EuH1"

def ask_directory(title="Selecciona una carpeta"):
    """Abre un cuadro de diálogo gráfico para seleccionar una carpeta."""
    root = tk.Tk()
    root.withdraw()              # Evita que se muestre la ventana principal
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None

def download_dataset(target_path: str):
    """Descarga y extrae el dataset desde Google Drive si no existe."""
    target = Path(target_path)

    # Si ya existe y tiene contenido, no se vuelve a descargar
    if target.exists() and any(target.iterdir()):
        print(f"Dataset ya existe en: {target}")
        return

    target.mkdir(parents=True, exist_ok=True)

    # Ruta temporal del zip descargado
    zip_path = target.parent / "dataset.zip"
    print(f"Descargando dataset desde Google Drive...")

    # Descarga con gdown
    gdown.download(GOOGLE_DRIVE_URL, str(zip_path), quiet=False, fuzzy=True)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("El archivo descargado NO es un ZIP válido.")

    # Extraemos el dataset
    print("Extrayendo dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target.parent)

    zip_path.unlink()  # Eliminamos el ZIP después de extraerlo
    print(f"Dataset extraído en {target}")

def train():
    parser = argparse.ArgumentParser(description="Entrena el modelo de celebridades")
    parser.add_argument("--dataset_path", type=str, help="Ruta del dataset")
    parser.add_argument("--output_path", type=str, help="Carpeta donde guardar el modelo entrenado")
    args = parser.parse_args()

    # Se puede seleccionar el dataset por CLI o manualmente, o se descarga si no existe
    if args.dataset_path:
        DATASET = args.dataset_path
    else:
        print("No se especificó dataset. Seleccioná una carpeta o descargaremos el dataset.")
        folder = ask_directory("Selecciona la carpeta del dataset (o cancelar para descargar)")
        if folder:
            DATASET = folder
        else:
            DATASET = DATA_DIR
            download_dataset(DATASET)

    # Se puede seleccionar el output por CLI o manualmente
    if args.output_path:
        OUTPUT_PATH = args.output_path
    else:
        print("No se especificó output. Seleccioná dónde guardar el modelo entrenado.")
        folder = ask_directory("Selecciona una carpeta para el modelo entrenado")
        OUTPUT_PATH = folder if folder else MODEL_DIR

    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    #Cargamos los dataloaders (imagenes que pasaron por el data augmentation) y numero de clases
    train_loader, val_loader, num_classes, class_names = get_dataloaders(DATASET)

    mapping_file = Path(OUTPUT_PATH) / "celebrity_mapping.json"
    id2label = {i: name for i, name in enumerate(class_names)}
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)
    print(f"Mapping de celebridades guardado en {mapping_file}")
    exit(1)
    model = create_model(num_classes, train_noHead=False).to(device)
    criterion = nn.CrossEntropyLoss() #Funcion de perdida

    best_acc = 0
    history_train, history_val = [], []
    history_train_acc, history_val_acc = [], []

    # FASE 1 - Entrenamiento inicial - Entrenamos solo la cabeza
    print("\nEntrenamiento inicial")

    #Congelamos las capas
    for p in model.features.parameters():
        p.requires_grad = False

    #Inicializamos el Optimizador, por ahora solo para la cabeza
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(EPOCHS_INITIAL):
        train_loss, train_acc = training_for_each_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc     = eval_for_each_epoch(model, val_loader, device, criterion)

        #Guardamos los datos para poder graficar despues
        history_train.append(train_loss)
        history_val.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        print(f"[{epoch+1}/{EPOCHS_INITIAL}] Train {train_acc:.4f} | Val {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(OUTPUT_PATH) / "best_model.pth")
            print("Se guardo el modelo para utilizar")
            

    # FASE 2 — Fine tuning - Descongelamos algunas capas para probar si hay mejora
    print("\n Entrenamiento del Fine-Tuning")

    #Descongelamos las ultimas 5 capas para ver si hay mejora
    feature_layers = list(model.features.children())
    for layer in feature_layers[-5:]:
        for p in layer.parameters():
            p.requires_grad = True

    #Otro optimizador, pero esta vez más bajo el learnig-rate para ver si mejora
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    for epoch in range(EPOCHS_FINETUNE):
        train_loss, train_acc = training_for_each_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc     = eval_for_each_epoch(model, val_loader, device, criterion)

        #Guardamos los datos para poder graficar
        history_train.append(train_loss)
        history_val.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        print(f"[{epoch+1}/{EPOCHS_FINETUNE}] FT Train {train_acc:.4f} | Val {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(OUTPUT_PATH) / "best_model.pth")
            print("Se guardo el modelo de Fine Tuning para utilizar ")
    

    #GRAFICOS DEL ENTRENAMIENTO

    #Curva de perdida
    plt.figure(figsize=(10,5))
    plt.plot(history_train, label="Training Loss")
    plt.plot(history_val, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("model/loss_curve.png")
    plt.show()

    #Curva de precision 
    plt.figure(figsize=(10,5))
    plt.plot(history_train_acc, label="Training Acc")
    plt.plot(history_val_acc, label="Validation Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("model/acc_curve.png")
    plt.show()

    print("\nEntrenamiento completo. Modelo guardado en:", Path(OUTPUT_PATH) / "best_model.pth")
        
if __name__ == "__main__":
    train()