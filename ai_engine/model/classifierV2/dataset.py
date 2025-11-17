import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from ai_engine.config import IMG_SIZE, BATCH_SIZE

def get_dataloaders(data_dir):
    # Transformaciones para las imágenes de entrenamiento:
    # reescalo, meto algo de ruido (rotación/crops) y normalizo.
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Para validación no quiero alterar demasiado la imagen:
    # solo reescalo, centro y normalizo.
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Cargo el dataset usando el formato de carpetas típico de ImageFolder
    path = Path(data_dir)
    train_dataset = datasets.ImageFolder(path, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(path, transform=val_transforms)

    # Armo los dataloaders. 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    class_names = sorted(os.listdir(data_dir))  # cada carpeta = nombre de celebridad
    # Devuelvo loaders y cantidad de clases
    return train_loader, val_loader, len(train_dataset.classes), class_names
