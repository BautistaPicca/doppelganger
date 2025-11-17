import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import TripletMarginLoss
from tqdm import tqdm
import torch.optim as optim


#   BLOQUE RESIDUAL ESTILO RESNET

class ResidualBlock(nn.Module):
    """
    Implementa un bloque residual clásico: Conv -> BN -> ReLU -> Conv -> BN
    con shortcut opcional si cambian dimensiones o stride.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Convolución principal
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        # Segunda convolución del bloque
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut para igualar dimensiones si es necesario
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Conexión residual
        return F.relu(out)


#   RED PARA GENERAR EMBEDDINGS (128D–256D)

class EmbeddingNet(nn.Module):
    """
    Red convolucional con bloques residuales para generar embeddings
    normalizados en un espacio L2 (útil para reconocimiento facial).
    """
    def __init__(self, embedding_dim=256):
        super().__init__()

        # Capa inicial estilo ResNet
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Tres grupos residuales
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Capa final -> embedding vector
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, eps=1e-5, momentum=0.1)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Proyección a embedding
        x = self.fc(x)
        x = self.bn(x)

        # Normalización L2 para comparación por distancia
        return F.normalize(x, p=2, dim=1)


#   DATASET DE TRIPLETAS PARA ENTRENAMIENTO SIAMESE

class TripletFaceDataset(Dataset):
    """
    Cada clase = una persona.
    Cada item devuelve (anchor, positive, negative).
    Se ignoran carpetas con menos de 2 imágenes.
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.images = {}

        skipped_classes = []
        
        # Recorre carpetas/personas
        for c in os.listdir(root):
            path = os.path.join(root, c)
            if not os.path.isdir(path):
                continue

            # Solo carga archivos de imagen
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            if len(files) >= 2:
                self.images[c] = files
            else:
                skipped_classes.append(c)

        if skipped_classes:
            print(f"[TripletFaceDataset] Se ignoraron personas con menos de 2 imágenes:\n{skipped_classes}")

        self.classes = list(self.images.keys())
        if len(self.classes) == 0:
            raise ValueError("No hay clases con al menos 2 imágenes.")

    def __len__(self):
        # Not essential exact count, but sum of all images
        return sum(len(v) for v in self.images.values())

    def _load_image(self, cls, file):
        return Image.open(os.path.join(self.root, cls, file)).convert('RGB')

    def __getitem__(self, _):
        # Anchor (persona A)
        anchor_cls = random.choice(self.classes)
        anchor_file = random.choice(self.images[anchor_cls])

        # Positive = otra imagen de la misma persona
        pos_file = anchor_file
        while pos_file == anchor_file:
            pos_file = random.choice(self.images[anchor_cls])

        # Negative = otra persona
        neg_cls = random.choice([c for c in self.classes if c != anchor_cls])
        neg_file = random.choice(self.images[neg_cls])

        # Cargar imágenes
        a = self._load_image(anchor_cls, anchor_file)
        p = self._load_image(anchor_cls, pos_file)
        n = self._load_image(neg_cls, neg_file)

        if self.transform:
            a, p, n = self.transform(a), self.transform(p), self.transform(n)

        return a, p, n


#   LOOP DE ENTRENAMIENTO CON TRIPLET LOSS

def train(model, dataloader, optimizer, device, margin=0.2):
    model.train()
    criterion = TripletMarginLoss(margin=margin, p=2)
    running_loss = 0.0

    for a, p, n in tqdm(dataloader, desc="Entrenando", leave=False):
        a, p, n = a.to(device), p.to(device), n.to(device)

        optimizer.zero_grad()
        ea, ep, en = model(a), model(p), model(n)
        loss = criterion(ea, ep, en)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


#   FUNCIÓN PRINCIPAL: ARMADO DE DATASET, MODELO Y ENTRENAMIENTO

def main():
    torch.backends.cudnn.benchmark = True  # Optimización para GPUs

    root = "run"
    data_root = root + "/vectorizer_dataset"
    os.makedirs(root + "/checkpoints", exist_ok=True)

    # Augmentaciones para mejorar generalización
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataset de tripletas
    dataset = TripletFaceDataset(data_root, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Modelo y optimización
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=256).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1.4e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Entrenamiento por 20 épocas
    for epoch in range(1, 21):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/20 - Loss: {loss:.4f}")

        scheduler.step()

        # Guardar checkpoint
        torch.save(model.state_dict(), f"run/checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()

