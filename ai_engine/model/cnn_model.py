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

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, eps=1e-5, momentum=0.1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class TripletFaceDataset(Dataset):
    """
    Dataset de tripletes.
    Cada carpeta debe corresponder a una persona y contener al menos 2 imágenes.
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.images = {}

        skipped_classes = []
        for c in os.listdir(root):
            path = os.path.join(root, c)
            if not os.path.isdir(path):
                continue

            # Filtra solo imágenes válidas
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(files) >= 2:
                self.images[c] = files
            else:
                skipped_classes.append(c)

        if skipped_classes:
            print(f"[TripletFaceDatasetLFW] Se ignoraron {len(skipped_classes)} personas con <2 imágenes:")
            print(", ".join(skipped_classes))

        self.classes = list(self.images.keys())
        if len(self.classes) == 0:
            raise ValueError("No hay clases válidas con al menos 2 imágenes en el dataset.")

    def __len__(self):
        return sum(len(v) for v in self.images.values())

    def _load_image(self, cls, file):
        img = Image.open(os.path.join(self.root, cls, file)).convert('RGB')
        return img

    def __getitem__(self, _):
        # Selecciona anchor
        anchor_cls = random.choice(self.classes)
        anchor_file = random.choice(self.images[anchor_cls])

        # Selecciona positive
        pos_file = anchor_file
        while pos_file == anchor_file:
            pos_file = random.choice(self.images[anchor_cls])

        # Selecciona negative
        neg_cls = random.choice([c for c in self.classes if c != anchor_cls])
        neg_file = random.choice(self.images[neg_cls])

        # Carga imágenes
        a = self._load_image(anchor_cls, anchor_file)
        p = self._load_image(anchor_cls, pos_file)
        n = self._load_image(neg_cls, neg_file)

        if self.transform:
            a, p, n = self.transform(a), self.transform(p), self.transform(n)

        return a, p, n
    
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

def main():
    torch.backends.cudnn.benchmark = True
    root = "run"
    data_root = root + "/vectorizer_dataset"
    os.makedirs(root + "/checkpoints", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = TripletFaceDataset(data_root, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(1, 31):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/30 - Loss: {loss:.4f}")
        torch.save(model.state_dict(), f"run/checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()