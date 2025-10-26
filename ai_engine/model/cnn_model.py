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

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
    
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Bloques residuales: [64, 128, 256]
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, eps=1e-5, momentum=0.1)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride))
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
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)
    
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
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = TripletFaceDataset(data_root, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4, 
        persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.4e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(1, 21):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/30 - Loss: {loss:.4f}")
        scheduler.step()  # Actualiza el learning rate después de cada epoch
        torch.save(model.state_dict(), f"run/checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()