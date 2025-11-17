import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.classes = os.listdir(root_dir)
        self.paths = {
            c: [os.path.join(root_dir, c, img) for img in os.listdir(os.path.join(root_dir, c))]
            for c in self.classes
        }

        # lista plana para elegir negativos
        self.all_images = [
            (c, img) for c in self.classes for img in self.paths[c]
        ]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_class, anchor_path = self.all_images[idx]

        # Imagen anchor
        anchor_img = transform(Image.open(anchor_path).convert("RGB"))

        # Positivo
        positive_path = random.choice(self.paths[anchor_class])
        positive_img = transform(Image.open(positive_path).convert("RGB"))

        # Negativo
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = random.choice(self.paths[negative_class])
        negative_img = transform(Image.open(negative_path).convert("RGB"))

        return anchor_img, positive_img, negative_img