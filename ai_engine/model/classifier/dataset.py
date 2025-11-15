from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CelebrityDataset(Dataset):
    """
    La estructura está en carpetas, por ejemplo:
    dataset/
        Messi/
            01.jpg
            02.jpg
            ...
        Brad_Pitt/
            01.jpg
            02.jpg
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or self.default_transform()
        
        self.images = []
        self.celebrity_to_idx = {}
        self.idx_to_celebrity = {}

        # Mapea carpetas a índices
        for idx, folder in enumerate(sorted(self.data_dir.iterdir())):
            if not folder.is_dir():
                continue

            self.celebrity_to_idx[folder.name] = idx
            self.idx_to_celebrity[idx] = folder.name

            # Carga todas las imágenes de la carpeta, en cualquiera de los formatos indicados
            for img_path in folder.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    self.images.append((img_path, idx))

        print(f"Dataset: {len(self.images)} imágenes / {len(self.celebrity_to_idx)} clases")

    def default_transform(self):
        """
        Aplica un conjunto de transformaciones a las imágenes del dataset
        """
        return transforms.Compose([
            transforms.Resize((224, 224)), # Redimensiona a 224x224, por lo general es el tamaño estandar en modelos preentrenados
            transforms.RandomHorizontalFlip(), # Data augmentation, voltea orizontalmente con 50% de probabilidad
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Cambia brillo, contraste y saturación aleatoriamente
            transforms.RandomRotation(10), # Aplica una rotación aleatoria de hasta 10 grados
            transforms.ToTensor(),
            # Estos valores son los usados en modelos como ImageNet, es lo que esperan
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label