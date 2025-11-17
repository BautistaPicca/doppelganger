import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_engine.model.vectorizer.dataset import TripletFaceDataset
from ai_engine.model.vectorizer.model import FaceEmbeddingNet

def train_triplet(model, loader, device, epochs=20, save_path="run/models/best_face_embedder.pth"):
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)

        for a, p, n in progress:
            a, p, n = a.to(device), p.to(device), n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")

        # Guardar solo el mejor
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Se guardó el mejor modelo (loss={best_loss:.6f}), ubicación: {save_path}")

    print("Entrenamiento terminado.")
    print(f"Mejor perdida: {best_loss:.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TripletFaceDataset("run/output/vectorizer_dataset_filtered")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

    model = FaceEmbeddingNet().to(device)

    train_triplet(model, loader, device)