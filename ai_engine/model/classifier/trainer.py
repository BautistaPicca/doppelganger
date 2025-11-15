import torch
import torch.nn as nn
import torch.optim as optim

class ClassifierTrainer:
    """
    Esta clase sirve para entrenar un modelo de clasificación en PyTorch.
    
    Básicamente tiene métodos para entrenar, validar y guardar el mejor modelo
    según la exactitud de validación.
    """
    def __init__(self, model, device="cuda"):
        """
         Inicializa la clase con el modelo y el dispositivo donde se va a entrenar.
        
        Args:
            model: El modelo que vamos a entrenar
            device: 'cuda' si queremos GPU, 'cpu' si no
        """
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.CrossEntropyLoss() # Función de pérdida
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        return total_loss / len(loader), 100 * correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        return total_loss / len(loader), 100 * correct / total

    def train(self, train_loader, val_loader, epochs=20):
        """
        Entrena el modelo varias veces (epochs) y guarda el mejor según la exactitud de validación.
        
        Args:
            train_loader: Datos de entrenamiento.
            val_loader: Datos de validación.
            epochs: Cuántas veces recorrer todos los datos
        """
        best_acc = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f} Acc={val_acc:.2f}%")

            self.scheduler.step()

            # Guardamos el modelo si mejoró respecto al anterior
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model": self.model.state_dict(),
                    "acc": val_acc # accuracy, basicamente la exactitud
                }, "best_model.pth")

                print("Se guardó el mejor modelo.")