import torch 

def training_for_each_epoch(model, loader, optimizer, device, criterion): 
    # Configuramos el modelo en modo entrenamiento
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        # Movemos los datos al dispositivo seleccionado (CPU/GPU)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # Reiniciamos los gradientes
        outputs = model(images)          # Forward pass

        loss = criterion(outputs, labels)  # Calculamos la pérdida
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Actualizamos los parámetros

        running_loss += loss.item() * images.size(0)  # Acumulamos pérdida
        _, preds = outputs.max(1)                     # Predicción más probable
        correct += (preds == labels).sum().item()     # Contamos aciertos
        total += labels.size(0)                       # Total de muestras

    # Devolvemos loss promedio y accuracy
    return running_loss / total, correct / total
