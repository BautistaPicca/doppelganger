import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from ai_engine.config import *
from ai_engine.model.classifierV2.dataset import get_dataloaders
from ai_engine.model.classifierV2.build_model import create_model
from ai_engine.model.classifierV2.engine import (
    eval_for_each_epoch,
    training_for_each_epoch,
)


import os

def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    #Cargamos los dataloaders (imagenes que pasaron por el data augmentation) y numero de clases
    train_loader, val_loader, num_classes = get_dataloaders()

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
            torch.save(model.state_dict(), MODEL_PATH)
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
            torch.save(model.state_dict(), MODEL_PATH)
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

    print("\nEntrenamiento completo. Modelo guardado en:", MODEL_PATH)
        
