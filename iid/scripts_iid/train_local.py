"""
Baseline local: entrena cada hospital en aislamiento total (sin federación).
Mismo presupuesto computacional que el entrenamiento federado (E × num_rounds épocas).
Guarda predicciones de test locales y globales para cada hospital.

Uso:
    python train_local.py

Lee la configuración del YAML del escenario (config_iid.yaml, config_non_iid.yaml, etc.)
"""

import os
import csv
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from create_dataloaders import create_dataloaders, CLASSES


# ── Modelo (idéntico al federado) ────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Funciones auxiliares ─────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), correct / total


def save_predictions(model, dataloader, device, csv_path, hospital_name=None):
    """
    Guarda predicciones: y_true, y_pred, probabilidades por clase.
    Si hospital_name es None, no incluye columna de hospital.
    """
    model.eval()
    rows = []
    img_idx = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                row = {}
                if hospital_name is not None:
                    row["hospital"] = hospital_name
                row["imagen_idx"] = img_idx
                row["y_true"] = labels[i].item()
                row["y_pred"] = preds[i].item()
                for c in range(probs.size(1)):
                    row[f"prob_{c}"] = round(probs[i][c].item(), 6)
                rows.append(row)
                img_idx += 1

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def save_global_predictions(model, hospital_loaders, hospitals, device, csv_path):
    """
    Evalúa el modelo contra los test de TODOS los hospitales y guarda un único CSV.
    """
    model.eval()
    rows = []
    img_idx = 0

    with torch.no_grad():
        for other_hospital in hospitals:
            other_loader = hospital_loaders[other_hospital]["test_loader"]
            for images, labels in other_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                for i in range(images.size(0)):
                    row = {
                        "hospital": other_hospital,
                        "imagen_idx": img_idx,
                        "y_true": labels[i].item(),
                        "y_pred": preds[i].item(),
                    }
                    for c in range(probs.size(1)):
                        row[f"prob_{c}"] = round(probs[i][c].item(), 6)
                    rows.append(row)
                    img_idx += 1

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar config - busca el config del escenario actual
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    config_path = None
    for name in ["config_iid.yaml", "config_semi_iid.yaml", "config_non_iid.yaml"]:
        candidate = parent_dir / name
        if candidate.exists():
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError("No se encontró ningún config YAML en el directorio padre.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    n_hospitals = config["n_hospitals"]
    hospitals = [f"Hospital_{i}" for i in range(1, n_hospitals + 1)]
    num_rounds = config.get("num_rounds", 20)
    local_epochs = config.get("local_epochs", 5)
    learning_rate = config.get("learning_rate", 0.1)
    seed = config.get("random_seed", config.get("seed", 42))

    # Mismo presupuesto computacional que federado
    total_epochs = local_epochs * num_rounds

    # Seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Determinar escenario a partir del nombre del config
    scenario = config_path.stem.replace("config_", "")

    # Crear directorio de resultados
    results_base = parent_dir / f"results_{scenario}" / f"K{n_hospitals}" / "baseline_local" / f"seed_{seed}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Copiar config usado
    config_dir = results_base.parent.parent
    config_copy_path = config_dir / "config_usado.yaml"
    if not config_copy_path.exists():
        import shutil
        shutil.copy2(config_path, config_copy_path)

    print(f"Dispositivo: {device}")
    print(f"Escenario: {scenario}")
    print(f"Hospitales: {n_hospitals}")
    print(f"Épocas totales por hospital: {total_epochs} (local_epochs={local_epochs} × num_rounds={num_rounds})")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {seed}")
    print(f"Resultados en: {results_base}")
    print()

    criterion = nn.CrossEntropyLoss()
    tiempos = []

    # Cargar TODOS los DataLoaders una sola vez
    hospital_loaders = {}
    for hospital_name in hospitals:
        ds_tr, train_loader, ds_va, val_loader, ds_te, test_loader, class_names = \
            create_dataloaders(hospital_name)
        hospital_loaders[hospital_name] = {
            "ds_tr": ds_tr, "train_loader": train_loader,
            "ds_va": ds_va, "val_loader": val_loader,
            "ds_te": ds_te, "test_loader": test_loader,
        }

    # Entrenar cada hospital en aislamiento
    for hospital_name in hospitals:
        data = hospital_loaders[hospital_name]
        train_loader = data["train_loader"]
        val_loader = data["val_loader"]
        test_loader = data["test_loader"]
        ds_tr = data["ds_tr"]
        ds_va = data["ds_va"]
        ds_te = data["ds_te"]

        print(f"{'='*40}")
        print(f"Entrenando {hospital_name} en aislamiento...")
        print(f"{'='*40}")

        t_start = time.time()

        print(f"Train: {len(ds_tr)} | Val: {len(ds_va)} | Test: {len(ds_te)}")

        # Crear modelo desde cero para este hospital
        model = SimpleCNN(num_classes=len(CLASSES)).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Entrenar
        for epoch in range(total_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(
                    f"  Época {epoch+1}/{total_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

        # Evaluar en test
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        t_end = time.time()
        elapsed = t_end - t_start
        tiempos.append({"hospital": hospital_name, "tiempo_seg": round(elapsed, 2)})
        print(f"  Tiempo: {elapsed:.1f}s")

        # Guardar predicciones de test local
        pred_path = results_base / f"predicciones_test_{hospital_name}.csv"
        save_predictions(model, test_loader, device, pred_path)
        print(f"  Predicciones locales guardadas en: {pred_path}")

        # Guardar predicciones contra test global (todos los hospitales)
        global_pred_path = results_base / f"predicciones_test_global_{hospital_name}.csv"
        save_global_predictions(model, hospital_loaders, hospitals, device, global_pred_path)
        print(f"  Predicciones globales guardadas en: {global_pred_path}")

        # Guardar modelo
        model_path = results_base / f"modelo_{hospital_name}.pt"
        torch.save(model.state_dict(), model_path)

        print()

    # Guardar tiempos
    tiempos_path = results_base / "tiempos.csv"
    with open(tiempos_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hospital", "tiempo_seg"])
        writer.writeheader()
        writer.writerows(tiempos)
        total_time = sum(t["tiempo_seg"] for t in tiempos)
        writer.writerow({"hospital": "TOTAL", "tiempo_seg": round(total_time, 2)})

    print(f"Tiempos guardados en: {tiempos_path}")
    print(f"Tiempo total: {total_time:.1f}s")


if __name__ == "__main__":
    main()