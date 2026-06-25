"""
Barrido de λ (lambda_reg) y γ (gamma_reg) para la función objetivo J(α).
Ejecuta MH-Fed con DE como solver en el escenario non-IID.

Grid: λ ∈ {0.001, 0.01, 0.1} × γ ∈ {0.001, 0.01, 0.1} = 9 combinaciones.

Uso:
    python sweep_lambda_gamma.py

Requisitos:
    - Estar en la carpeta scripts_non_iid (o ajustar config_path)
    - AGGREGATOR = "mh" y SOLVER = "de" se fuerzan internamente
"""

import copy
import os
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from datetime import datetime

from create_dataloaders import create_dataloaders, CLASSES
from funciones_de_agregacion.mh_aggregation import mh_aggregate

# ── Configuración del barrido ────────────────────────────────────────────────

LAMBDA_CANDIDATES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
GAMMA_CANDIDATES = [0.005, 0.01, 0.02]

# Forzar DE como solver para el barrido
import funciones_de_agregacion.mh_aggregation as mh_module
mh_module.SOLVER = "de"

# ── Modelo ───────────────────────────────────────────────────────────────────

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


# ── Entrenamiento federado con MH ────────────────────────────────────────────

def run_mh_fed(lambda_reg, gamma_reg, hospitals, hospital_loaders, device,
               num_rounds, local_epochs, learning_rate, seed):
    """Ejecuta MH-Fed con DE y devuelve val_acc final media."""

    random.seed(seed)
    torch.manual_seed(seed)

    global_model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    for round_idx in range(num_rounds):
        global_state_before = copy.deepcopy(global_model.state_dict())

        local_states = []
        for hospital_name in hospitals:
            data = hospital_loaders[hospital_name]
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

            for _ in range(local_epochs):
                train_one_epoch(local_model, data["train_loader"], criterion, optimizer, device)

            local_states.append(copy.deepcopy(local_model.state_dict()))

        # Aggregate with MH (DE)
        new_global_state, alphas, score = mh_aggregate(
            local_states=local_states,
            global_state=global_state_before,
            lambda_reg=lambda_reg,
            gamma_reg=gamma_reg,
            n_iter=200,
            global_lr=1.0
        )
        global_model.load_state_dict(new_global_state)

    # Evaluate final val_acc
    val_accs = []
    for hospital_name in hospitals:
        data = hospital_loaders[hospital_name]
        _, val_acc = evaluate(global_model, data["val_loader"], criterion, device)
        val_accs.append(val_acc)

    return sum(val_accs) / len(val_accs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar config
    config_path = Path(__file__).resolve().parent.parent / "config_non_iid.yaml"
    if not config_path.exists():
        # Intentar en el directorio actual
        config_path = Path("config_non_iid.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"No se encuentra config_non_iid.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    n_hospitals = config["n_hospitals"]
    hospitals = [f"Hospital_{i}" for i in range(1, n_hospitals + 1)]
    num_rounds = config.get("num_rounds", 20)
    local_epochs = config.get("local_epochs", 5)
    learning_rate = config.get("learning_rate", 0.001)
    seed = config.get("seed", 42)

    print(f"Dispositivo: {device}")
    print(f"Hospitales: {n_hospitals}")
    print(f"Rondas: {num_rounds}, Épocas locales: {local_epochs}, LR: {learning_rate}")
    print(f"Solver: DE")
    print(f"Grid: λ ∈ {LAMBDA_CANDIDATES} × γ ∈ {GAMMA_CANDIDATES}")
    print()

    # Cargar DataLoaders una sola vez
    hospital_loaders = {}
    for hospital_name in hospitals:
        ds_tr, train_loader, ds_va, val_loader, ds_te, test_loader, class_names = \
            create_dataloaders(hospital_name)
        hospital_loaders[hospital_name] = {
            "ds_tr": ds_tr, "train_loader": train_loader,
            "ds_va": ds_va, "val_loader": val_loader,
            "ds_te": ds_te, "test_loader": test_loader,
            "class_names": class_names
        }

    # Crear directorio de resultados
    output_dir = Path("sweep_lambda_gamma")
    output_dir.mkdir(exist_ok=True)

    # Ejecutar grid
    results = []

    print(f"{'λ':>10} | {'γ':>10} | {'val_acc':>10} | {'val_loss':>10}")
    print("-" * 50)

    for lam in LAMBDA_CANDIDATES:
        for gam in GAMMA_CANDIDATES:
            print(f"Ejecutando λ={lam}, γ={gam}...", end=" ", flush=True)

            val_acc = run_mh_fed(
                lambda_reg=lam,
                gamma_reg=gam,
                hospitals=hospitals,
                hospital_loaders=hospital_loaders,
                device=device,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                seed=seed
            )

            results.append({"lambda": lam, "gamma": gam, "val_acc": val_acc})
            print(f"val_acc = {val_acc:.4f}")

    # Guardar CSV
    csv_path = output_dir / "sweep_lambda_gamma_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "gamma", "val_acc"], delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    # Tabla de sensibilidad
    print()
    print("=" * 50)
    print("TABLA DE SENSIBILIDAD (val_acc)")
    print("=" * 50)

    # Header
    label = "λ \\ γ"
    header = f"{label:>10}"
    for gam in GAMMA_CANDIDATES:
        header += f" | {gam:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    best_val_acc = -1
    best_lam = None
    best_gam = None

    for lam in LAMBDA_CANDIDATES:
        row = f"{lam:>10}"
        for gam in GAMMA_CANDIDATES:
            match = [r for r in results if r["lambda"] == lam and r["gamma"] == gam][0]
            val = match["val_acc"]
            marker = ""
            if val > best_val_acc:
                best_val_acc = val
                best_lam = lam
                best_gam = gam
            row += f" | {val:>10.4f}"
        print(row)

    # Marcar el mejor en la tabla
    print()
    print(f"MEJOR: λ = {best_lam}, γ = {best_gam}, val_acc = {best_val_acc:.4f}")
    print()
    print(f"Usa estos valores en mh_aggregation.py o en train_federated.py:")
    print(f"    lambda_reg={best_lam}, gamma_reg={best_gam}")
    print()
    print(f"Resultados guardados en: {csv_path}")


if __name__ == "__main__":
    main()
