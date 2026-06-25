"""
Grid Search de hiperparámetros para todos los métodos de agregación.
Ejecutar desde: escenarios/non_iid/scripts_non_iid/

Busca los mejores hiperparámetros por método usando val_acc media
con early stopping (patience=5, max_rounds=30).

Resultados en: escenarios/non_iid/grid_search/
"""

import copy
import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
import random
import math

# ── Imports de agregación ────────────────────────────────────────────────────
from funciones_de_agregacion.fedAVG import fedavg_aggregate
from funciones_de_agregacion.fedEWA import fedewa_aggregate, train_one_epoch_ewa
from funciones_de_agregacion.fedAG import fedAG_aggregate, train_one_epoch_ag, contar_clases_por_hospital
from funciones_de_agregacion.fedPROX import fedprox_aggregate, train_one_epoch_fedprox
from funciones_de_agregacion.mh_aggregation import mh_aggregate
import funciones_de_agregacion.mh_aggregation as mh_module

from create_dataloaders import create_dataloaders, CLASSES


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
        return self.classifier(self.features(x))


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
    if len(dataloader) == 0 or total == 0:
        return 0.0, 0.0
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
    if len(dataloader) == 0 or total == 0:
        return None, None
    return running_loss / len(dataloader), correct / total


# ── Ejecución de una configuración ──────────────────────────────────────────

def run_one_config(method, hyperparams, hospitals, hospital_loaders,
                   device, seed, total_samples_all,
                   class_counts=None, scenario="non_iid"):
    """
    Entrena con un método y una combinación de hiperparámetros.
    Devuelve: best_val_acc, best_round, total_rounds, elapsed_time
    """
    PATIENCE = 3
    MAX_ROUNDS = 50

    lr = hyperparams["lr"]
    local_epochs = hyperparams["local_epochs"]

    random.seed(seed)
    torch.manual_seed(seed)

    global_model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    client_fraction = 0.6

    # FedAG state
    if method == "fedag":
        beta_dict = {h: None for h in hospitals}
        n_total = sum(sum(class_counts[h].values()) for h in hospitals)
        all_indicators = {}
        for h in hospitals:
            n_k = sum(class_counts[h].values())
            mu_k = n_k / (n_total + 1e-12)
            p_local = [class_counts[h].get(cls, 0) / (n_k + 1e-12) for cls in CLASSES]
            n_per_class = {cls: sum(class_counts[hp].get(cls, 0) for hp in hospitals) for cls in CLASSES}
            p_global = [n_per_class[cls] / (n_total + 1e-12) for cls in CLASSES]
            v_k = math.sqrt(sum((p_local[i] - p_global[i]) ** 2 for i in range(len(CLASSES))))
            all_indicators[h] = {"mu": mu_k, "v": v_k, "theta": 1.0 / len(CLASSES)}

    best_val_acc = -1.0
    best_model_state = None
    best_round = 0
    rounds_without_improvement = 0
    t_start = time.time()

    for round_idx in range(MAX_ROUNDS):
        m = int(max(client_fraction * len(hospitals), 1))
        selected_hospitals = random.sample(hospitals, m)
        global_state_before = copy.deepcopy(global_model.state_dict())

        local_states = []
        local_omegas = []
        sample_counts = []

        for hospital_name in selected_hospitals:
            data = hospital_loaders[hospital_name]
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.SGD(local_model.parameters(), lr=lr)

            if method == "fedewa":
                for ep in range(local_epochs):
                    train_loss, train_acc, omega = train_one_epoch_ewa(
                        local_model, data["train_loader"], criterion, optimizer,
                        device, val_loader=data["val_loader"]
                    )
                local_omegas.append(omega)

            elif method == "fedag":
                lr_beta = hyperparams.get("lr_beta", 0.01)
                beta_local = beta_dict[hospital_name]
                for ep in range(local_epochs):
                    train_loss, train_acc, ag_indicators = train_one_epoch_ag(
                        local_model, data["train_loader"], criterion, optimizer, device,
                        hospital_name, class_counts, hospitals, CLASSES,
                        beta_prev=beta_local, lr_beta=lr_beta
                    )
                    beta_local = ag_indicators["beta"]
                beta_dict[hospital_name] = beta_local
                all_indicators[hospital_name]["theta"] = ag_indicators["theta"]

            elif method == "fedprox":
                mu = hyperparams.get("mu", 0.01)
                for ep in range(local_epochs):
                    train_loss, train_acc = train_one_epoch_fedprox(
                        local_model, data["train_loader"], criterion, optimizer,
                        device, global_state_before, mu=mu
                    )

            elif method == "fedavg" or method.startswith("mh_"):
                for ep in range(local_epochs):
                    train_loss, train_acc = train_one_epoch(
                        local_model, data["train_loader"], criterion, optimizer, device
                    )

            local_states.append(copy.deepcopy(local_model.state_dict()))
            sample_counts.append(len(data["ds_tr"]))

        # Agregación
        if method == "fedavg":
            new_state = fedavg_aggregate(local_states, global_state_before,
                                         sample_counts, total_samples_all)
        elif method == "fedprox":
            new_state = fedprox_aggregate(local_states)
        elif method == "fedewa":
            new_state = fedewa_aggregate(local_states, global_state_before, local_omegas)
        elif method == "fedag":
            new_state, _, _ = fedAG_aggregate(
                local_states, selected_hospitals, all_indicators, hospitals
            )
        elif method.startswith("mh_"):
            mh_module.SOLVER = method.replace("mh_", "", 1)
            lam = hyperparams.get("lambda_reg", 0.0001)
            gam = hyperparams.get("gamma_reg", 0.005)
            new_state, _, _ = mh_aggregate(
                local_states, global_state_before,
                lambda_reg=lam, gamma_reg=gam,
                n_iter=200, global_lr=1.0
            )

        global_model.load_state_dict(new_state)

        # Evaluación en validación
        val_accs = []
        for hospital_name, data in hospital_loaders.items():
            _, val_acc = evaluate(global_model, data["val_loader"], criterion, device)
            if val_acc is None:
                continue
            val_accs.append(val_acc)
        if not val_accs:
            raise ValueError("No hay conjuntos de validacion con muestras para evaluar.")
        avg_val_acc = sum(val_accs) / len(val_accs)

        # Early stopping
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = copy.deepcopy(global_model.state_dict())
            best_round = round_idx + 1
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= PATIENCE:
                break

        # Check NaN
        if math.isnan(avg_val_acc):
            break

    elapsed = time.time() - t_start
    total_rounds = round_idx + 1

    return best_val_acc, best_round, total_rounds, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    config_path = None
    for name in ["config_iid.yaml", "config_semi_iid.yaml", "config_non_iid.yaml"]:
        candidate = parent_dir / name
        if candidate.exists():
            config_path = candidate
            break
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    scenario = config_path.stem.replace("config_", "")
    n_hospitals = config["n_hospitals"]
    hospitals = [f"Hospital_{i}" for i in range(1, n_hospitals + 1)]
    seed = config.get("random_seed", 16)

    # Output
    grid_dir = parent_dir / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Pre-cargar DataLoaders
    print("Cargando DataLoaders...")
    hospital_loaders = {}
    for h in hospitals:
        ds_tr, tr, ds_va, va, ds_te, te, cls = create_dataloaders(h)
        hospital_loaders[h] = {
            "ds_tr": ds_tr, "train_loader": tr,
            "ds_va": ds_va, "val_loader": va,
            "ds_te": ds_te, "test_loader": te,
        }
    total_samples_all = sum(len(hospital_loaders[h]["ds_tr"]) for h in hospitals)
    print(f"DataLoaders cargados. Total muestras: {total_samples_all}")

    # FedAG class counts
    data_root = os.path.join(os.path.dirname(__file__), f"../data_{scenario}")
    class_counts = contar_clases_por_hospital(hospitals, data_root, CLASSES)

    # ── Definir grids ────────────────────────────────────────────────────
    grids = {
        "fedavg": [
            {"lr": 0.05, "local_epochs": 3},
            {"lr": 0.01, "local_epochs": 3},
            {"lr": 0.05, "local_epochs": 1},
            {"lr": 0.05, "local_epochs": 10},
            {"lr": 0.1, "local_epochs": 1},
            {"lr": 0.05, "local_epochs": 5},
            {"lr": 0.01, "local_epochs": 5},
            {"lr": 0.01, "local_epochs": 10},
            {"lr": 0.1, "local_epochs": 5},
            {"lr": 0.1, "local_epochs": 3},
        ],
        "fedprox": [
            {"lr": 0.1, "local_epochs": 5, "mu": 0.001},
            {"lr": 0.1, "local_epochs": 5, "mu": 0.01},
            {"lr": 0.05, "local_epochs": 5, "mu": 0.001},
            {"lr": 0.1, "local_epochs": 1, "mu": 0.01},
            {"lr": 0.05, "local_epochs": 3, "mu": 0.1},
            {"lr": 0.05, "local_epochs": 3, "mu": 0.001},
            {"lr": 0.05, "local_epochs": 10, "mu": 0.001},
            {"lr": 0.01, "local_epochs": 10, "mu": 0.001},
            {"lr": 0.1, "local_epochs": 5, "mu": 0.1},
            {"lr": 0.1, "local_epochs": 10, "mu": 0.1},
        ],
        "fedewa": [
            {"lr": 0.01, "local_epochs": 5},
            {"lr": 0.01, "local_epochs": 3},
            {"lr": 0.01, "local_epochs": 10},
            {"lr": 0.05, "local_epochs": 5},
            {"lr": 0.001, "local_epochs": 10},
            {"lr": 0.05, "local_epochs": 10},
            {"lr": 0.01, "local_epochs": 1},
            {"lr": 0.1, "local_epochs": 5},
            {"lr": 0.05, "local_epochs": 3},
            {"lr": 0.001, "local_epochs": 3},
        ],
        "fedag": [
            {"lr": 0.01, "local_epochs": 10, "lr_beta": 0.001},
            {"lr": 0.1, "local_epochs": 10, "lr_beta": 0.01},
            {"lr": 0.01, "local_epochs": 3, "lr_beta": 0.01},
            {"lr": 0.05, "local_epochs": 5, "lr_beta": 0.01},
            {"lr": 0.01, "local_epochs": 10, "lr_beta": 0.1},
            {"lr": 0.05, "local_epochs": 5, "lr_beta": 0.1},
            {"lr": 0.01, "local_epochs": 5, "lr_beta": 0.1},
            {"lr": 0.01, "local_epochs": 3, "lr_beta": 0.1},
            {"lr": 0.01, "local_epochs": 5, "lr_beta": 0.01},
            {"lr": 0.001, "local_epochs": 10, "lr_beta": 0.001},
        ],
        "mh_cro": [
            {"lr": 0.1, "local_epochs": 10, "lambda_reg": 0.01, "gamma_reg": 0.05},
            {"lr": 0.1, "local_epochs": 10, "lambda_reg": 0.01, "gamma_reg": 0.005},
            {"lr": 0.1, "local_epochs": 5, "lambda_reg": 0.001, "gamma_reg": 0.005},
            {"lr": 0.05, "local_epochs": 3, "lambda_reg": 0.0001, "gamma_reg": 0.05},
            {"lr": 0.1, "local_epochs": 10, "lambda_reg": 0.00001, "gamma_reg": 0.005},
            {"lr": 0.05, "local_epochs": 10, "lambda_reg": 0.00001, "gamma_reg": 0.01},
            {"lr": 0.1, "local_epochs": 3, "lambda_reg": 0.00001, "gamma_reg": 0.005},
            {"lr": 0.1, "local_epochs": 10, "lambda_reg": 0.001, "gamma_reg": 0.01},
            {"lr": 0.1, "local_epochs": 10, "lambda_reg": 0.00001, "gamma_reg": 0.05},
            {"lr": 0.1, "local_epochs": 5, "lambda_reg": 0.001, "gamma_reg": 0.01},
        ],
    }

    # Contar total
    total_configs = sum(len(v) for v in grids.values())
    print(f"\nTotal de configuraciones: {total_configs}")
    print(f"Escenario: {scenario}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print()

    # ── CSV de resultados ────────────────────────────────────────────────
    csv_path = grid_dir / "resultados_grid_search.csv"
    already_done = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # skip header
            for row in reader:
                key = (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                already_done.add(key)
        print(f"Retomando: {len(already_done)} configuraciones ya completadas")
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    else:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file, delimiter=";")
    if not already_done:
        csv_writer.writerow([
            "metodo", "lr", "local_epochs", "mu", "lr_beta",
            "lambda_reg", "gamma_reg",
            "best_val_acc", "best_round", "total_rounds", "tiempo_seg"
        ])

    # ── Ejecutar grid search ─────────────────────────────────────────────
    config_num = 0
    for method, configs in grids.items():
        print(f"\n{'='*60}")
        print(f"MÉTODO: {method.upper()} ({len(configs)} configuraciones)")
        print(f"{'='*60}")

        best_method_acc = -1.0
        best_method_params = None

        for i, hp in enumerate(configs):
            config_num += 1
            key = (
                method,
                str(hp.get("lr", "")),
                str(hp.get("local_epochs", "")),
                str(hp.get("mu", "")),
                str(hp.get("lr_beta", "")),
                str(hp.get("lambda_reg", "")),
                str(hp.get("gamma_reg", "")),
            )
            if key in already_done:
                print(f"  → Ya completado, saltando")
                continue
            hp_str = " | ".join(f"{k}={v}" for k, v in hp.items())
            print(f"\n[{config_num}/{total_configs}] {method} | {hp_str}")

            try:
                best_acc, best_rnd, total_rnds, elapsed = run_one_config(
                    method, hp, hospitals, hospital_loaders,
                    device, seed, total_samples_all,
                    class_counts=class_counts, scenario=scenario
                )

                acc_str = f"{best_acc:.4f}" if not math.isnan(best_acc) else "NaN"
                print(f"  → val_acc={acc_str} (ronda {best_rnd}/{total_rnds}) [{elapsed:.0f}s]")

                # Escribir al CSV
                csv_writer.writerow([
                    method,
                    hp.get("lr", ""),
                    hp.get("local_epochs", ""),
                    hp.get("mu", ""),
                    hp.get("lr_beta", ""),
                    hp.get("lambda_reg", ""),
                    hp.get("gamma_reg", ""),
                    round(best_acc, 6) if not math.isnan(best_acc) else "NaN",
                    best_rnd,
                    total_rnds,
                    round(elapsed, 1)
                ])
                csv_file.flush()

                if not math.isnan(best_acc) and best_acc > best_method_acc:
                    best_method_acc = best_acc
                    best_method_params = hp.copy()

            except Exception as e:
                print(f"  → ERROR: {e}")
                csv_writer.writerow([
                    method,
                    hp.get("lr", ""), hp.get("local_epochs", ""),
                    hp.get("mu", ""), hp.get("lr_beta", ""),
                    hp.get("lambda_reg", ""), hp.get("gamma_reg", ""),
                    "ERROR", "", "", ""
                ])
                csv_file.flush()

        if best_method_params:
            print(f"\n  *** MEJOR {method.upper()}: val_acc={best_method_acc:.4f}")
            print(f"      Params: {best_method_params}")

    csv_file.close()
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETADO")
    print(f"Resultados en: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
