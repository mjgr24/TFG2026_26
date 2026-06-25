import copy
import os
import csv
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
import random
import math

# ============================================================
# SELECCIONA AQUÍ EL AGREGADOR QUE QUIERES USAR:
#   "fedavg"   → FedAvg
#   "fedprox"  → FedProx
#   "fedewa"   → FedEWA
#   "fedag"    → FedAG
#   "mh"       → MH-Fed (usa el solver configurado en mh_aggregation.py)
# ============================================================
AGGREGATOR = "fedavg"

from funciones_de_agregacion.fedAVG import fedavg_aggregate
from funciones_de_agregacion.fedEWA import fedewa_aggregate, train_one_epoch_ewa
from funciones_de_agregacion.fedAG import fedAG_aggregate, train_one_epoch_ag, contar_clases_por_hospital
from funciones_de_agregacion.fedPROX import fedprox_aggregate, train_one_epoch_fedprox
from funciones_de_agregacion.mh_aggregation import mh_aggregate, SOLVER as MH_SOLVER

from create_dataloaders import create_dataloaders, CLASSES

import psutil
import time

def get_env_override(name, cast, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return cast(raw)

def log_resources():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    print(f"    CPU: {cpu}% | RAM: {ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB ({ram.percent}%)")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t_data, t_gpu, t_move = 0, 0, 0
    t0 = time.time()
    for images, labels in dataloader:
        t1 = time.time()
        t_data += t1 - t0

        images, labels = images.to(device), labels.to(device)
        t_transfer = time.time() - t1

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = time.time()
        t_gpu += t2 - t1 - t_transfer
        t_move += t_transfer

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        t0 = time.time()

    if len(dataloader) == 0 or total == 0:
        return 0.0, 0.0
    ram = psutil.virtual_memory()
    print(f"    DataLoader: {t_data:.2f}s | Transfer: {t_move:.2f}s | GPU: {t_gpu:.2f}s | RAM: {ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB")
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


def save_predictions(model, hospital_loaders, device, csv_path, phase):
    """
    Guarda predicciones de todos los hospitales para una fase (train/val/test).
    phase: "train_loader", "val_loader" o "test_loader"
    """
    model.eval()
    rows = []

    for hospital_name, data in hospital_loaders.items():
        dataloader = data[phase]
        img_idx = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                for i in range(images.size(0)):
                    row = {
                        "hospital": hospital_name,
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


def aggregate(aggregator_name, local_states, global_state, sample_counts=None,
              local_omegas=None, local_ag_indicators=None,
              selected_names=None, all_indicators=None, all_hospitals=None,
              total_samples_all=None, mh_lambda_reg=0.0001, mh_gamma_reg=0.005):
    """
    Interfaz unificada para todos los agregadores.
    Devuelve siempre: (new_global_state, alphas_or_None, score_or_None)
    """
    name = aggregator_name.lower()

    if name == "fedavg":
        return fedavg_aggregate(local_states, global_state, sample_counts, total_samples_all), None, None

    elif name == "fedewa":
        return fedewa_aggregate(local_states, global_state, local_omegas), None, None

    elif name == "fedag":
        new_state, alphas, server_payload = fedAG_aggregate(
            local_states, selected_names, all_indicators, all_hospitals
        )
        return new_state, alphas, server_payload

    elif name == "fedprox":
        new_state = fedprox_aggregate(local_states)
        return new_state, None, None

    elif name == "mh":
        new_state, alphas, score = mh_aggregate(
            local_states=local_states,
            global_state=global_state,
            lambda_reg=mh_lambda_reg,
            gamma_reg=mh_gamma_reg,
            n_iter=200,
            global_lr=1.0
        )
        return new_state, alphas, score

    else:
        raise ValueError(
            f"Agregador desconocido: '{aggregator_name}'. "
            "Opciones: fedavg, fedewa, fedag, fedprox, mh"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_total_start = time.time()

    # ── Cargar config ────────────────────────────────────────────────────
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

    config = load_config(config_path)

    n_hospitals = config["n_hospitals"]
    hospitals = [f"Hospital_{i}" for i in range(1, n_hospitals + 1)]
    num_rounds = config["num_rounds"]
    local_epochs = config["local_epochs"]
    client_fraction = config["client_fraction"]
    random_seed = config.get("random_seed", config.get("seed", 42))
    learning_rate = config["learning_rate"]
    early_stopping_patience = config.get("early_stopping_patience", 3)

    random_seed = get_env_override("RUN_RANDOM_SEED", int, random_seed)

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    local_epochs = get_env_override("RUN_LOCAL_EPOCHS", int, local_epochs)
    learning_rate = get_env_override("RUN_LR", float, learning_rate)
    early_stopping_patience = get_env_override(
        "RUN_EARLY_STOPPING_PATIENCE", int, early_stopping_patience
    )

    fedprox_variable_local_epochs = bool(get_env_override("RUN_FEDPROX_VARIABLE_LOCAL_EPOCHS", int, 1))
    fedprox_mu = get_env_override("RUN_FEDPROX_MU", float, 0.01)
    fedag_lr_beta = get_env_override("RUN_FEDAG_LR_BETA", float, 0.01)
    mh_lambda_reg = get_env_override("RUN_MH_LAMBDA_REG", float, 0.0001)
    mh_gamma_reg = get_env_override("RUN_MH_GAMMA_REG", float, 0.005)

    # ── Determinar escenario y nombre del método ────────────────────────
    scenario = config_path.stem.replace("config_", "")  # "iid", "non_iid", etc.

    if AGGREGATOR == "mh":
        method_name = f"mh/{MH_SOLVER}"
    else:
        method_name = AGGREGATOR

    # ── Crear estructura de carpetas ────────────────────────────────────
    results_base = parent_dir / f"results_{scenario}" / f"K{n_hospitals}" / method_name / f"seed_{random_seed}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Copiar config usado (una vez por K)
    config_copy_dir = parent_dir / f"results_{scenario}" / f"K{n_hospitals}"
    config_copy_path = config_copy_dir / "config_usado.yaml"
    if not config_copy_path.exists():
        shutil.copy2(config_path, config_copy_path)

    # ── Archivos de log y métricas ──────────────────────────────────────
    results_path = results_base / "log.txt"
    results_file = open(results_path, "w", encoding="utf-8")

    csv_rondas_path = results_base / "metricas_por_ronda.csv"
    csv_rondas_file = open(csv_rondas_path, "w", newline="", encoding="utf-8")
    csv_rondas_writer = csv.writer(csv_rondas_file)
    csv_rondas_writer.writerow([
        "ronda", "hospital",
        "train_loss", "train_acc",
        "val_loss", "val_acc",
        "train_samples", "val_samples", "test_samples"
    ])

    tiempos_ronda = []

    def log(msg):
        print(msg)
        results_file.write(msg + "\n")
        results_file.flush()

    log(f"Agregador: {AGGREGATOR.upper()}")
    if AGGREGATOR == "mh":
        log(f"Solver MH: {MH_SOLVER}")
    log(f"Dispositivo: {device}")
    log(f"Escenario: {scenario}")
    log(f"Config cargada desde: {config_path}")
    log(f"Número de hospitales: {n_hospitals}")
    log(f"Hospitales: {hospitals}")
    log(f"Rondas federadas: {num_rounds}")
    log(f"Épocas locales por ronda: {local_epochs}")
    log(f"Fracción de clientes por ronda: {client_fraction}")
    log(f"Semilla aleatoria: {random_seed}")
    log(f"Learning rate: {learning_rate}")
    log(f"Early stopping patience: {early_stopping_patience}")
    if AGGREGATOR == "fedprox":
        log(f"FedProx mu: {fedprox_mu}")
        log(f"FedProx variable local epochs: {fedprox_variable_local_epochs}")
    if AGGREGATOR == "fedag":
        log(f"FedAG lr_beta: {fedag_lr_beta}")
    if AGGREGATOR == "mh":
        log(f"MH lambda_reg: {mh_lambda_reg}")
        log(f"MH gamma_reg: {mh_gamma_reg}")
    log(f"Resultados en: {results_base}")

    global_model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    round_metrics = []
    best_val_acc = -1.0
    best_round = 0
    prev_val_acc = None
    rounds_without_improvement = 0

    # ── Cargar DataLoaders una sola vez ──────────────────────────────────
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

    total_samples_all = sum(len(hospital_loaders[h]["ds_tr"]) for h in hospitals)

    # ── FedAG: precomputar indicadores estáticos ────────────────────────
    if AGGREGATOR == "fedag":
        data_root = os.path.join(os.path.dirname(__file__), f"../data_{scenario}")
        class_counts = contar_clases_por_hospital(hospitals, data_root, CLASSES)
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

        log(f"Indicadores estáticos FedAG precomputados para {len(hospitals)} hospitales.")
    else:
        class_counts = None
        beta_dict = None
        all_indicators = None

    # ── Bucle federado ──────────────────────────────────────────────────
    for round_idx in range(num_rounds):
        t_round_start = time.time()

        log(f"\n{'='*30}")
        log(f"RONDA FEDERADA {round_idx + 1}/{num_rounds}")
        log(f"{'='*30}")

        m = int(max(client_fraction * len(hospitals), 1))
        selected_hospitals = random.sample(hospitals, m)

        log(f"Hospitales seleccionados en esta ronda: {selected_hospitals}")

        global_state_before = copy.deepcopy(global_model.state_dict())

        # 1. ENTRENAMIENTO LOCAL
        local_states = []
        local_omegas = []
        local_ag_indicators = []
        train_metrics_round = {}

        for hospital_name in selected_hospitals:
            log(f"\nEntrenamiento local en {hospital_name}...")
            data = hospital_loaders[hospital_name]
            ds_tr = data["ds_tr"]
            ds_va = data["ds_va"]
            ds_te = data["ds_te"]

            log(f"Train: {len(ds_tr)} | Val: {len(ds_va)} | Test: {len(ds_te)}")

            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

            if AGGREGATOR == "fedewa":
                for local_epoch in range(local_epochs):
                    train_loss, train_acc, omega = train_one_epoch_ewa(
                        local_model, data["train_loader"], criterion, optimizer,
                        device, val_loader=data["val_loader"]
                    )
                local_omegas.append(omega)

            elif AGGREGATOR == "fedag":
                beta_local = beta_dict[hospital_name]
                for local_epoch in range(local_epochs):
                    train_loss, train_acc, ag_indicators = train_one_epoch_ag(
                        local_model, data["train_loader"], criterion, optimizer, device,
                        hospital_name, class_counts, hospitals, CLASSES,
                        beta_prev=beta_local, lr_beta=fedag_lr_beta
                    )
                    beta_local = ag_indicators["beta"]

                beta_dict[hospital_name] = beta_local
                client_upload = {
                    "mu": ag_indicators["mu"],
                    "v": ag_indicators["v"],
                    "theta": ag_indicators["theta"],
                    "beta": ag_indicators["beta"],
                }
                local_ag_indicators.append(client_upload)
                all_indicators[hospital_name]["theta"] = ag_indicators["theta"]

            elif AGGREGATOR == "fedprox":
                if fedprox_variable_local_epochs:
                    local_epochs_client = random.randint(1, local_epochs)
                else:
                    local_epochs_client = local_epochs

                log(f"{hospital_name} realizará {local_epochs_client} épocas locales con FedProx.")

                for local_epoch in range(local_epochs_client):
                    train_loss, train_acc = train_one_epoch_fedprox(
                        local_model, data["train_loader"], criterion, optimizer,
                        device, global_state_before, mu=fedprox_mu
                    )

            else:
                for local_epoch in range(local_epochs):
                    train_loss, train_acc = train_one_epoch(
                        local_model, data["train_loader"], criterion, optimizer, device
                    )

            log(f"{hospital_name} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            train_metrics_round[hospital_name] = {
                "train_loss": train_loss, "train_acc": train_acc,
                "train_samples": len(ds_tr),
                "val_samples": len(ds_va),
                "test_samples": len(ds_te),
            }
            local_states.append(copy.deepcopy(local_model.state_dict()))

        log(f"\n{len(local_states)} modelos locales entrenados correctamente.")

        # 2. AGREGACIÓN
        log(f"\nAgregando con {AGGREGATOR.upper()}...")
        t_agg_start = time.time()

        sample_counts = [
            train_metrics_round[h]["train_samples"] for h in selected_hospitals
        ]
        new_global_state, alphas, extra_info = aggregate(
            AGGREGATOR,
            local_states,
            global_state_before,
            sample_counts,
            local_omegas,
            local_ag_indicators,
            selected_names=selected_hospitals,
            all_indicators=all_indicators,
            all_hospitals=hospitals,
            total_samples_all=total_samples_all,
            mh_lambda_reg=mh_lambda_reg,
            mh_gamma_reg=mh_gamma_reg
        )

        t_agg_end = time.time()
        t_agg = t_agg_end - t_agg_start

        global_model.load_state_dict(new_global_state)
        log(f"Modelo global actualizado con {AGGREGATOR.upper()}.")
        log(f"Tiempo de agregación: {t_agg:.2f}s")

        if alphas is not None:
            log(f"Alphas: {[round(a, 4) for a in alphas]}")

        if AGGREGATOR == "fedag" and extra_info is not None:
            log(f"rho_mu: {extra_info['rho_mu']:.4f}")
            log(f"rho_v: {extra_info['rho_v']:.4f}")
            log(f"rho_theta: {extra_info['rho_theta']:.4f}")
        elif AGGREGATOR == "mh" and extra_info is not None:
            log(f"Valor función objetivo: {extra_info:.6f}")

        # 3. EVALUACIÓN EN VALIDACIÓN
        log(f"\n=== Evaluación en VALIDACIÓN después de ronda {round_idx + 1} ===")
        val_losses, val_accuracies = [], []

        for hospital_name, data in hospital_loaders.items():
            val_loss, val_acc = evaluate(global_model, data["val_loader"], criterion, device)
            if val_loss is None or val_acc is None:
                log(f"{hospital_name} -> Val vacio, se omite en la media")
                continue
            log(f"{hospital_name} -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            train_loss_h = train_metrics_round.get(hospital_name, {}).get("train_loss", "NA")
            train_acc_h = train_metrics_round.get(hospital_name, {}).get("train_acc", "NA")

            csv_rondas_writer.writerow([
                round_idx + 1, hospital_name,
                train_loss_h, train_acc_h,
                val_loss, val_acc,
                train_metrics_round.get(hospital_name, {}).get("train_samples", "NA"),
                len(data["ds_va"]), len(data["ds_te"])
            ])
            csv_rondas_file.flush()

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        if not val_losses:
            raise ValueError("No hay conjuntos de validacion con muestras para evaluar.")
        avg_loss = sum(val_losses) / len(val_losses)
        avg_acc = sum(val_accuracies) / len(val_accuracies)
        log(f"Val Loss medio: {avg_loss:.4f} | Val Acc media: {avg_acc:.4f}")
        if prev_val_acc is None:
            log("Cambio vs ronda anterior: NA (primera ronda)")
        else:
            delta_prev = avg_acc - prev_val_acc
            delta_best = avg_acc - best_val_acc
            log(
                f"Cambio vs ronda anterior: {delta_prev:+.4f} | "
                f"cambio vs mejor: {delta_best:+.4f} | "
                f"umbral mejora: +0.0010"
            )

        t_round_end = time.time()
        t_round = t_round_end - t_round_start

        log(f"Tiempo ronda: {t_round:.2f}s")

        round_metrics.append({
            "round": round_idx + 1,
            "val_loss": avg_loss,
            "val_acc": avg_acc
        })
        tiempos_ronda.append({
            "ronda": round_idx + 1,
            "tiempo_ronda_seg": round(t_round, 2),
            "tiempo_agregacion_seg": round(t_agg, 2)
        })
        prev_val_acc = avg_acc

        if avg_acc > best_val_acc + 0.001:
            best_val_acc = avg_acc
            best_round = round_idx + 1
            rounds_without_improvement = 0
            log(f"Mejor val_acc hasta ahora: {best_val_acc:.4f} (ronda {best_round})")
        else:
            rounds_without_improvement += 1
            log(
                f"Sin mejora en validacion: {rounds_without_improvement}/"
                f"{early_stopping_patience}"
            )
            if rounds_without_improvement >= early_stopping_patience:
                log(
                    f"Early stopping activado en la ronda {round_idx + 1}. "
                    f"Mejor val_acc={best_val_acc:.4f} en ronda {best_round}."
                )
                break

    # ── Resumen final por rondas ────────────────────────────────────────
    log(f"\n{'='*30}")
    log("RESUMEN FINAL POR RONDAS")
    log(f"{'='*30}")
    for m in round_metrics:
        log(f"Ronda {m['round']} -> Val Loss: {m['val_loss']:.4f} | Val Acc: {m['val_acc']:.4f}")

    # ── Evaluación final en test ────────────────────────────────────────
    log(f"\n{'='*30}")
    log("EVALUACIÓN FINAL EN TEST")
    log(f"{'='*30}")

    final_test_losses, final_test_accuracies = [], []
    for hospital_name, data in hospital_loaders.items():
        test_loss, test_acc = evaluate(global_model, data["test_loader"], criterion, device)
        if test_loss is None or test_acc is None:
            log(f"{hospital_name} -> Test vacio, se omite en la media")
            continue
        log(f"{hospital_name} -> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        final_test_losses.append(test_loss)
        final_test_accuracies.append(test_acc)

    if not final_test_losses:
        raise ValueError("No hay conjuntos de test con muestras para evaluar.")
    avg_test_loss = sum(final_test_losses) / len(final_test_losses)
    avg_test_acc = sum(final_test_accuracies) / len(final_test_accuracies)
    log(f"\nTest Loss medio: {avg_test_loss:.4f}")
    log(f"Test Acc media:  {avg_test_acc:.4f}")

    # ── Guardar predicciones ────────────────────────────────────────────
    log(f"\nGuardando predicciones...")

    save_predictions(global_model, hospital_loaders, device,
                     results_base / "predicciones_train.csv", "train_loader")
    save_predictions(global_model, hospital_loaders, device,
                     results_base / "predicciones_val.csv", "val_loader")
    save_predictions(global_model, hospital_loaders, device,
                     results_base / "predicciones_test.csv", "test_loader")

    log(f"Predicciones guardadas en: {results_base}")

    # ── Guardar modelo final ────────────────────────────────────────────
    model_path = results_base / "modelo_final.pt"
    torch.save(global_model.state_dict(), model_path)
    log(f"Modelo guardado en: {model_path}")

    # ── Guardar tiempos ─────────────────────────────────────────────────
    t_total_end = time.time()
    t_total = t_total_end - t_total_start

    tiempos_path = results_base / "tiempos.csv"
    with open(tiempos_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ronda", "tiempo_ronda_seg", "tiempo_agregacion_seg"])
        writer.writeheader()
        writer.writerows(tiempos_ronda)
        writer.writerow({
            "ronda": "TOTAL",
            "tiempo_ronda_seg": round(t_total, 2),
            "tiempo_agregacion_seg": round(sum(t["tiempo_agregacion_seg"] for t in tiempos_ronda), 2)
        })

    log(f"\nTiempo total: {t_total:.2f}s")
    log(f"Tiempos guardados en: {tiempos_path}")

    # ── Cerrar archivos ─────────────────────────────────────────────────
    results_file.close()
    csv_rondas_file.close()

    print(f"\n{'='*30}")
    print(f"EJECUCIÓN COMPLETADA")
    print(f"Resultados en: {results_base}")
    print(f"{'='*30}")


if __name__ == "__main__":
    main()
