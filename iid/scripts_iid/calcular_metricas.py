"""
Script de métricas: lee las predicciones guardadas y calcula las 5 piezas de evaluación.

Piezas:
  1. Comparación con baseline local
  2. Test global unificado
  3. Accuracy por clase en cada hospital
  4. Test de clases foráneas
  5. Evaluación por hospital

Además:
  - Tabla de tiempos
  - Test estadístico (Wilcoxon)

Uso:
    python calcular_metricas.py

Configura el escenario, K y la tabla de distribución directamente en este script.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


# ── Configuración ────────────────────────────────────────────────────────────

CLASSES = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]
FOREIGN_THRESHOLD = 0.05
SEEDS = [16, 999, 1935]
SCENARIO = "iid"
K = 50
DISTRIBUTION_TABLE = "tabla_iid_k50.csv"

FED_METHODS = ["fedavg", "fedprox", "fedewa", "fedag"]
MH_SOLVERS = ["random_search", "de", "ga", "pso", "cro"]


# ── Utilidades ───────────────────────────────────────────────────────────────

def load_predictions(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = ";" if ";" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            row["y_true"] = int(row["y_true"])
            row["y_pred"] = int(row["y_pred"])
            rows.append(row)
    return rows


def load_baseline_local_predictions(base_dir, seed, hospitals):
    """Lee predicciones del baseline local (test propio de cada hospital)."""
    rows = []
    seed_dir = base_dir / f"seed_{seed}"
    for hospital in hospitals:
        csv_path = seed_dir / f"predicciones_test_{hospital}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            delimiter = ";" if ";" in first_line else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                row["hospital"] = hospital
                row["y_true"] = int(row["y_true"])
                row["y_pred"] = int(row["y_pred"])
                rows.append(row)
    return rows


def load_baseline_global_predictions(base_dir, seed, hospitals):
    """
    Lee predicciones del baseline local evaluado contra el test global.
    Devuelve: {hospital_modelo: [rows]}
    """
    result = {}
    seed_dir = base_dir / f"seed_{seed}"
    for hospital in hospitals:
        csv_path = seed_dir / f"predicciones_test_global_{hospital}.csv"
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            delimiter = ";" if ";" in first_line else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                row["y_true"] = int(row["y_true"])
                row["y_pred"] = int(row["y_pred"])
                rows.append(row)
        if rows:
            result[hospital] = rows
    return result


def load_distribution_table(csv_path):
    dist = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row["Split"].strip() != "Train":
                continue
            hospital_num = row["Hospital"].strip()
            hospital_name = f"Hospital_{hospital_num}"
            dist[hospital_name] = {}
            for cls in CLASSES:
                pct_str = row[cls].strip().replace("%", "").replace(",", ".")
                dist[hospital_name][cls] = float(pct_str) / 100.0
    return dist


def get_foreign_classes(distribution, threshold=0.05):
    foreign = {}
    for hospital, class_pcts in distribution.items():
        foreign[hospital] = []
        for cls_name, pct in class_pcts.items():
            if pct < threshold:
                cls_idx = CLASSES.index(cls_name)
                foreign[hospital].append(cls_idx)
    return foreign


def accuracy(rows):
    if not rows:
        return None
    correct = sum(1 for r in rows if r["y_true"] == r["y_pred"])
    return correct / len(rows)


def _confusion_matrix(rows, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for row in rows:
        matrix[row["y_true"], row["y_pred"]] += 1
    return matrix


def _probability_matrix(rows, n_classes):
    prob_cols = [f"prob_{i}" for i in range(n_classes)]
    if not rows or any(col not in rows[0] for col in prob_cols):
        return None

    probs = []
    for row in rows:
        try:
            probs.append([float(row[col]) for col in prob_cols])
        except (KeyError, TypeError, ValueError):
            return None
    return np.array(probs, dtype=float)


def _binary_auc_score(y_binary, scores):
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_binary == 1))
    n_neg = int(np.sum(y_binary == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = np.sum(ranks[y_binary == 1])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _multiclass_ovr_auc_score(y_true, probs, n_classes):
    aucs = []
    for cls_idx in range(n_classes):
        y_binary = (y_true == cls_idx).astype(int)
        auc = _binary_auc_score(y_binary, probs[:, cls_idx])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None
    return float(np.mean(aucs))


def compute_classification_metrics(rows, n_classes=None):
    if not rows:
        return None

    if n_classes is None:
        n_classes = len(CLASSES)

    y_true = np.array([r["y_true"] for r in rows], dtype=int)
    y_pred = np.array([r["y_pred"] for r in rows], dtype=int)
    matrix = _confusion_matrix(rows, n_classes)

    accuracy_value = float(np.mean(y_true == y_pred))

    sensitivities = []
    f1_values = []
    for cls_idx in range(n_classes):
        tp = matrix[cls_idx, cls_idx]
        fn = matrix[cls_idx, :].sum() - tp
        fp = matrix[:, cls_idx].sum() - tp

        support = tp + fn
        recall = tp / support if support > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        sensitivities.append(recall)
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1_values.append(np.nan)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))

    valid_sensitivities = [v for v in sensitivities if not np.isnan(v)]
    balanced_accuracy = float(np.mean(valid_sensitivities)) if valid_sensitivities else None
    min_sensitivity = float(np.min(valid_sensitivities)) if valid_sensitivities else None

    valid_f1_values = [v for v in f1_values if not np.isnan(v)]
    f1score = float(np.mean(valid_f1_values)) if valid_f1_values else None

    total = matrix.sum()
    if total > 0:
        observed = np.trace(matrix) / total
        expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total * total)
        kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
        kappa = None if np.isnan(kappa) else float(kappa)
    else:
        kappa = None

    auc = None
    probs = _probability_matrix(rows, n_classes)
    if probs is not None and len(np.unique(y_true)) > 1:
        if roc_auc_score is not None:
            try:
                auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                          labels=list(range(n_classes))))
            except ValueError:
                auc = None
        else:
            auc = _multiclass_ovr_auc_score(y_true, probs, n_classes)

    return {
        "balanced_accuracy": balanced_accuracy,
        "f1score": f1score,
        "accuracy": accuracy_value,
        "auc": auc,
        "kappa": kappa,
        "minima_sensibilidad": min_sensitivity,
    }


def compute_classification_metrics_by_seed(predictions_by_seed):
    result = {}
    for seed, rows in predictions_by_seed.items():
        metrics = compute_classification_metrics(rows)
        if metrics is not None:
            result[seed] = metrics
    return result


def summarize_metric_values(metrics_by_seed):
    metric_names = [
        "balanced_accuracy",
        "f1score",
        "accuracy",
        "auc",
        "kappa",
        "minima_sensibilidad",
    ]
    summary = {}
    for metric in metric_names:
        values = [
            metrics[metric]
            for metrics in metrics_by_seed.values()
            if metrics.get(metric) is not None
        ]
        if values:
            summary[metric] = (float(np.mean(values)), float(np.std(values)))
        else:
            summary[metric] = (None, None)
    return summary


def format_metric_value(value, digits=6):
    if value is None:
        return "NA"
    return round(value, digits)


def accuracy_by_class(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["y_true"]].append(r["y_pred"])
    result = {}
    for cls_idx, preds in by_class.items():
        correct = sum(1 for p in preds if p == cls_idx)
        result[cls_idx] = (correct / len(preds), len(preds))
    return result


def accuracy_by_hospital(rows):
    by_hospital = defaultdict(list)
    for r in rows:
        by_hospital[r["hospital"]].append(r)
    result = {}
    for hospital, hospital_rows in by_hospital.items():
        result[hospital] = accuracy(hospital_rows)
    return result


# ── Funciones de cada pieza ──────────────────────────────────────────────────

def compute_global_accuracy(predictions_by_seed):
    accs = []
    for seed, rows in predictions_by_seed.items():
        acc = accuracy(rows)
        if acc is not None:
            accs.append(acc)
    return np.mean(accs), np.std(accs), accs


def compute_per_hospital(predictions_by_seed):
    all_results = defaultdict(list)
    for seed, rows in predictions_by_seed.items():
        by_hosp = accuracy_by_hospital(rows)
        for hospital, acc in by_hosp.items():
            all_results[hospital].append(acc)
    summary = {}
    for hospital, accs in all_results.items():
        summary[hospital] = (np.mean(accs), np.std(accs))
    return summary


def compute_per_class_per_hospital(predictions_by_seed):
    all_results = defaultdict(lambda: defaultdict(list))
    for seed, rows in predictions_by_seed.items():
        by_hosp = defaultdict(list)
        for r in rows:
            by_hosp[r["hospital"]].append(r)
        for hospital, hospital_rows in by_hosp.items():
            by_cls = accuracy_by_class(hospital_rows)
            for cls_idx, (acc, n) in by_cls.items():
                all_results[hospital][cls_idx].append(acc)
    summary = {}
    for hospital, classes in all_results.items():
        summary[hospital] = {}
        for cls_idx, accs in classes.items():
            summary[hospital][cls_idx] = (np.mean(accs), np.std(accs))
    return summary


def compute_foreign_accuracy(predictions_by_seed, foreign_classes):
    """Pieza 4 para métodos federados: accuracy en clases foráneas sobre test global."""
    # Per-hospital per-seed
    per_hospital_per_seed = defaultdict(dict)
    media_per_seed = {}

    for seed, rows in predictions_by_seed.items():
        for hospital, foreign_cls_indices in foreign_classes.items():
            if not foreign_cls_indices:
                continue
            foreign_rows = [r for r in rows if r["y_true"] in foreign_cls_indices]
            acc = accuracy(foreign_rows)
            if acc is not None:
                per_hospital_per_seed[hospital][seed] = acc

        # MEDIA across all foreign classes
        all_foreign_indices = set()
        for indices in foreign_classes.values():
            all_foreign_indices.update(indices)
        if all_foreign_indices:
            foreign_rows = [r for r in rows if r["y_true"] in all_foreign_indices]
            acc = accuracy(foreign_rows)
            if acc is not None:
                media_per_seed[seed] = acc

    summary = {}
    for hospital, seed_accs in per_hospital_per_seed.items():
        vals = list(seed_accs.values())
        summary[hospital] = (np.mean(vals), np.std(vals))

    if media_per_seed:
        vals = list(media_per_seed.values())
        summary["MEDIA"] = (np.mean(vals), np.std(vals))
    else:
        summary["MEDIA"] = (None, None)

    return summary, per_hospital_per_seed, media_per_seed


def compute_baseline_foreign_accuracy(baseline_global_by_seed, foreign_classes, hospitals):
    """
    Pieza 4 para baseline local: para cada hospital-modelo, calcula accuracy
    en las clases foráneas de ESE hospital sobre el test global.
    """
    per_hospital_per_seed = defaultdict(dict)

    for seed, hospital_preds in baseline_global_by_seed.items():
        for hospital, rows in hospital_preds.items():
            foreign_cls_indices = foreign_classes.get(hospital, [])
            if not foreign_cls_indices:
                continue
            foreign_rows = [r for r in rows if r["y_true"] in foreign_cls_indices]
            acc = accuracy(foreign_rows)
            if acc is not None:
                per_hospital_per_seed[hospital][seed] = acc

    summary = {}
    for hospital, seed_accs in per_hospital_per_seed.items():
        vals = list(seed_accs.values())
        summary[hospital] = (np.mean(vals), np.std(vals))

    # MEDIA: mean of hospital means per seed
    media_per_seed = {}
    all_seeds_present = set()
    for seed_accs in per_hospital_per_seed.values():
        all_seeds_present.update(seed_accs.keys())
    for seed in all_seeds_present:
        hosp_accs = [per_hospital_per_seed[h][seed]
                     for h in per_hospital_per_seed if seed in per_hospital_per_seed[h]]
        if hosp_accs:
            media_per_seed[seed] = np.mean(hosp_accs)

    if media_per_seed:
        vals = list(media_per_seed.values())
        summary["MEDIA"] = (np.mean(vals), np.std(vals))
    else:
        summary["MEDIA"] = (None, None)

    return summary, per_hospital_per_seed, media_per_seed


def load_times(results_dir, seeds):
    total_times = []
    agg_times = []
    for seed in seeds:
        tiempos_path = results_dir / f"seed_{seed}" / "tiempos.csv"
        if not tiempos_path.exists():
            continue
        with open(tiempos_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            delimiter = ";" if ";" in first_line else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if row["ronda"] == "TOTAL":
                    total_times.append(float(row["tiempo_ronda_seg"]))
                    agg_times.append(float(row["tiempo_agregacion_seg"]))
    if total_times:
        return {
            "tiempo_total_mean": np.mean(total_times),
            "tiempo_total_std": np.std(total_times),
            "tiempo_agg_mean": np.mean(agg_times),
            "tiempo_agg_std": np.std(agg_times),
        }
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    scenario = SCENARIO
    config_path = None
    for name in ["config_iid.yaml", "config_semi_iid.yaml", "config_non_iid.yaml"]:
        candidate = parent_dir / name
        if candidate.exists():
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError("No se encontró ningún config YAML en el directorio padre.")

    # K y escenario se controlan arriba; no se leen del YAML.

    scenario = SCENARIO
    hospitals = [f"Hospital_{i}" for i in range(1, K + 1)]

    results_root = parent_dir / f"results_{scenario}" / f"K{K}"
    output_dir = results_root / "metricas_finales"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Escenario: {scenario}")
    print(f"Hospitales: {K}")
    print(f"Results root: {results_root}")
    print(f"Output: {output_dir}")
    print()

    # ── Cargar distribución de clases ────────────────────────────────────
    dist_table_path = parent_dir / DISTRIBUTION_TABLE
    if not dist_table_path.exists():
        dist_table_path = script_dir / DISTRIBUTION_TABLE

    if dist_table_path.exists():
        distribution = load_distribution_table(dist_table_path)
        foreign_classes = get_foreign_classes(distribution, FOREIGN_THRESHOLD)
        print(f"Distribución cargada desde: {dist_table_path}")
        print("Clases foráneas por hospital:")
        for h, indices in foreign_classes.items():
            names = [CLASSES[i] for i in indices]
            print(f"  {h}: {names if names else 'ninguna'}")
        print()
    else:
        print(f"AVISO: No se encontró {dist_table_path}. No se calcularán clases foráneas.")
        distribution = None
        foreign_classes = None

    # ── Descubrir métodos disponibles ────────────────────────────────────
    methods = {}

    baseline_dir = results_root / "baseline_local"
    if baseline_dir.exists():
        methods["baseline_local"] = baseline_dir

    for method in FED_METHODS:
        method_dir = results_root / method
        if method_dir.exists():
            methods[method] = method_dir

    for solver in MH_SOLVERS:
        solver_dir = results_root / "mh" / solver
        if solver_dir.exists():
            methods[f"mh_{solver}"] = solver_dir

    print(f"Métodos encontrados: {list(methods.keys())}")
    print()

    # ── Cargar predicciones ──────────────────────────────────────────────
    all_predictions = {}
    baseline_global_preds = {}

    for method_name, method_dir in methods.items():
        all_predictions[method_name] = {}

        for seed in SEEDS:
            if method_name == "baseline_local":
                rows = load_baseline_local_predictions(method_dir, seed, hospitals)
                global_preds = load_baseline_global_predictions(method_dir, seed, hospitals)
                if global_preds:
                    baseline_global_preds[seed] = global_preds
            else:
                pred_path = method_dir / f"seed_{seed}" / "predicciones_test.csv"
                if not pred_path.exists():
                    continue
                rows = load_predictions(pred_path)

            if rows:
                all_predictions[method_name][seed] = rows

        n_seeds = len(all_predictions[method_name])
        if n_seeds > 0:
            print(f"  {method_name}: {n_seeds} seeds cargadas")
        else:
            print(f"  {method_name}: SIN DATOS")

    print()

    # ══════════════════════════════════════════════════════════════════════
    # Metricas globales de clasificacion
    print("=" * 60)
    print("METRICAS GLOBALES DE CLASIFICACION")
    print("=" * 60)

    classification_metrics = {}
    classification_metrics_by_seed = {}
    metric_names = [
        "balanced_accuracy",
        "f1score",
        "accuracy",
        "auc",
        "kappa",
        "minima_sensibilidad",
    ]

    for method_name, preds_by_seed in all_predictions.items():
        if not preds_by_seed:
            continue
        by_seed = compute_classification_metrics_by_seed(preds_by_seed)
        summary = summarize_metric_values(by_seed)
        classification_metrics_by_seed[method_name] = by_seed
        classification_metrics[method_name] = summary

        printable = []
        for metric in metric_names:
            mean_value, std_value = summary[metric]
            if mean_value is None:
                printable.append(f"{metric}=NA")
            else:
                printable.append(f"{metric}={mean_value:.4f}+/-{std_value:.4f}")
        print(f"  {method_name:20s}: " + " | ".join(printable))

    csv_path = output_dir / "tabla_metricas_clasificacion.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        header = ["metodo"]
        for metric in metric_names:
            header.extend([f"{metric}_mean", f"{metric}_std"])
        for seed in SEEDS:
            for metric in metric_names:
                header.append(f"seed_{seed}_{metric}")
        writer.writerow(header)

        for method, summary in classification_metrics.items():
            row = [method]
            for metric in metric_names:
                mean_value, std_value = summary[metric]
                row.extend([
                    format_metric_value(mean_value),
                    format_metric_value(std_value),
                ])
            by_seed = classification_metrics_by_seed.get(method, {})
            for seed in SEEDS:
                seed_metrics = by_seed.get(seed, {})
                for metric in metric_names:
                    row.append(format_metric_value(seed_metrics.get(metric)))
            writer.writerow(row)
    print(f"\n  Guardado: {csv_path}\n")

    # PIEZA 2: Test global unificado
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PIEZA 2: TEST GLOBAL UNIFICADO")
    print("=" * 60)

    global_results = {}
    for method_name, preds_by_seed in all_predictions.items():
        if not preds_by_seed:
            continue
        mean_acc, std_acc, accs = compute_global_accuracy(preds_by_seed)
        global_results[method_name] = {"mean": mean_acc, "std": std_acc, "accs": accs}
        print(f"  {method_name:20s}: {mean_acc:.4f} ± {std_acc:.4f}")

    csv_path = output_dir / "tabla_accuracy_global.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["metodo", "accuracy_mean", "accuracy_std"] + [f"seed_{s}" for s in SEEDS])
        for method, data in global_results.items():
            row = [method, round(data["mean"], 6), round(data["std"], 6)]
            row += [round(a, 6) for a in data["accs"]]
            writer.writerow(row)
    print(f"\n  Guardado: {csv_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PIEZA 5: Evaluación por hospital
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PIEZA 5: ACCURACY POR HOSPITAL")
    print("=" * 60)

    hospital_results = {}
    hospital_per_seed = {}  # {method: {hospital: {seed: acc}}}
    for method_name, preds_by_seed in all_predictions.items():
        if not preds_by_seed:
            continue
        summary = compute_per_hospital(preds_by_seed)
        hospital_results[method_name] = summary
        # Collect per-seed
        per_seed = defaultdict(dict)
        for seed, rows in preds_by_seed.items():
            by_hosp = accuracy_by_hospital(rows)
            for hospital, acc in by_hosp.items():
                per_seed[hospital][seed] = acc
        hospital_per_seed[method_name] = per_seed
        print(f"\n  {method_name}:")
        for hospital in hospitals:
            if hospital in summary:
                m, s = summary[hospital]
                print(f"    {hospital}: {m:.4f} ± {s:.4f}")

    csv_path = output_dir / "tabla_por_hospital.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["metodo"] + hospitals)
        for method, summary in hospital_results.items():
            row = [method]
            for hospital in hospitals:
                if hospital in summary:
                    m, s = summary[hospital]
                    row.append(f"{m:.4f}±{s:.4f}")
                else:
                    row.append("NA")
            writer.writerow(row)
    print(f"\n  Guardado: {csv_path}")

    csv_path_seeds = output_dir / "tabla_por_hospital_por_seed.csv"
    with open(csv_path_seeds, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["metodo", "seed"] + hospitals)
        for method in hospital_results.keys():
            per_seed = hospital_per_seed.get(method, {})
            seeds_present = set()
            for h_data in per_seed.values():
                seeds_present.update(h_data.keys())
            for seed in sorted(seeds_present):
                row = [method, seed]
                for hospital in hospitals:
                    if hospital in per_seed and seed in per_seed[hospital]:
                        row.append(f"{per_seed[hospital][seed]:.4f}")
                    else:
                        row.append("NA")
                writer.writerow(row)
    print(f"  Guardado: {csv_path_seeds}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PIEZA 3: Accuracy por clase en cada hospital
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PIEZA 3: ACCURACY POR CLASE EN CADA HOSPITAL")
    print("=" * 60)

    csv_path = output_dir / "tabla_por_clase.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["metodo", "hospital"] + CLASSES)

        for method_name, preds_by_seed in all_predictions.items():
            if not preds_by_seed:
                continue
            summary = compute_per_class_per_hospital(preds_by_seed)

            for hospital in hospitals:
                if hospital not in summary:
                    continue
                row = [method_name, hospital]
                for cls_idx in range(len(CLASSES)):
                    if cls_idx in summary[hospital]:
                        m, s = summary[hospital][cls_idx]
                        row.append(f"{m:.4f}±{s:.4f}")
                    else:
                        row.append("NA")
                writer.writerow(row)

                print(f"  {method_name} | {hospital}:")
                for cls_idx in range(len(CLASSES)):
                    if cls_idx in summary[hospital]:
                        m, s = summary[hospital][cls_idx]
                        print(f"    {CLASSES[cls_idx]:12s}: {m:.4f} ± {s:.4f}")

    print(f"\n  Guardado: {csv_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PIEZA 4: Test de clases foráneas
    # ══════════════════════════════════════════════════════════════════════
    if foreign_classes:
        print("=" * 60)
        print("PIEZA 4: TEST DE CLASES FORÁNEAS")
        print("=" * 60)

        foreign_results = {}
        foreign_per_seed_data = {}  # {method: {hospital: {seed: acc}}}
        foreign_media_per_seed = {}  # {method: {seed: acc}}

        # Baseline local: usa predicciones globales por hospital-modelo
        if baseline_global_preds:
            summary, per_hosp_seed, media_seed = compute_baseline_foreign_accuracy(
                baseline_global_preds, foreign_classes, hospitals)
            foreign_results["baseline_local"] = summary
            foreign_per_seed_data["baseline_local"] = per_hosp_seed
            foreign_media_per_seed["baseline_local"] = media_seed
            print(f"\n  baseline_local:")
            for hospital in hospitals:
                if hospital in summary:
                    m, s = summary[hospital]
                    print(f"    {hospital}: {m:.4f} ± {s:.4f}")
            if "MEDIA" in summary and summary["MEDIA"][0] is not None:
                m, s = summary["MEDIA"]
                print(f"    MEDIA:      {m:.4f} ± {s:.4f}")

        # Métodos federados y MH
        for method_name, preds_by_seed in all_predictions.items():
            if method_name == "baseline_local" or not preds_by_seed:
                continue
            summary, per_hosp_seed, media_seed = compute_foreign_accuracy(
                preds_by_seed, foreign_classes)
            foreign_results[method_name] = summary
            foreign_per_seed_data[method_name] = per_hosp_seed
            foreign_media_per_seed[method_name] = media_seed
            print(f"\n  {method_name}:")
            for hospital in hospitals:
                if hospital in summary:
                    m, s = summary[hospital]
                    print(f"    {hospital}: {m:.4f} ± {s:.4f}")
            if "MEDIA" in summary and summary["MEDIA"][0] is not None:
                m, s = summary["MEDIA"]
                print(f"    MEDIA:      {m:.4f} ± {s:.4f}")

        # --- CSV resumen (media±std) ---
        csv_path = output_dir / "tabla_clases_foraneas.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["metodo"] + hospitals + ["MEDIA"])
            for method, summary in foreign_results.items():
                row = [method]
                for hospital in hospitals:
                    if hospital in summary:
                        m, s = summary[hospital]
                        row.append(f"{m:.4f}±{s:.4f}")
                    else:
                        row.append("NA")
                if "MEDIA" in summary and summary["MEDIA"][0] is not None:
                    m, s = summary["MEDIA"]
                    row.append(f"{m:.4f}±{s:.4f}")
                else:
                    row.append("NA")
                writer.writerow(row)
        print(f"\n  Guardado: {csv_path}")

        # --- CSV por semilla ---
        csv_path_seeds = output_dir / "tabla_clases_foraneas_por_seed.csv"
        with open(csv_path_seeds, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["metodo", "seed"] + hospitals + ["MEDIA"])
            for method in foreign_results.keys():
                per_hosp = foreign_per_seed_data.get(method, {})
                media_seeds = foreign_media_per_seed.get(method, {})
                seeds_present = set()
                for h_data in per_hosp.values():
                    seeds_present.update(h_data.keys())
                seeds_present.update(media_seeds.keys())
                for seed in sorted(seeds_present):
                    row = [method, seed]
                    for hospital in hospitals:
                        if hospital in per_hosp and seed in per_hosp[hospital]:
                            row.append(f"{per_hosp[hospital][seed]:.4f}")
                        else:
                            row.append("NA")
                    if seed in media_seeds:
                        row.append(f"{media_seeds[seed]:.4f}")
                    else:
                        row.append("NA")
                    writer.writerow(row)
        print(f"  Guardado: {csv_path_seeds}\n")
    else:
        print("PIEZA 4: OMITIDA (no hay tabla de distribución)\n")

    # ══════════════════════════════════════════════════════════════════════
    # PIEZA 1: Comparación con baseline local
    # ══════════════════════════════════════════════════════════════════════
    if "baseline_local" in all_predictions and all_predictions["baseline_local"]:
        print("=" * 60)
        print("PIEZA 1: COMPARACIÓN CON BASELINE LOCAL")
        print("=" * 60)

        baseline_global_mean, baseline_global_std, _ = compute_global_accuracy(all_predictions["baseline_local"])
        print(f"\n  Baseline local (accuracy en test propio): {baseline_global_mean:.4f} ± {baseline_global_std:.4f}")

        print(f"\n  Ganancia sobre baseline:")
        for method_name, data in global_results.items():
            if method_name == "baseline_local":
                continue
            gain = data["mean"] - baseline_global_mean
            print(f"    {method_name:20s}: +{gain:.4f} ({data['mean']:.4f} vs {baseline_global_mean:.4f})")

        if foreign_classes and "baseline_local" in foreign_results:
            bl_foreign = foreign_results["baseline_local"]
            if "MEDIA" in bl_foreign and bl_foreign["MEDIA"][0] is not None:
                bl_mean = bl_foreign["MEDIA"][0]
                print(f"\n  Ganancia en clases foráneas sobre baseline:")
                for method_name, summary in foreign_results.items():
                    if method_name == "baseline_local":
                        continue
                    if "MEDIA" in summary and summary["MEDIA"][0] is not None:
                        gain = summary["MEDIA"][0] - bl_mean
                        print(f"    {method_name:20s}: +{gain:.4f} ({summary['MEDIA'][0]:.4f} vs {bl_mean:.4f})")

        print()
    else:
        print("PIEZA 1: OMITIDA (no hay baseline local)\n")

    # ══════════════════════════════════════════════════════════════════════
    # TABLA DE TIEMPOS
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TABLA DE TIEMPOS")
    print("=" * 60)

    time_results = {}
    for method_name, method_dir in methods.items():
        if method_name == "baseline_local":
            continue
        times = load_times(method_dir, SEEDS)
        if times:
            time_results[method_name] = times
            print(f"  {method_name:20s}: total {times['tiempo_total_mean']:.1f}±{times['tiempo_total_std']:.1f}s | "
                  f"agregación {times['tiempo_agg_mean']:.2f}±{times['tiempo_agg_std']:.2f}s")

    csv_path = output_dir / "tabla_tiempos.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["metodo", "tiempo_total_mean", "tiempo_total_std",
                         "tiempo_agg_mean", "tiempo_agg_std"])
        for method, times in time_results.items():
            writer.writerow([method,
                             round(times["tiempo_total_mean"], 2),
                             round(times["tiempo_total_std"], 2),
                             round(times["tiempo_agg_mean"], 2),
                             round(times["tiempo_agg_std"], 2)])
    print(f"\n  Guardado: {csv_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    # TEST ESTADÍSTICO (Wilcoxon)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST ESTADÍSTICO (Wilcoxon signed-rank)")
    print("=" * 60)

    reference = "fedavg"
    if wilcoxon is None:
        print("  No se puede calcular: scipy no esta instalado.\n")
    elif reference in global_results and len(global_results[reference]["accs"]) >= 5:
        ref_accs = global_results[reference]["accs"]

        stat_results = []
        for method_name, data in global_results.items():
            if method_name == reference or method_name == "baseline_local":
                continue
            if len(data["accs"]) < 5:
                continue

            method_accs = data["accs"]
            try:
                stat, p_value = wilcoxon(method_accs, ref_accs)
                sig = "SÍ" if p_value < 0.05 else "NO"
                print(f"  {method_name:20s} vs {reference}: p={p_value:.4f} ({sig} significativo)")
                stat_results.append({
                    "metodo": method_name,
                    "referencia": reference,
                    "statistic": round(stat, 4),
                    "p_value": round(p_value, 4),
                    "significativo": sig
                })
            except ValueError as e:
                print(f"  {method_name:20s} vs {reference}: no se pudo calcular ({e})")

        csv_path = output_dir / "test_estadistico.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metodo", "referencia", "statistic", "p_value", "significativo"],
                                    delimiter=";")
            writer.writeheader()
            writer.writerows(stat_results)
        print(f"\n  Guardado: {csv_path}\n")
    else:
        print(f"  No se puede calcular: {reference} no tiene suficientes seeds.\n")

    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("MÉTRICAS COMPLETADAS")
    print(f"Todos los resultados en: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
