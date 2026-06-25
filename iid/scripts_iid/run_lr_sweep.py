import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

VAL_RE = re.compile(r"Val Loss:\s*([0-9]*\.?[0-9]+)\s*\|\s*Val Acc:\s*([0-9]*\.?[0-9]+)")
TEST_RE = re.compile(r"Test Loss medio:\s*([0-9]*\.?[0-9]+).*?Test Acc media:\s*([0-9]*\.?[0-9]+)", re.S)
ROUND_RE = re.compile(r"Ronda\s+(\d+)\s*->\s*Val Loss:\s*([0-9]*\.?[0-9]+)\s*\|\s*Val Acc:\s*([0-9]*\.?[0-9]+)")


def parse_args():
    p = argparse.ArgumentParser(description="Run LR + local_epochs grid search for federated training and summarize results.")
    p.add_argument("--train-script", default="train_federated.py", help="Path to train_federated.py")
    p.add_argument("--base-config", default="config_iid.yaml", help="Base YAML config")
    p.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02],
        help="Learning rates to test",
    )
    p.add_argument(
        "--local-epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 7, 10, 15, 20],
        help="Local epochs to test",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--output-root", default="grid_sweeps", help="Folder to store all sweep outputs")
    p.add_argument("--keep-raw-results", action="store_true", help="Copy raw result txt/csv files")
    return p.parse_args()


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def lr_to_tag(lr: float) -> str:
    s = f"{lr:.10g}"
    return s.replace("-", "m").replace(".", "p")


def find_results_dir(train_script: Path, aggregator: str) -> Path:
    return (train_script.resolve().parent / f"../results_iid/{aggregator}").resolve()


def parse_training_output(stdout_text: str, results_txt: Path):
    out = {
        "final_val_loss": None,
        "final_val_acc": None,
        "best_val_acc": None,
        "best_round": None,
        "final_test_loss": None,
        "final_test_acc": None,
        "rounds": [],
    }

    round_matches = ROUND_RE.findall(stdout_text)
    if not round_matches and results_txt.exists():
        txt = results_txt.read_text(encoding="utf-8", errors="ignore")
        round_matches = ROUND_RE.findall(txt)
        test_match = TEST_RE.search(txt)
    else:
        test_match = TEST_RE.search(stdout_text)
        if not test_match and results_txt.exists():
            txt = results_txt.read_text(encoding="utf-8", errors="ignore")
            test_match = TEST_RE.search(txt)

    if round_matches:
        for r, loss, acc in round_matches:
            out["rounds"].append(
                {"round": int(r), "val_loss": float(loss), "val_acc": float(acc)}
            )
        last = out["rounds"][-1]
        best = max(out["rounds"], key=lambda x: x["val_acc"])
        out["final_val_loss"] = last["val_loss"]
        out["final_val_acc"] = last["val_acc"]
        out["best_val_acc"] = best["val_acc"]
        out["best_round"] = best["round"]

    if test_match:
        out["final_test_loss"] = float(test_match.group(1))
        out["final_test_acc"] = float(test_match.group(2))

    return out


def stability_note(rounds):
    if len(rounds) < 3:
        return "insuficientes rondas"
    accs = [r["val_acc"] for r in rounds]
    drops = sum(1 for i in range(1, len(accs)) if accs[i] < accs[i - 1])
    max_drop = max((accs[i - 1] - accs[i] for i in range(1, len(accs))), default=0.0)
    if drops == 0:
        return "muy estable"
    if max_drop <= 0.01 and drops <= 2:
        return "bastante estable"
    if max_drop <= 0.03:
        return "algo inestable"
    return "inestable"


def choose_recommendation(results):
    valid = [r for r in results if r["final_val_acc"] is not None]
    if not valid:
        return None

    # Prioridad:
    # 1. mayor final_val_acc
    # 2. menor final_val_loss
    # 3. menor local_epochs si todo lo demás es parecido
    valid.sort(
        key=lambda x: (
            x["final_val_acc"],
            -(x["final_val_loss"] if x["final_val_loss"] is not None else 1e9),
            -1 / x["local_epochs"],
        ),
        reverse=True,
    )
    return valid[0]


def main():
    args = parse_args()
    train_script = Path(args.train_script).resolve()
    base_config = Path(args.base_config).resolve()

    if not train_script.exists():
        raise FileNotFoundError(f"No existe el script de entrenamiento: {train_script}")
    if not base_config.exists():
        raise FileNotFoundError(f"No existe la config base: {base_config}")

    base_cfg = load_yaml(base_config)
    aggregator = base_cfg.get("aggregator_name", "fedavg")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_root).resolve() / f"sweep_{timestamp}"
    configs_dir = sweep_dir / "configs"
    logs_dir = sweep_dir / "logs"
    raw_dir = sweep_dir / "raw_results"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_raw_results:
        raw_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"Sweep directory: {sweep_dir}")
    print(f"Training script: {train_script}")
    print(f"Base config: {base_config}")
    print(f"LRs: {args.lrs}")
    print(f"Local epochs: {args.local_epochs}")

    original_results_dir = find_results_dir(train_script, "fedavg")

    total_runs = len(args.lrs) * len(args.local_epochs)
    run_counter = 0

    for lr in args.lrs:
        for le in args.local_epochs:
            run_counter += 1
            lr_tag = lr_to_tag(lr)
            run_name = f"lr_{lr_tag}_le_{le}"
            run_config = configs_dir / f"config_{run_name}.yaml"
            run_log = logs_dir / f"{run_name}.log"

            cfg = dict(base_cfg)
            cfg["learning_rate"] = float(lr)
            cfg["local_epochs"] = int(le)
            save_yaml(cfg, run_config)

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(run_config)

            print(f"\n=== [{run_counter}/{total_runs}] Running {run_name} (lr={lr}, local_epochs={le}) ===")

            captured_lines = []

            with open(run_log, "w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    [args.python, str(train_script)],
                    cwd=str(train_script.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                )

                for line in proc.stdout:
                    print(line, end="")          # lo ves en consola
                    log_f.write(line)            # se guarda en el log en tiempo real
                    log_f.flush()                # fuerza escritura inmediata
                    captured_lines.append(line)

                proc.wait()

            stdout_text = "".join(captured_lines)

            results_txt = original_results_dir / "fedavg_resultados.txt"
            parsed = parse_training_output(stdout_text, results_txt)
            note = stability_note(parsed["rounds"])

            raw_result_copy = None
            if args.keep_raw_results and original_results_dir.exists():
                target = raw_dir / run_name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(original_results_dir, target)
                raw_result_copy = str(target)

            summary_rows.append(
                {
                    "lr": lr,
                    "local_epochs": le,
                    "status": "ok" if proc.returncode == 0 else f"error_{proc.returncode}",
                    "final_val_acc": parsed["final_val_acc"],
                    "final_val_loss": parsed["final_val_loss"],
                    "best_val_acc": parsed["best_val_acc"],
                    "best_round": parsed["best_round"],
                    "final_test_acc": parsed["final_test_acc"],
                    "final_test_loss": parsed["final_test_loss"],
                    "stability": note,
                    "log_file": str(run_log),
                    "raw_results_dir": raw_result_copy or "",
                }
            )

    summary_csv = sweep_dir / "summary_grid_results.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr",
                "local_epochs",
                "status",
                "final_val_acc",
                "final_val_loss",
                "best_val_acc",
                "best_round",
                "final_test_acc",
                "final_test_loss",
                "stability",
                "log_file",
                "raw_results_dir",
            ],
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    recommendation = choose_recommendation(summary_rows)

    report_md = sweep_dir / "summary_grid_results.md"
    lines = []
    lines.append("# Resumen de grid search: learning_rate + local_epochs\n\n")
    lines.append(f"Generado: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append("## Criterio recomendado\n\n")
    lines.append(
        "Elegir principalmente por **final_val_acc**, usando **final_val_loss** como desempate, revisando la estabilidad, "
        "y si dos opciones son muy parecidas, preferir la de menor **local_epochs**.\n\n"
    )

    if recommendation:
        lines.append("## Recomendación automática\n\n")
        lines.append(
            f"Mejor candidato: **lr = {recommendation['lr']}**, **local_epochs = {recommendation['local_epochs']}**  \n"
            f"- final_val_acc: {recommendation['final_val_acc']}  \n"
            f"- final_val_loss: {recommendation['final_val_loss']}  \n"
            f"- estabilidad: {recommendation['stability']}\n\n"
        )
    else:
        lines.append("## Recomendación automática\n\n")
        lines.append("No se pudo generar una recomendación porque no hubo resultados válidos.\n\n")

    lines.append("## Tabla resumen\n\n")
    lines.append("| lr | local_epochs | status | final_val_acc | final_val_loss | best_val_acc | best_round | final_test_acc | estabilidad |\n")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---|\n")
    for row in summary_rows:
        lines.append(
            f"| {row['lr']} | {row['local_epochs']} | {row['status']} | {row['final_val_acc']} | {row['final_val_loss']} | "
            f"{row['best_val_acc']} | {row['best_round']} | {row['final_test_acc']} | {row['stability']} |\n"
        )

    lines.append("\n## Qué mirar mañana\n\n")
    lines.append("1. Quédate primero con el mayor **final_val_acc**.\n")
    lines.append("2. Si dos salen parecidos, elige el menor **final_val_loss**.\n")
    lines.append("3. Si siguen muy parecidos, prefiere el más estable.\n")
    lines.append("4. Si aún así están muy cerca, prefiere el menor **local_epochs**.\n")
    lines.append("5. Usa **final_test_acc** solo como referencia final, no para tunear.\n")

    report_md.write_text("".join(lines), encoding="utf-8")

    print(f"\nResumen CSV: {summary_csv}")
    print(f"Resumen MD:  {report_md}")
    if recommendation:
        print(
            f"Recomendación automática: lr = {recommendation['lr']}, "
            f"local_epochs = {recommendation['local_epochs']}"
        )


if __name__ == "__main__":
    main()