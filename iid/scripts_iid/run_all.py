"""
Ejecuta train_federated.py con una configuración concreta por método.

Uso:
    python run_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path


TRAIN_SCRIPT = "train_federated.py"
MH_AGGREGATION = Path("funciones_de_agregacion") / "mh_aggregation.py"
SEEDS = [16, 999, 1935]

RUN_CONFIGS = [
    {
        "aggregator": "fedavg",
        "solver": None,
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
        },
    },
    {
        "aggregator": "fedprox",
        "solver": None,
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "10",
            "RUN_FEDPROX_MU": "0.01",
            "RUN_FEDPROX_VARIABLE_LOCAL_EPOCHS": "0",
        },
    },
    {
        "aggregator": "fedewa",
        "solver": None,
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "10",
        },
    },
    {
        "aggregator": "fedag",
        "solver": None,
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "3",
            "RUN_FEDAG_LR_BETA": "0.1",
        },
    },
    {
        "aggregator": "mh",
        "solver": "random_search",
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
            "RUN_MH_LAMBDA_REG": "0.0001",
            "RUN_MH_GAMMA_REG": "0.05",
        },
    },
    {
        "aggregator": "mh",
        "solver": "de",
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
            "RUN_MH_LAMBDA_REG": "0.0001",
            "RUN_MH_GAMMA_REG": "0.05",
        },
    },
    {
        "aggregator": "mh",
        "solver": "ga",
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
            "RUN_MH_LAMBDA_REG": "0.0001",
            "RUN_MH_GAMMA_REG": "0.05",
        },
    },
    {
        "aggregator": "mh",
        "solver": "pso",
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
            "RUN_MH_LAMBDA_REG": "0.0001",
            "RUN_MH_GAMMA_REG": "0.05",
        },
    },
    {
        "aggregator": "mh",
        "solver": "cro",
        "env": {
            "RUN_LR": "0.1",
            "RUN_LOCAL_EPOCHS": "5",
            "RUN_MH_LAMBDA_REG": "0.0001",
            "RUN_MH_GAMMA_REG": "0.05",
        },
    },
]


def set_aggregator(aggregator):
    """Cambia AGGREGATOR en train_federated.py"""
    path = Path(TRAIN_SCRIPT)
    content = path.read_text(encoding="utf-8")

    # Buscar línea AGGREGATOR = "..."
    import re
    content = re.sub(
        r'^AGGREGATOR\s*=\s*"[^"]*"',
        f'AGGREGATOR = "{aggregator}"',
        content,
        flags=re.MULTILINE
    )
    path.write_text(content, encoding="utf-8")


def set_solver(solver):
    """Cambia SOLVER en mh_aggregation.py"""
    path = Path(MH_AGGREGATION)
    content = path.read_text(encoding="utf-8")

    import re
    content = re.sub(
        r'^SOLVER\s*=\s*"[^"]*"',
        f'SOLVER = "{solver}"',
        content,
        flags=re.MULTILINE
    )
    path.write_text(content, encoding="utf-8")


def main():
    total_start = time.time()
    results = []

    print("=" * 60)
    print("EJECUCIÓN DE MÉTODOS CONFIGURADOS")
    print("=" * 60)
    print(f"Seeds a ejecutar: {SEEDS}")
    print(f"Ejecuciones totales: {len(RUN_CONFIGS) * len(SEEDS)}")
    print()

    run_index = 0
    total_runs = len(RUN_CONFIGS) * len(SEEDS)
    for seed in SEEDS:
        for run_cfg in RUN_CONFIGS:
            run_index += 1
            aggregator = run_cfg["aggregator"]
            solver = run_cfg["solver"]
            method_name = f"{aggregator}/{solver}" if solver else aggregator
            print(f"\n{'='*60}")
            print(f"[{run_index}/{total_runs}] Ejecutando: {method_name} | seed={seed}")
            print(f"{'='*60}")

            # Configurar archivos
            set_aggregator(aggregator)
            if solver:
                set_solver(solver)

            # Ejecutar
            t_start = time.time()
            env = dict(os.environ)
            env.update(run_cfg["env"])
            env["RUN_RANDOM_SEED"] = str(seed)
            proc = subprocess.run(
                [sys.executable, TRAIN_SCRIPT],
                text=True,
                env=env
            )
            t_end = time.time()
            elapsed = t_end - t_start

            status = "OK" if proc.returncode == 0 else f"ERROR (code {proc.returncode})"
            results.append({
                "method": method_name,
                "seed": seed,
                "status": status,
                "time": elapsed
            })

            print(f"\n  {method_name} | seed={seed}: {status} ({elapsed:.1f}s)")

    # Resumen
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['method']:25s} | seed={r['seed']:4d} | {r['status']:10s} | {r['time']:.1f}s")
    print(f"\nTiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Restaurar a fedavg por defecto
    set_aggregator("fedavg")
    set_solver("cro")


if __name__ == "__main__":
    main()
