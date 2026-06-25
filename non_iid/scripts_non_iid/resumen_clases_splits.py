"""
Genera una tabla resumen por hospital y split (train/val/test) en formato ancho:
- Una fila por hospital y split
- Columnas para cada clase
- Incluye total de imagenes del hospital
- Incluye porcentaje del hospital respecto al total global

Todo queda configurado dentro del script.
Solo hay que ejecutar:
python resumen_clases_splits.py
"""

import csv
from collections import Counter
from pathlib import Path


# ==========================================
# CONFIGURACION FIJA
# ==========================================
SPLITS_ROOT = Path("../splits_non_iid")
OUTPUT_CSV = Path("tabla_non_iid.csv")
DECIMALES = 2

CLASES = ["BreastMRI", "CXR", "HeadCT", "ChestCT", "Hand", "AbdomenCT"]


def extraer_clase_desde_ruta(ruta_relativa: str) -> str:
    """Extrae la clase desde rutas del tipo Hospital_X/Clase/imagen.ext."""
    partes = ruta_relativa.strip().replace("\\", "/").split("/")
    if len(partes) >= 2 and partes[1].strip():
        return partes[1].strip()
    return "RUTA_INVALIDA"


def procesar_split(path_split: Path) -> Counter:
    """Lee un split y devuelve un contador de clases."""
    conteo = Counter()
    if not path_split.exists():
        return conteo

    with path_split.open("r", encoding="utf-8") as f:
        for linea in f:
            ruta = linea.strip()
            if not ruta:
                continue
            clase = extraer_clase_desde_ruta(ruta)
            conteo[clase] += 1

    return conteo


def obtener_total_hospital(hospital_dir: Path) -> int:
    total = 0
    for split in ("train", "val", "test"):
        split_file = hospital_dir / f"{split}.txt"
        conteo = procesar_split(split_file)
        total += sum(conteo.values())
    return total


def nombre_hospital_corto(nombre: str) -> str:
    """Convierte 'Hospital_1' -> '1'."""
    if "_" in nombre:
        return nombre.split("_")[-1]
    return nombre


def main() -> None:
    if not SPLITS_ROOT.exists() or not SPLITS_ROOT.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de splits: {SPLITS_ROOT.resolve()}"
        )

    hospitales = sorted([p for p in SPLITS_ROOT.iterdir() if p.is_dir()])
    if not hospitales:
        print(f"No se encontraron hospitales en {SPLITS_ROOT.resolve()}")
        return

    total_global = sum(obtener_total_hospital(h) for h in hospitales)

    filas = []

    for hospital_dir in hospitales:
        hospital_nombre = hospital_dir.name
        hospital_corto = nombre_hospital_corto(hospital_nombre)

        total_hospital = obtener_total_hospital(hospital_dir)
        pct_hospital = (
            round((total_hospital / total_global) * 100, DECIMALES)
            if total_global > 0 else 0
        )

        for split in ("train", "val", "test"):
            split_file = hospital_dir / f"{split}.txt"
            conteo = procesar_split(split_file)
            total_split = sum(conteo.values())

            fila = {
                "Hospital": hospital_corto,
                "%_Hospital": f"{pct_hospital:.{DECIMALES}f}%",
                "Split": split.capitalize(),
                "Total_split": total_split,
            }

            for clase in CLASES:
                n = conteo.get(clase, 0)
                pct = (n / total_split) * 100 if total_split > 0 else 0
                fila[clase] = f"{pct:.{DECIMALES}f}%"

            filas.append(fila)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    columnas = ["Hospital", "%_Hospital", "Split", "Total_split"] + CLASES

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, delimiter=";")
        writer.writeheader()
        writer.writerows(filas)

    print(f"CSV guardado en: {OUTPUT_CSV.resolve()}")
    print(f"Total global de imagenes: {total_global}")
    print(f"Filas escritas: {len(filas)}")


if __name__ == "__main__":
    main()