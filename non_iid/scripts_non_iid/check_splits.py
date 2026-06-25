# ==============================================================
# PARTE 1 — Verificar el contenido del split (solo lectura del .txt)
# ==============================================================

import os
from PIL import Image

# Ruta al archivo del split
SPLIT_PATH = os.path.join("..", "splits_non_iid", "Hospital_1", "train.txt")

# Leer las líneas del archivo (cada línea = ruta de una imagen)
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    rutas = [line.strip() for line in f.readlines() if line.strip()]

print(f"Total de imágenes en {SPLIT_PATH}: {len(rutas)}\n")

# Mostrar las primeras 10 rutas
print("=== Ejemplos de rutas ===")
for r in rutas[:10]:
    print(" ", r)

# Extraer clases a partir de las rutas
clases = set([r.split("/")[1] for r in rutas])
print(f"\nClases detectadas: {sorted(list(clases))}")



# ==============================================================
# PARTE 2 — Comprobar que las rutas del split apuntan a imágenes reales
# ==============================================================

# Carpeta raíz donde están los hospitales y sus clases
DATA_ROOT  = os.path.join("..", "data_non_iid")

print("\n=== Verificación de carga de imágenes (primeras 10) ===")

ok, fails = 0, 0
for r in rutas[:10]:
    abs_path = os.path.join(DATA_ROOT, r)  # Ruta absoluta a la imagen
    cls = r.split("/")[1]                  # Clase = nombre de la carpeta
    try:
        with Image.open(abs_path) as im:
            im.load()                      # Carga real de la imagen
            print(f"[OK] {cls} | {r} | {im.mode} {im.size}")
            ok += 1
    except Exception as e:
        print(f"[FALLO] {r} -> {e}")
        fails += 1

print(f"\nResumen: OK={ok}  FALLOS={fails}")
