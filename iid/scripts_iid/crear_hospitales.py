"""
Este script crea las carpetas de hospitales y reparte las imágenes
usando una partición IID (reparto uniforme estricto).
"""
import os, sys, yaml, shutil, random
from collections import defaultdict
from tqdm import tqdm

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config_iid.yaml")

def cargar_config(path):
    if not os.path.isfile(path):
        print(f"[ERROR] No encontré la config: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clases_presentes(raw_dir):
    return sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

def imagenes_por_clase(raw_dir, valid_ext):
    imgs_by_class = defaultdict(list)
    for cls in clases_presentes(raw_dir):
        cls_path = os.path.join(raw_dir, cls)
        for f in os.listdir(cls_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_ext:
                imgs_by_class[cls].append(os.path.join(cls_path, f))
    return imgs_by_class

def asegurar_estructura(out_dir, n_h, clases):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, n_h + 1):
        hpath = os.path.join(out_dir, f"Hospital_{i}")
        os.makedirs(hpath, exist_ok=True)
        for c in clases:
            os.makedirs(os.path.join(hpath, c), exist_ok=True)

def limpiar_hospitales(out_dir, n_h, clases):
    for i in range(1, n_h + 1):
        hpath = os.path.join(out_dir, f"Hospital_{i}")
        for c in clases:
            cpath = os.path.join(hpath, c)
            if os.path.isdir(cpath):
                for f in os.listdir(cpath):
                    fp = os.path.join(cpath, f)
                    if os.path.isfile(fp):
                        os.remove(fp)

def main():
    cfg = cargar_config(CONFIG_PATH)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = cfg["raw_dataset_dir"] if os.path.isabs(cfg["raw_dataset_dir"]) else os.path.join(base_dir, cfg["raw_dataset_dir"])
    out_dir = cfg["federated_output_dir"] if os.path.isabs(cfg["federated_output_dir"]) else os.path.join(base_dir, cfg["federated_output_dir"])

    n_h = int(cfg["n_hospitals"])
    valid_ext = set(e.lower() for e in cfg.get("valid_ext", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]))

    print(f"RAW = {raw_dir}")
    print(f"FED = {out_dir}\n")

    clases = clases_presentes(raw_dir)
    asegurar_estructura(out_dir, n_h, clases)
    limpiar_hospitales(out_dir, n_h, clases)

    imgs_by_class = imagenes_por_clase(raw_dir, valid_ext)

    print(f"== Reparto IID Uniforme ==")
    print(f"Hospitales={n_h} | clases={len(clases)} | seed={seed}\n")

    total_copiadas = 0
    for cls in clases:
        imgs = imgs_by_class[cls]
        random.shuffle(imgs)
        N = len(imgs)

        # Reparto equitativo
        base = N // n_h
        resto = N % n_h
        plan = [base + (1 if i < resto else 0) for i in range(n_h)]
        random.shuffle(plan) # Mezclar quién se lleva el resto

        idx = 0
        with tqdm(total=N, desc=f"Copiando {cls:12}", unit="img") as pbar:
            for i, count in enumerate(plan, start=1):
                if count <= 0: continue
                subset = imgs[idx:idx+count]
                idx += count
                dest = os.path.join(out_dir, f"Hospital_{i}", cls)
                for src in subset:
                    shutil.copy(src, os.path.join(dest, os.path.basename(src)))
                    pbar.update(1)
                total_copiadas += count

        resumen = ", ".join([f"H{i}:{plan[i-1]}" for i in range(1, n_h+1)])
        print(f"  -> Reparto: {resumen} (suma={sum(plan)})")

    print(f"\nListo. Imágenes copiadas en total: {total_copiadas}")

if __name__ == "__main__":
    main()