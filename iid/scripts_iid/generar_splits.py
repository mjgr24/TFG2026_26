# scripts/generar_splits_todos.py
import os
import yaml
from math import floor

# --- Config y rutas robustas ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_PATH = os.path.join(BASE, "config_iid.yaml")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

N_H = int(cfg["n_hospitals"])
FED_ROOT = os.path.join(BASE, cfg["federated_output_dir"])  # data/federated
SPLITS_ROOT = os.path.join(BASE, "splits_iid")
os.makedirs(SPLITS_ROOT, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
P_TRAIN, P_VAL = 0.75, 0.15  # P_TEST = 1 - (P_TRAIN + P_VAL)

def gen_split_for_hospital(hname: str):
    h_root = os.path.join(FED_ROOT, hname)           # data/federated/Hospital_i
    out_dir = os.path.join(SPLITS_ROOT, hname)       # splits/Hospital_i
    os.makedirs(out_dir, exist_ok=True)

    train_p = os.path.join(out_dir, "train.txt")
    val_p   = os.path.join(out_dir, "val.txt")
    test_p  = os.path.join(out_dir, "test.txt")
    open(train_p, "w", encoding="utf-8").close()
    open(val_p,   "w", encoding="utf-8").close()
    open(test_p,  "w", encoding="utf-8").close()

    t_train = t_val = t_test = 0

    # Recorrer clases del hospital
    for cls in sorted(os.listdir(h_root)):
        cdir = os.path.join(h_root, cls)
        if not os.path.isdir(cdir):
            continue

        files = [f for f in sorted(os.listdir(cdir))
                 if os.path.splitext(f)[1].lower() in VALID_EXT]
        N = len(files)
        if N == 0:
            continue

        n_train = floor(P_TRAIN * N)
        n_val   = floor(P_VAL * N)
        n_test  = N - n_train - n_val

        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

        with open(train_p, "a", encoding="utf-8") as ft:
            for f in train_files:
                ft.write(f"{hname}/{cls}/{f}\n")
        with open(val_p, "a", encoding="utf-8") as fv:
            for f in val_files:
                fv.write(f"{hname}/{cls}/{f}\n")
        with open(test_p, "a", encoding="utf-8") as fs:
            for f in test_files:
                fs.write(f"{hname}/{cls}/{f}\n")

        t_train += n_train
        t_val   += n_val
        t_test  += n_test

    total = t_train + t_val + t_test
    print(f"{hname}:")
    print(f"  Train: {t_train}  Val: {t_val}  Test: {t_test}  Total: {total}")
    print(f"  -> Guardado en {out_dir}\n")

def main():
    print(f"Generando splits 75/15/10 en {N_H} hospitales...\n")
    for i in range(1, N_H + 1):
        gen_split_for_hospital(f"Hospital_{i}")
    print("Listo.")

if __name__ == "__main__":
    main()
