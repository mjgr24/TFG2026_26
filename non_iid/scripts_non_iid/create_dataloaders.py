import os
import torch
from torch.utils.data import DataLoader
from medical_dataset import MedicalDataset, CLASSES

# Rutas base del escenario actual
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FED_ROOT = os.path.join(BASE_DIR, "data_non_iid")
SPLITS_ROOT = os.path.join(BASE_DIR, "splits_non_iid")

# Hiperparámetros
IMG_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 0         # si da problemas en Windows, pon 0
PIN_MEMORY = torch.cuda.is_available()


def make_loader(hospital_name, txt_name, phase):
    split_txt = os.path.join(SPLITS_ROOT, hospital_name, txt_name)
    ds = MedicalDataset(
        split_txt=split_txt,
        data_root=FED_ROOT,
        img_size=IMG_SIZE,
        phase=phase
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(phase == "train"),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    return ds, dl


def create_dataloaders(hospital_name):
    ds_tr, tr = make_loader(hospital_name, "train.txt", "train")
    ds_va, va = make_loader(hospital_name, "val.txt", "val")
    ds_te, te = make_loader(hospital_name, "test.txt", "test")
    return ds_tr, tr, ds_va, va, ds_te, te, CLASSES


def main():
    hospital_name = "Hospital_1"   # cambia aquí para probar otro hospital
    print(f"Creando DataLoaders para {hospital_name}...")

    ds_tr, tr, ds_va, va, ds_te, te, class_names = create_dataloaders(hospital_name)

    print(f"Clases: {class_names}")
    print(f"Train: {len(ds_tr)}  Val: {len(ds_va)}  Test: {len(ds_te)}")

    xb, yb = next(iter(tr))
    print(
        f"Batch train -> x {tuple(xb.shape)}  y {tuple(yb.shape)}  "
        f"ejemplo: {yb[0].item()} ({class_names[yb[0].item()]})"
    )


if __name__ == "__main__":
    main()
