# ==============================================================
# medical_dataset.py — Dataset desde splits .txt con transforms
# - Lee rutas de train/val/test (.txt)
# - Deduce la clase por la carpeta (Hospital_i/Clase/archivo)
# - Aplica: resize -> (opcional) augmentation -> ToTensor -> Normalize
# - Devuelve tensores [C,H,W] y etiquetas numéricas consistentes
# ==============================================================

import os
from typing import List, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Mapea clases a índices (fijo para todos los hospitales)
CLASSES = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class MedicalDataset(Dataset):
    def __init__(
        self,
        split_txt: str,                  # p.ej. "splits/Hospital_1/train.txt"
        data_root: str = "../data_non_iid",
        img_size: int = 64,
        phase: str = "train",            # "train" | "val" | "test"
        to_grayscale: bool = True
    ):
        """
        Carga rutas desde split_txt y prepara transforms según el 'phase'.
        """
        self.data_root = data_root
        self.phase = phase.lower()
        assert self.phase in {"train", "val", "test"}

        # Leer rutas
        with open(split_txt, "r", encoding="utf-8") as f:
            self.paths = [ln.strip() for ln in f if ln.strip()]

        # Transforms
        t_list = []
        if to_grayscale:
            t_list.append(transforms.Grayscale(num_output_channels=1))  # 1 canal
        t_list.append(transforms.Resize((img_size, img_size)))

        if self.phase == "train":
            # augmentation ligero y seguro para med-imaging 2D
            t_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # ±10°
            ])

        t_list.extend([
            transforms.ToTensor(),                     # [0,1]
            transforms.Normalize(mean=[0.5] if to_grayscale else [0.5,0.5,0.5],
                                 std=[0.5] if to_grayscale else [0.5,0.5,0.5])  # [−1,1]
        ])
        self.transform = transforms.Compose(t_list)

        # Cache: pre-cargar todas las imágenes en RAM
        self.cache = []
        for rel_path in self.paths:
            abs_path = os.path.join(self.data_root, rel_path)
            cls_name = rel_path.split("/")[1]
            label = CLASS_TO_IDX[cls_name]
            with Image.open(abs_path) as im:
                im = im.convert("RGB")
                im = im.copy()  # necesario para cerrar el file handle
            self.cache.append((im, label))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        im, label = self.cache[idx]
        x = self.transform(im)
        return x, label


# Pequeña prueba local (ejecuta: python scripts/medical_dataset.py)
if __name__ == "__main__":
    # Ajusta rutas relativas (desde /scripts/ subimos una carpeta)
    split_txt = os.path.join("..", "splits_non_iid", "Hospital_1", "train.txt")
    data_root = os.path.join("..", "data_non_iid")

    ds = MedicalDataset(split_txt=split_txt, data_root=data_root, img_size=64, phase="train")
    print("Clases:", CLASSES)
    print("Total muestras:", len(ds))

    # Inspecciona 3 ejemplos
    for i in range(3):
        x, y = ds[i]
        print(f"ejemplo {i}: tensor={tuple(x.shape)} etiqueta_idx={y} clase={CLASSES[y]}")