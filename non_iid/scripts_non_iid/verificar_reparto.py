"""
asegurarte de que cada hospital tiene imágenes, y que las clases dominantes están donde deberían. para asegurarte de que los datos realmente 
están bien distribuidos antes de entrenar nada.
"""

import os

root = "../data_non_iid"

for h in sorted(os.listdir(root)):
    hpath = os.path.join(root, h)
    if not os.path.isdir(hpath):
        continue

    print(f"\n{h}:")
    conteos = {}
    total = 0

    # Contar imágenes por clase
    for c in sorted(os.listdir(hpath)):
        cpath = os.path.join(hpath, c)
        if not os.path.isdir(cpath):
            continue
        n = len(os.listdir(cpath))
        conteos[c] = n
        total += n

    # Mostrar resultados con porcentajes
    for c, n in conteos.items():
        pct = (n / total * 100) if total > 0 else 0
        print(f"  {c}: {n} imágenes ({pct:.2f}%)")

    print(f"  Total: {total}")
