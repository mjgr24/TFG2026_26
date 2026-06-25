import os
import shutil

def limpiar_federated():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    federated_dir = os.path.join(base_dir, "data", "federated")

    if not os.path.isdir(federated_dir):
        print(f"[ERROR] No existe la carpeta: {federated_dir}")
        return

    subcarpetas = [os.path.join(federated_dir, d) for d in os.listdir(federated_dir) if os.path.isdir(os.path.join(federated_dir, d))]

    if not subcarpetas:
        print(f"[AVISO] No hay subcarpetas dentro de {federated_dir}")
        return

    total_borradas = 0
    for carpeta in subcarpetas:
        shutil.rmtree(carpeta)
        print(f"Borrada: {carpeta}")
        total_borradas += 1

    print(f"\nListo. Carpetas eliminadas: {total_borradas}")

if __name__ == "__main__":
    limpiar_federated()
