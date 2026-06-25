"""
FedEWA: Federated Learning with Elastic Weighted Averaging
Referencia: IEEE IJCNN 2022, DOI 10.1109/IJCNN55064.2022.9892851

Implementación fiel al paper (Algorithm 1, Eqs. 11-16):
  - train_one_epoch_ewa: calcula BPI (Eq. 11) y OPI (Eq. 13) durante el
    entrenamiento local, y devuelve el vector de importancia híbrido
    HPI (Eq. 14) usando β = val_acc.
  - fedewa_aggregate: agregación parámetro a parámetro (Eq. 15-16).

Nota sobre BPI:
  La Eq. 11 requiere gradientes POR MUESTRA: (1/|P_k|) Σ_{x} (∂F(x)/∂ω_j)².
  Esto NO es lo mismo que (gradiente_medio_del_batch)².
  Para ser fieles al paper, se itera muestra a muestra en el pase de BPI.
"""

import copy
import torch


def _compute_val_accuracy(model, val_loader, device):
    """Calcula la accuracy sobre el val_loader sin modificar el modelo."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_epoch_ewa(model, dataloader, criterion, optimizer, device,
                        val_loader=None):
    """
    Entrena una época y calcula los coeficientes de importancia HPI.

    Sigue fielmente el Algorithm 1 del paper:
      Líneas 23-27: Entrenamiento local + acumulación de OPI por batch.
      Línea  29:    Cálculo de BPI (pase extra, muestra a muestra).
      Línea  30:    HPI = β · BPI + (1 − β) · OPI

    BPI (Eq. 11): proxy de Fisher diagonal.
        BPI_j = (1/|P_k|) Σ_{x∈P_k} (∂F(x;ω)/∂ω_j)²
        Requiere gradientes POR MUESTRA → se itera sample-by-sample.

    OPI (Eq. 13): contribución de cada parámetro a la reducción de pérdida.
        OPI_j = (1/|P_k|) Σ_{x∈P_k} g_j(x) · δ_j(x)

    HPI (Eq. 14): β · BPI + (1 − β) · OPI
        β = accuracy de validación local (paper: "the test accuracy rate
        of the local model in each round learning").

    Args:
        model:      modelo local (copia del global)
        dataloader: DataLoader de entrenamiento del cliente
        criterion:  función de pérdida (debe usar reduction='mean')
        optimizer:  optimizador
        device:     dispositivo
        val_loader: DataLoader de validación del cliente (para calcular β)

    Returns:
        train_loss, train_acc, omega (dict con importancia HPI por parámetro)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Inicializar acumuladores de OPI a cero (Eq. 13)
    opi = {name: torch.zeros_like(param.data, dtype=torch.float32)
           for name, param in model.named_parameters() if param.requires_grad}

    n_batches = len(dataloader)

    # ── Algorithm 1, líneas 23-27: entrenamiento local + OPI ────────────
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Guardar parámetros antes del step
        params_before = {name: param.data.clone()
                         for name, param in model.named_parameters()
                         if param.requires_grad}

        # Paso de entrenamiento estándar (línea 25)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Acumular OPI (línea 26, Eq. 13)
        # param.grad = (1/B) Σ_{x∈b} g_j(x)  (reduction='mean')
        # delta = θ_after − θ_before
        # Dado que δ es el mismo para todas las x del batch:
        #   Σ_{x∈b} g_j(x)·δ_j = (Σ g_j) · δ = param.grad · B · δ
        # Acumulamos esto y al final dividimos por |P_k|.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    delta = param.data - params_before[name]
                    opi[name] += param.grad.float() * delta.float() * batch_size

        # Métricas de entrenamiento
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / n_batches
    train_acc = correct / total

    # Normalizar OPI: dividir por |P_k| (Eq. 13)
    for name in opi:
        opi[name] /= total

    # ── Algorithm 1, línea 29: Cálculo de BPI (Eq. 11) ─────────────────
    # BPI_j = (1/|P_k|) Σ_{x∈P_k} (∂F(x;ω)/∂ω_j)²
    #
    # IMPORTANTE: Eq. 11 requiere el gradiente POR MUESTRA individual,
    # NO el gradiente medio del batch. (E[g²] ≠ E[g]²)
    # Por eso iteramos muestra a muestra.
    bpi = {name: torch.zeros_like(param.data, dtype=torch.float32)
           for name, param in model.named_parameters() if param.requires_grad}
    n_samples_bpi = 0

    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        # Iterar muestra a muestra dentro del batch
        for i in range(images.size(0)):
            single_img = images[i:i+1]   # [1, C, H, W]
            single_lbl = labels[i:i+1]   # [1]

            model.zero_grad()
            with torch.enable_grad():
                output = model(single_img)
                loss_i = criterion(output, single_lbl)
                loss_i.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Con batch_size=1, param.grad = gradiente individual exacto
                        bpi[name] += param.grad.float() ** 2

            n_samples_bpi += 1

    with torch.no_grad():
        for name in bpi:
            bpi[name] /= n_samples_bpi

    # ── Algorithm 1, línea 30: HPI (Eq. 14) ────────────────────────────
    # Ω_j^k = β · BPI_j + (1 − β) · OPI_j
    # β = accuracy de validación local del modelo entrenado
    if val_loader is not None:
        beta = _compute_val_accuracy(model, val_loader, device)
    else:
        beta = train_acc

    omega = {}
    with torch.no_grad():
        for name in bpi:
            omega[name] = beta * bpi[name] + (1 - beta) * opi[name]

    return train_loss, train_acc, omega


def fedewa_aggregate(local_states, global_state, local_omegas, eps=1e-12):
    """
    Agregación elástica parámetro a parámetro (Eq. 15-16 del paper).

    Eq. 15 — Normalización de importancias por posición:
        Ω̄_{t+1,j}^k = Ω_{t+1,j}^k / Σ_{k'} Ω_{t+1,j}^{k'}

    Eq. 16 — Agregación ponderada:
        ω_{t+1,j}^g = Σ_k Ω̄_{t+1,j}^k · ω_{t+1,j}^k

    Args:
        local_states  : lista de state_dict (uno por cliente)
        global_state  : state_dict del modelo global ANTES de la ronda
        local_omegas  : lista de dicts con importancia HPI por parámetro
        eps           : constante de estabilidad numérica

    Returns:
        new_global_state : state_dict del nuevo modelo global
    """
    if not local_states:
        raise ValueError("local_states está vacío.")

    if local_omegas is None:
        raise ValueError(
            "FedEWA requiere local_omegas según la formulación del paper."
        )

    if len(local_states) != len(local_omegas):
        raise ValueError(
            "La longitud de local_states y local_omegas debe coincidir."
        )

    K = len(local_states)
    new_global_state = copy.deepcopy(global_state)

    for key in global_state.keys():
        stacked_w = torch.stack(
            [local_states[k][key].detach().float() for k in range(K)], dim=0
        )  # [K, *param_shape]

        # Usar HPI calculado en el cliente (fiel al paper)
        # Para parámetros sin omega (p. ej. buffers), usar peso uniforme
        importances = torch.stack(
            [local_omegas[k][key].float()
             if key in local_omegas[k]
             else torch.ones_like(local_states[k][key], dtype=torch.float32)
             for k in range(K)],
            dim=0
        )  # [K, *param_shape]

        # Eq. 15: Normalización por posición
        total = importances.sum(dim=0, keepdim=True) + eps
        weights = importances / total  # [K, *param_shape], suma 1 por posición

        # Eq. 16: Agregación ponderada
        new_param = (weights * stacked_w).sum(dim=0)
        new_global_state[key] = new_param.to(global_state[key].dtype)

    return new_global_state