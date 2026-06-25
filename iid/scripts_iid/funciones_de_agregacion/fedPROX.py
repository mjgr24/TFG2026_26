"""
FedProx: Federated Optimization in Heterogeneous Networks
Referencia: Li et al., MLSys 2020

Idea central:
  Añade un término de regularización proximal a la loss local que penaliza
  cuánto se aleja el modelo local del modelo global:

      loss_total = loss_CE + (µ/2) * ||w_local - w_global||²

  Esto limita el client drift en escenarios Non-IID sin cambiar nada
  en el servidor. La agregación es idéntica a FedAvg.

Interfaz:
    train_one_epoch_fedprox(model, dataloader, criterion, optimizer,
                            device, global_state, mu) -> train_loss, train_acc
    fedprox_aggregate(local_states, sample_counts) -> new_global_state
"""

import copy
import torch


def train_one_epoch_fedprox(model, dataloader, criterion, optimizer,
                             device, global_state, mu=0.01):
    """
    Entrena una época con el término proximal de FedProx.

    Args:
        model        : modelo local (copia del modelo global)
        dataloader   : DataLoader del hospital
        criterion    : función de pérdida (CrossEntropyLoss)
        optimizer    : optimizador Adam
        device       : cpu o cuda
        global_state : state_dict del modelo global ANTES de la ronda
        mu           : hiperparámetro proximal (default 0.01)

    Returns:
        train_loss, train_acc
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_ce = criterion(outputs, labels)

        # Término proximal: (µ/2) * ||w_local - w_global||²
        proximal_term = 0.0
        for name, param in model.named_parameters():
            if name in global_state:
                global_param = global_state[name].detach().to(device)
                proximal_term += ((param - global_param) ** 2).sum()
        proximal_term = (mu / 2) * proximal_term

        loss = loss_ce + proximal_term
        loss.backward()
        optimizer.step()

        running_loss += loss_ce.item()  # logueamos solo la CE loss
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def fedprox_aggregate(local_states, eps=1e-12):
    """
    Agregación FedProx = media simple (1/K) de los modelos locales.
    Fiel al Algorithm 2 del paper (Li et al., MLSys 2020):
        w^{t+1} = (1/K) * Σ_{k∈S_t} w^{k}_{t+1}

    Nota: los experimentos del paper (Sección 5.1) usan media ponderada
    por muestras, pero el algoritmo define media simple. Aquí seguimos
    el algoritmo.

    Args:
        local_states : lista de state_dict (uno por hospital seleccionado)
        eps          : estabilidad numérica

    Returns:
        new_global_state : state_dict del nuevo modelo global
    """
    if not local_states:
        raise ValueError("local_states está vacío.")

    K = len(local_states)

    new_global_state = copy.deepcopy(local_states[0])
    for key in new_global_state.keys():
        new_global_state[key] = sum(
            local_states[k][key].detach().float()
            for k in range(K)
        ).to(local_states[0][key].dtype) / K

    return new_global_state