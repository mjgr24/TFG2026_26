"""
FedAG: Federated Learning Based on Data Importance Weighted Aggregation
Referencia: IEEE ICCC 2023, DOI 10.1109/ICCC57788.2023.10233464

Implementación fiel al paper (Algorithm 1, Eqs. 1-9):
  El cliente calcula y envía tres indicadores al servidor:
    - µk (Eq. 1): volumen de datos relativo
    - vk (Eq. 2): sesgo de distribución de clases (norma L2)
    - θk (Eq. 3): calidad de datos via Dirichlet — αk = max del modo
  El servidor construye el modelo de importancia con KL divergence (Eq. 5-8)
  y agrega los modelos locales ponderados por Ck (Eq. 9)
"""

import os
import copy
import math
import torch
from torch.distributions.dirichlet import Dirichlet


# ── Utilidades ──────────────────────────────────────────────────────────────

def contar_clases_por_hospital(hospitals, data_root, classes):
    """
    Cuenta imágenes por clase para cada hospital leyendo la estructura
    de carpetas. No carga ninguna imagen.

    Returns:
        dict: {hospital_name: {clase: n_imagenes}}
    """
    conteos = {}
    for hospital in hospitals:
        conteos[hospital] = {}
        for cls in classes:
            carpeta = os.path.join(data_root, hospital, cls)
            if os.path.isdir(carpeta):
                conteos[hospital][cls] = len(os.listdir(carpeta))
            else:
                conteos[hospital][cls] = 0
    return conteos


def _normalize(values, eps=1e-12):
    """Normaliza una lista de valores para que sumen 1."""
    total = sum(values) + eps
    return [v / total for v in values]


def _kl_divergence(p, q, eps=1e-12):
    """
    KL divergence KL(P, Q) = Σ P_i · log(P_i / Q_i)  (Eq. 4)
    p y q son listas que suman 1.
    """
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > eps:
            kl += pi * math.log((pi + eps) / (qi + eps))
    return kl


# ── Función que ejecuta el cliente ──────────────────────────────────────────

def train_one_epoch_ag(model, dataloader, criterion, optimizer, device,
                       hospital_name, class_counts, hospitals, classes,
                       beta_prev=None, lr_beta=0.01, eps=1e-12):
    """
    Entrena una época y calcula los tres indicadores de FedAG.

    Sigue fielmente el Algorithm 1 del paper:
      Líneas 12-14: Entrenamiento local + actualización de β por batch.
      Tras el entrenamiento, calcula los tres indicadores:

    µk (Eq. 1): |Dk| / |D| → volumen relativo de datos
    vk (Eq. 2): ||p^(k) − p|| → distancia L2 entre distribución local y global
    θk (Eq. 3): Normalize(αk) donde αk = max del modo Dirichlet
                (paper: "αk represents the value with the highest probability
                 for client k")

    Args:
        hospital_name : nombre del hospital (cliente k)
        class_counts  : dict {hospital: {clase: count}} para todos los hospitales
        hospitals     : lista de todos los hospitales
        classes       : lista de nombres de clases
        beta_prev     : parámetros Dirichlet β del round anterior (o None)

    Returns:
        train_loss, train_acc, indicators (dict con µk, vk, θk, beta)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    n_batches = len(dataloader)

    # ── Actualizar parámetros Dirichlet β para calidad de datos ──────────
    # β es un vector de dimensión C (número de clases) para este cliente.
    # Se ajusta a la distribución empírica de etiquetas del cliente usando
    # la log-verosimilitud Dirichlet con implicit reparameterization (Eq. 3).
    C = len(classes)
    if beta_prev is None:
            beta = torch.ones(C, device=device, requires_grad=True)
    else:
        beta = beta_prev.clone().detach().to(device).requires_grad_(True)

    beta_optimizer = torch.optim.Adam([beta], lr=lr_beta)

    # ── Algorithm 1, líneas 4-8: entrenamiento local + actualización β ──
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Entrenamiento local del modelo (línea 13: ModelUpdate)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Actualizar β con implicit reparameterization (líneas 5-6)
        # Distribución empírica de clases en el batch
        label_counts = torch.bincount(labels, minlength=C).float()

        # Suavizar y proyectar al simplex: sumar eps y renormalizar para
        # que sume exactamente 1 (requisito de Dirichlet.log_prob)
        label_dist = label_counts + eps
        label_dist = label_dist / label_dist.sum()

        beta_optimizer.zero_grad()
        beta_clamped = torch.clamp(beta, min=eps)
        dirichlet = Dirichlet(beta_clamped)
        beta_loss = -dirichlet.log_prob(label_dist)
        beta_loss.backward()
        beta_optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / n_batches
    train_acc = correct / total

    # ── Calcular µk: volumen de datos (Eq. 1) ───────────────────────────
    # µk = |Dk| / |D|  (se normaliza en el servidor con Normalize)
    n_k = sum(class_counts[hospital_name].values())
    n_total = sum(
        sum(class_counts[h].values()) for h in hospitals
    )
    mu_k = n_k / (n_total + eps)

    # ── Calcular vk: sesgo de distribución (Eq. 2) ──────────────────────
    # vk = ||p^(k) − p||  (norma L2, notación ||·|| del paper)
    # Se normaliza en el servidor con Normalize.
    p_local = [class_counts[hospital_name].get(cls, 0) / (n_k + eps)
               for cls in classes]

    n_per_class_global = {
        cls: sum(class_counts[h].get(cls, 0) for h in hospitals)
        for cls in classes
    }
    p_global = [n_per_class_global[cls] / (n_total + eps) for cls in classes]

    # Norma L2 (euclidiana) de la diferencia
    v_k = math.sqrt(
        sum((p_local[i] - p_global[i]) ** 2 for i in range(len(classes)))
    )

    # ── Calcular θk: calidad de datos (Eq. 3) ───────────────────────────
    # El modo de Dir(β) es: mode_c = (βc − 1) / (Σ βc' − C)
    # El paper dice: "αk represents the value with the highest probability
    # for client k" → αk = max(mode)
    # θk = Normalize(αk) se aplica en el servidor.
    beta_vals = torch.clamp(beta, min=eps).detach()
    sum_beta = beta_vals.sum().item()
    denom = sum_beta - C

    if denom > eps:
        # Modo de la distribución Dirichlet
        mode = (beta_vals - 1.0) / denom
        # αk = valor con mayor probabilidad (el máximo del modo)
        alpha_k = mode.max().item()
        theta_k = max(alpha_k, 0.0)
    else:
        # β no ha convergido lo suficiente → asignar valor neutro
        theta_k = 1.0 / C

    indicators = {
        "mu": mu_k,
        "v": v_k,
        "theta": theta_k,
        "beta": beta_vals.clone()
    }

    return train_loss, train_acc, indicators


# ── Agregación en el servidor ────────────────────────────────────────────────

def fedAG_aggregate(local_states, selected_names, all_indicators, all_hospitals, eps=1e-12):
    """
    Agrega los modelos locales usando el modelo de importancia de datos (Eq. 8-9).
    Calcula KL y ρ sobre TODOS los hospitales, pero agrega solo los seleccionados.

    Args:
        local_states     : lista de state_dict (solo hospitales seleccionados)
        selected_names   : lista de nombres de hospitales seleccionados en esta ronda
        all_indicators   : dict {hospital: {"mu", "v", "theta"}} para TODOS los hospitales
        all_hospitals    : lista de TODOS los hospitales del sistema
        eps              : estabilidad numérica

    Returns:
        new_global_state, alphas, server_payload
    """
    if not local_states:
        raise ValueError("local_states está vacío.")

    K_total = len(all_hospitals)

    mu_all    = [all_indicators[h]["mu"]    for h in all_hospitals]
    v_all     = [all_indicators[h]["v"]     for h in all_hospitals]
    theta_all = [all_indicators[h]["theta"] for h in all_hospitals]

    mu_norm    = _normalize(mu_all)
    v_norm     = _normalize(v_all)
    theta_norm = _normalize(theta_all)

    T = [1.0 / K_total] * K_total

    kl_mu    = _kl_divergence(mu_norm,    T)
    kl_v     = _kl_divergence(v_norm,     T)
    kl_theta = _kl_divergence(theta_norm, T)

    kl_total = kl_mu + kl_v + kl_theta + eps

    rho_mu    = kl_mu    / kl_total
    rho_v     = kl_v     / kl_total
    rho_theta = kl_theta / kl_total

    C_all = {}
    for i, h in enumerate(all_hospitals):
        C_all[h] = rho_mu * mu_norm[i] + rho_v * v_norm[i] + rho_theta * theta_norm[i]

    C_selected = [C_all[h] for h in selected_names]
    C_sum = sum(C_selected) + eps
    alphas = [c / C_sum for c in C_selected]

    new_global_state = copy.deepcopy(local_states[0])
    for key in new_global_state.keys():
        new_global_state[key] = sum(
            alphas[k] * local_states[k][key].detach().float()
            for k in range(len(local_states))
        ).to(local_states[0][key].dtype)

    server_payload = {
        "rho_mu": rho_mu,
        "rho_v": rho_v,
        "rho_theta": rho_theta,
        "C_all": C_all,
        "C_selected": {h: C_all[h] for h in selected_names},
    }

    return new_global_state, alphas, server_payload