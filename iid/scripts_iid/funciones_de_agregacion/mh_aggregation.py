import copy
import random
import torch

# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN: cambia aquí la metaheurística que quieres usar
# Opciones: "random_search", "pso", "ga", "de"
# ══════════════════════════════════════════════════════════════════════
SOLVER = "cro"

# ── Imports de metaheurísticas ───────────────────────────────────────
from metaheuristicas.random_search import solve as solve_random_search
from metaheuristicas.pso import solve as solve_pso
from metaheuristicas.ga import solve as solve_ga
from metaheuristicas.de import solve as solve_de
from metaheuristicas.cro import solve as solve_cro

SOLVERS = {
    "random_search": solve_random_search,
    "pso":           solve_pso,
    "ga":            solve_ga,
    "de":            solve_de,
    "cro":           solve_cro,
}


# ── Utilidades internas ──────────────────────────────────────────────

def _flatten_state_dict(state_dict):
    flats = []
    for _, tensor in state_dict.items():
        flats.append(tensor.detach().float().reshape(-1))
    return torch.cat(flats)


def _compute_delta_state(local_state, global_state):
    delta = {}
    for key in global_state.keys():
        delta[key] = local_state[key].detach().clone() - global_state[key].detach().clone()
    return delta


def _weighted_sum_states(states, alphas):
    result = {}
    keys = states[0].keys()
    for key in keys:
        weighted = torch.zeros_like(states[0][key], dtype=states[0][key].dtype)
        for alpha, state in zip(alphas, states):
            weighted += alpha * state[key]
        result[key] = weighted
    return result


def _l2_norm_state(state_dict):
    vec = _flatten_state_dict(state_dict)
    return torch.norm(vec, p=2).item()


def _cosine_between_states(state_a, state_b, eps=1e-12):
    vec_a = _flatten_state_dict(state_a)
    vec_b = _flatten_state_dict(state_b)
    norm_a = torch.norm(vec_a, p=2)
    norm_b = torch.norm(vec_b, p=2)
    if norm_a.item() < eps or norm_b.item() < eps:
        return 0.0
    cos_val = torch.dot(vec_a, vec_b) / (norm_a * norm_b + eps)
    return cos_val.item()


# ── Función objetivo ─────────────────────────────────────────────────

def objective_function(alphas, delta_states, lambda_reg=0.01, gamma_reg=0.01):
    delta_alpha = _weighted_sum_states(delta_states, alphas)

    k = len(delta_states)
    uniform_alphas = [1.0 / k] * k
    delta_bar = _weighted_sum_states(delta_states, uniform_alphas)

    term_alignment = _cosine_between_states(delta_alpha, delta_bar)
    term_stability = _l2_norm_state(delta_alpha)

    term_robustness = 0.0
    for alpha, delta_k in zip(alphas, delta_states):
        term_robustness += alpha * _l2_norm_state(delta_k)

    score = term_alignment - lambda_reg * term_stability - gamma_reg * term_robustness
    return score


# ── Agregación principal ─────────────────────────────────────────────

def mh_aggregate(local_states, global_state,
                 lambda_reg=0.01, gamma_reg=0.01,
                 n_iter=200, global_lr=1.0,
                 **solver_kwargs):
    """
    Agregación basada en la función objetivo J(α).
    Usa la metaheurística definida en SOLVER.
    """
    if not local_states:
        raise ValueError("local_states está vacío.")

    # 1. Calcular deltas
    delta_states = [_compute_delta_state(ls, global_state) for ls in local_states]
    dim = len(delta_states)

    # 2. Construir fitness_fn
    def fitness_fn(alphas):
        return objective_function(alphas, delta_states,
                                  lambda_reg=lambda_reg, gamma_reg=gamma_reg)

    # 3. Optimizar α con la metaheurística configurada
    solver_fn = SOLVERS[SOLVER]
    best_alphas, best_score = solver_fn(fitness_fn, dim, n_iter=n_iter, **solver_kwargs)

    # 4. Agregar: w_new = w_global + lr * Δ(α)
    delta_agg = _weighted_sum_states(delta_states, best_alphas)
    new_global_state = copy.deepcopy(global_state)
    for key in new_global_state.keys():
        new_global_state[key] = new_global_state[key] + global_lr * delta_agg[key]

    return new_global_state, best_alphas, best_score