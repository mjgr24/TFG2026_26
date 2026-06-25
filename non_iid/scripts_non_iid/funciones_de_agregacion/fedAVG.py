import copy


def fedavg_aggregate(local_states, global_state, sample_counts, total_samples_all=None):
    """
    FedAvg expresado explícitamente en términos de updates y con
    ponderación obligatoria por tamaño local, como en el paper.

    Cada cliente devuelve su modelo local final w_k^(t+1).
    El servidor calcula:
        delta_k = w_k^(t+1) - w_t
    y agrega:
        delta = sum_k (n_k / n) * delta_k
        w_(t+1) = w_t + delta
    """
    if not local_states:
        raise ValueError("local_states está vacío.")

    if sample_counts is None:
        raise ValueError("FedAvg requiere sample_counts según la formulación del paper.")

    if len(local_states) != len(sample_counts):
        raise ValueError(
            "La longitud de local_states y sample_counts debe coincidir."
        )

    total = total_samples_all if total_samples_all is not None else sum(sample_counts)
    if total <= 0:
        raise ValueError("La suma de sample_counts debe ser mayor que 0.")

    K = len(local_states)
    weights = [n / total for n in sample_counts]

    new_global_state = copy.deepcopy(global_state)

    for key in new_global_state.keys():
        delta_avg = sum(
            weights[i] * (local_states[i][key] - global_state[key])
            for i in range(K)
        )
        new_global_state[key] = global_state[key] + delta_avg

    return new_global_state