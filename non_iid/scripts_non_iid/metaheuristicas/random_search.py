import random


def _sample_simplex(n):
    """Genera un vector alpha aleatorio en el simplex (alpha_k >= 0, sum = 1)."""
    xs = [random.random() for _ in range(n)]
    s = sum(xs)
    return [x / s for x in xs]


def solve(fitness_fn, dim, n_iter=200, **kwargs):
    """
    Búsqueda aleatoria en el simplex.
    Baseline: genera n_iter candidatos aleatorios y se queda con el mejor.

    Parámetros:
    - fitness_fn: función a maximizar, recibe lista de alphas y devuelve score.
    - dim:        dimensión del vector alpha (número de hospitales).
    - n_iter:     número de candidatos a evaluar.

    Retorna:
    - best_alphas: mejor vector alpha encontrado.
    - best_score:  valor de fitness del mejor alpha.
    """
    # Empezar con la solución uniforme como referencia
    best_alphas = [1.0 / dim] * dim
    best_score = fitness_fn(best_alphas)

    for _ in range(n_iter):
        candidate = _sample_simplex(dim)
        score = fitness_fn(candidate)
        if score > best_score:
            best_score = score
            best_alphas = candidate

    return best_alphas, best_score