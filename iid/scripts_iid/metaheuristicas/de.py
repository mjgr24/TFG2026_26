# diferential evolution

import random


def _sample_simplex(n):
    """Genera un vector aleatorio en el simplex (>=0, sum=1)."""
    xs = [random.random() for _ in range(n)]
    s = sum(xs)
    return [x / s for x in xs]


def _project_to_simplex(vec):
    """Proyecta un vector al simplex: clamp negativos a 0 y renormaliza."""
    clamped = [max(x, 0.0) for x in vec]
    s = sum(clamped)
    if s < 1e-12:
        n = len(vec)
        return [1.0 / n] * n
    return [x / s for x in clamped]


def solve(fitness_fn, dim, n_iter=200, pop_size=30, F=0.8, CR=0.9, **kwargs):
    """
    Differential Evolution (DE/rand/1/bin) en el simplex.

    Parámetros:
    - fitness_fn: función a maximizar.
    - dim:        dimensión del vector alpha (número de hospitales).
    - n_iter:     número de generaciones.
    - pop_size:   tamaño de la población (mínimo 4).
    - F:          factor de escala de mutación diferencial.
    - CR:         probabilidad de cruce (crossover rate).

    Retorna:
    - best_alphas: mejor vector alpha encontrado.
    - best_score:  valor de fitness del mejor alpha.
    """
    pop_size = max(pop_size, 4)

    # Inicializar población en el simplex
    population = [_sample_simplex(dim) for _ in range(pop_size)]
    scores = [fitness_fn(ind) for ind in population]

    # Mejor global
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_alphas = population[best_idx][:]
    best_score = scores[best_idx]

    for _ in range(n_iter):
        for i in range(pop_size):
            # Seleccionar tres individuos distintos de i
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = random.sample(candidates, 3)

            # Mutación diferencial: donor = r1 + F * (r2 - r3)
            donor = [
                population[r1][d] + F * (population[r2][d] - population[r3][d])
                for d in range(dim)
            ]

            # Cruce binomial
            j_rand = random.randint(0, dim - 1)
            trial = [
                donor[d] if (random.random() < CR or d == j_rand) else population[i][d]
                for d in range(dim)
            ]

            # Proyectar al simplex
            trial = _project_to_simplex(trial)

            # Selección: el trial reemplaza al padre solo si es mejor
            trial_score = fitness_fn(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score

                # Actualizar mejor global
                if trial_score > best_score:
                    best_score = trial_score
                    best_alphas = trial[:]

    return best_alphas, best_score