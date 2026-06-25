# Genetic Algorithms

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


def _tournament_selection(population, scores, k=3):
    """Selección por torneo: elige k individuos al azar y devuelve el mejor."""
    indices = random.sample(range(len(population)), k)
    best = max(indices, key=lambda i: scores[i])
    return population[best][:]


def _blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    BLX-α crossover para codificación real.
    Para cada gen, el hijo se genera en el rango
    [min - alpha*d, max + alpha*d] donde d = |p1 - p2|.
    """
    child = []
    for g1, g2 in zip(parent1, parent2):
        lo = min(g1, g2)
        hi = max(g1, g2)
        d = hi - lo
        child.append(random.uniform(lo - alpha * d, hi + alpha * d))
    return _project_to_simplex(child)


def _gaussian_mutation(individual, sigma=0.1, prob=0.2):
    """Mutación gaussiana: perturba cada gen con probabilidad prob."""
    mutated = individual[:]
    for i in range(len(mutated)):
        if random.random() < prob:
            mutated[i] += random.gauss(0, sigma)
    return _project_to_simplex(mutated)


def solve(fitness_fn, dim, n_iter=200, pop_size=40, crossover_rate=0.8,
          mutation_rate=0.2, mutation_sigma=0.1, tournament_k=3, elitism=2, **kwargs):
    """
    Algoritmo Genético con codificación real en el simplex.

    Parámetros:
    - fitness_fn:      función a maximizar.
    - dim:             dimensión del vector alpha (número de hospitales).
    - n_iter:          número de generaciones.
    - pop_size:        tamaño de la población.
    - crossover_rate:  probabilidad de cruce.
    - mutation_rate:   probabilidad de mutar cada gen.
    - mutation_sigma:  desviación estándar de la mutación gaussiana.
    - tournament_k:    tamaño del torneo de selección.
    - elitism:         número de mejores individuos que pasan directamente.

    Retorna:
    - best_alphas: mejor vector alpha encontrado.
    - best_score:  valor de fitness del mejor alpha.
    """
    # Inicializar población en el simplex
    population = [_sample_simplex(dim) for _ in range(pop_size)]
    scores = [fitness_fn(ind) for ind in population]

    # Mejor global
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_alphas = population[best_idx][:]
    best_score = scores[best_idx]

    for _ in range(n_iter):
        # Ordenar por fitness (mayor = mejor)
        ranked = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)

        # Elitismo: los mejores pasan directamente
        new_population = [population[ranked[i]][:] for i in range(elitism)]
        new_scores = [scores[ranked[i]] for i in range(elitism)]

        # Generar el resto de la nueva población
        while len(new_population) < pop_size:
            parent1 = _tournament_selection(population, scores, k=tournament_k)
            parent2 = _tournament_selection(population, scores, k=tournament_k)

            # Cruce
            if random.random() < crossover_rate:
                child = _blx_alpha_crossover(parent1, parent2)
            else:
                child = parent1[:]

            # Mutación
            child = _gaussian_mutation(child, sigma=mutation_sigma, prob=mutation_rate)

            score = fitness_fn(child)
            new_population.append(child)
            new_scores.append(score)

            # Actualizar mejor global
            if score > best_score:
                best_score = score
                best_alphas = child[:]

        population = new_population
        scores = new_scores

    return best_alphas, best_score