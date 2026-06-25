# Coral Reef Optimization

import random
import math


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


def _gaussian_larva(parent, sigma=0.1):
    """Genera una larva perturbando al padre con ruido gaussiano."""
    child = [g + random.gauss(0, sigma) for g in parent]
    return _project_to_simplex(child)


def _crossover_larva(parent1, parent2):
    """Genera una larva por cruce uniforme de dos padres."""
    child = [
        p1 if random.random() < 0.5 else p2
        for p1, p2 in zip(parent1, parent2)
    ]
    return _project_to_simplex(child)


def solve(fitness_fn, dim, n_iter=200, reef_size=40, rho0=0.6,
          Fb=0.8, Fa=0.1, Fd=0.1, sigma=0.1, **kwargs):
    """
    Coral Reef Optimization (CRO) en el simplex.

    Parámetros:
    - fitness_fn: función a maximizar.
    - dim:        dimensión del vector alpha (número de hospitales).
    - n_iter:     número de iteraciones.
    - reef_size:  tamaño del arrecife (número de posiciones).
    - rho0:       proporción inicial de posiciones ocupadas (0-1).
    - Fb:         fracción de corales que se reproducen por broadcast spawning.
    - Fa:         fracción de corales que se reproducen asexualmente (budding).
    - Fd:         fracción de los peores corales que se eliminan (depredación).
    - sigma:      desviación estándar para mutación gaussiana.

    Retorna:
    - best_alphas: mejor vector alpha encontrado.
    - best_score:  valor de fitness del mejor alpha.
    """
    # ── Inicializar arrecife ─────────────────────────────────────
    # Cada posición es None (vacía) o (coral, score)
    reef = [None] * reef_size
    n_initial = max(2, int(reef_size * rho0))

    for i in range(n_initial):
        coral = _sample_simplex(dim)
        score = fitness_fn(coral)
        reef[i] = (coral, score)

    # Mejor global
    occupied = [(entry[0], entry[1]) for entry in reef if entry is not None]
    best_coral, best_score = max(occupied, key=lambda x: x[1])
    best_alphas = best_coral[:]

    for _ in range(n_iter):
        larvae = []
        occupied_indices = [i for i in range(reef_size) if reef[i] is not None]

        # ── 1. Broadcast spawning (reproducción sexual) ──────────
        n_broadcast = max(2, int(len(occupied_indices) * Fb))
        broadcasters = random.sample(occupied_indices, min(n_broadcast, len(occupied_indices)))
        random.shuffle(broadcasters)

        for j in range(0, len(broadcasters) - 1, 2):
            parent1 = reef[broadcasters[j]][0]
            parent2 = reef[broadcasters[j + 1]][0]
            larva = _crossover_larva(parent1, parent2)
            larvae.append(larva)

        # ── 2. Brooding (reproducción asexual / budding) ─────────
        n_budding = max(1, int(len(occupied_indices) * Fa))
        budders = random.sample(occupied_indices, min(n_budding, len(occupied_indices)))

        for j in budders:
            parent = reef[j][0]
            larva = _gaussian_larva(parent, sigma=sigma)
            larvae.append(larva)

        # ── 3. Asentamiento de larvas ────────────────────────────
        for larva in larvae:
            larva_score = fitness_fn(larva)

            # Intentar asentarse en una posición aleatoria
            pos = random.randint(0, reef_size - 1)

            if reef[pos] is None:
                # Posición vacía: la larva se asienta
                reef[pos] = (larva, larva_score)
            else:
                # Posición ocupada: la larva reemplaza si es mejor
                if larva_score > reef[pos][1]:
                    reef[pos] = (larva, larva_score)

            # Actualizar mejor global
            if larva_score > best_score:
                best_score = larva_score
                best_alphas = larva[:]

        # ── 4. Depredación ───────────────────────────────────────
        occupied_indices = [i for i in range(reef_size) if reef[i] is not None]
        n_depredation = max(1, int(len(occupied_indices) * Fd))

        # Ordenar ocupados por fitness ascendente (peores primero)
        occupied_indices.sort(key=lambda i: reef[i][1])

        for j in range(min(n_depredation, len(occupied_indices))):
            reef[occupied_indices[j]] = None

    return best_alphas, best_score