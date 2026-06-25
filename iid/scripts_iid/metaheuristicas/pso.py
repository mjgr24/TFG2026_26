# Particle Swarm Optimization

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


def solve(fitness_fn, dim, n_iter=200, pop_size=30, w=0.7, c1=1.5, c2=1.5, **kwargs):
    """
    Particle Swarm Optimization en el simplex.

    Parámetros:
    - fitness_fn: función a maximizar.
    - dim:        dimensión del vector alpha (número de hospitales).
    - n_iter:     número de iteraciones.
    - pop_size:   número de partículas.
    - w:          inercia.
    - c1:         componente cognitivo (atracción hacia mejor personal).
    - c2:         componente social (atracción hacia mejor global).

    Retorna:
    - best_alphas: mejor vector alpha encontrado.
    - best_score:  valor de fitness del mejor alpha.
    """
    # Inicializar partículas en el simplex
    positions = [_sample_simplex(dim) for _ in range(pop_size)]
    velocities = [[0.0] * dim for _ in range(pop_size)]

    # Evaluar posiciones iniciales
    scores = [fitness_fn(p) for p in positions]

    # Mejor personal de cada partícula
    personal_best_pos = [p[:] for p in positions]
    personal_best_score = scores[:]

    # Mejor global
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    global_best_pos = positions[best_idx][:]
    global_best_score = scores[best_idx]

    # Bucle principal
    for _ in range(n_iter):
        for i in range(pop_size):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()

                # Actualizar velocidad
                velocities[i][d] = (
                    w * velocities[i][d]
                    + c1 * r1 * (personal_best_pos[i][d] - positions[i][d])
                    + c2 * r2 * (global_best_pos[d] - positions[i][d])
                )

                # Actualizar posición
                positions[i][d] += velocities[i][d]

            # Proyectar al simplex
            positions[i] = _project_to_simplex(positions[i])

            # Evaluar
            score = fitness_fn(positions[i])

            # Actualizar mejor personal
            if score > personal_best_score[i]:
                personal_best_score[i] = score
                personal_best_pos[i] = positions[i][:]

                # Actualizar mejor global
                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = positions[i][:]

    return global_best_pos, global_best_score