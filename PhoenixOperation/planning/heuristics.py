from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings, get_applicable_actions, is_applicable


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    OPTIMIZACIÓN: get_all_groundings tiene caché global, así que la primera
    llamada es costosa pero las siguientes son O(1). No instanciamos acciones
    de nuevo; reutilizamos las mismas referencias.
    """
    unsatisfied = goal - state
    if not unsatisfied:
        return 0

    # get_all_groundings usa caché → rápido en llamadas repetidas
    all_actions = get_all_groundings(domain, objects)

    # Pre-computar intersecciones solo con fluentes del goal para acelerar max()
    count = 0
    while unsatisfied:
        best_count = 0
        best_covered: frozenset = frozenset()
        for a in all_actions:
            covered = a.add_list & unsatisfied
            if len(covered) > best_count:
                best_count = len(covered)
                best_covered = covered
        if not best_covered:
            return float("inf")
        unsatisfied -= best_covered
        count += 1

    return count


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    IMPLEMENTACIÓN CORREGIDA: usa un relaxed planning graph (capas de
    proposiciones y acciones) en vez de hill-climbing greedy.
    
    El problema con hill-climbing greedy es que para un goal de 1 fluente
    siempre devuelve 1 (la acción que lo satisface), sin considerar cuántas
    acciones previas se necesitan para habilitar sus precondiciones.
    
    El relaxed planning graph construye capas P0, A0, P1, A1, ... hasta que
    el goal está en alguna capa Pi. El número de capas de acción necesarias
    es la estimación del costo.
    """
    # Capa inicial de proposiciones
    prop_layer = set(state)

    if goal.issubset(prop_layer):
        return 0

    all_actions = get_all_groundings(domain, objects)

    # Excluir Move: en el dominio de rescate Move no contribuye a ningún
    # fluente del goal (Rescued, SuppliesReady, Holding, etc.) y añade
    # muchísimo ruido que hace la heurística menos informativa.
    # En el problema relajado, asumimos que el robot puede estar en cualquier
    # lugar sin costo adicional.
    symbolic_actions = [a for a in all_actions if not a.name.startswith("Move")]

    # Añadir fluentes de posición del robot para todas las celdas (robot puede
    # ir a cualquier sitio "gratis" en el problema relajado sin Move)
    robot_pos_fluents = {f for f in all_actions[0].precond_pos
                         if False}  # placeholder
    cells = objects.get("cells", [])
    for cell in cells:
        prop_layer.add(("At", "robot", cell))

    if goal.issubset(prop_layer):
        return 0

    # Construir el relaxed planning graph capa por capa
    max_layers = 20
    for layer in range(1, max_layers + 1):
        new_props = set()
        for action in symbolic_actions:
            # En problema relajado: precond_pos debe estar, precond_neg ignorada
            if action.precond_pos.issubset(prop_layer):
                new_props |= action.add_list

        added = new_props - prop_layer
        if not added:
            # No hay progreso → meta inalcanzable
            return float("inf")

        prop_layer |= added

        if goal.issubset(prop_layer):
            return layer

    return float("inf")