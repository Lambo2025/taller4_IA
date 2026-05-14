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

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    OPTIMIZACIÓN PRINCIPAL: en vez de llamar get_applicable_actions() que
    internamente llama get_all_groundings() en cada iteración del hill-climbing
    (O(|groundings|) repetidas veces), usamos la caché de groundings y filtramos
    directamente sobre la lista cacheada. Esto evita re-crear miles de objetos
    Action en cada paso del hill-climbing.
    """
    # Usar set mutable para el estado relajado (más rápido que frozenset para |=)
    relaxed_state = set(state)
    goal_set = set(goal)
    count = 0

    # get_all_groundings usa caché → se computa UNA sola vez por run
    all_actions = get_all_groundings(domain, objects)

    while not goal_set.issubset(relaxed_state):
        # Filtrar acciones aplicables directamente (sin crear frozenset temporal)
        unsatisfied = goal_set - relaxed_state
        best_action = None
        best_gain = 0

        for action in all_actions:
            # Verificar aplicabilidad sin crear frozenset (usando issubset sobre set)
            if not action.precond_pos.issubset(relaxed_state):
                continue
            if not action.precond_neg.isdisjoint(relaxed_state):
                continue
            gain = len(action.add_list - relaxed_state)
            if gain > best_gain:
                best_gain = gain
                best_action = action

        if best_action is None or best_gain == 0:
            return float("inf")

        # Aplicar en modo relajado: solo agregar, nunca borrar
        relaxed_state |= best_action.add_list
        count += 1

    return count