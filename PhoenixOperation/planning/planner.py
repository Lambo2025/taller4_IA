from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, Stack, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    OPTIMIZACIÓN: en vez de copiar plan + [action] en cada nodo (O(d) por nodo,
    O(d²) total), guardamos un dict de punteros padre {state: (parent_state, action)}.
    El plan se reconstruye UNA sola vez al final en O(d).
    """
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    # parent[state] = (parent_state, action_taken)
    parent: dict = {start: None}
    frontier = Queue()
    frontier.push(start)

    while not frontier.isEmpty():
        state = frontier.pop()
        for next_state, action, _ in problem.getSuccessors(state):
            if next_state not in parent:
                parent[next_state] = (state, action)
                if problem.isGoalState(next_state):
                    # Reconstruir plan hacia atrás
                    plan = []
                    cur = next_state
                    while parent[cur] is not None:
                        prev, act = parent[cur]
                        plan.append(act)
                        cur = prev
                    plan.reverse()
                    return plan
                frontier.push(next_state)

    return []


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    if action.add_list.isdisjoint(goal_set):
        return None
    if not action.del_list.isdisjoint(goal_set):
        return None
    return (goal_set - action.add_list) | action.precond_pos
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    OPTIMIZACIONES:
    1. Usa _add_index para filtrar acciones relevantes en O(k) en vez de O(|groundings|).
    2. Separa acciones de navegación (Move) de acciones simbólicas: solo regresa
       acciones simbólicas (PickUp, PutDown, Rescue, SetupSupplies) en el BFS
       principal. La navegación entre celdas se resuelve con BFS de movimientos
       puros cuando el goal requiere At(robot, destino).
    3. Filtros de poda agresivos para dead-ends.
    """
    ### Your code here ###
    static = {"MedicalPost", "Adjacent"}
    add_index = getattr(problem, "_add_index", None)
    all_actions = problem._all_groundings if hasattr(problem, "_all_groundings") \
        else get_all_groundings(problem.domain, problem.objects)

    # Separar acciones de movimiento de acciones simbólicas
    move_actions    = [a for a in all_actions if a.name.startswith("Move")]
    symbolic_actions = [a for a in all_actions if not a.name.startswith("Move")]

    # Celdas adyacentes para navegar
    adj: dict = {}
    for f in problem.initial_state:
        if f[0] == "Adjacent":
            adj.setdefault(f[1], []).append(f[2])

    def robot_can_reach(from_cell, to_cell) -> bool:
        """BFS de celdas para comprobar si el robot puede ir de from→to."""
        if from_cell == to_cell:
            return True
        visited_cells: set = {from_cell}
        q: list = [from_cell]
        while q:
            cur = q.pop()
            for nb in adj.get(cur, []):
                if nb == to_cell:
                    return True
                if nb not in visited_cells:
                    visited_cells.add(nb)
                    q.append(nb)
        return False

    def nav_plan(from_cell, to_cell) -> list:
        """BFS de celdas para obtener la secuencia de acciones Move."""
        if from_cell == to_cell:
            return []

        # Índice: (from, to) → Action de Move, solo para pares adyacentes reales
        move_index: dict = {}
        for a in move_actions:
            # Extraer from/to de las precondiciones
            at_pre = next((f for f in a.precond_pos if f[0] == "At" and f[1] == "robot"), None)
            at_add = next((f for f in a.add_list if f[0] == "At" and f[1] == "robot"), None)
            if at_pre and at_add:
                # Solo guardar si hay adyacencia real (la acción la exige)
                adj_pre = next((f for f in a.precond_pos if f[0] == "Adjacent"), None)
                if adj_pre:
                    move_index[(at_pre[2], at_add[2])] = a

        parent_cell: dict = {from_cell: None}
        parent_action: dict = {from_cell: None}
        q: list = [from_cell]
        found = False
        while q and not found:
            cur = q.pop(0)
            for nb in adj.get(cur, []):
                if nb not in parent_cell and (cur, nb) in move_index:
                    parent_cell[nb] = cur
                    parent_action[nb] = move_index[(cur, nb)]
                    if nb == to_cell:
                        found = True
                        break
                    q.append(nb)

        if not found:
            return []

        path = []
        c = to_cell
        while parent_action[c] is not None:
            path.append(parent_action[c])
            c = parent_cell[c]
        path.reverse()
        return path

    def simplify(goal_desc):
        """Elimina fluents estáticos satisfechos."""
        return frozenset(f for f in goal_desc
                         if f[0] not in static or f not in problem.initial_state)

    def is_consistent(goal_desc):
        """Descarta goals donde la misma entidad aparece At en dos lugares."""
        at_locs: dict = {}
        for f in goal_desc:
            if f[0] == "At":
                entity, loc = f[1], f[2]
                if entity in at_locs and at_locs[entity] != loc:
                    return False
                at_locs[entity] = loc
        return True

    def get_relevant_symbolic(unsatisfied):
        """Solo acciones simbólicas relevantes (no Move)."""
        if add_index is not None:
            seen: set = set()
            relevant = []
            for fluent in unsatisfied:
                for action in add_index.get(fluent, []):
                    if not action.name.startswith("Move") and id(action) not in seen:
                        seen.add(id(action))
                        relevant.append(action)
            return relevant
        return [a for a in symbolic_actions if not a.add_list.isdisjoint(unsatisfied)]

    def resolve_navigation(goal_desc, partial_plan):
        """
        Si el goal tiene At(robot, dest) y el robot empieza en robot_start,
        inserta los movimientos necesarios al principio del plan.
        """
        robot_start = None
        for f in problem.initial_state:
            if f[0] == "At" and f[1] == "robot":
                robot_start = f[2]
                break

        # Encontrar destino del robot en el goal
        robot_dest = None
        for f in goal_desc:
            if f[0] == "At" and f[1] == "robot":
                robot_dest = f[2]
                break

        if robot_dest is None or robot_dest == robot_start:
            return partial_plan

        moves = nav_plan(robot_start, robot_dest)
        return moves + partial_plan

def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    OPTIMIZACIONES para evitar la explosión con Move:
    1. Move solo se regresa cuando At(robot, dest) está en el goal Y dest no
       es la posición inicial del robot (ya que si ya está ahí no hace falta mover).
    2. Para Move, solo se generan acciones cuyo from_cell es adyacente al dest
       (ya pre-filtrado en el índice de adyacencia).
    3. Poda de consistencia fuerte: entidad en dos sitios, o dos objetos en mano.
    4. Poda de Pickable: si el goal requiere Pickable(x) pero x ya no es pickable
       en initial_state, es dead-end.
    """
    ### Your code here ###
    static = {"MedicalPost", "Adjacent"}

    all_actions = problem._all_groundings if hasattr(problem, "_all_groundings") \
        else get_all_groundings(problem.domain, problem.objects)
    add_index = getattr(problem, "_add_index", None)

    # Posición inicial del robot
    robot_init_pos = next(
        (f[2] for f in problem.initial_state if f[0] == "At" and f[1] == "robot"), None
    )

    # Índice de adyacencia: celda → celdas adyacentes
    adj: dict = {}
    for f in problem.initial_state:
        if f[0] == "Adjacent":
            adj.setdefault(f[1], []).append(f[2])

    # Para Move: solo groundings adyacentes reales (from, to adyacentes)
    adjacent_pairs: set = set()
    for f in problem.initial_state:
        if f[0] == "Adjacent":
            adjacent_pairs.add((f[1], f[2]))

    # Índice (from_cell, to_cell) → Move action, SOLO pares adyacentes reales
    move_adj_index: dict = {}
    for a in all_actions:
        if not a.name.startswith("Move"):
            continue
        at_pre = next((f for f in a.precond_pos if f[0] == "At" and f[1] == "robot"), None)
        at_add = next((f for f in a.add_list    if f[0] == "At" and f[1] == "robot"), None)
        if at_pre and at_add:
            frm, to = at_pre[2], at_add[2]
            # Solo incluir si (from, to) es un par adyacente real en el mapa
            if (frm, to) in adjacent_pairs:
                move_adj_index[(frm, to)] = a

    def simplify(goal_desc):
        return frozenset(f for f in goal_desc
                         if f[0] not in static or f not in problem.initial_state)

    def is_consistent(goal_desc):
        at_locs: dict = {}
        holding: list = []
        for f in goal_desc:
            if f[0] == "At":
                e, l = f[1], f[2]
                if e in at_locs and at_locs[e] != l:
                    return False
                at_locs[e] = l
            elif f[0] == "Holding":
                holding.append(f[2])
        return len(set(holding)) <= 1

    def get_relevant(unsatisfied):
        """
        Acciones simbólicas para fluentes simbólicos.
        Acciones Move SOLO para fluentes At(robot, dest) donde dest ≠ robot_init_pos
        y se devuelven solo las Move cuyo to_cell = dest (las adyacentes al goal).
        """
        seen: set = set()
        result: list = []

        for fluent in unsatisfied:
            if fluent[0] == "At" and fluent[1] == "robot":
                dest = fluent[2]
                if dest == robot_init_pos:
                    continue   # el robot ya está ahí en el initial state
                # Solo Move acciones que llevan al robot a dest
                for (frm, to), action in move_adj_index.items():
                    if to == dest and id(action) not in seen:
                        seen.add(id(action))
                        result.append(action)
            else:
                if add_index is not None:
                    for action in add_index.get(fluent, []):
                        if not action.name.startswith("Move") and id(action) not in seen:
                            seen.add(id(action))
                            result.append(action)
                else:
                    for action in all_actions:
                        if not action.name.startswith("Move") and \
                           fluent in action.add_list and id(action) not in seen:
                            seen.add(id(action))
                            result.append(action)

        return result

    frontier = Queue()
    start = simplify(problem.goal)
    frontier.push((start, []))
    visited = {start}

    while not frontier.isEmpty():
        goal_desc, plan = frontier.pop()
        problem._expanded += 1

        if goal_desc.issubset(problem.initial_state):
            return plan

        unsatisfied = goal_desc - problem.initial_state
        relevant = get_relevant(unsatisfied)

        for action in relevant:
            new_goal = regress(goal_desc, action)
            if new_goal is None:
                continue
            if any(f[0] in static and f not in problem.initial_state for f in new_goal):
                continue
            new_goal = simplify(new_goal)
            if new_goal in visited or not is_consistent(new_goal):
                continue
            visited.add(new_goal)
            frontier.push((new_goal, [action] + plan))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    OPTIMIZACIÓN: igual que forwardBFS, usamos dict de punteros padre para
    evitar copiar la lista de acciones en cada nodo (O(d²) → O(d) total).
    """
    ### Your code here ###
    start = problem.getStartState()

    if problem.isGoalState(start):
        return []

    h0 = heuristic(start, problem.goal, problem.domain, problem.objects)
    frontier = PriorityQueue()
    frontier.push(start, h0)

    # parent[state] = (parent_state, action_taken)
    parent: dict = {start: None}
    best_g: dict = {start: 0}

    while not frontier.isEmpty():
        state = frontier.pop()
        g = best_g[state]

        if problem.isGoalState(state):
            # Reconstruir plan
            plan = []
            cur = state
            while parent[cur] is not None:
                prev, act = parent[cur]
                plan.append(act)
                cur = prev
            plan.reverse()
            return plan

        for next_state, action, cost in problem.getSuccessors(state):
            new_g = g + cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                parent[next_state] = (state, action)
                h = heuristic(next_state, problem.goal, problem.domain, problem.objects)
                frontier.push(next_state, new_g + h)

    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner

# Aliases for command-line names used by main.py
forwardSearch = forwardBFS