from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    from planning.utils import Queue
    from planning.pddl import get_all_groundings

    def execute_primitives(plan_prefix, start_state):
        """Ejecuta acciones primitivas hasta el primer HLA y retorna el estado."""
        state = start_state
        for step in plan_prefix:
            if not is_primitive(step):
                break
            if not is_applicable(state, step):
                return None
            state = apply_action(state, step)
        return state

    def expand_hla(hla, state, problem):
        """
        Genera los refinamientos de un HLA dado el estado actual.
        Si el HLA es una DynamicHLA, llama a su función de refinamiento con el estado.
        Si es un HLA estático, devuelve sus refinamientos directamente.
        """
        if hasattr(hla, 'expand'):
            return hla.expand(state, problem)
        return hla.refinements

    # BFS: cada nodo es (plan, estado_antes_del_primer_HLA)
    frontier = Queue()
    start = problem.getStartState()
    frontier.push((list(hlas), start))

    visited: set = set()

    while not frontier.isEmpty():
        plan, state_at_first_hla = frontier.pop()

        sig = tuple(s.name for s in plan)
        if sig in visited:
            continue
        visited.add(sig)
        problem._expanded += 1

        if is_plan_primitive(plan):
            # Verificar ejecución completa desde initial_state
            s = problem.getStartState()
            ok = True
            for a in plan:
                if not is_applicable(s, a):
                    ok = False
                    break
                s = apply_action(s, a)
            if ok and problem.isGoalState(s):
                return plan
            continue

        # Encontrar primer HLA y estado justo antes de él
        state = problem.getStartState()
        idx = -1
        for i, step in enumerate(plan):
            if not is_primitive(step):
                idx = i
                break
            if not is_applicable(state, step):
                state = None
                break
            state = apply_action(state, step)

        if idx == -1 or state is None:
            continue

        hla = plan[idx]
        refs = expand_hla(hla, state, problem)

        for refinement in refs:
            new_plan = plan[:idx] + list(refinement) + plan[idx + 1:]
            frontier.push((new_plan, state))

    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.
    """
    robot    = problem.objects["robots"][0]
    patients = problem.objects["patients"]
    supplies = problem.objects["supplies"]
    medposts = problem.objects["medical_posts"]
    medical_post = medposts[0]

    # Grafo de adyacencia
    adj: dict = {}
    for f in problem.initial_state:
        if f[0] == "Adjacent":
            adj.setdefault(f[1], []).append(f[2])

    # Índice (from, to) → Action Move adyacente
    move_index: dict = {}
    for a in problem._all_groundings:
        if not a.name.startswith("Move"):
            continue
        at_pre = next((f for f in a.precond_pos if f[0] == "At" and f[1] == robot), None)
        at_add = next((f for f in a.add_list    if f[0] == "At" and f[1] == robot), None)
        adj_f  = next((f for f in a.precond_pos if f[0] == "Adjacent"), None)
        if at_pre and at_add and adj_f:
            frm, to = at_pre[2], at_add[2]
            if to in adj.get(frm, []):
                move_index[(frm, to)] = a

    def find_actions_by_schema(schema_name: str) -> list:
        return [a for a in problem._all_groundings
                if a.name.startswith(schema_name + "(")]

    def bfs_path(from_cell, to_cell):
        """BFS mínimo entre dos celdas. Devuelve lista de Move actions."""
        if from_cell == to_cell:
            return []
        parent = {from_cell: None}
        parent_act = {from_cell: None}
        q = [from_cell]
        while q:
            cur = q.pop(0)
            for nb in adj.get(cur, []):
                if nb not in parent and (cur, nb) in move_index:
                    parent[nb] = cur
                    parent_act[nb] = move_index[(cur, nb)]
                    if nb == to_cell:
                        path = []
                        c = to_cell
                        while parent_act[c]:
                            path.append(parent_act[c])
                            c = parent[c]
                        path.reverse()
                        return path
                    q.append(nb)
        return []

    def get_robot_pos(state):
        return next((f[2] for f in state if f[0] == "At" and f[1] == robot), None)

    def get_entity_pos(state, entity):
        return next((f[2] for f in state if f[0] == "At" and f[1] == entity), None)

    # -----------------------------------------------------------------------
    # DynamicHLA: HLA que genera refinamientos según el estado actual
    # -----------------------------------------------------------------------
    class DynamicHLA(HLA):
        def __init__(self, name, expand_fn):
            super().__init__(name, refinements=[])
            self._expand_fn = expand_fn

        def expand(self, state, problem):
            return self._expand_fn(state, problem)

    # Navigate: mueve robot de su posición actual hasta to_cell
    def make_navigate(to_cell, label=None):
        name = label or f"Navigate(→{to_cell})"

        def expand_navigate(state, problem):
            robot_pos = get_robot_pos(state)
            if robot_pos is None or robot_pos == to_cell:
                return [[]]  # Ya está ahí → refinamiento vacío
            moves = bfs_path(robot_pos, to_cell)
            if not moves:
                return []
            return [moves]

        return DynamicHLA(name, expand_navigate)

    # PrepareSupplies: ir a recoger supply, luego llevarlo a medical_post
    def make_prepare_supplies(supply, med_post):
        name = f"PrepareSupplies({supply}, {med_post})"

        def expand_prepare(state, problem):
            robot_pos  = get_robot_pos(state)
            supply_pos = get_entity_pos(state, supply)

            # Si SuppliesReady ya está en el estado actual, no hace falta preparar
            if ("SuppliesReady", med_post) in state:
                return [[]]  # refinamiento vacío — ya está listo

            if robot_pos is None or supply_pos is None:
                return []

            pickup = next(
                (a for a in find_actions_by_schema("PickUp")
                 if ("At", supply, supply_pos) in a.precond_pos
                 and ("At", robot, supply_pos) in a.precond_pos),
                None
            )
            setup = next(
                (a for a in find_actions_by_schema("SetupSupplies")
                 if ("Holding", robot, supply) in a.precond_pos
                 and ("MedicalPost", med_post) in a.precond_pos),
                None
            )
            if not pickup or not setup:
                return []

            nav1 = make_navigate(supply_pos, f"Nav→{supply_pos}")
            nav2 = make_navigate(med_post,   f"Nav→{med_post}")
            return [[nav1, pickup, nav2, setup]]

        return DynamicHLA(name, expand_prepare)

    # ExtractPatient: ir a recoger paciente, llevarlo al puesto, dejarlo
    def make_extract_patient(patient, med_post):
        name = f"ExtractPatient({patient}, {med_post})"

        def expand_extract(state, problem):
            robot_pos   = get_robot_pos(state)
            patient_pos = get_entity_pos(state, patient)
            if robot_pos is None or patient_pos is None:
                return []

            pickup = next(
                (a for a in find_actions_by_schema("PickUp")
                 if ("At", patient, patient_pos) in a.precond_pos
                 and ("At", robot, patient_pos) in a.precond_pos),
                None
            )
            putdown = next(
                (a for a in find_actions_by_schema("PutDown")
                 if ("Holding", robot, patient) in a.precond_pos
                 and ("At", robot, med_post) in a.precond_pos),
                None
            )
            if not pickup or not putdown:
                return []

            nav1 = make_navigate(patient_pos, f"Nav→{patient_pos}")
            nav2 = make_navigate(med_post,    f"Nav→{med_post}")
            return [[nav1, pickup, nav2, putdown]]

        return DynamicHLA(name, expand_extract)

    # FullRescueMission: preparar + extraer + rescatar
    def make_full_rescue(supply, patient, med_post):
        name = f"FullRescueMission({supply}, {patient}, {med_post})"

        rescue = next(
            (a for a in find_actions_by_schema("Rescue")
             if ("MedicalPost", med_post) in a.precond_pos
             and ("At", patient, med_post) in a.precond_pos),
            None
        )

        prepare = make_prepare_supplies(supply, med_post)
        extract = make_extract_patient(patient, med_post)

        def expand_full(state, problem):
            if not rescue:
                return []
            return [[prepare, extract, rescue]]

        return DynamicHLA(name, expand_full)

    # -----------------------------------------------------------------------
    # Construir la jerarquía top-level
    # -----------------------------------------------------------------------
    top_level: list = []

    if len(patients) == 1:
        top_level = [make_full_rescue(supplies[0], patients[0], medical_post)]
    else:
        for i, patient in enumerate(patients):
            supply = supplies[i] if i < len(supplies) else supplies[-1]
            top_level.append(make_full_rescue(supply, patient, medical_post))

    return top_level
