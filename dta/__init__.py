"""
Schedule-based dynamic transit assignment with hard capacity and boarding
priority.

The implementation is a small, dependency-light library:

* ``networkx`` replaces Sage for topological sorting / linear extensions.
* ``scipy.optimize.linprog`` replaces Gurobi for the system-optimum LP.
* Cycle breaking is deterministic (timely-last boarding-priority rule) instead
  of ``random.choice``.
* Assignment routines return structured result objects instead of printing and
  calling ``exit``.

See the repository README for setup and experiment instructions.
"""

from .network import BoardingConflict, TransitInstance, Edge
from .paths import enumerate_paths, restrict_pathset, PathSet
from .priority_graph import build_priority_graph, break_cycles, linear_extension
from .assignment import (
    classical_ue_assignment,
    system_optimum_assignment,
    quasi_ue_assignment,
    componentwise_quasi_ue_assignment,
    improving_switches,
    lexicographic_assignment,
    lp_priority_assignment,
    que_order,
    greedy_fill,
    lp_solve,
    AssignmentResult,
)
from . import diagnostics
from . import io

__all__ = [
    "TransitInstance",
    "Edge",
    "BoardingConflict",
    "enumerate_paths",
    "restrict_pathset",
    "PathSet",
    "build_priority_graph",
    "break_cycles",
    "linear_extension",
    "system_optimum_assignment",
    "classical_ue_assignment",
    "quasi_ue_assignment",
    "componentwise_quasi_ue_assignment",
    "improving_switches",
    "lexicographic_assignment",
    "lp_priority_assignment",
    "que_order",
    "greedy_fill",
    "lp_solve",
    "AssignmentResult",
    "diagnostics",
    "io",
]
