---
name: qclab
description: "QC Lab quantum-classical dynamics simulation skill. Trigger on: QC Lab, qclab, tempelaar-team/qclab, quantum-classical dynamics simulations, spin-boson, Holstein lattice, FMO, Tully problem I/II/III, mean-field, Ehrenfest, fewest-switches surface hopping, FSSH, ab initio FSSH, new model/ingredient/task/algorithm in quantum-classical context, debugging RK4 integration/surface hopping/gauge fixing/trajectory averaging, the complex-classical coordinate formalism (Miyazaki Krotz Tempelaar 2024). Also trigger when the request involves trajectory-based quantum-classical dynamics with complex coordinates, even without explicitly naming QC Lab."
---

# QC Lab Skill

QC Lab is a Python package for trajectory-based quantum-classical dynamics simulations, built around the complex-classical coordinate formalism where classical phase-space points (q, p) are represented as a single complex coordinate z. It is developed by the Tempelaar group at Northwestern and lives at `tempelaar-team/qclab`.

## The five-object architecture

| Object | Role |
|---|---|
| `Simulation` | Top-level container: settings, model, algorithm, initial state |
| `Model` | Physical system: constants + ingredient list |
| `Algorithm` | Numerical recipe: three task lists (initialization, update, collect) |
| `Constants` | Attribute-bag with a `__setattr__` hook that re-fires `_init_*` methods on change |
| `Data` | Trajectory-averaged output; supports HDF5 and `.npz` I/O |

## The recipe / task / ingredient pattern

Every simulation is assembled from three layers:

- **Ingredients** are physics functions `f(model, parameters, **kwargs)` that compute quantities like Hamiltonians and gradients. They live on the Model as `(slot_name, callable)` tuples. The same ingredient can be shared across models.
- **Tasks** are algorithm steps `task(sim, state, parameters, **opts) -> (state, parameters)`. They read/write named entries in the `state` dict and call ingredients via `sim.model.get(...)`. Every state key a task touches is exposed as a `*_name` keyword argument.
- **Recipes** are plain Python lists of tasks on the Algorithm class. `Algorithm.execute_recipe` threads `(state, parameters)` through each task in order.

A new algorithm = a new ordering of existing tasks. A new model = a new ingredient list.

## Decision tree: which reference file to read

| You need to... | Read this file |
|---|---|
| Understand the overall architecture or module layout | `references/summary.md` |
| Write new code that must match QC Lab style (naming, docstrings, module layout) | `references/style_guide.md` |
| Look up ingredient slot names, state-dict keys, model constants, algorithm settings | `references/conventions.md` |
| Run a simulation, read Data, add a collect task, do a parameter scan, build a new model | `references/handbook.md` |
| Open a PR / contribute changes upstream / understand what CI will run | `references/contributing.md` |
| See a complete runnable script | `worked_examples/` — pick the one closest to your task |

For most code-generation requests, read `references/conventions.md` (for the exact key/slot vocabulary) AND `references/handbook.md` (for the recipe to follow). For new-model or new-ingredient work, also read `references/style_guide.md`.

## Critical mistakes to avoid

These are the errors Claude is most likely to make when generating QC Lab code. Internalize them before writing anything:

1. **Don't invent state-dict keys.** The wavefunction key is `wf_db` (not `wavefunction`, `psi`, `state_vector`, etc.). The density matrix is `dm_db`. The classical coordinate is `z`. Read `references/conventions.md` section 3.2 for the full vocabulary.

2. **Don't invent ingredient slot names.** The standard slots are: `h_q`, `h_qc`, `h_c`, `dh_qc_dzc`, `dh_c_dzc`, `init_classical`, `hop`, `derivative_coupling_dzc`, `gauge_field_force`, `ab_initio_property_calculator`. Read `references/conventions.md` section 3.1 for signatures. Do not create new slot names without also writing the tasks that consume them.

3. **Don't hard-code state keys in tasks.** Every key a task reads or writes must be a `*_name` keyword argument with a sensible default. This is what lets recipes rebind tasks to different keys via `functools.partial`.

4. **Don't bypass the Simulation/Model/Algorithm pattern.** Never write a hand-rolled integration loop. The correct pattern is always: `Simulation(settings)` -> set `sim.model` -> set `sim.algorithm` -> set `sim.initial_state["wf_db"]` -> call `serial_driver(sim)`.

5. **Don't confuse `z` with `(q, p)`.** The canonical representation is the complex coordinate `z`. Use `z_to_q`, `qp_to_z`, `dqdp_to_dzc`, `dzdzc_to_dqdp` from `qclab.functions` to convert.

6. **Don't forget `update_dh_qc_dzc` / `update_h_q` flags.** Set these to `False` on the model when the quantity doesn't depend on `z` — it's a significant speedup. But if a constant that affects the gradient can change, set to `True` or you'll get wrong gradients from the cache.

7. **Don't mutate `collect_recipe` in-place before instantiation.** `Algorithm.__init__` deep-copies the class attribute. Always use: `sim.algorithm.collect_recipe = sim.algorithm.collect_recipe + [my_task]` *after* the algorithm is instantiated.

8. **Don't use `print` or f-strings in log messages.** Use `logger = logging.getLogger(__name__)` and `%s` placeholders.

9. **Sparse gradients must return `(inds, mels, shape)`.** Always in that order. `inds` comes from `np.where` on the dense array. The consumer is `qclab.functions.calc_sparse_inner_product`.

10. **The batch axis is always first.** Shapes are `(B, C)`, `(B, N)`, `(B, N, N)`, `(B, C, N, N)`.

11. **Set `initial_state["wf_db"]` to an `np.array` with `dtype=complex`.** A Python list will be silently skipped.

12. **Don't reach into `state` from inside an ingredient.** Ingredients see `model`, `parameters`, and `kwargs` only.

13. **Get `batch_size` from `sim.settings.batch_size`, not `len(z)`.** In tasks, always use the canonical source `sim.settings.batch_size` rather than inferring the batch size from the shape of a state variable.

14. **Keep imports at module level, not inside task or ingredient functions.** Imports from `qclab.functions` (like `z_to_q`) belong at the top of the file, not inside the task body.

15. **Don't write new ingredients when modifying constants achieves the same thing.** The stock ingredients are parametric — they read constants like `diagonal_linear_coupling`, `harmonic_frequency`, etc. from the model. If you want a "spin-boson-like" model with a different spectral density, just subclass the existing model and override `_init_h_c` / `_init_h_qc` to set up different frequency distributions and coupling constants. Only write a new ingredient when the *functional form* of the physics is genuinely new (e.g., off-diagonal coupling when only diagonal coupling exists).

16. **Collect tasks must only copy values to `output_dict` — no computation.** If your observable requires computation (e.g., converting `z` to real-space position, projecting onto an adiabatic surface), put the computation in an **update task** and have a separate **collect task** that just copies the result into `state["output_dict"]`. This is the QC Lab separation of concerns: update tasks compute, collect tasks record.

17. **`gauge_fixing: "phase_der_couple"` is only needed when coupling is complex-valued.** For real-valued problems (Tully models, standard spin-boson), the default `"sign_overlap"` is sufficient and correct. Only use `"phase_der_couple"` when the derivative couplings or Hamiltonians are genuinely complex.

18. **When a request is ambiguous, ask for clarification.** If the user asks for something like "expectation value of position on the lower adiabatic surface", there are multiple reasonable interpretations (population-weighted position, trajectory-resolved position, etc.). Ask the user what they mean rather than guessing.

19. **Don't push directly to `dev` or `main`, and don't open PRs into `main`.**
    QC Lab uses a PR-into-`dev` workflow; release PRs (`dev` → `main`)
    are a maintainer action. See `references/contributing.md` section 6
    for the full list of operations Claude should not perform without
    explicit confirmation.

## Quick-start template

```python
import numpy as np
from qclab import Simulation
from qclab.models import SpinBoson          # or any Model
from qclab.algorithms import MeanField      # or any Algorithm
from qclab.dynamics import serial_driver

sim = Simulation({
    "tmax": 10.0, "dt_update": 0.01, "dt_collect": 0.1,
    "num_trajs": 100, "batch_size": 50, "progress_bar": False,
})
sim.model     = SpinBoson()
sim.algorithm = MeanField()

wf0 = np.zeros(sim.model.constants.num_quantum_states, dtype=complex)
wf0[0] = 1.0
sim.initial_state["wf_db"] = wf0

data = serial_driver(sim)

t   = data.data_dict["t"]
dm  = data.data_dict["dm_db"]
pop = dm[:, 0, 0].real
```

## Worked examples

Three complete, runnable scripts in `worked_examples/`:

| Script | What it demonstrates |
|---|---|
| `spin_boson_meanfield.py` | Minimum-viable simulation: spin-boson + mean-field, plot populations |
| `tully_momentum_scan.py` | Parameter scan over momentum + custom collect task for transmit/reflect |
| `lvc_conical_intersection.py` | New model from scratch with novel ingredients, run + plot |

Read the one closest to the user's request as a starting template.
