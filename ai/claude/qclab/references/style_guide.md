# QC Lab — Style Guide

This guide reflects the conventions in the existing `src/qclab` source. New code should match them.

For prose written into the QC Lab documentation (Sphinx `.rst` files under `docs/`, the README, or collaborator handouts), the corresponding style file is `documentation_style.md`. Read it before writing or revising documentation.

## Table of contents

- General Python style
- Naming conventions
- Module layout
- Numerical conventions
- Adding a Model
- Writing an Ingredient
- Writing a Task
- Writing an Algorithm
- Working with Constants
- Logging, errors, debug checks
- Don'ts

---

## General Python style

- Target Python >= 3.8.
- PEP 8. Indent 4 spaces. Lines ~88-100 columns.
- `snake_case` for functions/variables/modules; `CamelCase` for classes.
- Explicit absolute imports rooted at `qclab` (`from qclab import functions`, `from qclab.model import Model`).
- No third-party deps beyond `numpy` and `tqdm`. `numba`, `h5py`, `mpi4py`, `ase` are optional — guard through `qclab.utils.DISABLE_*` flags.
- Standard logging: `logger = logging.getLogger(__name__)`. Never `print`. Use `%s` placeholders, never f-strings in log calls.

## Naming conventions

### Casing

- **Classes** — `CamelCase`. Acronyms stay capitalized (`FMO`, `QC`, `MPI`).
- **Modules** — `lower_snake_case`, singular for single concepts (`simulation.py`), plural for collections (`ingredients.py`, `tasks/`).
- **Functions, tasks, ingredients, variables, dict keys** — `lower_snake_case`.
- **Module-level numerical constants** — `UPPER_SNAKE_CASE` (`SMALL`, `EV_TO_INVCM`).
- **Private internals** — leading underscore (`_init_complete`, `_updating`).
- **Physical constants** — allowed to keep conventional symbol: `kBT`, `V`, `E`, `A`, `W`, `J`, `N`.

### Task naming

Task functions follow strict prefix-based naming:

| Prefix | Meaning | Module |
|---|---|---|
| `initialize_` | Runs once at t_ind == 0 | `initialization_tasks.py` |
| `update_` | Runs every update step | `update_tasks.py` |
| `collect_` | Runs at collect steps | `collect_tasks.py` |
| `copy_in_` / `copy_to_` | Duplicates a value in same dict or between dicts | `initialization_tasks.py` |
| `add_` | Augments existing state entry | `update_tasks.py` |
| `diagonalize_` | Utility task | `update_tasks.py` |

After the prefix, name the quantity produced: physical noun first, modifier second, algorithm suffix last.
- `update_dm_db_fssh` = update density matrix in diabatic basis, FSSH variant
- `update_z_rk4_k123` = update z via RK4, sub-steps k1/k2/k3

Suffix with `_wf` (mean-field) or `_fssh` (active-surface) when tasks differ by wavefunction type. Suffix with integration scheme when tasks differ there: `_rk4`, `_propagator`, `_velocity_verlet`.

### Ingredient naming

Ingredients are named `<quantity>_<form>`:

| Prefix | Meaning |
|---|---|
| `h_q_` | Quantum Hamiltonian by functional form |
| `h_qc_` | Quantum-classical Hamiltonian |
| `h_c_` | Classical Hamiltonian |
| `dh_qc_dzc_` | Gradient of h_qc w.r.t. z* |
| `dh_c_dzc_` | Gradient of h_c w.r.t. z* |
| `init_classical_` | Initial sampler, named after distribution |
| `hop_` | FSSH hop rule, named after classical H |
| `rescaling_direction_` | Rescaling direction after hop |
| `gauge_field_force_` | Optional gauge-field force |

Standard slot names (what goes in the first element of `(name, callable)` tuples): `h_q`, `h_qc`, `h_c`, `dh_qc_dzc`, `dh_c_dzc`, `init_classical`, `hop`, `derivative_coupling_dzc`, `gauge_field_force`, `ab_initio_property_calculator`.

Model-internal initializers use `_init_` prefix with leading underscore: `_init_model`, `_init_h_q`, `_init_h_qc`, `_init_h_c`.

### Low-level function naming

Functions in `functions.py`:
- Coordinate conversion: `z_to_q`, `z_to_p`, `qp_to_z`, `dqdp_to_dzc`, `dzdzc_to_dqdp`
- Linear algebra: `batch_matvec`, `transform_vec`, `transform_mat`
- Integration kernels: `update_z_rk4_k123_sum`, `update_z_rk4_k4_sum` (`_sum` = JIT inner kernel)
- JIT kernels: append `_jit` to distinguish from the ingredient wrapper
- Decorators: `make_ingredient_sparse`, `vectorize_ingredient`
- Utilities: `calc_` for computations, `gen_` for samplers

### Local variable names

Allowed short forms in function bodies:
- coordinates: `z`, `q`, `p`
- per-coordinate: `m` (mass), `h` (weight), `w` (frequency)
- thermal: `kBT`
- eigenpairs: `evec_i`, `evec_j`, `eval_i`, `eval_j`, `eigval_diff`
- sparse triple (always this order): `inds`, `mels`, `shape`
- sizes: `batch_size`, `num_classical_coordinates`, `num_quantum_states`, etc.
- indices: `traj_ind`, `state_ind`, `act_surf_ind`, `t_ind`, etc.

---

## Module layout

Every module starts with a one-sentence docstring. Imports grouped: stdlib, third-party, qclab. Module-level `logger` after imports.

## Numerical conventions

- **Complex classical coordinates `z` are canonical.** Accept and return `z`, not `(q, p)`. Convert only via `z_to_q`, `qp_to_z`, etc.
- **`complex128` for z, wavefunctions, Hamiltonians, gradients.** `float64` for energies, time, probabilities.
- **First axis is batch (trajectory).** Shapes: `(B, C)`, `(B, N)`, `(B, N, N)`, `(B, C, N, N)`.
- **Sparse gradients: `(inds, mels, shape)` triple.** `inds` from `np.where`. Wrap with `make_ingredient_sparse`; consume with `calc_sparse_inner_product`.
- **Numerical constants in `numerical_constants.py`.** No inline magic numbers.
- **Random seeds per-trajectory.** Init ingredients accept `seed` array and call `np.random.seed(seed_value)` per trajectory.

## Adding a Model

A new Model subclass should:

1. Live in `src/qclab/models/` (or be self-contained outside the package).
2. Define `default_constants` in `__init__`, call `super().__init__(self.default_constants, constants)`.
3. Set `self.update_h_q` and `self.update_dh_qc_dzc` flags.
4. Provide `_init_model` (and optionally `_init_h_q`, `_init_h_qc`, `_init_h_c`) that derive per-coordinate constants from user-facing constants.
5. Define `ingredients = [...]` class attribute with standard slot names + `_init_*` callables.
6. Reuse stock ingredients from `qclab.ingredients` when possible.
7. Cite the reference paper in the class docstring.

Use `SpinBoson` and `HolsteinLattice` as canonical examples.

## Writing an Ingredient

Signature: `f(model, parameters, **kwargs)`. Returns the physical quantity.

- Pull data from `kwargs`, not positional args.
- Pull constants from `model.constants`.
- Vectorize over batch axis, or decorate with `@functions.vectorize_ingredient`.
- For sparse gradients, decorate with `@functions.make_ingredient_sparse`.
- Docstring sections (in order): Keyword Args, Model Constants, Returns, plus LaTeX formula at top.

## Writing a Task

Signature: `def my_task(sim, state, parameters, *, foo_name="foo") -> (state, parameters)`

- Every state-dict key is a `*_name` keyword argument with sensible default.
- Read once at top, write once at bottom.
- Call ingredients via `sim.model.get("name")` which returns `(callable, found_bool)`.
- **Get `batch_size` from `sim.settings.batch_size`**, not from `len(z)` or other indirect methods. The simulation settings are the canonical source.
- **Keep imports at module level**, not inside task function bodies.
- Use `sim.settings.debug` for expensive checks.
- Place in the right module: `initialization_tasks.py`, `update_tasks.py`, or `collect_tasks.py`.
- **Collect tasks must ONLY copy values into `state["output_dict"]` — no computation.** If your observable needs to be computed from other state variables (e.g., converting `z` to position, projecting onto a surface), put that computation in a separate **update task** and have the collect task just copy the result. This is the QC Lab separation of concerns: update tasks compute, collect tasks record.

## Writing an Algorithm

A class with three class attributes: `initialization_recipe`, `update_recipe`, `collect_recipe`.

- Define `default_settings` in `__init__`, forward through `super().__init__`.
- Recipes should be linear and readable top-to-bottom.
- Never write physics in the algorithm class — factor into tasks.

## Working with Constants

- `constants.get("name", default)` for optional constants; direct access for required ones.
- Don't stash mutable state on constants.
- Private names (leading `_`) are exempt from the change hook.

## Logging, errors, debug checks

- Errors that halt: raise `ValueError`/`AttributeError`/`NameError` with clear message.
- Recoverable: `logger.warning`.
- Reproducibility info: `logger.info`.
- Use `@njit` from `qclab.utils` (not `numba` directly) so the no-op shim works.

## Don'ts

- Don't put physics in `dynamics/run_dynamics`.
- Don't add new dependencies without a strong reason.
- Don't introduce per-trajectory Python loops in tasks — vectorize or use `@njit`.
- Don't reach into `state` from inside an ingredient.
- Don't hard-code state-dict keys inside tasks.
- Don't bypass `Constants` to set model parameters.
- Don't use `print` — use `logger`.
- Don't use f-strings inside log messages — use `%s` placeholders.
- Don't write new ingredients when you can achieve the same thing by modifying constants. The stock ingredients are parametric — if you want a different spectral density or coupling distribution, subclass the existing model and override the `_init_*` methods to set up different constants. Only write a new ingredient when the functional form itself changes (e.g., off-diagonal coupling when only diagonal exists).
- Don't put computation in collect tasks. Collect tasks only copy values to `output_dict`. Computation goes in update tasks.
- Don't get `batch_size` from `len(z)` — use `sim.settings.batch_size`.
- Don't put imports inside task or ingredient functions — imports belong at module level.
