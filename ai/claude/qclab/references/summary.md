# QC Lab — Architecture Summary

## What QC Lab is

QC Lab is a Python package for quantum-classical (QC) dynamics simulations, developed by the Tempelaar group at Northwestern. It is built around the complex-classical coordinate formalism of Miyazaki, Krotz & Tempelaar (J. Chem. Theory Comput. 2024), in which classical phase-space points (q, p) are represented as a single complex coordinate:

```
z = sqrt(m*h/2) * q + i * sqrt(1/(2*m*h)) * p
```

where `m` is the coordinate mass and `h` is a per-coordinate weight. All forces, gradients, and integrators are written in terms of `z` and its conjugate `z*`.

The package itself is described in Krotz, Garzón-Ramírez, Byrd, Miyazaki & Tempelaar, *J. Chem. Theory Comput.* **2026**, *22*, 3144–3152 (DOI [10.1021/acs.jctc.5c01818](https://doi.org/10.1021/acs.jctc.5c01818)). For paper- and SI-level questions — exact equations, figure details, reference numbers, or wording — see `references/publication.md`, which links to the open-access JCTC article and SI.

## Shipped algorithms

- **Mean-field (Ehrenfest)** — `MeanField`, `MeanFieldAbInitio`
- **Fewest-Switches Surface Hopping (FSSH)** — `FewestSwitchesSurfaceHopping`, `FewestSwitchesSurfaceHoppingAbInitio`

## Shipped models

- `SpinBoson` — two-level system coupled to a harmonic bath
- `HolsteinLattice` — electron on a lattice with phonon coupling
- `HolsteinLatticeReciprocalSpace` — same in k-space
- `FMOComplex` — Fenna-Matthews-Olson photosynthetic complex
- `TullyProblemOne`, `TullyProblemTwo`, `TullyProblemThree` — standard benchmark problems
- `AbInitio` — general atomistic model (Q-Chem interface)

## The five-object architecture

| Object | Role | Defined in |
|---|---|---|
| `Simulation` | Top-level container; holds settings, model, algorithm, initial state, and per-run time index `t_ind` | `simulation.py` |
| `Model` | Physical system: holds `constants` and a list of ingredients (callables) that define its Hamiltonians and initialization | `model.py` |
| `Algorithm` | Numerical recipe: holds three lists of tasks — `initialization_recipe`, `update_recipe`, `collect_recipe` | `algorithm.py` |
| `Constants` | Attribute-bag with `__setattr__` hook that re-runs a registered initializer whenever a constant changes after init | `constants.py` |
| `Data` | Trajectory-averaged output container; supports HDF5 and `.npz` saving, log capture, and incremental merging via `add_data` | `data.py` |

A simulation is run by handing a `Simulation` to one of the drivers in `qclab.dynamics`: `serial_driver`, `parallel_driver_multiprocessing`, or `parallel_driver_mpi`.

## The recipe / task / ingredient pattern

Three-layer split between what the physics is, what the algorithm does, and how each step is implemented:

- **Ingredients** (`ingredients.py`, plus model-specific ones) are functions `f(model, parameters, **kwargs)` that compute physical quantities. They live on the Model as `(name, callable)` tuples. A model declares which ingredients it provides; the same ingredient is reused across models. Ingredient names beginning with `_init_` are special: the Model calls them whenever any constant is changed.

- **Tasks** (`tasks/`) are functions `task(sim, state, parameters, **opts) -> (state, parameters)`. They read named entries from `state`/`parameters`, call ingredients via `sim.model.get(...)`, and write named entries back. Every task takes `*_name` keyword arguments for key rebinding via `functools.partial`.

- **Recipes** are plain Python lists of tasks on the Algorithm class. `Algorithm.execute_recipe` iterates the list and threads `(state, parameters)` through each call.

## The dynamics core

`qclab.dynamics.run_dynamics`:
1. On `t_ind == 0`, run `initialization_recipe`
2. At every collect step (`t_ind % dt_collect_n == 0`), run `collect_recipe` and call `data.add_output_to_data_dict`
3. On every step, run `update_recipe`

Drivers build per-batch `state`/`parameters` dicts, run `run_dynamics`, and merge `Data` objects via trajectory-count-weighted `Data.add_data`.

## Numerical kernels

`functions.py` collects low-level math: complex-coordinate conversions (`z_to_q`, `qp_to_z`, `dqdp_to_dzc`, etc.), batched matrix-vector helpers, RK4 sub-steps, JIT'd kernels, the sparse-gradient inner product `calc_sparse_inner_product`, gauge fixing, and `numerical_fssh_hop`. Hot loops are `@njit`-decorated.

Optional dependencies are handled through `qclab.utils` with `DISABLE_NUMBA`, `DISABLE_H5PY`, `DISABLE_ASE` flags.

## Module map

```
src/qclab/
├── __init__.py               # Top-level imports and version
├── simulation.py             # Simulation class
├── model.py                  # Model base class and ingredient lookup
├── algorithm.py              # Algorithm base class and recipe executor
├── constants.py              # Constants attribute-bag with change hook
├── data.py                   # Data: collection, merging, HDF5/npz I/O
├── utils.py                  # JIT shims, in-memory logging, optional-dep flags
├── numerical_constants.py    # SMALL, finite-difference delta, unit conversions
├── ingredients.py            # Reusable model ingredients
├── functions.py              # Low-level numerics, JIT kernels, gauge fixing
├── algorithms/               # MeanField, FSSH (and ab initio variants)
├── dynamics/                 # run_dynamics core + serial/MP/MPI drivers
├── models/                   # Spin-boson, Holstein, FMO, Tully I/II/III, AbInitio
├── tasks/                    # Initialization, update, and collect tasks
└── interfaces/               # Q-Chem ab initio interface
```
