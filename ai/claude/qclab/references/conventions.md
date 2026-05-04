# QC Lab — Convention Reference

Quick lookup tables for every standard name in QC Lab.

## Table of contents

- 3.1 Standard ingredient slots
- 3.2 Standard state dict keys
- 3.3 Standard model constants
- 3.4 Standard algorithm settings
- 3.5 Local variable names

---

## 3.1 Standard ingredient slots

These are the names that algorithms use when calling `sim.model.get(...)`. The first element of each `(name, callable)` tuple in a model's `ingredients` list must match one of these (or be an `_init_*` initializer). Signature is always `f(model, parameters, **kwargs)`.

| Slot | Required kwargs | Returns | Used by |
|---|---|---|---|
| `h_q` | `batch_size` | `(B, N, N)` complex Hamiltonian | every algorithm |
| `h_qc` | `z` | `(B, N, N)` complex Hamiltonian | every algorithm |
| `h_c` | `z` | `(B,)` real classical energy | mean-field, FSSH |
| `dh_qc_dzc` | `z` | sparse `(inds, mels, shape)` for `(B, C, N, N)` | every algorithm; falls back to finite differences if absent |
| `dh_c_dzc` | `z` | `(B, C)` complex gradient | every algorithm; falls back to finite differences if absent |
| `init_classical` | `seed` | `(B, C)` complex initial coordinates | every algorithm; falls back to MCMC if absent |
| `hop` | `z`, `resc_dir_z`, `eigval_diff` | `(shift, hop_bool)` | FSSH only |
| `derivative_coupling_dzc` | `z` | `(B, C, N, N)` complex | ab initio only |
| `gauge_field_force` | `z`, `state_ind` | `(B, C)` complex | optional, when `use_gauge_field_force == True` |
| `ab_initio_property_calculator` | `property_dict`, `traj_ind` | dict of energies/gradients/couplings | ab initio only |

### _init_* initializer "ingredients"

Called by `Model.initialize_constants` whenever a constant changes. Names always start with underscore:

- `_init_model` — derives sizes (`num_quantum_states`, `num_classical_coordinates`) and per-coordinate metadata (`classical_coordinate_mass`, `classical_coordinate_weight`)
- `_init_h_q` — derives constants for the `h_q` ingredient
- `_init_h_qc` — derives constants for the `h_qc` ingredient
- `_init_h_c` — derives constants for the `h_c` ingredient

---

## 3.2 Standard state dict keys

Keys follow `lower_snake_case`. Shapes use B = batch_size, C = num_classical_coordinates, N = num_quantum_states.

### Trajectory bookkeeping

| Key | Shape / type | Meaning |
|---|---|---|
| `seed` | `(B,)` int | Per-trajectory random seed |
| `branch_ind` | `(B,)` int | FSSH branch index (deterministic mode) |
| `t` | `(B,)` float64 | Current time |
| `output_dict` | dict | Values to collect this step |
| `norm_factor` | scalar | Normalization (= batch_size) |

### Classical coordinates and RK4 intermediates

| Key | Shape / type | Meaning |
|---|---|---|
| `z` | `(B, C)` complex128 | Current classical coordinate |
| `z_1`, `z_2`, `z_3` | `(B, C)` complex128 | RK4 sub-step intermediates |
| `z_previous` | `(B, C)` complex128 | Previous timestep z |
| `z_rk4_k1`, `z_rk4_k2`, `z_rk4_k3` | `(B, C)` complex128 | RK4 slopes |

### Hamiltonian matrices

| Key | Shape / type | Meaning |
|---|---|---|
| `h_q` | `(B, N, N)` complex128 | Quantum Hamiltonian |
| `h_qc` | `(B, N, N)` complex128 | Quantum-classical Hamiltonian |
| `h_q_tot` | `(B, N, N)` complex128 | h_q + h_qc |
| `h_q_tot_previous` | `(B, N, N)` complex128 | Previous-step value |

### Forces

| Key | Shape / type | Meaning |
|---|---|---|
| `classical_force` | `(B, C)` complex128 | Force from dh_c_dzc |
| `quantum_classical_force` | `(B, C)` complex128 | Force from <wf\|dh_qc_dzc\|wf> |
| `*_force_previous` | as above | Previous-step value |

### Diagonalization output

| Key | Shape / type | Meaning |
|---|---|---|
| `eigvals` | `(B, N)` float64 | Eigenvalues of h_q_tot |
| `eigvecs` | `(B, N, N)` complex128 | Eigenvectors (columns are states) |
| `eigvecs_previous` | `(B, N, N)` complex128 | Previous-step eigenvectors |

### Wavefunctions

| Key | Shape / type | Meaning |
|---|---|---|
| `wf_db` | `(B, N)` complex128 | Wavefunction in diabatic basis |
| `wf_adb` | `(B, N)` complex128 | Wavefunction in adiabatic basis |
| `act_surf_wf` | `(B, N)` complex128 | Active-surface unit vector (FSSH) |

### FSSH-specific

| Key | Shape / type | Meaning |
|---|---|---|
| `act_surf` | `(B, N)` int | One-hot active surface |
| `act_surf_ind` | `(B,)` int | Active surface index |
| `act_surf_ind_0` | `(B,)` int | Initial active surface |
| `hop_prob` | `(B, N)` float64 | Hopping probabilities |
| `hop_prob_rand_vals` | `(B/branches, t)` float64 | Pre-drawn random numbers |
| `hop_ind` | `(H,)` int | Indices of hopping trajectories |
| `hop_dest` | `(H,)` int | Destination surfaces |
| `hop_bool` | `(B,)` bool | Whether each trajectory hops |
| `hop_pairs` | `(B, 2)` int | (initial, final) state pairs |
| `dm_adb_0` | `(B, N, N)` complex128 | Initial adiabatic density matrix |

### Density matrices and energies

| Key | Shape / type | Meaning |
|---|---|---|
| `dm_db` | `(B, N, N)` complex128 | Diabatic density matrix |
| `dm_adb` | `(B, N, N)` complex128 | Adiabatic density matrix |
| `classical_energy` | `(B,)` float64 | Per-trajectory classical energy |
| `quantum_energy` | `(B,)` float64 | Per-trajectory quantum energy |

### Ab initio extras

| Key | Shape / type | Meaning |
|---|---|---|
| `aip_excited_amplitudes` | varies | From property calculator |
| `derivative_coupling_dzc` | `(B, C, N, N)` complex128 | Derivative coupling tensor |
| `adb_connection` | `(B, N, N)` complex128 | Adiabatic connection matrix |

### Suffix conventions

- `_previous` — value from prior timestep
- `_0` — initial-time reference (e.g. `dm_adb_0`, `act_surf_ind_0`)
- `_ind` — integer index (e.g. `act_surf_ind`, `traj_ind`)
- `_name` — only for task keyword arguments whose value is a string key name

---

## 3.3 Standard model constants

### Sizes (always `num_` prefix)

- `num_quantum_states`, `num_classical_coordinates`, `num_atoms`

### Per-coordinate metadata

- `classical_coordinate_mass` `(C,)`, `classical_coordinate_weight` `(C,)`, `harmonic_frequency` `(C,)`

### Initial conditions (`init_` prefix)

- `init_position`, `init_momentum`

### Coupling constants (named after consuming ingredient)

- `diagonal_linear_coupling` — used by `h_qc_diagonal_linear`
- `nearest_neighbor_hopping_energy`, `nearest_neighbor_periodic` — used by `h_q_nearest_neighbor`
- `two_level_00`, `two_level_11`, `two_level_01_re`, `two_level_01_im` — used by `h_q_two_level`
- `coherent_state_displacement` — used by `init_classical_wigner_coherent_state`

### Atomistic/ab initio

- `atom_names`, `atom_masses`, `atom_positions`, `normal_mode`, `energy_offset`, `calculator_args`

### Numerical tuning knobs (`<consumer>_<knob>`)

- `numerical_fssh_hop_gamma_range`, `numerical_fssh_hop_max_iter`, `numerical_fssh_hop_num_points`, `numerical_fssh_hop_threshold`
- `dh_c_dzc_finite_difference_delta`

### User-facing physical constants (conventional symbols)

- `kBT`, `V`, `E`, `A`, `W`, `J`, `N`, `g`, `w`, `l_reorg`, `w_c`

---

## 3.4 Standard algorithm settings

| Setting | Type | Default | Used by |
|---|---|---|---|
| `tmax` | float | 10.0 | every simulation |
| `dt_update` | float | 0.001 | every simulation |
| `dt_collect` | float | 0.1 | every simulation |
| `num_trajs` | int | 100 | every simulation |
| `batch_size` | int | 25 | every simulation |
| `progress_bar` | bool | True | every simulation |
| `debug` | bool | False | gates expensive sanity checks |
| `fssh_deterministic` | bool | False | FSSH |
| `gauge_fixing` | str | "sign_overlap" | FSSH |
| `use_gauge_field_force` | bool | False | FSSH |
| `update_wf_adb_eig_num_substeps` | int | 10 | ab initio |
| `use_wf_overlaps_for_adb_connection` | bool | varies | ab initio |

Boolean flags start with `use_` or `is_`; mode strings are descriptive snake-case.

---

## 3.5 Local variable names allowed in physics code

- coordinates: `z`, `q`, `p`
- per-coordinate: `m` (mass), `h` (weight), `w` (frequency)
- thermal: `kBT`
- eigenpairs: `evec_i`, `evec_j`, `eval_i`, `eval_j`, `eigval_diff`
- sparse triple (always this order): `inds`, `mels`, `shape`
- sizes: `batch_size`, `num_classical_coordinates`, `num_quantum_states`, `num_trajs`, `num_branches`, `num_batches`
- indices: `traj_ind`, `state_ind`, `act_surf_ind`, `t_ind`, `init_state_ind`, `final_state_ind`
- hop algebra: `disc`, `gamma`, `shift`

Avoid one-letter names outside this list (except in `for i, j in ...` index loops).
