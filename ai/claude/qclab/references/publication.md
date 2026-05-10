# QC Lab — Publication Reference

This file is the canonical reference for questions about the QC Lab paper
and its supporting information. The published article (open access,
Apache 2.0) is hosted on the JCTC website:

- Main article: <https://pubs.acs.org/doi/10.1021/acs.jctc.5c01818>
- Supporting information PDF: <https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5c01818/suppl_file/ct5c01818_si_001.pdf>
- DOI: [10.1021/acs.jctc.5c01818](https://doi.org/10.1021/acs.jctc.5c01818)

When the user asks something that goes beyond this page — exact wording,
a specific equation, a reference number, a figure caption — point them at
the JCTC article and SI links above. This file should already cover the
common cases.

## Table of contents

- Citation
- Abstract
- What the paper actually argues
- The complex-classical coordinate formalism
- Key equations (from the SI)
- Architecture-to-code mapping
- Algorithms and models discussed in the paper
- Worked examples in the paper
- Numerical schemes and performance notes
- Ab initio interface
- Position relative to other software
- Important external references
- When to crack open the PDFs

---

## Citation

> Krotz, A.; Garzón-Ramírez, A. J.; Byrd, E.; Miyazaki, K.; Tempelaar, R.
> **QC Lab: A Python Package for Quantum–Classical Dynamics.**
> *J. Chem. Theory Comput.* **2026**, *22*, 3144–3152.
> DOI: [10.1021/acs.jctc.5c01818](https://doi.org/10.1021/acs.jctc.5c01818)

Repository: [`tempelaar-team/qclab`](https://github.com/tempelaar-team/qclab).
Released version 1.1.1 archived at
[Zenodo 10.5281/zenodo.18964248](https://zenodo.org/doi/10.5281/zenodo.18964248).
License: Apache 2.0.

Corresponding author: Roel Tempelaar, Department of Chemistry,
Northwestern University, Evanston, IL 60208 (roel.tempelaar@northwestern.edu).
Funding: NSF grants 2145433 and 2513048; the Mark A. Ratner Postdoctoral
Fellowship and the International Institute for Nanotechnology (K.M.).
The package was first described in part in the Ph.D. dissertation of
A. Krotz (Northwestern University, 2025).

## Abstract (paraphrased)

QC Lab is an open-source Python package for trajectory-based
quantum-classical (QC) dynamics. Its design goal is to promote the
development of new QC algorithms and their application to a wide range
of model problems. The package is organized so that **algorithms and
models are cross-compatible**: each is decomposed into reusable units —
"tasks" for algorithms, "ingredients" for models — that can be swapped
in and out without rewriting the rest of the simulation. The paper
introduces the first stable release (v1.0, October 2025; v1.1
February 2026 added an ab initio Q-Chem interface) and describes its
design philosophy.

## What the paper actually argues

1. The single biggest source of friction in QC software is that
   algorithms and models are coupled. Adding a new model often forces
   tweaks to every algorithm and vice versa.
2. The fix is to define a **minimum model contract** — three Hamiltonians
   (quantum `H_q`, classical `H_c`, quantum-classical `H_qc`) — so that
   any algorithm can run on any model that satisfies the contract.
3. Models can optionally supply additional ingredients (analytic
   gradients, samplers, hop rules, …) that **override the default
   numerical procedures** in tasks. This is the performance escape
   hatch: contract-conformant out of the box, fast when ingredients
   are available.
4. Classical coordinates are stored as a **single complex coordinate
   `z`** (combining position and momentum) rather than as `(q, p)`. This
   makes arbitrary unitary basis transformations of the classical
   subsystem natural — real-space, reciprocal-space, normal modes, etc.
5. The package is a Python-first citizen: pip-installable, NumPy-vectorized,
   optional Numba JIT, and serial / multiprocessing / MPI dynamics drivers.

## The complex-classical coordinate formalism

The classical coordinate used everywhere in QC Lab is

```
z_ξ = sqrt(m_ξ * h_ξ / 2) * q_ξ + i * sqrt(1 / (2 * m_ξ * h_ξ)) * p_ξ
```

where `ξ` indexes classical coordinates in the chosen basis, `m_ξ` is
that coordinate's mass and `h_ξ` is a per-coordinate weight that
modulates the relative weighting of position and momentum. `h_ξ` is
formally arbitrary, but **for harmonic potentials with characteristic
frequency `ω_ξ` the natural choice is `h_ξ = ω_ξ`**. The real and
imaginary parts of `z_ξ` correspond to position and momentum only when
the basis is itself physical; under a unitary basis transformation the
parts mix and `Re z`, `Im z` no longer have direct (q, p) meaning.

This is the formalism of Miyazaki, Krotz & Tempelaar
(*JCTC* **2024**, *20*, 6500–6509), and it is the source of the
QC Lab convention that all forces, gradients, and integrators are
expressed in terms of `z` and its conjugate `z*`. Helper conversions
between `(q, p)` and `z` live in `qclab.functions`
(`z_to_q`, `z_to_p`, `qp_to_z`, `dqdp_to_dzc`, `dzdzc_to_dqdp`).

## Key equations (from the supporting information)

The SI is short — two technical sections, six equations.

**Mean-field dynamics.** With the wavefunction `|Ψ⟩` evolving under

```
i ℏ |Ψ̇⟩ = (Ĥ_q + Ĥ_{q-c}({z_ξ})) |Ψ⟩            (S2)
```

the classical coordinates obey complex Hamilton equations with a
mean-field feedback term:

```
ż_ξ = -i ∂/∂z*_ξ [ H_c({z_ξ}) + ⟨Ψ| Ĥ_{q-c}({z_ξ}) |Ψ⟩ ]   (S3)
```

**Fewest-Switches Surface Hopping.** Eigenstates `|α⟩` of the full
electronic Hamiltonian satisfy

```
(Ĥ_q + Ĥ_{q-c}({z_n})) |α⟩ = ε_α |α⟩            (S4)
```

The active surface `a` replaces `Ψ` in the EOM:

```
ż_ξ = -i ∂/∂z*_ξ [ H_c({z_ξ}) + ⟨a|Ĥ_{q-c}({z_ξ})|a⟩ ]    (S5)
```

The instantaneous switching probability between eigenstates `α` and `β`
is

```
P_{a:α→β} = 2 Re ⟨α| ∂_t β⟩ (A_β / A_α) Δt        (S6)
```

with `A_α` the expansion coefficients of the mean-field wavefunction
`|Ψ⟩ = Σ_α A_α |α⟩` (S7).

After a hop, the classical coordinate is shifted to conserve total
energy:

```
z'_ξ = z_ξ - i γ ⟨α̃| ∂/∂z*_ξ |β̃⟩                (S8)
```

with `γ` solving the energy-conservation condition. `α̃` and `β̃` are
real-valued projections of the (possibly complex) eigenvectors — a
global gauge fix appropriate for topologically trivial problems. For
the nontrivial case see Krotz & Tempelaar, *PRA* **2024**, *109*, 032210.

**On frustrated hops.** QC Lab's default FSSH **does not** invert the
classical momentum on a frustrated hop (the convention of Müller & Stock,
*JCP* **1997**). The paper shows in Figure 4 how to re-enable momentum
reversal by appending a `reverse_momenta` task to `update_recipe` —
the task acts in a physical basis by taking the complex conjugate of `z`.

## Architecture-to-code mapping

The capitalised concepts in the paper are concrete Python objects:

| Paper concept | Code |
|---|---|
| Simulation | `qclab.Simulation` (settings + model + algorithm + initial state) |
| Algorithm | `Algorithm` subclass holding three Recipes |
| Recipe | Plain `list` of Tasks on the Algorithm (`initialization_recipe`, `update_recipe`, `collect_recipe`) |
| Task | Function `task(sim, state, parameters, **opts) -> (state, parameters)` with `*_name` keyword args for state-key rebinding |
| Model | `Model` subclass holding `constants` plus an `ingredients = [(name, callable), …]` list |
| Ingredient | Function `f(model, parameters, **kwargs)` registered on a Model under a standard slot name |
| State | Per-batch dict built by the Dynamics Driver (holds `z`, wavefunctions, RK4 intermediates, …) |
| Parameters | Per-batch dict relayed to the Model (e.g., current time for time-dependent Hamiltonians) |
| Dynamics Driver | `serial_driver`, `parallel_driver_multiprocessing`, `parallel_driver_mpi` in `qclab.dynamics` |
| Data | `qclab.Data` — output dict, RNG seeds, log; HDF5 / npz I/O; trajectory-weighted merging via `add_data` |
| Calculator (ab initio) | Special ingredient (`ab_initio_property_calculator`) called from the algorithm to update an ab initio property dict |
| Interface (ab initio) | The bridge between the Calculator and an external QM code (Q-Chem v6 via ASE) |

The override pattern works as follows. A task that needs, say, the
classical force first asks the model `sim.model.get("dh_c_dzc")`. If
the model supplies the analytic ingredient, the task uses it; otherwise
the task falls back to a finite-difference computation. This is what
keeps "contract-conformant" and "fast" from being mutually exclusive.

## Algorithms and models discussed in the paper

**Algorithms.**
- Mean-field / Ehrenfest dynamics (Ehrenfest, *Z. Phys.* **1927**;
  Tully, *Faraday Discuss.* **1998**).
- Fewest-Switches Surface Hopping (Tully, *JCP* **1990**, *93*, 1061),
  implemented per Hammes-Schiffer & Tully (*JCP* **1994**, *101*, 4657)
  with the Müller–Stock no-momentum-reversal convention as default.
- Several gauge-fixing procedures enforcing parallel transport, plus
  flexible classical momentum rescaling for nontrivial gauge constraints
  — including FSSH on systems with geometric-phase effects and on
  topological materials (Krotz & Tempelaar, *PRA* **2024**, *109*, 032210).

**Models shipped.**
- Spin–boson (parameters of Tempelaar & Reichman, *JCP* **2018**, *148*, 102309).
- One-electron Holstein lattice (real space and reciprocal space).
- Fenna–Matthews–Olson light-harvesting complex (parameters of
  Mulvihill et al., *JCP* **2021**, *154*, 204109).
- Tully Problems I, II, III (Tully, *JCP* **1990**).
- General atomistic / ab initio model (Q-Chem interface).

## Worked examples in the paper

**Figure 1.** A complete FSSH simulation on the spin–boson model in a
single screen of Python — including a programmatic adjustment of the
reorganization energy on the model's `constants` before the simulation
is handed to the dynamics driver. This is the canonical "what does a
QC Lab script look like" example.

**Figure 4.** Adding a `reverse_momenta` task to `update_recipe` to
toggle frustrated-hop momentum reversal back on. The point of the
example is to show that user-level customisation is just standard
Python list manipulation on the recipe.

**Figures 5 & 6.** An ab initio FSSH simulation of the photoisomerization
of **protonated formaldimine**, initiated in S2, run with TD-DFT at the
**cc-pVDZ / ωB97X-D** level via the Q-Chem v6 interface, following
Tapavicza, Tavernelli & Rothlisberger (*PRL* **2007**, *98*, 023001).
Figure 6 shows the time-dependent state populations averaged over **158
trajectories**.

## Numerical schemes and performance notes

- **Diabatic models** integrate the classical coordinates with 4th-order
  Runge–Kutta. RK4 sub-step intermediates appear in the state dict as
  `z_1`, `z_2`, `z_3`, `z_rk4_k1`, `z_rk4_k2`, `z_rk4_k3`.
- **Ab initio models** use velocity Verlet by default. Switching the
  integrator is just swapping a few tasks in the update recipe.
- Vectorisation across **trajectory batches** (the leading axis `B`) is
  pervasive. Most built-in tasks and ingredients are already vectorised,
  and decorators (`vectorize_ingredient`) are available for new
  ingredients that aren't.
- **Numba JIT** is optional. Hot loops in `functions.py` are
  `@njit`-decorated through the `qclab.utils` shim so the package still
  runs without Numba.
- **Three drivers** in `qclab.dynamics`: serial (desktop / debugging),
  multiprocessing (single node), MPI (clusters; `mpi4py`).
- Required dependencies: NumPy plus standard library (`logging`,
  `multiprocessing`, `functools`). Optional: Numba, h5py, mpi4py, ASE.

## Ab initio interface

The ab initio model uses a special **Calculator** ingredient that is
called directly from the algorithm. The Calculator's job is to update
a single dictionary of ab initio properties — energies, gradients,
derivative couplings — by invoking the external electronic-structure
code, batching as much as possible to minimise expensive QM calls. The
Calculator talks to the QM code through an **Interface**, also part of
QC Lab. Currently shipped: one Calculator and one Interface backed by
**Q-Chem v6**, accessed through ASE.

Because ab initio calculations lack a global diabatic basis, the ab
initio model also supplies a **derivative-coupling** ingredient
(`derivative_coupling_dzc`); together with the three Hamiltonians this
fully specifies the system in a coordinate-dependent quantum basis.
The reformatted ab-initio Ehrenfest and FSSH algorithms ship as
`MeanFieldAbInitio` and `FewestSwitchesSurfaceHoppingAbInitio`.

## Position relative to other software

The paper situates QC Lab among existing nonadiabatic-dynamics codes —
Newton-X, SHARC, NEXMD, pyUNIxMD, JADE, Hefei-NAMD, PYXAID, Libra —
emphasising the Python ecosystem (PySCF, HOOMD-blue, TenPy, ASE) and
the pip-installable distribution model. The complex-classical
coordinate convention and the strict task/ingredient decomposition are
the primary differentiators.

The paper notes that QC Lab has already been used in research-grade
applications, specifically the optical line widths and spin-valley
depolarization in monolayer transition-metal dichalcogenides (Krotz &
Tempelaar, arXiv:2505.06953, 2025), where size-converged sampling
resolutions were reached.

## Important external references

The paper's bibliography is the right place to send users for primary
sources. The references that come up most often in QC Lab questions:

- **Miyazaki, Krotz, Tempelaar.** *JCTC* **2024**, *20*, 6500–6509 —
  the foundational paper for the complex-classical coordinate formalism
  that QC Lab implements. Notation source for the SI.
- **Krotz, Provazza, Tempelaar.** *JCP* **2021**, *154*, 224101 —
  reciprocal-space MQC dynamics.
- **Krotz, Tempelaar.** *JCP* **2022**, *156*, 024105 —
  reciprocal-space surface hopping.
- **Krotz, Tempelaar.** *PRA* **2024**, *109*, 032210 — geometric phase
  / gauge fixing in nonadiabatic dynamics.
- **Krotz, Tempelaar.** *JCP* **2024**, *161*, 044117 — MoS2 optical
  linewidths.
- **Krotz, Tempelaar.** arXiv:2505.06953 (2025) — valley depolarisation,
  the application that drove much of QC Lab's development.
- **Tully.** *JCP* **1990**, *93*, 1061–1071 — original FSSH paper and
  Tully Problems I/II/III.
- **Tempelaar, Reichman.** *JCP* **2018**, *148*, 102309 — FSSH for
  coherences; source of the spin–boson parameters in Figure 1.
- **Hammes-Schiffer, Tully.** *JCP* **1994**, *101*, 4657–4667 — FSSH
  reference implementation.
- **Müller, Stock.** *JCP* **1997**, *107*, 6230–6245 — convention of
  not reversing momenta on frustrated hops (QC Lab default).
- **Mulvihill et al.** *JCP* **2021**, *154*, 204109 — FMO complex
  parametrisation shipped in QC Lab.
- **Tapavicza, Tavernelli, Rothlisberger.** *PRL* **2007**, *98*,
  023001 — basis for the protonated-formaldimine ab initio FSSH demo.

## When to point users at the JCTC article

Send users to the JCTC article when they ask about:

- a specific numbered equation, figure, or table,
- exact wording from the abstract, introduction, or discussion,
- a reference number (the paper's bibliography is the source),
- the precise statement of the gauge-fixing procedure or the FSSH
  rescaling rule (the SI is the authoritative source),
- timing, parameter values, or settings used in Figures 5–6
  (protonated formaldimine ab initio run),
- anything where this summary would be a paraphrase and the user
  needs the primary text.

The article and SI are open access. Direct links:

- <https://pubs.acs.org/doi/10.1021/acs.jctc.5c01818>
- <https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5c01818/suppl_file/ct5c01818_si_001.pdf>

If web fetching is available in the environment and the user wants the
exact text quoted, use `WebFetch` on those URLs rather than guessing
from this summary.
