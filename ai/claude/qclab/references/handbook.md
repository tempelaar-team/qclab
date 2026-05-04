# QC Lab — Practical Handbook

Concrete recipes for getting things done, with code verified against QC Lab v1.1.1.

## Table of contents

- Installing QC Lab
- Running your first simulation
- Reading the Data object
- Adding a custom collect task
- Doing a parameter scan
- Implementing a new model from scratch
- Common pitfalls and how to avoid them

---

## Installing QC Lab

```bash
pip install qclab
```

Pulls numpy, tqdm, h5py, and numba automatically. For a minimal install without HDF5/JIT:

```bash
pip install qclab --no-deps
pip install numpy tqdm
```

## Running your first simulation

The pattern is always: build a Simulation, attach a Model and Algorithm, set the initial wavefunction, hand it to a driver.

```python
import numpy as np
from qclab import Simulation
from qclab.models import SpinBoson
from qclab.algorithms import MeanField
from qclab.dynamics import serial_driver

sim = Simulation({
    "tmax":        10.0,
    "dt_update":    0.01,
    "dt_collect":   0.1,
    "num_trajs":  100,
    "batch_size":  50,
    "progress_bar": False,
})
sim.model     = SpinBoson()
sim.algorithm = MeanField()

# Initial diabatic wavefunction (localised on site 0)
wf0 = np.zeros(sim.model.constants.num_quantum_states, dtype=complex)
wf0[0] = 1.0
sim.initial_state["wf_db"] = wf0

data = serial_driver(sim)
```

Substitute any model (HolsteinLattice, FMOComplex, TullyProblemOne, etc.) and any algorithm (FewestSwitchesSurfaceHopping, etc.). Nothing else changes.

## Reading the Data object

`serial_driver` returns a Data instance whose `data_dict` holds trajectory-averaged outputs:

```python
print(sorted(data.data_dict.keys()))
# ['classical_energy', 'dm_db', 'norm_factor', 'seed', 't']

t  = data.data_dict["t"]                       # shape (n_collect,)
dm = data.data_dict["dm_db"]                   # shape (n_collect, N, N)
pop_0 = dm[:, 0, 0].real                       # population on diabat 0
pop_1 = dm[:, 1, 1].real                       # population on diabat 1
q_e = data.data_dict["quantum_energy"]         # shape (n_collect,)
c_e = data.data_dict["classical_energy"]       # shape (n_collect,)
```

Save and reload:

```python
data.save("results.h5")
loaded = qclab.Data().load("results.h5")
```

Data objects support incremental merging: `data1.add_data(data2)`.

## Adding a custom observable

In QC Lab, computation and collection are separated into different tasks: **update tasks** compute values, **collect tasks** copy them into `state["output_dict"]`. This separation matters because update and collect recipes run at different frequencies.

Here's the correct pattern for adding a new observable (e.g., mean position):

```python
import numpy as np
from qclab.functions import z_to_q

# --- UPDATE TASK: does the computation ---
def update_mean_position(sim, state, parameters,
                         *, z_name="z", mean_position_name="mean_position"):
    """Compute mean real-space position from complex coordinates."""
    z = state[z_name]
    batch_size = sim.settings.batch_size
    m = sim.model.constants.classical_coordinate_mass[np.newaxis, :]
    h = sim.model.constants.classical_coordinate_weight[np.newaxis, :]
    q = z_to_q(z, m, h)                              # shape (B, C)
    state[mean_position_name] = q.mean(axis=1)        # shape (B,)
    return state, parameters

# --- COLLECT TASK: only copies to output_dict ---
def collect_mean_position(sim, state, parameters,
                          *, mean_position_name="mean_position",
                          mean_position_output_name="mean_position"):
    """Copy mean position into output_dict for recording."""
    state["output_dict"][mean_position_output_name] = state[mean_position_name]
    return state, parameters

# --- Append AFTER algorithm instantiation ---
# The update task goes in BOTH initialization_recipe (so the value exists
# at t=0 before the first collect) AND update_recipe (every step).
sim.algorithm.initialization_recipe = (
    sim.algorithm.initialization_recipe + [update_mean_position]
)
sim.algorithm.update_recipe = (
    sim.algorithm.update_recipe + [update_mean_position]
)
sim.algorithm.collect_recipe = (
    sim.algorithm.collect_recipe + [collect_mean_position]
)
```

Key points:
- **Imports at module level**, not inside task functions.
- **`batch_size` from `sim.settings.batch_size`**, not from `len(z)`.
- **Every state key is a `*_name` kwarg** with a sensible default.
- Anything in `state["output_dict"]` shows up as a new key in `data.data_dict`. The shape must be `(batch_size, *)` — the leading axis is averaged over trajectories.
- Always use `sim.algorithm.collect_recipe = sim.algorithm.collect_recipe + [my_task]` after the algorithm is instantiated. Do NOT mutate the class attribute in-place with `.append()` before instantiation — `Algorithm.__init__` deep-copies the class attribute, so in-place changes can be lost.

## Doing a parameter scan

A parameter scan is a Python loop that builds a fresh Simulation for each value:

```python
import numpy as np
from qclab import Simulation
from qclab.models import TullyProblemOne
from qclab.algorithms import FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver

momenta = np.array([5, 10, 15, 20, 25, 30], dtype=float)
results = np.zeros_like(momenta)

for i, p0 in enumerate(momenta):
    sim = Simulation({
        "tmax":        2.0 * 30.0 * 2000.0 / p0,
        "dt_update":   2.0,
        "dt_collect":  1000.0,
        "num_trajs":  150,
        "batch_size": 150,
        "progress_bar": False,
    })
    sim.model = TullyProblemOne({
        "init_position": -15.0,
        "init_momentum":  p0,
        "mass":         2000.0,
    })
    sim.algorithm = FewestSwitchesSurfaceHopping({
        "gauge_fixing": "phase_der_couple",
    })
    wf0 = np.array([1.0+0j, 0.0+0j])
    sim.initial_state["wf_db"] = wf0

    data = serial_driver(sim)
    results[i] = data.data_dict["dm_db"][-1, 0, 0].real
```

## Modifying an existing model (preferred approach)

Before writing a model from scratch, consider whether you can achieve what you need by subclassing an existing model and changing its constants. The stock ingredients are parametric — they compute physics based on the constants you give them.

For example, a spin-boson model with a **Debye spectral density** instead of the default flat-tail distribution does NOT need new ingredients. The stock `h_qc_diagonal_linear` ingredient works with any `diagonal_linear_coupling` array, and `h_c_harmonic` works with any `harmonic_frequency` array. You just need to change how those arrays are computed:

```python
import numpy as np
from qclab.models import SpinBoson

class SpinBosonDebye(SpinBoson):
    """Spin-boson with Debye spectral density J(w) = 2*l_reorg*w_D*w / (w^2 + w_D^2)."""

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        # Add the Debye cutoff frequency to the default constants
        self.default_constants = {
            **SpinBoson({}).default_constants,
            "w_D": 0.5,   # Debye cutoff frequency
        }
        # Let SpinBoson.__init__ handle the rest
        super().__init__({**self.default_constants, **(constants or {})})

    def _init_h_c(self, parameters, **kwargs):
        """Override frequency distribution to use Debye spectral density."""
        A = self.constants.A
        w_D = self.constants.w_D
        l_reorg = self.constants.l_reorg

        # Sample frequencies from Debye distribution
        w_max = 10.0 * w_D
        w = np.linspace(w_max / A, w_max, A)
        dw = w[1] - w[0]

        # Debye spectral density
        J_w = 2.0 * l_reorg * w_D * w / (w**2 + w_D**2)

        self.constants.harmonic_frequency = w
        self.constants.classical_coordinate_weight = w.copy()

        # diagonal_linear_coupling from spectral density
        coupling = np.sqrt(2.0 * J_w * dw / np.pi) * np.eye(
            self.constants.num_quantum_states
        )
        # ... set self.constants.diagonal_linear_coupling appropriately
```

This is the right approach whenever the functional form of the Hamiltonian stays the same and only the distribution of parameters changes. Only write new ingredients when the functional form itself is different (e.g., off-diagonal coupling when only diagonal exists, or a position-dependent coupling when only linear coupling exists).

## Implementing a new model from scratch

A self-contained model file doesn't need to live inside `src/qclab/`. Subclass `qclab.model.Model` and follow the conventions.

```python
import numpy as np
import qclab
from qclab import Simulation
from qclab.model import Model
from qclab import ingredients
from qclab import functions
from qclab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver


# ---------- novel ingredient(s) -----------------------------------------
def h_qc_my_form(model, parameters, **kwargs):
    """My new quantum-classical Hamiltonian.

    .. rubric:: Keyword Args
    z : ndarray, shape (B, C), complex128

    .. rubric:: Model Constants
    my_coupling : float

    .. rubric:: Returns
    h_qc : ndarray, shape (B, N, N), complex128
    """
    z = kwargs["z"]
    batch_size = len(z)
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    q = functions.z_to_q(z, m, h)
    coupling = model.constants.my_coupling

    h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
    # ... fill in the matrix elements ...
    return h_qc


def dh_qc_dzc_my_form(model, parameters, **kwargs):
    """Sparse z*-derivative of h_qc_my_form.

    Returns (inds, mels, shape) for a (B, C, N, N) gradient.
    """
    z = kwargs["z"]
    batch_size = len(z)
    # ... build a dense (B, C, N, N) array, then ...
    inds = np.where(dense != 0)
    mels = dense[inds]
    shape = dense.shape
    return inds, mels, shape


# ---------- the Model class ---------------------------------------------
class MyModel(Model):
    """One-line description.

    Reference: Author. Journal year, vol, pages.
    https://doi.org/...
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT":         1.0,
            "my_coupling": 0.5,
        }
        super().__init__(self.default_constants, constants)
        self.update_h_q       = False
        self.update_dh_qc_dzc = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states     = 2
        self.constants.num_classical_coordinates = 2
        self.constants.classical_coordinate_mass = np.ones(2)
        self.constants.harmonic_frequency        = np.ones(2)
        self.constants.classical_coordinate_weight = (
            self.constants.harmonic_frequency.copy()
        )

    def _init_h_q(self, parameters, **kwargs):
        self.constants.two_level_00 = 1.5
        self.constants.two_level_11 = -1.5
        self.constants.two_level_01_re = 0.0
        self.constants.two_level_01_im = 0.0

    ingredients = [
        ("h_q",            ingredients.h_q_two_level),
        ("h_qc",           h_qc_my_form),
        ("h_c",            ingredients.h_c_harmonic),
        ("dh_qc_dzc",      dh_qc_dzc_my_form),
        ("dh_c_dzc",       ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_wigner_harmonic),
        ("hop",            ingredients.hop_harmonic),
        ("_init_h_q",      _init_h_q),
        ("_init_model",    _init_model),
    ]


# ---------- run it -------------------------------------------------------
sim = Simulation({"tmax": 20, "dt_update": 0.01, "dt_collect": 0.1,
                  "num_trajs": 200, "batch_size": 100, "progress_bar": False})
sim.model = MyModel()
sim.algorithm = MeanField()
wf0 = np.array([1.0+0j, 0.0+0j])
sim.initial_state["wf_db"] = wf0
data = serial_driver(sim)
```

**Checklist for new models:**
- `_init_model` sets `num_quantum_states`, `num_classical_coordinates`, `classical_coordinate_mass`, `classical_coordinate_weight`
- Novel ingredients are vectorized over batch axis
- Sparse `dh_qc_dzc` returns `(inds, mels, shape)` in that order
- `update_h_q` and `update_dh_qc_dzc` are set correctly

## Common pitfalls and how to avoid them

**"Variable in initial_state is not a numpy.ndarray, skipping initialization."**
You set `sim.initial_state["wf_db"]` to a Python list. Wrap in `np.array(..., dtype=complex)`.

**Trace of dm_db is not 1.**
Either the wavefunction wasn't normalized, or you're looking at a single trajectory. Check `np.trace(data.data_dict["dm_db"][0])` at t=0.

**Energy drift much larger than 1e-3.**
Reduce `dt_update`. RK4 is fourth-order — halving the step reduces drift by ~16x. For a harmonic bath, `dt_update` of order `0.01 / max(harmonic_frequency)` is usually safe.

**Frustrated hops dominate the FSSH dynamics.**
This is physical, not a bug. When `hop_harmonic` can't find a real solution, the trajectory stays on its current surface. Check that the init sampler is generating coordinates with sensible energy.

**"Cannot save ... with unsupported type" from Data.save.**
You wrote a non-array into `state["output_dict"]`. Convert to `np.array` first.

**FSSH momentum-scan reproduces the wrong figure.**
Wavepacket started too close to the coupling region. For Tully 1, `init_position = -15.0` is standard.

**Model with `update_dh_qc_dzc = False` gives wrong gradients after changing a constant.**
The gradient is cached, but the Constants change hook only re-fires `_init_*` methods — it doesn't invalidate the cache. If a constant affects the gradient, set `update_dh_qc_dzc = True`.

**Custom collect task not in data.data_dict.**
You probably mutated `collect_recipe` in-place before instantiation. Always use: `sim.algorithm.collect_recipe = sim.algorithm.collect_recipe + [my_task]` after building the algorithm.

**Using `gauge_fixing: "phase_der_couple"` on a real-valued problem.**
The `"phase_der_couple"` gauge fixing is only needed when the Hamiltonian or derivative couplings are complex-valued (e.g., models with magnetic fields or complex hopping). For real-valued problems like Tully models or standard spin-boson, the default `"sign_overlap"` is correct and sufficient. Using `"phase_der_couple"` unnecessarily adds overhead without benefit.

**Writing new ingredients when modifying constants would suffice.**
Before writing a new ingredient, check whether you can achieve the same result by subclassing the existing model and changing the constants in `_init_h_c`, `_init_h_qc`, etc. The stock ingredients like `h_qc_diagonal_linear` and `h_c_harmonic` are parametric — they work with any constants you provide. Only write a new ingredient when the functional form of the physics is genuinely different.
