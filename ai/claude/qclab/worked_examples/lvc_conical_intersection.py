"""
Linear Vibronic Coupling (LVC) model with a conical intersection.

Demonstrates how to implement a new model from scratch: a two-state,
two-mode system where one mode (tuning) shifts the diabatic energies and
the other (coupling) creates an off-diagonal linear coupling that produces
a conical intersection.

This requires a novel h_qc ingredient (because the stock
`h_qc_diagonal_linear` only handles diagonal coupling) and a matching
`dh_qc_dzc` gradient.

Usage:
    pip install qclab matplotlib
    python lvc_conical_intersection.py
"""

import numpy as np
from qclab import Simulation
from qclab.model import Model
from qclab import ingredients
from qclab import functions
from qclab.algorithms import MeanField
from qclab.dynamics import serial_driver


# ---------- novel ingredients for off-diagonal linear coupling ------------

def h_qc_lvc(model, parameters, **kwargs):
    """Quantum-classical Hamiltonian for the LVC model.

    Mode 0 (tuning): diagonal coupling, shifts diabatic energies.
    Mode 1 (coupling): off-diagonal coupling, creates the conical intersection.

    .. rubric:: Keyword Args
    z : ndarray, shape (B, 2), complex128

    .. rubric:: Model Constants
    kappa : float — tuning-mode coupling strength
    lam : float — coupling-mode coupling strength

    .. rubric:: Returns
    h_qc : ndarray, shape (B, 2, 2), complex128
    """
    z = kwargs["z"]
    batch_size = len(z)
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    q = functions.z_to_q(z, m, h)        # (B, 2) real coordinates

    kappa = model.constants.kappa
    lam   = model.constants.lam

    q_tune = q[:, 0]     # tuning mode
    q_coup = q[:, 1]     # coupling mode

    h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
    # Diagonal: tuning mode shifts energies with opposite sign
    h_qc[:, 0, 0] =  kappa * q_tune
    h_qc[:, 1, 1] = -kappa * q_tune
    # Off-diagonal: coupling mode creates the CI
    h_qc[:, 0, 1] = lam * q_coup
    h_qc[:, 1, 0] = lam * q_coup
    return h_qc


def dh_qc_dzc_lvc(model, parameters, **kwargs):
    """Sparse z*-derivative of h_qc_lvc.

    Returns (inds, mels, shape) for a (B, C, N, N) gradient.
    The derivative w.r.t. z* involves the chain rule through z_to_q:
        dq/dz* = sqrt(m*h/2)
    """
    z = kwargs["z"]
    batch_size = len(z)
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight

    kappa = model.constants.kappa
    lam   = model.constants.lam

    # dq/dzc = sqrt(m*h/2)  (the conjugate derivative)
    dq_dzc = np.sqrt(m * h / 2.0)

    dense = np.zeros((batch_size, 2, 2, 2), dtype=complex)

    # d(h_qc)/dz*_0 (tuning mode)
    dense[:, 0, 0, 0] =  kappa * dq_dzc[0]
    dense[:, 0, 1, 1] = -kappa * dq_dzc[0]

    # d(h_qc)/dz*_1 (coupling mode)
    dense[:, 1, 0, 1] = lam * dq_dzc[1]
    dense[:, 1, 1, 0] = lam * dq_dzc[1]

    inds = np.where(dense != 0)
    mels = dense[inds]
    shape = dense.shape
    return inds, mels, shape


# ---------- the LVC Model class -------------------------------------------

class LVCModel(Model):
    """Two-state, two-mode Linear Vibronic Coupling model.

    A tuning mode (kappa coupling, diagonal) and a coupling mode
    (lambda coupling, off-diagonal) produce a conical intersection.

    Reference: Krotz et al. J. Chem. Theory Comput. 2024.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT":        0.5,
            "kappa":      0.5,       # tuning-mode coupling
            "lam":        0.3,       # coupling-mode coupling (off-diagonal)
            "w_tune":     1.0,       # tuning-mode frequency
            "w_coup":     1.0,       # coupling-mode frequency
            "E_gap":      1.0,       # diabatic energy gap (half-gap)
        }
        super().__init__(self.default_constants, constants)
        # h_q is z-independent; dh_qc_dzc is also z-independent
        # (linear coupling -> constant gradient)
        self.update_h_q       = False
        self.update_dh_qc_dzc = False

    def _init_model(self, parameters, **kwargs):
        """Derive sizes and per-coordinate metadata."""
        self.constants.num_quantum_states        = 2
        self.constants.num_classical_coordinates = 2
        self.constants.classical_coordinate_mass = np.array([1.0, 1.0])
        self.constants.harmonic_frequency = np.array([
            self.constants.w_tune,
            self.constants.w_coup,
        ])
        self.constants.classical_coordinate_weight = (
            self.constants.harmonic_frequency.copy()
        )

    def _init_h_q(self, parameters, **kwargs):
        """Set up the two-level quantum Hamiltonian constants."""
        E = self.constants.E_gap
        self.constants.two_level_00    =  E
        self.constants.two_level_11    = -E
        self.constants.two_level_01_re =  0.0
        self.constants.two_level_01_im =  0.0

    ingredients = [
        ("h_q",            ingredients.h_q_two_level),        # stock
        ("h_qc",           h_qc_lvc),                        # novel
        ("h_c",            ingredients.h_c_harmonic),         # stock
        ("dh_qc_dzc",      dh_qc_dzc_lvc),                   # novel
        ("dh_c_dzc",       ingredients.dh_c_dzc_harmonic),    # stock
        ("init_classical", ingredients.init_classical_wigner_harmonic),
        ("hop",            ingredients.hop_harmonic),
        ("_init_h_q",      _init_h_q),
        ("_init_model",    _init_model),
    ]


# ---------- run the simulation -------------------------------------------

sim = Simulation({
    "tmax":        20.0,
    "dt_update":    0.01,
    "dt_collect":   0.1,
    "num_trajs":   200,
    "batch_size":  100,
    "progress_bar": True,
})
sim.model = LVCModel({
    "kBT":   0.5,
    "kappa": 0.5,
    "lam":   0.3,
    "E_gap": 1.0,
})
sim.algorithm = MeanField()

# Localise on upper diabat (state 0)
wf0 = np.array([1.0 + 0j, 0.0 + 0j])
sim.initial_state["wf_db"] = wf0

data = serial_driver(sim)

# ---------- extract results -----------------------------------------------
t     = data.data_dict["t"]
dm    = data.data_dict["dm_db"]
pop_0 = dm[:, 0, 0].real
pop_1 = dm[:, 1, 1].real

print(f"Final populations: state 0 = {pop_0[-1]:.4f}, state 1 = {pop_1[-1]:.4f}")
print(f"Trace at t=0: {np.trace(dm[0]).real:.6f}")

# ---------- plot (optional) -----------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(t, pop_0, label="diabat 0 (upper)")
    ax.plot(t, pop_1, label="diabat 1 (lower)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("LVC Conical Intersection — Mean-Field Dynamics")
    ax.legend()
    fig.savefig("lvc_conical_intersection.png", dpi=150)
    print("Plot saved to lvc_conical_intersection.png")
except ImportError:
    print("matplotlib not available — skipping plot")
