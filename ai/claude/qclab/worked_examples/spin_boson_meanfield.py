"""
Minimum-viable QC Lab simulation: spin-boson model with mean-field dynamics.

Runs 200 trajectories of the spin-boson model (two-level system coupled to
a harmonic bath) using Ehrenfest / mean-field dynamics, then plots the
diabatic populations as a function of time.

Usage:
    pip install qclab matplotlib
    python spin_boson_meanfield.py
"""

import numpy as np
from qclab import Simulation
from qclab.models import SpinBoson
from qclab.algorithms import MeanField
from qclab.dynamics import serial_driver

# ---------- build the simulation -----------------------------------------
sim = Simulation({
    "tmax":        10.0,
    "dt_update":    0.01,
    "dt_collect":   0.1,
    "num_trajs":   200,
    "batch_size":  100,
    "progress_bar": True,
})
sim.model = SpinBoson({
    "kBT":    1.0,
    "V":      0.5,      # electronic coupling
    "E":      0.5,      # site energy gap
    "A":      100,      # number of bath modes
    "W":      0.1,      # bandwidth
    "l_reorg": 0.005,   # reorganization energy
})
sim.algorithm = MeanField()

# Initial wavefunction: localised on diabat 0
wf0 = np.zeros(sim.model.constants.num_quantum_states, dtype=complex)
wf0[0] = 1.0
sim.initial_state["wf_db"] = wf0

# ---------- run -----------------------------------------------------------
data = serial_driver(sim)

# ---------- extract results -----------------------------------------------
t     = data.data_dict["t"]
dm    = data.data_dict["dm_db"]        # shape (n_collect, N, N)
pop_0 = dm[:, 0, 0].real              # population on diabat 0
pop_1 = dm[:, 1, 1].real              # population on diabat 1

print(f"Final populations: site 0 = {pop_0[-1]:.4f}, site 1 = {pop_1[-1]:.4f}")
print(f"Trace at t=0: {np.trace(dm[0]).real:.6f}")

# ---------- plot (optional) -----------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(t, pop_0, label="diabat 0")
    ax.plot(t, pop_1, label="diabat 1")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("Spin-Boson Mean-Field Dynamics")
    ax.legend()
    fig.savefig("spin_boson_meanfield.png", dpi=150)
    print("Plot saved to spin_boson_meanfield.png")
except ImportError:
    print("matplotlib not available — skipping plot")
