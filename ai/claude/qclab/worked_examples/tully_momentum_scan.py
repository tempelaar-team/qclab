"""
Tully Problem 1 momentum scan with a custom collect task.

Scans over initial momentum p0 for the simple avoided crossing (Tully 1)
with FSSH, and tracks transmission / reflection probabilities using a
custom collect task that classifies trajectories by final position and
active surface.

Usage:
    pip install qclab matplotlib
    python tully_momentum_scan.py
"""

import numpy as np
from qclab import Simulation
from qclab.models import TullyProblemOne
from qclab.algorithms import FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver
from qclab.functions import z_to_q


# ---------- update task: compute channel classification -------------------
def update_channel_classification(sim, state, parameters,
                                  *, z_name="z",
                                  act_surf_ind_name="act_surf_ind",
                                  channel_name="channel_classification"):
    """Classify each trajectory as transmit/reflect on upper/lower surface.

    Writes a dict of four (B,) arrays of 0s and 1s into state.
    """
    z = state[z_name]                                  # (B, 1) complex
    batch_size = sim.settings.batch_size
    m = sim.model.constants.classical_coordinate_mass[np.newaxis, :]
    h = sim.model.constants.classical_coordinate_weight[np.newaxis, :]
    q = z_to_q(z, m, h)[:, 0].real                    # (B,) position

    act = state[act_surf_ind_name]                     # (B,) int

    transmit = (q > 0).astype(float)
    reflect  = (q <= 0).astype(float)

    state["transmit_lower"] = transmit * (act == 0).astype(float)
    state["transmit_upper"] = transmit * (act == 1).astype(float)
    state["reflect_lower"]  = reflect  * (act == 0).astype(float)
    state["reflect_upper"]  = reflect  * (act == 1).astype(float)
    return state, parameters


# ---------- collect tasks: copy to output_dict ----------------------------
def collect_channel_populations(sim, state, parameters):
    """Copy channel classification into output_dict."""
    for key in ("transmit_lower", "transmit_upper",
                "reflect_lower", "reflect_upper"):
        state["output_dict"][key] = state[key]
    return state, parameters


# ---------- parameter scan ------------------------------------------------
momenta = np.array([5, 10, 15, 20, 25, 30], dtype=float)
transmit_lower = np.zeros_like(momenta)
transmit_upper = np.zeros_like(momenta)
reflect_lower  = np.zeros_like(momenta)
reflect_upper  = np.zeros_like(momenta)

for i, p0 in enumerate(momenta):
    print(f"Running p0 = {p0:.1f} ...")

    sim = Simulation({
        "tmax":        2.0 * 30.0 * 2000.0 / p0,   # adapt to velocity
        "dt_update":   2.0,
        "dt_collect":  1000.0,
        "num_trajs":   200,
        "batch_size":  200,
        "progress_bar": False,
    })
    sim.model = TullyProblemOne({
        "init_position": -15.0,
        "init_momentum":  p0,
        "mass":         2000.0,
    })
    sim.algorithm = FewestSwitchesSurfaceHopping()

    # Append update + collect tasks AFTER algorithm instantiation.
    # The update task goes in both initialization_recipe (so it runs at t=0
    # before the first collect) and update_recipe (so it runs every step).
    sim.algorithm.initialization_recipe = (
        sim.algorithm.initialization_recipe + [update_channel_classification]
    )
    sim.algorithm.update_recipe = (
        sim.algorithm.update_recipe + [update_channel_classification]
    )
    sim.algorithm.collect_recipe = (
        sim.algorithm.collect_recipe + [collect_channel_populations]
    )

    wf0 = np.array([1.0 + 0j, 0.0 + 0j])
    sim.initial_state["wf_db"] = wf0

    data = serial_driver(sim)

    transmit_lower[i] = data.data_dict["transmit_lower"][-1]
    transmit_upper[i] = data.data_dict["transmit_upper"][-1]
    reflect_lower[i]  = data.data_dict["reflect_lower"][-1]
    reflect_upper[i]  = data.data_dict["reflect_upper"][-1]

# ---------- print results -------------------------------------------------
print("\n--- Tully 1 Momentum Scan Results ---")
print(f"{'p0':>6s}  {'T_lower':>8s}  {'T_upper':>8s}  {'R_lower':>8s}  {'R_upper':>8s}")
for i, p0 in enumerate(momenta):
    print(f"{p0:6.1f}  {transmit_lower[i]:8.4f}  {transmit_upper[i]:8.4f}"
          f"  {reflect_lower[i]:8.4f}  {reflect_upper[i]:8.4f}")

# ---------- plot (optional) -----------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(momenta, transmit_lower, "o-", label="Transmit lower")
    ax.plot(momenta, transmit_upper, "s-", label="Transmit upper")
    ax.plot(momenta, reflect_lower,  "^-", label="Reflect lower")
    ax.plot(momenta, reflect_upper,  "v-", label="Reflect upper")
    ax.set_xlabel("Initial momentum p₀")
    ax.set_ylabel("Probability")
    ax.set_title("Tully Problem 1 — FSSH Momentum Scan")
    ax.legend()
    fig.savefig("tully_momentum_scan.png", dpi=150)
    print("Plot saved to tully_momentum_scan.png")
except ImportError:
    print("matplotlib not available — skipping plot")
