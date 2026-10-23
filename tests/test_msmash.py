"""
This module tests the MultiStateMappingApproachToSurfaceHopping algorithm.
"""

import numpy as np
from qclab import Simulation, functions
from qclab.models import (
    SpinBoson,
    ComplexAvoidedCrossingA,
)
from qclab.algorithms import MultiStateMappingApproachToSurfaceHopping
from qclab.dynamics import serial_driver

try:
    from tests.reference_settings import model_sim_settings, model_settings
except ImportError:
    from reference_settings import model_sim_settings, model_settings


def _make_sim(model, sim_settings, algorithm=None):
    """
    Build a Simulation with the model, MSMASH algorithm, and the first diabatic state
    populated.
    """
    sim = Simulation(sim_settings)
    sim.model = model
    sim.model.initialize_constants()
    if algorithm is None:
        algorithm = MultiStateMappingApproachToSurfaceHopping()
    sim.algorithm = algorithm
    sim.initial_state["wf_db"] = np.zeros(
        sim.model.constants.num_quantum_states, dtype=complex
    )
    sim.initial_state["wf_db"][0] = 1j
    return sim


def _collect_state_consistency(sim, state, parameters):
    """
    Test-only collect task recording how far the stored active-surface energy and
    wavefunction are from a fresh diagonalization at the current classical coordinates.
    """
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    h_q_tot = h_q(sim.model, parameters, batch_size=sim.settings.batch_size) + h_qc(
        sim.model, parameters, z=state["z"]
    )
    eigvals, eigvecs = np.linalg.eigh(h_q_tot)
    traj_inds = np.arange(sim.settings.batch_size)
    active = state["act_surf_ind"]
    eigval_error = np.abs(
        state["eigvals"][traj_inds, active] - eigvals[traj_inds, active]
    ).max()
    overlap = np.abs(
        np.sum(np.conj(state["act_surf_wf"]) * eigvecs[traj_inds, :, active], axis=1)
    )
    state["output_dict"]["eigval_error"] = eigval_error
    state["output_dict"]["act_surf_wf_error"] = np.abs(overlap - 1.0).max()
    return state, parameters


def test_msmash_population_estimator():
    """
    Verify that the MASH density matrix has unit trace at every collected time and that
    the initial population estimator of the focused state equals one.
    """
    model_name = "SpinBoson"
    sim = _make_sim(
        SpinBoson(model_settings[model_name]), model_sim_settings[model_name]
    )
    data = serial_driver(sim)
    dm_db = data.data_dict["dm_db"]
    np.testing.assert_allclose(
        np.trace(dm_db, axis1=1, axis2=2), 1.0, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(dm_db[0, 0, 0].real, 1.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        functions.calc_alpha_n_mash(2), 2.0, rtol=0.0, atol=1e-12
    )


def test_msmash_state_consistency_after_hops():
    """
    Verify that the stored eigenvalues and active-surface wavefunction are consistent with
    the current classical coordinates after hops are resolved.
    """
    model_name = "SpinBoson"
    sim = _make_sim(
        SpinBoson(model_settings[model_name]),
        {**model_sim_settings[model_name], "dt_collect": 0.01},
    )
    sim.algorithm.collect_recipe = sim.algorithm.collect_recipe + [
        _collect_state_consistency
    ]
    data = serial_driver(sim)
    assert np.max(data.data_dict["eigval_error"]) < 1e-10
    assert np.max(data.data_dict["act_surf_wf_error"]) < 1e-10


def test_msmash_complex_avoided_crossing():
    """
    Verify that MSMASH runs on a complex-valued Hamiltonian with a model-supplied
    rescaling direction and derivative-coupling gauge fixing.
    """
    sim = _make_sim(
        ComplexAvoidedCrossingA(),
        {
            "num_trajs": 10,
            "batch_size": 10,
            "tmax": 0.5,
            "dt_update": 0.001,
            "dt_collect": 0.05,
            "progress_bar": False,
        },
        algorithm=MultiStateMappingApproachToSurfaceHopping(
            {"gauge_fixing": "phase_der_couple"}
        ),
    )
    data = serial_driver(sim)
    dm_db = data.data_dict["dm_db"]
    assert np.all(np.isfinite(dm_db))
    np.testing.assert_allclose(
        np.trace(dm_db, axis1=1, axis2=2), 1.0, rtol=0.0, atol=1e-10
    )
