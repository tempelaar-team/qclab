"""
This module tests the MultiStateMappingApproachToSurfaceHopping algorithm.
"""

import numpy as np
from qclab import Simulation, functions
from qclab.models import (
    SpinBoson,
    ComplexAvoidedCrossingA,
    HolsteinLattice,
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


def _make_reciprocal_phonon_holstein(num_sites):
    """
    Holstein lattice Model with the phonons in reciprocal space and the electronic
    states in real space. Initial classical coordinates are the real-space samples,
    transformed, so that trajectories can be compared one to one.
    """
    sites = np.arange(num_sites)
    # z_n = sum_k U_nk z_k, written z_n = z_k @ U.T for row vectors.
    unitary = np.exp(-2j * np.pi * np.outer(sites, sites) / num_sites) / np.sqrt(
        num_sites
    )
    unitary_conj = unitary.conj()
    real_space = HolsteinLattice({"N": num_sites})
    h_c, _ = real_space.get("h_c")
    h_qc, _ = real_space.get("h_qc")
    dh_c_dzc, _ = real_space.get("dh_c_dzc")
    dh_qc_dzc, _ = real_space.get("dh_qc_dzc")
    init_classical, _ = real_space.get("init_classical")

    def h_c_reciprocal(model, parameters, **kwargs):
        return h_c(real_space, parameters, z=kwargs["z"] @ unitary.T)

    def h_qc_reciprocal(model, parameters, **kwargs):
        return h_qc(real_space, parameters, z=kwargs["z"] @ unitary.T)

    def dh_c_dzc_reciprocal(model, parameters, **kwargs):
        return dh_c_dzc(real_space, parameters, z=kwargs["z"] @ unitary.T) @ (
            unitary_conj
        )

    def dh_qc_dzc_reciprocal(model, parameters, **kwargs):
        inds, mels, shape = dh_qc_dzc(real_space, parameters, z=kwargs["z"] @ unitary.T)
        dense = np.zeros(shape, dtype=complex)
        dense[inds] = mels
        dense = np.einsum("bnij,nk->bkij", dense, unitary_conj)
        inds = np.nonzero(np.abs(dense) > 1e-14)
        return inds, dense[inds], shape

    def init_classical_reciprocal(model, parameters, **kwargs):
        return init_classical(real_space, parameters, seed=kwargs["seed"]) @ (
            unitary_conj
        )

    model = HolsteinLattice({"N": num_sites})
    # The hop_harmonic Ingredient is kept, since it is unchanged by the transformation
    # for phonon modes of equal frequency and weight.
    model.ingredients = model.ingredients + [
        ("h_c", h_c_reciprocal),
        ("h_qc", h_qc_reciprocal),
        ("dh_c_dzc", dh_c_dzc_reciprocal),
        ("dh_qc_dzc", dh_qc_dzc_reciprocal),
        ("init_classical", init_classical_reciprocal),
    ]
    return model


def test_msmash_independent_of_classical_basis(monkeypatch):
    """
    MSMASH gives the same diabatic density matrix with the phonons of the Holstein
    lattice in real space and in reciprocal space, including frustrated-hop reversals
    of the physical momentum.
    """
    num_reversals = []
    calc_z_physical_momentum_reversal = functions.calc_z_physical_momentum_reversal

    def counting_reversal(*args, **kwargs):
        num_reversals[-1] += 1
        return calc_z_physical_momentum_reversal(*args, **kwargs)

    monkeypatch.setattr(
        functions, "calc_z_physical_momentum_reversal", counting_reversal
    )
    sim_settings = {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 5,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    }
    dm_db = []
    for model in (HolsteinLattice({"N": 4}), _make_reciprocal_phonon_holstein(4)):
        num_reversals.append(0)
        data = serial_driver(_make_sim(model, dict(sim_settings)))
        dm_db.append(data.data_dict["dm_db"])
    assert num_reversals[0] > 0
    assert num_reversals[0] == num_reversals[1]
    np.testing.assert_allclose(dm_db[1], dm_db[0], rtol=0.0, atol=1e-10)
