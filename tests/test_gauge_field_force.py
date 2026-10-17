"""
This module contains tests of the gauge field force in the quantum-classical force.
"""

import numpy as np
from qclab import Simulation, tasks, ingredients
from qclab.models import SpinBoson
from qclab.algorithms import FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver

GAUGE_FIELD_STRENGTH = 0.05


def gauge_field_force_state_dependent(model, parameters, **kwargs):
    """
    Gauge field force G(z, a) = (a + 1) z, which depends on both the classical
    coordinates and the index a of the adiabatic state.
    """
    z = kwargs["z"]
    state_ind = kwargs["state_ind"]
    return (state_ind[:, np.newaxis] + 1) * z


def gauge_field_force_linear(model, parameters, **kwargs):
    """
    State-independent gauge field force G(z) = lambda z.
    """
    z = kwargs["z"]
    return GAUGE_FIELD_STRENGTH * z


def dh_c_dzc_harmonic_plus_linear(model, parameters, **kwargs):
    """
    Harmonic classical force plus lambda z, i.e., the classical force with
    ``gauge_field_force_linear`` added to it.
    """
    z = kwargs["z"]
    return ingredients.dh_c_dzc_harmonic(model, parameters, **kwargs) + (
        GAUGE_FIELD_STRENGTH * z
    )


def _make_sim(
    use_gauge_field_force,
    gauge_field_force=None,
    dh_c_dzc=None,
    num_trajs=4,
    tmax=1.0,
):
    sim = Simulation()
    sim.settings.progress_bar = False
    sim.settings.num_trajs = num_trajs
    sim.settings.batch_size = num_trajs
    sim.settings.tmax = tmax
    sim.settings.dt_update = 0.01
    sim.model = SpinBoson()
    if gauge_field_force is not None:
        sim.model.ingredients.append(("gauge_field_force", gauge_field_force))
    if dh_c_dzc is not None:
        sim.model.ingredients.append(("dh_c_dzc", dh_c_dzc))
    sim.algorithm = FewestSwitchesSurfaceHopping(
        {"use_gauge_field_force": use_gauge_field_force}
    )
    sim.initial_state["wf_db"] = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state["wf_db"][0] += 1.0
    return sim


def _initialize(sim):
    """
    Runs the initialization recipe and returns the State and Parameters objects.
    """
    sim.model.initialize_constants()
    sim.initialize_timesteps()
    state = {"seed": np.arange(sim.settings.batch_size, dtype=int)}
    parameters = {}
    state, parameters = sim.algorithm.execute_recipe(
        sim, state, parameters, sim.algorithm.initialization_recipe
    )
    return state, parameters


def _rk4_stage_forces(sim, state, parameters, quantum_classical_force_name):
    """
    Evaluates the quantum-classical force at four RK4-like stages in the same way as
    the FewestSwitchesSurfaceHopping update recipe: the first stage with
    ``wf_changed=True`` and the remaining stages with ``wf_changed=False``.
    Returns the list of stage coordinates and the list of stage forces.
    """
    rng = np.random.default_rng(0)
    z = state["z"]
    z_stages = [z] + [
        z + 0.1 * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
        for _ in range(3)
    ]
    z_names = ["z", "z_1", "z_2", "z_3"]
    forces = []
    for n, (z_name, z_k) in enumerate(zip(z_names, z_stages)):
        state[z_name] = z_k
        state, parameters = tasks.update_quantum_classical_force(
            sim,
            state,
            parameters,
            z_name=z_name,
            wf_db_name="act_surf_wf",
            wf_changed=(n == 0),
            quantum_classical_force_name=quantum_classical_force_name,
        )
        forces.append(np.copy(state[quantum_classical_force_name]))
    return z_stages, forces


def test_gauge_field_force_evaluated_per_stage():
    """
    The gauge field force must be evaluated at the coordinates and adiabatic state of
    each stage, must not accumulate when the base force is reused
    (``wf_changed=False``), and must be added to the force stored under the
    ``quantum_classical_force_name`` passed to the task.
    """
    # Reference: no gauge field force, default force name.
    sim_ref = _make_sim(use_gauge_field_force=False)
    state_ref, parameters_ref = _initialize(sim_ref)
    _, forces_ref = _rk4_stage_forces(
        sim_ref, state_ref, parameters_ref, "quantum_classical_force"
    )
    # Gauge field force G(z, a) = (a + 1) z, custom force name.
    sim = _make_sim(
        use_gauge_field_force=True,
        gauge_field_force=gauge_field_force_state_dependent,
    )
    state, parameters = _initialize(sim)
    assert np.allclose(state["z"], state_ref["z"])
    assert np.array_equal(state["act_surf_ind"], state_ref["act_surf_ind"])
    assert not sim.model.update_dh_qc_dzc
    z_stages, forces = _rk4_stage_forces(sim, state, parameters, "custom_force")
    state_factor = (state["act_surf_ind"] + 1)[:, np.newaxis]
    for z_k, force_ref, force in zip(z_stages, forces_ref, forces):
        assert np.allclose(force, force_ref + state_factor * z_k)
    # The base force must be stored under the name derived from the force name and
    # must not contain the gauge field force.
    assert np.allclose(state["custom_force_base"], forces_ref[-1])
    # Nothing must be written under the default force names.
    assert "quantum_classical_force" not in state
    assert "quantum_classical_force_base" not in state


def test_base_force_names_do_not_collide():
    """
    Forces stored under different names must keep separate base forces, so that
    reusing one (``wf_changed=False``) does not return the base force of the other.
    """
    sim = _make_sim(use_gauge_field_force=False)
    state, parameters = _initialize(sim)
    force_names_and_wfs = [("force_a", "act_surf_wf"), ("force_b", "wf_db")]
    forces = {}
    for force_name, wf_db_name in force_names_and_wfs:
        state, parameters = tasks.update_quantum_classical_force(
            sim,
            state,
            parameters,
            wf_db_name=wf_db_name,
            quantum_classical_force_name=force_name,
        )
        forces[force_name] = np.copy(state[force_name])
    assert not np.allclose(forces["force_a"], forces["force_b"])
    state, parameters = tasks.update_quantum_classical_force(
        sim,
        state,
        parameters,
        wf_db_name="act_surf_wf",
        quantum_classical_force_name="force_a",
        wf_changed=False,
    )
    assert np.allclose(state["force_a"], forces["force_a"])


def test_zero_gauge_field_force_matches_no_gauge_field_force():
    """
    A full FSSH simulation with a zero gauge field force must reproduce the
    simulation without a gauge field force.
    """
    sim_ref = _make_sim(use_gauge_field_force=False)
    data_ref = serial_driver(sim_ref)
    sim = _make_sim(
        use_gauge_field_force=True,
        gauge_field_force=ingredients.gauge_field_force_zero,
    )
    data = serial_driver(sim)
    for key, val in data_ref.data_dict.items():
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            assert np.allclose(val, data.data_dict[key]), key


def test_gauge_field_force_matches_classical_force():
    """
    A full FSSH simulation with the gauge field force G(z) = lambda z must reproduce
    the simulation without a gauge field force in which lambda z is instead added to
    the classical force, and must differ from the simulation without either.
    """
    kwargs = {"num_trajs": 20, "tmax": 5.0}
    sim_gauge = _make_sim(
        use_gauge_field_force=True,
        gauge_field_force=gauge_field_force_linear,
        **kwargs,
    )
    data_gauge = serial_driver(sim_gauge)
    sim_classical = _make_sim(
        use_gauge_field_force=False,
        dh_c_dzc=dh_c_dzc_harmonic_plus_linear,
        **kwargs,
    )
    data_classical = serial_driver(sim_classical)
    sim_ref = _make_sim(use_gauge_field_force=False, **kwargs)
    data_ref = serial_driver(sim_ref)
    differs_from_ref = False
    for key, val in data_classical.data_dict.items():
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            assert np.allclose(val, data_gauge.data_dict[key]), key
            if not np.allclose(val, data_ref.data_dict[key]):
                differs_from_ref = True
    assert differs_from_ref
