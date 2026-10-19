"""Exercise local diabatic transport with every supported PySCF state model."""

import copy
import pickle
from functools import partial

import numpy as np
import pytest

pytest.importorskip("pyscf")

from qclab import Simulation, functions, ingredients
from qclab.algorithms import (
    MeanField, MeanFieldAbInitio,
    FewestSwitchesSurfaceHopping, FewestSwitchesSurfaceHoppingAbInitio,
)
from qclab.dynamics import serial_driver
from qclab.interfaces import QCLabPySCFInterface
from qclab.models import AbInitio, LocalDiabaticAbInitio
from qclab.numerical_constants import AMU_TO_EMASS, ANGSTROM_TO_BOHR


POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
MASSES = np.array([7.0160034366, 1.00782503223]) * AMU_TO_EMASS
CASES = (
    "casscf", "cms-tPBE", "cms-tBLYP", "cms-ftPBE",
    "tda-pbe-singlet", "tda-pbe-triplet",
    "tda-b3lyp-singlet", "tda-b3lyp-triplet",
)


def model_constants(case="casscf"):
    """Keep spin, response roots, grids, and convergence identical in each frame."""
    arguments = {"basis": "sto-3g", "num_threads": 1,
                 "scf_options": {"conv_tol": 1e-12}}
    num_states = 2
    if case.startswith("tda-"):
        _, xc, spin = case.split("-")
        arguments.update({
            "method_es": "tddft", "tda": True, "xc": xc,
            "excited_multiplicity": 1 if spin == "singlet" else 3,
            "excited_state_indices": [0, 3], "use_etfs": False,
            "grids_level": 5, "td_options": {"conv_tol": 1e-11},
        })
        num_states = 3
    else:
        arguments.update({
            "method_es": "casscf", "ncas": 2, "nelecas": 2,
            "casscf_options": {"conv_tol": 1e-11, "conv_tol_grad": 1e-8,
                               "ah_conv_tol": 1e-14, "max_cycle_macro": 200},
        })
        if case.startswith("cms-"):
            arguments.update({
                "method_es": "cms-pdft", "otxc": case.split("-", 1)[1],
                "grids_level": 5, "cms_options": {"conv_tol_diabatize": 1e-10},
                "response_options": {"conv_atol": 1e-10, "conv_rtol": 0.0,
                                     "max_cycle": 200},
            })
    return {
        "calculator": "pyscf", "atom_names": ["Li", "H"],
        "atom_positions": POSITIONS.copy(), "atom_masses": MASSES.copy(),
        "num_quantum_states": num_states, "calculator_args": arguments,
        "energy_offset": -7.8,
        "init_momentum": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
    }


def coordinates(model, positions=POSITIONS):
    return functions.qp_to_z(
        np.asarray(positions).reshape(1, 6), np.zeros((1, 6)),
        model.constants.classical_coordinate_mass,
        model.constants.classical_coordinate_weight,
    )


def local_packet(model, parameters, positions=POSITIONS, action="evaluate"):
    ingredient, found = model.get("local_diabatization")
    assert found
    return ingredient(model, parameters, z=coordinates(model, positions),
                      action=action)[0]


def reference_properties(constants, packet):
    calculator = QCLabPySCFInterface(
        atom_names=constants["atom_names"], atom_masses=constants["atom_masses"],
        atom_positions=packet["snapshot"]["atom_positions"],
        num_quantum_states=constants["num_quantum_states"],
        **constants["calculator_args"],
    )
    return calculator.calculate({
        "energy": {"excited_amplitudes": True}, "gradient": {},
        "derivative_coupling": {},
        "wf_overlaps": {"amplitudes_previous": packet["snapshot"]},
    })


def adiabatic_gradient(properties):
    """Build the operator derivative, including both signs of every NAC pair."""
    energies = properties["energy"]
    gradient = np.zeros((6, len(energies), len(energies)), dtype=complex)
    for index in range(len(energies)):
        gradient[:, index, index] = properties["gradient"][:, :, index].ravel()
    for (bra, ket), coupling in properties["derivative_coupling"].items():
        value = ((energies[ket] - energies[bra])
                 * coupling.ravel() / ANGSTROM_TO_BOHR)
        gradient[:, bra, ket] = value
        gradient[:, ket, bra] = value.conj()
    return gradient


@pytest.mark.parametrize("case", CASES)
def test_three_geometry_transport_retains_all_coupling_forces(case):
    constants = model_constants(case)
    model = LocalDiabaticAbInitio(copy.deepcopy(constants))
    parameters = {}
    initial = local_packet(model, parameters, action="initialize")
    saved = pickle.dumps(initial["snapshot"])
    np.testing.assert_allclose(initial["adiabatic_to_diabatic"],
                               np.eye(constants["num_quantum_states"]), atol=1e-12)
    displaced = POSITIONS.copy()
    displaced[1, 2] += 0.04
    local_packet(model, parameters, displaced, action="evaluate")
    accepted = local_packet(model, parameters, displaced, action="commit")
    displaced[1, 2] += 0.04
    current = local_packet(model, parameters, displaced)
    assert current["frame_revision"] >= accepted["frame_revision"]
    assert np.all(current["overlap_singular_values"] > 0.99)
    assert pickle.dumps(initial["snapshot"]) == saved

    reference = reference_properties(constants, current)
    np.testing.assert_allclose(reference["wf_overlaps"],
                               np.eye(constants["num_quantum_states"]), atol=3e-7)
    transform = current["adiabatic_to_diabatic"]
    np.testing.assert_allclose(transform.conj().T @ transform,
                               np.eye(constants["num_quantum_states"]), atol=1e-12)
    np.testing.assert_allclose(current["hamiltonian"],
                               transform @ np.diag(reference["energy"] - constants["energy_offset"])
                               @ transform.conj().T,
                               atol=2e-8, rtol=0)
    gradient = adiabatic_gradient(reference)
    expected = np.einsum("ij,cjk,lk->cil", transform, gradient, transform.conj())
    np.testing.assert_allclose(current["gradient"], expected, atol=3e-7, rtol=0)
    assert np.linalg.norm(gradient[:, 0, -1] if len(transform) == 2
                          else gradient[:, 1, 2]) > 1e-5
    random = np.random.RandomState(91)
    coefficients = random.normal(size=len(transform)) + 1j * random.normal(size=len(transform))
    coefficients /= np.linalg.norm(coefficients)
    local_coefficients = transform @ coefficients
    force = -np.einsum("i,cij,j->c", coefficients.conj(), gradient, coefficients)
    local_force = -np.einsum("i,cij,j->c", local_coefficients.conj(),
                             current["gradient"], local_coefficients)
    np.testing.assert_allclose(local_force, force, atol=3e-7, rtol=0)
    # This superposition must distinguish the full force from diagonal-only forces.
    diagonal_force = -np.einsum("i,ci,i->c", coefficients.conj(),
                                np.diagonal(gradient, axis1=1, axis2=2), coefficients)
    assert np.linalg.norm(force - diagonal_force) > 1e-6


@pytest.mark.parametrize("case", ["casscf", "cms-tPBE", "tda-pbe-triplet"])
def test_repeated_property_requests_reuse_native_responses_and_snapshots(monkeypatch, case):
    model = LocalDiabaticAbInitio(model_constants(case))
    parameters = {}
    first = local_packet(model, parameters, action="initialize")
    saved = pickle.dumps(first["snapshot"])
    calculator = parameters["_pyscf_calculators"][0]["calculator"]

    def unexpected(*args, **kwargs):
        pytest.fail("A repeated request repeated a native electronic calculation")

    monkeypatch.setattr(calculator, "_run", unexpected)
    if hasattr(calculator, "_gradient_from_td"):
        monkeypatch.setattr(calculator, "_gradient_from_td", unexpected)
        from qclab.interfaces import pyscf as pyscf_interface
        monkeypatch.setattr(pyscf_interface, "derivative_coupling", unexpected)
    else:
        monkeypatch.setattr(calculator._gradient_solver, "kernel", unexpected)
        monkeypatch.setattr(calculator._coupling_solver, "kernel", unexpected)
    repeated = local_packet(model, parameters)
    local_packet(model, parameters, action="commit")
    final = local_packet(model, parameters)
    for packet in (repeated, final):
        np.testing.assert_allclose(packet["hamiltonian"], first["hamiltonian"], atol=1e-12)
        np.testing.assert_allclose(packet["gradient"], first["gradient"], atol=1e-12)
    assert pickle.dumps(first["snapshot"]) == saved


def simulation(case, local=True, fssh=False, dt=0.1, steps=1):
    sim = Simulation({"tmax": steps * dt, "dt_update": dt, "dt_collect": dt,
                      "num_trajs": 1, "batch_size": 1, "progress_bar": False})
    constants = model_constants(case)
    sim.model = (LocalDiabaticAbInitio if local else AbInitio)(constants)
    sim.model.ingredients = sim.model.ingredients + [
        ("init_classical", ingredients.init_classical_definite_position_momentum)]
    if local:
        sim.algorithm = FewestSwitchesSurfaceHopping() if fssh else MeanField()
    else:
        algorithm = FewestSwitchesSurfaceHoppingAbInitio if fssh else MeanFieldAbInitio
        sim.algorithm = algorithm({"use_wf_overlaps_for_adb_connection": True})
    num_states = constants["num_quantum_states"]
    coefficients = np.array([1, 1] if num_states == 2 else [0, 1, 1], dtype=complex)
    sim.initial_state["wf_db" if local else "wf_adb"] = coefficients / np.sqrt(2)
    sim.algorithm.collect_recipe = sim.algorithm.collect_recipe + [
        partial(collect_wavefunction, wf_name="wf_db" if local else "wf_adb")]
    return sim


def collect_wavefunction(sim, state, parameters, wf_name="wf_db",
                         output_name="electronic_coefficients"):
    """Copy coefficients for independent normalization checks after propagation."""
    state["output_dict"][output_name] = state[wf_name].copy()
    return state, parameters


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fssh", [False, True], ids=["meanfield", "fssh"])
def test_short_standard_algorithm_trajectories(case, fssh):
    sim = simulation(case, fssh=fssh)
    data = serial_driver(sim, seeds=np.array([13]))
    for values in data.data_dict.values():
        assert np.all(np.isfinite(values))
    density = data.data_dict["dm_adb"]
    assert len(density) == 2
    np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1, atol=1e-9)
    np.testing.assert_allclose(density, density.swapaxes(-1, -2).conj(), atol=1e-12)
    np.testing.assert_allclose(np.sum(np.abs(data.data_dict["electronic_coefficients"])**2,
                                     axis=-1), 1, atol=1e-9)
    # Identical nuclear input and the identity anchor make initial observables match.
    expected = 0.5 * np.array([1, 1] if len(density[0]) == 2 else [0, 1, 1])
    if fssh:
        np.testing.assert_allclose(np.max(np.diagonal(density[0]).real), 1, atol=1e-12)
    else:
        np.testing.assert_allclose(np.diagonal(density[0]).real, expected, atol=1e-12)
    reference = serial_driver(simulation(case, local=False, fssh=fssh),
                              seeds=np.array([13]))
    # Both routes use the same physical state labels; the stock time integrators
    # remain distinct, so agreement is assessed with a finite-step tolerance.
    np.testing.assert_allclose(density, reference.data_dict["dm_adb"], atol=2e-5, rtol=0)
    for name in ("classical_energy", "quantum_energy"):
        np.testing.assert_allclose(data.data_dict[name], reference.data_dict[name],
                                   atol=2e-6, rtol=0)


def test_casscf_meanfield_representation_difference_decreases_with_timestep():
    """Compare equal physical durations, retaining the distinct stock integrators."""
    errors = []
    for dt in (0.2, 0.1, 0.05, 0.025):
        steps = round(0.4 / dt)
        local = serial_driver(simulation("casscf", dt=dt, steps=steps))
        adiabatic = serial_driver(simulation("casscf", local=False, dt=dt, steps=steps))
        for name in ("dm_adb", "classical_energy", "quantum_energy"):
            np.testing.assert_allclose(local.data_dict[name][0], adiabatic.data_dict[name][0],
                                       atol=2e-9, rtol=0)
        errors.append(np.max(np.abs(local.data_dict["dm_adb"][-1]
                                     - adiabatic.data_dict["dm_adb"][-1])))
    assert errors[0] < 2e-6
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < 0.75 * coarse + 2e-9


def test_tda_electron_translation_factors_cannot_be_used_for_overlap_transport():
    constants = model_constants("tda-pbe-singlet")
    constants["calculator_args"]["use_etfs"] = True
    with pytest.raises((ValueError, NotImplementedError), match="(?i)etf|translation"):
        model = LocalDiabaticAbInitio(constants)
        local_packet(model, {}, action="initialize")
