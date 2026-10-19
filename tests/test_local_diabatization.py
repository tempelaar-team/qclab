"""Representation checks with exact electronic states, independent of PySCF.

The synthetic calculator diagonalizes a known diabatic Hamiltonian, so its
energies, gradients, derivative couplings, and overlaps have an independent
Cartesian oracle.  No optional electronic-structure program is imported.
"""

import copy

import numpy as np
import pytest

from qclab import Simulation, functions, ingredients, tasks
from qclab.algorithms import (
    FewestSwitchesSurfaceHopping,
    FewestSwitchesSurfaceHoppingAbInitio,
    MeanField,
    MeanFieldAbInitio,
)
from qclab.models import (
    AbInitio, LocalDiabaticAbInitio,
    TullyProblemOne, TullyProblemTwo, TullyProblemThree,
)
from qclab.dynamics import serial_driver
from qclab.numerical_constants import ANGSTROM_TO_BOHR


def _hamiltonian(position):
    """Tully's avoided-crossing shape with two transverse nuclear coordinates."""
    x, y, z = np.asarray(position).reshape(3)
    diagonal = 0.1 * np.sign(x) * (1.0 - np.exp(-abs(x)))
    coupling_base = 0.04 * np.exp(-0.6 * x * x)
    coupling = coupling_base * (1.0 + 0.2 * y) + 0.01 * z
    matrix = np.array([[diagonal, coupling], [coupling, -diagonal]])
    matrix += 0.005 * (y * y + z * z) * np.eye(2)
    gradient = np.zeros((3, 2, 2))
    diagonal_gradient = 0.1 * np.exp(-abs(x))
    coupling_gradient = -1.2 * x * coupling_base * (1.0 + 0.2 * y)
    gradient[0] = [[diagonal_gradient, coupling_gradient],
                   [coupling_gradient, -diagonal_gradient]]
    gradient[1] = 0.01 * y * np.eye(2)
    gradient[2] = 0.01 * z * np.eye(2)
    gradient[1, 0, 1] = gradient[1, 1, 0] = 0.2 * coupling_base
    gradient[2, 0, 1] = gradient[2, 1, 0] = 0.01
    return matrix, gradient


def _electronic_structure(position):
    matrix, gradient = _hamiltonian(position)
    energies, vectors = np.linalg.eigh(matrix)
    gradient_ad = np.einsum("ai,cab,bj->cij", vectors, gradient, vectors)
    coupling = np.zeros_like(gradient_ad)
    coupling[:, 0, 1] = gradient_ad[:, 0, 1] / (energies[1] - energies[0])
    coupling[:, 1, 0] = -coupling[:, 0, 1]
    return energies, vectors, gradient_ad, coupling


class SyntheticCalculator:
    """A recording implementation of the existing batched property contract."""

    def __init__(self, phase_changes=False):
        self.calls = []
        self.phase_changes = phase_changes
        self.overlap_override = None

    def __call__(self, model, parameters, **kwargs):
        requests = kwargs["property_dict"]
        size = kwargs.get("batch_size")
        if size is None:
            size = len(next(args["z"] for args in requests.values() if "z" in args))
        output = np.empty(size, dtype=object)
        for index in range(size):
            args_with_z = next(args for args in requests.values() if "z" in args)
            z = np.asarray(args_with_z["z"])[index]
            position = functions.z_to_q(
                z, model.constants.classical_coordinate_mass,
                model.constants.classical_coordinate_weight,
            )
            energies, vectors, gradient, coupling = _electronic_structure(position)
            if self.phase_changes:
                signs = np.array([(-1.0) ** (len(self.calls) + 1), -1.0])
                vectors = vectors * signs
                gradient = gradient * signs[None, :, None] * signs[None, None, :]
                coupling = coupling * signs[None, :, None] * signs[None, None, :]
            snapshot = {
                "atom_positions": position.reshape(1, 3).copy(),
                "positions": position.reshape(1, 3).copy(),
                "vectors": vectors.copy(),
                "energies": energies.copy(),
            }
            result = {}
            for name, args in requests.items():
                active = args.get("calc_property", True)
                if isinstance(active, np.ndarray):
                    active = active[index]
                if not active:
                    result[name] = None
                elif name == "energy":
                    result[name] = energies.copy()
                    if args.get("excited_amplitudes", False):
                        result["excited_amplitudes"] = copy.deepcopy(snapshot)
                elif name == "gradient":
                    result[name] = np.diagonal(gradient, axis1=-2, axis2=-1).reshape(1, 3, 2).copy()
                elif name == "derivative_coupling":
                    result[name] = {(0, 1): coupling[:, 0, 1].reshape(1, 3) * ANGSTROM_TO_BOHR}
                elif name == "wf_overlaps":
                    previous = args["amplitudes_previous"]
                    if not isinstance(previous, dict):
                        previous = previous[index]
                    result[name] = previous["vectors"].T @ vectors
                    if self.overlap_override is not None:
                        result[name] = self.overlap_override.copy()
            self.calls.append({
                "position": position.copy(), "requests": copy.deepcopy(requests),
                "trajectory": kwargs.get("cache_traj_ind", index),
                "snapshot": copy.deepcopy(snapshot),
            })
            output[index] = result
        return output


def _model(local=True, calculator=None, position=(-0.8, 0.2, -0.1), **overrides):
    constants = {
        "atom_names": ["H"], "atom_positions": np.asarray(position, dtype=float).reshape(1, 3),
        "atom_masses": np.array([10.0]), "num_quantum_states": 2,
        "calculator": "pyscf", "calculator_args": {"method_es": "casscf", "ncas": 2, "nelecas": 2},
        "init_momentum": np.array([5.0, 0.1, -0.2]),
    }
    constants.update(overrides)
    model = (LocalDiabaticAbInitio if local else AbInitio)(constants)
    model.ingredients.append(("ab_initio_property_calculator", calculator or SyntheticCalculator()))
    model.ingredients.append(("init_classical", ingredients.init_classical_definite_position_momentum))
    return model


def _z(model, positions):
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    return functions.qp_to_z(
        positions, np.zeros_like(positions), model.constants.classical_coordinate_mass,
        model.constants.classical_coordinate_weight,
    )


def _packet(model, parameters, positions, action="evaluate"):
    ingredient, found = model.get("local_diabatization")
    assert found
    return ingredient(model, parameters, z=_z(model, positions), action=action)[0]


def test_gradient_contains_coherence_force_and_converts_nac_units_once():
    model = _model()
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    _packet(model, parameters, anchor, "initialize")
    point = np.array([0.15, 0.3, -0.2])
    packet = _packet(model, parameters, point)
    _, reference_vectors, _, _ = _electronic_structure(anchor)
    matrix, gradient = _hamiltonian(point)
    expected_h = reference_vectors.T @ matrix @ reference_vectors
    expected_g = np.einsum("ai,cab,bj->cij", reference_vectors, gradient, reference_vectors)
    np.testing.assert_allclose(packet["hamiltonian"], expected_h, atol=2e-14)
    np.testing.assert_allclose(packet["gradient"], expected_g, atol=2e-14)
    coefficients = np.array([np.sqrt(0.7), np.sqrt(0.3) * np.exp(0.4j)])
    force = -np.einsum("i,cij,j->c", coefficients.conj(), packet["gradient"], coefficients).real
    physical_coefficients = reference_vectors @ coefficients
    expected_force = -np.einsum("i,cij,j->c", physical_coefficients.conj(), gradient, physical_coefficients).real
    np.testing.assert_allclose(force, expected_force, atol=2e-14)
    assert abs(expected_g[0, 0, 1]) > 0.01


@pytest.mark.parametrize("tully_class", [TullyProblemOne, TullyProblemTwo, TullyProblemThree])
def test_stock_tully_model_operators_are_recovered_from_adiabatic_properties(monkeypatch, tully_class):
    """Use the actual stock-model ingredients as independent diabatic oracles."""
    native = tully_class()
    native_hamiltonian, _ = native.get("h_qc")
    native_gradient, _ = native.get("dh_qc_dzc")
    masses = native.constants.classical_coordinate_mass
    weights = native.constants.classical_coordinate_weight

    def native_cartesian(position):
        x = float(np.asarray(position).reshape(3)[0])
        native_z = functions.qp_to_z(np.array([[x]]), np.zeros((1, 1)), masses, weights)
        hamiltonian = np.real_if_close(native_hamiltonian(native, {}, z=native_z)[0])
        indices, values, shape = native_gradient(native, {}, z=native_z)
        dense = np.zeros(shape, dtype=complex)
        dense[indices] = values
        dq, _ = functions.dzdzc_to_dqdp(None, dense, masses[None, :, None, None],
                                        weights[None, :, None, None])
        gradient = np.zeros((3, 2, 2))
        gradient[0] = np.real_if_close(dq[0, 0])
        return hamiltonian, gradient

    # The mock electronic calculation obtains its eigensystem/NACs from each
    # native Tully model. The production adapter sees only adiabatic properties.
    monkeypatch.setattr(__name__ + "._hamiltonian", native_cartesian)
    model = _model()
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    _packet(model, parameters, anchor, "initialize")
    initial_vectors = np.linalg.eigh(native_cartesian(anchor)[0])[1]
    for point in ([-0.3, 0.2, -0.1], [0.4, -0.1, 0.2], [1.2, 0.1, 0.0]):
        packet = _packet(model, parameters, point)
        matrix, gradient = native_cartesian(point)
        np.testing.assert_allclose(packet["hamiltonian"],
                                   initial_vectors.T @ matrix @ initial_vectors, atol=2e-14)
        np.testing.assert_allclose(packet["gradient"],
                                   np.einsum("ai,cab,bj->cij", initial_vectors, gradient, initial_vectors),
                                   atol=2e-14)
        _packet(model, parameters, point, "commit")


def test_reference_stays_fixed_across_nonmonotone_stage_queries():
    calculator = SyntheticCalculator()
    model = _model(calculator=calculator)
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    initial = _packet(model, parameters, anchor, "initialize")
    first = _packet(model, parameters, [0.3, 0.1, -0.1])
    _packet(model, parameters, [-0.4, 0.2, 0.1])
    repeated = _packet(model, parameters, [0.3, 0.1, -0.1])
    assert repeated["frame_revision"] == initial["frame_revision"]
    np.testing.assert_allclose(first["hamiltonian"], repeated["hamiltonian"], atol=1e-14)
    np.testing.assert_allclose(first["gradient"], repeated["gradient"], atol=1e-14)
    for call in calculator.calls:
        overlap_request = call["requests"].get("wf_overlaps")
        if overlap_request is not None:
            previous = overlap_request["amplitudes_previous"]
            if not isinstance(previous, dict):
                previous = previous[0]
            np.testing.assert_array_equal(previous["positions"].reshape(3), anchor)


def test_commit_preserves_endpoint_operators_and_accumulates_transport():
    model = _model()
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    _packet(model, parameters, anchor, "initialize")
    for point in ([0.1, 0.1, 0.2], [0.6, -0.1, 0.3], [-0.2, 0.0, 0.1]):
        before = _packet(model, parameters, point)
        _packet(model, parameters, point, "commit")
        after = _packet(model, parameters, point)
        assert after["frame_revision"] == before["frame_revision"] + 1
        for key in ("hamiltonian", "gradient", "adiabatic_to_diabatic"):
            np.testing.assert_allclose(after[key], before[key], atol=3e-14)
        _, vectors0, _, _ = _electronic_structure(anchor)
        matrix, gradient = _hamiltonian(point)
        np.testing.assert_allclose(after["hamiltonian"], vectors0.T @ matrix @ vectors0, atol=3e-14)
        np.testing.assert_allclose(after["gradient"], np.einsum("ai,cab,bj->cij", vectors0, gradient, vectors0), atol=3e-14)


def test_cumulative_frame_matches_rolling_resets_with_explicit_state_rotations():
    """Independent reset/rotation propagation represents the same physical state."""
    model = _model()
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    initial = _packet(model, parameters, anchor, "initialize")
    rolling_vectors = initial["snapshot"]["vectors"].copy()
    rolling_coefficients = np.array([np.sqrt(0.7), np.sqrt(0.3) * np.exp(0.4j)])
    persistent_coefficients = rolling_coefficients.copy()
    cumulative_rotation = np.eye(2, dtype=complex)

    def propagate(matrix, coefficients):
        energies, vectors = np.linalg.eigh(matrix)
        return vectors @ (np.exp(-0.2j * energies) * (vectors.conj().T @ coefficients))

    for point in ([0.1, 0.2, 0.1], [0.5, -0.1, 0.2], [-0.2, 0.3, -0.1]):
        packet = _packet(model, parameters, point)
        energies, vectors, gradient, _ = _electronic_structure(point)
        # This reference uses a fresh crude adiabatic basis each segment,
        # independently of the engine's persistent-coordinate bookkeeping.
        overlap = rolling_vectors.T @ vectors
        rolling_hamiltonian = (overlap * energies[None]) @ overlap.T
        rolling_gradient = overlap[None] @ gradient @ overlap.T[None]
        np.testing.assert_allclose(packet["hamiltonian"],
                                   cumulative_rotation @ rolling_hamiltonian @ cumulative_rotation.conj().T,
                                   atol=3e-14)
        np.testing.assert_allclose(packet["gradient"],
                                   cumulative_rotation[None] @ rolling_gradient @ cumulative_rotation.conj().T[None],
                                   atol=3e-14)
        persistent_coefficients = propagate(packet["hamiltonian"], persistent_coefficients)
        rolling_coefficients = propagate(rolling_hamiltonian, rolling_coefficients)
        physical_persistent = packet["adiabatic_to_diabatic"].conj().T @ persistent_coefficients
        physical_rolling = overlap.T @ rolling_coefficients
        np.testing.assert_allclose(physical_persistent, physical_rolling, atol=3e-14)
        force_persistent = np.einsum("i,cij,j->c", persistent_coefficients.conj(),
                                     packet["gradient"], persistent_coefficients)
        force_rolling = np.einsum("i,cij,j->c", rolling_coefficients.conj(),
                                  rolling_gradient, rolling_coefficients)
        np.testing.assert_allclose(force_persistent, force_rolling, atol=3e-14)
        # A rolling implementation rotates its state at reset. The cumulative
        # implementation changes only the accepted transform, leaving c fixed.
        rolling_coefficients = overlap.T @ rolling_coefficients
        cumulative_rotation = cumulative_rotation @ overlap
        rolling_vectors = vectors
        _packet(model, parameters, point, "commit")
        np.testing.assert_allclose(persistent_coefficients,
                                   cumulative_rotation @ rolling_coefficients, atol=3e-14)


def test_constant_complex_computational_basis_change_preserves_operators_and_forces():
    """A computational-coordinate rotation need not share the real electronic gauge."""
    model = _model()
    parameters = {}
    _packet(model, parameters, model.constants.atom_positions, "initialize")
    rotated_parameters = copy.deepcopy(parameters)
    unitary = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    # Change only the computational-coordinate convention. The physical anchor
    # and all electronic-structure results remain identical.
    frame = rotated_parameters["_local_diabatic_frames"][0]
    frame["transform"] = unitary @ frame["transform"]
    frame["cache"].clear()
    point = [0.2, -0.1, 0.3]
    original = _packet(model, parameters, point)
    rotated = _packet(model, rotated_parameters, point)
    np.testing.assert_allclose(rotated["hamiltonian"],
                               unitary @ original["hamiltonian"] @ unitary.conj().T,
                               atol=2e-14)
    np.testing.assert_allclose(rotated["gradient"],
                               unitary[None] @ original["gradient"] @ unitary.conj().T[None],
                               atol=2e-14)
    np.testing.assert_allclose(rotated["adiabatic_to_diabatic"],
                               unitary @ original["adiabatic_to_diabatic"], atol=2e-14)
    coefficients = np.array([np.sqrt(0.7), np.sqrt(0.3) * np.exp(0.4j)])
    rotated_coefficients = unitary @ coefficients
    original_force = np.einsum("i,cij,j->c", coefficients.conj(),
                                original["gradient"], coefficients)
    rotated_force = np.einsum("i,cij,j->c", rotated_coefficients.conj(),
                               rotated["gradient"], rotated_coefficients)
    np.testing.assert_allclose(rotated_force, original_force, atol=2e-14)
    for runtime, before in ((parameters, original), (rotated_parameters, rotated)):
        _packet(model, runtime, point, "commit")
        after = _packet(model, runtime, point)
        for key in ("hamiltonian", "gradient", "adiabatic_to_diabatic"):
            np.testing.assert_allclose(after[key], before[key], atol=2e-14)


def test_snapshot_results_cannot_mutate_the_committed_frame():
    model = _model()
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    initial = _packet(model, parameters, anchor, "initialize")
    initial["snapshot"]["vectors"][:] = 0.0
    initial["hamiltonian"][:] = 123.0
    point = [0.2, 0.1, -0.1]
    result = _packet(model, parameters, point)
    _, vectors0, _, _ = _electronic_structure(anchor)
    np.testing.assert_allclose(result["hamiltonian"], vectors0.T @ _hamiltonian(point)[0] @ vectors0, atol=2e-14)


def test_same_geometry_hamiltonian_and_gradient_share_one_property_request():
    calculator = SyntheticCalculator()
    model = _model(calculator=calculator)
    parameters = {}
    _packet(model, parameters, model.constants.atom_positions, "initialize")
    z = _z(model, [0.2, 0.1, -0.1])
    before = len(calculator.calls)
    h_qc, _ = model.get("h_qc")
    gradient, _ = model.get("dh_qc_dzc")
    h_qc(model, parameters, z=z)
    gradient(model, parameters, z=z)
    assert len(calculator.calls) == before + 1


def test_electronic_sign_changes_cancel_from_physical_operators():
    calculator = SyntheticCalculator(phase_changes=True)
    model = _model(calculator=calculator)
    parameters = {}
    anchor = model.constants.atom_positions.reshape(3)
    initial = _packet(model, parameters, anchor, "initialize")
    reference_vectors = initial["snapshot"]["vectors"].copy()
    for point in ([0.1, 0.2, 0.1], [0.4, -0.1, 0.2], [-0.1, 0.3, -0.2]):
        result = _packet(model, parameters, point)
        matrix, gradient = _hamiltonian(point)
        np.testing.assert_allclose(result["hamiltonian"], reference_vectors.T @ matrix @ reference_vectors, atol=2e-14)
        np.testing.assert_allclose(result["gradient"], np.einsum("ai,cab,bj->cij", reference_vectors, gradient, reference_vectors), atol=2e-14)
        _packet(model, parameters, point, "commit")


def test_singular_overlap_fails_without_committing_new_reference():
    calculator = SyntheticCalculator()
    model = _model(calculator=calculator)
    parameters = {}
    initial = _packet(model, parameters, model.constants.atom_positions, "initialize")
    calculator.overlap_override = np.diag([1.0, 0.0])
    with pytest.raises((ValueError, RuntimeError), match="(?i)(overlap|singular|subspace)"):
        _packet(model, parameters, [0.5, 0.0, 0.0])
    calculator.overlap_override = None
    after = _packet(model, parameters, [-0.1, 0.0, 0.0])
    assert after["frame_revision"] == initial["frame_revision"]


def test_nonunitary_overlap_is_unitarized_with_subspace_loss_visible():
    calculator = SyntheticCalculator()
    model = _model(calculator=calculator)
    parameters = {}
    _packet(model, parameters, model.constants.atom_positions, "initialize")
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    calculator.overlap_override = rotation @ np.diag([1.0, 0.8])
    result = _packet(model, parameters, [0.2, 0.1, 0.0])
    np.testing.assert_allclose(result["overlap_singular_values"], [1.0, 0.8], atol=1e-14)
    np.testing.assert_allclose(result["adiabatic_to_diabatic"], rotation, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(result["hamiltonian"]), result["energies"], atol=1e-14)


def test_missing_off_diagonal_gradient_information_is_not_silently_zeroed():
    class MissingCouplingCalculator(SyntheticCalculator):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            for properties in result:
                properties["derivative_coupling"] = {}
            return result

    model = _model(calculator=MissingCouplingCalculator())
    with pytest.raises(ValueError, match="(?i)(coupling|pair)"):
        _packet(model, {}, model.constants.atom_positions, "initialize")


def test_two_trajectories_keep_independent_reference_frames():
    model = _model()
    parameters = {}
    ingredient, _ = model.get("local_diabatization")
    anchors = np.array([[-0.8, 0.1, 0.0], [0.4, -0.2, 0.1]])
    ingredient(model, parameters, z=_z(model, anchors), action="initialize")
    points = np.array([[0.2, 0.0, 0.1], [-0.3, 0.1, 0.0]])
    results = ingredient(model, parameters, z=_z(model, points), action="evaluate")
    for anchor, point, result in zip(anchors, points, results):
        _, vectors0, _, _ = _electronic_structure(anchor)
        np.testing.assert_allclose(result["hamiltonian"], vectors0.T @ _hamiltonian(point)[0] @ vectors0, atol=2e-14)


def test_configuration_changes_require_explicit_reinitialization():
    model = _model()
    parameters = {}
    _packet(model, parameters, model.constants.atom_positions, "initialize")
    model.constants.calculator_args["ncas"] = 3
    with pytest.raises(ValueError, match="(?i)configuration.*reinitial"):
        _packet(model, parameters, [0.1, 0.0, 0.0])
    restarted = _packet(model, parameters, [0.1, 0.0, 0.0], "initialize")
    np.testing.assert_allclose(restarted["adiabatic_to_diabatic"], np.eye(2), atol=1e-14)


def test_cloned_trajectory_parameters_preserve_and_isolate_frame_history():
    model = _model()
    parameters = {}
    _packet(model, parameters, model.constants.atom_positions, "initialize")
    _packet(model, parameters, [0.2, 0.1, 0.0], "commit")
    branch = copy.deepcopy(parameters)
    probe = [0.5, -0.1, 0.0]
    original = _packet(model, parameters, probe)
    copied = _packet(model, branch, probe)
    np.testing.assert_allclose(original["hamiltonian"], copied["hamiltonian"], atol=1e-14)
    _packet(model, branch, probe, "commit")
    unchanged = _packet(model, parameters, probe)
    assert unchanged["frame_revision"] == original["frame_revision"]
    assert _packet(model, branch, probe)["frame_revision"] == original["frame_revision"] + 1


@pytest.mark.parametrize("etf_arguments", [{}, {"use_etfs": True}])
def test_tda_requires_raw_couplings_compatible_with_snapshot_overlaps(etf_arguments):
    with pytest.raises(ValueError, match="use_etfs=False"):
        _model(calculator_args={"method_es": "tddft", "tda": True, "xc": "pbe", **etf_arguments})


def test_evaluation_before_frame_initialization_is_actionable():
    with pytest.raises(RuntimeError, match="(?i)initializ"):
        _packet(_model(), {}, [0.1, 0.0, 0.0])


def _strict_h_qc(model, parameters, **kwargs):
    positions = functions.z_to_q(kwargs["z"], model.constants.classical_coordinate_mass,
                                  model.constants.classical_coordinate_weight)
    return np.array([_hamiltonian(point)[0] for point in positions])


@functions.make_ingredient_sparse
def _strict_gradient(model, parameters, **kwargs):
    positions = functions.z_to_q(kwargs["z"], model.constants.classical_coordinate_mass,
                                  model.constants.classical_coordinate_weight)
    gradients = np.array([_hamiltonian(point)[1] for point in positions])
    scale = functions.dqdp_to_dzc(np.ones(3), None,
                                  model.constants.classical_coordinate_mass,
                                  model.constants.classical_coordinate_weight)
    return gradients * scale[None, :, None, None]


def _simulation(model, algorithm, dt=0.05, tmax=2.0, electronic=None, strict=False, batch_size=1):
    sim = Simulation({"tmax": tmax, "dt_update": dt, "dt_collect": dt,
                      "num_trajs": batch_size, "batch_size": batch_size, "progress_bar": False})
    sim.model = model
    sim.algorithm = algorithm()
    coefficients = (np.array([np.sqrt(0.7), np.sqrt(0.3) * np.exp(0.4j)])
                    if electronic is None else np.asarray(electronic, dtype=complex))
    if strict:
        _, vectors, _, _ = _electronic_structure(model.constants.atom_positions)
        coefficients = vectors @ coefficients
        model.ingredients.extend([("h_qc", _strict_h_qc), ("dh_qc_dzc", _strict_gradient)])
    key = "wf_adb" if isinstance(sim.algorithm, (MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio)) else "wf_db"
    sim.initial_state[key] = coefficients.copy()
    model.initialize_constants()
    sim.initialize_timesteps()
    state, parameters = sim.algorithm.execute_recipe(
        sim, {"seed": 13 + np.arange(batch_size)}, {}, sim.algorithm.initialization_recipe,
    )
    return sim, state, parameters


def _advance(sim, state, parameters):
    for step in range(sim.settings.tmax_n):
        sim.t_ind = step
        state, parameters = sim.algorithm.execute_recipe(sim, state, parameters, sim.algorithm.update_recipe)
    return state, parameters


def _adiabatic_density(sim, state, parameters, local=False, strict=False):
    if local:
        packet = _packet(sim.model, parameters, functions.z_to_q(
            state["z"], sim.model.constants.classical_coordinate_mass,
            sim.model.constants.classical_coordinate_weight,
        ))
        coefficients = packet["adiabatic_to_diabatic"].conj().T @ state["wf_db"][0]
    elif strict:
        position = functions.z_to_q(state["z"], sim.model.constants.classical_coordinate_mass,
                                     sim.model.constants.classical_coordinate_weight)[0]
        coefficients = _electronic_structure(position)[1].T @ state["wf_db"][0]
    else:
        coefficients = state["wf_adb"][0]
    return np.outer(coefficients, coefficients.conj())


@pytest.mark.parametrize("algorithm", [MeanField, FewestSwitchesSurfaceHopping])
def test_known_complete_diabatic_model_matches_ordinary_algorithm_at_fixed_step(algorithm):
    """A complete two-state frame reproduces a known analytic diabatic baseline."""
    local_sim, local_state, local_parameters = _simulation(_model(), algorithm)
    strict_sim, strict_state, strict_parameters = _simulation(_model(local=False), algorithm, strict=True)
    local_state, local_parameters = _advance(local_sim, local_state, local_parameters)
    strict_state, strict_parameters = _advance(strict_sim, strict_state, strict_parameters)
    # Local FSSH explicitly refreshes the accepted-endpoint hopping gradient;
    # the ordinary recipe retains its last RK trial, a higher-order difference.
    np.testing.assert_allclose(local_state["z"], strict_state["z"], atol=2e-10, rtol=0)
    np.testing.assert_allclose(
        _adiabatic_density(local_sim, local_state, local_parameters, local=True),
        _adiabatic_density(strict_sim, strict_state, strict_parameters, strict=True),
        atol=2e-10, rtol=0,
    )
    if algorithm is FewestSwitchesSurfaceHopping:
        np.testing.assert_array_equal(local_state["act_surf_ind"], strict_state["act_surf_ind"])


def test_mean_field_representation_difference_decreases_under_three_step_halvings():
    """Different existing nuclear/electronic integrators converge to the same equations."""
    errors = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        local_sim, local_state, local_parameters = _simulation(_model(), MeanField, dt=dt)
        ad_sim, ad_state, ad_parameters = _simulation(_model(local=False), MeanFieldAbInitio, dt=dt)
        local_state, local_parameters = _advance(local_sim, local_state, local_parameters)
        ad_state, ad_parameters = _advance(ad_sim, ad_state, ad_parameters)
        local_dm = _adiabatic_density(local_sim, local_state, local_parameters, local=True)
        ad_dm = _adiabatic_density(ad_sim, ad_state, ad_parameters)
        # Populations are invariant to independent electronic-state signs.
        errors.append(np.linalg.norm(np.diag(local_dm - ad_dm)) + np.linalg.norm(local_state["z"] - ad_state["z"]))
    assert all(smaller < 0.8 * larger for larger, smaller in zip(errors[:-1], errors[1:])), errors
    assert errors[-1] < 0.005, errors


def test_mean_field_generic_force_uses_full_gradient_without_double_counting():
    sim, state, parameters = _simulation(_model(), MeanField, tmax=0.1)
    state, parameters = tasks.update_quantum_classical_force(sim, state, parameters)
    _, vectors, _, _ = _electronic_structure(sim.model.constants.atom_positions)
    physical_coefficients = vectors @ state["wf_db"][0]
    gradient = _hamiltonian(sim.model.constants.atom_positions)[1]
    expected_cartesian = np.einsum("i,cij,j->c", physical_coefficients.conj(), gradient, physical_coefficients)
    expected = functions.dqdp_to_dzc(expected_cartesian, None,
                                     sim.model.constants.classical_coordinate_mass,
                                     sim.model.constants.classical_coordinate_weight)
    np.testing.assert_allclose(state["quantum_classical_force"][0], expected, atol=2e-14)
    assert "derivative_coupling_dzc" not in state


def test_ab_initio_fssh_initial_superposition_uses_active_surface_force_and_verlet_drift():
    """The first nuclear half-kick must already follow the selected FSSH surface."""
    sim, state, parameters = _simulation(_model(local=False),
                                         FewestSwitchesSurfaceHoppingAbInitio,
                                         dt=0.05, tmax=0.05)
    model = sim.model
    masses = model.constants.classical_coordinate_mass
    weights = model.constants.classical_coordinate_weight
    initial_z = state["z"].copy()
    initial_q = functions.z_to_q(initial_z, masses, weights)
    initial_p = functions.z_to_p(initial_z, masses, weights)
    _, _, gradient, _ = _electronic_structure(initial_q[0])
    active = state["act_surf_ind"][0]
    active_gradient = gradient[:, active, active]
    expected_force = functions.dqdp_to_dzc(active_gradient, None, masses, weights)
    np.testing.assert_allclose(state["quantum_classical_force"][0],
                               expected_force, atol=2e-14)
    coefficients = state["wf_adb"][0]
    mean_field_gradient = np.einsum("i,cij,j->c", coefficients.conj(), gradient, coefficients)
    # The complex superposition and distinct state gradients make the previous
    # mean-field initialization detectably different from the FSSH force.
    assert np.linalg.norm(mean_field_gradient - active_gradient) > 0.001
    state, parameters = tasks.update_q_velocity_verlet(sim, state, parameters)
    expected_q = (initial_q + sim.settings.dt_update * initial_p / masses
                  - 0.5 * sim.settings.dt_update ** 2 * active_gradient / masses)
    np.testing.assert_allclose(functions.z_to_q(state["z"], masses, weights),
                               expected_q, atol=2e-14)
    np.testing.assert_allclose(functions.z_to_p(state["z"], masses, weights),
                               initial_p, atol=2e-14)


def test_fssh_rescaling_gradient_is_evaluated_at_accepted_endpoint():
    sim, state, parameters = _simulation(_model(), FewestSwitchesSurfaceHopping,
                                         dt=0.5, tmax=0.5, electronic=[1.0, 0.0])
    state["hop_prob_rand_vals"][:] = 1.0
    state, parameters = _advance(sim, state, parameters)
    position = functions.z_to_q(state["z"], sim.model.constants.classical_coordinate_mass,
                                 sim.model.constants.classical_coordinate_weight)
    packet = _packet(sim.model, parameters, position)
    indices, values, shape = state["dh_qc_dzc"]
    actual = np.zeros(shape, dtype=complex)
    actual[indices] = values
    expected = functions.dqdp_to_dzc(packet["gradient"], None,
                                     sim.model.constants.classical_coordinate_mass[:, None, None],
                                     sim.model.constants.classical_coordinate_weight[:, None, None])
    np.testing.assert_allclose(actual[0], expected, atol=2e-14)
    # This is a meaningful endpoint check: the final RK trial is different.
    assert np.linalg.norm(state["z"] - state["z_3"]) > 1e-8


def test_fssh_hopping_flux_converges_with_controlled_random_numbers():
    errors = []
    for dt in (0.05, 0.025, 0.0125, 0.00625):
        outputs = []
        for local, algorithm in ((True, FewestSwitchesSurfaceHopping),
                                 (False, FewestSwitchesSurfaceHoppingAbInitio)):
            sim, state, parameters = _simulation(_model(local=local), algorithm,
                                                 dt=dt, electronic=[1.0, 0.0])
            # Keep the active surface fixed to compare the same physical path;
            # probabilities still use the genuine unmodified electronic flux.
            state["hop_prob_rand_vals"][:] = 1.0
            integrated_probability = 0.0
            for step in range(sim.settings.tmax_n):
                sim.t_ind = step
                state, parameters = sim.algorithm.execute_recipe(
                    sim, state, parameters, sim.algorithm.update_recipe,
                )
                integrated_probability += max(0.0, float(state["hop_prob"][0, 1]))
                assert state["act_surf_ind"][0] == 0
            density = _adiabatic_density(sim, state, parameters, local=local)
            outputs.append((integrated_probability, np.diag(density), state["z"].copy()))
        local_result, ad_result = outputs
        assert local_result[0] > 1e-3
        errors.append(abs(local_result[0] - ad_result[0])
                      + np.linalg.norm(local_result[1] - ad_result[1])
                      + np.linalg.norm(local_result[2] - ad_result[2]))
    assert all(smaller < 0.8 * larger for larger, smaller in zip(errors[:-1], errors[1:])), errors
    assert errors[-1] < 0.01, errors


def test_fssh_fixed_random_grid_ensemble_statistics_converge_across_representations():
    """Paired stochastic estimates converge without requiring identical hop paths."""
    ensemble_size = 16
    finest_dt = 0.015
    tmax = 1.2
    fine_uniforms = np.random.default_rng(303).uniform(size=(ensemble_size, 80))
    representation_errors = []
    estimates = []
    for dt in (0.12, 0.06, 0.03, finest_dt):
        stride = int(round(dt / finest_dt))
        block_minimum = fine_uniforms.reshape(ensemble_size, -1, stride).min(axis=-1)
        # The CDF of min(U_1,...,U_k) is 1-(1-u)^k. This gives independent
        # uniform draws at every step size while coupling refinements to the
        # same fixed underlying grid and allowing ordinary FSSH hop decisions.
        uniform_grid = 1.0 - (1.0 - block_minimum) ** stride
        pair = []
        for local, algorithm in ((True, FewestSwitchesSurfaceHopping),
                                 (False, FewestSwitchesSurfaceHoppingAbInitio)):
            model = _model(local=local, position=(-0.3, 0.2, -0.1),
                           init_momentum=np.array([8.0, 0.1, -0.2]))
            sim, state, parameters = _simulation(model, algorithm, dt=dt, tmax=tmax,
                                                 electronic=[1.0, 0.0], batch_size=ensemble_size)
            state["hop_prob_rand_vals"][:, :uniform_grid.shape[1]] = uniform_grid
            accepted_hops = 0
            for step in range(sim.settings.tmax_n):
                sim.t_ind = step
                state, parameters = sim.algorithm.execute_recipe(
                    sim, state, parameters, sim.algorithm.update_recipe,
                )
                accepted_hops += np.count_nonzero(state["hop_successful"])
            state, parameters = tasks.update_dm_db_fssh(sim, state, parameters)
            # Use the FSSH ensemble estimator (active populations), separately
            # from continuous electronic populations and mean nuclear motion.
            estimate = np.diag(state["dm_adb"].mean(axis=0)).real
            electronic_population = (np.abs(state["wf_adb"]) ** 2).mean(axis=0)
            np.testing.assert_allclose(estimate.sum(), 1.0, atol=1e-13)
            assert np.all(estimate > 0.0), estimate
            assert accepted_hops > 0
            pair.append((estimate, electronic_population, state["z"].mean(axis=0)))
        # Finite stochastic samples resolve active populations in increments of
        # 1/N. Neither trajectory identities nor individual hop times are tested.
        np.testing.assert_allclose(pair[0][0], pair[1][0], atol=1.0 / ensemble_size, rtol=0)
        representation_errors.append(np.linalg.norm(pair[0][1] - pair[1][1])
                                     + np.linalg.norm(pair[0][2] - pair[1][2]))
        estimates.append(pair)
    assert all(smaller < 0.8 * larger for larger, smaller in
               zip(representation_errors[:-1], representation_errors[1:])), representation_errors
    assert representation_errors[-1] < 0.005, representation_errors
    for representation in (0, 1):
        np.testing.assert_allclose(estimates[-1][representation][0],
                                   estimates[-2][representation][0],
                                   atol=1.0 / ensemble_size, rtol=0)


def test_controlled_fssh_hop_runs_through_full_recipe_and_commits_endpoint():
    sim, state, parameters = _simulation(_model(), FewestSwitchesSurfaceHopping,
                                         dt=0.05, tmax=0.05, electronic=[1.0, 0.0])
    initial_revision = state["local_diabatic_property"][0]["frame_revision"]
    state["hop_prob_rand_vals"][:] = 0.0
    state, parameters = _advance(sim, state, parameters)
    assert state["hop_prob"][0, 1] > 0
    assert state["hop_successful"][0]
    assert state["act_surf_ind"][0] == 1
    assert state["local_diabatic_property"][0]["frame_revision"] == initial_revision + 1
    np.testing.assert_allclose(np.linalg.norm(state["wf_db"][0]), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(state["act_surf_wf"][0]), 1.0, atol=1e-13)


@pytest.mark.parametrize("algorithm", [MeanField, FewestSwitchesSurfaceHopping])
def test_driver_collects_physical_adiabatic_density_instead_of_local_frame_average(algorithm):
    sim, _, _ = _simulation(_model(), algorithm, dt=0.05, tmax=0.1)
    sim.settings.num_trajs = 2
    sim.settings.batch_size = 2
    result = serial_driver(sim)
    assert "dm_adb" in result.data_dict
    assert "dm_db" not in result.data_dict
    density = result.data_dict["dm_adb"]
    np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(density, density.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.all(np.isfinite(result.data_dict["quantum_energy"]))
    assert np.all(np.isfinite(result.data_dict["classical_energy"]))


@pytest.mark.parametrize("momentum,successful", [(5.0, True), (0.01, False)])
def test_fssh_hop_direction_and_energy_conservation_from_transformed_gradient(momentum, successful):
    model = _model(init_momentum=np.array([momentum, 0.0, 0.0]))
    sim, state, parameters = _simulation(model, FewestSwitchesSurfaceHopping,
                                         electronic=[1.0, 0.0], tmax=0.1)
    state["act_surf_ind"][:] = 0
    state["hop_ind"] = np.array([0])
    state["hop_dest"] = np.array([1])
    before = state["z"].copy()
    state, parameters = tasks.update_dh_qc_dzc(sim, state, parameters)
    state, parameters = tasks.update_hop_vals_fssh(sim, state, parameters)
    assert bool(state["hop_successful"][0]) is successful
    _, _, _, coupling = _electronic_structure(model.constants.atom_positions)
    expected_direction = functions.dqdp_to_dzc(
        coupling[:, 0, 1], None, model.constants.classical_coordinate_mass,
        model.constants.classical_coordinate_weight,
    )
    # Eigenvector phase choices can reverse the entire rescaling direction.
    actual_direction = state["resc_dir_z_traj"]
    np.testing.assert_allclose(np.abs(actual_direction), np.abs(expected_direction), atol=2e-14)
    if successful:
        after = before + state["z_shift"]
        mass = model.constants.classical_coordinate_mass
        weight = model.constants.classical_coordinate_weight
        initial_p = functions.z_to_p(before, mass, weight)
        final_p = functions.z_to_p(after, mass, weight)
        gap = state["eigvals"][0, 1] - state["eigvals"][0, 0]
        np.testing.assert_allclose(np.sum((initial_p ** 2 - final_p ** 2) / (2 * mass)), gap, atol=2e-13)
        np.testing.assert_allclose(functions.z_to_q(after, mass, weight), functions.z_to_q(before, mass, weight), atol=1e-14)
    else:
        np.testing.assert_array_equal(state["z_shift"], np.zeros_like(state["z_shift"]))
