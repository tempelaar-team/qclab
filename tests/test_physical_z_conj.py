"""
This module contains tests of the physical z-conjugation matrix initialization Tasks.
"""

import logging
from functools import partial
import numpy as np
import pytest
from qclab import Simulation, Model, tasks, ingredients, functions
from qclab.models import (
    SpinBoson,
    HolsteinLattice,
    FMOComplex,
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
)
from qclab.algorithms import MeanField
from qclab.dynamics import serial_driver
from qclab.tasks import (
    initialize_z_probe,
    initialize_physical_z_conj_matrix,
)

TOLERANCE = 1e-4
BATCH_SIZE = 4
# Recipe entries that determine the physical z-conjugation matrix, to be placed
# after the initialization of the classical coordinates.
PHYSICAL_Z_CONJ_RECIPE = [
    tasks.update_dh_qc_dzc,
    initialize_z_probe,
    partial(
        tasks.update_classical_force,
        z_name="z_probe",
        classical_force_name="classical_force_probe",
    ),
    initialize_physical_z_conj_matrix,
]


def _random_unitary(size, seed):
    """
    Random unitary matrix.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    unitary, upper = np.linalg.qr(matrix)
    return unitary * (np.diag(upper) / np.abs(np.diag(upper)))


def _rotate_model(model, unitary, keep_dh_c_dzc=False):
    """
    Express a Model in the classical basis z = U z', where z' is the basis in which
    the Model was defined. Only the Hamiltonians are provided in the new basis,
    optionally with the analytical gradient of the classical Hamiltonian.
    """
    h_c, _ = model.get("h_c")
    h_qc, _ = model.get("h_qc")
    dh_c_dzc, has_dh_c_dzc = model.get("dh_c_dzc")
    unitary_conj = unitary.conj()

    def h_c_rotated(model, parameters, **kwargs):
        return h_c(model, parameters, z=kwargs["z"] @ unitary_conj)

    def h_qc_rotated(model, parameters, **kwargs):
        return h_qc(model, parameters, z=kwargs["z"] @ unitary_conj)

    def dh_c_dzc_rotated(model, parameters, **kwargs):
        return dh_c_dzc(model, parameters, z=kwargs["z"] @ unitary_conj) @ unitary.T

    model.ingredients.append(("h_c", h_c_rotated))
    model.ingredients.append(("h_qc", h_qc_rotated))
    model.ingredients.append(("dh_qc_dzc", None))
    if keep_dh_c_dzc and has_dh_c_dzc:
        model.ingredients.append(("dh_c_dzc", dh_c_dzc_rotated))
    else:
        model.ingredients.append(("dh_c_dzc", None))
    model.update_dh_qc_dzc = True
    return model


def _execute_physical_z_conj_recipe(model, seed=0, recipe=PHYSICAL_Z_CONJ_RECIPE):
    """
    Execute the Recipe entries on random classical coordinates, returning the physical
    z-conjugation matrix, the coordinates, and the Simulation object.
    """
    sim = Simulation({"batch_size": BATCH_SIZE})
    sim.model = model
    sim.algorithm = MeanField()
    num_classical_coordinates = model.constants.num_classical_coordinates
    rng = np.random.default_rng(seed)
    shape = (BATCH_SIZE, num_classical_coordinates)
    z = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    state, _ = sim.algorithm.execute_recipe(sim, {"z": z}, {}, recipe)
    return state["physical_z_conj_matrix"], z, sim


def _assert_valid(physical_z_conj_matrix, z, sim):
    """
    Assert the defining properties of a physical z-conjugation matrix.
    """
    identity = np.eye(len(physical_z_conj_matrix))
    assert np.allclose(physical_z_conj_matrix, physical_z_conj_matrix.T, atol=TOLERANCE)
    assert np.allclose(
        physical_z_conj_matrix @ physical_z_conj_matrix.conj().T,
        identity,
        atol=TOLERANCE,
    )
    physical_z_conj = z.conj() @ physical_z_conj_matrix.T
    z_physical_position = 0.5 * (z + physical_z_conj)
    h_c, _ = sim.model.get("h_c")
    h_qc, _ = sim.model.get("h_qc")
    h_c_val = h_c(sim.model, {}, z=z)
    h_qc_val = h_qc(sim.model, {}, z=z)
    # Reversing momenta leaves both Hamiltonians unchanged.
    assert np.allclose(h_c(sim.model, {}, z=physical_z_conj), h_c_val, rtol=TOLERANCE)
    assert np.allclose(
        h_qc(sim.model, {}, z=physical_z_conj),
        h_qc_val,
        atol=TOLERANCE * np.abs(h_qc_val).max(),
    )
    # The quantum–classical interaction Hamiltonian depends on the physical position
    # component only.
    assert np.allclose(
        h_qc(sim.model, {}, z=z_physical_position),
        h_qc_val,
        atol=TOLERANCE * np.abs(h_qc_val).max(),
    )


class SpinBosonDegenerate(SpinBoson):
    """
    Spin-boson Model class with pairwise degenerate bath frequencies.
    """

    def _init_h_c(self, parameters, **kwargs):
        A = self.constants.get("A")
        self.constants.harmonic_frequency = np.repeat(0.1 * np.arange(1, A // 2 + 1), 2)

    ingredients = [
        ingredient
        for ingredient in SpinBoson.ingredients
        if ingredient[0] != "_init_h_c"
    ] + [("_init_h_c", _init_h_c)]


def h_q_zero(model, parameters, **kwargs):
    """
    Vanishing quantum Hamiltonian of a two-level system.
    """
    return np.zeros((kwargs["batch_size"], 2, 2), dtype=complex)


def h_qc_momentum_dependent(model, parameters, **kwargs):
    """
    Quantum–classical interaction Hamiltonian coupling the real part of z to sigma_z
    and the imaginary part of z to sigma_x.
    """
    z = kwargs["z"]
    h_qc = np.zeros((len(z), 2, 2), dtype=complex)
    h_qc[:, 0, 0] = z[:, 0].real
    h_qc[:, 1, 1] = -z[:, 0].real
    h_qc[:, 0, 1] = z[:, 0].imag
    h_qc[:, 1, 0] = z[:, 0].imag
    return h_qc


class MomentumCoupledModel(Model):
    """
    Model class with a momentum-dependent quantum–classical interaction Hamiltonian.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {"kBT": 1.0}
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states = 2
        self.constants.num_classical_coordinates = 1
        self.constants.classical_coordinate_weight = np.ones(1)
        self.constants.classical_coordinate_mass = np.ones(1)
        self.constants.harmonic_frequency = np.ones(1)

    ingredients = [
        ("h_q", h_q_zero),
        ("h_qc", h_qc_momentum_dependent),
        ("h_c", ingredients.h_c_harmonic),
        ("_init_model", _init_model),
    ]


SHIPPED_MODELS = [
    lambda: SpinBoson({"A": 20}),
    lambda: HolsteinLattice({"N": 8}),
    lambda: FMOComplex({"N": 5}),
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
]


@pytest.mark.parametrize("make_model", SHIPPED_MODELS)
def test_physical_basis_gives_identity(make_model):
    """
    Models shipped with QC Lab are defined in a physical basis, for which S is the
    identity.
    """
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(make_model())
    assert np.allclose(
        physical_z_conj_matrix, np.eye(len(physical_z_conj_matrix)), atol=TOLERANCE
    )
    _assert_valid(physical_z_conj_matrix, z, sim)


@pytest.mark.parametrize("make_model", SHIPPED_MODELS)
@pytest.mark.parametrize("keep_dh_c_dzc", [False, True])
def test_rotated_basis_recovers_matrix(make_model, keep_dh_c_dzc):
    """
    For a Model rotated by a random unitary U, the Tasks recover S = U U^T.
    """
    model = make_model()
    unitary = _random_unitary(model.constants.num_classical_coordinates, seed=1)
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(
        _rotate_model(model, unitary, keep_dh_c_dzc)
    )
    assert np.allclose(physical_z_conj_matrix, unitary @ unitary.T, atol=TOLERANCE)
    _assert_valid(physical_z_conj_matrix, z, sim)


def test_reciprocal_space_basis():
    """
    For a discrete Fourier transform, S pairs the coordinates k and -k.
    """
    num_sites = 8
    sites = np.arange(num_sites)
    unitary = np.exp(2j * np.pi * np.outer(sites, sites) / num_sites) / np.sqrt(
        num_sites
    )
    model = _rotate_model(HolsteinLattice({"N": num_sites}), unitary)
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(model)
    expected = np.zeros((num_sites, num_sites))
    expected[sites, (-sites) % num_sites] = 1.0
    assert np.allclose(physical_z_conj_matrix, expected, atol=TOLERANCE)
    _assert_valid(physical_z_conj_matrix, z, sim)


@pytest.mark.parametrize("rotate", [False, True])
def test_degenerate_frequencies_give_valid_matrix(rotate):
    """
    With degenerate bath frequencies, S is not unique, but the Tasks return a valid
    one.
    """
    model = SpinBosonDegenerate({"A": 20})
    if rotate:
        model = _rotate_model(model, _random_unitary(20, seed=2))
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(model)
    _assert_valid(physical_z_conj_matrix, z, sim)


def _randomize_phases(monkeypatch, seed):
    """
    Replace the singular value and eigenvalue decompositions by versions that return
    singular vectors and eigenvectors with random phases.
    """
    rng = np.random.default_rng(seed)
    svd, eigh = np.linalg.svd, np.linalg.eigh

    def random_phases(size):
        return np.exp(2j * np.pi * rng.random(size))

    def svd_random_phases(matrix, *args, **kwargs):
        left, singvals, right_dagger = svd(matrix, *args, **kwargs)
        left_phases = random_phases(left.shape[1])
        right_phases = random_phases(right_dagger.shape[0])
        # Phases of paired singular vectors must match to leave the matrix unchanged.
        right_phases[: len(singvals)] = left_phases[: len(singvals)]
        return left * left_phases, singvals, right_phases.conj()[:, None] * right_dagger

    def eigh_random_phases(matrix, *args, **kwargs):
        eigvals, eigvecs = eigh(matrix, *args, **kwargs)
        return eigvals, eigvecs * random_phases(eigvecs.shape[1])

    monkeypatch.setattr(np.linalg, "svd", svd_random_phases)
    monkeypatch.setattr(np.linalg, "eigh", eigh_random_phases)


@pytest.mark.parametrize("make_model", SHIPPED_MODELS)
@pytest.mark.parametrize("seed", [8, 9])
def test_independent_of_eigenvector_phases(monkeypatch, make_model, seed):
    """
    The physical z-conjugation matrix does not depend on the arbitrary phases of the
    singular vectors and eigenvectors used to determine it.
    """
    model = make_model()
    unitary = _random_unitary(model.constants.num_classical_coordinates, seed=seed)
    _randomize_phases(monkeypatch, seed)
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(
        _rotate_model(model, unitary)
    )
    assert np.allclose(physical_z_conj_matrix, unitary @ unitary.T, atol=TOLERANCE)
    _assert_valid(physical_z_conj_matrix, z, sim)


def test_degenerate_frequencies_with_random_phases(monkeypatch):
    """
    With degenerate bath frequencies, random phases of the eigenvectors still yield a
    valid physical z-conjugation matrix.
    """
    model = _rotate_model(SpinBosonDegenerate({"A": 20}), _random_unitary(20, seed=10))
    _randomize_phases(monkeypatch, 10)
    physical_z_conj_matrix, z, sim = _execute_physical_z_conj_recipe(model)
    _assert_valid(physical_z_conj_matrix, z, sim)


def test_physical_position_and_momentum_components():
    """
    The physical z-conjugation matrix reproduces the kinetic energy and an isotropic
    rescaling of the physical momentum as evaluated in the physical basis.
    """
    model = SpinBoson({"A": 20})
    weight = model.constants.classical_coordinate_weight
    unitary = _random_unitary(20, seed=3)
    physical_z_conj_matrix, _, sim = _execute_physical_z_conj_recipe(
        _rotate_model(model, unitary)
    )
    rng = np.random.default_rng(4)
    z_physical = rng.normal(size=(3, 20)) + 1j * rng.normal(size=(3, 20))
    z = z_physical @ unitary.T
    z_physical_position = 0.5 * (z + z.conj() @ physical_z_conj_matrix.T)
    z_physical_momentum = 0.5 * (z - z.conj() @ physical_z_conj_matrix.T)
    h_c, _ = sim.model.get("h_c")
    kinetic_energy = h_c(sim.model, {}, z=z) - h_c(sim.model, {}, z=z_physical_position)
    assert np.allclose(
        kinetic_energy, np.sum(weight * z_physical.imag**2, axis=1), rtol=TOLERANCE
    )
    z_rescaled = (z_physical.real + 0.7j * z_physical.imag) @ unitary.T
    assert np.allclose(
        z_physical_position + 0.7 * z_physical_momentum, z_rescaled, atol=TOLERANCE
    )


def test_ingredient_overrides_numerical_determination(caplog):
    """
    A physical_z_conj_matrix Ingredient is used as given, here for a Model whose
    matrix cannot be determined numerically.
    """
    supplied = np.array([[1j]])
    model = MomentumCoupledModel()
    model.ingredients.append(
        ("physical_z_conj_matrix", lambda model, parameters, **kwargs: supplied)
    )
    with caplog.at_level(logging.WARNING):
        physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(model)
    assert np.array_equal(physical_z_conj_matrix, supplied)
    assert len(caplog.records) == 0


def test_invalid_ingredient_logs_warning(caplog):
    """
    An Ingredient that is not symmetric and unitary is used, but a warning is logged.
    """
    supplied = np.array([[2.0]])
    model = MomentumCoupledModel()
    model.ingredients.append(
        ("physical_z_conj_matrix", lambda model, parameters, **kwargs: supplied)
    )
    with caplog.at_level(logging.WARNING):
        physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(model)
    assert np.array_equal(physical_z_conj_matrix, supplied)
    assert "not symmetric and unitary" in caplog.text


def test_momentum_dependent_coupling_logs_warning(caplog):
    """
    For a momentum-dependent quantum–classical interaction Hamiltonian no valid
    matrix exists; a warning is logged and the identity is used.
    """
    with caplog.at_level(logging.WARNING):
        physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(
            MomentumCoupledModel()
        )
    assert np.array_equal(physical_z_conj_matrix, np.eye(1))
    assert "could not be determined" in caplog.text


def test_task_in_initialization_recipe():
    """
    The Tasks are executed as part of an initialization Recipe, after the classical
    coordinates have been initialized, and their output remains in the State object.
    """
    captured = {}

    def capture_physical_z_conj_matrix(sim, state, parameters):
        captured["matrix"] = state["physical_z_conj_matrix"]
        return state, parameters

    sim = Simulation({"num_trajs": 4, "batch_size": 4, "tmax": 0.1, "dt_update": 0.01})
    sim.model = SpinBoson({"A": 20})
    sim.algorithm = MeanField()
    sim.algorithm.initialization_recipe = (
        sim.algorithm.initialization_recipe + PHYSICAL_Z_CONJ_RECIPE
    )
    sim.algorithm.update_recipe = sim.algorithm.update_recipe + [
        capture_physical_z_conj_matrix
    ]
    sim.initial_state["wf_db"] = np.array([1.0, 0.0], dtype=complex)
    serial_driver(sim)
    assert np.allclose(captured["matrix"], np.eye(20), atol=TOLERANCE)


def test_large_models():
    """
    The Tasks remain tractable for many classical coordinates.
    """
    physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(
        HolsteinLattice({"N": 200})
    )
    assert np.allclose(physical_z_conj_matrix, np.eye(200), atol=TOLERANCE)
    unitary = _random_unitary(300, seed=5)
    model = _rotate_model(SpinBoson({"A": 300}), unitary, keep_dh_c_dzc=True)
    physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(model)
    assert np.allclose(physical_z_conj_matrix, unitary @ unitary.T, atol=TOLERANCE)


def test_probe_coordinates_only_when_needed():
    """
    Probe coordinates for the classical Hessian are written only if the gradient of
    the quantum–classical interaction Hamiltonian leaves directions undetermined.
    """
    for model, num_probes in (
        (HolsteinLattice({"N": 8}), 0),
        (TullyProblemOne(), 0),
        (SpinBoson({"A": 20}), 80),
        (_rotate_model(SpinBoson({"A": 20}), _random_unitary(20, seed=6)), 80),
    ):
        sim = Simulation({"batch_size": BATCH_SIZE})
        sim.model = model
        num_classical_coordinates = model.constants.num_classical_coordinates
        state = {"z": np.ones((BATCH_SIZE, num_classical_coordinates), dtype=complex)}
        parameters = {}
        for task in PHYSICAL_Z_CONJ_RECIPE[:3]:
            state, parameters = task(sim, state, parameters)
        assert state["z_probe"].shape == (num_probes, num_classical_coordinates)
        assert state["classical_force_probe"].shape == state["z_probe"].shape


def test_without_probe_recipe_entries(caplog):
    """
    Without the probe Recipe entries, the gradient of the quantum–classical
    interaction Hamiltonian alone is used; a warning is logged if that does not
    suffice.
    """
    recipe = [tasks.update_dh_qc_dzc, initialize_physical_z_conj_matrix]
    unitary = _random_unitary(8, seed=7)
    model = _rotate_model(HolsteinLattice({"N": 8}), unitary)
    physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(model, recipe=recipe)
    assert np.allclose(physical_z_conj_matrix, unitary @ unitary.T, atol=TOLERANCE)
    model = _rotate_model(SpinBoson({"A": 20}), _random_unitary(20, seed=7))
    with caplog.at_level(logging.WARNING):
        physical_z_conj_matrix, _, _ = _execute_physical_z_conj_recipe(
            model, recipe=recipe
        )
    assert np.array_equal(physical_z_conj_matrix, np.eye(20))
    assert "could not be determined" in caplog.text


def test_physical_position_and_momentum_component_functions():
    """
    In a physical basis, the physical position and momentum components are the real
    part and i times the imaginary part; in a rotated basis they transform with the
    rotation.
    """
    rng = np.random.default_rng(11)
    vec = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    identity = np.eye(5)
    assert np.allclose(
        functions.calc_z_physical_position_component(vec, identity), vec.real
    )
    assert np.allclose(
        functions.calc_z_physical_momentum_component(vec, identity), 1j * vec.imag
    )
    unitary = _random_unitary(5, seed=12)
    physical_z_conj_matrix = unitary @ unitary.T
    # Row vectors: z = z' @ U^T.
    assert np.allclose(
        functions.calc_z_physical_position_component(
            vec @ unitary.T, physical_z_conj_matrix
        ),
        vec.real @ unitary.T,
    )
    assert np.allclose(
        functions.calc_z_physical_momentum_component(
            vec @ unitary.T, physical_z_conj_matrix
        ),
        1j * vec.imag @ unitary.T,
    )


def test_z_physical_momentum_reversal():
    """
    In a physical basis, the reversal of the physical momentum reproduces the
    reflection of the mass-weighted momentum along the direction and conserves the
    classical energy; in a rotated basis it transforms with the rotation.
    """
    model = SpinBoson({"A": 6})
    mass = model.constants.classical_coordinate_mass
    weight = model.constants.classical_coordinate_weight
    rng = np.random.default_rng(13)
    z = rng.normal(size=6) + 1j * rng.normal(size=6)
    direction = rng.normal(size=6) + 1j * rng.normal(size=6)
    z_reversed = functions.calc_z_physical_momentum_reversal(
        model, {}, z, direction, np.eye(6)
    )
    # Reflect p / sqrt(m) along the mass-weighted q form of the direction.
    momentum = functions.z_to_p(z, mass, weight) / np.sqrt(mass)
    direction_q = np.sqrt(2.0 * mass * weight) * direction.real / np.sqrt(mass)
    momentum -= (
        2.0 * np.dot(momentum, direction_q) / np.dot(direction_q, direction_q)
    ) * direction_q
    expected = functions.qp_to_z(
        functions.z_to_q(z, mass, weight), momentum * np.sqrt(mass), mass, weight
    )
    assert np.allclose(z_reversed, expected, rtol=0.0, atol=1e-10)
    h_c, _ = model.get("h_c")
    assert np.isclose(
        h_c(model, {}, z=z_reversed[None])[0], h_c(model, {}, z=z[None])[0]
    )
    unitary = _random_unitary(6, seed=14)
    rotated = _rotate_model(SpinBoson({"A": 6}), unitary)
    z_reversed_rotated = functions.calc_z_physical_momentum_reversal(
        rotated, {}, unitary @ z, unitary @ direction, unitary @ unitary.T
    )
    assert np.allclose(z_reversed_rotated, unitary @ z_reversed, rtol=0.0, atol=1e-10)
