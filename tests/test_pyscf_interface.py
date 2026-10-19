"""Physical and wavefunction-overlap checks for the optional PySCF backend."""

import copy
import pickle

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci, gto, lib, mcscf, scf

from qclab import Simulation, functions, numerical_constants
from qclab.algorithms import (
    FewestSwitchesSurfaceHoppingAbInitio,
    MeanFieldAbInitio,
)
from qclab.interfaces.pyscf import (
    QCLabPySCFInterface,
    determinant_overlaps,
    wavefunction_overlaps,
)
from qclab.models import AbInitio


LIH_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
LIH_MASSES = np.array([7.0, 1.0]) * numerical_constants.AMU_TO_EMASS
SCF_OPTIONS = {"conv_tol": 1e-12}
CASSCF_OPTIONS = {
    "conv_tol": 1e-10,
    "conv_tol_grad": 1e-7,
    "ah_conv_tol": 1e-14,
    "max_cycle_macro": 100,
}


def _calculator(atom_positions=None, **kwargs):
    options = {
        "atom_positions": LIH_POSITIONS.copy(),
        "atom_masses": LIH_MASSES.copy(),
        "atom_names": ["Li", "H"],
        "num_quantum_states": 2,
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "scf_options": dict(SCF_OPTIONS),
        "casscf_options": dict(CASSCF_OPTIONS),
        "num_threads": 1,
    }
    if atom_positions is not None:
        options["atom_positions"] = atom_positions
    options.update(kwargs)
    return QCLabPySCFInterface(**options)


@pytest.fixture(scope="module")
def lih_result():
    """One converged two-singlet calculation, including response properties."""
    calculator = _calculator()
    result = calculator.calculate(
        {
            "energy": {"excited_amplitudes": True},
            "gradient": {"state_inds_gradient": None},
            "derivative_coupling": {"state_inds_derivative_coupling": None},
        }
    )
    return calculator, copy.deepcopy(result)


@pytest.fixture(scope="module")
def lih_displaced_results():
    """Independent displaced calculations for numerical response checks."""
    delta = 1e-3
    results = []
    for sign in [-1, 1]:
        positions = LIH_POSITIONS.copy()
        positions[1, 2] += sign * delta
        results.append(
            _calculator(positions).calculate(
                {"energy": {"excited_amplitudes": True}}
            )
        )
    return delta, results[0], results[1]


def _embed_ci(ci, ncore, ncas, nelec):
    """Small-space oracle using PySCF's full determinant indexing."""
    norb = ncore + ncas
    core_string = (1 << ncore) - 1
    addresses = []
    for count in nelec:
        active = fci.cistring.make_strings(range(ncas), count)
        full_strings = (active << ncore) | core_string
        addresses.append(
            fci.cistring.strs2addr(norb, ncore + count, full_strings)
        )
    full = np.zeros(
        tuple(fci.cistring.num_strings(norb, ncore + n) for n in nelec)
    )
    full[np.ix_(*addresses)] = ci
    return full


@pytest.mark.parametrize("ncore", [0, 2])
def test_core_aware_overlap_matches_full_fci_oracle(ncore):
    """Core-active mixing and unequal spin populations retain all electrons."""
    rng = np.random.default_rng(123)
    ncas, nelec = 4, (2, 1)
    mo_overlap = rng.normal(size=(ncore + ncas, ncore + ncas))
    ci_shape = tuple(fci.cistring.num_strings(ncas, n) for n in nelec)
    bra, ket = rng.normal(size=(2, *ci_shape))
    alpha, beta = determinant_overlaps(mo_overlap, ncore, ncas, nelec)
    actual = np.einsum("ab,ac,bd,cd->", bra, alpha, beta, ket, optimize=True)
    expected = fci.addons.overlap(
        _embed_ci(bra, ncore, ncas, nelec),
        _embed_ci(ket, ncore, ncas, nelec),
        ncore + ncas,
        tuple(ncore + n for n in nelec),
        s=mo_overlap,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)


def test_singular_core_overlap_remains_well_defined():
    """An invertible full occupied block can have a singular core block."""
    mo_overlap = np.array([[0.0, 1.0], [1.0, 0.0]])
    alpha, beta = determinant_overlaps(mo_overlap, 1, 1, (1, 1))
    np.testing.assert_allclose(alpha, [[-1.0]])
    np.testing.assert_allclose(beta, [[-1.0]])
    np.testing.assert_allclose(alpha * beta, [[1.0]])


def test_empty_active_spin_sector_keeps_core_overlap():
    """Zero active electrons still carry the occupied core determinant."""
    mo_overlap = np.diag([0.7, 0.8, 0.9])
    alpha, beta = determinant_overlaps(mo_overlap, 1, 2, (1, 0))
    np.testing.assert_allclose(alpha, np.diag([0.56, 0.63]))
    np.testing.assert_allclose(beta, [[0.7]])


def test_inactive_and_active_orbital_rotations_preserve_wavefunction():
    """Changing an orbital representation must preserve the physical roots."""
    mol = gto.M(atom="O 0 0 0", charge=2, basis="sto-3g", verbose=0)
    eigenvalues, eigenvectors = np.linalg.eigh(mol.intor("int1e_ovlp"))
    orbitals = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    rng = np.random.default_rng(47)
    root_vectors, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    previous = {
        "mol": mol.dumps(),
        "mo_coeff": orbitals,
        "ncore": 2,
        "ncas": 2,
        "nelecas": (1, 1),
        "ci": [root_vectors[:, root].reshape(2, 2) for root in range(2)],
    }
    inactive_rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    active_rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    rotation = np.eye(orbitals.shape[1])
    rotation[:2, :2] = inactive_rotation
    rotation[2:4, 2:4] = active_rotation
    current = copy.deepcopy(previous)
    current["mo_coeff"] = orbitals @ rotation
    current["ci"] = [
        fci.addons.transform_ci_for_orbital_rotation(ci, 2, (1, 1), active_rotation)
        for ci in previous["ci"]
    ]
    np.testing.assert_allclose(
        wavefunction_overlaps(previous, current), np.eye(2), atol=2e-12
    )


def test_lih_energies_match_direct_state_averaged_pyscf(lih_result):
    """The calculator returns root energies rather than the weighted average."""
    _, result = lih_result
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["Li", "H"], LIH_POSITIONS)),
            unit="Bohr",
            basis="sto-3g",
            verbose=0,
        )
        reference_hf = scf.RHF(mol).set(**SCF_OPTIONS).run()
        reference = mcscf.CASSCF(reference_hf, 2, 2)
        reference.fcisolver = fci.direct_spin0.FCI(mol)
        reference.state_average_([0.5, 0.5])
        reference.set(**CASSCF_OPTIONS).run()
    assert reference.converged
    assert np.shape(result["energy"]) == (2,)
    np.testing.assert_allclose(result["energy"], reference.e_states, atol=2e-8)
    assert result["energy"][1] > result["energy"][0]


def test_lih_state_gradients_match_energy_finite_differences(
    lih_result, lih_displaced_results
):
    _, result = lih_result
    delta, minus, plus = lih_displaced_results
    assert result["gradient"].shape == (2, 3, 2)
    finite_difference = (plus["energy"] - minus["energy"]) / (2 * delta)
    np.testing.assert_allclose(
        result["gradient"][1, 2], finite_difference, atol=3e-5, rtol=2e-3
    )
    np.testing.assert_allclose(np.sum(result["gradient"], axis=0), 0, atol=2e-6)


def test_lih_derivative_coupling_matches_overlap_finite_difference(
    lih_result, lih_displaced_results
):
    """Overlap derivatives check NAC direction, phase, and Angstrom conversion."""
    _, result = lih_result
    delta, minus, plus = lih_displaced_results
    reference = result["excited_amplitudes"]
    overlap_minus = wavefunction_overlaps(reference, minus["excited_amplitudes"])
    overlap_plus = wavefunction_overlaps(reference, plus["excited_amplitudes"])
    overlap_minus *= np.sign(np.diag(overlap_minus))[None, :]
    overlap_plus *= np.sign(np.diag(overlap_plus))[None, :]
    finite_difference = (overlap_plus - overlap_minus) / (2 * delta)
    coupling = result["derivative_coupling"][(0, 1)]
    assert coupling.shape == (2, 3)
    coupling_per_bohr = coupling / numerical_constants.ANGSTROM_TO_BOHR
    np.testing.assert_allclose(
        coupling_per_bohr[1, 2], finite_difference[0, 1], atol=2e-4, rtol=5e-3
    )


def test_wavefunction_snapshots_preserve_phase_orientation_and_serialization(
    lih_result, lih_displaced_results
):
    _, result = lih_result
    _, _, plus = lih_displaced_results
    previous = result["excited_amplitudes"]
    current = plus["excited_amplitudes"]
    assert isinstance(previous["mol"], str)
    assert previous["ncore"] == 1
    assert previous["ncas"] == 2
    restored = pickle.loads(pickle.dumps(previous))
    np.testing.assert_allclose(
        wavefunction_overlaps(previous, restored), np.eye(2), atol=2e-10
    )
    forward = wavefunction_overlaps(previous, current)
    np.testing.assert_allclose(
        wavefunction_overlaps(current, previous), forward.conj().T, atol=2e-10
    )
    flipped = copy.deepcopy(current)
    flipped["ci"] = [np.array(ci, copy=True) for ci in flipped["ci"]]
    flipped["ci"][1] *= -1
    np.testing.assert_allclose(
        wavefunction_overlaps(previous, flipped),
        forward * np.array([1.0, -1.0])[None, :],
        atol=2e-10,
    )


def test_requested_gradient_subset_keeps_state_order(lih_result):
    _, result = lih_result
    selected = _calculator().calculate({"gradient": {"state_inds_gradient": [1]}})
    assert selected["gradient"].shape == (2, 3, 2)
    np.testing.assert_array_equal(selected["gradient"][:, :, 0], 0)
    np.testing.assert_allclose(
        selected["gradient"][:, :, 1], result["gradient"][:, :, 1], atol=2e-7
    )


def test_requested_reverse_derivative_coupling_is_antisymmetric(lih_result):
    calculator, reference = lih_result
    selected = calculator.calculate(
        {"derivative_coupling": {"state_inds_derivative_coupling": [1, 0]}}
    )
    assert set(selected["derivative_coupling"]) == {(1, 0)}
    np.testing.assert_allclose(
        selected["derivative_coupling"][(1, 0)],
        -reference["derivative_coupling"][(0, 1)],
        atol=1e-12,
    )


def test_previous_snapshot_controls_phase_across_three_geometries(lih_result):
    """An arbitrary initial sign must survive successive orbital projections."""
    _, result = lih_result
    previous = copy.deepcopy(result["excited_amplitudes"])
    previous["ci"] = [np.array(ci, copy=True) for ci in previous["ci"]]
    previous["ci"][0] *= -1
    calculator = _calculator()
    for shift in [0.001, -0.001, 0.0]:
        positions = LIH_POSITIONS.copy()
        positions[1, 2] += shift
        saved_ci = [ci.copy() for ci in previous["ci"]]
        current = calculator.calculate(
            {
                "energy": {"excited_amplitudes": True},
                "derivative_coupling": {"state_inds_derivative_coupling": None},
                "wf_overlaps": {"amplitudes_previous": previous},
            },
            atom_positions=positions,
        )
        assert np.all(np.diag(current["wf_overlaps"]) > 0.99)
        for old, retained in zip(saved_ci, previous["ci"]):
            np.testing.assert_array_equal(old, retained)
        previous = current["excited_amplitudes"]
    # Flipping only root zero reverses the final raw NAC relative to the
    # independently initialized calculation at the same reference geometry.
    np.testing.assert_allclose(
        current["derivative_coupling"][(0, 1)],
        -result["derivative_coupling"][(0, 1)],
        atol=2e-6,
        rtol=2e-4,
    )


def test_lih_numerical_normal_mode_uses_electron_masses():
    """Only the internal stretch survives rigid-body projection for a diatomic."""
    calculator = _calculator()
    modes = calculator.normal_modes(state_ind=0, delta=1e-3)
    frequencies = np.asarray(modes["harmonic_frequency"])
    displacement = np.asarray(modes["normal_mode"])
    assert frequencies.shape == (1,)
    assert displacement.shape == (6, 1)
    assert np.isrealobj(frequencies)
    assert frequencies[0] > 0
    assert modes["normal_mode_convention"] == "mass_weighted"
    np.testing.assert_allclose(displacement.T @ displacement, [[1.0]], atol=2e-7)
    np.testing.assert_allclose(
        np.sum(
            displacement.reshape(2, 3, 1) * np.sqrt(LIH_MASSES)[:, None, None],
            axis=0,
        ),
        0,
        atol=1e-7,
    )
    delta = 1e-2
    energies = []
    for shift in [-delta, 0, delta]:
        positions = LIH_POSITIONS.copy()
        positions[1, 2] += shift
        energies.append(_calculator(positions).calculate({"energy": {}})["energy"][0])
    curvature = (energies[2] - 2 * energies[1] + energies[0]) / delta**2
    expected = np.sqrt(curvature * np.sum(1 / LIH_MASSES))
    np.testing.assert_allclose(frequencies[0], expected, rtol=5e-3, atol=1e-7)


def _fixed_lih_coordinates(model, parameters, **kwargs):
    """Deterministic initial geometry for the dynamics integration tests."""
    coordinates = functions.qp_to_z(
        LIH_POSITIONS.ravel(),
        np.zeros(6),
        model.constants.classical_coordinate_mass,
        model.constants.classical_coordinate_weight,
    )
    return np.tile(coordinates, (len(kwargs["seed"]), 1))


@pytest.mark.parametrize(
    "algorithm", [MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio]
)
def test_ab_initio_algorithms_take_a_pyscf_step(algorithm):
    """Both stock ab initio recipes consume all PySCF outputs in one update."""
    sim = Simulation(
        {
            "num_trajs": 1,
            "batch_size": 1,
            "tmax": 0.1,
            "dt_update": 0.1,
            "dt_collect": 0.1,
            "progress_bar": False,
        }
    )
    sim.model = AbInitio(
        {
            "calculator": "pyscf",
            "atom_names": ["Li", "H"],
            "atom_positions": LIH_POSITIONS.copy(),
            "atom_masses": LIH_MASSES.copy(),
            "num_quantum_states": 2,
            "calculator_args": {
                "method_es": "casscf",
                "basis": "sto-3g",
                "ncas": 2,
                "nelecas": 2,
                "num_threads": 1,
                "scf_options": dict(SCF_OPTIONS),
                "casscf_options": dict(CASSCF_OPTIONS),
            },
        }
    )
    sim.model.ingredients.append(("init_classical", _fixed_lih_coordinates))
    sim.algorithm = algorithm()
    sim.initial_state["wf_adb"] = np.array([1.0, 0.0], dtype=complex)
    sim.initialize_timesteps()
    sim.t_ind = 0
    state, parameters = sim.algorithm.execute_recipe(
        sim, {"seed": np.array([0])}, {}, sim.algorithm.initialization_recipe
    )
    state, parameters = sim.algorithm.execute_recipe(
        sim, state, parameters, sim.algorithm.update_recipe
    )
    assert state["wf_adb"].shape == (1, 2)
    assert np.all(np.isfinite(state["z"]))
    assert np.all(np.isfinite(state["wf_adb"]))
    np.testing.assert_allclose(np.sum(np.abs(state["wf_adb"]) ** 2, axis=1), 1, atol=1e-9)
    assert state["aip_wf_overlaps"].shape == (1, 2, 2)
    assert np.all(np.isfinite(state["aip_wf_overlaps"]))
