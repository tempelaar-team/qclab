"""Verify calculator integration and Cartesian sampling without external programs."""

import copy

import numpy as np
import pytest

from qclab import Simulation, ingredients
from qclab.algorithms import MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio
from qclab.dynamics import serial_driver
from qclab.functions import dqdp_to_dzc, qp_to_z, z_to_p, z_to_q
from qclab.models import AbInitio
from qclab.numerical_constants import ANGSTROM_TO_BOHR


def make_model(**overrides):
    """Construct a small heteronuclear system with distinct atomic masses."""
    constants = {
        "atom_names": ["H", "He"],
        "atom_positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "atom_masses": np.array([2.0, 8.0]),
        "num_quantum_states": 2,
        "calculator": "pyscf",
        "calculator_args": {"ncas": 2, "nelecas": 2},
    }
    constants.update(overrides)
    return AbInitio(constants)


def batched_z(model, positions):
    """Convert a batch of atom-major geometries to QC Lab coordinates."""
    positions = np.asarray(positions).reshape((-1, 6))
    return qp_to_z(
        positions,
        np.zeros_like(positions),
        model.constants.classical_coordinate_mass[np.newaxis],
        model.constants.classical_coordinate_weight[np.newaxis],
    )


@pytest.fixture
def mock_interface(monkeypatch):
    """Replace the optional electronic structure package with a recording fake."""
    class Calculator:
        instances = []

        def __init__(self, **kwargs):
            self.constructor_args = copy.deepcopy(kwargs)
            self.calls = []
            Calculator.instances.append(self)

        def calculate(self, property_dict, atom_positions=None):
            self.calls.append((copy.deepcopy(property_dict), atom_positions.copy()))
            results = {}
            if "energy" in property_dict:
                base = float(np.sum(atom_positions))
                results["energy"] = np.array([base, base + 0.1])
                if property_dict["energy"].get("excited_amplitudes", False):
                    results["excited_amplitudes"] = {"positions": atom_positions.copy()}
            if "gradient" in property_dict:
                results["gradient"] = np.zeros((2, 3, 2))
            if "derivative_coupling" in property_dict:
                results["derivative_coupling"] = {}
            if "wf_overlaps" in property_dict:
                results["wf_overlaps"] = np.eye(2)
            self.results = results
            return results

    monkeypatch.setattr(ingredients, "QCLabPySCFInterface", Calculator)
    return Calculator


def test_atom_major_masses_and_ase_free_model_export():
    model = make_model()
    np.testing.assert_array_equal(
        model.constants.classical_coordinate_mass, [2, 2, 2, 8, 8, 8]
    )


def test_calculator_dispatch_preserves_qchem_default(monkeypatch):
    model = make_model(calculator="qchem")
    monkeypatch.setattr(ingredients, "ab_initio_property_calculator_qchem", lambda *a, **k: "qchem")
    monkeypatch.setattr(ingredients, "ab_initio_property_calculator_pyscf", lambda *a, **k: "pyscf")
    calculator, found = model.get("ab_initio_property_calculator")
    assert found
    assert calculator(model, {}) == "qchem"
    model.constants.calculator = "pyscf"
    assert calculator(model, {}) == "pyscf"
    model.constants.calculator = "unknown"
    with pytest.raises(ValueError, match="calculator"):
        calculator(model, {})
    defaults = make_model()
    del defaults.constants.calculator
    assert calculator(defaults, {}) == "qchem"


@pytest.mark.parametrize("with_seeds", [False, True])
def test_pyscf_failure_identifies_original_trajectory(monkeypatch, with_seeds):
    """A single-trajectory request retains its original batch index and seed."""
    failure = RuntimeError("PySCF SCF did not converge after one Newton SCF retry")

    class Calculator:
        def __init__(self, **options):
            pass

        def calculate(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(ingredients, "QCLabPySCFInterface", Calculator)
    parameters = {"seed": np.array([11, 79])} if with_seeds else {}
    with pytest.raises(RuntimeError, match="trajectory 1") as caught:
        ingredients.ab_initio_property_calculator_pyscf(
            make_model(), parameters, batch_size=1, cache_traj_ind=1,
            property_dict={"energy": {}},
        )
    assert ("seed 79" in str(caught.value)) == with_seeds
    assert "atom_positions_bohr=" in str(caught.value)
    assert "SCF did not converge after one Newton SCF retry" in str(caught.value)
    assert caught.value.__cause__ is failure


def test_qchem_missing_ase_has_actionable_error(monkeypatch):
    monkeypatch.setattr(ingredients, "DISABLE_ASE", True)
    with pytest.raises(ImportError, match="requires ASE"):
        ingredients.ab_initio_property_calculator_qchem(
            make_model(), {}, batch_size=1, property_dict={}
        )


def test_qchem_ingredient_forwards_shared_settings_and_expands_selected_gradient(monkeypatch):
    recorded = {}

    class Calculator:
        def __init__(self, **options):
            recorded.update(options)
            self.results = {}

        def write_input(self, **requests):
            self.requests = requests

        def execute(self):
            pass

        def read_results(self, **requests):
            self.results = {"gradient": np.ones((2, 3, 1)), "energy": np.arange(3)}

    monkeypatch.setattr(ingredients, "DISABLE_ASE", False)
    monkeypatch.setattr(ingredients, "QCLabQChemInterface", Calculator, raising=False)
    model = make_model(calculator="qchem", num_quantum_states=3, calculator_args={
        "method_es": "tddft", "tda": True, "xc": "pbe",
        "excited_multiplicity": 3, "excited_state_indices": [0, 3],
        "qchem_gs_args": {"basis": "sto-3g"},
        "qchem_es_args": {"cis_convergence": 10},
    })
    results = ingredients.ab_initio_property_calculator_qchem(
        model, {"seed": np.array([5])}, batch_size=1,
        property_dict={"gradient": {"state_inds_gradient": [2]}, "energy": {}},
    )[0]
    assert recorded["num_quantum_states"] == 3
    assert recorded["excited_state_indices"] == [0, 3]
    assert recorded["excited_multiplicity"] == 3
    assert recorded["basis"] == "sto-3g"
    assert recorded["cis_convergence"] == 10
    np.testing.assert_array_equal(results["gradient"][:, :, :2], 0)
    np.testing.assert_array_equal(results["gradient"][:, :, 2], 1)


def test_nested_qchem_request_preserves_trajectory_label_and_local_snapshot_slice(monkeypatch):
    recorded = {}

    class Calculator:
        def __init__(self, **options):
            recorded.update(options)
            self.results = {}

        def write_input(self, **requests):
            recorded["requests"] = requests

        def execute(self):
            pass

        def read_results(self, **requests):
            self.results = {"wf_overlaps": np.eye(2)}

    monkeypatch.setattr(ingredients, "DISABLE_ASE", False)
    monkeypatch.setattr(ingredients, "QCLabQChemInterface", Calculator, raising=False)
    model = make_model(calculator="qchem")
    positions = model.constants.atom_positions + 0.2
    previous = np.empty(1, dtype=object)
    previous[0] = {"ci": np.arange(4).reshape(2, 2)}
    ingredients.ab_initio_property_calculator_qchem(
        model, {"seed": np.array([11, 79])}, batch_size=1, cache_traj_ind=1,
        property_dict={"wf_overlaps": {
            "z": batched_z(model, positions),
            "z_previous": batched_z(model, model.constants.atom_positions),
            "amplitudes_previous": previous,
        }},
    )
    assert recorded["label"] == "qchem_job_79"
    assert recorded["folder_scratch"] == "qclab_job_79"
    np.testing.assert_allclose(recorded["atom_positions"], positions)
    request = recorded["requests"]["wf_overlaps"]
    np.testing.assert_array_equal(request["amplitudes_previous"]["ci"], previous[0]["ci"])
    np.testing.assert_allclose(request["atom_positions_previous"], model.constants.atom_positions)


@pytest.mark.parametrize("convention", ["legacy", "mass_weighted"])
def test_wigner_projection_is_canonical_and_preserves_constants(monkeypatch, convention):
    mode = np.zeros((6, 1))
    mode[0, 0] = np.sqrt(8.0 / 10.0)
    mode[3, 0] = -np.sqrt(2.0 / 10.0)
    model = make_model(
        normal_mode=mode,
        harmonic_frequency=np.array([0.2]),
        normal_mode_convention=convention,
    )
    constants = model.constants
    q_mode = np.array([[1.25], [-0.75]])
    p_mode = np.array([[0.4], [-0.5]])

    def sample(mode_model, parameters, **kwargs):
        assert mode_model is not model
        assert mode_model.constants is not constants
        return qp_to_z(q_mode, p_mode, np.ones((1, 1)), np.array([[0.2]]))

    monkeypatch.setattr(ingredients, "init_classical_wigner_harmonic", sample)
    z = model.init_classical({}, seed=np.array([1, 2]))
    masses = constants.classical_coordinate_mass
    weights = constants.classical_coordinate_weight
    q = z_to_q(z, masses, weights) - constants.init_position
    p = z_to_p(z, masses, weights)
    expected_q = q_mode @ mode.T
    expected_p = p_mode @ mode.T
    if convention == "mass_weighted":
        expected_q = expected_q / np.sqrt(masses)
        expected_p = expected_p * np.sqrt(masses)
        np.testing.assert_allclose(q[:, 0] * 2 + q[:, 3] * 8, 0, atol=1e-14)
        np.testing.assert_allclose(p[:, 0] + p[:, 3], 0, atol=1e-14)
        np.testing.assert_allclose(np.sum(p * p / (2 * masses), axis=1), p_mode[:, 0] ** 2 / 2)
    np.testing.assert_allclose(q, expected_q, atol=1e-14)
    np.testing.assert_allclose(p, expected_p)
    assert model.constants is constants
    assert constants._update_function.__self__ is model


def test_wigner_failure_preserves_constants(monkeypatch):
    model = make_model(normal_mode=np.eye(6), harmonic_frequency=np.ones(6))
    constants = model.constants

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(ingredients, "init_classical_wigner_harmonic", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        model.init_classical({}, seed=np.array([0]))
    assert model.constants is constants
    assert constants._update_function.__self__ is model


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"harmonic_frequency": np.array([0.0])}, "positive"),
        ({"harmonic_frequency": np.array([0.2j])}, "real, positive"),
        ({"normal_mode": np.eye(6, dtype=complex)[:, :1] * 1j}, "real normal modes"),
        ({"normal_mode": np.ones((2, 3))}, "shape"),
        (
            {"normal_mode": np.ones((6, 1)), "normal_mode_convention": "mass_weighted"},
            "orthonormal",
        ),
        ({"normal_mode_convention": "unknown"}, "normal_mode_convention"),
    ],
)
def test_wigner_rejects_invalid_modes_without_mutation(overrides, message):
    options = {"normal_mode": np.eye(6)[:, :1], "harmonic_frequency": np.array([0.2])}
    options.update(overrides)
    model = make_model(**options)
    constants = model.constants
    with pytest.raises(ValueError, match=message):
        model.init_classical({}, seed=np.array([0]))
    assert model.constants is constants


def test_wigner_sampling_is_seed_reproducible():
    model = make_model(
        normal_mode=np.eye(6)[:, :2],
        harmonic_frequency=np.array([0.2, 0.3]),
        normal_mode_convention="mass_weighted",
    )
    seeds = np.arange(4)
    np.testing.assert_array_equal(
        model.init_classical({}, seed=seeds), model.init_classical({}, seed=seeds)
    )


def test_model_converts_legacy_nac_units_once():
    model = make_model()
    z = batched_z(model, [model.constants.atom_positions])
    parameters = {
        "ab_initio_property": [
            {"derivative_coupling": {(0, 1): np.ones((2, 3)) * ANGSTROM_TO_BOHR}}
        ]
    }
    coupling = model.derivative_coupling_dzc(parameters, z=z)
    expected = dqdp_to_dzc(
        np.ones(6), None,
        model.constants.classical_coordinate_mass,
        model.constants.classical_coordinate_weight,
    )
    np.testing.assert_allclose(coupling[0, :, 0, 1], expected)
    np.testing.assert_allclose(coupling[0, :, 1, 0], -expected)


def test_pyscf_adapter_slices_batch_requests_and_keeps_snapshots(mock_interface):
    model = make_model()
    geometries = np.array([model.constants.atom_positions, model.constants.atom_positions + 0.2])
    z = batched_z(model, geometries)
    snapshots = np.empty(2, dtype=object)
    snapshots[:] = [{"ci": np.eye(2)}, {"ci": np.ones((2, 2))}]
    requests = {
        "energy": {"z": z, "excited_amplitudes": True},
        "gradient": {"z": z, "state_inds_gradient": np.array([0, 1])},
        "derivative_coupling": {
            "z": z,
            "calc_property": np.array([True, False]),
            "state_inds_derivative_coupling": np.array([[0, 1], [1, 0]]),
        },
        "wf_overlaps": {
            "z": z,
            "z_previous": z,
            "amplitudes_previous": snapshots,
            "amplitudes_current": snapshots,
        },
    }
    results = ingredients.ab_initio_property_calculator_pyscf(
        model, {}, batch_size=2, property_dict=requests
    )
    assert results.shape == (2,)
    assert len(mock_interface.instances) == 2
    for index, calculator in enumerate(mock_interface.instances):
        args, positions = calculator.calls[0]
        np.testing.assert_allclose(positions, geometries[index])
        assert args["gradient"]["state_inds_gradient"] == index
        overlaps_args = args["wf_overlaps"]
        np.testing.assert_allclose(overlaps_args["atom_positions_previous"], geometries[index])
        np.testing.assert_array_equal(
            overlaps_args["amplitudes_previous"]["ci"], snapshots[index]["ci"]
        )
    pair = mock_interface.instances[0].calls[0][0]["derivative_coupling"]
    np.testing.assert_array_equal(
        pair["state_inds_derivative_coupling"], [0, 1]
    )
    assert "derivative_coupling" not in mock_interface.instances[1].calls[0][0]
    assert results[1]["derivative_coupling"] is None
    np.testing.assert_array_equal(
        requests["wf_overlaps"]["amplitudes_previous"][0]["ci"], np.eye(2)
    )


def test_pyscf_adapter_skips_without_constructing_calculator(mock_interface):
    parameters = {}
    results = ingredients.ab_initio_property_calculator_pyscf(
        make_model(), parameters, batch_size=2,
        property_dict={"derivative_coupling": {"calc_property": np.array([False, False])}},
    )
    assert all(result == {"derivative_coupling": None} for result in results)
    assert mock_interface.instances == []
    assert "_pyscf_calculators" not in parameters


def test_pyscf_adapter_rejects_mixed_geometries(mock_interface):
    model = make_model()
    z = batched_z(model, [model.constants.atom_positions])
    other_z = batched_z(model, [model.constants.atom_positions + 0.1])
    with pytest.raises(ValueError, match="same geometry"):
        ingredients.ab_initio_property_calculator_pyscf(
            model, {}, batch_size=1,
            property_dict={"energy": {"z": z}, "gradient": {"z": other_z}},
        )
    assert mock_interface.instances == []


def test_pyscf_adapter_reuses_by_trajectory_and_invalidates_configuration(mock_interface):
    model = make_model(
        calculator_args={"ncas": 2, "nelecas": 2, "casscf_options": {"max_cycle_macro": 20}}
    )
    parameters = {}
    geometry = model.constants.atom_positions.copy()
    for shift in (0.0, 0.2):
        z = batched_z(model, [geometry + shift, geometry - shift])
        ingredients.ab_initio_property_calculator_pyscf(
            model, parameters, batch_size=2, property_dict={"energy": {"z": z}}
        )
    assert len(mock_interface.instances) == 2
    assert all(len(calculator.calls) == 2 for calculator in mock_interface.instances)
    model.constants.calculator_args["casscf_options"]["max_cycle_macro"] = 30
    ingredients.ab_initio_property_calculator_pyscf(
        model, parameters, batch_size=2, property_dict={"energy": {"z": z}}
    )
    assert len(mock_interface.instances) == 4
    assert mock_interface.instances[0].constructor_args["casscf_options"]["max_cycle_macro"] == 20


def test_direct_model_fallback_retains_cache_trajectory_indices(mock_interface):
    model = make_model()
    z = batched_z(model, [model.constants.atom_positions, model.constants.atom_positions + 0.2])
    parameters = {}
    energies = model.h_qc(parameters, z=z)
    model.derivative_coupling_dzc(parameters, z=z)
    assert set(parameters["_pyscf_calculators"]) == {0, 1}
    assert len(mock_interface.instances) == 2
    assert energies[0, 0, 0] != energies[1, 0, 0]


@pytest.mark.parametrize("algorithm", [MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio])
def test_ab_initio_recipes_accept_pyscf_adapter(mock_interface, algorithm):
    sim = Simulation({
        "tmax": 0.01, "dt_update": 0.01, "dt_collect": 0.01,
        "num_trajs": 1, "batch_size": 1, "progress_bar": False,
    })
    sim.model = make_model(init_momentum=np.zeros(6))
    sim.model.ingredients.append(
        ("init_classical", ingredients.init_classical_definite_position_momentum)
    )
    sim.algorithm = algorithm()
    sim.initial_state["wf_adb"] = np.array([1.0, 0.0], dtype=complex)
    data = serial_driver(sim)
    assert np.all(np.isfinite(data.data_dict["quantum_energy"]))
    np.testing.assert_allclose(np.trace(data.data_dict["dm_adb"], axis1=-2, axis2=-1), 1)
    assert len(mock_interface.instances) == 1
