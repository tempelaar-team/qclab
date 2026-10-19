"""Check CMS-PDFT physical roots and the complete ab initio property contract."""

import copy
import pickle

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import fci, gto, lib, mcpdft, scf

from qclab import Simulation, functions, ingredients
from qclab.algorithms import MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio
from qclab.interfaces.pyscf import QCLabPySCFInterface, wavefunction_overlaps
from qclab.models import AbInitio
from qclab.numerical_constants import AMU_TO_EMASS, ANGSTROM_TO_BOHR


FUNCTIONALS = ("tPBE", "tBLYP", "ftPBE")
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
MASSES = np.array([7.0, 1.0]) * AMU_TO_EMASS
SCF_OPTIONS = {"conv_tol": 1e-12}
CASSCF_OPTIONS = {
    "conv_tol": 1e-11,
    "conv_tol_grad": 1e-8,
    "ah_conv_tol": 1e-14,
    "max_cycle_macro": 200,
}
CMS_OPTIONS = {"conv_tol_diabatize": 1e-10}
RESPONSE_OPTIONS = {"conv_atol": 1e-10, "conv_rtol": 0.0, "max_cycle": 200}


def calculator(otxc="tPBE", **overrides):
    """Use a small active space with an inactive core and accurate responses."""
    options = {
        "method_es": "cms-pdft",
        "otxc": otxc,
        "grids_level": 5,
        "cms_options": dict(CMS_OPTIONS),
        "atom_positions": POSITIONS.copy(),
        "atom_masses": MASSES.copy(),
        "atom_names": ["Li", "H"],
        "num_quantum_states": 2,
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "scf_options": dict(SCF_OPTIONS),
        "casscf_options": dict(CASSCF_OPTIONS),
        "response_options": dict(RESPONSE_OPTIONS),
        "num_threads": 1,
    }
    options.update(overrides)
    return QCLabPySCFInterface(**options)


def all_properties():
    return {
        "energy": {"excited_amplitudes": True},
        "gradient": {"state_inds_gradient": None},
        "derivative_coupling": {"state_inds_derivative_coupling": None},
    }


def direct_cms(otxc):
    """Independent stock PySCF reference, without the QC Lab interface."""
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["Li", "H"], POSITIONS)),
            unit="Bohr", basis="sto-3g", verbose=0,
        )
        mf = scf.RHF(mol)
        mf.chkfile = None
        mf.set(**SCF_OPTIONS).run()
        mc = mcpdft.CASSCF(mf, otxc, 2, 2, grids_level=5)
        mc.fcisolver = fci.direct_spin0.FCI(mol)
        mc.fcisolver.conv_tol = 1e-12
        mc.fcisolver.max_cycle = 200
        mc.fix_spin_(ss=0.0)
        mc = mc.multi_state([0.5, 0.5], "cms")
        mc.chkfile = None
        mc.set(**CASSCF_OPTIONS, **CMS_OPTIONS).run()
        assert mc.converged
    return mc


def physical_snapshot(mc):
    """Represent the final physical PDFT eigenstates, rather than its diabats."""
    return {
        "mol": mc.mol.dumps(),
        "atom_positions": mc.mol.atom_coords().copy(),
        "mo_coeff": mc.mo_coeff.copy(),
        "ci": tuple(ci.copy() for ci in mc.get_ci_adiabats(uci="MSPDFT")),
        "ncore": mc.ncore,
        "ncas": mc.ncas,
        "nelecas": mc.nelecas,
        "energies": np.asarray(mc.e_states).copy(),
    }


@pytest.fixture(scope="module")
def cases():
    """Share expensive reference calculations while retaining immutable results."""
    cache = {}

    def get(otxc="tPBE"):
        if otxc not in cache:
            calc = calculator(otxc)
            cache[otxc] = {
                "calculator": calc,
                "result": copy.deepcopy(calc.calculate(all_properties())),
                "direct": direct_cms(otxc),
                "displaced": {},
            }
        return cache[otxc]

    return get


def finite_differences(case, otxc, delta):
    """Cold displaced solves avoid importing the reference's orbital path."""
    if delta not in case["displaced"]:
        reference = case["result"]["excited_amplitudes"]
        results = []
        overlaps = []
        for direction in (-1, 1):
            positions = POSITIONS.copy()
            positions[1, 2] += direction * delta
            result = calculator(otxc, atom_positions=positions).calculate(
                {"energy": {"excited_amplitudes": True}}
            )
            overlap = wavefunction_overlaps(reference, result["excited_amplitudes"])
            overlap *= np.sign(np.diag(overlap))[None, :]
            results.append(result)
            overlaps.append(overlap)
        case["displaced"][delta] = (
            (results[1]["energy"] - results[0]["energy"]) / (2 * delta),
            (overlaps[1] - overlaps[0]) / (2 * delta),
        )
    return case["displaced"][delta]


@pytest.mark.parametrize("otxc", FUNCTIONALS)
def test_final_energies_and_snapshots_match_stock_cms(cases, otxc):
    case = cases(otxc)
    result, reference = case["result"], case["direct"]
    assert result["energy"].shape == (2,)
    np.testing.assert_allclose(result["energy"], reference.e_states, atol=2e-9, rtol=0)
    snapshot = result["excited_amplitudes"]
    assert snapshot["ncore"] == 1
    assert "ci_intermediate" in snapshot and "si" in snapshot
    combined = np.einsum("ri,rab->iab", snapshot["si"], snapshot["ci_intermediate"])
    np.testing.assert_allclose(snapshot["ci"], combined, atol=1e-12)
    overlap = wavefunction_overlaps(snapshot, physical_snapshot(reference))
    np.testing.assert_allclose(np.abs(overlap), np.eye(2), atol=2e-8)


@pytest.mark.parametrize("otxc", FUNCTIONALS)
def test_all_response_components_match_stock_cms(cases, otxc):
    case = cases(otxc)
    reference = case["direct"]
    overlap = wavefunction_overlaps(
        case["result"]["excited_amplitudes"], physical_snapshot(reference)
    )
    signs = np.sign(np.diag(overlap))
    with lib.with_omp_threads(1):
        gradients = reference.nuc_grad_method().set(**RESPONSE_OPTIONS)
        expected_gradient = np.stack(
            [gradients.kernel(state=index) for index in range(2)], axis=-1
        )
        assert gradients.converged
        couplings = reference.nac_method().set(**RESPONSE_OPTIONS)
        expected_coupling = couplings.kernel(
            state=(0, 1), use_etfs=False, mult_ediff=False
        )
        assert couplings.converged
    np.testing.assert_allclose(
        case["result"]["gradient"], expected_gradient, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        case["result"]["derivative_coupling"][(0, 1)],
        expected_coupling * np.prod(signs) * ANGSTROM_TO_BOHR,
        atol=1e-8, rtol=0,
    )


@pytest.mark.parametrize("otxc", FUNCTIONALS)
def test_root_gradients_match_final_energy_derivatives_at_two_steps(cases, otxc):
    case = cases(otxc)
    gradient = case["result"]["gradient"]
    assert gradient.shape == (2, 3, 2)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0, atol=1e-7)
    for delta in (1e-3, 5e-4):
        energy_derivative, _ = finite_differences(case, otxc, delta)
        np.testing.assert_allclose(gradient[1, 2], energy_derivative, atol=1e-6, rtol=0)


@pytest.mark.parametrize("otxc", FUNCTIONALS)
def test_nac_matches_physical_final_state_overlap_derivatives(cases, otxc):
    case = cases(otxc)
    coupling = case["result"]["derivative_coupling"][(0, 1)] / ANGSTROM_TO_BOHR
    assert coupling.shape == (2, 3)
    assert np.all(np.isfinite(coupling))
    for delta in (1e-3, 5e-4):
        _, overlap_derivative = finite_differences(case, otxc, delta)
        np.testing.assert_allclose(coupling[1, 2], overlap_derivative[0, 1], atol=1e-6, rtol=0)


def test_final_snapshots_are_serializable_and_basis_invariant(cases):
    snapshot = cases()["result"]["excited_amplitudes"]
    restored = pickle.loads(pickle.dumps(snapshot))
    np.testing.assert_allclose(wavefunction_overlaps(snapshot, restored), np.eye(2), atol=1e-12)
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = copy.deepcopy(snapshot)
    rotated["ci_intermediate"] = np.einsum("ri,rab->iab", rotation, snapshot["ci_intermediate"])
    rotated["si"] = rotation.T @ snapshot["si"]
    rotated["ci"] = tuple(np.einsum("ri,rab->iab", rotated["si"], rotated["ci_intermediate"]))
    np.testing.assert_allclose(wavefunction_overlaps(snapshot, rotated), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(snapshot["ci"], rotated["ci"], atol=1e-12)


def test_selection_reverse_pairs_and_skips_preserve_contract(cases, monkeypatch):
    calc = calculator()
    result = calc.calculate(all_properties())
    selected = calc.calculate({
        "gradient": {"state_inds_gradient": 1},
        "derivative_coupling": {"state_inds_derivative_coupling": [1, 0]},
    })
    np.testing.assert_array_equal(selected["gradient"][:, :, 0], 0)
    np.testing.assert_allclose(selected["gradient"][:, :, 1], result["gradient"][:, :, 1])
    assert set(selected["derivative_coupling"]) == {(1, 0)}
    np.testing.assert_allclose(
        selected["derivative_coupling"][(1, 0)], -result["derivative_coupling"][(0, 1)]
    )

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Skipped properties must not solve or evaluate a response.")

    monkeypatch.setattr(calc, "_run", unexpected_solve)
    monkeypatch.setattr(calc, "_gradient", unexpected_solve)
    monkeypatch.setattr(calc, "_coupling", unexpected_solve)
    assert calc.calculate({"gradient": {"calc_property": False}}) == {"gradient": None}


def test_cached_phase_repair_changes_final_si_only(monkeypatch):
    calc = calculator()
    original = calc.calculate(all_properties())
    original_copy = pickle.dumps(original)
    previous = copy.deepcopy(original["excited_amplitudes"])
    previous["ci"] = tuple(
        ci * (-1 if index == 0 else 1) for index, ci in enumerate(previous["ci"])
    )
    previous["si"][:, 0] *= -1
    previous_copy = pickle.dumps(previous)
    intermediate = [ci.copy() for ci in calc._mc.ci]
    diagnostic = {
        name: np.asarray(getattr(calc._mc, name)).copy()
        for name in ("heff_mcscf", "hdiag_pdft", "si_mcscf")
    }
    old_si = calc._mc.si_pdft.copy()

    def unexpected_solve(*args, **kwargs):
        pytest.fail("A same-geometry phase change must reuse the solved CMS model.")

    monkeypatch.setattr(calc, "_run", unexpected_solve)
    requests = all_properties()
    requests["wf_overlaps"] = {"amplitudes_previous": previous}
    current = calc.calculate(requests)
    np.testing.assert_allclose(current["wf_overlaps"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(calc._mc.si_pdft, old_si * [-1, 1], atol=1e-12)
    for saved, current_ci in zip(intermediate, calc._mc.ci):
        np.testing.assert_array_equal(current_ci, saved)
    for name, saved in diagnostic.items():
        np.testing.assert_array_equal(getattr(calc._mc, name), saved)
    np.testing.assert_array_equal(current["excited_amplitudes"]["ci_intermediate"], intermediate)
    np.testing.assert_allclose(current["gradient"], original["gradient"], atol=1e-12)
    np.testing.assert_allclose(
        current["derivative_coupling"][(0, 1)], -original["derivative_coupling"][(0, 1)],
        atol=1e-8, rtol=0,
    )
    assert pickle.dumps(previous) == previous_copy
    assert pickle.dumps(original) == original_copy


def test_three_geometry_phases_ignore_stale_current_snapshot(cases):
    previous = copy.deepcopy(cases()["result"]["excited_amplitudes"])
    previous["ci"] = tuple(
        ci * (-1 if index == 0 else 1) for index, ci in enumerate(previous["ci"])
    )
    previous["si"][:, 0] *= -1
    calc = calculator()
    for shift in (0.001, -0.001, 0.0):
        positions = POSITIONS.copy()
        positions[1, 2] += shift
        retained = pickle.dumps(previous)
        current = calc.calculate(
            {
                "energy": {"excited_amplitudes": True},
                "wf_overlaps": {
                    "amplitudes_previous": previous,
                    "amplitudes_current": previous,
                },
            },
            atom_positions=positions,
        )
        assert np.all(np.diag(current["wf_overlaps"]) > 0.999)
        assert pickle.dumps(previous) == retained
        np.testing.assert_allclose(current["excited_amplitudes"]["atom_positions"], positions)
        previous = current["excited_amplitudes"]
    reference_ci = np.asarray(cases()["result"]["excited_amplitudes"]["ci"])
    np.testing.assert_allclose(
        previous["ci"], reference_ci * np.array([-1, 1])[:, None, None], atol=1e-8
    )


def test_changed_geometry_agrees_with_independent_cold_solution():
    positions = POSITIONS.copy()
    positions[1, 2] += 0.001
    calc = calculator()
    previous = calc.calculate({"energy": {"excited_amplitudes": True}})["excited_amplitudes"]
    requests = all_properties()
    requests["wf_overlaps"] = {"amplitudes_previous": previous}
    path_result = calc.calculate(requests, atom_positions=positions)
    independent = calculator(atom_positions=positions).calculate(all_properties())
    np.testing.assert_allclose(path_result["energy"], independent["energy"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(path_result["gradient"], independent["gradient"], atol=1e-8, rtol=0)
    overlap = wavefunction_overlaps(
        path_result["excited_amplitudes"], independent["excited_amplitudes"]
    )
    np.testing.assert_allclose(np.abs(overlap), np.eye(2), atol=1e-10)
    signs = np.sign(np.diag(overlap))
    np.testing.assert_allclose(
        path_result["derivative_coupling"][(0, 1)],
        independent["derivative_coupling"][(0, 1)] * np.prod(signs),
        atol=1e-8, rtol=0,
    )


def cms_model(**overrides):
    options = {
        "calculator": "pyscf", "atom_positions": POSITIONS.copy(),
        "atom_masses": MASSES.copy(), "atom_names": ["Li", "H"],
        "num_quantum_states": 2,
        "calculator_args": {
            "method_es": "cms-pdft", "otxc": "tPBE", "grids_level": 5,
            "ncas": 2, "nelecas": 2, "cms_options": dict(CMS_OPTIONS),
            "scf_options": dict(SCF_OPTIONS), "casscf_options": dict(CASSCF_OPTIONS),
            "response_options": dict(RESPONSE_OPTIONS), "num_threads": 1,
        },
    }
    options.update(overrides)
    return AbInitio(options)


def test_batched_cms_requests_are_isolated_and_configuration_invalidates_cache(cases):
    model = cms_model()
    positions = np.array([POSITIONS, POSITIONS + [0.0, 0.0, 0.001]])
    z = functions.qp_to_z(
        positions.reshape(2, 6), np.zeros((2, 6)),
        model.constants.classical_coordinate_mass, model.constants.classical_coordinate_weight,
    )
    saved_z = z.copy()
    parameters = {}
    result = ingredients.ab_initio_property_calculator_pyscf(
        model, parameters, batch_size=2, property_dict={"energy": {"z": z}}
    )
    calculators = [parameters["_pyscf_calculators"][index]["calculator"] for index in range(2)]
    assert calculators[0] is not calculators[1]
    np.testing.assert_allclose(result[0]["energy"], cases()["result"]["energy"], atol=1e-9, rtol=0)
    subset = ingredients.ab_initio_property_calculator_pyscf(
        model, parameters, batch_size=2,
        property_dict={
            "gradient": {"z": z, "state_inds_gradient": np.array([0, 1])},
            "derivative_coupling": {
                "z": z, "calc_property": np.array([False, True]),
                "state_inds_derivative_coupling": np.array([[0, 1], [1, 0]]),
            },
        },
    )
    assert subset[0]["derivative_coupling"] is None
    assert set(subset[1]["derivative_coupling"]) == {(1, 0)}
    np.testing.assert_array_equal(subset[0]["gradient"][:, :, 1], 0)
    np.testing.assert_array_equal(subset[1]["gradient"][:, :, 0], 0)
    np.testing.assert_array_equal(z, saved_z)
    assert all(
        parameters["_pyscf_calculators"][index]["calculator"] is calculators[index]
        for index in range(2)
    )
    model.constants.calculator_args["otxc"] = "tBLYP"
    changed = ingredients.ab_initio_property_calculator_pyscf(
        model, parameters, batch_size=2, property_dict={"energy": {"z": z}}
    )
    assert all(
        parameters["_pyscf_calculators"][index]["calculator"] is not calculators[index]
        for index in range(2)
    )
    np.testing.assert_allclose(changed[0]["energy"], cases("tBLYP")["result"]["energy"], atol=1e-9)


def test_cms_normal_modes_use_cms_energy_curvature_and_leave_cache_unchanged(cases):
    calc = calculator()
    initial = calc.calculate({"energy": {"excited_amplitudes": True}})
    saved = pickle.dumps(initial["excited_amplitudes"])
    modes = calc.normal_modes(state_ind=0, delta=1e-3)
    frequency = np.asarray(modes["harmonic_frequency"])
    mode = np.asarray(modes["normal_mode"])
    assert frequency.shape == (1,) and mode.shape == (6, 1)
    assert np.isrealobj(frequency) and frequency[0] > 0
    assert modes["normal_mode_convention"] == "mass_weighted"
    np.testing.assert_allclose(mode.T @ mode, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(
        np.sum(mode.reshape(2, 3, 1) * np.sqrt(MASSES)[:, None, None], axis=0),
        0, atol=1e-10,
    )
    delta = 1e-2
    energies = []
    for sign in (-1, 1):
        positions = POSITIONS.copy()
        positions[1, 2] += sign * delta
        energies.append(calculator(atom_positions=positions).calculate({"energy": {}})["energy"][0])
    curvature = (sum(energies) - 2 * cases()["result"]["energy"][0]) / delta**2
    np.testing.assert_allclose(frequency[0], np.sqrt(curvature * np.sum(1 / MASSES)), rtol=5e-3)
    retained = calc.calculate({"energy": {"excited_amplitudes": True}})["excited_amplitudes"]
    assert pickle.dumps(retained) == saved


def test_open_shell_doublet_responses_match_physical_derivatives():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    options = {
        "atom_positions": positions, "atom_names": ["H", "H"],
        "atom_masses": np.array([1837.0, 1837.0]),
        "charge": 1, "multiplicity": 2, "nelecas": 1,
    }
    result = calculator(**options).calculate(all_properties())
    assert result["gradient"].shape == (2, 3, 2)
    assert np.all(np.isfinite(result["gradient"]))
    displaced, overlaps = [], []
    delta = 1e-3
    for sign in (-1, 1):
        shifted = positions.copy()
        shifted[1, 2] += sign * delta
        other = calculator(**dict(options, atom_positions=shifted)).calculate(
            {"energy": {"excited_amplitudes": True}}
        )
        displaced.append(other["energy"])
        overlap = wavefunction_overlaps(result["excited_amplitudes"], other["excited_amplitudes"])
        overlaps.append(overlap * np.sign(np.diag(overlap))[None, :])
    np.testing.assert_allclose(
        result["gradient"][1, 2], (displaced[1] - displaced[0]) / (2 * delta),
        atol=1e-6, rtol=0,
    )
    np.testing.assert_allclose(
        result["derivative_coupling"][(0, 1)][1, 2] / ANGSTROM_TO_BOHR,
        (overlaps[1] - overlaps[0])[0, 1] / (2 * delta), atol=1e-6, rtol=0,
    )


@pytest.mark.parametrize("algorithm", [MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio])
def test_both_algorithms_advance_with_final_cms_states(algorithm):
    sim = Simulation({
        "num_trajs": 1, "batch_size": 1, "dt_update": 0.05,
        "dt_collect": 0.05, "tmax": 0.1, "progress_bar": False,
    })
    sim.model = cms_model(init_momentum=np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.1]))
    sim.model.ingredients.append(
        ("init_classical", ingredients.init_classical_definite_position_momentum)
    )
    sim.algorithm = algorithm()
    if algorithm is MeanFieldAbInitio:
        sim.initial_state["wf_adb"] = np.sqrt(np.array([0.4, 0.6])).astype(complex)
    else:
        sim.initial_state["wf_adb"] = np.array([0.0, 1.0], dtype=complex)
    sim.initialize_timesteps()
    sim.t_ind = 0
    state, parameters = sim.algorithm.execute_recipe(
        sim, {"seed": np.array([7])}, {}, sim.algorithm.initialization_recipe
    )
    initial_z = state["z"].copy()
    for sim.t_ind in range(2):
        state, parameters = sim.algorithm.execute_recipe(
            sim, state, parameters, sim.algorithm.update_recipe
        )
    assert not np.array_equal(state["z"], initial_z)
    for key in ("z", "wf_adb", "h_q_tot", "aip_wf_overlaps", "quantum_classical_force"):
        assert np.all(np.isfinite(state[key]))
    np.testing.assert_allclose(np.sum(abs(state["wf_adb"]) ** 2, axis=1), 1, atol=1e-10)
    assert state["aip_wf_overlaps"].shape == (1, 2, 2)
    if algorithm is FewestSwitchesSurfaceHoppingAbInitio:
        assert state["act_surf_ind"][0] in (0, 1)
