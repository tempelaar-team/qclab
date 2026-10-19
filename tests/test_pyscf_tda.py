"""Validate analytic TDA properties and dynamics on LiH without Q-Chem."""

import copy
import itertools
import pickle

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import dft, gto, lib

from qclab import Simulation, functions, ingredients
from qclab.algorithms import MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio
from qclab.dynamics import serial_driver
from qclab.interfaces.pyscf import QCLabPySCFInterface, wavefunction_overlaps
from qclab.models import AbInitio
from qclab.numerical_constants import AMU_TO_EMASS, ANGSTROM_TO_BOHR


POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
MASSES = np.array([7.0160034366, 1.00782503223]) * AMU_TO_EMASS
ROOTS = [0, 3]
FUNCTIONALS = ("pbe", "b3lyp")


def rotation_matrix():
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    antisymmetric = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(0.7) * antisymmetric
            + (1 - np.cos(0.7)) * antisymmetric @ antisymmetric)


def calculator(xc="pbe", excited_multiplicity=1, **overrides):
    options = {
        "atom_positions": POSITIONS.copy(), "atom_masses": MASSES.copy(),
        "atom_names": ["Li", "H"], "num_quantum_states": 3,
        "method_es": "tddft", "tda": True, "xc": xc, "basis": "sto-3g",
        "excited_multiplicity": excited_multiplicity,
        "excited_state_indices": list(ROOTS), "use_etfs": False,
        "grids_level": 5, "scf_options": {"conv_tol": 1e-12},
        "td_options": {"conv_tol": 1e-11}, "num_threads": 1,
    }
    options.update(overrides)
    return QCLabPySCFInterface(**options)


def all_properties():
    return {
        "energy": {"excited_amplitudes": True}, "gradient": {},
        "derivative_coupling": {},
    }


@pytest.fixture(scope="module")
def cases():
    cache = {}

    def get(xc="pbe", spin=1):
        key = xc, spin
        if key not in cache:
            calc = calculator(xc, spin)
            cache[key] = calc, copy.deepcopy(calc.calculate(all_properties()))
        return cache[key]

    return get


def direct_tda(xc, spin, positions=POSITIONS, grids_options=None):
    """Stock PySCF reference with no QC Lab calculator calls."""
    mol = gto.M(atom=list(zip(["Li", "H"], positions)), unit="Bohr",
                basis="sto-3g", spin=0, verbose=0)
    mf = dft.RKS(mol).set(xc=xc, conv_tol=1e-12)
    mf.chkfile = None
    mf.grids.level = 5
    for name, value in (grids_options or {}).items():
        setattr(mf.grids, name, value)
    mf.kernel()
    td = mf.TDA().set(singlet=spin == 1, nstates=4, conv_tol=1e-11)
    td.kernel()
    assert mf.converged and np.all(td.converged)
    return mf, td


@pytest.mark.parametrize("xc", FUNCTIONALS)
@pytest.mark.parametrize("spin", [1, 3])
def test_selected_energies_and_gradients_match_direct_pyscf(cases, xc, spin):
    _, result = cases(xc, spin)
    with lib.with_omp_threads(1):
        mf, td = direct_tda(xc, spin)
        ground_gradient = mf.nuc_grad_method()
        ground_gradient.grid_response = True
        expected = np.stack([ground_gradient.kernel()] + [
            td.nuc_grad_method().kernel(state=index + 1) for index in ROOTS
        ], axis=-1)
    np.testing.assert_allclose(result["energy"],
                               np.r_[mf.e_tot, mf.e_tot + td.e[ROOTS]], atol=1e-10)
    # Stock TD gradients omit grid motion, which is significant on Li. The
    # independent H derivative and translational invariance determine both
    # atom forces in this diatomic; all six derivatives are also checked below
    # against energies without making this symmetry assumption.
    expected[0] = -expected[1]
    np.testing.assert_allclose(result["gradient"], expected, atol=2e-7, rtol=0)
    assert result["gradient"].shape == (2, 3, 3)
    assert set(result["derivative_coupling"]) == {(0, 1), (0, 2), (1, 2)}
    if spin == 3:
        for pair in ((0, 1), (0, 2)):
            np.testing.assert_array_equal(result["derivative_coupling"][pair], 0)


def explicit_overlap(previous, current):
    """Expand the small LiH CIS determinants, retaining every occupied orbital."""
    cross = gto.intor_cross("int1e_ovlp", gto.loads(previous["mol"]),
                           gto.loads(current["mol"]))
    cross = previous["mo_coeff"].T @ cross @ current["mo_coeff"]

    def states(snapshot):
        nocc = snapshot["nocc"]
        occupied = tuple(range(nocc))
        ground = [(1.0, occupied, occupied)]
        result = [ground]
        for amplitudes in snapshot["ci"][1:]:
            determinants = []
            for spin in (0, 1):
                for i, a in itertools.product(range(nocc), range(amplitudes.shape[-1])):
                    orbitals = tuple(j for j in occupied if j != i) + (nocc + a,)
                    occupations = [occupied, occupied]
                    occupations[spin] = orbitals
                    coefficient = (-1) ** (nocc - 1 - i) * amplitudes[spin, i, a]
                    determinants.append((coefficient, *occupations))
            result.append(determinants)
        return result

    overlap = np.zeros((len(previous["ci"]), len(current["ci"])))
    for i, bra in enumerate(states(previous)):
        for j, ket in enumerate(states(current)):
            for cb, ab, bb in bra:
                for ck, ak, bk in ket:
                    overlap[i, j] += cb * ck * np.linalg.det(cross[np.ix_(ab, ak)]) * np.linalg.det(cross[np.ix_(bb, bk)])
    return overlap


@pytest.mark.parametrize("spin", [1, 3])
def test_overlaps_include_core_determinants_and_spin_adaptation(cases, spin):
    _, result = cases("pbe", spin)
    old = result["excited_amplitudes"]
    new_positions = POSITIONS.copy()
    new_positions[1, 2] += 0.03
    new = calculator(excited_multiplicity=spin, atom_positions=new_positions).calculate({
        "energy": {"excited_amplitudes": True},
        "wf_overlaps": {"amplitudes_previous": old},
    })
    assert old["nocc"] == 2
    np.testing.assert_allclose(wavefunction_overlaps(old, old), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(new["wf_overlaps"],
                               explicit_overlap(old, new["excited_amplitudes"]), atol=1e-12)
    if spin == 3:
        np.testing.assert_allclose(new["wf_overlaps"][0, 1:], 0, atol=1e-14)
        np.testing.assert_allclose(new["wf_overlaps"][1:, 0], 0, atol=1e-14)


@pytest.mark.parametrize("xc", FUNCTIONALS)
@pytest.mark.parametrize("spin", [1, 3])
@pytest.mark.parametrize("delta", [1e-3, 5e-4])
@pytest.mark.parametrize("rotated", [False, True])
def test_analytic_responses_match_cartesian_energy_and_overlap_derivatives(cases, xc, spin, delta, rotated):
    positions = POSITIONS @ rotation_matrix().T + [0.3, -0.8, 0.4] if rotated else POSITIONS
    if rotated:
        reference = calculator(xc, spin, atom_positions=positions).calculate(all_properties())
    else:
        _, reference = cases(xc, spin)
    snapshot = reference["excited_amplitudes"]
    gradient = np.empty((6, 3))
    coupling = np.empty((6, 3, 3))
    for coordinate in range(6):
        displaced = []
        for sign in (-1, 1):
            displaced_positions = positions.copy()
            displaced_positions.flat[coordinate] += sign * delta
            displaced.append(calculator(xc, spin, atom_positions=displaced_positions).calculate({
                "energy": {}, "wf_overlaps": {"amplitudes_previous": snapshot},
            }))
        gradient[coordinate] = (displaced[1]["energy"] - displaced[0]["energy"]) / (2 * delta)
        coupling[coordinate] = (displaced[1]["wf_overlaps"] - displaced[0]["wf_overlaps"]) / (2 * delta)
    np.testing.assert_allclose(reference["gradient"].reshape(6, 3), gradient, atol=1e-6, rtol=0)
    np.testing.assert_allclose(coupling + coupling.transpose(0, 2, 1), 0, atol=1e-6)
    for (i, j), vector in reference["derivative_coupling"].items():
        np.testing.assert_allclose(vector.ravel() / ANGSTROM_TO_BOHR,
                                   coupling[:, i, j], atol=1e-6, rtol=0)


@pytest.mark.parametrize("spin", [1, 3])
def test_rotated_translated_geometry_and_etf_covariance(cases, spin):
    # A finite Lebedev grid has cubic symmetry, not arbitrary rotational
    # invariance. Use a grid-symmetry rotation here; the separate finite-
    # difference tests check arbitrary molecular orientations directly.
    reference_positions = POSITIONS @ rotation_matrix().T
    original = calculator("b3lyp", spin, atom_positions=reference_positions).calculate(all_properties())
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    positions = reference_positions @ rotation.T + [0.3, -0.8, 0.4]
    rotated = calculator("b3lyp", spin, atom_positions=positions).calculate(all_properties())
    overlap = wavefunction_overlaps(original["excited_amplitudes"],
                                   rotated["excited_amplitudes"])
    signs = np.sign(np.diag(overlap))
    np.testing.assert_allclose(rotated["energy"], original["energy"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(rotated["gradient"],
                               np.einsum("ab,ibn->ian", rotation, original["gradient"]), atol=2e-6, rtol=0)
    for (i, j), vector in original["derivative_coupling"].items():
        np.testing.assert_allclose(rotated["derivative_coupling"][i, j],
                                   vector @ rotation.T * signs[i] * signs[j], atol=2e-6, rtol=0)
    etf = calculator("b3lyp", spin, atom_positions=positions, use_etfs=True).calculate(all_properties())
    for pair, vector in etf["derivative_coupling"].items():
        np.testing.assert_allclose(vector.sum(axis=0), 0, atol=1e-8)


def test_occupied_and_virtual_orbital_rotations_leave_snapshots_invariant(cases):
    _, result = cases()
    original = result["excited_amplitudes"]
    rotated = copy.deepcopy(original)
    rng = np.random.default_rng(23)
    occupied, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    virtual, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated["mo_coeff"][:, :2] = original["mo_coeff"][:, :2] @ occupied
    rotated["mo_coeff"][:, 2:] = original["mo_coeff"][:, 2:] @ virtual
    rotated["ci"] = tuple(np.einsum("ij,sja,ab->sib", occupied.T, ci, virtual)
                           for ci in original["ci"])
    np.testing.assert_allclose(wavefunction_overlaps(original, rotated), np.eye(3), atol=1e-12)


def test_selection_skips_reuse_cached_rephasing_and_snapshot_independence(monkeypatch):
    calc = calculator()
    reference = calc.calculate(all_properties())
    old = reference["excited_amplitudes"]
    saved = pickle.dumps(old)
    monkeypatch.setattr(calc, "_run", lambda *a, **k: pytest.fail("Repeated geometry was solved again"))
    subset = calc.calculate({"gradient": {"state_inds_gradient": [2]},
                             "derivative_coupling": {"state_inds_derivative_coupling": [2, 1]}})
    np.testing.assert_array_equal(subset["gradient"][:, :, :2], 0)
    np.testing.assert_allclose(subset["derivative_coupling"][2, 1],
                               -reference["derivative_coupling"][1, 2], atol=1e-12)
    skipped = calc.calculate({"energy": {"calc_property": False},
                              "derivative_coupling": {"calc_property": False}})
    assert skipped == {"energy": None, "derivative_coupling": None}
    flipped = copy.deepcopy(old)
    flipped["ci"] = tuple(ci * sign for ci, sign in zip(flipped["ci"], [1, -1, 1]))
    rephased = calc.calculate({**all_properties(),
                               "wf_overlaps": {"amplitudes_previous": flipped}})
    np.testing.assert_allclose(rephased["wf_overlaps"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rephased["derivative_coupling"][1, 2],
                               -reference["derivative_coupling"][1, 2], atol=1e-12)
    assert pickle.dumps(old) == saved


@pytest.mark.parametrize("spin", [1, 3])
def test_three_geometry_phase_continuity(cases, spin):
    calc = calculator(excited_multiplicity=spin)
    previous = None
    saved = []
    for distance in (2.8, 3.0, 3.2):
        positions = POSITIONS.copy()
        positions[1, 2] = distance
        request = all_properties()
        if previous is not None:
            request["wf_overlaps"] = {"amplitudes_previous": previous}
        current = calc.calculate(request, atom_positions=positions)
        if previous is not None:
            assert np.all(np.diag(current["wf_overlaps"]) > 0.95)
        previous = current["excited_amplitudes"]
        saved.append((previous, pickle.dumps(previous)))
    assert all(pickle.dumps(snapshot) == serialized for snapshot, serialized in saved)


def test_analytic_ground_modes_and_excited_modes_preserve_geometry_cache(monkeypatch):
    calc = calculator()
    initial = calc.calculate({"energy": {"excited_amplitudes": True}})
    saved = pickle.dumps(initial["excited_amplitudes"])
    run = calc._run
    monkeypatch.setattr(calc, "_run", lambda *a, **k: pytest.fail("Ground Hessian used displaced solves"))
    modes = calc.normal_modes()
    assert modes["harmonic_frequency"].shape == (1,)
    assert modes["normal_mode"].shape == (6, 1)
    assert modes["harmonic_frequency"][0] > 0
    np.testing.assert_allclose(modes["normal_mode"].T @ modes["normal_mode"], np.eye(1), atol=1e-12)
    delta = 0.002
    energies = []
    for sign in (-1, 1):
        positions = POSITIONS.copy()
        positions[1, 2] += sign * delta
        energies.append(calculator(atom_positions=positions).calculate({"energy": {}})["energy"][0])
    curvature = (sum(energies) - 2 * initial["energy"][0]) / delta**2
    reduced_mass = 1 / np.sum(1 / MASSES)
    np.testing.assert_allclose(modes["harmonic_frequency"][0]**2 * reduced_mass,
                               curvature, atol=1e-6, rtol=0)
    monkeypatch.setattr(calc, "_run", run)
    excited = calc.normal_modes(state_ind=1)
    assert excited["normal_mode"].shape == (6, 1)
    assert np.all(np.isfinite(excited["harmonic_frequency"]))
    current = calc.calculate({"energy": {"excited_amplitudes": True}})
    assert pickle.dumps(current["excited_amplitudes"]) == saved


def tda_model(spin=1):
    return AbInitio({
        "calculator": "pyscf", "atom_names": ["Li", "H"],
        "atom_positions": POSITIONS.copy(), "atom_masses": MASSES.copy(),
        "num_quantum_states": 3, "init_momentum": np.zeros(6),
        "calculator_args": {
            "method_es": "tddft", "tda": True, "xc": "pbe", "basis": "sto-3g",
            "excited_multiplicity": spin, "excited_state_indices": ROOTS,
            "use_etfs": True, "grids_level": 5,
            "scf_options": {"conv_tol": 1e-12}, "td_options": {"conv_tol": 1e-11},
        },
    })


def test_trajectory_cache_isolation_and_configuration_invalidation():
    model = tda_model()
    parameters = {}
    z = functions.qp_to_z(np.array([POSITIONS, POSITIONS + [0, 0, 0.01]]).reshape(2, 6),
                          np.zeros((2, 6)), model.constants.classical_coordinate_mass,
                          model.constants.classical_coordinate_weight)
    request = {"energy": {"z": z, "excited_amplitudes": True}}
    results = ingredients.ab_initio_property_calculator_pyscf(model, parameters,
                                                              batch_size=2, property_dict=request)
    first = [parameters["_pyscf_calculators"][i]["calculator"] for i in range(2)]
    assert first[0] is not first[1]
    assert results[0]["excited_amplitudes"] is not results[1]["excited_amplitudes"]
    model.constants.calculator_args["excited_state_indices"] = [3, 0]
    changed = ingredients.ab_initio_property_calculator_pyscf(model, parameters,
                                                              batch_size=2, property_dict=request)
    for i in range(2):
        assert parameters["_pyscf_calculators"][i]["calculator"] is not first[i]
        np.testing.assert_allclose(changed[i]["energy"][[0, 2, 1]], results[i]["energy"], atol=1e-10)


def test_reordered_response_roots_map_every_property(cases):
    _, reference = cases()
    result = calculator(excited_state_indices=[3, 0]).calculate(all_properties())
    order = [0, 2, 1]
    overlap = wavefunction_overlaps(reference["excited_amplitudes"],
                                   reference["excited_amplitudes"])
    np.testing.assert_allclose(overlap, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(result["energy"][order], reference["energy"], atol=1e-10)
    np.testing.assert_allclose(result["gradient"][:, :, order], reference["gradient"], atol=1e-9)
    for pair, old_pair in (((0, 1), (0, 2)), ((0, 2), (0, 1)), ((1, 2), (1, 2))):
        sign = -1 if pair == (1, 2) else 1
        np.testing.assert_allclose(result["derivative_coupling"][pair],
                                   sign * reference["derivative_coupling"][old_pair], atol=1e-9)


@pytest.mark.parametrize("spin", [1, 3])
@pytest.mark.parametrize("algorithm", [MeanFieldAbInitio, FewestSwitchesSurfaceHoppingAbInitio])
def test_short_trajectories_with_excited_electronic_superposition(spin, algorithm):
    sim = Simulation({"tmax": 0.4, "dt_update": 0.2, "dt_collect": 0.2,
                      "num_trajs": 1, "batch_size": 1, "progress_bar": False})
    sim.model = tda_model(spin)
    sim.model.ingredients = sim.model.ingredients + [
        ("init_classical", ingredients.init_classical_definite_position_momentum)]
    sim.algorithm = algorithm()
    sim.initial_state["wf_adb"] = np.array([0, 1, 1], dtype=complex) / np.sqrt(2)
    data = serial_driver(sim)
    for value in data.data_dict.values():
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(np.trace(data.data_dict["dm_adb"], axis1=-2, axis2=-1), 1, atol=1e-10)


@pytest.mark.parametrize("overrides, message", [
    ({"tda": False}, "(?i)tda"),
    ({"xc": "scan"}, "functional|PBE|pbe"),
    ({"multiplicity": 3}, "reference|multiplicity"),
    ({"excited_multiplicity": 2}, "multiplicity"),
    ({"excited_state_indices": [0, 0]}, "distinct|duplicate"),
    ({"excited_state_indices": [0]}, "num_quantum_states|length|exactly"),
    ({"excited_state_indices": [-1, 3]}, "nonnegative|at least"),
    ({"excited_state_indices": [0, 50]}, "roots|excitations|space"),
    ({"td_options": {"frozen": 1}}, "structural|frozen|Unsupported"),
    ({"scf_options": {"_eri": np.zeros((6, 6, 6, 6))}}, "structural|unsupported"),
    ({"scf_options": {"small_rho_cutoff": 1e-6}}, "small_rho_cutoff"),
    ({"grids_options": {"atom_grid": (0, 590)}}, "radial|at least"),
    ({"grids_options": {"atom_grid": (50, 123)}}, "Lebedev"),
    ({"grids_options": {"prune": "none"}}, "prune"),
    ({"grids_options": {"radi_method": None}}, "grids_options"),
])
def test_invalid_configuration_is_explicit(overrides, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        calculator(**overrides).calculate({"energy": {}})


def test_explicit_scf_and_tda_convergence_failures():
    with pytest.raises(RuntimeError, match="SCF.*converg"):
        calculator(scf_options={"max_cycle": 0}).calculate({"energy": {}})
    with pytest.raises(RuntimeError, match="TDA.*converg|roots.*converg"):
        calculator(td_options={"max_cycle": 0}).calculate({"energy": {}})


@pytest.mark.parametrize("xc", FUNCTIONALS)
def test_notebook_rks_recovery_preserves_tda_properties(monkeypatch, xc):
    """The kicked Wigner run stalled at this translated R=5.46937 geometry."""
    from pyscf.scf import hf
    from pyscf.soscf import newton_ah

    original = hf.SCF.kernel
    attempts = []

    def stalled_diis(reference, *args, **kwargs):
        attempts.append(reference)
        value = original(reference, *args, **kwargs)
        # Force the same recovery path on platforms/functionals where DIIS
        # succeeds here; the orbitals and the Newton solve remain native PySCF.
        reference.converged = False
        return value

    monkeypatch.setattr(hf.SCF, "kernel", stalled_diis)
    positions = np.array([[0.0, 0.0, -0.31016296672], [0.0, 0.0, 5.1592085638]])
    calc = calculator(
        xc, atom_positions=positions, num_quantum_states=2,
        excited_state_indices=[0], use_etfs=True,
    )
    result = calc.calculate(all_properties())
    reference = calc._mc._scf
    assert len(attempts) == 1
    assert calc._scf_recovery["algorithm"] == "newton"
    assert reference.converged and np.all(calc._mc.converged)
    assert type(reference) is dft.rks.RKS
    assert not isinstance(reference, newton_ah._CIAH_SOSCF)
    assert reference.xc == xc and reference.grids.level == 5
    assert reference.grids is attempts[0].grids
    assert reference._numint is attempts[0]._numint
    assert reference.conv_tol == 1e-12 and reference.max_cycle == 200
    assert result["energy"].shape == (2,)
    assert result["gradient"].shape == (2, 3, 2)
    assert np.all(np.isfinite(result["energy"]))
    assert np.all(np.isfinite(result["gradient"]))
    assert np.all(np.isfinite(result["derivative_coupling"][(0, 1)]))
    saved = pickle.dumps(result["excited_amplitudes"])
    repeated = calc.calculate({
        "energy": {"excited_amplitudes": True}, "gradient": {},
        "derivative_coupling": {},
        "wf_overlaps": {"amplitudes_previous": result["excited_amplitudes"]},
    })
    assert len(attempts) == 1
    np.testing.assert_allclose(repeated["wf_overlaps"], np.eye(2), atol=1e-10)
    assert pickle.dumps(result["excited_amplitudes"]) == saved


def test_response_failure_restores_native_solver_and_degeneracies_are_explicit():
    from pyscf.scf import cphf

    native = cphf.solve
    for property_name in ("gradient", "derivative_coupling"):
        with pytest.raises(RuntimeError, match="CPKS.*converg|response.*converg"):
            calculator(response_options={"max_cycle": 0}).calculate({property_name: {}})
        assert cphf.solve is native
    with pytest.raises((ValueError, RuntimeError), match="degenerate"):
        calculator(excited_state_indices=[1, 2]).calculate({"derivative_coupling": {}})


def test_ground_only_atom_without_virtual_orbitals():
    calc = calculator(atom_positions=np.zeros((1, 3)), atom_masses=[4 * AMU_TO_EMASS],
                      atom_names=["He"], num_quantum_states=1, excited_state_indices=[])
    result = calc.calculate(all_properties())
    np.testing.assert_allclose(result["gradient"], 0, atol=1e-10)
    assert result["derivative_coupling"] == {}
    np.testing.assert_allclose(wavefunction_overlaps(result["excited_amplitudes"],
                                                     result["excited_amplitudes"]), [[1]], atol=1e-12)
    modes = calc.normal_modes()
    assert modes["harmonic_frequency"].shape == (0,)
    assert modes["normal_mode"].shape == (3, 0)


def test_explicit_grid_is_copied_used_by_solver_and_checked_in_phase_history():
    grid = {"atom_grid": (40, 194), "prune": None}
    calc = calculator(grids_options=grid)
    grid["atom_grid"] = (50, 302)
    result = calc.calculate({"energy": {"excited_amplitudes": True}})
    with lib.with_omp_threads(1):
        mf, td = direct_tda("pbe", 1, grids_options={"atom_grid": (40, 194), "prune": None})
    np.testing.assert_allclose(result["energy"],
                               np.r_[mf.e_tot, mf.e_tot + td.e[ROOTS]], atol=1e-10)
    assert result["excited_amplitudes"]["grids_options"] == {"atom_grid": (40, 194), "prune": None}
    with pytest.raises(ValueError, match="grids_options"):
        calculator().calculate({"wf_overlaps": {"amplitudes_previous": result["excited_amplitudes"]}})
