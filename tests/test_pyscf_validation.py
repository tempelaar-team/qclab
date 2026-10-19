"""Validate optional dependencies, solver failures, phase reuse, and spin sectors."""

import copy
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from qclab.interfaces import pyscf as backend
from qclab.functions import z_to_p, z_to_q
from qclab.models import AbInitio
from qclab.numerical_constants import ANGSTROM_TO_BOHR


H2_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])


@pytest.fixture
def pyscf_backend():
    """Keep numerical helper and dependency checks active without PySCF."""
    if backend.gto is None:
        pytest.skip("PySCF is not installed.")
    return backend


def calculator(**overrides):
    """Make a small, inexpensive restricted CASSCF calculation."""
    options = {
        "atom_positions": H2_POSITIONS.copy(),
        "atom_masses": np.array([1837.0, 1837.0]),
        "atom_names": ["H", "H"],
        "num_quantum_states": 2,
        "ncas": 2,
        "nelecas": 2,
        "num_threads": 1,
    }
    options.update(overrides)
    return backend.QCLabPySCFInterface(**options)


def test_missing_pyscf_has_installation_instructions(monkeypatch):
    monkeypatch.setattr(backend, "gto", None)
    with pytest.raises(ImportError, match=r"qclab\[pyscf\]"):
        calculator()


def test_ab_initio_import_without_pyscf_or_ase():
    """Keep the model importable when neither electronic backend is installed."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingBackends(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'pyscf', 'ase'}:
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, MissingBackends())
        from qclab.models import AbInitio
        from qclab.interfaces import QCLabPySCFInterface
        from qclab.interfaces import pyscf as backend
        assert backend.gto is None
        assert backend.mcpdft is None
        try:
            QCLabPySCFInterface(None, None, None, 2, method_es='cms-pdft')
        except ImportError as error:
            assert 'qclab[pyscf]' in str(error)
        else:
            raise AssertionError('Missing PySCF must raise an installation error')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_missing_cms_capability_leaves_casscf_usable(pyscf_backend, monkeypatch):
    """An unavailable optional PDFT module must not disable ordinary CASSCF."""
    monkeypatch.setattr(backend, "mcpdft", None)
    with pytest.raises(ImportError, match="CMS-PDFT"):
        calculator(method_es="cms-pdft")
    result = calculator(num_quantum_states=1).calculate({"energy": {}})
    assert result["energy"].shape == (1,)
    assert np.all(np.isfinite(result["energy"]))


def test_nonlinear_modes_remove_rigid_motion_and_preserve_frequency():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = np.array([2.0, 3.0, 5.0])
    frequency = 0.125
    hessian = np.diag(np.repeat(masses, 3) * frequency**2)
    result = backend._harmonic_modes(hessian, positions, masses)
    modes = result["normal_mode"]
    assert modes.shape == (9, 3)
    assert result["normal_mode_convention"] == "mass_weighted"
    np.testing.assert_allclose(result["harmonic_frequency"], frequency, atol=1e-13)
    np.testing.assert_allclose(modes.T @ modes, np.eye(3), atol=1e-13)
    weighted = modes.reshape((3, 3, 3)) * np.sqrt(masses)[:, None, None]
    np.testing.assert_allclose(np.sum(weighted, axis=0), 0, atol=1e-13)
    centered = positions - np.average(positions, weights=masses, axis=0)
    for mode_ind in range(3):
        np.testing.assert_allclose(
            np.sum(np.cross(centered, weighted[:, :, mode_ind]), axis=0),
            0,
            atol=1e-13,
        )


def test_unstable_modes_remain_imaginary():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = np.array([2.0, 3.0, 5.0])
    hessian = -np.diag(np.repeat(masses, 3) * 0.125**2)
    frequencies = backend._harmonic_modes(hessian, positions, masses)["harmonic_frequency"]
    assert frequencies.shape == (3,)
    assert np.iscomplexobj(frequencies)
    np.testing.assert_allclose(frequencies, 0.125j, atol=1e-13)


def test_sampled_wigner_energy_matches_harmonic_zero_point_energy():
    masses = np.array([2.0, 8.0])
    mode = np.zeros((6, 1))
    mode[0, 0] = np.sqrt(8.0 / 10.0)
    mode[3, 0] = -np.sqrt(2.0 / 10.0)
    frequency = 0.2
    model = AbInitio(
        {
            "atom_names": ["H", "He"],
            "atom_positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "atom_masses": masses,
            "num_quantum_states": 1,
            "normal_mode": mode,
            "normal_mode_convention": "mass_weighted",
            "harmonic_frequency": np.array([frequency]),
            "kBT": 0.0,
        }
    )
    z = model.init_classical({}, seed=np.arange(4000))
    assert np.all(np.isfinite(z))
    coordinate_mass = model.constants.classical_coordinate_mass
    weight = model.constants.classical_coordinate_weight
    q = z_to_q(z, coordinate_mass, weight) - model.constants.init_position
    p = z_to_p(z, coordinate_mass, weight)
    q_mode = np.einsum("tc,cm->tm", q * np.sqrt(coordinate_mass), mode)
    energy = np.sum(p * p / (2 * coordinate_mass), axis=1)
    energy += 0.5 * frequency**2 * q_mode[:, 0] ** 2
    np.testing.assert_allclose(np.mean(energy), 0.5 * frequency, rtol=0.05)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"method_es": "tdhf"}, "method_es"),
        ({"weights": [0.75, 0.25]}, "equal state-average"),
        ({"ncas": 0}, "ncas"),
        ({"nelecas": 3}, "spin multiplicity"),
        ({"nelecas": 2.0}, "electron count"),
        ({"num_quantum_states": 4}, "roots available"),
        (
            {"num_quantum_states": 3, "multiplicity": 2, "charge": 1, "nelecas": 1},
            "roots available",
        ),
        ({"active_orbitals": [0, 0]}, "distinct orbital"),
        ({"response_options": {"atmlst": [0]}}, "response_options"),
        ({"response_options": {"ci": np.ones((2, 2))}}, "response_options"),
    ],
)
def test_invalid_configuration_is_actionable(pyscf_backend, options, message):
    with pytest.raises(ValueError, match=message):
        calculator(**options)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"num_quantum_states": 1}, "num_quantum_states"),
        ({"otxc": "tPBE0"}, "otxc"),
        ({"otxc": "tLDA"}, "otxc"),
        ({"grids_level": -1}, "grids_level"),
        ({"grids_level": 2.5}, "grids_level"),
        ({"grids_level": True}, "grids_level"),
        ({"cms_options": {"diabatization": "XMS"}}, "cms_options"),
        ({"cms_options": {"si": np.eye(2)}}, "cms_options"),
        ({"cms_options": {"max_cyc_diabatize": 0}}, "max_cyc_diabatize"),
        ({"cms_options": {"max_cyc_diabatize": 1.5}}, "max_cyc_diabatize"),
        ({"cms_options": {"conv_tol_diabatize": -1}}, "conv_tol_diabatize"),
        ({"cms_options": {"conv_tol_diabatize": np.nan}}, "conv_tol_diabatize"),
        ({"cms_options": {"sing_tol_diabatize": 0}}, "sing_tol_diabatize"),
        ({"cms_options": {"nudge_tol_diabatize": True}}, "nudge_tol_diabatize"),
    ],
)
def test_invalid_cms_configuration_is_actionable(pyscf_backend, options, message):
    with pytest.raises(ValueError, match=message):
        calculator(method_es="cms-pdft", **options)


@pytest.mark.parametrize(
    "name, value",
    [
        ("frozen", 0),
        ("otxc", "tPBE0"),
        ("diabatization", "LIN"),
        ("si", np.eye(2)),
        ("with_df", None),
    ],
)
def test_cms_numerical_options_cannot_replace_the_model(pyscf_backend, name, value):
    calc = calculator(method_es="cms-pdft", casscf_options={name: value})
    with pytest.raises(ValueError, match=name):
        calc.calculate({"energy": {}})
    assert calc._snapshot is None


def test_active_space_must_fit_basis(pyscf_backend):
    calc = calculator(ncas=3)
    with pytest.raises(ValueError, match="exceeds the molecular orbital basis"):
        calc.calculate({"energy": {}})


def test_skipped_request_does_not_solve(pyscf_backend, monkeypatch):
    calc = calculator()

    def unexpected_solve(*args, **kwargs):
        pytest.fail("A skipped property must not trigger an electronic structure solve.")

    monkeypatch.setattr(calc, "_run", unexpected_solve)
    result = calc.calculate({"gradient": {"calc_property": False}})
    assert result == {"gradient": None}
    assert calc._snapshot is None


def test_single_state_energy_and_gradient(pyscf_backend):
    result = calculator(num_quantum_states=1).calculate({"energy": {}, "gradient": {}})
    assert result["energy"].shape == (1,)
    assert -1.2 < result["energy"][0] < -1.0
    assert result["gradient"].shape == (2, 3, 1)
    assert np.all(np.isfinite(result["gradient"]))
    np.testing.assert_allclose(np.sum(result["gradient"], axis=0), 0, atol=1e-10)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"scf_options": {"max_cycle": 0}}, "SCF did not converge"),
        ({"casscf_options": {"max_cycle_macro": 0}}, "CASSCF did not converge"),
    ],
)
def test_nonconvergence_stops_calculation(pyscf_backend, options, message):
    calc = calculator(**options)
    with pytest.raises(RuntimeError, match=message):
        calc.calculate({"energy": {}})
    assert calc._snapshot is None


@pytest.mark.parametrize(
    "method, bond, otxc",
    [("casscf", bond, "tPBE") for bond in (12.0, 20.0, 31.39409966)]
    + [("cms-pdft", bond, "tPBE") for bond in (12.0, 20.0, 31.39409966)]
    + [("cms-pdft", 20.0, otxc) for otxc in ("tBLYP", "ftPBE")],
)
def test_stretched_lih_scf_recovery_preserves_responses_and_reuse(
    pyscf_backend, monkeypatch, method, bond, otxc
):
    """Recover the notebook's failed DIIS references without changing its model."""
    from pyscf.soscf import newton_ah

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]])
    is_cms = method == "cms-pdft"
    scf_options = {"conv_tol": 1e-12 if is_cms else 1e-10}
    casscf_options = {
        "conv_tol": 1e-12 if is_cms else 1e-9,
        "conv_tol_grad": 1e-8 if is_cms else 1e-6,
        "ah_conv_tol": 1e-14,
    }
    calc = calculator(
        atom_positions=positions, atom_names=["Li", "H"],
        method_es=method, otxc=otxc, scf_options=scf_options,
        casscf_options=casscf_options,
        cms_options={"conv_tol_diabatize": 1e-10} if is_cms else None,
        response_options={"conv_atol": 1e-10, "conv_rtol": 0.0},
    )
    result = calc.calculate({
        "energy": {"excited_amplitudes": True}, "gradient": {},
        "derivative_coupling": {},
    })
    # DIIS trajectories can differ across BLAS implementations. A reference
    # that converges without recovery is equally valid on another platform.
    recovery = getattr(calc, "_scf_recovery", None)
    if recovery is not None:
        assert recovery["algorithm"] == "newton"
        assert recovery["conv_tol"] == scf_options["conv_tol"]
        assert recovery["max_cycle"] == (200 if is_cms else 50)
    reference = calc._mc._scf
    assert reference.converged and calc._mc.converged
    assert not isinstance(reference, newton_ah._CIAH_SOSCF)
    assert calc._gradient_solver.converged and calc._coupling_solver.converged
    assert result["energy"].shape == (2,)
    assert result["gradient"].shape == (2, 3, 2)
    assert np.all(np.isfinite(result["energy"]))
    assert np.all(np.isfinite(result["gradient"]))
    coupling = result["derivative_coupling"][(0, 1)]
    assert np.all(np.isfinite(coupling))

    if bond == 20.0:
        # An independent cold Newton reference reaches the same physical states.
        with backend.lib.with_omp_threads(1):
            mol = backend.gto.M(
                atom=list(zip(["Li", "H"], positions)), unit="Bohr",
                basis="sto-3g", verbose=0,
            )
            hf = backend.scf.RHF(mol).set(**scf_options)
            hf.chkfile = None
            hf.max_cycle = 200 if is_cms else 50
            hf = hf.newton().run().remove_soscf()
            assert hf.converged
            direct = (
                backend.mcpdft.CASSCF(hf, otxc, 2, 2, grids_level=5)
                if is_cms else backend.mcscf.CASSCF(hf, 2, 2)
            )
            direct.chkfile = None
            direct.fcisolver = backend.fci.direct_spin0.FCI(mol)
            direct.fcisolver.conv_tol = 1e-14 if is_cms else 1e-12
            direct.fix_spin_(ss=0.0)
            if is_cms:
                direct = direct.multi_state([0.5, 0.5], method="CMS")
                direct.conv_tol_diabatize = 1e-10
            else:
                direct.state_average_([0.5, 0.5])
            direct.set(**casscf_options).run()
            assert direct.converged
            gradients = direct.nuc_grad_method().set(conv_atol=1e-10, conv_rtol=0.0)
            expected = np.stack([gradients.kernel(state=i) for i in range(2)], axis=-1)
            assert gradients.converged
        np.testing.assert_allclose(result["energy"], direct.e_states, atol=1e-9, rtol=0)
        np.testing.assert_allclose(result["gradient"], expected, atol=1e-8, rtol=0)

    saved_snapshot = copy.deepcopy(result["excited_amplitudes"])

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Recovered geometry must be reused across property requests.")

    monkeypatch.setattr(calc, "_run", unexpected_solve)
    repeated = calc.calculate({
        "energy": {"excited_amplitudes": True}, "gradient": {},
        "derivative_coupling": {"state_inds_derivative_coupling": [1, 0]},
        "wf_overlaps": {"amplitudes_previous": saved_snapshot},
    })
    assert calc._mc._scf is reference
    np.testing.assert_allclose(repeated["wf_overlaps"], np.eye(2), atol=1e-10)
    np.testing.assert_array_equal(repeated["gradient"], result["gradient"])
    np.testing.assert_array_equal(repeated["derivative_coupling"][(1, 0)], -coupling)
    for before, after in zip(saved_snapshot["ci"], result["excited_amplitudes"]["ci"]):
        np.testing.assert_array_equal(before, after)


def test_scf_recovery_respects_explicit_iteration_limit(pyscf_backend):
    calc = calculator(
        atom_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
        atom_names=["Li", "H"], scf_options={"max_cycle": 1},
    )
    with pytest.raises(RuntimeError, match="ordinary SCF and one Newton SCF retry") as caught:
        calc.calculate({"energy": {}})
    assert "max_cycle=1" in str(caught.value)
    assert "minimum_nuclear_distance_bohr=20.0" in str(caught.value)
    assert calc._snapshot is None and calc._mc is None


@pytest.mark.parametrize("open_shell", [False, True])
def test_scf_retry_preserves_native_reference_and_thresholds(
    pyscf_backend, monkeypatch, open_shell
):
    """Exercise successful recovery deterministically with real RHF and ROHF."""
    from pyscf.soscf import newton_ah

    original_diis = backend.scf.hf.SCF.kernel
    original_newton = newton_ah._CIAH_SOSCF.kernel
    attempts = []

    def stalled_diis(solver, *args, **kwargs):
        attempts.append("diis")
        value = original_diis(solver, *args, **kwargs)
        # Keep genuine PySCF orbitals, but force the nonconvergence path even
        # on platforms where ordinary SCF readily converges for this molecule.
        solver.converged = False
        return value

    def newton(solver, *args, **kwargs):
        attempts.append("newton")
        assert solver.conv_tol == 1e-10
        assert solver.conv_tol_grad == 3e-6
        assert solver.max_cycle == 12
        return original_newton(solver, *args, **kwargs)

    monkeypatch.setattr(backend.scf.hf.SCF, "kernel", stalled_diis)
    monkeypatch.setattr(newton_ah._CIAH_SOSCF, "kernel", newton)
    calc = calculator(
        charge=int(open_shell), multiplicity=2 if open_shell else 1,
        nelecas=1 if open_shell else 2,
    )
    with backend.lib.with_omp_threads(1):
        mol = backend.gto.M(
            atom=list(zip(["H", "H"], H2_POSITIONS)), unit="Bohr",
            basis="sto-3g", charge=int(open_shell), spin=int(open_shell), verbose=0,
        )
        mf = (backend.scf.ROHF if open_shell else backend.scf.RHF)(mol)
        mf.chkfile = None
        mf.set(conv_tol=1e-10, conv_tol_grad=3e-6, max_cycle=12)
        recovered = calc._solve_scf(mf, H2_POSITIONS)
        gradient = recovered.nuc_grad_method().kernel()
    assert attempts == ["diis", "newton"]
    assert type(recovered) is type(mf)
    assert recovered.converged and np.all(np.isfinite(gradient))
    assert not isinstance(recovered, newton_ah._CIAH_SOSCF)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"scf_options": {"max_cycle": 0}}, "SCF did not converge"),
        ({"casscf_options": {"max_cycle_macro": 0}}, "did not converge"),
        ({"cms_options": {"max_cyc_diabatize": 1}}, "did not converge"),
    ],
)
def test_cms_nonconvergence_stops_calculation(pyscf_backend, options, message):
    calc = calculator(method_es="cms-pdft", **options)
    with pytest.raises(RuntimeError, match=message):
        calc.calculate({"energy": {}})
    assert calc._snapshot is None


@pytest.mark.parametrize("property_name", ["gradient", "derivative_coupling"])
@pytest.mark.parametrize(
    "response, converged, message",
    [
        (np.full((2, 3), np.nan), True, "invalid"),
        (np.zeros((1, 3)), True, "invalid"),
        (np.ones((2, 3)) * 1j, True, "invalid"),
        (np.zeros((2, 3)), False, "did not converge"),
    ],
)
def test_invalid_response_is_rejected_before_caching(
    pyscf_backend, property_name, response, converged, message
):
    calc = calculator()
    calc.calculate({"energy": {}})

    class Response:
        def kernel(self, **kwargs):
            return response

    solver = Response()
    solver.converged = converged
    if property_name == "gradient":
        calc._gradient_solver = solver
    else:
        calc._coupling_solver = solver
    with pytest.raises(RuntimeError, match=message):
        calc.calculate({property_name: {}})
    assert calc._gradients == {}
    assert calc._couplings == {}


def test_same_geometry_properties_reuse_solve(pyscf_backend, monkeypatch):
    calc = calculator(num_quantum_states=1)
    calls = []
    original_run = calc._run

    def recording_run(*args, **kwargs):
        calls.append(np.asarray(args[0]).copy())
        return original_run(*args, **kwargs)

    monkeypatch.setattr(calc, "_run", recording_run)
    calc.calculate({"energy": {}})
    calc.calculate({"gradient": {}})
    calc.calculate({"energy": {}})
    assert len(calls) == 1
    shifted = H2_POSITIONS.copy()
    shifted[1, 2] += 0.001
    calc.calculate({"energy": {}}, atom_positions=shifted)
    assert len(calls) == 2


def test_cached_geometry_rephases_snapshot_and_nac_without_solving(
    pyscf_backend, monkeypatch
):
    calc = calculator()
    original = calc.calculate(
        {"energy": {"excited_amplitudes": True}, "derivative_coupling": {}}
    )
    previous = copy.deepcopy(original["excited_amplitudes"])
    previous["ci"] = tuple(
        ci * (-1 if index == 0 else 1) for index, ci in enumerate(previous["ci"])
    )
    saved_previous = copy.deepcopy(previous)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Changing only the reference phase must reuse the solved wavefunction.")

    monkeypatch.setattr(calc, "_run", unexpected_solve)
    result = calc.calculate(
        {
            "energy": {"excited_amplitudes": True},
            "derivative_coupling": {},
            "wf_overlaps": {"amplitudes_previous": previous},
        }
    )
    np.testing.assert_allclose(result["wf_overlaps"], np.eye(2), atol=1e-12)
    for index, ci in enumerate(result["excited_amplitudes"]["ci"]):
        sign = -1 if index == 0 else 1
        np.testing.assert_allclose(ci, sign * original["excited_amplitudes"]["ci"][index])
        np.testing.assert_array_equal(calc._mc.ci[index], ci)
        np.testing.assert_array_equal(previous["ci"][index], saved_previous["ci"][index])
    np.testing.assert_allclose(
        result["derivative_coupling"][(0, 1)],
        -original["derivative_coupling"][(0, 1)],
        atol=1e-12,
    )


def test_doublet_gradient_and_nac_match_overlap_derivative(pyscf_backend):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    options = {"atom_positions": positions, "charge": 1, "multiplicity": 2, "nelecas": 1}
    calc = calculator(**options)
    result = calc.calculate(
        {
            "energy": {"excited_amplitudes": True},
            "gradient": {},
            "derivative_coupling": {},
        }
    )
    assert result["gradient"].shape == (2, 3, 2)
    assert np.all(np.isfinite(result["gradient"]))
    np.testing.assert_allclose(np.sum(result["gradient"], axis=0), 0, atol=1e-10)
    reference = calc._mc._scf
    assert reference.mol.spin == 1
    calc.calculate({"derivative_coupling": {}})
    assert calc._mc._scf is reference

    delta = 1e-3
    overlaps = []
    for direction in (-1, 1):
        shifted = positions.copy()
        shifted[1, 2] += direction * delta
        shifted_options = dict(options, atom_positions=shifted)
        displaced = calculator(**shifted_options).calculate(
            {"energy": {"excited_amplitudes": True}}
        )
        overlap = backend.wavefunction_overlaps(
            result["excited_amplitudes"], displaced["excited_amplitudes"]
        )
        overlaps.append(overlap * np.sign(np.diag(overlap))[None, :])
    finite_difference = (overlaps[1] - overlaps[0])[0, 1] / (2 * delta)
    coupling = result["derivative_coupling"][(0, 1)] / ANGSTROM_TO_BOHR
    assert np.all(np.isfinite(coupling))
    np.testing.assert_allclose(coupling[1, 2], finite_difference, atol=2e-5, rtol=5e-3)
