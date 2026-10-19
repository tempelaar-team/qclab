"""Exercise Q-Chem input and output contracts without an installed executable."""

import copy
import io

import numpy as np
import pytest

from qclab.interfaces.qchem import QCLabQChemInterface
from qclab.numerical_constants import ANGSTROM_TO_BOHR


def calculator(**kwargs):
    args = dict(method_es="tddft", charge=0, multiplicity=1,
                num_quantum_states=3, excited_state_indices=[2, 0])
    args.update(kwargs)
    return QCLabQChemInterface(np.zeros((1, 3)), np.array([1836.0]), ["H"], **args)


def write_jobs(calc, **properties):
    output = io.StringIO()
    calc._write_job(output, calc._build_job_specs(properties), **properties)
    return output.getvalue()


def rem_sections(text):
    return [dict(line.lower().split(None, 1) for line in part.split("$end")[0].splitlines()
                 if line.strip()) for part in text.split("$rem\n")[1:]]


def test_shared_theory_reaches_all_jobs_and_both_overlap_geometries():
    calc = calculator(xc="PBE0", basis="STO-3G", purecart=1111, xc_grid="000099000590",
                      scf_convergence=10, cis_convergence=9, thresh=14)
    text = write_jobs(calc, energy={"excited_amplitudes": True},
                      gradient={"state_inds_gradient": [1, 0, 2]}, frequency={},
                      derivative_coupling={"state_inds_derivative_coupling": [(1, 2)]},
                      wf_overlaps={"atom_positions_previous": np.ones((1, 3))})
    sections = rem_sections(text)
    assert len(sections) == 8
    for rem in sections:
        assert rem["exchange"] == "pbe0"
        assert rem["basis"] == "sto-3g"
        assert rem["cis_n_roots"] == "3"
        assert rem["cis_singlets"] == "true"
        assert rem["cis_triplets"] == "false"
        assert rem["rpa"] == "false"
        assert rem["set_quadratic"] == "false"
        assert rem["purecart"] == "1111"
        assert rem["xc_grid"] == "000099000590"
        assert rem["scf_convergence"] == "10"
        assert rem["cis_convergence"] == "9"
        assert rem["thresh"] == "14"
    assert sections[1]["cis_state_deriv"] == "3"
    assert "cis_state_deriv" not in sections[2]
    assert sections[3]["cis_state_deriv"] == "1"
    assert sections[-2]["mo_overlaps_two_geoms"] == "1"
    assert sections[-1]["mo_overlaps_two_geoms"] == "2"
    assert "\n3 1\n" in text
    assert "cis_state_deriv" not in calc.job_templates["gradient"]["qchem_parameters"]["tddft"]


def test_legacy_cis_constructor_and_explicit_full_tddft():
    calc = calculator(method_es="cis", num_quantum_states=None, excited_state_indices=None,
                      cis_n_roots=2)
    assert calc.state_indices == (0, 1, 2)
    assert calc.kwargs["exchange"] == "HF"
    assert calculator(tda=False).kwargs["rpa"] is True
    assert calculator().tda is True


@pytest.mark.parametrize("kwargs,match", [
    ({"tda": False, "rpa": False}, "contradicts"),
    ({"method_es": "cis", "tda": False}, "CIS requires"),
    ({"excited_multiplicity": 5}, "excited_multiplicity"),
    ({"excited_multiplicity": 3, "cis_singlets": True}, "contradicts"),
    ({"excited_state_indices": [0]}, "num_quantum_states - 1"),
    ({"excited_state_indices": [0, 0]}, "unique"),
    ({"cis_n_roots": 2}, "include every selected"),
    ({"excited_multiplicity": 1, "multiplicity": 3}, "singlet reference"),
    ({"excited_multiplicity": 1, "unrestricted": True}, "restricted reference"),
    ({"xc": "PBE", "exchange": "PBE0"}, "same functional"),
])
def test_rejects_ambiguous_theory_and_invalid_root_selections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        calculator(**kwargs)


def test_property_override_cannot_change_shared_model_or_mutate_input():
    calc = calculator()
    request = {"gradient": {"qchem_args": {"RPA": "TRUE"}}}
    before = copy.deepcopy(request)
    with pytest.raises(ValueError, match="shared Q-Chem parameter rpa"):
        write_jobs(calc, **request)
    assert request == before
    # Job-local printing settings are still accepted.
    assert "SCF_PRINT" in write_jobs(calc, energy={"qchem_args": {"scf_print": 2}})


@pytest.mark.parametrize("key,value", [
    ("unrestricted", True), ("spin_flip", True),
    ("correlation", "MP2"), ("purecart", 2222),
])
def test_property_override_cannot_add_structural_setting_absent_from_shared_model(key, value):
    calc = calculator(excited_multiplicity=1)
    request = {"energy": {}, "gradient": {"qchem_args": {key: value}}}
    before = copy.deepcopy(request)
    with pytest.raises(ValueError, match="shared Q-Chem parameter " + key):
        write_jobs(calc, **request)
    assert request == before
    assert key not in calc._shared_parameters


def test_property_override_can_repeat_shared_structural_setting():
    calc = calculator(purecart=1111)
    text = write_jobs(calc, energy={}, gradient={"qchem_args": {"purecart": 1111}})
    assert all(rem["purecart"] == "1111" for rem in rem_sections(text))


def test_legacy_open_shell_quadratic_and_property_convergence_controls_are_preserved():
    calc = calculator(multiplicity=3, unrestricted=True, spin_flip=True,
                      set_quadratic=True)
    assert calc.excited_multiplicity is None
    assert calc.kwargs["set_quadratic"] is True
    text = write_jobs(calc, energy={"qchem_args": {"scf_convergence": 11}},
                      derivative_coupling={})
    assert rem_sections(text)[0]["scf_convergence"] == "11"
    assert "\n0 3 1\n" in text
    assert calculator(rpa=True).tda is False
    assert calculator(cis_singlets=True, cis_triplets=True).excited_multiplicity is None
    assert calculator(cis_singlets="FALSE").excited_multiplicity == 3


def test_energy_projection_uses_native_labels_not_output_order():
    calc = calculator()
    calc._pull_energy([
        "Total energy in the final basis set = -8.0",
        "Total energy for state 2: -7.2 au",
        "Total energy for state 1: -7.5 au",
        "Total energy for state 3: -7.0 au",
    ])
    np.testing.assert_array_equal(calc.results["energy"], [-8.0, -7.0, -7.5])
    with pytest.raises(ValueError, match="missing total energies"):
        calc._pull_energy(["Total energy in the final basis set = -8.0"])


def snapshot():
    # Three orthonormal spin-adapted single excitations: one occupied,
    # three virtual orbitals; x_alpha=x_beta ensures singlet normalization.
    x = np.eye(3).reshape((3, 1, 3)) / np.sqrt(2.0)
    return dict(x_alpha=x, x_beta=x.copy(), y_alpha=np.zeros_like(x),
                y_beta=np.zeros_like(x), num_basis_functions=4,
                num_alpha_electrons=1, num_beta_electrons=1,
                num_excited_states=3, mo_alpha=np.eye(4).ravel(order="F"),
                mo_beta=np.eye(4).ravel(order="F"))


def test_snapshot_projection_is_independent_and_overlap_order_is_consistent(monkeypatch):
    calc = calculator()
    original = snapshot()
    before = copy.deepcopy(original)
    projected = calc._project_amplitudes(original)
    assert projected["state_indices"] == (0, 3, 1)
    assert projected["num_excited_states"] == 2
    np.testing.assert_array_equal(projected["x_alpha"], original["x_alpha"][[2, 0]])
    projected["x_alpha"][0, 0, 0] = 99
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])
    projected = calc._project_amplitudes(original)
    monkeypatch.setattr(calc, "_compute_mo_overlaps", lambda: (np.eye(4), np.eye(4)))
    calc._pull_overlaps(original, projected)
    np.testing.assert_allclose(calc.results["wf_overlaps"], np.eye(3), atol=1e-15)
    # Projected snapshots are not indexed a second time with native root 3.
    calc._pull_overlaps(projected, projected)
    np.testing.assert_allclose(calc.results["wf_overlaps"], np.eye(3), atol=1e-15)
    wrong = dict(projected, state_indices=(0, 1, 3))
    with pytest.raises(ValueError, match="different selected states"):
        calc._pull_overlaps(wrong, projected)


def coupling_output(first=1, second=3):
    return ["Derivative coupling between states %d and %d" % (first, second),
            "without ETF", "", "Atom X Y Z", "1 1.0 2.0 3.0",
            "with ETF", "", "Atom X Y Z", "1 4.0 5.0 6.0"]


@pytest.mark.parametrize("use_etfs,expected", [(True, [4, 5, 6]), (False, [1, 2, 3])])
def test_coupling_maps_selected_pairs_converts_units_and_handles_reverse_sign(use_etfs, expected):
    calc = calculator(use_etfs=use_etfs)
    write_jobs(calc, derivative_coupling={"state_inds_derivative_coupling": [(1, 2)]})
    calc._pull_derivative_coupling(coupling_output(), 1)
    assert set(calc.results["derivative_coupling"]) == {(1, 2)}
    # Native (1,3) is selected (2,1), the reverse of requested (1,2).
    np.testing.assert_allclose(calc.results["derivative_coupling"][(1, 2)],
                               -np.array([expected]) * ANGSTROM_TO_BOHR)


def test_triplet_jobs_exclude_reference_from_nacs_and_return_exact_cross_spin_zeros(monkeypatch):
    calc = calculator(excited_multiplicity=3)
    text = write_jobs(calc, derivative_coupling={})
    assert "\n0 3 1\n" not in text
    assert "\n3 1\n" in text
    assert rem_sections(text)[0]["cis_der_numstate"] == "2"
    calc._pull_derivative_coupling(coupling_output(), 1)
    np.testing.assert_array_equal(calc.results["derivative_coupling"][(0, 1)], 0)
    np.testing.assert_array_equal(calc.results["derivative_coupling"][(0, 2)], 0)
    # A request containing only cross-spin pairs needs no native NAC job.
    text = write_jobs(calc, derivative_coupling={"state_inds_derivative_coupling": [(0, 1)]})
    assert "$derivative_coupling" not in text
    assert rem_sections(text)[0]["calc_nac"] == "false"
    assert rem_sections(text)[0]["cis_der_numstate"] == "0"
    calc._pull_derivative_coupling([], 1)
    np.testing.assert_array_equal(calc.results["derivative_coupling"][(0, 1)], 0)
    amplitudes = snapshot()
    amplitudes["x_beta"] *= -1
    monkeypatch.setattr(calc, "_compute_mo_overlaps", lambda: (np.eye(4), np.eye(4)))
    calc._pull_overlaps(amplitudes, amplitudes)
    np.testing.assert_allclose(calc.results["wf_overlaps"], np.eye(3), atol=1e-15)


def test_skip_requests_and_repeated_input_writes_do_not_leak_parameters(tmp_path):
    calc = calculator(label=str(tmp_path / "qchem"))
    calc.write_input(energy={"calc_property": False},
                     gradient={"state_inds_gradient": [1]})
    text = (tmp_path / "qchem.inp").read_text()
    assert len(rem_sections(text)) == 1
    assert rem_sections(text)[0]["cis_state_deriv"] == "3"
    calc.write_input(gradient={"state_inds_gradient": [0]})
    assert "cis_state_deriv" not in rem_sections((tmp_path / "qchem.inp").read_text())[0]
    calc.read_results(energy={"calc_property": False})
    assert calc.results == {}


def test_combined_output_parses_selected_gradients_and_native_couplings(tmp_path):
    calc = calculator(label=str(tmp_path / "qchem"))
    request = dict(energy={}, gradient={"state_inds_gradient": [1, 0, 2]},
                   derivative_coupling={"state_inds_derivative_coupling": [(1, 2)]})
    calc.write_input(**request)
    assert calc._get_number_of_jobs(request) == 5
    sections = rem_sections((tmp_path / "qchem.inp").read_text())
    assert len(sections) == 5

    def gradient(value):
        return ("Gradient of SCF Energy\n 1\n X %s\n Y %s\n Z %s\n"
                "Gradient time\n" % (value, value, value))

    jobs = ["Total energy in the final basis set = -8.0\n"
            "Total energy for state 1: -7.5 au\n"
            "Total energy for state 2: -7.2 au\n"
            "Total energy for state 3: -7.0 au\n",
            gradient(3), gradient(0), gradient(1), "\n".join(coupling_output())]
    (tmp_path / "qchem.out").write_text("".join(
        "Running Job %d\n%s\nThank you for using Q-Chem\n" % (i, job)
        for i, job in enumerate(jobs, 1)))
    calc.read_results(**request)
    np.testing.assert_array_equal(calc.results["energy"], [-8, -7, -7.5])
    assert calc.results["gradient"].shape == (1, 3, 3)
    np.testing.assert_array_equal(calc.results["gradient"][0, 0], [3, 0, 1])
    np.testing.assert_allclose(calc.results["derivative_coupling"][(1, 2)],
                               -np.array([[4, 5, 6]]) * ANGSTROM_TO_BOHR)
