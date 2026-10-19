"""
This module runs the models and algorithms in QC Lab for a given simulation setting and
compares them against a reference dataset.
"""

import pytest
import os
import time
from functools import partial
import numpy as np
from qclab import Simulation, Data
from qclab import tasks
from qclab.models import (
    SpinBoson,
    HolsteinLattice,
    HolsteinLatticeReciprocalSpace,
    FMOComplex,
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
)
from qclab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver, parallel_driver_multiprocessing

try:
    from tests.reference_settings import model_sim_settings, model_settings
except ImportError:
    from reference_settings import model_sim_settings, model_settings


def test_output_serial():
    """
    This test verifies that the serial driver produces the expected results.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = serial_driver(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_multiprocessing():
    """
    This test verifies that the multiprocessing driver produces the same results
    as the serial driver.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            sim.settings.batch_size = sim.settings.num_trajs // 2
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = parallel_driver_multiprocessing(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_different_h():
    """
    This test ensures that changing the value of the ``classical_coordinate_weight`` does
    not change the result of the simulation.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            np.random.seed(10)
            # Adjust the weights to be a random positive number between 1 and 2.
            sim.model.constants.classical_coordinate_weight = (
                np.random.rand(sim.model.constants.num_classical_coordinates) + 1.0
            )
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = serial_driver(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_fssh_gauge_fixing():
    """
    This test ensures that the initial choice of phase in FSSH does not affect
    the final result of a calculation.

    The test proceeds by adding a random phase to the eigenvectors computed at the
    outset of the simulation. The relative phases of these vectors are fixed so that
    the derivative couplings are real-valued ("phase_der_couple"). At subsequent
    timesteps, the relative phase of the new eigenvectors is determined from maximizing the
    real part of their overlap with the previous set of eigenvectors ("phase_overlap").

    Note that this test will only work if the ``add_random_phase`` task is between the
    ``diagonalize_matrix`` and ``update_eigvecs_gauge`` tasks in the initialization recipe.

    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    my_FSSH = FewestSwitchesSurfaceHopping({"gauge_fixing": "phase_overlap"})

    def add_random_phase(sim, state, parameters):
        np.random.seed(10)
        random_phases = np.exp(
            1j
            * 2.0
            * np.pi
            * np.random.rand(
                sim.settings.batch_size, sim.model.constants.num_quantum_states
            )
        )
        state["eigvecs"] = random_phases[:, np.newaxis, :] * state["eigvecs"]
        return state, parameters

    diagonalization_index = next(
        index for index, task in enumerate(my_FSSH.initialization_recipe)
        if (task.func if isinstance(task, partial) else task) is tasks.diagonalize_matrix
    )
    my_FSSH.initialization_recipe.insert(diagonalization_index + 1, add_random_phase)
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [my_FSSH]:
            print(f"Testing {model_class.__name__} with FewestSwitchesSurfaceHopping")
            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = "FewestSwitchesSurfaceHopping"
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = serial_driver(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_fssh_deterministic():
    """
    This test verifies the deterministic version of FSSH.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        algorithm_class = FewestSwitchesSurfaceHopping
        print(
            f"Testing {model_class.__name__} with deterministic {algorithm_class.__name__}"
        )

        sim = Simulation(model_sim_settings[model_class.__name__])
        model_name = model_class.__name__
        algorithm_name = algorithm_class.__name__
        sim.model = model_class(model_settings[model_class.__name__])
        sim.model.initialize_constants()
        sim.settings.batch_size *= sim.model.constants.num_quantum_states
        sim.settings.num_trajs *= sim.model.constants.num_quantum_states
        sim.algorithm = algorithm_class({"fssh_deterministic": True})
        sim.initial_state["wf_db"] = np.zeros(
            sim.model.constants.num_quantum_states, dtype=complex
        )
        sim.initial_state["wf_db"][0] = 1j
        st = time.time()
        data = serial_driver(sim)
        et = time.time()
        print(f"Finished in {et - st:.2f} seconds.")
        data_correct = Data().load(
            os.path.join(
                reference_folder, f"{model_name}_{algorithm_name}_deterministic.h5"
            )
        )
        for key, val in data.data_dict.items():
            np.testing.assert_allclose(
                val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
            )
    return


def test_dh_qc_dzc_finite_differences():
    """
    Tests the output when using finite differences to compute dh_qc_dzc.

    We change the atol to 1e-5 so that it matches better in TullyProblemThree.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.ingredients.append(("dh_qc_dzc", None))
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = serial_driver(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-5
                )
    return


def test_dh_c_dzc_finite_differences():
    """
    Tests the output when using finite differences to compute dh_c_dzc.
    We change the atol to 1e-4 and reduce the finite difference delta to 1e-9.
    This is just to obtain agreement with reference data,
    the defaults may be appropriate in other cases.

    We exclude SpinBoson and FMOComplex here because they take too long.
    """
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        # SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        # FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.ingredients.append(("dh_c_dzc", None))
            sim.model.initialize_constants()
            sim.model.constants.dh_c_dzc_finite_difference_delta = 1e-9
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            st = time.time()
            data = serial_driver(sim)
            et = time.time()
            print(f"Finished in {et - st:.2f} seconds.")
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-4
                )
    return


if __name__ == "__main__":
    st = time.time()
    test_output_serial()
    et1 = time.time()
    print(f"Serial tests completed in {et1 - st:.2f} seconds.")
    test_output_multiprocessing()
    et2 = time.time()
    print(f"Multiprocessing tests completed in {et2 - et1:.2f} seconds.")
    test_output_different_h()
    et3 = time.time()
    print(f"Different h tests completed in {et3 - et2:.2f} seconds.")
    test_output_fssh_gauge_fixing()
    et4 = time.time()
    print(f"FSSH gauge fixing tests completed in {et4 - et3:.2f} seconds.")
    test_output_fssh_deterministic()
    et5 = time.time()
    print(f"FSSH deterministic tests completed in {et5 - et4:.2f} seconds.")
    test_dh_qc_dzc_finite_differences()
    et6 = time.time()
    print(f"dh_qc_dzc finite difference tests completed in {et6 - et5:.2f} seconds.")
    test_dh_c_dzc_finite_differences()
    et7 = time.time()
    print(f"dh_c_dzc finite difference tests completed in {et7 - et6:.2f} seconds.")
    print(f"All tests completed in {et7 - st:.2f} seconds.")
