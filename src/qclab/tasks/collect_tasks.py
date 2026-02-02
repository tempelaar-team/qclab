"""
This module contains tasks that are used to collect data from the state or
parameters objects into the output dictionary of the State object.
"""

from qclab import Simulation


def collect_t(
    sim: Simulation,
    state: dict,
    parameters: dict,
    t_name: str = "t",
    t_output_name: str = "t",
):
    """
    Collects the time from the State object and stores it in the output
    dictionary.

    Optional Keyword Arguments
    --------------------------
    t_name:
        Name of the time in the State object.
    t_output_name:
        Name of the time in the output dictionary.

    Reads
    -----
    state[t_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Writes
    ------
    state["output_dict"][t_output_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    """
    state["output_dict"][t_output_name] = state[t_name]
    return state, parameters


def collect_dm_db(
    sim: Simulation,
    state: dict,
    parameters: dict,
    dm_db_name: str = "dm_db",
    dm_db_output_name: str = "dm_db",
):
    """
    Collects the diabatic density matrix from the State object and stores it
    in the output dictionary.

    Optional Keyword Arguments
    --------------------------
    dm_db_name:
        Name of the diabatic density matrix in the State object.
    dm_db_output_name:
        Name of the diabatic density matrix in the output dictionary.

    Reads
    -----
    state[dm_db_name]: ndarray of shape (B, N, N), dtype=complex128
        Density matrix in the diabatic basis.

    Writes
    ------
    state["output_dict"][dm_db_output_name]: ndarray of shape (B, N, N), dtype=complex128
        Density matrix in the diabatic basis.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    state["output_dict"][dm_db_output_name] = state[dm_db_name]
    return state, parameters


def collect_classical_energy(
    sim: Simulation,
    state: dict,
    parameters: dict,
    classical_energy_name: str = "classical_energy",
    classical_energy_output_name: str = "classical_energy",
):
    """
    Collects the classical energy from the State object and stores it in the
    output dictionary.

    Optional Keyword Arguments
    --------------------------
    classical_energy_name:
        Name of the classical energy in the State object.
    classical_energy_output_name:
        Name of the classical energy in the output dictionary.

    Reads
    -----
    state[classical_energy_name]: ndarray of shape (B,), dtype=float64
        Classical energy of each trajectory.

    Writes
    ------
    state["output_dict"][classical_energy_output_name] : ndarray of shape (B,), dtype=float64
        Classical energy of each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    state["output_dict"][classical_energy_output_name] = state[classical_energy_name]
    return state, parameters


def collect_quantum_energy(
    sim: Simulation,
    state: dict,
    parameters: dict,
    quantum_energy_name: str = "quantum_energy",
    quantum_energy_output_name: str = "quantum_energy",
):
    """
    Collects the quantum energy from the State object and stores it in the
    output dictionary.

    Optional Keyword Arguments
    --------------------------
    quantum_energy_name:
        Name of the quantum energy in the State object.
    quantum_energy_output_name:
        Name of the quantum energy in the output dictionary.

    Reads
    -----
    state[quantum_energy_name]: ndarray of shape (B,) with dtype=float64
        Quantum energy in each trajectory.

    Writes
    ------
    state["output_dict"][quantum_energy_output_name]: ndarray of shape (B,) with dtype=float64
        Quantum energy in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    """
    state["output_dict"][quantum_energy_output_name] = state[quantum_energy_name]
    return state, parameters
