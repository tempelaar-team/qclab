"""Tasks connecting ordinary dynamics recipes to trajectory-local diabatization."""

import numpy as np

from qclab import Simulation, functions
from qclab.tasks.collect_tasks import collect_dm_db
from qclab.tasks.update_tasks import update_dh_qc_dzc


def initialize_local_diabatization(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    wf_db_name: str = "wf_db",
    seed_name: str = "seed",
    parameters_seed_name: str = "seed",
    local_diabatic_property_name: str = "local_diabatic_property",
):
    """Initialize the transported electronic frame after nuclear initialization.

    This task is a no-op for models without a ``local_diabatization`` ingredient.
    The initial transported basis is the adiabatic basis at each trajectory's
    own initial geometry, so ``wf_db`` contains initial adiabatic coefficients.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Initial nuclear coordinates.
    state[wf_db_name]: ndarray of shape (B, N)
        Electronic coefficients in the initial local basis.
    state[seed_name]: ndarray of shape (B,), optional
        Trajectory seeds, copied for electronic-structure error reporting.

    Writes
    ------
    state[local_diabatic_property_name]: sequence of dict
        Initial frame packets returned by the ingredient.
    """
    local_diabatization, has_local_diabatization = sim.model.get("local_diabatization")
    if not has_local_diabatization:
        return state, parameters
    if sim.algorithm.settings.get("use_gauge_field_force"):
        raise ValueError(
            "Local diabatization currently supports real electronic states with "
            "use_gauge_field_force=False."
        )
    if wf_db_name not in state:
        raise ValueError(
            "Local diabatization with the ordinary dynamics algorithms requires "
            "initial_state['%s'] as a complex NumPy array of initial adiabatic "
            "coefficients." % wf_db_name
        )
    wavefunction = np.asarray(state[wf_db_name], dtype=complex)
    expected_shape = (sim.settings.batch_size, sim.model.constants.num_quantum_states)
    if wavefunction.shape != expected_shape or not np.all(np.isfinite(wavefunction)):
        raise ValueError(
            "Local-diabatization %s must be finite and have batched shape %s."
            % (wf_db_name, expected_shape)
        )
    if not np.allclose(
        np.sum(np.abs(wavefunction) ** 2, axis=-1), 1.0, rtol=0, atol=1e-7
    ):
        raise ValueError(
            "Local-diabatization %s must be normalized separately in each trajectory."
            % wf_db_name
        )
    state[wf_db_name] = wavefunction.copy()
    if seed_name in state:
        parameters[parameters_seed_name] = state[seed_name].copy()
    state[local_diabatic_property_name] = local_diabatization(
        sim.model, parameters, z=state[z_name], action="initialize"
    )
    return state, parameters


def begin_local_diabatic_step(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    local_diabatic_property_name: str = "local_diabatic_property",
):
    """Evaluate the accepted start geometry without advancing its frozen anchor.

    The ingredient retains the accepted reference throughout all subsequent RK
    trial evaluations. Ordinary models are unaffected.
    """
    local_diabatization, has_local_diabatization = sim.model.get("local_diabatization")
    if has_local_diabatization:
        state[local_diabatic_property_name] = local_diabatization(
            sim.model, parameters, z=state[z_name], action="evaluate"
        )
    return state, parameters


def update_local_diabatic_endpoint(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    dh_qc_dzc_name: str = "dh_qc_dzc",
    local_diabatic_property_name: str = "local_diabatic_property",
):
    """Refresh the accepted endpoint gradient before FSSH hopping decisions.

    The ordinary RK recipe otherwise leaves its gradient at the final trial
    geometry. Hop rescaling needs the gradient at the accepted endpoint instead.
    """
    local_diabatization, has_local_diabatization = sim.model.get("local_diabatization")
    if has_local_diabatization:
        state, parameters = update_dh_qc_dzc(
            sim, state, parameters, z_name=z_name, dh_qc_dzc_name=dh_qc_dzc_name
        )
        state[local_diabatic_property_name] = local_diabatization(
            sim.model, parameters, z=state[z_name], action="evaluate"
        )
    return state, parameters


def commit_local_diabatic_step(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    local_diabatic_property_name: str = "local_diabatic_property",
):
    """Commit the accepted endpoint after all force and hopping requests.

    Updating the ingredient's anchor preserves the persistent computational
    coordinates. Wavefunctions and eigenvectors therefore require no rotation.
    """
    local_diabatization, has_local_diabatization = sim.model.get("local_diabatization")
    if has_local_diabatization:
        state[local_diabatic_property_name] = local_diabatization(
            sim.model, parameters, z=state[z_name], action="commit"
        )
    return state, parameters


def update_local_diabatic_density(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    dm_db_name: str = "dm_db",
    dm_adb_name: str = "dm_adb",
    local_diabatic_property_name: str = "local_diabatic_property",
):
    """Express an already computed local density in the adiabatic basis.

    Each trajectory has its own transported frame. Its raw local density cannot
    be averaged with other trajectories as a shared diabatic observable.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Current nuclear coordinates.
    state[dm_db_name]: ndarray of shape (B, N, N), dtype=complex128
        Density already formed by the mean-field or FSSH update task.

    Writes
    ------
    state[dm_adb_name]: ndarray of shape (B, N, N), dtype=complex128
        Density in instantaneous adiabatic states.
    """
    local_diabatization, has_local_diabatization = sim.model.get("local_diabatization")
    if has_local_diabatization:
        packets = local_diabatization(
            sim.model, parameters, z=state[z_name], action="evaluate"
        )
        basis = np.asarray([packet["adiabatic_to_diabatic"] for packet in packets])
        state[dm_adb_name] = functions.transform_mat(state[dm_db_name], basis)
        state[local_diabatic_property_name] = packets
    return state, parameters


def collect_local_diabatic_density(
    sim: Simulation,
    state: dict,
    parameters: dict,
    dm_db_name: str = "dm_db",
    dm_db_output_name: str = "dm_db",
    dm_adb_name: str = "dm_adb",
    dm_adb_output_name: str = "dm_adb",
):
    """Copy adiabatic local-frame densities or the ordinary diabatic density.

    All matrix transformations are performed by the preceding update task.
    """
    _, has_local_diabatization = sim.model.get("local_diabatization")
    if has_local_diabatization:
        return collect_dm_db(
            sim, state, parameters,
            dm_db_name=dm_adb_name, dm_db_output_name=dm_adb_output_name,
        )
    return collect_dm_db(
        sim, state, parameters,
        dm_db_name=dm_db_name, dm_db_output_name=dm_db_output_name,
    )
