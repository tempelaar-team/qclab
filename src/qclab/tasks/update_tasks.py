"""
This module contains tasks that update the state and parameters objects
during propagation.
"""

import logging
import numpy as np
import copy
from qclab import functions, Simulation
import qclab.numerical_constants as numerical_constants

logger = logging.getLogger(__name__)


def update_t(sim: Simulation, state: dict, parameters: dict, t_name: str = "t"):
    """
    Updates the time in the State object with the time index in each trajectory
    multiplied by the update timestep.

    Optional Keyword Arguments
    --------------------------
    t_name:
        Name of the time variable in the State object.

    Writes
    ------
    state[t_name]: ndarray of shape (B,) dtype=float64
        Time variable in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size

    """
    batch_size = sim.settings.batch_size
    state[t_name] = np.broadcast_to(sim.t_ind * sim.settings.dt_update, (batch_size,))
    return state, parameters


def update_dh_c_dzc_finite_differences(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    dh_c_dzc_name: str = "dh_c_dzc",
):
    """
    Updates the gradient of the classical Hamiltonian using finite differences.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of the classical coordinates in the State object.
    dh_c_dzc_name:
        Name under which to store the finite-difference gradient in the State object.

    Constants and Settings
    ----------------------
    sim.model.constants.dh_c_dzc_finite_difference_delta: float, default: numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    Ingredients
    -----------
    h_c:
        Classical Hamiltonian.

    Reads
    -----
    state[z_name] : ndarray, (batch_size, num_classical_coordinates), complex
        Classical coordinates.

    Writes
    ------
    state[dh_c_dzc_name] : ndarray, (batch_size, num_classical_coordinates), complex
        Gradient of the classical Hamiltonian.

    """
    z = state[z_name]
    delta_z = sim.model.constants.get(
        "dh_c_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    batch_size = len(z)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_c, _ = sim.model.get("h_c")
    # Calculate it at the original z coordinate.
    h_c_0 = h_c(sim.model, parameters, z=z)
    # Calculate h_c at the offset coordinates.
    h_c_all = h_c(sim.model, parameters, z=z_offset_all).reshape(
        batch_size, 2 * num_classical_coordinates
    )
    # Split real/imag blocks of the offset h_c.
    h_c_re = h_c_all[:, :num_classical_coordinates]
    h_c_im = h_c_all[:, num_classical_coordinates:]
    # Calculate finite-difference derivatives.
    h_c_0_new_ind = h_c_0[:, None]
    dh_c_dzc_re = (h_c_re - h_c_0_new_ind) / delta_z
    dh_c_dzc_im = (h_c_im - h_c_0_new_ind) / delta_z
    dh_c_dzc = 0.5 * (dh_c_dzc_re + 1j * dh_c_dzc_im)
    state[dh_c_dzc_name] = dh_c_dzc
    return state, parameters


def update_classical_force(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    classical_force_name: str = "classical_force",
):
    """
    Updates the gradient of the classical Hamiltonian w.r.t. the conjugate classical
    coordinate.

    Optional Keyword Arguments
    --------------------------
    z_name : str, default:
        Name of the classical coordinates in the State object.
    classical_force_name:
        Name under which to store the classical force in the State object.

    Ingredients
    -----------
    dh_c_dzc:
        Gradient of the classical Hamiltonian with respect to the conjugate classical
        coordinate.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinate.

    Writes
    ------
    state[classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the classical Hamiltonian.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state[classical_force_name] = dh_c_dzc(sim.model, parameters, z=z)
    else:
        if sim.settings.debug:
            logger.info("dh_c_dzc not found; using finite differences.")
        state, parameters = update_dh_c_dzc_finite_differences(
            sim, state, parameters, dh_c_dzc_name=classical_force_name, z_name=z_name
        )
    return state, parameters


def update_dh_qc_dzc_finite_differences(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    dh_qc_dzc_name: str = "dh_qc_dzc",
):
    """
    Updates the gradient of the quantum-classical Hamiltonian using finite
    differences.

    .. rubric:: Required Constants
    dh_qc_dzc_finite_difference_delta : float, default : numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    dh_qc_dzc_name:
        Name under which to store the gradient of the quantum-classical Hamiltonian in the State object.

    Constants and Settings
    sim.model.constants.dh_qc_dzc_finite_difference_delta: float, default: numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    Ingredients
    -----------
    h_qc:
        Quantum-classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Writes
    ------
    state[dh_qc_dzc_name]: tuple
        Gradient of the quantum-classical Hamiltonian in sparse matrix format.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    batch_size = len(z)
    delta_z = sim.model.constants.get(
        "dh_qc_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_qc, _ = sim.model.get("h_qc")
    # Calculate it at the original z coordinate.
    h_qc_0 = h_qc(sim.model, parameters, z=z)
    # Calculate h_qc at the offset coordinates.
    h_qc_all = h_qc(sim.model, parameters, z=z_offset_all).reshape(
        batch_size,
        2 * num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    # Split real/imag blocks of the offset h_qc.
    h_qc_re = h_qc_all[:, :num_classical_coordinates, :, :]
    h_qc_im = h_qc_all[:, num_classical_coordinates:, :, :]
    # Calculate finite-difference derivatives.
    h_qc_0_new_ind = h_qc_0[:, None, :, :]
    dh_qc_dzc_re = (h_qc_re - h_qc_0_new_ind) / delta_z
    dh_qc_dzc_im = (h_qc_im - h_qc_0_new_ind) / delta_z
    dh_qc_dzc = 0.5 * (dh_qc_dzc_re + 1j * dh_qc_dzc_im)
    # Get sparse representation of dh_qc_dzc.
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    # Update it in the State object.
    state[dh_qc_dzc_name] = (inds, mels, shape)
    return state, parameters


def update_dh_qc_dzc(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    dh_qc_dzc_name: str = "dh_qc_dzc",
):
    """
    Updates the gradient of the quantum-classical Hamiltonian w.r.t. the conjugate
    classical coordinate.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in State object.
    dh_qc_dzc_name:
        Name under which to store the gradient of the quantum-classical Hamiltonian in the State object.

    Ingredients
    -----------
    dh_qc_dzc:
        Gradient of the quantum-classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Writes
    ------
    state[dh_qc_dzc_name]: tuple
        Gradient of the quantum-classical Hamiltonian in spare format.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    if not (dh_qc_dzc_name in state) or sim.model.update_dh_qc_dzc:
        # If dh_qc_dzc has not been calculated yet, or if the
        # model requires it to be updated, calculate it.
        dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
        if has_dh_qc_dzc:
            state[dh_qc_dzc_name] = dh_qc_dzc(sim.model, parameters, z=z)
        else:
            if sim.settings.debug:
                logger.info("dh_qc_dzc not found; using finite differences.")
            state, parameters = update_dh_qc_dzc_finite_differences(
                sim, state, parameters, z_name=z_name, dh_qc_dzc_name=dh_qc_dzc_name
            )
    # If dh_qc_dzc has already been calculated and does not need to be updated,
    # return the existing parameters and State objects.
    return state, parameters


def update_derivative_coupling_dzc(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    derivative_coupling_dzc_name: str = "derivative_coupling_dzc",
):
    """
    Updates the derivative coupling matrix.

    .. math::

        d^{i}_{\\alpha,\\beta} = \\langle\\alpha(z)\\vert\\partial_{z_{i}^{*}}\\vert\\beta(z)\\rangle

    Raises

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of the classical coordinates in the State object.

    Constants and Settings
    ----------------------
    sim.model.update_dh_qc_dzc: Bool, default: False
        Model flag indicating if the quantum-classical Hamiltonian is to be updated at each timestep.
    sim.algorithm.settings.use_gauge_field_force: Bool, default: False
        Boolean indicating if a gauge field force is to be added to the quantum-classical force.

    Ingredients
    -----------
    derivative_coupling_dzc:
        Derivative coupling tensor.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Writes
    ------
    state[derivative_coupling_dzc_name]: ndarray of shape (B, C, N, N), dtype=complex128
        Quantum-classical force.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    derivative_coupling_dzc_func, has_derivative_coupling_dzc = sim.model.get(
        "derivative_coupling_dzc"
    )
    if has_derivative_coupling_dzc:
        state[derivative_coupling_dzc_name] = derivative_coupling_dzc_func(
            sim.model, parameters, z=z
        )
    else:
        raise AttributeError(
            "Model class must have a derivative_coupling_dzc ingredint."
        )
    return state, parameters


def update_quantum_classical_force(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    wf_db_name: str = "wf_db",
    dh_qc_dzc_name: str = "dh_qc_dzc",
    quantum_classical_force_name: str = "quantum_classical_force",
    state_ind_name: str = "act_surf_ind",
    wf_changed: bool = True,
    h_q_tot_name: str = "h_q_tot",
    derivative_coupling_dzc_name: str = "derivative_coupling_dzc",
    update_dh_qc_dzc_flag: bool = True,
):
    """
    Updates the quantum-classical force w.r.t. the wavefunction defined by ``wf_db``.

    If the model has a ``gauge_field_force`` ingredient, this term will be added
    to the quantum-classical force.

    If the model has a ``derivative_coupling_dzc`` ingredient, this contribution will
    be added to the quantum-classical force.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    wf_db_name:
        Name of the wavefunction (in the diabatic basis) in the State object.
    dh_qc_dzc_name:
        Name of the gradient of the quantum-classical Hamiltonian in the State object.
    quantum_classical_force_name:
        Name under which to store the quantum-classical force in the State object.
    state_ind_name:
        Name in the State object of the state index for which to obtain the gauge field force.
        Required if ``algorithm.settings.use_gauge_field_force`` is ``True``.
    wf_changed:
        If ``True``, the wavefunction has changed since the last time the force were calculated.
    h_q_tot_name:
        Name under which to store the total Hamiltonian in the State object.
    derivative_coupling_dzc_name:
        Name of the derivative coupling tensor in the State object.

    Constants and Settings
    ----------------------
    sim.model.update_dh_qc_dzc: Bool, default: False
        Model flag indicating if the quantum-classical Hamiltonian is to be updated at each timestep.
    sim.algorithm.settings.use_gauge_field_force: Bool, default: False
        Boolean indicating if a gauge field force is to be added to the quantum-classical force.

    Ingredients
    -----------
    derivative_coupling_dzc:
        Derivative coupling tensor.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[dh_qc_dzc_name]: tuple
        Gradient of the quantum-classical Hamiltonian in sparse format.
    state[h_qc_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian.
    state[derivative_coupling_dzc_name]: ndarray of shape (B, C, N, N), dtype=complex128 (optional)
        The derivative coupling tensor.

    Writes
    ------
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Quantum-classical force.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    wf_db = state[wf_db_name]
    if update_dh_qc_dzc_flag:
        # Update the gradient of h_qc.
        state, parameters = update_dh_qc_dzc(
            sim, state, parameters, z_name=z_name, dh_qc_dzc_name=dh_qc_dzc_name
        )
    # Calculate the expectation value w.r.t. the wavefunction.
    # If not(wf_changed) and sim.model.update_dh_qc_dzc then recalculate.
    # If wf_changed then recalculate.
    # If quantum_classical_force_name not in state then recalculate.
    if (
        not (quantum_classical_force_name in state)
        or wf_changed
        or (not (wf_changed) and sim.model.update_dh_qc_dzc)
    ):
        if not (quantum_classical_force_name in state):
            state[quantum_classical_force_name] = np.zeros_like(z)
        state[quantum_classical_force_name] = functions.calc_sparse_inner_product(
            *state[dh_qc_dzc_name],
            wf_db.conj(),
            wf_db,
            out=state[quantum_classical_force_name].reshape(-1),
        ).reshape(np.shape(z))
        # Add force arising from derivative coupling if it is defined in the model.
        if derivative_coupling_dzc_name in state:
            derivative_coupling_dzc = state[derivative_coupling_dzc_name]
            diag_e = np.einsum("tii->ti", state[h_q_tot_name])
            diff_e = diag_e[:, None, :] - diag_e[:, :, None]  # e_{ji}
            derivative_coupling_force = np.einsum(
                "ti,tcij,tj->tc",
                np.conj(wf_db),
                diff_e[:, np.newaxis, :, :] * derivative_coupling_dzc,
                wf_db,
                optimize="greedy",
            )
            state[quantum_classical_force_name] += derivative_coupling_force
    if sim.algorithm.settings.get("use_gauge_field_force"):
        state, parameters = add_gauge_field_force(
            sim, state, parameters, z=z, state_ind_name=state_ind_name
        )

    return state, parameters


def add_gauge_field_force(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    adb_state_ind_name: str = "act_surf_ind",
    quantum_classical_force_name: str = "quantum_classical_force",
):
    """
    Adds the quantum-classical force with the gauge field force if the model has a
    ``gauge_field_force`` ingredient.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in State object.
    adb_state_ind_name:
        Name of the adiabatic state index for which to obtain the gauge field force.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.

    Ingredients
    -----------
    gauge_field_force:
        Forse originating from the gauge field.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[adb_state_ind_name]: ndarray of shape (B,), dtype=int
        Index of the adiabatic state from which the gauge field originates.

    Writes
    ------
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Quantum-classical force.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    adb_state_ind = state[adb_state_ind_name]
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force:
        gauge_field_force_val = gauge_field_force(
            sim.model, parameters, z=z, state_ind=adb_state_ind
        )
        state[quantum_classical_force_name] += gauge_field_force_val
    else:
        if sim.settings.debug:
            logger.warning("gauge_field_force not found; skipping.")
    return state, parameters


def diagonalize_matrix(
    sim: Simulation,
    state: dict,
    parameters: dict,
    matrix_name: str,
    eigvals_name: str,
    eigvecs_name: str,
):
    """
    Diagonalizes a given matrix from the State object and stores the eigenvalues and
    eigenvectors in the State object.

    Optional Keyword Arguments
    --------------------------
    matrix_name:
        Name of the matrix to diagonalize in the State object.
    eigvals_name:
        Name of the eigenvalues in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.

    Reads
    -----
    state[matrix_name]: ndarray
        A Hermitian matrix.

    Writes
    ------
    state[eigvals_name]: ndarray
        Eigenvalues.
    state[eigvecs_name]: ndarray
        Eigenvectors.
    """
    matrix = state[matrix_name]
    state[eigvals_name], state[eigvecs_name] = np.linalg.eigh(matrix)
    return state, parameters


def update_eigvecs_gauge(
    sim: Simulation,
    state: dict,
    parameters: dict,
    eigvals_name: str = "eigvals",
    eigvecs_name: str = "eigvecs",
    eigvecs_previous_name: str = "eigvecs_previous",
    output_eigvecs_name: str = "eigvecs_name",
    z_name: str = "z",
    gauge_fixing: str = None,
    dh_qc_dzc_name: str = "dh_qc_dzc",
):
    """
    Updates the gauge of the eigenvectors as specified by the gauge_fixing parameter.


    if gauge_fixing == "sign_overlap":
        The sign of the eigenvector is changed so the real part of its
        overlap with the previous eigenvector is positive.

    if gauge_fixing == "phase_overlap":
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and used to maximize the real
        part of the overlap. The sign is then changed so the real
        part of the overlap is positive.

    if gauge_fixing == "phase_der_couple":
        The phase of the eigenvector is determined by calculating the
        derivative couplings and changed so that all the derivative
        couplings are real-valued.

    Optional Keyword Arguments
    --------------------------
    eigvals_name:
        Name of the eigenvalues in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.
    eigvecs_previous_name:
        Name of the previous eigenvectors in the State object.
    output_eigvecs_name:
        Name of the output gauge-fixed eigenvectors in the State object.
    z_name:
        Name of classical coordinates in the State object.
    gauge_fixing: default: ``sim.algorithm.settings.gauge_fixing``
        Gauge-fixing method to use.
    dh_qc_dzc_name:
        Name of the gradient of the quantum-classical Hamiltonian in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.gauge_fixing: str
        Gauge fixing method to use.

    Reads
    -----
    state[eigvals_name]: ndarray of shape (B, N), dtype=float64
        Eigenvalues of the total quantum Hamiltonian.
    state[eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.
    state[eigvecs_previous_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian at the prior timestep.
    state[dh_qc_dzc]: tuple

    Writes
    ------
    state[output_eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Gauge-fixed eigenvectors of the total quantum Hamiltonian.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    eigvals = state[eigvals_name]
    eigvecs = state[eigvecs_name]
    eigvecs_previous = state[eigvecs_previous_name]
    if gauge_fixing is None:
        gauge_fixing = sim.algorithm.settings.gauge_fixing
    gauge_fixing_numerical_values = {
        "sign_overlap": 0,
        "phase_overlap": 1,
        "phase_der_couple": 2,
    }
    try:
        gauge_fixing_value = gauge_fixing_numerical_values[gauge_fixing]
    except KeyError:
        logger.critical("Invalid gauge_fixing value: %s", gauge_fixing)
        raise ValueError(f"Invalid gauge_fixing value: {gauge_fixing}")
    if gauge_fixing_value == 1:
        # Maximize the real part of the overlap (guaranteed to be positive).
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        phase = np.exp(-1j * np.angle(overlap))
        eigvecs *= phase[:, None, :]
    if gauge_fixing_value == 2:
        # Make the derivative couplings real-valued (but not necessarily positive).
        state, parameters = update_dh_qc_dzc(sim, state, parameters, z_name=z_name)
        der_couple_dq_phase, _ = functions.analytic_der_couple_phase(
            sim, state[dh_qc_dzc_name], eigvals, eigvecs
        )
        eigvecs *= np.conj(der_couple_dq_phase)[:, None, :]
    if gauge_fixing_value == 0 or gauge_fixing_value == 2:
        # Make the real part positive based on the sign of the real part of the overlap.
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        signs = np.sign(np.real(overlap))
        if sim.settings.debug:
            if np.any(np.abs(signs) < numerical_constants.SMALL):
                logger.error(
                    "Zero overlap encountered when fixing gauge.\n"
                    "This may indicate a trivial crossing or degeneracy.\n"
                    "Normalization will be broken and results will be incorrect."
                )
        eigvecs *= signs[:, None, :]
    if gauge_fixing_value == 2 and sim.settings.debug:
        der_couple_dq_phase_new, der_couple_dp_phase_new = (
            functions.analytic_der_couple_phase(
                sim, state[dh_qc_dzc_name], eigvals, eigvecs
            )
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_dq_phase_new)) ** 2
                + np.abs(np.imag(der_couple_dp_phase_new)) ** 2
            )
            > numerical_constants.SMALL
        ):
            logger.error(
                "Phase error encountered when fixing gauge analytically. %s",
                np.sum(
                    np.abs(np.imag(der_couple_dq_phase_new)) ** 2
                    + np.abs(np.imag(der_couple_dp_phase_new)) ** 2
                ),
            )
    state[output_eigvecs_name] = eigvecs
    return state, parameters


def update_vector_basis(
    sim: Simulation,
    state: dict,
    parameters: dict,
    input_vec_name: str,
    basis_name: str,
    output_vec_name: str,
    adb_to_db: bool = False,
):
    """
    Transforms a vector to a new basis.

    Optional Keyword Arguments
    --------------------------
    input_vec_name:
        Name of the vector to transform in the State object.
    basis_name:
        Name of the basis to transform to in the State object.
        Assumed to be column vectors corresponding to adiabatic
        states.
    output_vec_name:
        Name of the output vector in the State object.
    adb_to_db:
        If True, transforms from adiabatic to diabatic. If False, transforms from
        adiabatic to diabatic.

    Reads
    -----
    state[input_vec_name]: ndarray
        Vector to be transformed
    state[basis_name]: ndarray
        Column vectors that form the new basis.

    Writes
    ------
    state[output_vec_name]: ndarray
        Vector expressed in the new basis.
    """
    input_vec = state[input_vec_name]
    basis = state[basis_name]
    state[output_vec_name] = functions.transform_vec(
        input_vec, basis, adb_to_db=adb_to_db
    )
    return state, parameters


def update_act_surf_wf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    act_surf_wf_name: str = "act_surf_wf",
    act_surf_ind_name: str = "act_surf_ind",
    eigvecs_name: str = "eigvecs",
):
    """
    Updates the wavefunction corresponding to the active surface.

    Optional Keyword Arguments
    --------------------------
    act_surf_wf_name:
        Name of the active surface wavefunction in the State object.
    act_surf_ind_name:
        Name of the active surface index in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.

    Reads
    -----
    state[eigvecs_name]: ndarray of shape (B, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.
    state[act_surf_ind_name]: ndarray of shape (B,), dtype=int
        Active surface index in each trajectory.

    Writes
    ------
    state[act_surf_wf_name] : ndarray of shape (B, N), dtype=complex128
        Wavefunction of the active surface in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    state[act_surf_wf_name] = state[eigvecs_name][
        np.arange(sim.settings.batch_size, dtype=int), :, state[act_surf_ind_name]
    ]
    return state, parameters


def update_wf_db_propagator(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_db_name: str = "wf_db",
    eigvals_name: str = "eigvals",
    eigvecs_name: str = "eigvecs",
):
    """
    Updates the diabatic wavefunction by calculating and applying the propagator.

    Optional Keyword Arguments
    --------------------------
    wf_db_name:
        Name of the diabatic wavefunction in the State object.
    eigvals_name:
        Name of the eigenvalues in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.

    Reads
    -----
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[eigvals_name]: ndarray of shape (B, N), dtype=float64
        Eigenvalues of the total quantum Hamiltonian.
    state[eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.

    Writes
    ------
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Propagated wavefunction coefficients in the diabatic basis.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    wf_db = state[wf_db_name]
    eigvals = state[eigvals_name]
    eigvecs = state[eigvecs_name]
    batch_size = sim.settings.batch_size
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate the propagator in the adiabatic basis.
    prop_adb = np.zeros(
        (batch_size, num_quantum_states, num_quantum_states), dtype=complex
    )
    idx = np.arange(num_quantum_states)
    prop_adb[:, idx, idx] = np.exp(-1j * eigvals * sim.settings.dt_update)
    # Transform propagator to the diabatic basis.
    prop_db = functions.transform_mat(prop_adb, eigvecs, adb_to_db=True)
    state[wf_db_name] = functions.batch_matvec(prop_db, wf_db)
    return state, parameters


def update_wf_db_rk4(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_db_name: str = "wf_db",
    h_q_tot_name: str = "h_q_tot",
):
    """
    Updates the wavefunction using the 4th-order Runge-Kutta method.

    Optional Keyword Arguments
    --------------------------
    wf_db_name:
        Name of the diabatic wavefunction in the State object.
    h_q_tot_name:
        Name of the quantum Hamiltonian in the State object.

    Reads
    -----
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian.

    Writes
    ------
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Updated wavefunction coefficients.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    dt_update = sim.settings.dt_update
    wf_db = state[wf_db_name]
    h_q_tot = state[h_q_tot_name]
    k1 = -1j * functions.batch_matvec(h_q_tot, wf_db)
    k2 = -1j * functions.batch_matvec(h_q_tot, (wf_db + 0.5 * dt_update * k1))
    k3 = -1j * functions.batch_matvec(h_q_tot, (wf_db + 0.5 * dt_update * k2))
    k4 = -1j * functions.batch_matvec(h_q_tot, (wf_db + dt_update * k3))
    wf_db += dt_update * 0.16666666666666666 * k1
    wf_db += dt_update * 0.3333333333333333 * k2
    wf_db += dt_update * 0.3333333333333333 * k3
    wf_db += dt_update * 0.16666666666666666 * k4
    return state, parameters


def update_hop_prob_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    act_surf_ind_name: str = "act_surf_ind",
    wf_adb_name: str = "wf_adb",
    eigvecs_name: str = "eigvecs",
    eigvecs_previous_name: str = "eigvecs_previous",
    hop_prob_name: str = "hop_prob",
    adb_connection_name: str = "adb_connection",
):
    """
    Calculates the hopping probabilities according to the FSSH algorithm.

    :math:`P_{a \\rightarrow b} = -2 \\Re \\left( (C_{b}/C_{a}) \\langle a(t) | b(t-dt) \\rangle \\right)`

    Optional Keyword Arguments
    --------------------------
    act_surf_ind_name:
        Name of the active surface index in the State object.
    wf_adb_name:
        Name of the adiabatic wavefunction in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.
    eigvecs_previous_name:
        Name of the previous eigenvectors in the State object.
    hop_prob_name:
        Name under which to store the hopping probabilities in the State object.
    adb_connection_name:
        Name of the adiabatic connection in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.fssh_deterministic: Bool
        Boolean indicating if the FSSH simulation is deterministic.

    Reads
    -----
    state[act_surf_ind_name]: ndarray of shape (B,), dtype=int
        Active surface index in each trajectory.
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the adiabatic basis.
    state[eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.
    state[eigvecs_previous_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian at the previous timestep.
    state[adb_connection_name]: ndarray of shape (B, C, N, N), dtype=complex128

    Writes
    ------
    state[hop_prob_name]: ndarray of shape (B, N), dtype=float64
        Hopping probabilities between the active surface and all other surfaces.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    act_surf_ind = state[act_surf_ind_name]
    wf_adb = state[wf_adb_name]
    eigvecs = state[eigvecs_name]
    eigvecs_previous = state[eigvecs_previous_name]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    # Check if any of the coefficients on the active surface are zero.
    if sim.settings.debug:
        if np.any(
            np.abs(wf_adb[np.arange(num_trajs * num_branches), act_surf_ind]) == 0.0
        ):
            logger.warning(
                "Zero coefficient on active surface encountered when calculating hopping probabilities."
            )
    if adb_connection_name in state.keys():
        # Calculates < act_surf(t) | b(t-dt)> = - A_{act_surf, b} * dt
        adb_connection = state[adb_connection_name]
        nac_prod = (
            -adb_connection[
                np.arange(num_trajs * num_branches, dtype=int), act_surf_ind, :
            ]
            * sim.settings.dt_update
        )

    else:
        # Calculates < act_surf(t) | b(t-dt) > = -\dot{q} \cdot d_{act_surf,b} dt
        nac_prod = functions.batch_matvec(
            np.swapaxes(eigvecs_previous, -1, -2),
            np.conj(
                eigvecs[np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind]
            ),
        )
    # Calculates -2 Re( (C_b / C_act_surf) < act_surf(t) | b(t-dt) > )
    hop_prob = -2.0 * np.real(
        nac_prod
        * wf_adb
        / wf_adb[np.arange(num_trajs * num_branches), act_surf_ind][:, np.newaxis]
    )
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind] = 0.0
    # Check for singular values.
    if sim.settings.debug:
        if np.any(np.isnan(hop_prob)):
            logger.warning("Singluar value encountered in hopping probabilities.")
    state[hop_prob_name] = hop_prob
    return state, parameters


def update_hop_inds_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    act_surf_ind_name: str = "act_surf_ind",
    hop_prob_name: str = "hop_prob",
    hop_prob_rand_vals_name: str = "hop_prob_rand_vals",
    hop_ind_name: str = "hop_ind",
    hop_dest_name: str = "hop_dest",
    hop_bool_name: str = "hop_bool",
    hop_pairs_name: str = "hop_pairs",
):
    """
    Updates indices of trajectories that hop according to their probabilities (but may later be frustrated) and their destination state indices.

    Optional Keyword Arguments
    --------------------------
    act_surf_ind_name:
        Name of the active surface in the State object.
    hop_prob_name:
        Name of the hopping probabilities in the State object.
    hop_prob_rand_vals_name:
        Name of the random values for hopping probabilities in the State object.
    hop_ind_name:
        Name under which to store the indices of the hopping trajectories in the State object.
    hop_dest_name:
        Name under which to store the destination indices of the hopping trajectories in the State object.
    hop_pairs_name:
        Name under which to store the pairs of initial and final indices of hopping trajectories in the
        State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.fssh_deterministic: Bool
        Boolean indicating if the FSSH simulation is deterministic.

    Reads
    -----
    state[act_surf_ind_name]: ndarray of shape (B,) dtype=int
        Active surface indices.
    state[hop_prob_name]: ndarray of shape (B, N), dtype=float64
        Hopping probabilities between the active surface and all other surfaces.
    state[hop_prob_rand_vals_name]: ndarray of shape (B//b, t), dtype=float64
        Random numbers for hop decisions.

    Writes
    ------
    state[hop_ind_name]: ndarray of shape (H,), dtype=int
        Indices of trajectories that hop.
    state[hop_dest_name]: ndarray of shape (H,), dtype=int
        Destination surface for each hop.
    state[hop_bool_name]: ndarray of shape (B,), dtype=bool
        Boolean indicating if the trajectory hops.
    state[hop_paris_name]: ndarray of shape (B, 2), dtype=int
        Indices of initial/final states of hops. Zero for trajectories that don't hop.

    Notes
    -----
    * H = The number of trajectories that hop.
    * B = sim.settings.batch_size
    * b = sim.model.constants.num_quantum_states if fssh_deterministic == True, b = 1 otherwise.
    * t = The number of update timesteps.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    hop_prob = state[hop_prob_name]
    # Create a copy of hop_prob and set negative values to 0.
    hop_prob_positive = np.copy(hop_prob)
    hop_prob_positive[np.where(hop_prob < 0)] *= 0
    rand = state[hop_prob_rand_vals_name][:, sim.t_ind]
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob_positive, nan=0, posinf=100e100, neginf=0, copy=False),
        axis=1,
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    hop_bool = (
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )
    hop_ind = np.where(hop_bool == True)[0]
    hop_dest = np.argmax(
        (cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1
    )[hop_ind]
    state[hop_ind_name] = hop_ind
    state[hop_dest_name] = hop_dest
    state[hop_bool_name] = hop_bool
    hop_pairs = np.zeros((sim.settings.batch_size, 2), dtype=int)
    hop_pairs[hop_ind, 0] = state[act_surf_ind_name][hop_ind]
    hop_pairs[hop_ind, 1] = hop_dest
    state[hop_pairs_name] = hop_pairs
    return state, parameters


def update_z_shift_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_traj_name: str = "z_traj",
    resc_dir_z_traj_name: str = "resc_dir_z_traj",
    eigval_diff_traj_name: str = "eigval_diff_traj",
    hop_successful_traj_name: str = "hop_successful_traj",
    z_shift_traj_name: str = "z_shift_traj",
):
    """
    Determines if a hop occurs and calculates the shift in the classical coordinate
    at the single trajectory level.

    Optional Keyword Arguments
    --------------------------
    z_traj_name:
        Name of the classical coordinates for this trajectory in the State object.
    resc_dir_z_traj_name:
        Name of the rescaling direction for this trajectory in the State object.
    eigval_diff_traj_name:
        Name of the difference in eigenvalues between the initial and final states
        ``(e_final - e_initial)`` for this trajectory in the State object.
    hop_successful_traj_name:
        Name under which to store whether the hop was successful for this trajectory in the State object.
    z_shift_traj_name:
        Name under which to store the shift in classical coordinates for this trajectory in the State object.

    Ingredients
    -----------
    hop: optional, default: ``functions.numerical_fssh_hop``
        Hopping ingredient that determines the energy conservation criterion for a given classical Hamiltonian.

    Reads
    -----
    state[z_traj_name]: ndarray of shape (C,), dtype=complex128
        Complex-valued classical coordinates in a single trajectory.

    Writes
    ------
    state[hop_successful_traj_name]: Bool
        Boolean value indicating if the hop was successful.
    state[z_shift_traj_name]: ndarray of shape (C,), dtype=complex128
        Shift of the classical coordinate required to conserve energy.

    Notes
    -----
    * C = sim.settings.batch_size
    """
    z = state[z_traj_name]
    resc_dir_z = state[resc_dir_z_traj_name]
    eigval_diff = state[eigval_diff_traj_name]
    hop, has_hop = sim.model.get("hop")
    if has_hop:
        z_shift, hopped = hop(
            sim.model,
            parameters,
            z=z,
            resc_dir_z=resc_dir_z,
            eigval_diff=eigval_diff,
        )
    else:
        z_shift, hopped = functions.numerical_fssh_hop(
            sim.model,
            parameters,
            z=z,
            resc_dir_z=resc_dir_z,
            eigval_diff=eigval_diff,
        )
    state[hop_successful_traj_name] = hopped
    state[z_shift_traj_name] = z_shift
    return state, parameters


def update_hop_vals_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_shift_name: str = "z_shift",
    hop_successful_name: str = "hop_successful",
    hop_ind_name: str = "hop_ind",
    hop_dest_name: str = "hop_dest",
    eigvals_name: str = "eigvals",
    eigvecs_name: str = "eigvecs",
    z_name: str = "z",
    act_surf_ind_name: str = "act_surf_ind",
    dh_qc_dzc_name: str = "dh_qc_dzc",
    z_traj_name: str = "z_traj",
    resc_dir_z_traj_name: str = "resc_dir_z_traj",
    eigval_diff_traj_name: str = "eigval_diff_traj",
    hop_successful_traj_name: str = "hop_successful_traj",
    z_shift_traj_name: str = "z_shift_traj",
    derivative_coupling_dzc_name: str = None,
):
    """
    Updates trajectory hopping information for FSSH.

    Executes the hopping function for the hopping trajectories and stores the rescaled
    coordinates in ``state.z_rescaled`` and a Boolean registering if the hop was
    successful in ``state.hop_successful``.

    If the model has a ``rescaling_direction_fssh`` ingredient, it will be used to
    determine the direction in which to rescale the coordinates. Otherwise, the
    direction will be calculated with ``functions.calc_resc_dir_z_fssh``.

    .. rubric:: Required Constants
    None

    Optional Keyword Arguments
    --------------------------
    z_shift_name:
        Name under which to store the shift in coordinates for each hopping trajectory in the State object.
    hop_successful_name:
        Name under which to store flags indicating if each hop was successful in the State object.
    hop_ind_name:
        Name of the indices of the trajectories that are attempting to hop in the State object.
    hop_dest_name:
        Name of the destination states of the trajectories that are attempting to hop in the State object.
    eigvals_name:
        Name of the eigenvalues in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.
    z_name:
        Name of classical coordinates in the State object.
    act_surf_ind_name:
        Name of the active surface index in the State object.
    dh_qc_dzc_name :
        Name of the gradient of the quantum-classical Hamiltonian in the State object.
    z_traj_name:
        Name of the classical coordinates for the intermediate hopping trajectory in the State object.
    resc_dir_z_traj_name:
        Name of the rescaling direction for the intermediate hopping trajectory in the State object.
    eigval_diff_traj_name:
        Name of the difference in eigenvalues between the initial and final states
        ``(e_final - e_initial)`` for the intermediate hopping trajectory in the State object.
    hop_successful_traj_name:
        Name under which to store whether the hop was successful for the intermediate hopping trajectory in the State object.
    z_shift_traj_name:
        Name under which to store the shift in classical coordinates for the intermediate hopping trajectory in the State object.
    derivative_coupling_dzc_name:
        Name of the derivative coupling in the State object.

    Ingredients
    -----------
    rescaling_direction_fssh: optional, default: derivative coupling
        Rescaling direction for FSSH.

    Reads
    -----
    state[hop_ind_name]: ndarray of shape (h,), dtype=int
        Trajectory indices of trajectories that could hop.
    state[hop_dest_name]: ndarray of shape (h,), dtype=int
        Destination states of trajectories that hop.
    state[eigvals_name]: ndarray of shape (B, N), dtype=float64
        Eigenvalues of the total quantum Hamiltonian.
    state[eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinate.
    state[act_surf_ind_name]: ndarray of shape (B,), dtype=int
        Active surface index in each trajectory.
    state[derivative_couplign_dzc_name]: ndarray of shape (B, C, N, N), dtype=complex128
        Derivative coupling tensor.

    Writes
    ------
    state[hop_successful_name]: ndarray of shape (h,), dtype=Bool
        Boolean indicating if the hop was successful or frustrated in the trajectories that could hop.
    state[z_shift_name]: ndarray of shape (h,C), dtype=complex128
        Shift in classical coordinates in trajectories that could hop. 0 for frustrated hops, nonzero otherwise.

    Notes
    -----
    * h is the number of trajectories that could hop as determined by the hopping probabilities
        and random number, which are now being evaluated for energy conservation.
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    * N = sim.model.constants.num_quantum_states
    """
    hop_ind = state[hop_ind_name]
    hop_dest = state[hop_dest_name]
    state[z_shift_name] = np.zeros(
        (len(hop_ind), sim.model.constants.num_classical_coordinates), dtype=complex
    )
    state[hop_successful_name] = np.zeros(len(hop_ind), dtype=bool)
    eigvals = state[eigvals_name]
    eigvecs = state[eigvecs_name]
    z = state[z_name]
    act_surf_ind = state[act_surf_ind_name]
    state_hop_successful = state[hop_successful_name]
    state_z_shift = state[z_shift_name]
    hop_traj_ind = 0
    for traj_ind in hop_ind:
        final_state_ind = hop_dest[hop_traj_ind]
        init_state_ind = act_surf_ind[traj_ind]
        eigval_init_state = eigvals[traj_ind][init_state_ind]
        eigval_final_state = eigvals[traj_ind][final_state_ind]
        eigval_diff = eigval_final_state - eigval_init_state
        eigvec_init_state = eigvecs[traj_ind, :, init_state_ind]
        eigvec_final_state = eigvecs[traj_ind, :, final_state_ind]
        rescaling_direction_fssh, has_rescaling_direction_fssh = sim.model.get(
            "rescaling_direction_fssh"
        )
        if has_rescaling_direction_fssh:
            resc_dir_z = rescaling_direction_fssh(
                sim.model,
                parameters,
                z=z[traj_ind],
                init_state_ind=init_state_ind,
                final_state_ind=final_state_ind,
            )
        elif derivative_coupling_dzc_name in state.keys():
            derivative_coupling_dzc = state[derivative_coupling_dzc_name]
            resc_dir_z = derivative_coupling_dzc[
                traj_ind, :, init_state_ind, final_state_ind
            ]
        else:
            inds, mels, shape = state[dh_qc_dzc_name]
            dh_qc_dzc_traj_ind = inds[0] == traj_ind
            inds_traj = (
                inds[0][dh_qc_dzc_traj_ind],
                inds[1][dh_qc_dzc_traj_ind],
                inds[2][dh_qc_dzc_traj_ind],
                inds[3][dh_qc_dzc_traj_ind],
            )
            mels_traj = mels[dh_qc_dzc_traj_ind]
            shape_traj = (1, shape[1], shape[2], shape[3])
            dh_qc_dzc_traj = (inds_traj, mels_traj, shape_traj)
            resc_dir_z = functions.calc_resc_dir_z_fssh(
                sim, eigval_diff, eigvec_init_state, eigvec_final_state, dh_qc_dzc_traj
            )
        state[z_traj_name] = z[traj_ind]
        state[resc_dir_z_traj_name] = resc_dir_z
        state[eigval_diff_traj_name] = eigval_diff
        state, parameters = update_z_shift_fssh(
            sim,
            state,
            parameters,
            z_traj_name,
            resc_dir_z_traj_name,
            eigval_diff_traj_name,
            hop_successful_traj_name,
            z_shift_traj_name,
        )
        state_hop_successful[hop_traj_ind] = state[hop_successful_traj_name]
        state_z_shift[hop_traj_ind] = state[z_shift_traj_name]
        hop_traj_ind += 1
    return state, parameters


def update_z_hop(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_shift_name: str = "z_shift",
    hop_ind_name: str = "hop_ind",
    z_name: str = "z",
):
    """
    Updates the classical coordinates in trajectories that have hopped.

    Optional Keyword Arguments
    --------------------------
    z_shift_name:
        Name of the shift in coordinates for each hopping trajectory in the State object.
    hop_ind_name:
        Name of the indices of the trajectories that are attempting to hop in the State object.
    z_name: ndarray of shape (B, C), dtype=complex128
        Name of classical coordinates in the State object.

    Reads
    -----
    state[z_shift_name]: ndarray of shape (h, C), dtype=complex128
        Shift in classical coordinates for trajectories that could hop. Zero for frustrated hops, nonzero otherwise.
    state[hop_ind_name]: ndarray of shape (h,), dtype=int
        Trajectory indices of trajectories that could hop.

    Writes
    ------
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Notes
    -----
    * h is the number of trajectories that could hop as determined by the hopping probabilities
        and random number, which are now being evaluated for energy conservation.
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    * N = sim.model.constants.num_quantum_states
    """
    z = state[z_name]
    hop_ind = state[hop_ind_name]
    z_shift = state[z_shift_name]
    z[hop_ind] += z_shift
    state[z_name] = z
    return state, parameters


def update_act_surf_hop(
    sim: Simulation,
    state: dict,
    parameters: dict,
    hop_ind_name: str = "hop_ind",
    hop_dest_name: str = "hop_dest",
    hop_successful_name: str = "hop_successful",
    act_surf_name: str = "act_surf",
    act_surf_ind_name: str = "act_surf_ind",
):
    """
    Updates the active surface, active surface index, and active surface wavefunction
    following a hop.

    Optional Keyword Arguments
    --------------------------
    hop_ind_name:
        Name of the indices of the trajectories that are attempting to hop in the State object.
    hop_dest_name:
        Name of the destination states of the trajectories that are attempting to hop in the State object.
    hop_successful_name:
        Name of the flags indicating if each hop was successful in the State object.
    act_surf_name:
        Name of the active surface wavefunction in the State object.
    act_surf_ind_name:
        Name of the active surface index in the State object.

    Reads
    -----
    state[hop_successful_name]: ndarray of shape (h,), dtype=Bool
        Boolean indicating if the hop was successful or frustrated in the trajectories that could hop.
    state[hop_ind_name]: ndarray of shape (h,), dtype=int
        Trajectory indices of trajectories that could hop.
    state[hop_dest_name]: ndarray of shape (h,), dtype=int
        Destination states of trajectories that hop.

    Writes
    ------
    state[act_surf_name]: ndarray of shape (B, N), dtype=int
        Active surface vector in adiabatic basis: 1 if active 0 if not.
    state[act_surf_ind_name]: ndarray of shape (B,), dtype=int
        Active surface index.

    Notes
    -----
    * h is the number of trajectories that could hop as determined by the hopping probabilities
        and random number, which are now being evaluated for energy conservation.
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    hop_successful = state[hop_successful_name]
    hop_ind = state[hop_ind_name]
    hop_dest = state[hop_dest_name]
    act_surf = state[act_surf_name]
    act_surf_ind = state[act_surf_ind_name]
    # Get the index of the trajectories that successfully hopped.
    hop_successful_traj_ind = hop_ind[hop_successful]
    # Get their destination states.
    hop_dest_ind = hop_dest[hop_successful]
    # Zero out the active surface in the ones that hopped.
    act_surf[hop_successful_traj_ind] = 0
    # Set the new active surface to 1.
    act_surf[hop_successful_traj_ind, hop_dest_ind] = 1
    # Update the active surface index.
    act_surf_ind[hop_successful_traj_ind] = hop_dest_ind
    return state, parameters


def update_h_q_tot(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    h_q_name: str = "h_q",
    h_qc_name: str = "h_qc",
    h_q_tot_name: str = "h_q_tot",
):
    """
    Updates the Hamiltonian matrix of the quantum subsystem.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in State object.
    h_q_name:
        Name of the quantum Hamiltonian in the State object.
    h_qc_name:
        Name of the quantum-classical coupling Hamiltonian in the State object.
    h_q_tot_name:
        Name of the total Hamiltonian of the quantum subsystem in the State object.
        (``h_q + h_qc``)

    Constants and Settings
    ----------------------
    sim.model.update_h_q: Bool
        Flag used to determine if the quantum Hamiltonian should be updated at each timestep.

    Ingredients
    -----------
    h_q:
        Quantum Hamiltonian.
    h_qc:
        Quantum-classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinate.

    Writes
    ------
    state[h_q_name]: ndarray of shape (B, N, N), dtype=complex128
        Quantum Hamiltonian.
    state[h_qc_name]: ndarray of shape (B, N, N), dtype=complex128
        Quantum-classical Hamiltonian.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian (quantum plus quantum-classical).

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    * N = sim.model.constants.num_quantum_states
    """
    z = state[z_name]
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    if sim.model.update_h_q or not (h_q_name in state):
        # Update the quantum Hamiltonian if required or if it is not set.
        state[h_q_name] = h_q(sim.model, parameters, batch_size=sim.settings.batch_size)
    # Update the quantum-classical Hamiltonian.
    state[h_qc_name] = h_qc(sim.model, parameters, z=z)
    # Update the total Hamiltonian of the quantum subsystem.
    state[h_q_tot_name] = state[h_q_name] + state[h_qc_name]
    return state, parameters


def update_z_rk4_k123(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    z_k_name: str = "z_1",
    k_name: str = "z_rk4_k1",
    classical_force_name: str = "classical_force",
    quantum_classical_force_name: str = "quantum_classical_force",
    dt_factor: float = 0.5,
):
    """
    Computes the first three RK4 intermediates for evolving the classical coordinates.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of input coordinates in State object.
    z_k_name:
        Name of the output coordinates in the State object.
    k_name:
        Name of the k-th RK4 slope in the State object.
    classical_force_name:
        Name of the classical force in the State object.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.
    dt_factor:
        Factor to multiply the time step by. Typical values are 0.5 (for k1 and k2)
        and 1.0 (for k3).

    Reads
    -----
    state[classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force on the classical coordinates arising from the classical Hamiltonian.
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force on the classical coordinates arising from the quantum-classical Hamiltonian.
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Writes
    ------
    state[z_k_name]: ndarray of shape (B, C), dtype=complex128
        Intermediate classical coordinate after the k-th step of the RK4 algorithm.
    state[k_name]: ndarray of shape (B, C), dtype=complex128
        Slope of the k-th step of the RK4 algorithm.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    if sim.settings.debug:
        if dt_factor not in [0.5, 1.0]:
            logger.warning(
                "Unusual dt_factor %s passed to update_z_rk4_k123_sum. Typical values are 0.5 or 1.0.",
                dt_factor,
            )
    classical_force = state[classical_force_name]
    quantum_classical_force = state[quantum_classical_force_name]
    dt_update = sim.settings.dt_update
    z = state[z_name]
    z_k, k = functions.update_z_rk4_k123_sum(
        z, classical_force, quantum_classical_force, dt_factor * dt_update
    )
    state[z_k_name] = z_k
    state[k_name] = k
    return state, parameters


def update_z_rk4_k4(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    k1_name: str = "z_rk4_k1",
    k2_name: str = "z_rk4_k2",
    k3_name: str = "z_rk4_k3",
    classical_force_name: str = "classical_force",
    quantum_classical_force_name: str = "quantum_classical_force",
):
    """
    Computes the final RK4 update for evolving the classical coordinates.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of input coordinates in State object.
    k1_name:
        Name of the first RK4 slope in the State object.
    k2_name:
        Name of the second RK4 slope in the State object.
    k3_name:
        Name of the third RK4 slope in the State object.
    classical_force_name:
        Name of the classical force in the State object.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[k1_name]: ndarray of shape (B, C), dtype=complex128
        First RK4 slope.
    state[k2_name]: ndarray of shape (B, C), dtype=complex128
        Second RK4 slope.
    state[k3_name]: ndarray of shape (B, C), dtype=complex128
        Third RK4 slope.
    state[classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force on the classical coordinates arising from the classical Hamiltonian.
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force on the classical coordinates arising from the quantum-classical Hamiltonian.

    Writes
    ------
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Updated complex-valued classical coordinates.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    k1 = state[k1_name]
    k2 = state[k2_name]
    k3 = state[k3_name]
    classical_force = state[classical_force_name]
    quantum_classical_force = state[quantum_classical_force_name]
    out = functions.update_z_rk4_k4_sum(
        z,
        k1,
        k2,
        k3,
        classical_force,
        quantum_classical_force,
        sim.settings.dt_update,
    )
    state[z_name] = out
    return state, parameters


def update_dm_db_wf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_db_name: str = "wf_db",
    dm_db_name: str = "dm_db",
):
    """
    Updates the diabatic density matrix based on the wavefunction.

    Optional Keyword Arguments
    --------------------------
    wf_db_name:
        Name of the diabatic wavefunction in the State object.
    dm_db_name:
        Name of the diabatic density matrix in the State object.

    Reads
    -----
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.

    Writes
    ------
    state[dm_db_name]: ndarray of shape (B, N, N), dtype=complex128
        Density matrix elements in the diabatic basis.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    wf_db = state[wf_db_name]
    state[dm_db_name] = np.einsum(
        "ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy"
    )
    return state, parameters


def update_classical_energy(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    classical_energy_name: str = "classical_energy",
):
    """
    Updates the classical energy.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of the classical coordinates in the State object.
    classical_energy_name:
        Name under which to store the classical energy in the State object.

    Ingredients
    -----------
    h_c:
        Classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.

    Writes
    ------
    state[classical_energy_name]: ndarray of shape (B,), dtype=float64
        Classical energy in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    h_c, _ = sim.model.get("h_c")
    state[classical_energy_name] = np.real(h_c(sim.model, parameters, z=state[z_name]))
    return state, parameters


def update_classical_energy_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    classical_energy_name: str = "classical_energy",
    dm_adb_0_name: str = "dm_adb_0",
    branch_ind_name: str = "branch_ind",
):
    """
    Updates the classical energy for FSSH simulations.

    If deterministic, the energy in
    each branch is summed together with weights determined by the initial adiabatic
    populations. If not deterministic (and so there is only one branch), the energy is
    computed for the single branch.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in State object.
    dm_adb_0_name:
        Name of the initial adiabatic density matrix in the State object.
    branch_ind_name:
        Name of the branch indices in the State object.
    classical_energy_name:
        Name under which to store the classical energy in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.fssh_deterministic: Bool
        Boolean indicating if the FSSH simulation is deterministic.


    Ingredients
    -----------
    h_c:
        Classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[dm_adb_0_name]: ndarray of shape (B, N, N), dtype=complex128
        Initial adiabatic density matrix.
    state[branch_ind_name]: ndarray of shape (B,), dtype=int
        Branch index.

    Writes
    state[classical_energy_name]: ndarray of shape (B,), dtype=float64
        Classical energy in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    dm_adb_0 = state[dm_adb_0_name]
    branch_ind = state[branch_ind_name]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        classical_energy = 0.0
        branch_weights = num_branches * np.einsum(
            "tbbb->tb",
            dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
        )
        for branch_num in range(num_branches):
            z_branch = z[branch_ind == branch_num]
            classical_energy += np.real(
                branch_weights[:, branch_num]
                * h_c(
                    sim.model,
                    parameters,
                    z=z_branch,
                )
            )
    else:
        z_branch = z[branch_ind == 0]
        classical_energy = np.real(
            h_c(
                sim.model,
                parameters,
                z=z_branch,
            )
        )
    state[classical_energy_name] = classical_energy
    return state, parameters


def update_quantum_energy_wf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_db_name: str = "wf_db",
    h_q_tot_name: str = "h_q_tot",
    quantum_energy_name: str = "quantum_energy",
):
    """
    Updates the quantum energy w.r.t. the wavefunction.

    Optional Keyword Arguments
    --------------------------
    wf_db_name:
        Name of the wavefunction in the State object.
    h_q_tot_name:
        Name of the total Hamiltonian of the quantum subsystem in the State object.
        (``h_q + h_qc``)
    quantum_energy_name:
        Name under which to store the quantum energy in the State object.

    Reads
    -----
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian.

    Writes
    ------
    state[quantum_energy_name]: ndarray of shape (B,) dtype=float64
        Quantum energy in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    wf_db = state[wf_db_name]
    h_q_tot = state[h_q_tot_name]
    quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf_db), h_q_tot, wf_db, optimize="greedy")
    )
    state[quantum_energy_name] = quantum_energy
    return state, parameters


def update_quantum_energy_act_surf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_db_name: str = "act_surf_wf",
    h_q_tot_name: str = "h_q_tot",
    quantum_energy_name: str = "quantum_energy",
    dm_adb_0_name: str = "dm_adb_0",
):
    """
    Updates the quantum energy using the active surface wavefunction.

    Accounts for both stochastic and deterministic FSSH modes.

    Optional Keyword Arguments
    --------------------------
    wf_db_name:
        Name of the wavefunction in the State object.
    h_q_tot_name:
        Name of the total Hamiltonian of the quantum subsystem in the State object.
        (``h_q + h_qc``)
    quantum_energy_name:
        Name under which to store the quantum energy in the State object.
    dm_adb_0_name:
        Name of the initial adiabatic density matrix in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.fssh_deterministic: Bool
        Boolean indicating if the FSSH simulation is deterministic.

    Reads
    -----
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian.
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[dm_adb_0_name]: ndarray of shape (B, N, N), dtype=complex128
        Initial density matrix in the adiabatic basis.

    Writes
    ------
    state[quantum_energy_name]: ndarray of shape (B,), dtype=float64
        Quantum energy in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    h_q_tot = state[h_q_tot_name]
    wf_db = state[wf_db_name]
    dm_adb_0 = state[dm_adb_0_name]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf_db = wf_db * np.sqrt(
            num_branches
            * np.einsum(
                "tbbb->tb",
                dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
                optimize="greedy",
            ).flatten()[:, np.newaxis]
        )
        quantum_energy = np.real(
            np.einsum("ti,tij,tj->t", np.conj(wf_db), h_q_tot, wf_db, optimize="greedy")
        )
    else:
        quantum_energy = np.real(
            np.einsum("ti,tij,tj->t", np.conj(wf_db), h_q_tot, wf_db, optimize="greedy")
        )
    state[quantum_energy_name] = quantum_energy
    return state, parameters


def update_dm_db_fssh(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_adb_name: str = "wf_adb",
    dm_adb_0_name: str = "dm_adb_0",
    act_surf_name: str = "act_surf",
    dm_db_name: str = "dm_db",
    dm_adb_name: str = "dm_adb",
    eigvecs_name: str = "eigvecs",
):
    """
    Updates the diabatic density matrix for FSSH.

    Accounts for both stochastic and deterministic FSSH modes.

    Optional Keyword Arguments
    --------------------------
    wf_adb_name:
        Name of the adiabatic wavefunction in the State object.
    dm_adb_0_name:
        Name of the initial adiabatic density matrix in the State object.
    act_surf_name:
        Name of the active surface wavefunction in the State object.
    dm_db_name:
        Name of the diabatic density matrix in the State object.
    dm_adb_name:
        Name of the adiabatic density matrix in the State object.
    eigvecs_name:
        Name of the eigenvectors in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.fssh_deterministic: Bool
        Boolean indicating if the FSSH simulation is deterministic.

    Reads
    -----
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the adiabatic basis.
    state[dm_adb_0_name]: ndarray of shape (B, N, N), dtype=complex128
        Initial density matrix in the adiabatic basis.
    state[act_surf_name]: ndarray of shape (B, N), dtype=complex128
        Active surface.
    state[eigvecs_name]: ndarray of shape (B, N, N), dtype=complex128
        Eigenvectors of the total quantum Hamiltonian.

    Writes
    ------
    state[dm_adb_name]: ndarray of shape (B, N, N), dtype=complex128
        Density matrix in the adiabatic basis.
    state[dm_db_name]: ndarray of shape (B, N, N), dtype=complex128
        Density matrix in the diabatic basis.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    wf_adb = state[wf_adb_name]
    dm_adb_0 = state[dm_adb_0_name]
    act_surf = state[act_surf_name]
    eigvecs = state[eigvecs_name]
    dm_adb = np.einsum(
        "ti,tj->tij",
        wf_adb,
        np.conj(wf_adb),
        optimize="greedy",
    )
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt, _ in enumerate(dm_adb):
        np.einsum("jj->j", dm_adb[nt])[...] = act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        # This reweighting by num_branches simplifies the subsequent averaging.
        dm_adb = num_branches * (
            np.einsum(
                "tbbb->tb",
                dm_adb_0.reshape(
                    (batch_size, num_branches, num_quantum_states, num_quantum_states)
                ),
            ).flatten()[:, np.newaxis, np.newaxis]
            * dm_adb
        )
    dm_db = functions.transform_mat(dm_adb, eigvecs, adb_to_db=True)
    state[dm_adb_name] = dm_adb
    state[dm_db_name] = dm_db
    return state, parameters


def update_adb_connection(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    adb_connection_name: str = "adb_connection",
    classical_force_name: str = "classical_force",
    quantum_classical_force_name: str = "quantum_classical_force",
    derivative_coupling_dzc_name: str = "derivative_coupling_dzc",
    wf_overlaps_name: str = "aip_wf_overlaps",
    update_derivative_coupling: bool = False,
):
    """
    Updates the adiabatic connection matrix.


    This matrix describes the coupling between different adiabatic states.

    .. math::

        A = U^{\\dagger}\\partial_{t}U = B - B^{\\dagger}

    where

    .. math::

        B = \\dot{z}^{*}\\cdot U^{\\dagger}\\partial_{z^{*}} U

    and :math:`U` is a matrix of adiabatic states (column vectors).

    Updates the derivative coupling if needed.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    adb_connection_name:
        Name under which to store the adiabatic connection in the State object.
    classical_force_name:
        Name of the classical force in the State object.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.
    derivative_coupling_dzc_name:
        Name of the derivative coupling tensor in the State object.
    use_wf_overlaps:
        Boolean indicating if the wavefunction overlaps should be used to construct
        the adiabatic connection.
    wf_overlaps_name:
        Name of the wavefunction overlaps in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.use_wf_overlaps_for_adb_connection: bool, default: False
        If True, uses a finite difference formula to construct the adiabatic connection
        from wavefunction overlaps. If False, uses the derivative coupling.

    Ingredients
    -----------
    adb_connection:
        Adiabatic connection matrix.
    derivative_coupling_dzc:
        Derivative coupling matrix.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[wf_overlaps_name]: ndarray of shape (B, N, N), dtype=float64
        Wavefunction overlaps.

    Writes
    ------
    state[adb_connection_name]: ndarray of shape (B, N, N), dtype=complex128
        Adiabatic connection.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    * C = sim.model.constants.num_classical_coordinates
    """

    use_wf_overlaps = sim.algorithm.settings.use_wf_overlaps_for_adb_connection

    if wf_overlaps_name in state.keys() and use_wf_overlaps:
        state, parameters = update_wf_overlaps_gauge(
            sim, state, parameters, wf_overlaps_name=wf_overlaps_name
        )
        A = state[wf_overlaps_name]
        state[adb_connection_name] = (1 / (2 * sim.settings.dt_update)) * (
            A - np.transpose(A, axes=(0, 2, 1))
        )
    else:
        dz_dt = -1j * (
            state[classical_force_name] + state[quantum_classical_force_name]
        )
        need_derivative_coupling = True
        if derivative_coupling_dzc_name in state:
            derivative_coupling_dzc = state[derivative_coupling_dzc_name]
            need_derivative_coupling = False
        if update_derivative_coupling or need_derivative_coupling:
            state, parameters = update_derivative_coupling_dzc(
                sim,
                state,
                parameters,
                z_name=z_name,
                derivative_coupling_dzc_name=derivative_coupling_dzc_name,
            )
            state, parameters = update_derivative_coupling_dzc_gauge(
                sim,
                state,
                parameters,
                wf_overlaps_name=wf_overlaps_name,
                derivative_coupling_dzc_name=derivative_coupling_dzc_name,
            )
            state, parameters = update_wf_overlaps_gauge(
                sim, state, parameters, wf_overlaps_name=wf_overlaps_name
            )
            derivative_coupling_dzc = state[derivative_coupling_dzc_name]
        B = np.sum(
            np.conj(dz_dt)[:, :, np.newaxis, np.newaxis] * derivative_coupling_dzc,
            axis=1,
        )
        state[adb_connection_name] = B - np.conj(B).transpose((0, 2, 1))
    return state, parameters


def update_wf_adb_rk4(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_adb_name: str = "wf_adb",
    h_q_tot_name: str = "h_q_tot",
    adb_connection_name: str = "adb_connection",
):
    """
    Updates the adiabatic wavefunction using the 4th-order Runge-Kutta method.

    Optional Keyword Arguments
    --------------------------
    wf_adb_name:
        Name of the diabatic wavefunction in the State object.
    h_q_tot_name:
        Name of the quantum Hamiltonian in the State object.
    adb_connection_name:
        Name of the adiabatic connection in the State object.

    Reads
    -----
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the adiabatic basis.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian in the adiabatic basis.
    state[adb_connection_name]: ndarray of shape (B, N, N), dtype=complex128
        Adiabatic connection matrix.

    Writes
    ------
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Updated wavefunction coefficients in the adiabatic basis.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    dt_update = sim.settings.dt_update
    wf_adb = state[wf_adb_name]
    h_q_tot = state[h_q_tot_name]
    adb_connection = state[adb_connection_name]
    # Include the adiabatic connection in the Hamiltonian.
    h_q_tot_adb = h_q_tot - 1j * adb_connection
    k1 = -1j * functions.batch_matvec(h_q_tot_adb, wf_adb)
    k2 = -1j * functions.batch_matvec(h_q_tot_adb, (wf_adb + 0.5 * dt_update * k1))
    k3 = -1j * functions.batch_matvec(h_q_tot_adb, (wf_adb + 0.5 * dt_update * k2))
    k4 = -1j * functions.batch_matvec(h_q_tot_adb, (wf_adb + dt_update * k3))
    wf_adb += dt_update * 0.16666666666666666 * k1
    wf_adb += dt_update * 0.3333333333333333 * k2
    wf_adb += dt_update * 0.3333333333333333 * k3
    wf_adb += dt_update * 0.16666666666666666 * k4
    return state, parameters


def update_wf_adb_hop_prob(
    sim: Simulation,
    state: dict,
    parameters: dict,
    h_q_tot_name: str = "h_q_tot",
    adb_connection_name: str = "adb_connection",
    h_q_tot_previous_name: str = "h_q_tot_previous",
    adb_connection_previous_name: str = "adb_connection_previous",
    wf_adb_name: str = "wf_adb",
    update_hopping_probabilities: bool = False,
    hop_prob_name: str = "hop_prob",
):
    """
    Updates the adiabatic wavefunction by diagonalizing the Hamiltonian.

    Optionally integrates the hopping probability over the time interval.

    Hopping probability formula is taken from:

    Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.

    .. math::

        g_{k->j} = ( \int_{t}^{t+\Delta} b_{jk}(t') dt' ) / (c^{*}_{k}(t)c_{k}(t))

    .. math::

        b_{jk}(t) = -2\Re(c_{j}^{*}(t)c_{k}(t) A_{jk}(t))

    .. math::

        A_{jk}(t) = \dot{q}(t) \cdot d_{jk}(t)

    Note that this is consistent with Eq. 19-21 for real and complex :math:`d_{jk}`.

    Eq. 30 (follwoing) is equivalent to the above expression for :math:`b_{jk}` when :math:`d_{jk}` is real-valued:

    .. math::

        b_{jk}(t) = -2\Re(c_{j}(t)c_{k}^{*}(t) A_{jk}(t))

    Optional Keyword Arguments
    --------------------------
    h_q_tot_name:
        Name of the quantum Hamiltonian in the State object.
    adb_connection_name:
        Name of the adiabatic connection in the State object.
    h_q_tot_previous_name:
        Name of the quantum Hamiltonian from the previous time step in the State object.
    adb_connection_previous_name:
        Name of the adiabatic connection from the previous time step in the State object.
    wf_adb_name:
        Name of the adiabatic wavefunction in the State object.
    update_hopping_probabilities:
        Boolean indicating if to update the hopping probabilities.
    hop_prob_name:
        Name of the hopping probabilities in the State object.

    Constants and Settings
    ----------------------
    sim.algorithm.settings.update_wf_adb_eig_num_substeps: int, default: 1
        Number of substeps to use when updating the adiabatic wavefunction.

    Reads
    -----
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the adiabatic basis.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian in the adiabatic basis.
    state[adb_connection_name]: ndarray of shape (B, N, N), dtype=complex128
        Adiabatic connection matrix.
    state[h_q_tot_previous_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian in the adiabatic basis at the previous timestep.
    state[adb_connection_previous_name]: ndarray of shape (B, N, N), dtype=complex128
        Adiabatic connection matrix at the previous timestep.

    Writes
    ------
    state[wf_adb_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the adiabatic basis.
    state[hop_prob_name]: ndarray of shape (B, N), dtype=float64
        Hopping probabilities.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    num_substeps = sim.algorithm.settings.get("update_wf_adb_eig_num_substeps", 1)
    wf_adb = state[wf_adb_name]
    h_q_tot = state[h_q_tot_name]
    adb_connection = state[adb_connection_name]
    h_q_tot_previous = state[h_q_tot_previous_name]
    adb_connection_previous = state[adb_connection_previous_name]
    dt_update = sim.settings.dt_update
    num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    if update_hopping_probabilities:
        hop_prob_numerator = np.zeros(
            (sim.settings.batch_size, sim.model.constants.num_quantum_states)
        )
        act_surf_ind = state["act_surf_ind"]
        if sim.algorithm.settings.fssh_deterministic:
            num_branches = sim.model.constants.num_quantum_states

        num_trajs = sim.settings.batch_size // num_branches
        act_surf_population = (
            np.abs(
                wf_adb[np.arange(num_trajs * num_branches), act_surf_ind][:, np.newaxis]
            )
            ** 2
        )
    num_quantum_states = sim.model.constants.num_quantum_states
    for substep_ind in range(num_substeps):
        h_q_tot_interp = (substep_ind / num_substeps) * (
            h_q_tot - h_q_tot_previous
        ) + h_q_tot_previous
        adb_connection_interp = (substep_ind / num_substeps) * (
            adb_connection - adb_connection_previous
        ) + adb_connection_previous
        h_q_tot_adb_interp = h_q_tot_interp - 1j * adb_connection_interp
        eigvals, eigvecs = np.linalg.eigh(h_q_tot_adb_interp)
        if update_hopping_probabilities:
            hop_prob_numerator += -2 * np.real(
                np.conj(wf_adb)
                * wf_adb[np.arange(num_trajs * num_branches), act_surf_ind][
                    :, np.newaxis
                ]
                * adb_connection_interp[
                    np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind
                ].reshape((num_trajs * num_branches, num_quantum_states))
                * dt_update
                / num_substeps
            )
        wf_adb = np.einsum(
            "tia,ta,tja,tj->ti",
            eigvecs,
            np.exp(-1j * eigvals * dt_update / num_substeps),
            np.conj(eigvecs),
            wf_adb,
            optimize="greedy",
        )
    if update_hopping_probabilities:
        state[hop_prob_name] = hop_prob_numerator / act_surf_population
    state[wf_adb_name] = wf_adb
    return state, parameters


def update_q_velocity_verlet(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    classical_force_name: str = "classical_force",
    quantum_classical_force_name: str = "quantum_classical_force",
):
    """
    Updates the position component of the classical coordinates using velocity Verlet.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    classical_force_name:
        Name of the classical force in the State object.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.

    Reads
    -----
    state[z_name] : ndarray of shape (B, C), dtype=complex128
        Updated classical coordinates.
    state[classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the classical Hamiltonian.
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the quantum-classical Hamiltonian.

    Writes
    ------
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Updated complex-valued classical coordinates.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    dt_update = sim.settings.dt_update
    z = state[z_name]
    classical_force = state[classical_force_name]
    quantum_classical_force = state[quantum_classical_force_name]
    m = sim.model.constants.classical_coordinate_mass
    h = sim.model.constants.classical_coordinate_weight
    q = functions.z_to_q(z, m[np.newaxis], h[np.newaxis])
    p = functions.z_to_p(z, m[np.newaxis], h[np.newaxis])
    f = classical_force + quantum_classical_force
    f_dp, f_dq = functions.dzdzc_to_dqdp(None, f, m[np.newaxis], h[np.newaxis])
    q_dt = q + f_dq * dt_update - 0.5 * (f_dp / m[np.newaxis]) * dt_update**2
    z_dt = functions.qp_to_z(q_dt, p, m[np.newaxis], h[np.newaxis])
    state[z_name] = z_dt
    return state, parameters


def update_p_velocity_verlet(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    classical_force_name: str = "classical_force",
    classical_force_previous_name: str = "classical_force_previous",
    quantum_classical_force_name: str = "quantum_classical_force",
    quantum_classical_force_previous_name: str = "quantum_classical_force_previous",
):
    """
    Updates the momentum component of the classical coordinates using velocity Verlet.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    classical_force_name:
        Name of the classical force in the State object.
    quantum_classical_force_name:
        Name of the quantum-classical force in the State object.
    quantum_classical_force_previous_name:
        Name of the quantum-classical force from the previous time step in the State object.

    Reads
    -----
    state[z_name] : ndarray of shape (B, C), dtype=complex128
        Updated classical coordinates.
    state[classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the classical Hamiltonian.
    state[classical_force_previous_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the classical Hamiltonian at the previous timestep.
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the quantum-classical Hamiltonian.
    state[quantum_classical_force_previous_name]: ndarray of shape (B, C), dtype=complex128
        Force arising from the quantum-classical Hamiltonian at the previous timestep.

    Writes
    ------
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Updated complex-valued classical coordinates.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    dt_update = sim.settings.dt_update
    z = state[z_name]
    classical_force = state[classical_force_name]
    classical_force_previous = state[classical_force_previous_name]
    quantum_classical_force = state[quantum_classical_force_name]
    quantum_classical_force_previous = state[quantum_classical_force_previous_name]
    m = sim.model.constants.classical_coordinate_mass
    h = sim.model.constants.classical_coordinate_weight
    p = functions.z_to_p(z, m[np.newaxis], h[np.newaxis])

    f_previous = classical_force_previous + quantum_classical_force_previous
    f_dq_previous, _ = functions.dzdzc_to_dqdp(
        None, f_previous, m[np.newaxis], h[np.newaxis]
    )
    f = classical_force + quantum_classical_force
    f_previous = classical_force_previous + quantum_classical_force_previous
    f_dq, _ = functions.dzdzc_to_dqdp(None, f, m[np.newaxis], h[np.newaxis])
    f_dq_previous, _ = functions.dzdzc_to_dqdp(
        None, f_previous, m[np.newaxis], h[np.newaxis]
    )
    p_dt = p - 0.5 * (f_dq + f_dq_previous) * dt_update
    q_dt = functions.z_to_q(z, m[np.newaxis], h[np.newaxis])
    z_dt = functions.qp_to_z(q_dt, p_dt, m[np.newaxis], h[np.newaxis])
    state[z_name] = z_dt
    return state, parameters


def update_derivative_coupling_dzc_gauge(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_overlaps_name: str = "aip_wf_overlaps",
    derivative_coupling_dzc_name: str = "derivative_coupling_dzc",
):
    """
    Updates the gauge of the derivative coupling tensor using the wavefunction overlaps.
    Assumes real-valued wavefunction overlaps.

    Optional Keyword Arguments
    --------------------------
    wf_overlaps_name:
        Name of the wavefunction overlaps in the State object.
    derivative_coupling_dzc_name:
        Name of the derivative coupling tensor in the State object.

    Reads
    -----
    state[wf_overlaps_name]: ndarray of shape (B, N), dtype=float64
        Wavefunction overlaps.
    state[derivative_coupling_dzc_name]: ndarray of shape (B, C, N, N), dtype=float64
        Derivative coupling tensor.

    Writes
    ------
    state[derivative_coupling_dzc_name]: ndarray of shape (B, C, N, N), dtype=float64
        Derivative coupling tensor in the updated gauge.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    * C = sim.model.constants.num_classical_coordinates
    """
    if wf_overlaps_name in state.keys():
        wf_overlaps = np.einsum("tii->ti", state[wf_overlaps_name])
        signs = np.sign(wf_overlaps)
        state[derivative_coupling_dzc_name] = np.einsum(
            "ti,tcij,tj->tcij",
            signs,
            state[derivative_coupling_dzc_name],
            signs,
            optimize="greedy",
        )
    return state, parameters


def update_wf_overlaps_gauge(
    sim: Simulation,
    state: dict,
    parameters: dict,
    wf_overlaps_name: str = "aip_wf_overlaps",
):
    """
    Updates the gauge of the overlap matrix. Assumes real-valued overlaps.

    Optional Keyword Arguments
    --------------------------
    wf_overlaps_name:
        The name of the wavefunction overlaps in the State object.

    Reads
    -----
    state[wf_overlaps_name]: ndarray of shape (B, N), dtype=float64
        Wavefunction overlaps.

    Writes
    ------
    state[wf_overlaps_name]: ndarray of shape (B, N), dtype=float64
        Wavefunction overlaps in the updated gauge.

    Notes
    -----
    * B = sim.settings.batch_size
    * N = sim.model.constants.num_quantum_states
    """
    if wf_overlaps_name in state.keys():
        wf_overlaps = np.einsum("tii->ti", state[wf_overlaps_name])
        signs = np.sign(wf_overlaps)
        state[wf_overlaps_name] = np.einsum(
            "tij,tj->tij", state[wf_overlaps_name], signs, optimize="greedy"
        )
    return state, parameters


def update_ab_initio_property(
    sim: Simulation,
    state: dict,
    parameters: dict,
    ab_initio_property_name: str = "ab_initio_property",
    property_dict: dict = {
        "energy": {"z": "z", "excited_amplitudes": True},
        "gradient": {"z": "z", "state_inds_gradient": None},
        "derivative_coupling": {"z": "z", "state_inds_derivative_coupling": None},
        "wf_overlaps": {"z": "z", "z_previous": "z_previous"},
    },
):
    """
    Calculates ab initio properties using the ab initio property calculator ingredient.

    Stores the properties as a list of dictionaries in the Parameters object, a dictionary
    in the State object containing vectorized properties, and as new variables in the State object
    with the same name in property_dict appended with "aip_" where aip stands for "ab initio property".

    Optional Keyword Arguments
    --------------------------
    ab_initio_property_name:
        The name under which to store the ab initio properties in the State and Parameters objects.
    property_dict:
        A dictionary of properties and associated argument dictionaries, i.e. {property_name:{arg1:arg1_name,...}}.
        The argument dictionaries contain names of objects in the State object, Booleans, or None.

    Ingredients
    -----------
    ab_initio_property_calculator:
        Ab initio property calculator.

    Reads
    -----

    Writes
    ------
    parameters[ab_initio_property_name]: List
        A List of dictionaries containing the ab initio properties calculated for each
        trajectory.
    state[ab_initio_property_name]: Dict
        A dictionary containing the calculated ab initio properties restructured into a
        ndarray where the first index corresponds to the trajectory index.
    state["aip_" + property]: ndarray of shape (B, shape(property)), dtype=type(property)
        A new variable in the State object giving the result of the calculation of property.

    Notes
    -----
    * B = sim.settings.batch_size
    """
    parameters[ab_initio_property_name] = np.array(
        [{} for _ in range(sim.settings.batch_size)]
    )
    state[ab_initio_property_name] = {}

    ab_initio_property_calculator, has_ab_intio_property_calculator = sim.model.get(
        "ab_initio_property_calculator"
    )
    if has_ab_intio_property_calculator:
        new_property_dict = copy.deepcopy(property_dict)
        for property_key in new_property_dict.keys():
            args_dict = new_property_dict[property_key]
            for args_key in args_dict.keys():
                if args_dict[args_key] is None:
                    args_dict[args_key] = None
                elif type(args_dict[args_key]) is bool:
                    args_dict[args_key] = args_dict[args_key]
                else:
                    args_dict[args_key] = state[args_dict[args_key]]
            new_property_dict[property_key] = args_dict
        new_ab_initio_property = ab_initio_property_calculator(
            sim.model,
            parameters,
            batch_size=sim.settings.batch_size,
            property_dict=new_property_dict,
        )
        parameters[ab_initio_property_name] = new_ab_initio_property
        new_results_dict = {}
        ab_initio_property = parameters[ab_initio_property_name]
        for key in ab_initio_property[0].keys():
            new_results_dict[key] = np.array(
                [ab_initio_property[n][key] for n in range(len(ab_initio_property))]
            )
        state[ab_initio_property_name] = new_results_dict
        for key in new_results_dict.keys():
            state["aip_" + key] = new_results_dict[key]

    return state, parameters
