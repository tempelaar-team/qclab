"""
This module contains functions used in QC Lab. This includes
functions that are used by tasks, ingredients, and the
dynamics drivers.
"""

import logging
import functools
import math
import numpy as np
from qclab.utils import njit
import qclab.numerical_constants as numerical_constants

logger = logging.getLogger(__name__)

@njit
def erf_jit(x):
    """
    Error function of each element of a one-dimensional array.

    Parameters
    ----------
    x : ndarray of shape (B,), dtype=float64
        Input values.

    Returns
    -------
    out : ndarray of shape (B,), dtype=float64
        Error function of each element.
    """
    out = np.empty(x.shape[0])
    for idx in range(x.shape[0]):
        out[idx] = math.erf(x[idx])
    return out

def batch_matvec(mat, vec):
    """
    Multiplies a matrix with a vector where there is an initial set of indices.

    Assumes that the last two indices of ``mat`` are the matrix indices,
    and the last index of ``vec`` is the vector index. The other indices
    must be broadcastable. This rule is needed to ensure that the vector
    is not interpreted as a single matrix.

    (..., i, j) x (..., j) -> (..., i)

    .. rubric:: Args

    mat : ndarray
        Input matrix.
    vec : ndarray
        Input vector.

    .. rubric:: Returns

    out : ndarray
        Result of the multiplication.
    """
    return np.matmul(mat, vec[..., None])[..., 0]


def transform_vec(vec, basis, adb_to_db=False):
    """
    Transforms a vector ``vec`` to  a new basis defined by the
    column vectors of the matrix ``basis``.

    If ``basis`` are eigenvectors of a diabatic Hamiltonian, then this
    transformation ammounts to a transformation from the diabatic
    to adiabatic basis.

    Assumes that the last two indices of ``basis`` are the matrix indices,
    and the last index of ``vec`` is the vector index. The other indices
    must be broadcastable.

    .. rubric:: .. rubric:: Args

    vec : ndarray
        Input vector.
    basis : ndarray
        Basis matrix.
    adb_to_db : bool, optional, default: False
        If True, reverses the direction of the transformation
        by using the Hermitian conjugate of basis.

    .. rubric:: Returns

    out : ndarray
        Transformed matrix
    """
    if adb_to_db:
        out = batch_matvec(basis, vec)
    else:
        out = batch_matvec(np.swapaxes(np.conj(basis), -1, -2), vec)
    return out


def transform_mat(mat, basis, adb_to_db=False):
    """
    Transforms a matrix ``mat`` to a new basis defined by the
    column vectors of the basis ``basis``.

    If ``basis`` are eigenvectors of a diabatic Hamiltonian, then this
    transformation ammounts to a transformation from the diabatic
    to adiabatic basis.

    Assumes that the last two indices of ``mat`` and ``basis`` are the matrix
    indices. The other indices must be broadcastable.

    .. rubric:: Args

    mat : ndarray
        Input matrix.
    basis : ndarray
        Unitary matrix.
    adb_to_db : bool, optional, default: False
        If True, reverses the direction of the transformation
        by using the Hermitian conjugate of basis.

    .. rubric:: Returns

    out : ndarray
        Transformed matrix
    """
    if adb_to_db:
        out = np.matmul(basis, np.matmul(mat, np.swapaxes(basis.conj(), -1, -2)))
    else:
        out = np.matmul(np.swapaxes(basis.conj(), -1, -2), np.matmul(mat, basis))
    return out


@njit
def update_z_rk4_k123_sum(z_k, classical_force, quantum_classical_force, dt_update):
    """
    Low-level function to calculate the intermediate z coordinate and k values
    for RK4 update. Applies to steps 1-3.

    .. rubric:: Args

    z_k : ndarray
        Initial complex coordinate for that update step.
    classical_force : ndarray
        Classical force.
    quantum_classical_force : ndarray
        Quantum-classical force.
    dt_update : float
        Time step for the update.

    .. rubric:: Returns

    out : ndarray
        Updated complex coordinates after applying the RK4 step.
    k : ndarray
        The k value used in the RK4 update.
    """
    batch_size, num_classical_coordinates = z_k.shape
    k = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    out = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    for i in range(z_k.shape[0]):
        for j in range(z_k.shape[1]):
            k[i, j] = -1j * (classical_force[i, j] + quantum_classical_force[i, j])
            out[i, j] = z_k[i, j] + dt_update * k[i, j]
    return out, k


@njit
def update_z_rk4_k4_sum(
    z_0, k1, k2, k3, classical_force, quantum_classical_force, dt_update
):
    """
    Low-level function to calculate the fourth (and final) step for the RK4 update.

    .. rubric:: Args

    z_0 : ndarray
        Initial complex coordinate.
    k1 : ndarray
        First RK4 slope.
    k2 : ndarray
        Second RK4 slope.
    k3 : ndarray
        Third RK4 slope.
    classical_force : ndarray
        Classical force.
    quantum_classical_force : ndarray
        Quantum-classical force.
    dt_update : float
        Time step for the update.

    .. rubric:: Returns

    z_0 : ndarray
        Updated complex coordinates after applying the RK4 step.
    """
    for i in range(z_0.shape[0]):
        for j in range(z_0.shape[1]):
            z_0[i, j] = z_0[i, j] + (dt_update / 6.0) * (
                k1[i, j]
                + 2.0 * k2[i, j]
                + 2.0 * k3[i, j]
                - 1j * (classical_force[i, j] + quantum_classical_force[i, j])
            )
    return z_0


@njit
def dqdp_to_dzc(dq, dp, m, h):
    """
    Convert derivatives w.r.t. q and p (``dq`` and ``dp``, respectively) to
    the derivative w.r.t. zc (``dzc``).

    .. rubric:: Args

    dq : ndarray | None
        Derivative w.r.t. position coordinate.
    dp : ndarray | None
        Derivative w.r.t. momentum coordinate.
    m : ndarray
        classical coordinate mass.
    h : ndarray
        classical coordinate weight.

    .. rubric:: Returns

    dzc : ndarray
        Derivative w.r.t. conjugate complex coordinate.
    """
    dp_present = dp is not None
    dq_present = dq is not None
    if dq_present and dp_present:
        return np.sqrt(0.5 / (m * h)) * dq + 1j * np.sqrt(m * h / 2) * dp
    if dp_present:
        return 1j * np.sqrt(m * h / 2) * dp
    if dq_present:
        return np.sqrt(0.5 / (m * h)) * dq.astype(np.complex128)
    raise ValueError("At least one of dq or dp must be provided.")


@njit
def dzdzc_to_dqdp(dz, dzc, m, h):
    """
    Convert derivatives w.r.t. z and zc (``dz`` and ``dzc``) to derivatives w.r.t.
    q and p (``dq`` and ``dp``). If only one of ``dz`` or ``dzc`` is provided, then
    ``dq`` and ``dp`` are calculated assuming that :math:`dz = dzc^{*}`.

    .. rubric:: Args

    dz : ndarray | None
        Derivative w.r.t. complex z coordinate.
    dzc : ndarray | None
        Derivative w.r.t. conjugate z coordinate.
    m : ndarray
        classical coordinate mass.
    h : ndarray
        classical coordinate weight.

    .. rubric:: Returns

    dq : ndarray
        Derivative w.r.t. position coordinate.
    dp : ndarray
        Derivative w.r.t. momentum coordinate.
    """
    dz_present = dz is not None
    dzc_present = dzc is not None
    if dz_present and dzc_present:
        dq = np.sqrt(0.5 * m * h) * (dz + dzc)
        dp = 1j * np.sqrt(0.5 / (m * h)) * (dz - dzc)
        return dq, dp
    if dz_present:
        dq = np.sqrt(0.5 * m * h) * (dz + np.conj(dz))
        dp = 1j * np.sqrt(0.5 / (m * h)) * (dz - np.conj(dz))
        return dq, dp
    if dzc_present:
        dq = np.sqrt(0.5 * m * h) * (np.conj(dzc) + dzc)
        dp = 1j * np.sqrt(0.5 / (m * h)) * (np.conj(dzc) - dzc)
        return dq, dp
    raise ValueError("At least one of dz or dzc must be provided.")


@njit
def z_to_q(z, m, h):
    """
    Convert complex coordinates to position coordinate.

    .. rubric:: Args


    z : ndarray
        Complex coordinates.
    m : ndarray
        Classical coordinate mass.
    h : ndarray
        Classical coordinate weight.

    .. rubric:: Returns


    q : ndarray
        Position coordinates.
    """
    return np.sqrt(2.0 / (m * h)) * z.real


@njit
def z_to_p(z, m, h):
    """
    Convert complex coordinates to momentum coordinate.

    .. rubric:: Args


    z : ndarray
        Complex coordinates.
    m : ndarray
        Classical coordinate mass.
    h : ndarray
        Classical coordinate weight.

    .. rubric:: Returns


    p : ndarray
        Momentum coordinates.
    """
    return np.sqrt(2.0 * m * h) * z.imag


@njit
def qp_to_z(q, p, m, h):
    """
    Convert real coordinates to complex coordinates.
    If only one of ``q`` or ``p`` is provided, then the other is assumed to be zero.

    .. rubric:: Args


    q : ndarray | None
        Position coordinates.
    p : ndarray | None
        Momentum coordinates.
    m : ndarray
        Classical coordinate mass.
    h : ndarray
        Classical coordinate weight.

    .. rubric:: Returns


    z : ndarray
        Complex coordinates.
    """
    q_present = q is not None
    p_present = p is not None
    if q_present and p_present:
        return np.sqrt(0.5 * m * h) * q + 1j * np.sqrt(0.5 / (m * h)) * p
    if q_present and not p_present:
        return np.sqrt(0.5 * m * h) * q.astype(np.complex128)
    if not q_present and p_present:
        return 1j * np.sqrt(0.5 / (m * h)) * p
    raise ValueError("At least one of q or p must be provided.")


def make_ingredient_sparse(ingredient):
    """
    Wrapper that converts a vectorized ingredient output to a sparse format consisting
    of the indices (``inds``), nonzero elements (``mels``), and ``shape``.

    .. rubric:: Args

    ingredient : function
        Ingredient to be converted to sparse format.

    .. rubric:: Returns

    sparse_ingredient : function
        Sparse version of the ingredient.
    """

    @functools.wraps(ingredient)
    def sparse_ingredient(*args, **kwargs):
        (model, parameters) = args
        out = ingredient(model, parameters, **kwargs)
        inds = np.where(out != 0)
        mels = out[inds]
        shape = np.shape(out)
        return inds, mels, shape

    return sparse_ingredient


def vectorize_ingredient(ingredient):
    """
    Wrapper that vectorizes ingredient functions.
    It assumes that any kwarg is an ndarray that is vectorized over its first
    index. Other kwargs are assumed to not be vectorized.

    .. rubric:: Args

    ingredient : function
        Ingredient to be vectorized.

    .. rubric:: Returns

    vectorized_ingredient : function
        Vectorized version of the ingredient.
    """

    @functools.wraps(ingredient)
    def vectorized_ingredient(*args, **kwargs):
        (model, parameters) = args
        if kwargs.get("z") is not None:
            batch_size = len(kwargs["z"])
        else:
            batch_size = kwargs["batch_size"]
        keys = kwargs.keys()
        kwargs_list = []
        for n in range(batch_size):
            kwargs_n = {"traj_ind": n}
            for key in keys:
                if isinstance(kwargs[key], np.ndarray):
                    kwargs_n[key] = kwargs[key][n]
                else:
                    kwargs_n[key] = kwargs[key]
            kwargs_list.append(kwargs_n)
        out = np.array(
            [ingredient(model, parameters, **kwargs_list[n]) for n in range(batch_size)]
        )
        return out

    return vectorized_ingredient


@njit
def dh_c_dzc_harmonic_jit(z, h, w):
    """
    Derivative of the harmonic oscillator classical Hamiltonian function with respect to
    the conjugate z coordinate.

    .. rubric:: Args

    z : ndarray
        Complex coordinates.
    h : ndarray
        Classical coordinate weight.
    w : ndarray
        Harmonic frequency.

    .. rubric:: Returns

    out : ndarray
        Derivative of the harmonic oscillator classical Hamiltonian function with respect to
        the conjugate z coordinate.
    """

    batch_size, num_classical_coordinates = z.shape
    out = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    w2_over_h = (w**2) / h
    for i in range(batch_size):
        for j in range(num_classical_coordinates):
            zij = z[i, j]
            out[i, j] = complex(w2_over_h[j] * zij.real, h[j] * zij.imag)

    return out


@njit
def h_qc_diagonal_linear_jit(z, gamma):
    """
    Low-level function to generate the diagonal linear quantum-classical interaction Hamiltonian.

    :math:`H_{nm} = \\delta_{nm}\\sum_{\\xi} \\gamma_{n\\xi} (z_{\\xi} + z_{\\xi}^*)`

    .. rubric:: Args

    z : ndarray
        Complex coordinates.
    gamma : ndarray
        Classical coordinate coupling strengths.

    .. rubric:: Returns

    h_qc : ndarray
        Diagonal linear quantum-classical interaction Hamiltonian.
    """
    batch_size = z.shape[0]
    num_classical_coordinates = z.shape[1]
    num_sites = gamma.shape[0]
    h_qc = np.zeros((batch_size, num_sites, num_sites), dtype=np.complex128)
    for b in range(batch_size):
        for i in range(num_sites):
            acc = 0.0
            for j in range(num_classical_coordinates):
                acc += gamma[i, j] * 2.0 * z[b, j].real
            h_qc[b, i, i] = acc
    return h_qc


def gen_sample_gaussian(constants, z_initial=None, seed=None, separable=True):
    """
    Generates a complex number sampled from a Gaussian distribution.

    If ``z_initial`` is provided, then a Gaussian distribution centered around
    ``z_initial`` is sampled.
    If ``z_initial`` is not provided, then a Gaussian distribution centered around
    the origin is sampled.

    If ``separable`` is ``True``, then a different random number is generated
    for each classical coordinate (i.e., each coordinate corresponds to
    a different random walker). If ``False``, then the same random number
    is generated for all classical coordinates (i.e., a single random walker).

    .. rubric:: Args

    constants : Constants
        Constants object.
    z_initial : ndarray | None, default: None
        Center of the Gaussian distribution. If ``None``, the distribution is
        centered around the origin.
    seed : int | None
        Random seed for reproducibility.
    separable : bool, default: True
        Whether to generate a different random number for each classical coordinate.

    .. rubric:: Model Constants

    mcmc_std : float, default: 1.0
        Standard deviation of the Gaussian distribution.

    .. rubric:: Returns

    z : ndarray
        Complex number sampled from a Gaussian distribution.
    rand : ndarray
        Random number(s) used to generate the complex number.
    """
    if seed is not None:
        np.random.seed(seed)
    num_classical_coordinates = constants.num_classical_coordinates
    if separable:
        rand = np.random.rand(num_classical_coordinates)
    else:
        rand = np.random.rand()
    if z_initial is None:
        z_initial = np.zeros(num_classical_coordinates, dtype=complex)
    mcmc_std = constants.get("mcmc_std", 1.0)
    z_re = np.random.normal(
        loc=z_initial.real, scale=mcmc_std, size=num_classical_coordinates
    )
    z_im = np.random.normal(
        loc=z_initial.imag, scale=mcmc_std, size=num_classical_coordinates
    )
    z = z_re + 1j * z_im
    return z, rand


@njit
def calc_sparse_inner_product(inds, mels, shape, vec_l_conj, vec_r, out=None):
    """
    Take a sparse gradient matrix with shape ``(batch_size, num_classical_coordinates,
    num_quantum_state, num_quantum_states)`` and calculate the matrix element of
    the vectors ``vec_l_conj`` and ``vec_r`` with  shape ``(batch_size*num_quantum_states)``.

    .. rubric:: Args

    inds : tuple of ndarrays
        Indices of the nonzero elements in the sparse matrix.
    mels : ndarray
        Nonzero elements of the sparse matrix.
    shape : tuple
        Shape of the sparse matrix.
    vec_l_conj : ndarray
        Left vector (conjugated) for the inner product.
    vec_r : ndarray
        Right vector for the inner product.
    out : ndarray | None
        Preallocated output array. If ``None``, a new array is created.

    .. rubric:: Returns

    out : ndarray
        Result of the inner product with shape ``(batch_size, num_classical_coordinates)``.
    """
    batch_size, num_classical_coordinates, num_quantum_states = (
        shape[0],
        shape[1],
        shape[2],
    )
    r = inds[0]
    c = inds[1]
    a = inds[2]
    b = inds[3]

    if out is None:
        out = np.zeros(batch_size * num_classical_coordinates, dtype=np.complex128)
    if out is not None:
        out.fill(0.0j)

    l_flat = vec_l_conj.reshape(batch_size * num_quantum_states)
    r_flat = vec_r.reshape(batch_size * num_quantum_states)

    for i in range(mels.shape[0]):
        ri = r[i]
        idx_out = ri * num_classical_coordinates + c[i]
        idx_l = ri * num_quantum_states + a[i]
        idx_r = ri * num_quantum_states + b[i]
        out[idx_out] += l_flat[idx_l] * mels[i] * r_flat[idx_r]

    return out


def analytic_der_couple_phase(sim, dh_qc_dzc, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytical derivative
    couplings, i.e., calculates the phase-factors :math:`u^{q}_{\\alpha}` and :math:`u^{p}_{\\alpha}`
    such that :math:`d_{\\alpha\\beta}^{q}u_{\\alpha}^{q*}u_{\\beta}^{q}` and :math:`d_{\\alpha\\beta}^{p}u_{\\alpha}^{p*}u_{\\beta}^{p}`
    are real-valued.

    It does this by calculating the derivative couplings analytically. In the event of
    degenerate eigenvalues, an error is logged and a small offset is added to the energy
    differences.

    .. rubric:: Args

    sim: Simulation
        Simulation object.
    dh_qc_dzc : tuple
        Sparse representation of the derivative of the quantum-classical interaction Hamiltonian
        with respect to the conjugate complex coordinate.
    eigvals : ndarray
        Eigenvalues of the quantum subsystem.
    eigvecs : ndarray
        Eigenvectors of the quantum subsystem.

    .. rubric::
    der_couple_dq_phase : ndarray
        Phase factor for derivative couplings obtained by differentiating
        w.r.t. the position coordinate.
    der_couple_dp_phase : ndarray
        Phase factor for derivative couplings obtained by differentiating
        w.r.t. the momentum coordinate.
    """
    inds, mels, shape = dh_qc_dzc
    batch_size = shape[0]
    m = sim.model.constants.classical_coordinate_mass
    h = sim.model.constants.classical_coordinate_weight
    num_classical_coords = shape[1]
    num_quantum_states = shape[2]
    der_couple_dq_phase = np.ones(
        (
            batch_size,
            num_quantum_states,
        ),
        dtype=complex,
    )
    der_couple_dp_phase = np.ones(
        (
            batch_size,
            num_quantum_states,
        ),
        dtype=complex,
    )
    for i in range(num_quantum_states - 1):
        j = i + 1
        evec_i = eigvecs[..., i]
        evec_j = eigvecs[..., j]
        eval_i = eigvals[..., i]
        eval_j = eigvals[..., j]
        eigval_diff = eval_j - eval_i
        plus = np.zeros_like(eigval_diff)
        if sim.settings.debug:
            if np.any(np.abs(eigval_diff) < numerical_constants.SMALL):
                plus[np.where(np.abs(eigval_diff) < numerical_constants.SMALL)] = 1
                logger.error("Degenerate eigenvalues detected.")
        der_couple_dzc = np.zeros(
            (
                batch_size,
                num_classical_coords,
            ),
            dtype=complex,
        )
        der_couple_dz = np.zeros(
            (
                batch_size,
                num_classical_coords,
            ),
            dtype=complex,
        )
        np.add.at(
            der_couple_dzc,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[2]]
            * mels
            * evec_j[inds[0], inds[3]]
            / ((eigval_diff + plus)[inds[0]]),
        )
        np.add.at(
            der_couple_dz,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[3]]
            * np.conj(mels)
            * evec_j[inds[0], inds[2]]
            / ((eigval_diff + plus)[inds[0]]),
        )
        der_couple_dq, der_couple_dp = dzdzc_to_dqdp(
            der_couple_dz, der_couple_dzc, m[np.newaxis, :], h[np.newaxis, :]
        )
        der_couple_dq_angle = np.angle(
            der_couple_dq[
                np.arange(len(der_couple_dq)),
                np.argmax(np.abs(der_couple_dq), axis=-1),
            ]
        )
        der_couple_dp_angle = np.angle(
            der_couple_dp[
                np.arange(len(der_couple_dp)),
                np.argmax(np.abs(der_couple_dp), axis=-1),
            ]
        )
        der_couple_dq_angle[
            np.where(np.abs(der_couple_dq_angle) < numerical_constants.SMALL)
        ] = 0
        der_couple_dp_angle[
            np.where(np.abs(der_couple_dp_angle) < numerical_constants.SMALL)
        ] = 0
        der_couple_dq_phase[..., i + 1 :] = (
            np.exp(1j * der_couple_dq_angle[..., np.newaxis])
            * der_couple_dq_phase[..., i + 1 :]
        )
        der_couple_dp_phase[..., i + 1 :] = (
            np.exp(1j * der_couple_dp_angle[..., np.newaxis])
            * der_couple_dp_phase[..., i + 1 :]
        )
    return der_couple_dq_phase, der_couple_dp_phase


def calc_resc_dir_z_fssh(
    sim, eigval_diff, eigvec_init_state, eigvec_final_state, dh_qc_dzc_traj
):
    """
    Calculates the rescaling direction for FSSH using analytical derivative couplings.

    This function is not vectorized over multiple trajectories and is intended to be
    only called when needed in that trajectory.

    It calculates both the derivative coupling w.r.t. z and z*, and checks that they
    are properly aligned to correspond with real-valued phase space derivative couplings.
    If they are not, an error is logged.

    .. rubric:: Args

    eigval_diff : ndarray
        Difference in eigenvalues between the initial and final states, ``e_final - e_initial``.
    eigvec_init_state : ndarray
        Eigenvector of the initial state.
    eigvec_final_state : ndarray
        Eigenvector of the final state.
    dh_qc_dzc_traj : tuple
        Sparse representation of the derivative of the quantum-classical Hamiltonian
        with respect to the conjugate complex coordinate.
    m : ndarray
        Classical coordinate mass.
    h : ndarray
        Classical coordinate weight.

    .. rubric:: Returns

    dkj_zc : ndarray
        Nonadiabatic coupling vector for rescaling the z coordinate.
        Derivative is w.r.t. z*.
    """
    inds, mels, shape = dh_qc_dzc_traj
    num_classical_coordinates = shape[1]
    dkj_z = np.zeros((num_classical_coordinates), dtype=complex)
    dkj_zc = np.zeros((num_classical_coordinates), dtype=complex)
    np.add.at(
        dkj_zc,
        (inds[1]),
        np.conj(eigvec_init_state)[inds[2]]
        * mels
        * eigvec_final_state[inds[3]]
        / eigval_diff,
    )
    if sim.settings.debug:
        np.add.at(
            dkj_z,
            (inds[1]),
            np.conj(eigvec_init_state)[inds[3]]
            * np.conj(mels)
            * eigvec_final_state[inds[2]]
            / eigval_diff,
        )
        # Determine the position of the maximum derivative coupling.
        max_pos = np.argmax(np.abs(dkj_zc))
        dkj_zc_max = dkj_zc[max_pos]
        dkj_z_max = dkj_z[max_pos]
        # Calculate the fraction of error in the real and imaginary parts,
        # i.e., the deviation from dkj_zc = dkj_z*.
        alignment_error = np.abs(dkj_zc_max - np.conj(dkj_z_max)) / np.abs(dkj_zc_max)
        if alignment_error > numerical_constants.GAUGE_FIX_THRESHOLD:
            logger.warning(
                "Derivative couplings are not properly aligned. "
                "Gauge fixing may be needed.\n"
                "Fraction of alignment error: %s",
                alignment_error,
            )

    return dkj_zc


def numerical_fssh_hop(model, parameters, **kwargs):
    """
    Determines the shift required to conserve energy during a hop in FSSH using
    an iterative numerical method. The complex-valued classical coordinates following
    the hop are ``z + shift``.

    The shift is ``-1j * gamma * u``, where ``u`` is the unit vector along
    ``resc_dir_z`` and ``gamma`` is the root of the energy mismatch closest to zero.
    The energy mismatch is the change of the classical energy under the shift plus
    ``eigval_diff``, which is zero when the shift conserves the energy. The root is
    found by ``calc_nearest_root``. If there is none, or its refinement does not
    converge, a zero shift is returned and the hop is frustrated. If ``resc_dir_z`` is
    zero, a zero shift is returned, and the hop is successful only if the magnitude of
    ``eigval_diff`` is less than the threshold. The function is called for a single
    trajectory; shapes use C for the number of classical coordinates.

    .. rubric:: Keyword Arguments

    z : ndarray of shape (C,), dtype=complex128
        Current complex-valued classical coordinates.
    resc_dir_z : ndarray of shape (C,), dtype=complex128
        Rescaling direction.
    eigval_diff : float
        Difference in eigenvalues between the initial and final states,
        ``e_final - e_initial``.

    .. rubric:: Model Constants

    numerical_fssh_hop_gamma_initial : float, default: 1e-6
        First value of ``gamma`` on each side of zero, from which the search doubles.
    numerical_fssh_hop_max_doublings : int, default: 60
        Maximum number of doublings of ``gamma`` on each side of zero.
    numerical_fssh_hop_threshold : float, default: 1e-6
        Energy threshold for convergence.

    .. rubric:: Returns

    shift : ndarray of shape (C,), dtype=complex128
        The shift to apply to the complex-valued classical coordinates to conserve
        energy.
    hop_successful : bool
        Whether the hop was successful (i.e., energy conservation was achieved).
    """
    z = kwargs["z"]
    resc_dir_z = kwargs["resc_dir_z"]
    eigval_diff = kwargs["eigval_diff"]
    gamma_initial = model.constants.get("numerical_fssh_hop_gamma_initial", 1e-6)
    max_doublings = model.constants.get("numerical_fssh_hop_max_doublings", 60)
    thresh = model.constants.get("numerical_fssh_hop_threshold", 1e-6)
    h_c, _ = model.get("h_c")
    init_energy = h_c(model, parameters, z=np.array([z]), batch_size=1)[0]
    resc_dir_z_norm = np.linalg.norm(resc_dir_z)
    if resc_dir_z_norm == 0:
        return np.zeros_like(z), np.abs(eigval_diff) < thresh
    # A unit direction makes gamma independent of the length of resc_dir_z.
    unit_dir = resc_dir_z / resc_dir_z_norm

    def calc_energy_mismatch(gamma):
        energy = h_c(
            model, parameters, z=np.array([z - 1j * gamma * unit_dir]), batch_size=1
        )[0]
        return np.real(energy - init_energy + eigval_diff)

    mismatch_zero = calc_energy_mismatch(0.0)
    if np.abs(mismatch_zero) < thresh:
        return np.zeros_like(z), True
    gamma, converged = calc_nearest_root(
        calc_energy_mismatch,
        mismatch_zero,
        (1.0, -1.0),
        gamma_initial,
        max_doublings,
        thresh,
    )
    if not converged:
        return np.zeros_like(z), False
    return -1j * gamma * unit_dir, True


def calc_nearest_root(
    calc_value, value_zero, sides, gamma_initial, max_doublings, threshold
):
    """
    Find the root of a real function of ``gamma`` closest to zero.

    On each given side of zero, the search steps outward geometrically, to
    ``gamma_initial``, ``2 * gamma_initial``, ``4 * gamma_initial``, and so on, up to
    ``max_doublings`` doublings. It stops at the first point where the function has the
    opposite sign to its value at zero, or where its magnitude stops decreasing:

    * At a sign change, the root is refined by bisection.
    * Where the magnitude stops decreasing, the magnitude has a minimum between the
      last three points, which is located by golden-section search. If the function
      changes sign there, the root between zero and the minimum is refined by
      bisection. If the magnitude at the minimum is less than ``threshold``, the
      minimum is taken as the root. Otherwise, the side has no root.
    * If the magnitude increases already at ``gamma_initial``, the side has no root.

    Of the roots found on the given sides, the one closest to zero is returned. The
    search does not use a fixed scale of ``gamma``, so roots close to zero and roots
    close to each other are found as well, within the assumption given in the Notes.

    Parameters
    ----------
    calc_value : callable
        Real function of ``gamma``, called only for nonzero ``gamma``.
    value_zero : float
        Value of the function at zero.
    sides : tuple of float
        Sides of zero to search, ``1.0`` for positive and ``-1.0`` for negative
        ``gamma``.
    gamma_initial : float
        First value of ``gamma`` on each side, from which the search doubles.
    max_doublings : int
        Maximum number of doublings of ``gamma`` on each side.
    threshold : float
        Threshold on the magnitude of the function for convergence.

    Returns
    -------
    gamma : float or None
        Root closest to zero, or the last bisection midpoint if its bisection does not
        converge; ``None`` if no root is found.
    converged : bool
        Whether the magnitude of the function at ``gamma`` is less than ``threshold``.

    Notes
    -----
    * On each side, the function is assumed to approach zero before turning away from
      it, as for the energy mismatch of a hop in ``numerical_fssh_hop`` and the energy
      change divided by ``gamma`` in ``physical_momentum_reversal_numerical``. A root
      that is preceded by a stretch in which the magnitude of the function increases
      is not found.
    """
    if value_zero == 0:
        return 0.0, True
    roots = []
    for side in sides:
        gamma, converged = _calc_nearest_root_side(
            calc_value, value_zero, side, gamma_initial, max_doublings, threshold
        )
        if gamma is not None:
            roots.append((np.abs(gamma), gamma, converged))
    if not roots:
        return None, False
    _, gamma, converged = min(roots, key=lambda root: root[0])
    return gamma, converged


def _calc_nearest_root_side(
    calc_value, value_zero, side, gamma_initial, max_doublings, threshold
):
    """
    Finds the root of ``calc_value`` closest to zero on one side, for
    ``calc_nearest_root``. Works with the function times the sign of its value at zero,
    which is positive until the first root.
    """
    sign_zero = np.sign(value_zero)
    # Points (gamma, sign_zero * value) along the side, starting at zero.
    points = [(0.0, np.abs(value_zero))]
    gamma = side * gamma_initial
    for _ in range(max_doublings + 1):
        value = sign_zero * calc_value(gamma)
        if value <= 0:
            return _bisect_root(calc_value, sign_zero, points[-1][0], gamma, threshold)
        if value >= points[-1][1]:
            if len(points) == 1:
                return None, False
            return _calc_root_via_minimum(
                calc_value, sign_zero, points[-2], points[-1], (gamma, value), threshold
            )
        points.append((gamma, value))
        gamma = 2.0 * gamma
    return None, False


def _calc_root_via_minimum(calc_value, sign_zero, point_a, point_b, point_c, threshold):
    """
    Golden-section search for the minimum of ``sign_zero * calc_value`` between
    ``point_a`` and ``point_c``, given the point ``point_b`` between them with the
    lowest value, for ``calc_nearest_root``. Points are tuples
    ``(gamma, sign_zero * value)``, with ``point_a`` closest to zero. Stops at the first
    point with a nonpositive value and refines the root between it and zero by
    bisection.
    """
    golden_fraction = (3.0 - np.sqrt(5.0)) / 2.0
    evaluated = [point_a, point_b, point_c]
    # 100 golden-section steps reduce the bracket far below any useful resolution.
    for _ in range(100):
        gamma_a, gamma_b, gamma_c = point_a[0], point_b[0], point_c[0]
        if np.abs(gamma_c - gamma_a) <= 1e-12 * np.abs(gamma_c):
            break
        if np.abs(gamma_c - gamma_b) > np.abs(gamma_b - gamma_a):
            gamma_new = gamma_b + golden_fraction * (gamma_c - gamma_b)
        else:
            gamma_new = gamma_b - golden_fraction * (gamma_b - gamma_a)
        value_new = sign_zero * calc_value(gamma_new)
        if value_new <= 0:
            # The root closest to zero lies between this point and the farthest point
            # from zero that is closer to zero and has a positive value.
            gamma_lower = max(
                (
                    point[0]
                    for point in evaluated
                    if np.abs(point[0]) < np.abs(gamma_new)
                ),
                key=np.abs,
            )
            return _bisect_root(
                calc_value, sign_zero, gamma_lower, gamma_new, threshold
            )
        evaluated.append((gamma_new, value_new))
        beyond_b = np.abs(gamma_new) > np.abs(gamma_b)
        if value_new < point_b[1]:
            if beyond_b:
                point_a = point_b
            else:
                point_c = point_b
            point_b = (gamma_new, value_new)
        elif beyond_b:
            point_c = (gamma_new, value_new)
        else:
            point_a = (gamma_new, value_new)
    if point_b[1] < threshold:
        return point_b[0], True
    return None, False


def _bisect_root(calc_value, sign_zero, gamma_lower, gamma_upper, threshold):
    """
    Bisection for the root of ``calc_value`` between ``gamma_lower``, where it has the
    sign ``sign_zero``, and ``gamma_upper``, where it does not, for
    ``calc_nearest_root``.
    """
    gamma_mid = gamma_upper
    # 100 bisection steps reduce the bracket far below any useful resolution.
    for _ in range(100):
        gamma_mid = 0.5 * (gamma_lower + gamma_upper)
        value_mid = calc_value(gamma_mid)
        if np.abs(value_mid) < threshold:
            return gamma_mid, True
        if np.sign(value_mid) == sign_zero:
            gamma_lower = gamma_mid
        else:
            gamma_upper = gamma_mid
    return gamma_mid, False


def calc_alpha_n_mash(num_states):
    """
    Population-estimator scaling factor of multi-state MASH,

    .. math::
        \\alpha_N = (N - 1) / (H_N - 1),

    where :math:`H_N` is the N-th harmonic number (Runeson and Manolopoulos, J. Chem. Phys.
    159, 094115, 2023).

    .. rubric:: Args

    num_states : int
        Number of quantum states N.

    .. rubric:: Returns

    alpha_n : float
        Scaling factor :math:`\\alpha_N`.
    """
    harmonic_number = np.sum(1.0 / np.arange(1, num_states + 1))
    return (num_states - 1) / (harmonic_number - 1)


def slice_parameters(parameters: dict, inds, batch_size: int):
    """
    Slice the per-trajectory entries of ``parameters`` to ``inds``, passing all other
    entries through unchanged, so a Task executed on a sub-batch of trajectories sees
    exactly the parameters a full-batch Task would.

    An entry is treated as per-trajectory when it is an ndarray whose leading axis has
    length ``batch_size``. An entry that is not per-trajectory but whose leading axis
    happens to have length ``batch_size`` is sliced as well.

    .. rubric:: Args

    parameters : dict
        Parameters object of the full batch.
    inds : ndarray
        Indices of the trajectories to keep.
    batch_size : int
        Number of trajectories in the full batch.

    .. rubric:: Returns

    sliced : dict
        Parameters object restricted to ``inds``.
    """
    sliced = {}
    for key, value in parameters.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == batch_size
        ):
            sliced[key] = value[inds]
        else:
            sliced[key] = value
    return sliced


# Multi-state mapping approach to surface hopping (MSMASH).
#
# Portions of the MSMASH code below are derived from multimash
# (https://github.com/jruneson/multimash, commit 4756b9e), Copyright (c) 2023
# Johan Runeson, used under the MIT License; see
# src/qclab/algorithms/licenses/multimash_LICENSE.


def propagate_wf_db_adb(
    wf_db: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray, dt: np.ndarray
):
    """
    Propagate the diabatic wavefunction by dt, mirroring the native
    update_wf_db_propagator arithmetic at the array level: build the diagonal adiabatic
    propagator exp(-i E dt), rotate it to the diabatic basis (transform_mat), and apply
    it (batch_matvec). dt is a per-trajectory vector; the step size is the caller's
    choice of dt, not a property of this unit. It is a method-specific per-trajectory-dt
    helper, not a replacement for update_wf_db_propagator.

    This function can be removed if the shared RK4 and wavefunction-propagation Tasks
    (update_z_rk4_k123, update_z_rk4_k4, update_wf_db_propagator) gain support for a
    per-trajectory time step, so that _rk4_step can be composed from shared Tasks.
    """
    batch_size, num_quantum_states = wf_db.shape
    prop_adb = np.zeros(
        (batch_size, num_quantum_states, num_quantum_states), dtype=complex
    )
    idx = np.arange(num_quantum_states)
    prop_adb[:, idx, idx] = np.exp(-1j * eigvals * dt[:, None])
    prop_db = transform_mat(prop_adb, eigvecs, adb_to_db=True)
    return batch_matvec(prop_db, wf_db)


def calc_hop_dest_msmash(wf_adb, act_surf_ind):
    """
    Per-trajectory hop destination: the second-most-populated adiabatic surface, i.e.
    the argmax adiabatic population excluding the current active surface.
    """
    idx = np.arange(len(act_surf_ind))
    nonactive_populations = (np.abs(wf_adb) ** 2).copy()
    nonactive_populations[idx, act_surf_ind] = -np.inf
    return np.argmax(nonactive_populations, axis=1).astype(int)


def sample_wf_db_focused_msmash(num_states, wf_db, seeds):
    """
    Default focused (spin-mapping) sampling of the diabatic wavefunction. The
    populated state is the largest-population component of the supplied wavefunction;
    its squared magnitudes are set so the MASH population estimator gives
    :math:`\\Phi_{\\mathrm{init}}(0) = 1`,

    .. math::
        |c_{\\mathrm{init}}|^2 = 1/N + (N - 1) / (\\alpha_N N), \\qquad
        |c_{\\zeta}|^2 = (\\alpha_N - 1) / (\\alpha_N N)\\ \\ (\\zeta \\neq \\mathrm{init}),

    with :math:`\\alpha_N = (N - 1) / (H_N - 1)`; each state then gets a random global
    phase drawn from the per-trajectory seed. A Model whose populated initial state is
    not the largest-population component of the supplied wavefunction overrides this by
    providing the ``sample_wf_db_msmash`` Ingredient (see initialize_wf_db_msmash).

    Parameters
    ----------
    num_states : int
        Number of quantum states N.
    wf_db : ndarray of shape (B, N), dtype=complex128
        Supplied diabatic state; its first-trajectory argmax sets the focused index.
    seeds : ndarray of shape (B,), dtype=int
        Per-trajectory random seeds.

    Returns
    -------
    ndarray of shape (B, N), dtype=complex128
        Sampled focused diabatic wavefunction.

    Notes
    -----
    * B = number of trajectories (rows of the supplied wavefunction)
    * N = num_states (number of quantum states)
    """
    wf_db = np.asarray(wf_db).reshape(-1, num_states)
    batch_size = wf_db.shape[0]
    init_state_ind = int(np.argmax(np.abs(wf_db[0]) ** 2))
    alpha_n = calc_alpha_n_mash(num_states)
    sq_magnitudes = np.full(num_states, (alpha_n - 1) / (alpha_n * num_states))
    sq_magnitudes[init_state_ind] = 1.0 / num_states + (num_states - 1) / (
        alpha_n * num_states
    )
    sampled_wf_db = np.empty((batch_size, num_states), dtype=complex)
    for traj_ind in range(batch_size):
        rng = np.random.default_rng(int(seeds[traj_ind]))
        phase = rng.uniform(0.0, 2.0 * np.pi, num_states)
        sampled_wf_db[traj_ind] = (np.cos(phase) + 1j * np.sin(phase)) * np.sqrt(
            sq_magnitudes
        )
    return sampled_wf_db


def calc_dh_qc_dzc_matrices_physical_z_conj(dh_qc_dzc, num_trajs_used):
    """
    Arrange the gradient of the quantum–classical interaction Hamiltonian, and its
    Hermitian conjugate in the quantum indices, as dense matrices whose rows are
    indexed by the classical coordinates. There is one column for each (trajectory,
    row, column) index triple of the stored entries of the gradient, and for each of
    their transposes, so that the columns of the two matrices correspond.

    Parameters
    ----------
    dh_qc_dzc : tuple
        Gradient of the quantum–classical interaction Hamiltonian in sparse format,
        ``(inds, mels, shape)``.
    num_trajs_used : int
        Number of trajectories, counted from the first, whose gradients are included.

    Returns
    -------
    dh_qc_dzc_matrix : ndarray of shape (C, M), dtype=complex128
        Gradient of the quantum–classical interaction Hamiltonian.
    dh_qc_dzc_dagger_matrix : ndarray of shape (C, M), dtype=complex128
        Hermitian conjugate of the gradient in the quantum indices.

    Notes
    -----
    * C = number of classical coordinates
    * M = number of distinct (trajectory, row, column) index triples of the stored
      entries of the gradient, together with their transposes
    """
    inds, mels, shape = dh_qc_dzc
    num_classical_coordinates, num_quantum_states = shape[1], shape[2]
    keep = inds[0] < num_trajs_used
    traj_ind, coord_ind = inds[0][keep], inds[1][keep]
    row_ind, col_ind = inds[2][keep], inds[3][keep]
    mels = mels[keep]
    # Label each (trajectory, row, column) triple, and its transpose.
    label = (traj_ind * num_quantum_states + row_ind) * num_quantum_states + col_ind
    label_dagger = (
        traj_ind * num_quantum_states + col_ind
    ) * num_quantum_states + row_ind
    unique_label = np.unique(np.concatenate((label, label_dagger)))
    dh_qc_dzc_matrix = np.zeros(
        (num_classical_coordinates, len(unique_label)), dtype=complex
    )
    dh_qc_dzc_dagger_matrix = np.zeros_like(dh_qc_dzc_matrix)
    dh_qc_dzc_matrix[coord_ind, np.searchsorted(unique_label, label)] = mels
    dh_qc_dzc_dagger_matrix[coord_ind, np.searchsorted(unique_label, label_dagger)] = (
        np.conj(mels)
    )
    return dh_qc_dzc_matrix, dh_qc_dzc_dagger_matrix


def calc_determined_directions_physical_z_conj(
    dh_qc_dzc_matrix, dh_qc_dzc_dagger_matrix
):
    """
    Find the directions of the classical coordinates on which the relation
    :math:`G = S \\tilde{G}` determines the physical z-conjugation matrix
    :math:`S`, where ``dh_qc_dzc_matrix`` and ``dh_qc_dzc_dagger_matrix`` hold
    :math:`G` and :math:`\\tilde{G}`. The columns of :math:`\\tilde{G}` hold the
    Hermitian conjugate, in the quantum indices, of the gradient in :math:`G`.

    Parameters
    ----------
    dh_qc_dzc_matrix : ndarray of shape (C, M), dtype=complex128
        Gradient of the quantum–classical interaction Hamiltonian, as returned by
        ``calc_dh_qc_dzc_matrices_physical_z_conj``.
    dh_qc_dzc_dagger_matrix : ndarray of shape (C, M), dtype=complex128
        Hermitian conjugate of the gradient in the quantum indices, as returned by
        ``calc_dh_qc_dzc_matrices_physical_z_conj``.

    Returns
    -------
    determined_directions : ndarray of shape (C, R), dtype=complex128
        Orthonormal directions on which :math:`S` is determined.
    image : ndarray of shape (C, R), dtype=complex128
        Images of the columns of ``determined_directions`` under :math:`S`.

    Notes
    -----
    * C = number of classical coordinates
    * M = number of columns of the gradient matrices, see
      ``calc_dh_qc_dzc_matrices_physical_z_conj``
    * R = number of directions on which :math:`S` is determined
    """
    left, singvals, right_dagger = np.linalg.svd(
        dh_qc_dzc_dagger_matrix, full_matrices=False
    )
    num_determined = 0
    # Singular values are compared relative to the largest one.
    if len(singvals) > 0 and singvals[0] > 0:
        num_determined = int(
            np.sum(
                singvals
                > numerical_constants.PHYSICAL_Z_CONJ_RANK_THRESHOLD * singvals[0]
            )
        )
    determined_directions = left[:, :num_determined]
    image = (dh_qc_dzc_matrix @ right_dagger[:num_determined].conj().T) / singvals[
        :num_determined
    ]
    return determined_directions, image


def calc_h_c_hessian(classical_force_probe, delta_z):
    """
    Calculate the Hessian of the classical Hamiltonian,
    :math:`W_{\\xi\\xi'} = \\frac{\\partial^{2} H_{\\mathrm{c}}}{\\partial z_{\\xi}^{*} \\partial z_{\\xi'}}`,
    at the centre of the probe displacements, by central differences of the classical
    gradient evaluated at the probe coordinates. For the probe coordinates written by
    ``initialize_z_probe``, the centre is :math:`z = 0`.

    Parameters
    ----------
    classical_force_probe : ndarray of shape (4C, C), dtype=complex128
        Gradient of the classical Hamiltonian at the probe coordinates written by
        ``initialize_z_probe``. The rows form four blocks of C, in which each
        classical coordinate in turn is displaced from zero by :math:`+\\delta`,
        :math:`-\\delta`, :math:`+i\\delta` and :math:`-i\\delta`, respectively.
    delta_z : float
        Displacement of the probe coordinates from zero.

    Returns
    -------
    h_c_hessian : ndarray of shape (C, C), dtype=complex128
        Hessian of the classical Hamiltonian.

    Notes
    -----
    * C = number of classical coordinates
    """
    plus, minus, plus_imag, minus_imag = np.split(classical_force_probe, 4)
    # Real displacements of z_xi give 2 delta (d/dz_xi + d/dz_xi^*) of the gradient and
    # imaginary displacements give 2i delta (d/dz_xi - d/dz_xi^*); combining them
    # leaves 4 delta d/dz_xi. Rows are indexed by xi, hence the transpose.
    h_c_hessian = ((plus - minus) - 1j * (plus_imag - minus_imag)).T / (4.0 * delta_z)
    # The Hessian is Hermitian; symmetrizing removes finite-difference noise.
    return 0.5 * (h_c_hessian + h_c_hessian.conj().T)


def calc_physical_z_conj_matrix(determined_directions, image, h_c_hessian):
    """
    Construct the symmetric unitary matrix :math:`S` that maps the columns of
    ``determined_directions`` onto the columns of ``image``.

    Directions of the classical coordinates outside of ``determined_directions`` are
    resolved by requiring :math:`W S = S W^{*}`, where :math:`W` is the Hessian of the
    classical Hamiltonian. Any remaining freedom is fixed arbitrarily. If
    ``h_c_hessian`` is None, all directions outside of ``determined_directions`` are
    fixed arbitrarily.

    Parameters
    ----------
    determined_directions : ndarray of shape (C, R), dtype=complex128
        Orthonormal directions on which :math:`S` is determined, as returned by
        ``calc_determined_directions_physical_z_conj``.
    image : ndarray of shape (C, R), dtype=complex128
        Images of the columns of ``determined_directions`` under :math:`S`.
    h_c_hessian : ndarray of shape (C, C), dtype=complex128 | None
        Hessian of the classical Hamiltonian, as returned by ``calc_h_c_hessian``.

    Returns
    -------
    physical_z_conj_matrix : ndarray of shape (C, C), dtype=complex128
        Physical z-conjugation matrix :math:`S`.
    num_arbitrary : int
        Number of directions fixed arbitrarily.

    Notes
    -----
    * C = number of classical coordinates
    * R = number of directions on which :math:`S` is determined
    """
    num_classical_coordinates, num_determined = determined_directions.shape
    if num_determined == num_classical_coordinates:
        return image @ determined_directions.conj().T, 0
    # Otherwise, work within the degenerate subspaces of the classical Hessian.
    eigvecs = np.eye(num_classical_coordinates, dtype=complex)
    boundaries = []
    if h_c_hessian is not None:
        eigvals, eigvecs = np.linalg.eigh(h_c_hessian)
        scale = max(np.max(np.abs(eigvals)), np.finfo(float).tiny)
        boundaries = (
            np.flatnonzero(
                np.diff(eigvals)
                > numerical_constants.PHYSICAL_Z_CONJ_DEGENERACY_THRESHOLD * scale
            )
            + 1
        )
    physical_z_conj_matrix = np.zeros(
        (num_classical_coordinates, num_classical_coordinates), dtype=complex
    )
    num_arbitrary = 0
    for subspace_ind in np.split(np.arange(num_classical_coordinates), boundaries):
        subspace = eigvecs[:, subspace_ind]
        # The matrix transforms as V S V^T, hence the unconjugated transposes.
        determined_directions_sub = subspace.T @ determined_directions
        image_sub = subspace.conj().T @ image
        left_sub, singvals_sub, right_dagger_sub = np.linalg.svd(
            determined_directions_sub
        )
        # As the determined directions are orthonormal, these singular values are at
        # most 1 and are compared to the threshold directly.
        num_determined_sub = int(
            np.sum(singvals_sub > numerical_constants.PHYSICAL_Z_CONJ_RANK_THRESHOLD)
        )
        physical_z_conj_matrix_sub = (
            (image_sub @ right_dagger_sub[:num_determined_sub].conj().T)
            / singvals_sub[:num_determined_sub]
        ) @ left_sub[:, :num_determined_sub].conj().T
        # Complete the matrix on the directions left undetermined. The unconjugated
        # transpose keeps the completion symmetric and unitary.
        remainder = left_sub[:, num_determined_sub:].conj()
        physical_z_conj_matrix_sub = (
            physical_z_conj_matrix_sub + remainder @ remainder.T
        )
        num_arbitrary += remainder.shape[1]
        physical_z_conj_matrix += subspace @ physical_z_conj_matrix_sub @ subspace.T
    return physical_z_conj_matrix, num_arbitrary


def calc_symmetric_unitary_error(matrix):
    """
    Calculate the largest deviation of a matrix from being symmetric and unitary,
    as the largest absolute matrix element of :math:`A - A^{T}` and of
    :math:`A A^{\\dagger} - 1`.

    Parameters
    ----------
    matrix : ndarray of shape (D, D), dtype=complex128
        Matrix :math:`A` to be checked.

    Returns
    -------
    error : float
        Largest deviation.

    Notes
    -----
    * D = dimension of the matrix
    """
    identity = np.eye(len(matrix))
    return max(
        np.max(np.abs(matrix - matrix.T)),
        np.max(np.abs(matrix @ matrix.conj().T - identity)),
    )


def calc_physical_z_conj_matrix_error(
    sim,
    parameters,
    z,
    physical_z_conj_matrix,
    dh_qc_dzc_matrix,
    dh_qc_dzc_dagger_matrix,
):
    """
    Calculate the largest violation of the defining properties of the physical
    z-conjugation matrix: its symmetry and unitarity, the gradient relation
    :math:`G = S \\tilde{G}` (see ``calc_determined_directions_physical_z_conj``),
    and the invariance of the classical and quantum–classical interaction
    Hamiltonians under :math:`z \\rightarrow S z^{*}`. Symmetry and unitarity are
    measured in absolute terms, since the elements of a unitary matrix are at most 1
    in magnitude; the other properties are measured relative to the largest gradient
    or Hamiltonian matrix element.

    Parameters
    ----------
    sim : Simulation
        Simulation object, whose Model supplies the ``h_c`` and ``h_qc`` Ingredients.
    parameters : dict
        Parameters object.
    z : ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates at which the Hamiltonians are evaluated.
    physical_z_conj_matrix : ndarray of shape (C, C), dtype=complex128
        Physical z-conjugation matrix to be validated.
    dh_qc_dzc_matrix : ndarray of shape (C, M), dtype=complex128
        Gradient of the quantum–classical interaction Hamiltonian, as returned by
        ``calc_dh_qc_dzc_matrices_physical_z_conj``.
    dh_qc_dzc_dagger_matrix : ndarray of shape (C, M), dtype=complex128
        Hermitian conjugate of the gradient in the quantum indices, as returned by
        ``calc_dh_qc_dzc_matrices_physical_z_conj``.

    Returns
    -------
    error : float
        Largest violation.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = number of classical coordinates
    * M = number of columns of the gradient matrices, see
      ``calc_dh_qc_dzc_matrices_physical_z_conj``
    """
    errors = [calc_symmetric_unitary_error(physical_z_conj_matrix)]
    if dh_qc_dzc_matrix.size > 0 and np.max(np.abs(dh_qc_dzc_matrix)) > 0:
        errors.append(
            np.max(
                np.abs(
                    dh_qc_dzc_matrix - physical_z_conj_matrix @ dh_qc_dzc_dagger_matrix
                )
            )
            / np.max(np.abs(dh_qc_dzc_matrix))
        )
    # Apply z -> S z^*; the transpose arises because each row of z is one trajectory.
    physical_z_conj = np.conj(z) @ physical_z_conj_matrix.T
    for ingredient_name in ("h_c", "h_qc"):
        ingredient, _ = sim.model.get(ingredient_name)
        value = ingredient(sim.model, parameters, z=z)
        value_physical_z_conj = ingredient(sim.model, parameters, z=physical_z_conj)
        if np.max(np.abs(value)) > 0:
            errors.append(
                np.max(np.abs(value_physical_z_conj - value)) / np.max(np.abs(value))
            )
    return max(errors)


def calc_z_physical_position_component(vec, physical_z_conj_matrix):
    """
    Calculate the physical position component of a vector in the space of the
    complex-valued classical coordinates, :math:`(v + S v^{*})/2`, where :math:`S` is
    the physical z-conjugation matrix.

    Parameters
    ----------
    vec : ndarray of shape (..., C), dtype=complex128
        Vector :math:`v`, such as the classical coordinates or a direction in their
        space.
    physical_z_conj_matrix : ndarray of shape (C, C), dtype=complex128
        Physical z-conjugation matrix :math:`S`.

    Returns
    -------
    physical_position_component : ndarray of shape (..., C), dtype=complex128
        Physical position component of the vector.

    Notes
    -----
    * C = number of classical coordinates
    * For a Model defined in a physical basis, :math:`S` is the identity and the
      physical position component equals the real part.
    """
    # Apply v -> S v^*; the transpose arises because the last index of vec is the
    # classical index.
    return 0.5 * (vec + np.conj(vec) @ physical_z_conj_matrix.T)


def calc_z_physical_momentum_component(vec, physical_z_conj_matrix):
    """
    Calculate the physical momentum component of a vector in the space of the
    complex-valued classical coordinates, :math:`(v - S v^{*})/2`, where :math:`S` is
    the physical z-conjugation matrix.

    Parameters
    ----------
    vec : ndarray of shape (..., C), dtype=complex128
        Vector :math:`v`, such as the classical coordinates or a direction in their
        space.
    physical_z_conj_matrix : ndarray of shape (C, C), dtype=complex128
        Physical z-conjugation matrix :math:`S`.

    Returns
    -------
    physical_momentum_component : ndarray of shape (..., C), dtype=complex128
        Physical momentum component of the vector.

    Notes
    -----
    * C = number of classical coordinates
    * For a Model defined in a physical basis, :math:`S` is the identity and the
      physical momentum component equals :math:`i` times the imaginary part.
    """
    # Apply v -> S v^*; the transpose arises because the last index of vec is the
    # classical index.
    return 0.5 * (vec - np.conj(vec) @ physical_z_conj_matrix.T)


def calc_z_physical_momentum_reversal(
    model, parameters, z, reversal_dir_z, physical_z_conj_matrix
):
    """
    Reverse the physical momentum along a direction, leaving the physical positions
    and the kinetic energy unchanged.

    With :math:`\\mathcal{P}z = (z + S z^{*})/2` the physical position component of
    :math:`z` and :math:`z - \\mathcal{P}z` its physical momentum component, the
    latter is reflected along :math:`u = i \\mathcal{P}d`, where :math:`d` is the
    direction, in the inner product defined by the kinetic energy :math:`T`,

    .. math::

        z - \\mathcal{P}z \\rightarrow z - \\mathcal{P}z
        - 2 \\frac{T(z - \\mathcal{P}z, u)}{T(u, u)} u,

    where :math:`T(\\cdot, \\cdot)` is the symmetric bilinear form of the kinetic
    energy. It is obtained by polarization,
    :math:`T(v, w) = [T(v + w) - T(v - w)]/4`, from the kinetic energy
    :math:`T(v) = T(v, v) = [H_{\\mathrm{c}}(\\mathcal{P}z + v) +
    H_{\\mathrm{c}}(\\mathcal{P}z - v)]/2 - H_{\\mathrm{c}}(\\mathcal{P}z)`.

    Parameters
    ----------
    model : Model
        Model object, whose ``h_c`` Ingredient supplies the classical Hamiltonian.
    parameters : dict
        Parameters object of a single trajectory.
    z : ndarray of shape (C,), dtype=complex128
        Complex-valued classical coordinates of a single trajectory.
    reversal_dir_z : ndarray of shape (C,), dtype=complex128
        Direction along which the physical momentum is reversed. Only its physical
        position component is used.
    physical_z_conj_matrix : ndarray of shape (C, C), dtype=complex128
        Physical z-conjugation matrix :math:`S`.

    Returns
    -------
    z_reversed : ndarray of shape (C,), dtype=complex128
        Complex-valued classical coordinates after the reversal.

    Notes
    -----
    * C = number of classical coordinates
    * The classical Hamiltonian is assumed to be the sum of a kinetic energy that is
      quadratic in the physical momenta and a potential energy that depends on the
      physical positions only.
    * With a single classical coordinate, the full physical momentum is reversed.
    """
    z_physical_position = calc_z_physical_position_component(z, physical_z_conj_matrix)
    # Write the physical momentum component as i times a vector with the symmetry of
    # a physical position.
    physical_momentum = (
        calc_z_physical_momentum_component(z, physical_z_conj_matrix) / 1j
    )
    if len(z) == 1:
        return z_physical_position - 1j * physical_momentum
    physical_momentum_norm = np.linalg.norm(physical_momentum)
    reversal_dir_z_physical_position = calc_z_physical_position_component(
        reversal_dir_z, physical_z_conj_matrix
    )
    reversal_dir_z_norm = np.linalg.norm(reversal_dir_z_physical_position)
    if physical_momentum_norm == 0 or reversal_dir_z_norm == 0:
        return z.copy()
    h_c, _ = model.get("h_c")

    def calc_h_c(z_eval):
        return np.real(h_c(model, parameters, z=z_eval[None], batch_size=1)[0])

    h_c_physical_position = calc_h_c(z_physical_position)

    def calc_kinetic_energy(vec):
        # The even part in vec cancels any residual physical momentum in
        # z_physical_position.
        return (
            0.5 * calc_h_c(z_physical_position + 1j * vec)
            + 0.5 * calc_h_c(z_physical_position - 1j * vec)
            - h_c_physical_position
        )

    # Unit vectors keep the polarization identity well conditioned.
    unit_physical_momentum = physical_momentum / physical_momentum_norm
    unit_reversal_dir_z = reversal_dir_z_physical_position / reversal_dir_z_norm
    kinetic_energy_cross = 0.25 * (
        calc_kinetic_energy(unit_physical_momentum + unit_reversal_dir_z)
        - calc_kinetic_energy(unit_physical_momentum - unit_reversal_dir_z)
    )
    kinetic_energy_direction = calc_kinetic_energy(unit_reversal_dir_z)
    physical_momentum_reversed = (
        physical_momentum
        - 2.0
        * physical_momentum_norm
        * kinetic_energy_cross
        / kinetic_energy_direction
        * unit_reversal_dir_z
    )
    return z_physical_position + 1j * physical_momentum_reversed
