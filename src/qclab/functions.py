"""
This module contains functions used in QC Lab. This includes
functions that are used by tasks, ingredients, and the
dynamics drivers.
"""

import logging
import functools
import numpy as np
from qclab.utils import njit
import qclab.numerical_constants as numerical_constants

logger = logging.getLogger(__name__)


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
    if dp_present:
        return 1j * np.sqrt(m * h / 2) * dp
    if dq_present:
        return np.sqrt(0.5 / (m * h)) * dq.astype(np.complex128)
    if dq_present and dp_present:
        return np.sqrt(0.5 / (m * h)) * dq + 1j * np.sqrt(m * h / 2) * dp
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
    Low-level function to generate the diagonal linear quantum-classical Hamiltonian.

    :math:`H_{nm} = \\delta_{nm}\\sum_{j} \\gamma_{nj} (z_{j} + z_{j}^*)`

    .. rubric:: Args
    z : ndarray
        Complex coordinates.
    gamma : ndarray
        Classical coordinate coupling strengths.

    .. rubric:: Returns
    h_qc : ndarray
        Diagonal linear quantum-classical Hamiltonian.
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
    couplings, i.e., calculates the phase-factors :math:`u^{q}_{i}` and :math:`u^{p}_{i}`
    such that :math:`d_{ij}^{q}u_{i}^{q*}u_{j}^{q}` and :math:`d_{ij}^{p}u_{i}^{p*}u_{j}^{p}`
    are real-valued.

    It does this by calculating the derivative couplings analytically. In the event of
    degenerate eigenvalues, an error is logged and a small offset is added to the energy
    differences.

    .. rubric:: Args
    sim: Simulation
        Simulation object.
    dh_qc_dzc : tuple
        Sparse representation of the derivative of the quantum-classical Hamiltonian
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
    an iterative numerical method. The coordinate following the hop is ``z + shift``.

    The algorithm is as follows:
    1. Calculate the initial energy using the Hamiltonian function at the current ``z``.
    2. Define a grid from ``-gamma_range`` to ``+gamma_range`` with ``num_points`` points
       uniformly spaced.
    3. Calculate the energy at each point in the grid using the Hamiltonian function.
    4. Find the point in the grid that minimizes the difference between the energy
       difference and the calculated energy difference.
    5. Recenter the grid around the minimum point found in step 4, reduce the
       ``gamma_range`` by half, and repeat steps 3-5 until either the minimum energy
       difference is less than the threshold or the maximum number of iterations
       is reached.
    6. If the minimum energy difference is less than the threshold, return the
       corresponding shift. Otherwise, return a zero shift and indicate that the
       hop was frustrated.

    .. rubric:: Keyword Arguments
    z : ndarray
        Current complex coordinate.
    resc_dir_z : float
        Rescaling direction.
    eigval_diff : float
        Difference in eigenvalues between the initial and final states, ``e_final - e_initial``.


    .. rubric:: Model Constants
    numerical_fssh_hop_gamma_range : float, default: 5.0
        Initial range (negative to positive) of gamma values to search over.
    numerical_fssh_hop_max_iter : int, default: 20
        Maximum number of iterations to perform.
    numerical_fssh_hop_num_points : int, default: 10
        Number of points to sample in each iteration.
    numerical_fssh_hop_threshold : float, default: 1e-6
        Energy threshold for convergence.

    .. rubric:: Returns
    ndarray
        The shift to apply to the complex coordinate to conserve energy.
    hop_successful : bool
        Whether the hop was successful (i.e., energy conservation was achieved).
    """
    z = kwargs["z"]
    resc_dir_z = kwargs["resc_dir_z"]
    eigval_diff = kwargs["eigval_diff"]
    gamma_range = model.constants.get("numerical_fssh_hop_gamma_range", 5.0)
    max_iter = model.constants.get("numerical_fssh_hop_max_iter", 20)
    num_points = model.constants.get("numerical_fssh_hop_num_points", 10)
    thresh = model.constants.get("numerical_fssh_hop_threshold", 1e-6)
    h_c, _ = model.get("h_c")
    init_energy = h_c(model, parameters, z=np.array([z]), batch_size=1)[0]
    min_gamma = 0.0
    num_iter = 0
    min_energy = 1.0
    while min_energy > thresh and num_iter < max_iter:
        gamma_list = np.linspace(
            min_gamma - gamma_range, min_gamma + gamma_range, num_points
        )
        new_energies = np.abs(
            eigval_diff
            - np.array(
                [
                    init_energy
                    - h_c(
                        model,
                        parameters,
                        z=np.array([z - 1j * gamma * resc_dir_z]),
                        batch_size=1,
                    )[0]
                    for gamma in gamma_list
                ]
            )
        )
        min_gamma = gamma_list[np.argmin(new_energies)]
        min_energy = np.min(new_energies)
        gamma_range = gamma_range * 0.5
        num_iter += 1
    if min_energy > thresh:
        return np.zeros_like(z), False
    return -1j * min_gamma * resc_dir_z, True
