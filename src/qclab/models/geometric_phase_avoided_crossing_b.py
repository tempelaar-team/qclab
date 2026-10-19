"""
This module contains the GeometricPhaseAvoidedCrossingB model class.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients, functions


# Coordinate, row and column indices of the 8 dh_qc_dzc entries per trajectory.
DH_QC_DZC_COORD_INDS = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
DH_QC_DZC_ROW_INDS = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
DH_QC_DZC_COL_INDS = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)


def calc_mixing_angles(model: Model, z: np.ndarray):
    """
    Mixing angle, phase and their position gradients for model B.

    .. math::

        \\theta = -\\pi e^{-(B_x x^2 + B_y y^2)/3}, \\quad
        \\nabla\\theta = \\frac{2\\pi}{3} e^{-(B_x x^2 + B_y y^2)/3}\\,(B_x x, B_y y),
        \\quad \\phi = Wy, \\quad \\nabla\\phi = (0, W)

    Parameters
    ----------
    model : Model
        Model B instance.
    z : ndarray of shape (B, 2), dtype=complex128
        Complex classical coordinates (x, y).

    Returns
    -------
    theta, phi : ndarray of shape (B,), dtype=float64
        Mixing angle and phase.
    dtheta, dphi : ndarray of shape (B, 2), dtype=float64
        Their gradients with respect to (x, y); ``dphi`` is read-only.
    """
    Bx = model.constants.get("Bx")
    By = model.constants.get("By")
    W = model.constants.get("W")
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    q = functions.z_to_q(z, m, h)
    x = q[:, 0]
    y = q[:, 1]
    gaussian = np.exp(-(Bx * x**2 + By * y**2) / 3.0)
    theta = -np.pi * gaussian
    dtheta = np.empty((len(z), 2))
    dtheta[:, 0] = (2.0 * np.pi / 3.0) * Bx * x * gaussian
    dtheta[:, 1] = (2.0 * np.pi / 3.0) * By * y * gaussian
    phi = W * y
    dphi = np.broadcast_to(np.array([0.0, W]), (len(z), 2))
    return theta, dtheta, phi, dphi


def _init_model(model: Model, parameters: dict, **kwargs):
    """
    Initializes the masses, weights and initial wavepacket.

    Model Constants
    ---------------
    mass : float
        Mass of both classical coordinates.
    init_position_x, init_position_y : float
        Center of the initial wavepacket.
    init_momentum_x, init_momentum_y : float
        Mean momentum of the initial wavepacket.

    Notes
    -----
    The wavepacket :math:`e^{i\\mathbf{q}\\cdot\\mathbf{p}_{\\mathrm{init}}}
    e^{-\\vert\\mathbf{q}-\\mathbf{q}_{\\mathrm{init}}\\vert^2}` is a coherent state
    with :math:`\\omega = 2/m`; ``harmonic_frequency`` only sets its width.
    """
    mass = model.constants.get("mass")
    model.constants.num_quantum_states = 2
    model.constants.num_classical_coordinates = 2
    m = np.array([mass, mass])
    w = 2.0 / m
    init_position = np.array(
        [model.constants.get("init_position_x"), model.constants.get("init_position_y")]
    )
    init_momentum = np.array(
        [model.constants.get("init_momentum_x"), model.constants.get("init_momentum_y")]
    )
    model.constants.classical_coordinate_mass = m
    model.constants.classical_coordinate_weight = np.array([1.0, 1.0])
    model.constants.harmonic_frequency = w
    model.constants.coherent_state_displacement = (
        np.sqrt(0.5 * m * w) * init_position
        + 1j * init_momentum / np.sqrt(2.0 * m * w)
    )
    return


def h_qc_geometric_phase_avoided_crossing_b(model: Model, parameters: dict, **kwargs):
    """
    Quantum-classical Hamiltonian of model B.

    .. math::

        H_{\\mathrm{qc}} = A\\begin{pmatrix} -\\cos\\theta &
        \\sin\\theta\\, e^{i\\phi} \\\\ \\sin\\theta\\, e^{-i\\phi} & \\cos\\theta
        \\end{pmatrix}, \\quad
        \\theta = -\\pi e^{-(B_x x^2 + B_y y^2)/3}, \\quad \\phi = Wy

    Keyword Arguments
    -----------------
    z : ndarray of shape (B, 2), dtype=complex128
        Complex classical coordinates (x, y).

    Model Constants
    ---------------
    A : float
        Magnitude of the adiabatic energies.
    Bx, By : float
        Widths of the Gaussian mixing angle in x and y.
    W : float
        Strength of the phase; ``W = 0`` gives a real Hamiltonian.

    Returns
    -------
    h_qc : ndarray of shape (B, 2, 2), dtype=complex128
        Quantum-classical Hamiltonian.
    """
    z = kwargs["z"]
    A = model.constants.get("A")
    theta, _, phi, _ = calc_mixing_angles(model, z)
    h_qc = np.zeros((len(z), 2, 2), dtype=complex)
    h_qc[:, 0, 0] = -A * np.cos(theta)
    h_qc[:, 1, 1] = A * np.cos(theta)
    h_qc[:, 0, 1] = A * np.sin(theta) * np.exp(1j * phi)
    h_qc[:, 1, 0] = np.conj(h_qc[:, 0, 1])
    return h_qc


def dh_qc_dzc_geometric_phase_avoided_crossing_b(
    model: Model, parameters: dict, **kwargs
):
    """
    Gradient of the quantum-classical Hamiltonian with respect to :math:`z^*`.

    .. math::

        \\nabla H_{00} = -\\nabla H_{11} = A\\sin\\theta\\,\\nabla\\theta, \\qquad
        \\nabla H_{01} = \\nabla H_{10}^* = A(\\cos\\theta\\,\\nabla\\theta
        + i\\sin\\theta\\,\\nabla\\phi)e^{i\\phi}, \\qquad
        \\frac{\\partial H}{\\partial z_c^*} = \\frac{1}{\\sqrt{2 m_c h_c}}
        \\frac{\\partial H}{\\partial q_c}

    Keyword Arguments
    -----------------
    z : ndarray of shape (B, 2), dtype=complex128
        Complex classical coordinates.

    Model Constants
    ---------------
    A : float
        Magnitude of the adiabatic energies.

    Returns
    -------
    inds : tuple of four ndarrays of shape (8B,), dtype=int
        Batch, coordinate, row and column indices.
    mels : ndarray of shape (8B,), dtype=complex128
        Matrix elements.
    shape : tuple
        ``(B, 2, 2, 2)``.
    """
    z = kwargs["z"]
    batch_size = len(z)
    A = model.constants.get("A")
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    theta, dtheta, phi, dphi = calc_mixing_angles(model, z)
    sin_theta = np.sin(theta)[:, np.newaxis]
    cos_theta = np.cos(theta)[:, np.newaxis]
    phase = np.exp(1j * phi)[:, np.newaxis]
    # Position derivatives, shape (B, 2).
    dh_00_dq = A * sin_theta * dtheta
    dh_01_dq = A * (cos_theta * dtheta + 1j * sin_theta * dphi) * phase
    dh_00_dzc = functions.dqdp_to_dzc(dh_00_dq, None, m, h)
    dh_01_dzc = functions.dqdp_to_dzc(dh_01_dq, None, m, h)
    # Order per trajectory: x then y, each with elements 00, 01, 10, 11.
    inds = (
        np.repeat(np.arange(batch_size, dtype=int), 8),
        np.tile(DH_QC_DZC_COORD_INDS, batch_size),
        np.tile(DH_QC_DZC_ROW_INDS, batch_size),
        np.tile(DH_QC_DZC_COL_INDS, batch_size),
    )
    mels = np.empty((batch_size, 2, 4), dtype=complex)
    mels[:, :, 0] = dh_00_dzc
    mels[:, :, 1] = dh_01_dzc
    mels[:, :, 2] = np.conj(dh_01_dzc)
    mels[:, :, 3] = -dh_00_dzc
    shape = (batch_size, 2, 2, 2)
    return inds, mels.ravel(), shape


def gauge_field_force_geometric_phase_avoided_crossing_b(
    model: Model, parameters: dict, **kwargs
):
    """
    Pseudomagnetic (Berry) force on adiabatic state ``state_ind``, as a term of the
    quantum-classical force (a gradient with respect to :math:`z^*`).

    .. math::

        \\Omega_0 = -\\Omega_1 = \\frac{1}{2}\\sin\\theta\\,
        (\\partial_x\\theta\\,\\partial_y\\phi - \\partial_y\\theta\\,\\partial_x\\phi),
        \\qquad \\mathbf{F} = \\Omega\\,(v_y, -v_x), \\qquad
        G_c = -\\frac{F_c}{\\sqrt{2 m_c h_c}}

    with :math:`\\mathbf{v} = \\mathbf{p}/m`, so that :math:`\\dot{z} = -iG` gives
    :math:`\\dot{\\mathbf{p}} = \\mathbf{F}`.

    Keyword Arguments
    -----------------
    z : ndarray of shape (B, 2), dtype=complex128
        Complex classical coordinates.
    state_ind : ndarray of shape (B,), dtype=int
        Adiabatic state index: 0 for :math:`E = -A`, 1 for :math:`E = A`.

    Model Constants
    ---------------
    None

    Returns
    -------
    gauge_field_force : ndarray of shape (B, 2), dtype=complex128
        Gauge field term of the quantum-classical force.
    """
    z = kwargs["z"]
    state_ind = np.asarray(kwargs["state_ind"])
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    theta, dtheta, _, dphi = calc_mixing_angles(model, z)
    cross = dtheta[:, 0] * dphi[:, 1] - dtheta[:, 1] * dphi[:, 0]
    omega = 0.5 * np.sin(theta) * cross
    omega = np.where(state_ind == 0, omega, -omega)
    v = functions.z_to_p(z, m, h) / m
    force = np.stack([omega * v[:, 1], -omega * v[:, 0]], axis=1)
    return -functions.dqdp_to_dzc(force, None, m, h)


def rescaling_direction_fssh_dmin_geometric_phase_avoided_crossing_b(
    model: Model, parameters: dict, **kwargs
):
    """
    Gauge-fixed (:math:`\\mathfrak{D}_{\\min}`) momentum rescaling direction
    :math:`\\nabla\\theta/\\vert\\nabla\\theta\\vert`, in z-space.

    It is the real derivative coupling :math:`\\frac{1}{2}\\nabla\\theta` of the
    minimum-metric diabatic basis. The direction is zero where
    :math:`\\nabla\\theta = 0` (the origin). Called per trajectory (not vectorized).

    Keyword Arguments
    -----------------
    z : ndarray of shape (2,), dtype=complex128
        Complex classical coordinates of the hopping trajectory.
    init_state_ind, final_state_ind : int
        Indices of the initial and final adiabatic states.

    Model Constants
    ---------------
    None

    Returns
    -------
    resc_dir_z : ndarray of shape (2,), dtype=complex128
        Unit rescaling direction in z-space.
    """
    z = kwargs["z"]
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    _, dtheta, _, _ = calc_mixing_angles(model, z[np.newaxis, :])
    norm = np.linalg.norm(dtheta[0])
    if norm == 0.0:
        return np.zeros_like(z)
    return functions.dqdp_to_dzc(dtheta[0] / norm, None, m, h)


def rescaling_direction_fssh_rmax_geometric_phase_avoided_crossing_b(
    model: Model, parameters: dict, **kwargs
):
    """
    :math:`\\mathfrak{R}_{\\max}` momentum rescaling direction, in z-space: the real
    part of the derivative coupling in the gauge that maximizes its norm.

    .. math::

        \\mathbf{d}_{01} = \\frac{1}{2}(\\nabla\\theta + i\\sin\\theta\\,\\nabla\\phi),
        \\qquad \\eta^* = -\\frac{1}{2}\\arg(\\mathbf{d}_{01}\\cdot\\mathbf{d}_{01}),
        \\qquad \\mathrm{Re}(e^{i\\eta^*}\\mathbf{d}_{01})

    The dot product has no conjugate. Not in the default ingredient list: append it as
    ``("rescaling_direction_fssh", ...)``. Called per trajectory (not vectorized).

    Keyword Arguments
    -----------------
    z : ndarray of shape (2,), dtype=complex128
        Complex classical coordinates of the hopping trajectory.
    init_state_ind, final_state_ind : int
        Indices of the initial and final adiabatic states.

    Model Constants
    ---------------
    None

    Returns
    -------
    resc_dir_z : ndarray of shape (2,), dtype=complex128
        Unit rescaling direction in z-space.
    """
    z = kwargs["z"]
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    theta, dtheta, _, dphi = calc_mixing_angles(model, z[np.newaxis, :])
    der_couple = 0.5 * (dtheta[0] + 1j * np.sin(theta[0]) * dphi[0])
    eta = -0.5 * np.angle(np.sum(der_couple * der_couple))
    direction = np.real(np.exp(1j * eta) * der_couple)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return np.zeros_like(z)
    return functions.dqdp_to_dzc(direction / norm, None, m, h)


class GeometricPhaseAvoidedCrossingB(Model):
    """
    Geometric-phase avoided crossing model B class.

    Reference publication:
    Krotz and Tempelaar. Phys. Rev. A 2024, 109, 032210.
    https://doi.org/10.1103/PhysRevA.109.032210.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "A": 10.0,
            "Bx": 1.0,
            "By": 0.25,
            "W": 1.5,
            "mass": 1.0,
            "init_position_x": -4.0,
            "init_position_y": -2.0,
            "init_momentum_x": 16.0,
            "init_momentum_y": 0.0,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = True
        self.update_h_q = False

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc_geometric_phase_avoided_crossing_b),
        ("h_c", ingredients.h_c_free),
        ("dh_qc_dzc", dh_qc_dzc_geometric_phase_avoided_crossing_b),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("init_classical", ingredients.init_classical_wigner_coherent_state),
        ("hop", ingredients.hop_free),
        (
            "rescaling_direction_fssh",
            rescaling_direction_fssh_dmin_geometric_phase_avoided_crossing_b,
        ),
        ("gauge_field_force", gauge_field_force_geometric_phase_avoided_crossing_b),
        ("_init_model", _init_model),
    ]
