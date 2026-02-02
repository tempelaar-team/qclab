"""
The module defines the Model class for Tully's first problem, a simple avoided crossing.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients
from qclab import functions


class TullyProblemOne(Model):
    """
    A Model class for Tully's first problem: a simple avoided crossing.

    Reference publication:
    Tully. J. Chem. Phys. 1990, 93 (2), 1061–1071. https://doi.org/10.1063/1.459170.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "init_momentum": 10.0,
            "init_position": -25.0,
            "mass": 2000.0,
            "A": 0.01,
            "B": 1.6,
            "C": 0.005,
            "D": 1.0,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states = 2
        self.constants.num_classical_coordinates = 1
        self.constants.classical_coordinate_mass = np.array(
            [self.constants.get("mass")]
        )
        self.constants.classical_coordinate_weight = np.array([1.0])
        self.constants.init_position = np.array([self.constants.get("init_position")])
        self.constants.init_momentum = np.array([self.constants.get("init_momentum")])
        return

    def h_qc(self, parameters, **kwargs):
        """
        Quantum-classical Hamiltonian for Tully's first problem.
        """
        z = kwargs["z"]
        batch_size = len(z)
        num_quantum_states = self.constants.num_quantum_states
        A = self.constants.get("A")
        B = self.constants.get("B")
        C = self.constants.get("C")
        D = self.constants.get("D")
        # Calculate q.
        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)[:, 0]
        # Calculate matrix elements.
        v_11 = np.zeros(batch_size, dtype=complex)
        v_11[q >= 0.0] = A * (1.0 - np.exp(-B * q))[q >= 0.0]
        v_11[q < 0.0] = -A * (1.0 - np.exp(B * q))[q < 0.0]
        v_12 = C * np.exp(-D * (q**2))
        # Assemble Hamiltonian.
        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        h_qc[:, 0, 0] = v_11
        h_qc[:, 0, 1] = v_12
        h_qc[:, 1, 0] = v_12
        h_qc[:, 1, 1] = -v_11
        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        """
        Gradient w.r.t. to the conjugate z coordinate of the quantum-classical Hamiltonian for Tully's first problem.
        """
        z = kwargs["z"]
        batch_size = len(z)
        num_quantum_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        A = self.constants.get("A")
        B = self.constants.get("B")
        C = self.constants.get("C")
        D = self.constants.get("D")
        # Calculate q.
        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)[:, 0]
        # Calculate phase-space gradients.
        dv_11_dq = np.zeros(batch_size, dtype=complex)
        dv_11_dq[q >= 0] = A * B * np.exp(-B * q)[q >= 0]
        dv_11_dq[q < 0] = A * B * np.exp(B * q)[q < 0]
        dv_12_dq = -2 * C * D * q * np.exp(-D * q**2)
        # Convert to complex gradients.
        dv_11_dzc = functions.dqdp_to_dzc(dv_11_dq, None, m[0], h[0])
        dv_12_dzc = functions.dqdp_to_dzc(dv_12_dq, None, m[0], h[0])
        # Assemble indices.
        batch_idx = np.repeat(np.arange(batch_size, dtype=int), 4)
        coord_idx = np.zeros(4 * batch_size, dtype=int)
        state_i_idx = np.tile(np.array([0, 0, 1, 1], dtype=int), batch_size)
        state_j_idx = np.tile(np.array([0, 1, 0, 1], dtype=int), batch_size)
        inds = (batch_idx, coord_idx, state_i_idx, state_j_idx)
        # Assemble matrix elements.
        mels = np.empty(4 * batch_size, dtype=complex)
        mels[0::4] = dv_11_dzc
        mels[1::4] = dv_12_dzc
        mels[2::4] = dv_12_dzc
        mels[3::4] = -dv_11_dzc
        # Assemble shape.
        shape = (
            batch_size,
            num_classical_coordinates,
            num_quantum_states,
            num_quantum_states,
        )
        return inds, mels, shape

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("init_classical", ingredients.init_classical_definite_position_momentum),
        ("hop", ingredients.hop_free),
        ("_init_model", _init_model),
    ]
