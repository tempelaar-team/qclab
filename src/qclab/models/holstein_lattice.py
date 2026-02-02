"""
This module contains the Holstein lattice Model class.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients


class HolsteinLattice(Model):
    """
    A Model class representing a nearest-neighbor tight-binding model with Holstein-type
    electron-phonon coupling with a single optical mode.

    Reference publication:
    Krotz et al. J. Chem. Phys. 2021, 154 (22), 224101. https://doi.org/10.1063/5.0053177.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "g": 0.5,
            "w": 0.5,
            "N": 10,
            "J": 1.0,
            "phonon_mass": 1.0,
            "periodic": True,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        phonon_mass = self.constants.get("phonon_mass")
        self.constants.num_quantum_states = N
        self.constants.num_classical_coordinates = N
        self.constants.classical_coordinate_weight = w * np.ones(N)
        self.constants.classical_coordinate_mass = phonon_mass * np.ones(N)
        return

    def _init_h_q(self, parameters, **kwargs):
        J = self.constants.get("J")
        periodic = self.constants.get("periodic")
        self.constants.nearest_neighbor_hopping_energy = J
        self.constants.nearest_neighbor_periodic = periodic
        return

    def _init_h_qc(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        g = self.constants.get("g")
        h = self.constants.classical_coordinate_weight
        self.constants.diagonal_linear_coupling = np.diag(
            g * w * np.sqrt(w / h) * np.ones(N)
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        self.constants.harmonic_frequency = w * np.ones(N)
        return

    ingredients = [
        ("h_q", ingredients.h_q_nearest_neighbor),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_h_qc", _init_h_qc),
        ("_init_h_c", _init_h_c),
        ("_init_model", _init_model),
    ]


class HolsteinLatticeReciprocalSpace(Model):
    """
    A model representing a nearest-neighbor tight-binding model with Holstein-type
    electron-phonon coupling with a single optical mode implemented in reciprocal space.

    Reference publication:
    Krotz et al. J. Chem. Phys. 154, 224101 (2021); https://doi.org/10.1063/5.0053177
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "g": 0.5,
            "w": 0.5,
            "N": 10,
            "J": 1.0,
            "phonon_mass": 1.0,
            "periodic": True,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def init_model(self, parameters, **kwargs):
        N = self.constants.get("N")
        assert N % 2 == 0, "N must be even."
        self.constants.num_quantum_states = N
        self.constants.num_classical_coordinates = N
        self.constants.classical_coordinate_weight = self.constants.get("w") * np.ones(
            N
        )
        self.constants.classical_coordinate_mass = self.constants.get(
            "phonon_mass"
        ) * np.ones(N)
        self.constants.harmonic_frequency = self.constants.get("w") * np.ones(N)
        self.constants.k_inds = np.arange(-N / 2, N / 2).astype(int)
        self.constants.k_diff_inds = np.array(
            [np.roll(self.constants.k_inds, int(N / 2 + i)) for i in range(N)]
        ).astype(int)
        return

    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        J = self.constants.get("J")
        k = 2.0 * np.pi * self.constants.k_inds / self.constants.num_quantum_states
        out = np.diag(-2.0 * J * np.cos(k))
        h_q = np.broadcast_to(
            out[np.newaxis, :, :],
            (
                batch_size,
                self.constants.num_quantum_states,
                self.constants.num_quantum_states,
            ),
        )
        return h_q

    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        g = self.constants.get("g")
        w = self.constants.get("w")
        h = self.constants.classical_coordinate_weight
        z_kap_mat = z[:, self.constants.k_diff_inds]
        zc_mkap_mat = np.conj(z[:, self.constants.k_diff_inds.transpose()])
        h_qc = (g * w * np.sqrt(w / h) / np.sqrt(self.constants.num_quantum_states)) * (
            z_kap_mat + zc_mkap_mat
        )
        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        batch_size = len(z)
        g = self.constants.get("g")
        w = self.constants.get("w")
        h = self.constants.classical_coordinate_weight
        out = np.zeros(
            (
                batch_size,
                self.constants.num_classical_coordinates,
                self.constants.num_quantum_states,
                self.constants.num_quantum_states,
            ),
            dtype=complex,
        )
        for k_ind in self.constants.k_inds:
            pos = np.where(self.constants.k_diff_inds.transpose() == k_ind)
            out[:, k_ind, pos[0], pos[1]] = (
                g * w * np.sqrt(w / h) / np.sqrt(self.constants.num_quantum_states)
            )
        shape = (
            batch_size,
            self.constants.num_classical_coordinates,
            self.constants.num_quantum_states,
            self.constants.num_quantum_states,
        )
        inds = np.where(out != 0)
        mels = out[inds]
        return inds, mels, shape

    ingredients = [
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_model", init_model),
    ]
