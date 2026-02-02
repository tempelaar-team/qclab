"""
This module contains the Model class for the Fenna-Matthews-Olson (FMO) complex.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients
from qclab.numerical_constants import INVCM_TO_300K


class FMOComplex(Model):
    """
    A Model class representing the Fenna-Matthews-Olson (FMO) complex.

    All quantities in this model are taken to be in units of kBT at 300 K.

    At 300 K, kBT = 0.025852 eV = 208.521 cm^-1. Any quantity in wavenumbers
    is made unitless by dividing by kBT.

    One unit of time is given by \\hbar/kBT = 25.4595 fs.

    Reference publication:
    Mulvihill et al. J. Chem. Phys. 2021, 154 (20), 204109. https://doi.org/10.1063/5.0051101.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "mass": 1.0,
            "l_reorg": 35.0 * INVCM_TO_300K,  # Reorganization energy
            "w_c": 106.14 * INVCM_TO_300K,  # Characteristic frequency
            "N": 200,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        """
        Initialize the model-specific constants.
        """
        self.constants.num_quantum_states = 7
        N = self.constants.get("N")
        w_c = self.constants.get("w_c")
        mass = self.constants.get("mass")
        self.constants.w = (
            w_c
            * np.tan(np.arange(0.5, N + 0.5, 1.0) * np.pi * 0.5 / (N))[np.newaxis, :]
            * np.ones((self.constants.num_quantum_states, N))
        ).flatten()
        self.constants.num_classical_coordinates = self.constants.num_quantum_states * N

        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = mass * np.ones(
            self.constants.num_classical_coordinates
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        self.constants.harmonic_frequency = self.constants.w
        return

    def _init_h_qc(self, parameters, **kwargs):
        N = self.constants.get("N")
        l_reorg = self.constants.get("l_reorg")
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        w = self.constants.w
        self.constants.diagonal_linear_coupling = np.zeros(
            (
                self.constants.num_quantum_states,
                self.constants.num_classical_coordinates,
            )
        )
        for n in range(self.constants.num_quantum_states):
            self.constants.diagonal_linear_coupling[n, n * N : (n + 1) * N] = (
                w * np.sqrt(2.0 * l_reorg / N) * (1.0 / np.sqrt(2.0 * m * h))
            )[n * N : (n + 1) * N]
        return

    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        matrix_elements = (
            np.array(
                [
                    [12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
                    [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
                    [5.5, 30.8, 12210.0, -53.5, -2.2, -9.6, 6.0],
                    [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                    [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
                    [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
                    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440],
                ],
                dtype=complex,
            )
            * INVCM_TO_300K
        )
        # To reduce numerical errors we offset the diagonal elements by
        # their minimum value.
        matrix_elements = matrix_elements - np.min(
            np.diag(matrix_elements)
        ) * np.identity(7)
        # Finally we broadcast the array to the desired shape.
        return np.broadcast_to(matrix_elements, (batch_size, 7, 7))

    ingredients = [
        ("h_q", h_q),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_qc", _init_h_qc),
        ("_init_h_c", _init_h_c),
        ("_init_model", _init_model),
    ]
