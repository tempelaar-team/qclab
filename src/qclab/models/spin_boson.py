"""
This module contains the spin-boson Model class.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients


class SpinBoson(Model):
    """
    Spin-Boson Model class.

    Reference publication:
    Tempelaar & Reichman. J. Chem. Phys. 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_h_q(self, parameters, **kwargs):
        self.constants.two_level_00 = self.constants.get("E")
        self.constants.two_level_11 = -self.constants.get("E")
        self.constants.two_level_01_re = self.constants.get("V")
        self.constants.two_level_01_im = 0
        return

    def _init_h_qc(self, parameters, **kwargs):
        A = self.constants.get("A")
        l_reorg = self.constants.get("l_reorg")
        boson_mass = self.constants.get("boson_mass")
        h = self.constants.classical_coordinate_weight
        w = self.constants.harmonic_frequency
        self.constants.diagonal_linear_coupling = np.zeros((2, A))
        self.constants.diagonal_linear_coupling[0] = (
            w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        self.constants.diagonal_linear_coupling[1] = (
            -w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        A = self.constants.get("A")
        W = self.constants.get("W")
        self.constants.harmonic_frequency = W * np.tan(
            np.arange(0.5, A + 0.5, 1.0) * np.pi * 0.5 / A
        )
        return

    def _init_model(self, parameters, **kwargs):
        A = self.constants.get("A")
        boson_mass = self.constants.get("boson_mass")
        self.constants.num_classical_coordinates = A
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.classical_coordinate_mass = boson_mass * np.ones(A)
        return

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_h_qc", _init_h_qc),
        ("_init_model", _init_model),
        ("_init_h_c", _init_h_c),
    ]
