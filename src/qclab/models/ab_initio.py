"""
This module contains the Atomic Simulation Environment (ASE) Model class.
"""

import numpy as np
from qclab.functions import (
    vectorize_ingredient,
    make_ingredient_sparse,
    z_to_q,
    z_to_p,
    qp_to_z,
    dqdp_to_dzc,
)
from qclab.model import Model, Constants
from qclab import ingredients
from qclab.numerical_constants import (
    ANGSTROM_TO_BOHR,
)
import copy


class AbInitio(Model):
    """
    Model class for ab initio calculations.

    It is compatible with the ab initio algorithms implemented in QC Lab.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "atom_positions": None,  # In units of Bohr.
            "atom_masses": None,  # In units of electron mass.
            "atom_names": None,
            "calculator_args": {},
            "num_quantum_states": None,
            "normal_mode": None,  # Unitless and normalilzed.
            "harmonic_frequency": None,  # In units of Hartrees.
            "energy_offset": 0,  # In units of Hartrees.
            "kBT": 0.00095,  # In units of Hartrees.
        }
        self.update_dh_qc_dzc = True
        self.update_h_q = False
        super().__init__(self.default_constants, constants)

    def _init_model(self, parameters, **kwargs):
        num_atoms = len(self.constants.atom_names)
        self.constants.num_classical_coordinates = num_atoms * 3
        self.constants.classical_coordinate_mass = (
            self.constants.atom_masses[np.newaxis] * np.ones((3, num_atoms))
        ).flatten()
        self.constants.classical_coordinate_weight = np.ones(
            self.constants.num_classical_coordinates
        )
        self.constants.init_position = self.constants.atom_positions.flatten()
        self.constants.finite_difference_delta = 1e-2

    def init_classical(self, parameters, **kwargs):
        # Temporarily set the masses to 1 for mass-weighted normal mode sampling.
        normal_modes = self.constants.normal_mode
        constants_original = copy.deepcopy(self.constants)
        self.constants = Constants()
        # Set number of coordinates to number of normal modes.
        self.constants.num_classical_coordinates = len(
            constants_original.harmonic_frequency
        )
        self.constants.classical_coordinate_mass = np.ones(
            self.constants.num_classical_coordinates
        )
        self.constants.harmonic_frequency = constants_original.harmonic_frequency
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.kBT = constants_original.kBT
        z_normal_mode = ingredients.init_classical_wigner_harmonic(
            self, parameters, **kwargs
        )
        # Convert back to phase-space normal coordinates.
        q_normal_mode = z_to_q(
            z_normal_mode,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        p_normal_mode = z_to_p(
            z_normal_mode,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        # Convert back to Cartesian coordinates.
        q = (
            np.einsum("tm, cm ->tc", q_normal_mode, normal_modes)
            + constants_original.init_position
        )
        p = np.einsum("tm, cm ->tc", p_normal_mode, normal_modes)
        self.constants = constants_original
        z = qp_to_z(
            q,
            p,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        return z

    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        num_quantum_states = self.constants.num_quantum_states
        out = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        return out

    @vectorize_ingredient
    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_quantum_states = self.constants.num_quantum_states
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_property = "ab_initio_property" in parameters.keys()
        if has_ab_initio_property:
            properties = parameters["ab_initio_property"][traj_ind]
            if "energy" in properties.keys():
                diag_h_qc = (
                    properties["energy"][:num_quantum_states]
                    - self.constants.energy_offset
                )
                needs_energy = False
            else:
                needs_energy = True
        if not (has_ab_initio_property) or needs_energy:
            property_dict = {
                "energy": {"z": z[np.newaxis], "excited_amplitudes": True},
            }
            ab_initio_property_calculator, has_ab_intio_property_calculator = self.get(
                "ab_initio_property_calculator"
            )
            if has_ab_intio_property_calculator:
                properties = ab_initio_property_calculator(
                    self,
                    parameters,
                    batch_size=1,
                    property_dict=property_dict,
                )[0]
                diag_h_qc = (
                    properties["energy"][:num_quantum_states]
                    - self.constants.energy_offset
                )
            else:
                raise NameError("ab_initio_property_calculator must be provided")
        return np.diag(diag_h_qc)

    @make_ingredient_sparse
    @vectorize_ingredient
    def dh_qc_dzc(self, parameters, **kwargs):
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        z = kwargs["z"]
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_property = "ab_initio_property" in parameters.keys()
        if has_ab_initio_property:
            properties = parameters["ab_initio_property"][traj_ind]
            if "gradient" in properties.keys():
                for state_ind in range(num_quantum_states):
                    # Convert to derivative w.r.t. zc.
                    out[:, state_ind, state_ind] = dqdp_to_dzc(
                        properties["gradient"][:, :, state_ind].flatten(),
                        None,
                        m,
                        h,
                    )
                needs_gradient = False
            else:
                needs_gradient = True
        if not (has_ab_initio_property) or needs_gradient:
            property_dict = {
                "gradient": {"z": z[np.newaxis], "state_inds_gradient": None},
            }
            ab_initio_property_calculator, has_ab_intio_property_calculator = self.get(
                "ab_initio_property_calculator"
            )
            if has_ab_intio_property_calculator:
                properties = ab_initio_property_calculator(
                    self,
                    parameters,
                    batch_size=1,
                    property_dict=property_dict,
                )[0]
                for state_ind in range(num_quantum_states):
                    # Convert to derivative w.r.t. zc.
                    out[:, state_ind, state_ind] = dqdp_to_dzc(
                        properties["gradient"][:, :, state_ind].flatten(),
                        None,
                        m,
                        h,
                    )
            else:
                raise NameError("ab_initio_property_calculator must be provided")
        return out

    @vectorize_ingredient
    def derivative_coupling_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_property = "ab_initio_property" in parameters.keys()
        if has_ab_initio_property:
            properties = parameters["ab_initio_property"][traj_ind]
            if "derivative_coupling" in properties.keys():
                derivative_coupling_dq = properties["derivative_coupling"]
                if not (derivative_coupling_dq is None):
                    for key, val in derivative_coupling_dq.items():
                        out[:, key[0], key[1]] = (
                            dqdp_to_dzc(val.flatten(), None, m, h) / ANGSTROM_TO_BOHR
                        )  # Convert from 1/Angstrom to 1/Bohr.
                        out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
                needs_derivative_coupling = False
            else:
                needs_derivative_coupling = True
        if not (has_ab_initio_property) or needs_derivative_coupling:
            property_dict = {
                "derivative_coupling": {
                    "z": z[np.newaxis],
                    "state_inds_derivative_coupling": None,
                },
            }
            ab_initio_property_calculator, has_ab_intio_property_calculator = self.get(
                "ab_initio_property_calculator"
            )
            if has_ab_intio_property_calculator:
                properties = ab_initio_property_calculator(
                    self,
                    parameters,
                    batch_size=1,
                    property_dict=property_dict,
                )[0]
                derivative_coupling_dq = properties["derivative_coupling"]
                for key, val in derivative_coupling_dq.items():
                    out[:, key[0], key[1]] = (
                        dqdp_to_dzc(val.flatten(), None, m, h) / ANGSTROM_TO_BOHR
                    )  # Convert from 1/Angstrom to 1/Bohr.
                    out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
            else:
                raise NameError("ab_initio_property_calculator must be provided")
        return out

    ingredients = [
        (
            "ab_initio_property_calculator",
            ingredients.ab_initio_property_calculator_qchem,
        ),
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("hop", ingredients.hop_free),
        ("init_classical", init_classical),
        ("derivative_coupling_dzc", derivative_coupling_dzc),
        ("_init_model", _init_model),
    ]
