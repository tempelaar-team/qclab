"""
This module contains the model for ab initio dynamics.
"""

import copy

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


class AbInitio(Model):
    """
    Model class for *ab initio* calculations.

    It is compatible with the *ab initio* Algorithms implemented in QC Lab.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "atom_positions": None,  # In units of Bohr.
            "atom_masses": None,  # In units of electron mass.
            "atom_names": None,
            "calculator": "qchem",
            "calculator_args": {},
            "num_quantum_states": None,
            "normal_mode": None,  # Columns with shape (coordinates, modes).
            "normal_mode_convention": "legacy",
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
        self.constants.classical_coordinate_mass = np.repeat(
            np.asarray(self.constants.atom_masses, dtype=float), 3
        )
        self.constants.classical_coordinate_weight = np.ones(
            self.constants.num_classical_coordinates
        )
        self.constants.init_position = self.constants.atom_positions.flatten()
        self.constants.finite_difference_delta = 1e-2

    def init_classical(self, parameters, **kwargs):
        """Sample harmonic modes without changing the model constants.

        ``mass_weighted`` modes are orthonormal eigenvectors of the mass-weighted
        Cartesian Hessian. ``legacy`` retains the historical Cartesian projection.
        Frequencies must be positive and in atomic units in either convention.
        """
        constants = self.constants
        if constants.normal_mode is None or constants.harmonic_frequency is None:
            raise ValueError(
                "Wigner initialization requires normal_mode and harmonic_frequency; "
                "use init_classical_definite_position_momentum for fixed coordinates."
            )
        if np.iscomplexobj(constants.harmonic_frequency) and np.any(
            np.imag(constants.harmonic_frequency) != 0
        ):
            raise ValueError("Wigner initialization requires real, positive harmonic frequencies.")
        if np.iscomplexobj(constants.normal_mode) and np.any(
            np.imag(constants.normal_mode) != 0
        ):
            raise ValueError("Wigner initialization requires real normal modes.")
        normal_modes = np.asarray(np.real(constants.normal_mode), dtype=float)
        frequencies = np.asarray(np.real(constants.harmonic_frequency), dtype=float)
        convention = constants.normal_mode_convention
        if convention not in {"legacy", "mass_weighted"}:
            raise ValueError("normal_mode_convention must be 'legacy' or 'mass_weighted'.")
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
            raise ValueError("harmonic_frequency must be a finite one-dimensional array.")
        if np.any(frequencies <= 0):
            raise ValueError("Wigner initialization requires positive harmonic frequencies.")
        expected_shape = (constants.num_classical_coordinates, len(frequencies))
        if normal_modes.shape != expected_shape or not np.all(np.isfinite(normal_modes)):
            raise ValueError("normal_mode must have shape (coordinates, modes) and be finite.")
        masses = constants.classical_coordinate_mass
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("Atomic masses must be finite and positive.")
        if convention == "mass_weighted" and not np.allclose(
            normal_modes.T @ normal_modes,
            np.eye(len(frequencies)),
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError("Mass-weighted normal modes must have orthonormal columns.")

        # A local model supplies unit-mass oscillators to the standard sampler.
        # Keep the original Constants object and its update callback intact.
        mode_model = copy.copy(self)
        mode_model.constants = Constants()
        mode_model.constants.num_classical_coordinates = len(frequencies)
        mode_model.constants.classical_coordinate_mass = np.ones(len(frequencies))
        mode_model.constants.harmonic_frequency = frequencies
        mode_model.constants.classical_coordinate_weight = frequencies
        mode_model.constants.kBT = constants.kBT
        z_normal_mode = ingredients.init_classical_wigner_harmonic(
            mode_model, parameters, **kwargs
        )
        # Convert back to phase-space normal coordinates.
        q_normal_mode = z_to_q(
            z_normal_mode,
            mode_model.constants.classical_coordinate_mass[np.newaxis],
            mode_model.constants.classical_coordinate_weight[np.newaxis],
        )
        p_normal_mode = z_to_p(
            z_normal_mode,
            mode_model.constants.classical_coordinate_mass[np.newaxis],
            mode_model.constants.classical_coordinate_weight[np.newaxis],
        )
        # Convert back to Cartesian coordinates.
        q = np.einsum("tm,cm->tc", q_normal_mode, normal_modes)
        p = np.einsum("tm, cm ->tc", p_normal_mode, normal_modes)
        if convention == "mass_weighted":
            sqrt_mass = np.sqrt(masses)
            q = q / sqrt_mass
            p = p * sqrt_mass
        q = q + constants.init_position
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
                    cache_traj_ind=traj_ind,
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
                    cache_traj_ind=traj_ind,
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
                    cache_traj_ind=traj_ind,
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
            ingredients.ab_initio_property_calculator,
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
