"""Ab initio electronic structure in a locally transported diabatic representation."""

import copy
import numbers

import numpy as np

from qclab import functions
from qclab.models.ab_initio import AbInitio
from qclab.numerical_constants import ANGSTROM_TO_BOHR


class LocalDiabaticAbInitio(AbInitio):
    """Supply ordinary dynamics algorithms with local ab initio operators.

    Uses the quasi-diabatic propagation construction of Mandal, Yamijala, and
    Huo, J. Chem. Theory Comput. 14, 1828 (2018), DOI 10.1021/acs.jctc.7b01178.
    Calculator arguments and nuclear initialization are inherited from
    :class:`AbInitio`. Initial ``wf_db`` amplitudes refer to each trajectory's
    initial adiabatic states. Ordinary MeanField and FSSH recipes maintain the
    accepted reference; their collected electronic density is ``dm_adb``.
    """

    def _init_local_diabatization(self, parameters=None, **kwargs):
        """Check the electronic conventions required by ordinary overlap transport."""
        constants = self.constants
        num_states = constants.num_quantum_states
        if (isinstance(num_states, (bool, np.bool_))
                or not isinstance(num_states, numbers.Integral) or num_states < 1):
            raise ValueError("Local diabatization requires a positive integer num_quantum_states.")
        arguments = constants.calculator_args
        if not isinstance(arguments, dict):
            raise ValueError("calculator_args must be a dictionary.")
        if str(arguments.get("method_es", "casscf")).lower() == "tddft":
            if not arguments.get("tda", True):
                raise ValueError("Local diabatization currently supports TDA, not full TDDFT.")
            etfs = arguments.get("use_etfs", True)
            if not isinstance(etfs, (bool, np.bool_)) or etfs:
                raise ValueError(
                    "Local diabatization requires TDA calculator_args['use_etfs']=False "
                    "(use_etfs=False), so couplings match ordinary wavefunction overlaps."
                )

    @classmethod
    def _signature(cls, value):
        if isinstance(value, np.ndarray):
            return value.dtype.str, value.shape, value.tobytes()
        if isinstance(value, dict):
            return tuple((key, cls._signature(value[key])) for key in sorted(value, key=repr))
        if isinstance(value, (list, tuple)):
            return tuple(cls._signature(item) for item in value)
        return value

    def _configuration(self):
        constants = self.constants
        calculator, found = self.get("ab_initio_property_calculator")
        if not found:
            raise ValueError("Local diabatization requires an ab_initio_property_calculator.")
        # Store a portable description, not the mutable calculator object. Branch
        # copies must not turn an unchanged callable ingredient into a new model.
        implementation = getattr(calculator, "__func__", calculator)
        calculator_type = type(implementation)
        calculator_name = (
            getattr(implementation, "__module__", calculator_type.__module__),
            getattr(implementation, "__qualname__", calculator_type.__qualname__),
        )
        return self._signature((
            calculator_name, constants.get("calculator", "qchem"), constants.calculator_args,
            constants.atom_names, constants.atom_masses, constants.num_quantum_states,
            constants.classical_coordinate_weight, constants.energy_offset,
        ))

    @staticmethod
    def _polar_overlap(overlap, num_states):
        overlap = np.asarray(overlap, dtype=complex)
        if overlap.shape != (num_states, num_states) or not np.all(np.isfinite(overlap)):
            raise ValueError("Local diabatization requires a finite full (states, states) overlap.")
        left, singular_values, right = np.linalg.svd(overlap)
        tolerance = np.finfo(float).eps * num_states * singular_values[0]
        if singular_values[0] == 0 or singular_values[-1] <= tolerance:
            raise ValueError(
                "Local-diabatization overlap is rank deficient; reduce the time step "
                "or retain more electronic states. Singular values: %s" % singular_values
            )
        return left @ right, singular_values

    @staticmethod
    def _adiabatic_gradient(properties, energies, positions):
        num_states = len(energies)
        gradients = np.real_if_close(np.asarray(properties.get("gradient")))
        if (gradients.shape != (*positions.shape, num_states)
                or np.iscomplexobj(gradients) or not np.all(np.isfinite(gradients))):
            raise ValueError("Local diabatization requires finite real gradients for every state.")
        gradient = np.zeros((positions.size, num_states, num_states), dtype=complex)
        indices = np.arange(num_states)
        gradient[:, indices, indices] = gradients.reshape(positions.size, num_states)
        couplings = properties.get("derivative_coupling")
        if not isinstance(couplings, dict):
            if num_states > 1:
                raise ValueError("Local diabatization requires all derivative coupling pairs.")
            couplings = {}
        for bra in range(num_states):
            for ket in range(bra + 1, num_states):
                if (bra, ket) in couplings:
                    coupling = np.asarray(couplings[bra, ket], dtype=complex)
                elif (ket, bra) in couplings:
                    coupling = -np.asarray(couplings[ket, bra], dtype=complex).conj()
                else:
                    raise ValueError(
                        "Local diabatization requires all derivative coupling pairs; "
                        "missing pair (%s, %s)." % (bra, ket)
                    )
                if coupling.shape != positions.shape or not np.all(np.isfinite(coupling)):
                    raise ValueError("Local diabatization received an invalid derivative coupling.")
                if (bra, ket) in couplings and (ket, bra) in couplings:
                    if not np.allclose(coupling, -np.asarray(couplings[ket, bra]).conj()):
                        raise ValueError("Derivative coupling pairs must be anti-Hermitian.")
                # Calculator couplings use inverse Angstrom; energies and coordinates
                # in the force matrix use Hartree and Bohr, respectively.
                element = (energies[ket] - energies[bra]) * coupling.ravel() / ANGSTROM_TO_BOHR
                gradient[:, bra, ket] = element
                gradient[:, ket, bra] = element.conj()
        return gradient

    def _solve_properties(self, parameters, z, traj_ind, frame):
        calculator, _ = self.get("ab_initio_property_calculator")
        request = {
            "energy": {"z": z[None], "excited_amplitudes": True},
            "gradient": {"z": z[None], "state_inds_gradient": None},
            "derivative_coupling": {"z": z[None], "state_inds_derivative_coupling": None},
        }
        if frame is not None:
            previous = np.empty(1, dtype=object)
            previous[0] = copy.deepcopy(frame["anchor"])
            request["wf_overlaps"] = {
                "z": z[None], "z_previous": frame["anchor_z"][None],
                "amplitudes_previous": previous,
            }
        return calculator(
            self, parameters, batch_size=1, cache_traj_ind=traj_ind,
            property_dict=request,
        )[0]

    def _packet(self, parameters, z, positions, traj_ind, frame):
        key = positions.tobytes()
        if frame is not None and key in frame["cache"]:
            return frame["cache"][key]
        properties = self._solve_properties(parameters, z, traj_ind, frame)
        num_states = self.constants.num_quantum_states
        energies = np.real_if_close(np.asarray(properties.get("energy")))
        if (energies.shape != (num_states,) or np.iscomplexobj(energies)
                or not np.all(np.isfinite(energies))):
            raise ValueError("Local diabatization requires finite real energies for every state.")
        if properties.get("excited_amplitudes") is None:
            raise ValueError("Local diabatization requires independent excited_amplitudes snapshots.")
        if frame is None:
            transform = np.eye(num_states, dtype=complex)
            singular_values = np.ones(num_states)
        else:
            rotation, singular_values = self._polar_overlap(properties.get("wf_overlaps"), num_states)
            transform = frame["transform"] @ rotation
        gradient = self._adiabatic_gradient(properties, energies, positions)
        shifted_energies = energies - self.constants.energy_offset
        packet = {
            "hamiltonian": (transform * shifted_energies[None]) @ transform.conj().T,
            "gradient": transform[None] @ gradient @ transform.conj().T[None],
            "adiabatic_gradient": gradient,
            "adiabatic_to_diabatic": transform,
            "energies": shifted_energies.copy(),
            "atom_positions": positions.copy(),
            "snapshot": copy.deepcopy(properties["excited_amplitudes"]),
            "overlap_singular_values": singular_values,
            "frame_revision": 0 if frame is None else frame["revision"],
        }
        if frame is not None:
            frame["cache"][key] = packet
        return packet

    def local_diabatization(self, parameters, **kwargs):
        """Evaluate or commit locally transported Hamiltonians and gradient operators.

        Keyword Args
        ------------
        z : ndarray of shape (B, C)
            Nuclear complex coordinates. Electronic calculations depend on positions.
        action : str, optional
            ``initialize`` creates independent anchors with identity transforms;
            ``evaluate`` freezes those anchors; ``commit`` accepts the exact endpoint.

        Returns
        -------
        list of dict
            Independent Hamiltonian, Cartesian gradient, transformation, snapshot,
            and overlap-singular-value packets. Mutable frames live in ``parameters``.
        """
        self._init_local_diabatization()
        action = kwargs.get("action", "evaluate")
        if action not in {"initialize", "evaluate", "commit"}:
            raise ValueError("Local-diabatization action must be initialize, evaluate, or commit.")
        z = np.asarray(kwargs["z"], dtype=complex)
        constants = self.constants
        if (z.ndim != 2 or z.shape[1] != constants.num_classical_coordinates
                or not np.all(np.isfinite(z))):
            raise ValueError("Local diabatization requires finite batched Cartesian z coordinates.")
        positions = functions.z_to_q(
            z, constants.classical_coordinate_mass, constants.classical_coordinate_weight
        ).reshape((len(z), -1, 3))
        signature = self._configuration()
        frames = parameters.get("_local_diabatic_frames", {})
        if action != "initialize":
            for traj_ind in range(len(z)):
                if traj_ind not in frames:
                    raise RuntimeError("Initialize the local-diabatization frame before evaluating it.")
                if frames[traj_ind]["configuration"] != signature:
                    raise ValueError(
                        "The local-diabatization configuration changed; reinitialize the "
                        "trajectory instead of resetting its electronic frame."
                    )
        # Validate every trajectory before committing any accepted frame. An error
        # can populate a trial cache, but cannot partially advance the accepted batch.
        packets = [
            self._packet(
                parameters, z[index], positions[index], index,
                None if action == "initialize" else frames[index],
            )
            for index in range(len(z))
        ]
        if action in {"initialize", "commit"}:
            accepted = {}
            for index, packet in enumerate(packets):
                revision = 0 if action == "initialize" else frames[index]["revision"] + 1
                packet = copy.deepcopy(packet)
                packet["frame_revision"] = revision
                accepted[index] = {
                    "configuration": signature,
                    "anchor": copy.deepcopy(packet["snapshot"]),
                    "anchor_z": z[index].copy(),
                    "transform": packet["adiabatic_to_diabatic"].copy(),
                    "revision": revision,
                    "overlap_singular_values": packet["overlap_singular_values"].copy(),
                    "cache": {positions[index].tobytes(): packet},
                }
                packets[index] = packet
            parameters["_local_diabatic_frames"] = accepted
        return copy.deepcopy(packets)

    def h_qc(self, parameters, **kwargs):
        """Return the electronic Hamiltonian in persistent transported coordinates."""
        ingredient, _ = self.get("local_diabatization")
        return np.asarray([packet["hamiltonian"] for packet in ingredient(
            self, parameters, z=kwargs["z"], action="evaluate"
        )])

    @functions.make_ingredient_sparse
    def dh_qc_dzc(self, parameters, **kwargs):
        """Return the full local electronic gradient as a sparse z-conjugate tensor."""
        ingredient, _ = self.get("local_diabatization")
        gradients = np.asarray([packet["gradient"] for packet in ingredient(
            self, parameters, z=kwargs["z"], action="evaluate"
        )])
        return functions.dqdp_to_dzc(
            gradients, None, self.constants.classical_coordinate_mass[None, :, None, None],
            self.constants.classical_coordinate_weight[None, :, None, None],
        )

    ingredients = AbInitio.ingredients + [
        ("local_diabatization", local_diabatization),
        ("h_qc", h_qc),
        ("dh_qc_dzc", dh_qc_dzc),
        ("derivative_coupling_dzc", None),
        ("_init_local_diabatization", _init_local_diabatization),
    ]
