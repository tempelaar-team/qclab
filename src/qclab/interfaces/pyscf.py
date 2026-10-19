"""In-process PySCF electronic structure calculations for ab initio dynamics.

Coordinates are in Bohr, masses in electron masses, and energies in Hartree.
The public property dictionary follows the Q-Chem calculator convention,
including derivative couplings in inverse Angstrom rather than inverse Bohr.
PySCF is optional; importing QC Lab does not require it.

Restricted TDA states are normalized single excitations of a closed-shell
Kohn-Sham reference. Their alpha amplitudes are PySCF X, with +X for
singlet beta amplitudes and -X for triplet (Ms=0) beta amplitudes. Analytic
responses include orbital, XC, moving-grid, and basis-motion contributions.
"""

from contextlib import contextmanager
import copy
import itertools
import logging
import math
import numbers
import threading

import numpy as np

from qclab.numerical_constants import ANGSTROM_TO_BOHR


logger = logging.getLogger(__name__)

try:
    from pyscf import fci, gto, lib, mcscf, scf
    from pyscf.nac import sacasscf
except ImportError:
    fci = gto = lib = mcscf = scf = sacasscf = None

try:
    from pyscf import mcpdft
except ImportError:
    mcpdft = None


# Keep optional response capabilities separate from the CASSCF imports above.
try:
    from pyscf import dft
except ImportError:
    dft = None

try:
    from pyscf.grad import rks as rks_grad
except ImportError:
    rks_grad = None

try:
    from pyscf.grad import tdrks
    from pyscf.scf import cphf
except ImportError:
    tdrks = cphf = None


def _require_pyscf():
    if gto is None:
        raise ImportError(
            "The PySCF calculator requires PySCF 2.14.x. "
            "Install it with pip install 'qclab[pyscf]'."
        )


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be an integer." % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be at least %d." % (name, minimum))
    return int(value)


def determinant_overlaps(mo_overlap, ncore, ncas, nelecas):
    """Return alpha/beta occupied-determinant overlaps in PySCF CI order.

    Rows are previous determinants and columns are current determinants.
    Every occupied string includes the doubly occupied inactive orbitals;
    core-active mixing is therefore retained, even for singular core blocks.
    """
    _require_pyscf()
    mo_overlap = np.asarray(mo_overlap)
    nocc = ncore + ncas
    if mo_overlap.shape != (nocc, nocc):
        raise ValueError("MO overlap must span the inactive and active orbitals.")
    transformations = []
    for num_electrons in nelecas:
        active = fci.cistring.gen_occslst(range(ncas), num_electrons)
        core = np.tile(np.arange(ncore), (len(active), 1))
        occupied = np.concatenate((core, active + ncore), axis=1)
        transform = np.empty((len(occupied), len(occupied)), dtype=mo_overlap.dtype)
        for row, previous in enumerate(occupied):
            blocks = mo_overlap[previous[None, :, None], occupied[:, None, :]]
            transform[row] = np.linalg.det(blocks)
        transformations.append(transform)
    return tuple(transformations)


def wavefunction_overlaps(previous, current):
    """Compute ``<previous root i | current root j>`` including core orbitals.

    Snapshots contain only molecule serialization and copied arrays, so they
    can be retained independently of the calculator and sent between processes.
    """
    _require_pyscf()
    if previous.get("tda", False) or current.get("tda", False):
        return tda_wavefunction_overlaps(previous, current)
    for name in ("ncore", "ncas", "nelecas"):
        if previous[name] != current[name]:
            raise ValueError("Wavefunction snapshots have different active spaces.")
    mol_previous = gto.loads(previous["mol"])
    mol_current = gto.loads(current["mol"])
    if not np.array_equal(mol_previous.atom_charges(), mol_current.atom_charges()):
        raise ValueError("Wavefunction snapshots must describe the same atoms.")
    ao_overlap = gto.intor_cross("int1e_ovlp", mol_previous, mol_current)
    nocc = previous["ncore"] + previous["ncas"]
    mo_overlap = (
        previous["mo_coeff"][:, :nocc].conj().T
        @ ao_overlap
        @ current["mo_coeff"][:, :nocc]
    )
    alpha, beta = determinant_overlaps(
        mo_overlap, previous["ncore"], previous["ncas"], previous["nelecas"]
    )
    transformed = [alpha @ ci @ beta.T for ci in current["ci"]]
    overlaps = np.array(
        [[np.vdot(ci, ket) for ket in transformed] for ci in previous["ci"]]
    )
    return np.real_if_close(overlaps)


def _harmonic_modes(hessian, positions, masses):
    """Diagonalize a Hessian in the internal mass-weighted subspace (atomic units)."""
    mass = np.repeat(masses, 3)
    sqrt_mass = np.sqrt(mass)
    weighted = hessian / sqrt_mass[:, None] / sqrt_mass[None, :]
    center = np.average(positions, weights=masses, axis=0)
    centered = positions - center
    translations = [np.tile(axis, len(masses)) * sqrt_mass for axis in np.eye(3)]
    rotations = [
        np.cross(np.broadcast_to(axis, centered.shape), centered).ravel() * sqrt_mass
        for axis in np.eye(3)
    ]
    # Normalize generators first: translation and rotation have different units.
    rigid = np.array(translations + rotations).T
    norms = np.linalg.norm(rigid, axis=0)
    rigid = rigid[:, norms > 1e-10]
    rigid /= np.linalg.norm(rigid, axis=0)
    basis, singular, _ = np.linalg.svd(rigid, full_matrices=True)
    rank = np.count_nonzero(singular > 1e-8)
    internal = basis[:, rank:]
    eigenvalues, vectors = np.linalg.eigh(internal.T @ weighted @ internal)
    frequencies = np.real_if_close(np.lib.scimath.sqrt(eigenvalues))
    return {
        "harmonic_frequency": frequencies,
        "normal_mode": internal @ vectors,
        "normal_mode_convention": "mass_weighted",
    }


class QCLabPySCFInterface:
    """CASSCF, CMS-PDFT, and restricted TDA calculator for QC Lab.

    ``num_quantum_states`` includes state zero. ``ncas`` counts active orbitals;
    ``nelecas`` is an electron count or an ``(alpha, beta)`` pair.
    ``active_orbitals`` uses zero-based indices in the initial HF orbital set.
    ``scf_options``, ``casscf_options``, and ``response_options`` set numerical
    solver attributes. Structural settings belong in the explicit arguments.
    Real, restricted orbitals and one spin multiplicity are supported.
    CMS-PDFT supports ``tPBE``, ``tBLYP``, and ``ftPBE`` on-top functionals.
    Its geometries are solved independently; previous snapshots set phases.
    ``method_es='tddft', tda=True`` selects the TDA implementation, supporting
    PBE/B3LYP and a singlet reference with singlet or triplet excited states.
    Its ``excited_state_indices`` select native zero-based excited roots.
    """

    implemented_properties = ("energy", "gradient", "derivative_coupling", "wf_overlaps")

    def __new__(cls, *args, **kwargs):
        method = kwargs.get("method_es", args[4] if len(args) > 4 else "casscf")
        if cls is QCLabPySCFInterface and str(method).lower() == "tddft":
            return object.__new__(QCLabPySCFTDAInterface)
        return object.__new__(cls)

    def __init__(
        self,
        atom_positions,
        atom_masses,
        atom_names,
        num_quantum_states,
        method_es="casscf",
        basis="sto-3g",
        charge=0,
        multiplicity=1,
        ncas=None,
        nelecas=None,
        active_orbitals=None,
        weights=None,
        scf_options=None,
        casscf_options=None,
        response_options=None,
        max_memory=4000,
        num_threads=1,
        otxc="tPBE",
        grids_level=5,
        cms_options=None,
        tda=True,
        xc="pbe",
        excited_multiplicity=1,
        excited_state_indices=None,
        use_etfs=True,
        td_options=None,
        cart=False,
        grids_options=None,
    ):
        _require_pyscf()
        self.method_es = str(method_es).lower()
        if self.method_es not in {"casscf", "cms-pdft"}:
            raise ValueError("PySCF method_es supports 'casscf', 'cms-pdft', and 'tddft' with tda=True.")
        is_cms = self.method_es == "cms-pdft"
        if is_cms and mcpdft is None:
            raise ImportError(
                "CMS-PDFT requires the pyscf.mcpdft module supplied by PySCF 2.14.x. "
                "Install it with pip install 'qclab[pyscf]'."
            )
        functionals = {"tpbe": "tPBE", "tblyp": "tBLYP", "ftpbe": "ftPBE"}
        if not isinstance(otxc, str) or otxc.lower() not in functionals:
            raise ValueError("otxc supports only 'tPBE', 'tBLYP', and 'ftPBE'.")
        self.otxc = functionals[otxc.lower()]
        self.grids_level = _integer(grids_level, "grids_level")
        if self.grids_level > 9:
            raise ValueError("grids_level must be between 0 and 9.")
        self.cms_options = dict(cms_options or {})
        unknown_cms = set(self.cms_options).difference(
            {"conv_tol_diabatize", "max_cyc_diabatize", "sing_tol_diabatize", "nudge_tol_diabatize"}
        )
        if unknown_cms:
            raise ValueError(
                "cms_options supports only conv_tol_diabatize, max_cyc_diabatize, "
                "sing_tol_diabatize, and nudge_tol_diabatize."
            )
        for name, value in self.cms_options.items():
            if name == "max_cyc_diabatize":
                self.cms_options[name] = _integer(value, "cms_options max_cyc_diabatize", 1)
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, numbers.Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("cms_options %s must be a finite positive real number." % name)
        if self.cms_options and not is_cms:
            raise ValueError("cms_options requires method_es='cms-pdft'.")
        self.num_quantum_states = _integer(num_quantum_states, "num_quantum_states", 1)
        if is_cms and self.num_quantum_states < 2:
            raise ValueError("CMS-PDFT requires num_quantum_states to be at least two.")
        self.ncas = _integer(ncas, "ncas", 1)
        self.charge = _integer(charge, "charge", None)
        self.multiplicity = _integer(multiplicity, "multiplicity", 1)
        self.num_threads = _integer(num_threads, "num_threads", 1)
        self.atom_names = tuple(atom_names)
        if not self.atom_names:
            raise ValueError("atom_names must contain at least one atom.")
        if np.iscomplexobj(atom_masses):
            raise ValueError("atom_masses must be real.")
        self.atom_masses = np.asarray(atom_masses, dtype=float).copy()
        if self.atom_masses.shape != (len(self.atom_names),) or not np.all(
            np.isfinite(self.atom_masses) & (self.atom_masses > 0)
        ):
            raise ValueError("atom_masses must contain one positive electron mass per atom.")
        self.atom_positions = self._positions(atom_positions)
        self.basis = copy.deepcopy(basis)
        self.max_memory = float(max_memory)
        if not np.isfinite(self.max_memory) or self.max_memory <= 0:
            raise ValueError("max_memory must be a positive number of MB.")
        spin = self.multiplicity - 1
        if isinstance(nelecas, numbers.Integral):
            total = _integer(nelecas, "nelecas")
            if total < spin or (total - spin) % 2:
                raise ValueError("nelecas is incompatible with the spin multiplicity.")
            self.nelecas = ((total + spin) // 2, (total - spin) // 2)
        else:
            if not isinstance(nelecas, (tuple, list, np.ndarray)) or np.ndim(nelecas) != 1 or len(nelecas) != 2:
                raise ValueError("nelecas must be an electron count or (alpha, beta) pair.")
            self.nelecas = tuple(_integer(n, "nelecas") for n in nelecas)
        if self.nelecas[0] - self.nelecas[1] != spin or max(self.nelecas) > self.ncas:
            raise ValueError("Active electrons must fit ncas and the spin multiplicity.")
        alpha, beta = self.nelecas
        spin_dimension = math.comb(self.ncas, alpha) * math.comb(self.ncas, beta)
        if alpha < self.ncas and beta > 0:
            spin_dimension -= math.comb(self.ncas, alpha + 1) * math.comb(self.ncas, beta - 1)
        if self.num_quantum_states > spin_dimension:
            raise ValueError(
                "num_quantum_states exceeds the %d roots available for this active space and spin."
                % spin_dimension
            )
        self.weights = np.ones(self.num_quantum_states) / self.num_quantum_states
        if weights is not None and (
            np.shape(weights) != self.weights.shape
            or not np.allclose(weights, self.weights, rtol=0, atol=1e-12)
        ):
            raise ValueError("Only normalized, equal state-average weights are supported.")
        self.active_orbitals = None
        if active_orbitals is not None:
            self.active_orbitals = tuple(_integer(i, "active_orbitals") for i in active_orbitals)
            if len(set(self.active_orbitals)) != self.ncas:
                raise ValueError("active_orbitals must contain ncas distinct orbital indices.")
            if len(self.active_orbitals) != self.ncas:
                raise ValueError("active_orbitals must contain exactly ncas orbital indices.")
        self.scf_options = dict(scf_options or {})
        self.casscf_options = dict(casscf_options or {})
        self.response_options = {"conv_atol": 1e-11} if is_cms else {}
        self.response_options.update(response_options or {})
        unknown_response = set(self.response_options).difference(
            {"conv_atol", "conv_rtol", "max_cycle", "level_shift"}
        )
        if unknown_response:
            raise ValueError(
                "response_options supports only conv_atol, conv_rtol, max_cycle, and level_shift."
            )
        self.results = {}
        self._mc = None
        self._snapshot = None
        self._gradients = {}
        self._couplings = {}
        self._gradient_solver = None
        self._coupling_solver = None

    def _positions(self, positions):
        if np.iscomplexobj(positions):
            raise ValueError("atom_positions must be real Cartesian coordinates.")
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (len(self.atom_names), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("atom_positions must be a finite (atoms, 3) array in Bohr.")
        return positions.copy()

    @staticmethod
    def _options(solver, options, protected=()):
        for name, value in options.items():
            if not isinstance(name, str) or name.startswith("_") or name in protected or not hasattr(solver, name) or callable(getattr(solver, name)):
                raise ValueError("Unsupported or structural PySCF solver option: %s" % name)
            setattr(solver, name, value)

    def _solve_scf(self, mf, positions):
        """Solve the reference, with one bounded Newton retry after failure.

        The retry retains the reference's method, numerical thresholds, and
        iteration limit. Removing its SOSCF decorator restores the ordinary
        RHF/ROHF or RKS object expected by the response solvers.
        """
        def finite_orbitals(solver):
            return all(
                value is not None and np.all(np.isfinite(value))
                for value in (solver.mo_coeff, solver.mo_occ, solver.mo_energy)
            )

        attempts = ["ordinary SCF"]
        cause = None
        try:
            mf.kernel()
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            cause = error
        valid = mf.converged and finite_orbitals(mf) and np.isfinite(mf.e_tot)
        if valid:
            return mf
        if not mf.converged and mf.max_cycle > 0 and finite_orbitals(mf):
            attempts.append("one Newton SCF retry")
            try:
                recovered = mf.newton()
                recovered.kernel(mf.mo_coeff, mf.mo_occ)
                if (recovered.converged and finite_orbitals(recovered)
                        and np.isfinite(recovered.e_tot)):
                    recovered = recovered.remove_soscf()
                    self._scf_recovery = {
                        "algorithm": "newton", "atom_positions": positions.copy(),
                        "max_cycle": mf.max_cycle, "conv_tol": mf.conv_tol,
                        "conv_tol_grad": mf.conv_tol_grad,
                    }
                    if not getattr(self, "_scf_recovery_reported", False):
                        logger.warning(
                            "PySCF %s reference SCF recovered with one Newton retry "
                            "for %s at coordinates (Bohr) %s.",
                            type(mf).__name__, self.method_es.upper(),
                            np.array2string(positions, precision=8),
                        )
                        self._scf_recovery_reported = True
                    return recovered
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                cause = error
        if len(positions) > 1:
            distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            minimum_distance = float(distances[np.triu_indices(len(positions), 1)].min())
        else:
            minimum_distance = None
        message = (
            "PySCF SCF did not converge after %s; reference=%s, method_es=%s, "
            "max_cycle=%s, conv_tol=%s, conv_tol_grad=%s, "
            "minimum_nuclear_distance_bohr=%s, atom_positions_bohr=%s. "
            "Adjust scf_options or inspect the failing geometry."
            % (" and ".join(attempts), type(mf).__name__, self.method_es,
               mf.max_cycle, mf.conv_tol, mf.conv_tol_grad, minimum_distance,
               np.array2string(positions, precision=10, separator=","))
        )
        if cause is not None:
            raise RuntimeError(message) from cause
        raise RuntimeError(message)

    def _run(self, positions, previous=None):
        is_cms = self.method_es == "cms-pdft"
        method_name = "CMS-PDFT" if is_cms else "CASSCF"
        mol = gto.M(
            atom=list(zip(self.atom_names, positions)),
            unit="Bohr",
            basis=self.basis,
            charge=self.charge,
            spin=self.multiplicity - 1,
            symmetry=False,
            verbose=0,
            max_memory=self.max_memory,
        )
        inactive = mol.nelectron - sum(self.nelecas)
        if inactive < 0 or inactive % 2:
            raise ValueError("Active electrons must leave a nonnegative, doubly occupied core.")
        if inactive // 2 + self.ncas > mol.nao_nr():
            raise ValueError("The requested active space exceeds the molecular orbital basis.")
        mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
        mf.chkfile = None
        mf.conv_tol = 1e-13 if is_cms else 1e-10
        if is_cms:
            mf.max_cycle = 200
        self._options(mf, self.scf_options, ("mol", "mo_coeff", "mo_occ", "chkfile"))
        mf = self._solve_scf(mf, positions)
        if is_cms:
            mc = mcpdft.CASSCF(
                mf, self.otxc, self.ncas, self.nelecas, grids_level=self.grids_level
            )
        else:
            mc = mcscf.CASSCF(mf, self.ncas, self.nelecas)
        if mol.spin == 0:
            mc.fcisolver = fci.direct_spin0.FCI(mol)
        mc.fcisolver.conv_tol = 1e-14 if is_cms else 1e-12
        if is_cms:
            mc.fcisolver.conv_tol_residual = 1e-12
        mc.fcisolver.max_cycle = 200
        mc.chkfile = None
        mc.conv_tol = 1e-12 if is_cms else 1e-10
        mc.conv_tol_grad = 1e-8 if is_cms else 1e-7
        # The default AH residual can stall orbital optimization above 1e-7.
        mc.ah_conv_tol = 1e-14
        mc.max_cycle_macro = 200 if is_cms else 100
        # An Sz sector alone does not fix total spin for excited roots.
        spin = 0.5 * (self.multiplicity - 1)
        mc.fix_spin_(ss=spin * (spin + 1))
        if is_cms:
            mc = mc.multi_state(self.weights, method="CMS")
            mc.conv_tol_diabatize = 1e-12
            mc.max_cyc_diabatize = 200
            self._options(mc, self.cms_options)
        elif self.num_quantum_states > 1:
            mc.state_average_(self.weights)
        self._options(
            mc, self.casscf_options,
            ("mol", "_scf", "fcisolver", "ncas", "nelecas", "ncore", "weights", "frozen", "with_df",
             "mo_coeff", "ci", "chkfile", "natorb", "canonicalization",
             "otfnal", "otxc", "grids", "si", "si_pdft", "si_mcscf",
             "heff_mcscf", "hdiag_pdft", "diabatization",
             "conv_tol_diabatize", "max_cyc_diabatize",
             "sing_tol_diabatize", "nudge_tol_diabatize"),
        )
        orbitals = mf.mo_coeff
        ci_guess = None
        # Independent CMS optimizations avoid projected-guess bias in small
        # displacements. The previous physical states only determine phases.
        if previous is not None and not is_cms:
            orbitals = mcscf.addons.project_init_guess(
                mc, previous["mo_coeff"], prev_mol=gto.loads(previous["mol"])
            )
            ci_guess = [ci.copy() for ci in previous["ci"]]
            if self.num_quantum_states == 1:
                ci_guess = ci_guess[0]
        elif self.active_orbitals is not None:
            if max(self.active_orbitals) >= orbitals.shape[1]:
                raise ValueError("active_orbitals contains an index outside the HF orbital basis.")
            orbitals = mc.sort_mo(list(self.active_orbitals), base=0)
        mc.kernel(orbitals, ci0=ci_guess)
        if not mc.converged:
            options_name = "casscf_options, cms_options," if is_cms else "casscf_options"
            raise RuntimeError(
                "PySCF %s did not converge; adjust %s or the active space."
                % (method_name, options_name)
            )
        ci_vectors = mc.get_ci_adiabats(uci="MSPDFT") if is_cms else (
            list(mc.ci) if self.num_quantum_states > 1 else [mc.ci]
        )
        energies = np.atleast_1d(mc.e_states if self.num_quantum_states > 1 else mc.e_tot)
        if len(ci_vectors) != self.num_quantum_states or energies.shape != (self.num_quantum_states,):
            raise RuntimeError("PySCF did not return the requested number of %s roots." % method_name)
        ci_shape = tuple(fci.cistring.num_strings(self.ncas, n) for n in self.nelecas)
        if (
            not np.all(np.isfinite(energies))
            or np.iscomplexobj(energies)
            or not np.all(np.isfinite(mc.mo_coeff))
            or np.iscomplexobj(mc.mo_coeff)
            or any(np.shape(ci) != ci_shape or not np.all(np.isfinite(ci)) or np.iscomplexobj(ci)
                   for ci in ci_vectors)
        ):
            raise RuntimeError("PySCF returned invalid energies, orbitals, or %s coefficients." % method_name)
        for ci in ci_vectors:
            s_squared = fci.spin_op.spin_square(ci, self.ncas, self.nelecas)[0]
            if abs(s_squared - spin * (spin + 1)) > 1e-5:
                raise RuntimeError("A %s root has the wrong spin; choose a different active space." % method_name)
        snapshot = {
            "method_es": self.method_es,
            "mol": mol.dumps(),
            "atom_positions": positions.copy(),
            "mo_coeff": mc.mo_coeff.copy(),
            "ci": tuple(np.asarray(ci).copy() for ci in ci_vectors),
            "ncore": mc.ncore,
            "ncas": self.ncas,
            "nelecas": self.nelecas,
            "energies": energies.copy(),
        }
        if is_cms:
            intermediate = tuple(np.asarray(ci).copy() for ci in mc.ci)
            si = np.asarray(mc.si)
            if (
                len(intermediate) != self.num_quantum_states
                or any(ci.shape != ci_shape or np.iscomplexobj(ci) or not np.all(np.isfinite(ci))
                       for ci in intermediate)
                or si.shape != (self.num_quantum_states, self.num_quantum_states)
                or np.iscomplexobj(si)
                or not np.all(np.isfinite(si))
                or not np.allclose(si.T @ si, np.eye(self.num_quantum_states), atol=1e-8, rtol=0)
            ):
                raise RuntimeError("PySCF returned invalid CMS-PDFT intermediate states or SI coefficients.")
            snapshot.update(
                ci_intermediate=intermediate,
                si=si.copy(),
                otxc=self.otxc,
                grids_level=self.grids_level,
            )
        if previous is not None:
            diagonal = np.diag(wavefunction_overlaps(previous, snapshot))
            signs = np.where(diagonal < 0, -1.0, 1.0)
        else:
            signs = np.array([1.0 if ci.ravel()[np.argmax(abs(ci))] >= 0 else -1.0 for ci in ci_vectors])
        self._apply_phase_signs(mc, snapshot, signs)
        return mc, snapshot

    def _apply_phase_signs(self, mc, snapshot, signs):
        """Keep the physical snapshots and response solvers in the same gauge."""
        snapshot["ci"] = tuple(ci * sign for ci, sign in zip(snapshot["ci"], signs))
        if self.method_es == "cms-pdft":
            # CMS response uses the intermediate basis and final SI columns.
            # Rotating mc.ci to the adiabatic basis corrupts that representation.
            mc.si = np.asarray(mc.si).copy() * signs[None, :]
            snapshot["si"] = mc.si.copy()
        else:
            mc.ci = list(snapshot["ci"]) if self.num_quantum_states > 1 else snapshot["ci"][0]

    def _validate_previous(self, previous):
        method = previous.get("method_es", "casscf")
        if method != self.method_es:
            raise ValueError("amplitudes_previous method_es does not match this calculator.")
        if len(previous["ci"]) != self.num_quantum_states:
            raise ValueError("amplitudes_previous has a different number of quantum states.")
        if self.method_es == "cms-pdft" and (
            previous.get("otxc") != self.otxc or previous.get("grids_level") != self.grids_level
        ):
            raise ValueError("amplitudes_previous CMS-PDFT functional or grids_level does not match.")

    def _ensure_solved(self, positions, previous=None):
        if previous is not None:
            self._validate_previous(previous)
        if self._snapshot is not None and np.array_equal(positions, self._snapshot["atom_positions"]):
            if previous is not None:
                diagonal = np.diag(wavefunction_overlaps(previous, self._snapshot))
                signs = np.where(diagonal < 0, -1.0, 1.0)
                if np.any(signs < 0):
                    self._snapshot = copy.deepcopy(self._snapshot)
                    self._apply_phase_signs(self._mc, self._snapshot, signs)
                    self._couplings = {}
                    self._gradient_solver = None
                    self._coupling_solver = None
            return
        if previous is None:
            previous = self._snapshot
        mc, snapshot = self._run(positions, previous)
        self._mc, self._snapshot = mc, snapshot
        self.atom_positions = positions.copy()
        self._gradients = {}
        self._couplings = {}
        self._gradient_solver = None
        self._coupling_solver = None

    def _state_indices(self, indices):
        if indices is None:
            return list(range(self.num_quantum_states))
        if isinstance(indices, numbers.Integral):
            indices = [indices]
        indices = [_integer(i, "state index") for i in indices]
        if any(i >= self.num_quantum_states for i in indices):
            raise ValueError("State index is outside num_quantum_states.")
        return indices

    def _gradient(self, state_ind):
        if state_ind not in self._gradients:
            if self._gradient_solver is None:
                self._gradient_solver = self._mc.nuc_grad_method()
                self._options(self._gradient_solver, self.response_options, ("base", "mol", "state"))
            if self.num_quantum_states == 1:
                gradient = self._gradient_solver.kernel()
            else:
                gradient = self._gradient_solver.kernel(state=state_ind)
            if hasattr(self._gradient_solver, "converged") and not self._gradient_solver.converged:
                raise RuntimeError("PySCF %s gradient response did not converge." % self.method_es.upper())
            self._gradients[state_ind] = self._response_array(gradient, "gradient")
        return self._gradients[state_ind].copy()

    def _coupling(self, bra, ket):
        if bra == ket:
            return np.zeros_like(self.atom_positions)
        if (bra, ket) not in self._couplings:
            if np.isclose(self._snapshot["energies"][bra], self._snapshot["energies"][ket], rtol=0, atol=1e-12):
                raise ValueError(
                    "Derivative couplings are undefined at an exactly degenerate %s root pair."
                    % self.method_es.upper()
                )
            # PySCF 2.14's NAC core term assumes a spin-summed RHF gradient.
            # ROHF uses the same restricted orbitals, but its UHF-style gradient
            # has an extra spin axis. A temporary view supplies the core term
            # without changing the converged CASSCF solution or its spin sector.
            reference = self._mc._scf
            if self.multiplicity > 1:
                reference = reference.view(scf.hf.RHF)
            with lib.temporary_env(self._mc, _scf=reference):
                if self._coupling_solver is None:
                    self._coupling_solver = self._mc.nac_method()
                    self._options(
                        self._coupling_solver, self.response_options,
                        ("base", "mol", "state", "use_etfs", "mult_ediff"),
                    )
                # The 2.14 energy-gap sign makes state=(i,j) agree with
                # <i|d j> from cross-geometry overlap derivatives.
                native = self._coupling_solver.kernel(
                    state=(bra, ket), use_etfs=False, mult_ediff=False
                )
            if not self._coupling_solver.converged:
                raise RuntimeError(
                    "PySCF %s derivative-coupling response did not converge." % self.method_es.upper()
                )
            coupling = self._response_array(native, "derivative coupling") * ANGSTROM_TO_BOHR
            self._couplings[bra, ket] = coupling.copy()
            self._couplings[ket, bra] = -coupling.copy()
        return self._couplings[bra, ket].copy()

    def _response_array(self, value, name):
        array = np.real_if_close(np.asarray(value))
        if (
            array.shape != self.atom_positions.shape
            or np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
        ):
            raise RuntimeError("PySCF returned an invalid %s; expected finite real (atoms, 3) values." % name)
        return array.copy()

    def calculate(self, property_dict=None, atom_positions=None, **properties):
        """Calculate requested properties and return a fresh result dictionary.

        ``property_dict`` uses the same per-property arguments as the QC Lab
        ingredient, after removing its batch axis and converting ``z`` to Bohr.
        Wavefunction overlap arguments include ``amplitudes_previous`` and,
        optionally, ``atom_positions_previous`` for consistency checking.
        ``amplitudes_current`` is superseded by the current solved wavefunction.
        """
        requests = dict(property_dict or {})
        requests.update(properties)
        if not requests:
            requests = {"energy": {}}
        unknown = set(requests).difference(self.implemented_properties)
        if unknown:
            raise ValueError("Unsupported PySCF properties: %s" % ", ".join(sorted(unknown)))
        results = {name: None for name in requests}
        active = {name: dict(args) for name, args in requests.items() if args.get("calc_property", True)}
        if not active:
            self.results = results
            return results
        positions = self._positions(self.atom_positions if atom_positions is None else atom_positions)
        previous = active.get("wf_overlaps", {}).get("amplitudes_previous")
        if "wf_overlaps" in active:
            if previous is None:
                raise ValueError("wf_overlaps requires amplitudes_previous from an earlier calculation.")
            expected = active["wf_overlaps"].get("atom_positions_previous")
            if expected is not None and not np.allclose(previous["atom_positions"], expected, rtol=0, atol=1e-10):
                raise ValueError("amplitudes_previous does not match the requested previous geometry.")
        with lib.with_omp_threads(self.num_threads):
            self._ensure_solved(positions, previous)
            for name, args in active.items():
                if name == "energy":
                    results[name] = self._snapshot["energies"].copy()
                    if args.get("excited_amplitudes", False):
                        results["excited_amplitudes"] = copy.deepcopy(self._snapshot)
                elif name == "gradient":
                    gradient = np.zeros((*positions.shape, self.num_quantum_states))
                    for index in self._state_indices(args.get("state_inds_gradient")):
                        gradient[:, :, index] = self._gradient(index)
                    results[name] = gradient
                elif name == "derivative_coupling":
                    indices = args.get("state_inds_derivative_coupling")
                    if isinstance(indices, numbers.Integral):
                        indices = list(range(_integer(indices, "state_inds_derivative_coupling") + 1))
                    pairs = itertools.combinations(self._state_indices(indices), 2)
                    results[name] = {(i, j): self._coupling(i, j) for i, j in pairs}
                elif name == "wf_overlaps":
                    results[name] = wavefunction_overlaps(previous, self._snapshot)
        self.results = results
        return results

    def normal_modes(self, state_ind=0, delta=1e-3):
        """Return model-ready vibrational constants from a numerical Hessian.

        ``delta`` is a central-difference displacement in Bohr. Frequencies
        are atomic-unit angular frequencies, and the columns of ``normal_mode``
        are orthonormal mass-weighted eigenvectors. Unstable frequencies remain
        imaginary; they must not be passed to the Wigner sampler.
        The supplied reference geometry is not optimized automatically.
        """
        state_ind = _integer(state_ind, "state_ind")
        self._state_indices(state_ind)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be a positive displacement in Bohr.")
        positions = self.atom_positions.copy()
        with lib.with_omp_threads(self.num_threads):
            self._ensure_solved(positions)
            reference = copy.deepcopy(self._snapshot)
            hessian = np.empty((positions.size, positions.size))
            for coordinate in range(positions.size):
                gradients = []
                for direction in (1, -1):
                    displaced = positions.copy()
                    displaced.flat[coordinate] += direction * delta
                    # Displaced solves leave the trajectory cache untouched.
                    mc, _ = self._run(displaced, reference)
                    solver = mc.nuc_grad_method()
                    self._options(solver, self.response_options, ("base", "mol", "state"))
                    gradient = solver.kernel(state=state_ind) if self.num_quantum_states > 1 else solver.kernel()
                    if hasattr(solver, "converged") and not solver.converged:
                        raise RuntimeError("PySCF gradient response did not converge during the numerical Hessian.")
                    gradients.append(self._response_array(gradient, "Hessian gradient").ravel())
                hessian[:, coordinate] = (gradients[0] - gradients[1]) / (2 * delta)
        hessian = (hessian + hessian.T) * 0.5
        return _harmonic_modes(hessian, positions, self.atom_masses)


# Analytic TDA grid response.

# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
# Copyright 2026 QC Lab contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# The moving-grid and partition-weight contractions follow PySCF's
# grad/rks.py:get_vxc_full_response. Modified for the spin-resolved TDA
# stationary Lagrangian, including its relaxed and transition densities.

def _density_and_spatial_derivative(ao, density, lda):
    """Evaluate density variables and their spatial derivatives on a grid.

    GGA variables are ``(rho, d_x rho, d_y rho, d_z rho)``. Their spatial
    derivatives require AO derivatives through second order. The ordering
    below is PySCF's ``deriv=2`` AO ordering: value, x,y,z, xx,xy,xz,yy,yz,zz.
    """
    density = (density + density.T) * 0.5
    value = np.dot(ao[0], density)
    rho = np.empty((4, ao.shape[1]))
    rho[0] = np.einsum("gi,gi->g", value, ao[0])
    rho[1:] = 2 * np.einsum("gi,agi->ag", value, ao[1:4])
    derivative = np.empty((3, 4, ao.shape[1]))
    derivative[:, 0] = rho[1:]
    hessian_indices = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    for axis in range(3):
        first = np.dot(ao[axis + 1], density)
        for component in range(3):
            derivative[axis, component + 1] = 2 * (
                np.einsum("gi,gi->g", value, ao[hessian_indices[axis][component]])
                + np.einsum("gi,gi->g", first, ao[component + 1])
            )
    return (rho[:1], derivative[:, :1]) if lda else (rho, derivative)


def grid_response_correction(
    td, delta_density, transition_density=None, singlet=None,
    include_ground=True, block_size=4096,
):
    r"""Return missing atomic-grid response in Hartree/Bohr.

    The native CPU TDA gradient differentiates AOs on a fixed quadrature.
    This adds the derivative of its weights and the movement of each
    atom-centered grid. It involves no displaced geometries.

    The XC part of the stationary Lagrangian is

    .. math::

        L_{xc} = E_{xc}[\rho_0]
          + \sum_\sigma v_\sigma \rho_{D\sigma}
          + \sum_{\sigma\tau} f_{\sigma\tau}
                         \rho_{T\sigma}\rho_{T\tau}.

    Spin densities are ``rho_0/2``, relaxed densities ``rho_D/2``, and
    transition densities ``(rho_T, +rho_T)`` for singlets or
    ``(rho_T, -rho_T)`` for triplets. Thus the singlet kernel term is
    ``4 * f_charge * rho_T**2``, with PySCF's restricted ``2*X.T@X=1``
    normalization. GGA Cartesian density gradients are included in every
    contraction.

    ``delta_density`` is the native gradient's ``dmz1doo = dmzoo + z1ao``;
    ``transition_density`` is its ``dmxpy``. The real symmetric parts suffice.
    For a ground-to-excited coupling use its relaxed AO density, omit the
    transition density, and set ``include_ground=False``. The caller divides
    that force contribution by the excitation energy.
    """
    if rks_grad is None:
        raise ImportError("Analytic TDA grid response requires PySCF 2.14.x.")
    mf, mol = td._scf, td.mol
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return np.zeros((mol.natm, 3))
    if xctype not in ("LDA", "GGA"):
        raise NotImplementedError("Analytic TDA grid response supports LDA and GGA XC functionals.")
    if (isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, numbers.Integral) or block_size < 1):
        raise ValueError("TDA grid-response block_size must be a positive integer.")
    nao = mol.nao_nr()
    delta_density = np.asarray(delta_density)
    has_transition = transition_density is not None
    transition_density = (np.zeros((nao, nao)) if transition_density is None
                          else np.asarray(transition_density))
    for density in (delta_density, transition_density):
        if (density.shape != (nao, nao) or np.iscomplexobj(density)
                or not np.all(np.isfinite(density))):
            raise ValueError("TDA grid-response densities must be finite real AO matrices.")
    if singlet is None:
        singlet = td.singlet
    lda = xctype == "LDA"
    ground_density = mf.make_rdm1()
    correction = np.zeros((mol.natm, 3))
    signs = np.array([1.0, 1.0 if singlet else -1.0])
    for atom, (coords, weights, weights_derivative) in enumerate(
        rks_grad.grids_response_cc(mf.grids)
    ):
        for start in range(0, len(coords), block_size):
            stop = min(start + block_size, len(coords))
            ao = ni.eval_ao(mol, coords[start:stop], deriv=2)
            rho0, grad0 = _density_and_spatial_derivative(ao, ground_density, lda)
            delta, grad_delta = _density_and_spatial_derivative(ao, delta_density, lda)
            transition, grad_transition = _density_and_spatial_derivative(
                ao, transition_density, lda,
            )
            rho = np.repeat(rho0[None], 2, axis=0) * 0.5
            rho_delta = np.repeat(delta[None], 2, axis=0) * 0.5
            rho_transition = signs[:, None, None] * transition[None]
            exc, vxc, fxc, kxc = ni.eval_xc_eff(
                mf.xc, rho, deriv=3 if has_transition else 2,
                xctype=xctype, spin=1,
            )
            integrand = exc * rho0[0] if include_ground else np.zeros_like(exc)
            integrand += np.einsum("sug,sug->g", vxc, rho_delta)
            if has_transition:
                integrand += np.einsum(
                    "sutvg,sug,tvg->g", fxc, rho_transition, rho_transition,
                )
            grad_rho = np.repeat(grad0[:, None], 2, axis=1) * 0.5
            grad_delta = np.repeat(grad_delta[:, None], 2, axis=1) * 0.5
            grad_transition = signs[None, :, None, None] * grad_transition[:, None]
            first_ground = vxc.copy() if include_ground else np.zeros_like(vxc)
            first_ground += np.einsum("sutvg,tvg->sug", fxc, rho_delta)
            if has_transition:
                first_ground += np.einsum(
                    "sutvwzg,tvg,wzg->sug", kxc, rho_transition, rho_transition,
                )
            spatial = np.einsum("sug,asug->ag", first_ground, grad_rho)
            spatial += np.einsum("sug,asug->ag", vxc, grad_delta)
            if has_transition:
                first_transition = 2 * np.einsum("sutvg,tvg->sug", fxc, rho_transition)
                spatial += np.einsum("sug,asug->ag", first_transition, grad_transition)
            # Weight derivatives affect every atom; coordinate movement affects
            # the atom carrying this batch of radial/angular grid points.
            correction += np.einsum(
                "g,nag->na", integrand, weights_derivative[:, :, start:stop],
            )
            correction[atom] += np.einsum("ag,g->a", spatial, weights[start:stop])
    if not np.all(np.isfinite(correction)):
        raise RuntimeError("TDA analytic grid response returned nonfinite forces.")
    return correction


def attach_grid_response(gradient):
    """Add the analytic grid correction through PySCF's ``extra_force`` hook.

    The native gradient exposes its relaxed density after the CPKS solve.
    Retaining that matrix as the cache key computes one correction per
    gradient evaluation and avoids caching across different polarized states.
    Existing extension forces are preserved.
    """
    if getattr(gradient, "_qclab_grid_response_attached", False):
        return gradient
    original = gradient.extra_force
    cache = {}

    def extra_force(atom, environment):
        if cache.get("delta_density") is not environment["dmz1doo"]:
            cache["delta_density"] = environment["dmz1doo"]
            cache["correction"] = grid_response_correction(
                gradient.base, environment["dmz1doo"], environment["dmxpy"],
                environment["singlet"],
            )
        return original(atom, environment) + cache["correction"][atom]

    gradient.extra_force = extra_force
    gradient._qclab_grid_response_attached = True
    return gradient


# Analytic TDA orbital response and derivative couplings.

# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
# Copyright 2026 QC Lab contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# This CPU adaptation of GPU4PySCF nac/tdrks.py and nac/tdrhf.py uses
# bilinear polarization of PySCF's analytic gradient Lagrangian for excited
# pairs. The CPU triplet gradient supplies its spin-resolved XC kernels.
# Modified from upstream: NumPy/CPU contractions, triplet support, response
# residual checks, and the QC Lab <bra|d ket> convention.
# References: Li, Suo and Liu, JCP 141, 244105 (2014),
# doi:10.1063/1.4903986; Ou et al., JCP 141, 024114 (2014),
# doi:10.1063/1.4887256; Fatehi et al., JCP 135, 234105 (2011),
# doi:10.1063/1.3665031.

_RESPONSE_LOCK = threading.RLock()


def _response_options(options):
    values = {"conv_atol": 1e-10, "conv_rtol": 0.0, "max_cycle": 100, "level_shift": 0.0}
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError("TDA response_options must be a dictionary.")
        unknown = set(options) - set(values)
        if unknown:
            raise ValueError("Unsupported TDA response_options: %s." % sorted(unknown))
        values.update(options)
    for key in ("conv_atol", "conv_rtol", "level_shift"):
        value = values[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError("TDA response option %s must be a finite real number." % key)
        if not np.isfinite(value) or value < 0 or (key == "conv_atol" and value == 0):
            raise ValueError("TDA response tolerances must be finite and nonnegative; conv_atol > 0.")
    cycles = values["max_cycle"]
    if isinstance(cycles, (bool, np.bool_)) or not isinstance(cycles, numbers.Integral) or cycles < 0:
        raise ValueError("TDA response max_cycle must be a nonnegative integer.")
    return values


@contextmanager
def response_context(response_options=None):
    """Apply CPKS controls and check its unpreconditioned equation residual.

    PySCF's gradient implementation resolves ``cphf.solve`` from its module.
    A locked, temporary override retains that implementation and restores the
    module even when a calculation fails. The lock also serializes nested use.
    """
    if cphf is None:
        raise ImportError("Analytic TDA response requires PySCF 2.14.x.")
    options = _response_options(response_options)
    with _RESPONSE_LOCK:
        native_solve = cphf.solve

        def checked_solve(fvind, mo_energy, mo_occ, h1, s1=None, **kwargs):
            if options["max_cycle"] == 0:
                raise RuntimeError("TDA CPKS response did not converge: max_cycle=0.")
            energies = np.asarray(mo_energy)
            occupied = np.asarray(mo_occ) > 0
            gap = energies[~occupied, None] - energies[None, occupied]
            kwargs.update(
                max_cycle=options["max_cycle"],
                tol=options["conv_atol"] * 0.1 / max(1.0, np.max(np.abs(gap), initial=0.0)),
                level_shift=options["level_shift"],
            )
            try:
                result = native_solve(fvind, mo_energy, mo_occ, h1, s1=s1, **kwargs)
            except (RuntimeError, np.linalg.LinAlgError) as error:
                raise RuntimeError(
                    "TDA CPKS response did not converge; tighten the SCF or increase "
                    "response_options['max_cycle']."
                ) from error
            solution = np.asarray(result[0])
            if not np.all(np.isfinite(solution)):
                raise RuntimeError("TDA CPKS response returned nonfinite orbital derivatives.")
            rhs = np.asarray(h1)
            if s1 is not None:
                rhs = rhs - np.asarray(s1) * energies[occupied]

            def equation_residual():
                response = np.asarray(fvind(solution)).reshape(solution.shape)
                if s1 is None:
                    return gap * solution + response + rhs
                return (gap * solution[..., ~occupied, :] + response[..., ~occupied, :]
                        + rhs[..., ~occupied, :])

            residual = equation_residual()
            tolerance = options["conv_atol"] + options["conv_rtol"] * np.linalg.norm(rhs)
            # PySCF's absolute Krylov linear-dependence threshold can stop a
            # small RHS before the requested physical residual is reached.
            # Analytic iterative refinement rescales the correction problem;
            # no nuclear displacements or numerical derivatives are involved.
            for _ in range(3):
                magnitude = np.linalg.norm(residual)
                if not np.isfinite(magnitude) or magnitude <= tolerance:
                    break
                if s1 is None:
                    correction_response = fvind
                else:
                    def correction_response(vector):
                        full = np.zeros_like(solution)
                        full[..., ~occupied, :] = np.asarray(vector).reshape(residual.shape)
                        return np.asarray(fvind(full)).reshape(solution.shape)[..., ~occupied, :]
                correction = native_solve(
                    correction_response, mo_energy, mo_occ, residual / magnitude,
                    **kwargs,
                )[0] * magnitude
                if s1 is None:
                    solution = solution + correction
                else:
                    solution = solution.copy()
                    solution[..., ~occupied, :] += correction
                residual = equation_residual()
            if not np.all(np.isfinite(residual)) or np.linalg.norm(residual) > tolerance:
                raise RuntimeError(
                    "TDA CPKS response residual %.3g exceeds tolerance %.3g; tighten "
                    "SCF convergence or increase response_options['max_cycle']."
                    % (np.linalg.norm(residual), tolerance)
                )
            if s1 is None:
                return solution, result[1]
            response = np.asarray(fvind(solution)).reshape(solution.shape)
            energy_response = (
                (rhs + response)[..., occupied, :]
                + solution[..., occupied, :]
                * (energies[occupied, None] - energies[None, occupied])
            )
            return solution, energy_response

        with lib.temporary_env(cphf, solve=checked_solve):
            yield options


def gradient(td, state, response_options=None):
    """Evaluate a root gradient including quadrature and checked CPKS response."""

    grad = attach_grid_response(td.nuc_grad_method())
    with response_context(response_options):
        result = np.asarray(grad.kernel(state=state))
    if result.shape != (td.mol.natm, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("TDA gradient returned an invalid or nonfinite response array.")
    return result


def _overlap_connection(td, transition_density):
    """Antisymmetric AO-basis connection removed by electron translation factors."""
    s1 = td._scf.nuc_grad_method().get_ovlp(td.mol)
    antisymmetric = transition_density.T - transition_density
    return np.array([
        0.5 * np.einsum("xij,ij->x", s1[:, first:last], antisymmetric[first:last])
        for _, _, first, last in td.mol.aoslice_by_atom()
    ])


def _excited_pair(td, bra, ket, use_etfs):

    xi, xj = td.xy[bra - 1][0], td.xy[ket - 1][0]
    grad = attach_grid_response(td.nuc_grad_method())
    # The relaxed TDA Lagrangian is quadratic in X. Polarization gives its
    # mixed-state force exactly, including spin kernels and orbital response.
    force = 0.25 * (
        grad.grad_elec((xi + xj, 0), td.singlet)
        - grad.grad_elec((xi - xj, 0), td.singlet)
    )
    result = force / (td.e[ket - 1] - td.e[bra - 1])
    if not use_etfs:
        coeff = td._scf.mo_coeff
        occupied = td._scf.mo_occ > 0
        co, cv = coeff[:, occupied], coeff[:, ~occupied]
        transition_density = 2 * (
            -co @ (xj @ xi.T) @ co.T + cv @ (xi.T @ xj) @ cv.T
        )
        result += _overlap_connection(td, transition_density)
    return result


def _ground_pair(td, state, use_etfs):

    mf, mol = td._scf, td.mol
    coeff, energies, occupations = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    occupied = occupations > 0
    co, cv = coeff[:, occupied], coeff[:, ~occupied]
    nocc, nvir = co.shape[1], cv.shape[1]
    amplitude = td.xy[state - 1][0].T
    excitation = td.e[state - 1]
    vresp = mf.gen_response(singlet=None, hermi=1)

    def fvind(vector):
        density = cv @ np.asarray(vector).reshape(nvir, nocc) @ co.T * 2
        return (cv.T @ vresp(density + density.T) @ co).ravel()

    z = cphf.solve(fvind, energies, occupations, -amplitude * excitation)[0]
    z = z.reshape(nvir, nocc)
    density = cv @ z @ co.T
    density += density.T
    potential = coeff.T @ vresp(density) @ coeff
    energy_density = np.zeros((coeff.shape[1], coeff.shape[1]))
    energy_density[:nocc, :nocc] = potential[:nocc, :nocc]
    energy_density[:nocc, nocc:] = potential[:nocc, nocc:] + 0.5 * (z * energies[~occupied, None]).T
    energy_density[nocc:, :nocc] = 0.5 * amplitude * excitation + 0.5 * z * energies[None, occupied]
    energy_density = 2 * coeff @ energy_density @ coeff.T
    ground_density = 2 * co @ co.T
    mf_grad = mf.nuc_grad_method()
    grad = td.nuc_grad_method()
    _, _, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    vj, vk = grad.get_jk(mol, (ground_density, density))
    veff = vj - 0.5 * hyb * vk
    f1vo, _, vxc1, _ = tdrks._contract_xc_kernel(
        grad, mf.xc, density, None, True, False, True, grad.max_memory
    )
    s1 = mf_grad.get_ovlp(mol)
    hcore = mf_grad.hcore_generator(mol)
    xao = co @ amplitude.T @ cv.T
    result = np.zeros((mol.natm, 3))
    for atom, (_, _, first, last) in enumerate(mol.aoslice_by_atom()):
        result[atom] = np.einsum("xij,ij->x", hcore(atom), density)
        result[atom] -= np.einsum(
            "xij,ij->x", s1[:, first:last],
            (energy_density + energy_density.T)[first:last],
        )
        result[atom] += 2 * np.einsum("xij,ij->x", veff[0, :, first:last], density[first:last])
        result[atom] += 2 * np.einsum("xij,ij->x", veff[1, :, first:last], ground_density[first:last])
        result[atom] += 2 * np.einsum("xij,ij->x", vxc1[1:, first:last], density[first:last])
        result[atom] += np.einsum("xij,ij->x", f1vo[1:, first:last], ground_density[first:last])
        connection = xao + xao.T if use_etfs else 2 * xao.T
        result[atom] += excitation * np.einsum("xij,ij->x", s1[:, first:last], connection[first:last])
    result += grid_response_correction(td, density, include_ground=False)
    return result / excitation


def derivative_coupling(td, bra, ket, use_etfs=True, response_options=None):
    """Return ``<bra|d ket/dR>`` in inverse Bohr for native TDA root indices.

    Zero labels the closed-shell ground state. Excited labels are one-based
    indices into ``td.xy``. Opposite-spin couplings vanish without spin-orbit
    coupling; a requested degenerate same-spin pair raises an actionable error.
    """
    for index in (bra, ket):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, numbers.Integral):
            raise ValueError("TDA coupling root indices must be integers.")
        if not 0 <= index <= len(td.xy):
            raise ValueError("TDA coupling root index exceeds the solved roots.")
    if bra == ket or (not td.singlet and 0 in (bra, ket)):
        return np.zeros((td.mol.natm, 3))
    if bra > ket:
        return -derivative_coupling(td, ket, bra, use_etfs, response_options)
    energies = np.r_[0.0, td.e]
    if abs(energies[ket] - energies[bra]) < 1e-8:
        raise RuntimeError("TDA derivative coupling is undefined for a degenerate pair; select isolated roots.")
    with response_context(response_options):
        result = (_ground_pair(td, ket, use_etfs) if bra == 0
                  else _excited_pair(td, bra, ket, use_etfs))
    result = np.asarray(result)
    if result.shape != (td.mol.natm, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("TDA derivative coupling returned an invalid or nonfinite response array.")
    return result


# Restricted TDA wavefunctions, state overlaps, and calculator.

def _single_spin_overlaps(mo_overlap, nocc, previous, current):
    """Contract reference/single-excitation determinants for one spin.

    Orbital replacement stays in the original occupied slot, as required for
    the excitation operator a_a^dagger a_i. The usual inverse-occupied-block
    formula avoids forming the quadratic determinant matrix. A direct minor
    fallback retains valid overlaps when the occupied block is singular.
    """
    occupied = mo_overlap[:nocc, :nocc]
    determinant = np.linalg.det(occupied)
    singular = np.linalg.svd(occupied, compute_uv=False)
    if singular[-1] > 1e-10 * max(1.0, singular[0]):
        inverse = np.linalg.inv(occupied)
        left = mo_overlap[nocc:, :nocc] @ inverse
        right = inverse @ mo_overlap[:nocc, nocc:]
        schur = mo_overlap[nocc:, nocc:] - left @ mo_overlap[:nocc, nocc:]
        bra = np.einsum("rjb,bj->r", previous.conj(), left)
        ket = np.einsum("sia,ia->s", current, right)
        transformed = np.einsum("ij,sia,ba->sjb", inverse, current, schur, optimize=True)
        excited = determinant * (
            previous.conj().reshape(len(previous), previous.shape[1] * previous.shape[2])
            @ transformed.reshape(len(current), current.shape[1] * current.shape[2]).T
            + np.outer(bra, ket)
        )
        return determinant, determinant * ket, determinant * bra, excited

    nvir = mo_overlap.shape[0] - nocc
    replacements = [np.arange(nocc)]
    for orbital in range(nocc):
        for virtual in range(nvir):
            replacement = np.arange(nocc)
            replacement[orbital] = nocc + virtual
            replacements.append(replacement)
    determinants = np.empty((len(replacements), len(replacements)), dtype=mo_overlap.dtype)
    for row, bra_orbitals in enumerate(replacements):
        for column, ket_orbitals in enumerate(replacements):
            determinants[row, column] = np.linalg.det(
                mo_overlap[np.ix_(bra_orbitals, ket_orbitals)]
            )
    bra = previous.conj().reshape(len(previous), nocc * nvir)
    ket = current.reshape(len(current), nocc * nvir)
    return (
        determinants[0, 0],
        determinants[0, 1:] @ ket.T,
        bra @ determinants[1:, 0],
        bra @ determinants[1:, 1:] @ ket.T,
    )


def tda_wavefunction_overlaps(previous, current):
    """Return previous-bra/current-ket overlaps of normalized TDA states.

    State zero is the implicit doubly occupied reference determinant. Each
    remaining ``ci`` entry contains alpha/beta single-excitation coefficients,
    including every occupied orbital, rather than an exponentially large FCI
    expansion. Ground-state ``ci[0]`` is an array of zeros of the same shape.
    """
    _require_pyscf()
    if not previous.get("tda", False) or not current.get("tda", False):
        raise ValueError("Wavefunction snapshots must both describe TDA states.")
    for name in ("nocc", "xc", "excited_multiplicity", "excited_state_indices"):
        if previous.get(name) != current.get(name):
            raise ValueError("TDA wavefunction snapshots have different %s." % name)
    mol_previous, mol_current = gto.loads(previous["mol"]), gto.loads(current["mol"])
    if not np.array_equal(mol_previous.atom_charges(), mol_current.atom_charges()):
        raise ValueError("Wavefunction snapshots must describe the same atoms.")
    ao_overlap = gto.intor_cross("int1e_ovlp", mol_previous, mol_current)
    mo_overlap = previous["mo_coeff"].conj().T @ ao_overlap @ current["mo_coeff"]
    nocc = previous["nocc"]
    previous_ci, current_ci = np.asarray(previous["ci"]), np.asarray(current["ci"])
    shape = (2, nocc, mo_overlap.shape[0] - nocc)
    if (
        previous_ci.ndim != 4 or current_ci.ndim != 4
        or previous_ci.shape[1:] != shape or current_ci.shape[1:] != shape
        or mo_overlap.shape[0] != mo_overlap.shape[1]
    ):
        raise ValueError("TDA snapshots have incompatible orbital or amplitude dimensions.")
    alpha = _single_spin_overlaps(mo_overlap, nocc, previous_ci[1:, 0], current_ci[1:, 0])
    beta = _single_spin_overlaps(mo_overlap, nocc, previous_ci[1:, 1], current_ci[1:, 1])
    da, ga, ag, aa = alpha
    db, gb, bg, bb = beta
    overlaps = np.empty((len(previous_ci), len(current_ci)), dtype=np.result_type(mo_overlap, previous_ci, current_ci))
    overlaps[0, 0] = da * db
    overlaps[0, 1:] = ga * db + da * gb
    overlaps[1:, 0] = ag * db + da * bg
    overlaps[1:, 1:] = aa * db + da * bb + np.outer(ag, gb) + np.outer(bg, ga)
    return np.real_if_close(overlaps)


class QCLabPySCFTDAInterface(QCLabPySCFInterface):
    """Real restricted TDA backend selected by ``method_es='tddft'``.

    ``num_quantum_states`` includes the ground state. ``excited_state_indices``
    selects zero-based native TDA roots in the requested model order. Singlet
    or triplet excited states share one closed-shell singlet ground state.
    Full (non-TDA) TDDFT is intentionally rejected until its state transport
    and derivative-coupling conventions are supported together.
    """

    def __init__(
        self, atom_positions, atom_masses, atom_names, num_quantum_states,
        method_es="tddft", basis="sto-3g", charge=0, multiplicity=1,
        ncas=None, nelecas=None, active_orbitals=None, weights=None,
        scf_options=None, casscf_options=None, response_options=None,
        max_memory=4000, num_threads=1, otxc="tPBE", grids_level=5,
        cms_options=None, tda=True, xc="pbe", excited_multiplicity=1,
        excited_state_indices=None, use_etfs=True, td_options=None, cart=False,
        grids_options=None,
    ):
        _require_pyscf()
        self.method_es = str(method_es).lower()
        if self.method_es != "tddft":
            raise ValueError("TDA requires method_es='tddft'.")
        if not isinstance(tda, (bool, np.bool_)) or not tda:
            raise ValueError("Only TDA (tda=True) is supported; full TDDFT state transport and NACs are not implemented.")
        self.tda = True
        if not isinstance(xc, str) or xc.lower() not in {"pbe", "b3lyp"}:
            raise ValueError("xc supports only 'pbe' and 'b3lyp' for TDA.")
        self.xc = xc.lower()
        self.multiplicity = _integer(multiplicity, "multiplicity", 1)
        if self.multiplicity != 1:
            raise ValueError("TDA requires multiplicity=1 for a closed-shell restricted reference.")
        self.excited_multiplicity = _integer(excited_multiplicity, "excited_multiplicity", 1)
        if self.excited_multiplicity not in (1, 3):
            raise ValueError("excited_multiplicity must be 1 (singlets) or 3 (triplets).")
        for name, value in (("use_etfs", use_etfs), ("cart", cart)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("%s must be a boolean." % name)
        self.use_etfs, self.cart = bool(use_etfs), bool(cart)
        if any(value is not None for value in (ncas, nelecas, active_orbitals, weights)) or casscf_options or cms_options:
            raise ValueError("ncas, nelecas, active_orbitals, weights, casscf_options, and cms_options are not TDA settings.")
        self.num_quantum_states = _integer(num_quantum_states, "num_quantum_states", 1)
        if excited_state_indices is None:
            excited_state_indices = range(self.num_quantum_states - 1)
        if isinstance(excited_state_indices, (set, frozenset, dict, str, bytes)):
            raise ValueError("excited_state_indices must be an ordered sequence of zero-based root indices.")
        try:
            self.excited_state_indices = tuple(_integer(i, "excited_state_indices") for i in excited_state_indices)
        except TypeError as error:
            raise ValueError("excited_state_indices must be an ordered sequence of zero-based root indices.") from error
        if len(self.excited_state_indices) != self.num_quantum_states - 1 or len(set(self.excited_state_indices)) != len(self.excited_state_indices):
            raise ValueError("excited_state_indices must contain num_quantum_states-1 distinct zero-based root indices.")
        self.charge = _integer(charge, "charge", None)
        self.num_threads = _integer(num_threads, "num_threads", 1)
        self.atom_names = tuple(atom_names)
        if not self.atom_names:
            raise ValueError("atom_names must contain at least one atom.")
        if np.iscomplexobj(atom_masses):
            raise ValueError("atom_masses must be real.")
        self.atom_masses = np.asarray(atom_masses, dtype=float).copy()
        if self.atom_masses.shape != (len(self.atom_names),) or not np.all(np.isfinite(self.atom_masses) & (self.atom_masses > 0)):
            raise ValueError("atom_masses must contain one positive electron mass per atom.")
        self.atom_positions = self._positions(atom_positions)
        self.basis = copy.deepcopy(basis)
        self.max_memory = float(max_memory)
        if not np.isfinite(self.max_memory) or self.max_memory <= 0:
            raise ValueError("max_memory must be a positive number of MB.")
        self.grids_level = _integer(grids_level, "grids_level")
        if self.grids_level > 9:
            raise ValueError("grids_level must be between 0 and 9.")
        if grids_options is not None and not isinstance(grids_options, dict):
            raise ValueError("grids_options must be a dictionary.")
        self.grids_options = copy.deepcopy(dict(grids_options or {}))
        if set(self.grids_options).difference({"atom_grid", "prune"}):
            raise ValueError("grids_options supports only atom_grid and prune.")
        if "prune" in self.grids_options and self.grids_options["prune"] is not None:
            raise ValueError("grids_options prune supports only None; omit it to use PySCF's default pruning.")
        if "atom_grid" in self.grids_options:
            grid = self.grids_options["atom_grid"]

            def validate_grid(pair):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("grids_options atom_grid must contain (radial, angular) point counts.")
                radial = _integer(pair[0], "grids_options atom_grid radial count", 1)
                angular = _integer(pair[1], "grids_options atom_grid angular count", 1)
                if angular not in dft.gen_grid.LEBEDEV_NGRID:
                    raise ValueError("grids_options atom_grid angular count must be a supported Lebedev grid size.")
                return radial, angular

            if isinstance(grid, dict):
                if not grid or any(not isinstance(name, str) for name in grid):
                    raise ValueError("grids_options atom_grid must map atom symbols to point-count pairs.")
                if set(grid).difference(set(self.atom_names) | {"default"}):
                    raise ValueError("grids_options atom_grid contains an atom absent from atom_names.")
                self.grids_options["atom_grid"] = {name: validate_grid(pair) for name, pair in grid.items()}
            else:
                self.grids_options["atom_grid"] = validate_grid(grid)
        self.scf_options = dict(scf_options or {})
        unknown_scf = set(self.scf_options).difference({
            "conv_tol", "conv_tol_grad", "conv_tol_cpscf", "max_cycle",
            "diis", "diis_start_cycle", "diis_space", "diis_space_rollback",
            "diis_damp", "level_shift", "damp", "init_guess", "direct_scf",
            "direct_scf_tol", "max_memory", "small_rho_cutoff",
        })
        if unknown_scf:
            raise ValueError("scf_options contains unsupported or structural settings: %s" % ", ".join(sorted(map(str, unknown_scf))))
        if "small_rho_cutoff" in self.scf_options:
            cutoff = self.scf_options["small_rho_cutoff"]
            if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, numbers.Real) or cutoff != 0:
                raise ValueError("scf_options small_rho_cutoff must be zero for analytic grid response.")
        self.td_options = dict(td_options or {})
        unknown_td = set(self.td_options).difference(
            {"conv_tol", "max_cycle", "lindep", "level_shift", "deg_eia_thresh", "positive_eig_threshold"}
        )
        if unknown_td:
            raise ValueError("td_options contains unsupported or structural settings: %s" % ", ".join(sorted(map(str, unknown_td))))
        for name, value in self.td_options.items():
            if name == "max_cycle":
                self.td_options[name] = _integer(value, "td_options max_cycle")
            elif (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value < 0
                or (name in {"conv_tol", "lindep"} and value == 0)
            ):
                raise ValueError("td_options %s must be a finite numerical tolerance or shift." % name)
        self.response_options = {"conv_atol": 1e-10, "conv_rtol": 0.0, "max_cycle": 100, "level_shift": 0.0}
        self.response_options.update(response_options or {})
        unknown_response = set(self.response_options).difference({"conv_atol", "conv_rtol", "max_cycle", "level_shift"})
        if unknown_response:
            raise ValueError("response_options supports only conv_atol, conv_rtol, max_cycle, and level_shift.")
        for name, value in self.response_options.items():
            if name == "max_cycle":
                self.response_options[name] = _integer(value, "response_options max_cycle")
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0 or (name == "conv_atol" and value == 0):
                raise ValueError("response_options %s must be finite and %s." % (name, "positive" if name == "conv_atol" else "nonnegative"))
        self.results = {}
        self._mc = self._td = None
        self._snapshot = None
        self._gradients, self._couplings = {}, {}
        self._gradient_solver = self._coupling_solver = None

    def _run(self, positions, previous=None):
        mol = gto.M(
            atom=list(zip(self.atom_names, positions)), unit="Bohr", basis=self.basis,
            charge=self.charge, spin=None, cart=self.cart, symmetry=False,
            verbose=0, max_memory=self.max_memory,
        )
        if mol.nelectron <= 0 or mol.nelectron % 2:
            raise ValueError("TDA requires a positive even electron count and a closed-shell reference.")
        nocc = mol.nelectron // 2
        nvir = mol.nao_nr() - nocc
        native_roots = max(self.excited_state_indices, default=-1) + 1
        if nvir < 0 or native_roots > nocc * nvir:
            raise ValueError("excited_state_indices exceeds the TDA excitation space available in this basis.")
        mf = dft.RKS(mol)
        mf.xc, mf.grids.level = self.xc, self.grids_level
        for name, value in self.grids_options.items():
            setattr(mf.grids, name, copy.deepcopy(value))
        mf.chkfile, mf.conv_tol, mf.max_cycle = None, 1e-12, 200
        self._options(mf, self.scf_options, ("mol", "mo_coeff", "mo_occ", "mo_energy", "chkfile", "xc", "grids", "_numint", "with_df", "disp", "nlc"))
        mf = self._solve_scf(mf, positions)
        td = mf.TDA()
        td.nstates, td.singlet, td.chkfile = native_roots, self.excited_multiplicity == 1, None
        td.conv_tol, td.max_cycle = 1e-10, 100
        self._options(td, self.td_options)
        if native_roots:
            if td.max_cycle <= 0:
                raise RuntimeError("PySCF TDA did not converge; td_options max_cycle must allow solver iterations.")
            try:
                td.kernel()
            except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as error:
                raise RuntimeError("PySCF TDA solver failed; adjust td_options or the geometry.") from error
            if td.e is None or td.xy is None or len(td.e) != native_roots or len(td.xy) != native_roots or not np.all(td.converged):
                raise RuntimeError("PySCF TDA did not converge all requested roots; adjust td_options or the geometry.")
        else:
            td.e, td.xy, td.converged = np.empty(0), [], np.empty(0, dtype=bool)
        energies = np.concatenate(([mf.e_tot], mf.e_tot + np.asarray(td.e)[list(self.excited_state_indices)]))
        zero = np.zeros((2, nocc, nvir))
        coefficients = [zero]
        for index in self.excited_state_indices:
            x, y = td.xy[index]
            x = np.asarray(x)
            if x.shape != (nocc, nvir) or np.iscomplexobj(x) or not np.all(np.isfinite(x)) or not np.allclose(y, 0):
                raise RuntimeError("PySCF returned invalid real TDA excitation amplitudes.")
            if not np.isclose(2 * np.vdot(x, x).real, 1.0, rtol=0, atol=1e-7):
                raise RuntimeError("PySCF returned unnormalized TDA excitation amplitudes.")
            coefficients.append(np.stack((x, x if td.singlet else -x)))
        if not np.all(np.isfinite(energies)) or np.iscomplexobj(energies) or not np.all(np.isfinite(mf.mo_coeff)) or np.iscomplexobj(mf.mo_coeff):
            raise RuntimeError("PySCF returned invalid real TDA energies or orbitals.")
        snapshot = {
            "method_es": "tddft", "tda": True, "xc": self.xc,
            "excited_multiplicity": self.excited_multiplicity,
            "excited_state_indices": self.excited_state_indices,
            "grids_level": self.grids_level, "cart": self.cart,
            "grids_options": copy.deepcopy(self.grids_options),
            "mol": mol.dumps(), "atom_positions": positions.copy(),
            "mo_coeff": mf.mo_coeff.copy(), "nocc": nocc,
            "ci": tuple(ci.copy() for ci in coefficients), "energies": energies.copy(),
        }
        if previous is None:
            signs = np.array([1.0] + [1.0 if ci[0].ravel()[np.argmax(abs(ci[0]))] >= 0 else -1.0 for ci in coefficients[1:]])
        else:
            signs = np.where(np.diag(wavefunction_overlaps(previous, snapshot)) < 0, -1.0, 1.0)
        self._apply_phase_signs(td, snapshot, signs)
        return td, snapshot

    def _apply_phase_signs(self, td, snapshot, signs):
        snapshot["ci"] = tuple(ci * sign for ci, sign in zip(snapshot["ci"], signs))
        td.xy = list(td.xy)
        for state, index in enumerate(self.excited_state_indices, 1):
            x, y = td.xy[index]
            td.xy[index] = (np.asarray(x) * signs[state], np.asarray(y) * signs[state])

    def _validate_previous(self, previous):
        if not isinstance(previous, dict) or previous.get("method_es") != "tddft" or not previous.get("tda", False):
            raise ValueError("amplitudes_previous method_es does not match the TDA calculator.")
        for name in ("xc", "excited_multiplicity", "excited_state_indices", "grids_level", "cart", "grids_options"):
            if previous.get(name) != getattr(self, name):
                raise ValueError("amplitudes_previous TDA %s does not match this calculator." % name)
        if "ci" not in previous or len(previous["ci"]) != self.num_quantum_states:
            raise ValueError("amplitudes_previous has a different number of quantum states.")
        try:
            mol = gto.loads(previous["mol"])
            coefficients = np.asarray(previous["ci"])
            orbitals = np.asarray(previous["mo_coeff"])
            nocc = _integer(previous["nocc"], "amplitudes_previous nocc", 1)
            positions = self._positions(previous["atom_positions"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("amplitudes_previous is not a valid TDA wavefunction snapshot.") from error
        if (
            mol.charge != self.charge or mol.spin != 0 or mol.cart != self.cart
            or not np.array_equal(mol.atom_charges(), [gto.charge(name) for name in self.atom_names])
        ):
            raise ValueError("amplitudes_previous atoms, charge, or reference spin does not match this calculator.")
        configured = self._mc.mol if self._mc is not None else gto.M(
            atom=list(zip(self.atom_names, positions)), unit="Bohr", basis=self.basis,
            charge=self.charge, spin=0, cart=self.cart, symmetry=False, verbose=0,
        )
        if not gto.same_basis_set(mol, configured):
            raise ValueError("amplitudes_previous basis does not match this calculator.")
        if (
            orbitals.shape != (mol.nao_nr(), mol.nao_nr())
            or nocc != mol.nelectron // 2
            or coefficients.shape != (self.num_quantum_states, 2, nocc, mol.nao_nr() - nocc)
            or np.iscomplexobj(orbitals) or np.iscomplexobj(coefficients)
            or not np.all(np.isfinite(orbitals)) or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("amplitudes_previous contains invalid real orbitals or TDA amplitudes.")

    def _ensure_solved(self, positions, previous=None):
        super()._ensure_solved(positions, previous)
        self._td = self._mc

    def _native_state(self, index):
        return 0 if index == 0 else self.excited_state_indices[index - 1] + 1

    def _gradient_from_td(self, td, state_ind):
        if state_ind == 0:
            solver = td._scf.nuc_grad_method()
            solver.grid_response = True
            return solver.kernel()

        return gradient(td, self._native_state(state_ind), self.response_options)

    def _gradient(self, state_ind):
        if state_ind not in self._gradients:
            self._gradients[state_ind] = self._response_array(self._gradient_from_td(self._mc, state_ind), "TDA gradient")
        return self._gradients[state_ind].copy()

    def _coupling(self, bra, ket):
        if bra == ket:
            return np.zeros_like(self.atom_positions)
        if (bra, ket) not in self._couplings:
            if self.excited_multiplicity == 3 and (bra == 0 or ket == 0):
                coupling = np.zeros_like(self.atom_positions)
            else:
                if abs(self._snapshot["energies"][bra] - self._snapshot["energies"][ket]) < 1e-12:
                    raise ValueError("Derivative couplings are undefined at an exactly degenerate TDA root pair.")

                native = derivative_coupling(
                    self._mc, self._native_state(bra), self._native_state(ket),
                    use_etfs=self.use_etfs, response_options=self.response_options,
                )
                coupling = self._response_array(native, "TDA derivative coupling") * ANGSTROM_TO_BOHR
            self._couplings[bra, ket] = coupling.copy()
            self._couplings[ket, bra] = -coupling.copy()
        return self._couplings[bra, ket].copy()

    def normal_modes(self, state_ind=0, delta=1e-3):
        """Return mass-weighted modes; use an analytic ground-state Hessian.

        Excited-state Hessians use central differences of analytic gradients.
        Frequencies are atomic-unit angular frequencies. The supplied reference
        geometry is not optimized, and unstable modes remain imaginary.
        """
        state_ind = _integer(state_ind, "state_ind")
        self._state_indices(state_ind)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be a positive displacement in Bohr.")
        positions = self.atom_positions.copy()
        with lib.with_omp_threads(self.num_threads):
            self._ensure_solved(positions)
            if state_ind == 0:

                mf = self._mc._scf
                solver = mf.Hessian()
                solver.max_cycle = self.response_options["max_cycle"]
                solver.level_shift = self.response_options["level_shift"]
                with response_context(self.response_options), lib.temporary_env(
                    mf, conv_tol_cpscf=self.response_options["conv_atol"]
                ):
                    native = np.asarray(solver.kernel())
                hessian = native.transpose(0, 2, 1, 3).reshape(positions.size, positions.size)
            else:
                reference = copy.deepcopy(self._snapshot)
                hessian = np.empty((positions.size, positions.size))
                for coordinate in range(positions.size):
                    gradients = []
                    for direction in (1, -1):
                        displaced = positions.copy()
                        displaced.flat[coordinate] += direction * delta
                        td, _ = self._run(displaced, reference)
                        gradients.append(self._response_array(self._gradient_from_td(td, state_ind), "TDA Hessian gradient").ravel())
                    hessian[:, coordinate] = (gradients[0] - gradients[1]) / (2 * delta)
        if hessian.shape != (positions.size, positions.size) or np.iscomplexobj(hessian) or not np.all(np.isfinite(hessian)):
            raise RuntimeError("PySCF returned an invalid TDA Hessian.")
        return _harmonic_modes((hessian + hessian.T) * 0.5, positions, self.atom_masses)
