"""
This module contains the Q-Chem interface for ab initio calculations.
"""

import copy
import itertools
import os
import re
import subprocess
import logging
import numpy as np
from qclab import numerical_constants

logger = logging.getLogger(__name__)


class QCLabQChemInterface:
    """
    Q-Chem interface for ab initio calculations.
    """

    def __init__(
        self,
        atom_positions,  # In units of Bohr.
        atom_masses,  # In units of electron mass.
        atom_names,
        label="qchem",
        folder_scratch="qclab_job",
        num_threads=1,
        num_procs=1,
        **kwargs,
    ):
        kwargs = {str(k).lower(): v for k, v in kwargs.items()}
        self.method_es = str(kwargs.pop("method_es", "tddft")).lower()
        if self.method_es not in ("cis", "tddft"):
            raise ValueError("Q-Chem method_es must be 'cis' or 'tddft'.")
        explicit_tda = "tda" in kwargs
        legacy_rpa = kwargs.get("rpa", False)
        self.tda = self._logical(
            kwargs.pop("tda", not (legacy_rpa == 2 or self._logical(legacy_rpa, "rpa"))),
            "tda",
        )
        self.use_etfs = self._logical(kwargs.pop("use_etfs", True), "use_etfs")
        aliases = [kwargs.pop(key) for key in ("xc", "functional") if key in kwargs]
        if aliases:
            exchange = kwargs.get("exchange", aliases[0])
            if any(str(value).lower() != str(exchange).lower() for value in aliases):
                raise ValueError("xc, functional, and exchange must specify the same functional.")
            kwargs["exchange"] = exchange
        if kwargs.get("exchange") is None:
            kwargs["exchange"] = "HF" if self.method_es == "cis" else "B3LYP"
        if kwargs.get("basis") is None:
            kwargs["basis"] = "6-31G*"
        explicit_spin = kwargs.pop("excited_multiplicity", None)
        spin_adapted_selection = explicit_spin is not None
        if explicit_spin is None:
            default_triplets = (not self._logical(kwargs["cis_singlets"], "cis_singlets")
                                if "cis_singlets" in kwargs else False)
            triplets = self._logical(kwargs.get("cis_triplets", default_triplets), "cis_triplets")
            singlets = self._logical(kwargs.get("cis_singlets", not triplets), "cis_singlets")
            explicit_spin = (1 if singlets else 3) if singlets != triplets else None
        if spin_adapted_selection and (isinstance(explicit_spin, bool) or explicit_spin not in (1, 3)):
            raise ValueError("excited_multiplicity must be 1 or 3.")
        self.excited_multiplicity = explicit_spin
        for name, expected in (("cis_singlets", explicit_spin == 1),
                               ("cis_triplets", explicit_spin == 3)):
            if spin_adapted_selection and name in kwargs and self._logical(kwargs[name], name) != expected:
                raise ValueError("%s contradicts excited_multiplicity." % name)
            kwargs.setdefault(name, expected)
        unrestricted = self._logical(kwargs.get("unrestricted", False), "unrestricted")
        spin_flip = kwargs.get("spin_flip", False) not in (False, 0, "FALSE", "false")
        if spin_adapted_selection:
            if int(kwargs.get("multiplicity", 1)) != 1:
                raise ValueError("Spin-adapted Q-Chem states require a singlet reference (multiplicity=1).")
            if unrestricted or spin_flip:
                raise ValueError("Spin-adapted Q-Chem states require a restricted reference without spin_flip.")
        elif unrestricted or spin_flip or int(kwargs.get("multiplicity", 1)) != 1:
            # Legacy open-shell roots are not spin-pure; do not impose scalar
            # singlet/triplet zeros on those calculations.
            self.excited_multiplicity = None
        if explicit_tda and "rpa" in kwargs and (legacy_rpa == 2 or self._logical(legacy_rpa, "rpa")) != (not self.tda):
            raise ValueError("rpa contradicts tda; use tda=False for full TDDFT.")
        if self.method_es == "cis" and not self.tda:
            raise ValueError("CIS requires tda=True.")
        kwargs.setdefault("rpa", not self.tda)
        kwargs.setdefault("set_quadratic", False)
        num_states = kwargs.pop("num_quantum_states", None)
        roots = kwargs.get("cis_n_roots")
        if isinstance(roots, str):
            roots = int(roots)
        selected = kwargs.pop("excited_state_indices", None)
        if selected is None:
            count = self._integer(num_states, "num_quantum_states", 1) - 1 if num_states is not None else self._integer(roots or 0, "cis_n_roots", 0)
            selected = tuple(range(count))
        else:
            selected = tuple(self._integer(ind, "excited_state_indices", 0) for ind in selected)
            if len(set(selected)) != len(selected):
                raise ValueError("excited_state_indices must contain unique indices.")
        if num_states is not None and len(selected) != self._integer(num_states, "num_quantum_states", 1) - 1:
            raise ValueError("excited_state_indices must contain num_quantum_states - 1 roots.")
        required_roots = max(selected, default=-1) + 1
        if roots is not None and self._integer(roots, "cis_n_roots", 0) < required_roots:
            raise ValueError("cis_n_roots must include every selected excited_state_indices root.")
        kwargs["cis_n_roots"] = max(required_roots, self._integer(roots or 0, "cis_n_roots", 0))
        self.excited_state_indices = selected
        self.state_indices = (0,) + tuple(ind + 1 for ind in selected)
        self.num_quantum_states = len(self.state_indices)
        self._native_to_state = {native: ind for ind, native in enumerate(self.state_indices)}
        # Create atoms dict.
        atoms = {}
        for i, atom_name in enumerate(atom_names):
            atoms[i] = {
                "name": atom_name,
                "position": atom_positions[i] / numerical_constants.ANGSTROM_TO_BOHR,
                "mass": atom_masses[i] / numerical_constants.AMU_TO_EMASS,
            }
        # Store instance attributes.
        self.atom_names = atom_names
        self.atom_masses = atom_masses
        self.atom_positions = atom_positions
        self.atoms = atoms
        self.folder_scratch = folder_scratch
        self.label = label
        self.kwargs = kwargs
        self.num_threads = num_threads
        self.num_procs = num_procs
        self.results = {}
        # Properties implemented.
        self.implemented_properties = [
            "energy",
            "gradient",
            "derivative_coupling",
            "frequency",
            "wf_overlaps",
        ]
        # Keep the same electronic model in every property and overlap job.
        shared = {key: copy.deepcopy(value) for key, value in kwargs.items()
                  if key not in ("charge", "multiplicity")}
        shared.setdefault("sym_ignore", True)
        self._shared_parameters = shared
        self.job_templates = {}
        for name, jobtype in (("energy", "SP"), ("gradient", "FORCE"),
                              ("frequency", "FREQ"), ("derivative_coupling", "SP"),
                              ("wf_overlaps", "SP")):
            params = dict(shared, jobtype=jobtype)
            if name == "derivative_coupling":
                params.update(calc_nac=True, cis_der_numstate=self.num_quantum_states)
            self.job_templates[name] = {
                "name": name,
                "write_derivative_coupling": name == "derivative_coupling",
                "qchem_parameters": {self.method_es: params},
            }

    @staticmethod
    def _logical(value, name):
        """Interpret Q-Chem logical values without treating 'FALSE' as true."""
        if value in (True, 1) or str(value).lower() == "true":
            return True
        if value in (False, 0) or str(value).lower() == "false":
            return False
        raise ValueError("%s must be a boolean." % name)

    @staticmethod
    def _integer(value, name, minimum):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("%s must be an integer >= %d." % (name, minimum))
        return int(value)

    def _state_list(self, indices):
        if indices is None:
            return list(range(self.num_quantum_states))
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        states = [self._integer(ind, "state index", 0) for ind in indices]
        if any(ind >= self.num_quantum_states for ind in states):
            raise ValueError("Requested state index is outside the selected state space.")
        return states

    def execute(self):
        """
        Create and execute the command for the Q-Chem calculation.
        """
        # Build command.
        command = "qchem"
        if self.num_procs != 1:
            command += f" -np {self.num_procs}"
        if self.num_threads != 1:
            command += f" -nt {self.num_threads}"
        command += f" {self.label}.inp {self.label}.out"
        command += " " + self.folder_scratch
        command += " > /dev/null 2>&1"
        subprocess.run(command, shell=True, cwd=os.getcwd())

    def _build_job_specs(self, properties):
        """
        Build the job specifications for the requested properties.

        Each job spec is a dict with:
        name : label (e.g. 'gradient', 'derivative_coupling', 'frequency', 'energy')
        write_derivative_coupling : bool, whether to emit $derivative_coupling
        qchem_parameters : dict of dicts, keyed by method_es, of Q-Chem parameters to be written in $rem section.
        """

        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        # Check if the asked properties are implemented.
        self._check_requested_properties(properties)
        properties = list(dict.fromkeys(properties))
        return [copy.deepcopy(self.job_templates[p]) for p in properties]

    def _check_requested_properties(self, job_requested):
        """
        Checks if the requested properties are implemented.
        """
        for job in job_requested:
            if job not in self.implemented_properties:
                raise ValueError(f"Requested property '{job}' is not implemented.")

    def _compute_job_read_flag(self, num_jobs, indx):
        """
        Compute the flag for the $molecule section of each job in the input file.
        gradient/wf_overlaps: flag=0 for single-job input or first job,
        otherwise offset by +1
        """
        return 0 if (num_jobs == 1 or indx == 0) else (indx + 1)

    def _add_parameters_to_job_spec(self, job_spec, **kwargs):
        """
        Add parameters from kwargs to job_spec.
        """
        overrides = kwargs.get(job_spec["name"], {}).get("qchem_args") or {}
        protected = {"exchange", "correlation", "method", "basis", "purecart",
                     "cis_n_roots", "cis_singlets", "cis_triplets", "rpa",
                     "unrestricted", "spin_flip", "set_quadratic", "xc_grid",
                     "scf_convergence", "cis_convergence", "thresh"}
        structural = {"exchange", "correlation", "method", "basis", "purecart",
                      "cis_n_roots", "cis_singlets", "cis_triplets", "rpa",
                      "unrestricted", "spin_flip", "set_quadratic"}
        for key, value in overrides.items():
            key = str(key).lower()
            expected = self._shared_parameters.get(key)
            if key == "method":
                expected = self._shared_parameters["exchange"]
            # An absent structural setting still has a program default. Adding
            # it to one job can change its spin or AO basis relative to the
            # other properties and invalidate the overlap representation.
            if (key in structural and expected is None) or (
                key in protected and expected is not None
                and str(value).lower() != str(expected).lower()
            ):
                raise ValueError("Set shared Q-Chem parameter %s in calculator_args, not per property." % key)
            job_spec["qchem_parameters"][self.method_es][key] = value
        return job_spec

    def _write_job(self, file_obj, job_spec, **kwargs):
        """
        Write  Q-Chem jobs (single job or multiple jobs that share same geometry).

        The number of jobs is determined by the number of properties requested and
        the writing is handled by the corresponding method for each property.
        """
        num_jobs = len(job_spec)
        for i, job in enumerate(job_spec):
            flag = self._compute_job_read_flag(num_jobs, i)
            job = self._add_parameters_to_job_spec(job, **kwargs)
            if "gradient" in job["name"]:
                self._write_gradient_jobs(
                    file_obj,
                    job,
                    flag,
                    kwargs["gradient"].get("state_inds_gradient", None),
                )
                continue
            elif "wf_overlaps" in job["name"]:
                if kwargs["wf_overlaps"]["atom_positions_previous"] is None:
                    raise ValueError(
                        "Previous geometry must be provided"
                        "when requesting wf_overlaps."
                    )
                atoms_previous = {}
                for i, atom_name in enumerate(self.atom_names):
                    atoms_previous[i] = {
                        "name": atom_name,
                        "position": kwargs["wf_overlaps"]["atom_positions_previous"][i]
                        / numerical_constants.ANGSTROM_TO_BOHR,
                        "mass": self.atom_masses[i] / numerical_constants.AMU_TO_EMASS,
                    }
                self._write_wf_overlaps_jobs(file_obj, job, atoms_previous, flag)
                continue
            else:
                self._write_comments(file_obj, job["name"], ind=i)
                self._write_molecule_section(file_obj, ind=i)
                if job["name"] == "energy" and kwargs["energy"].get("excited_amplitudes", False):
                    job["qchem_parameters"][self.method_es]["GUI"] = "2"
                    job["qchem_parameters"][self.method_es]["IQMOL_FCHK"] = "TRUE"
                if job["name"] == "derivative_coupling":
                    self._get_state_inds_derivative_coupling(
                        job,
                        kwargs["derivative_coupling"].get(
                            "state_inds_derivative_coupling", None
                        ),
                    )
                    job["qchem_parameters"][self.method_es]["cis_der_numstate"] = (
                        len(self.state_inds_derivative_coupling)
                        if job["write_derivative_coupling"] else 0
                    )
                self._write_rem_section(
                    file_obj,
                    job,
                )
                if job["write_derivative_coupling"]:
                    self._write_derivative_coupling(
                        file_obj,
                        job,
                    )

    def _get_state_inds_gradient(self, state_inds_gradient):
        """
        Return a list of state indices for gradient calculations.

        If no indices are provided, defaults to all states (ground + excited).
        """
        return self._state_list(state_inds_gradient)

    def _write_gradient_jobs(
        self, file_obj, job_spec, flag=0, state_inds_gradient=None
    ):
        """
        Handles the writing for gradient jobs requested.
        """
        self.state_inds_gradient = self._get_state_inds_gradient(state_inds_gradient)
        for j, state_ind in enumerate(self.state_inds_gradient):
            job_spec = copy.deepcopy(job_spec)
            i = self.state_indices[state_ind]
            job_spec["qchem_parameters"][self.method_es].pop("cis_state_deriv", None)
            if i > 0:
                job_spec["qchem_parameters"][self.method_es]["cis_state_deriv"] = str(i)
                job_spec["qchem_parameters"][self.method_es]["cis_n_roots"] = (
                    self.kwargs.get("cis_n_roots")
                )
                job_spec["qchem_parameters"][self.method_es]["cis_singlets"] = (
                    self.kwargs.get("cis_singlets")
                )
                job_spec["qchem_parameters"][self.method_es]["cis_triplets"] = (
                    self.kwargs.get("cis_triplets")
                )
            if flag == 0:
                ind = j
            else:
                ind = flag
            self._write_comments(file_obj, f"gradient state {i}", ind)
            self._write_molecule_section(file_obj, ind)
            self._write_rem_section(file_obj, job_spec)

    def _write_wf_overlaps_jobs(self, file_obj, job_spec, atoms_previous, flag=0):
        """
        Handles the writing for wavefunction overlaps jobs requested.
        """
        ind = flag
        # Create the input for the previous geometry.
        self._write_comments(file_obj, job_spec["name"] + " previous step", ind)
        # Write geometry.
        file_obj.write("$molecule\n")
        charge, mult = self._get_charge_and_multiplicity()
        file_obj.write("   %d %d\n" % (charge, mult))
        self._write_geometry(file_obj, atoms_previous)
        file_obj.write("$end\n\n")
        # Job definition.
        keys_for_wf_overlaps = ["mo_overlaps_two_geoms"]
        if all(
            key in job_spec["qchem_parameters"][self.method_es]
            for key in keys_for_wf_overlaps
        ):
            pass
        else:
            job_spec["qchem_parameters"][self.method_es]["mo_overlaps_two_geoms"] = 1
        if (
            job_spec["qchem_parameters"][self.method_es].get("mo_overlaps_two_geoms")
            != 1
        ):
            job_spec["qchem_parameters"][self.method_es]["mo_overlaps_two_geoms"] = 1
        self._write_rem_section(file_obj, job_spec)

        # Creating input for the current geometry.
        self._write_comments(file_obj, job_spec["name"] + " current step", ind + 1)
        self._write_molecule_section(file_obj, ind=0)
        job_spec["qchem_parameters"][self.method_es]["mo_overlaps_two_geoms"] = 2
        self._write_rem_section(file_obj, job_spec)

    def _write_comments(self, file_obj, job, ind=0):
        """
        Write comments and job separator for Q-Chem input file.
        """
        if ind != 0:
            file_obj.write("@@@\n\n")
        file_obj.write("$comment\n")
        file_obj.write(f"QC Lab generated input file ({job} job)\n")
        file_obj.write("$end\n\n")

    def _get_charge_and_multiplicity(self):
        """
        Return the molecular charge and spin multiplicity from kwargs
        if provided, otherwise raise an error.
        """
        charge = self.kwargs.get("charge", None)
        multiplicity = self.kwargs.get("multiplicity", None)
        if multiplicity is None or charge is None:
            raise ValueError("Charge and multiplicity must be provided.")
        else:
            multiplicity = int(multiplicity)
            charge = int(charge)
        return charge, multiplicity

    def _write_geometry(self, file_obj, geometry):
        """
        Write atomic coordinates to the Q-Chem input file.
        """
        for i in range(len(self.atom_names)):
            file_obj.write(
                "   {}  {:f}  {:f}  {:f}\n".format(
                    geometry[i]["name"],
                    geometry[i]["position"][0],
                    geometry[i]["position"][1],
                    geometry[i]["position"][2],
                )
            )

    def _write_molecule_section(self, file_obj, ind=0):
        """
        Write the $molecule section to the Q-Chem input file.

        For the first job (ind=0), writes charge, multiplicity, and geometry.
        For subsequent jobs, writes 'read' to reuse the previous geometry.
        """
        file_obj.write("$molecule\n")
        if ind == 0:
            charge, multiplicity = self._get_charge_and_multiplicity()
            file_obj.write("   %d %d\n" % (charge, multiplicity))
            self._write_geometry(file_obj, self.atoms)
        else:
            file_obj.write("   read\n")
        file_obj.write("$end\n\n")

    def _write_rem_section(self, file_obj, job_spec):
        """
        Write the $rem section to the Q-Chem input file from the job spec parameters.
        """
        file_obj.write("$rem\n")
        for parameter, v_str in job_spec["qchem_parameters"][self.method_es].items():
            if isinstance(v_str, str):
                v_str = v_str.upper()
            if v_str == "NONE" or v_str is None:
                continue
            file_obj.write("   %-25s   %s\n" % (parameter.upper(), v_str))
        file_obj.write("$end\n\n")

    def _get_state_inds_derivative_coupling(
        self, job, state_inds_derivative_coupling=None
    ):
        """
        Return a list/tuple of state indices for derivative coupling calculations.

        If no indices are provided, defaults to all states (ground + excited) as specified
        by cis_der_numstate (cis_n_roots+1) in the input file.
        """
        indices = state_inds_derivative_coupling
        if isinstance(indices, (int, np.integer)):
            indices = list(range(indices + 1))
        if indices is not None and np.asarray(indices).ndim == 2:
            pairs = []
            for pair in indices:
                if len(pair) != 2:
                    raise ValueError("Derivative-coupling pairs must contain two states.")
                pair = tuple(self._state_list(pair))
                if pair[0] == pair[1]:
                    raise ValueError("Derivative-coupling pairs must contain distinct states.")
                pairs.append(pair)
            states = sorted({ind for pair in pairs for ind in pair})
        else:
            states = self._state_list(indices)
            pairs = list(itertools.combinations(states, 2))
        self._requested_coupling_pairs = pairs
        # The reference has a different spin from every triplet target. Its
        # scalar derivative couplings vanish and need not be sent to Q-Chem.
        self.state_inds_derivative_coupling = [self.state_indices[ind] for ind in states
                                              if self.excited_multiplicity != 3 or ind != 0]
        if len(self.state_inds_derivative_coupling) < 2:
            job["write_derivative_coupling"] = False
            job["qchem_parameters"][self.method_es]["calc_nac"] = False

    def _write_derivative_coupling(self, file_obj, job):
        """
        Write the $derivative_coupling section to the Q-Chem input file.
        """
        file_obj.write("$derivative_coupling\n")
        file_obj.write("Native Q-Chem state indices (0 is the SCF reference)\n")
        file_obj.write(
            " ".join(str(s) for s in self.state_inds_derivative_coupling) + "\n"
        )
        file_obj.write("$end\n\n")

    def read_results(self, **kwargs):
        """
        Read and parse the Q-Chem output file for the requested properties.

        Results are stored in ``self.results``.

        .. rubric:: Args

        **kwargs:
            Keyword arguments keyed by property name (e.g. 'energy', 'gradient').
            Each value is a dict of options specific to that property.
        """
        kwargs = {key: value for key, value in kwargs.items()
                  if value.get("calc_property", True)}
        properties = kwargs.keys()
        if not properties:
            self.results = {}
            return
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        num_atoms = len(self.atoms)
        filename = self.label + ".out"
        with open(filename, "r") as file_obj:
            file_content = file_obj.read()

        self._check_normal_termination(file_content, properties)
        self._organize_data_file(file_content, properties, num_atoms, **kwargs)

    def _get_number_of_jobs(self, properties):
        """
        Return the total number of Q-Chem jobs for the requested properties.

        Gradient and wavefunction overlap calculations may require
        multiple jobs beyond one per property.
        """
        num_jobs = len(properties)
        if "gradient" in properties:
            num_jobs += len(self.state_inds_gradient) - 1
        if "wf_overlaps" in properties:
            num_jobs += 1
        return num_jobs

    def _check_normal_termination(self, file_content, properties):
        """
        Check Q-Chem normal termination for the number of jobs
        corresponding to the requested properties. If the number of jobs
        is not equal to the  number of normal termination flags in
        the output file, raise an error.
        """
        normal_termination = "for using Q-Chem"
        num_jobs = self._get_number_of_jobs(properties)
        if file_content.count(normal_termination) != num_jobs:
            raise ValueError("Q-Chem did not terminate normally.")

    def _split_file_by_job(self, file_content, properties, **kwargs):
        """
        Split the Q-Chem output file content into sections for each job.
        """
        num_jobs = self._get_number_of_jobs(properties)
        splitting_flag = "Running Job"
        if num_jobs == 1:
            return [file_content]
        else:
            return file_content.split(splitting_flag)[1 : num_jobs + 1]

    def _organize_data_file(self, file_content, properties, num_atoms, **kwargs):
        """
        Organize Q-Chem data according to the requested properties.
        """
        split_files = self._split_file_by_job(file_content, properties, **kwargs)
        i = 0
        for job in properties:
            if "gradient" in job:
                gradient_files = split_files[i : i + len(self.state_inds_gradient)]
                i += len(self.state_inds_gradient)
                self._pull_property(
                    job,
                    gradient_files,
                    num_atoms,
                )
            elif "wf_overlaps" in job:
                overlaps_files = split_files[i : i + 1]
                i += 2
                self._pull_property(
                    job,
                    overlaps_files,
                    num_atoms,
                    amplitudes_previous=kwargs["wf_overlaps"].get(
                        "amplitudes_previous", None
                    ),
                    amplitudes_current=(
                        self.results["excited_amplitudes"]
                        if (
                            "energy" in properties
                            and kwargs["energy"].get("excited_amplitudes", False)
                        )
                        else kwargs["wf_overlaps"].get("amplitudes_current", None)
                    ),
                )
            else:
                job_file = split_files[i]
                i += 1
                excited_amplitudes = (
                    kwargs["energy"].get("excited_amplitudes", False)
                    if "energy" in job
                    else False
                )
                self._pull_property(
                    job,
                    job_file.splitlines(),
                    num_atoms,
                    excited_amplitudes=excited_amplitudes,
                )

    def _pull_property(self, property, file_obj, num_atoms, **kwargs):
        """
        Pull the requested property from the Q-Chem output file section.

        Determines which parsing method to call based on the property type and
        stores the results in ``self.results``.
        """
        if "energy" in property:
            self._pull_energy(
                file_obj, excited_amplitudes=kwargs.get("excited_amplitudes", False)
            )
        elif "gradient" in property:
            self._pull_gradient(file_obj, num_atoms)
        elif "frequency" in property:
            self._pull_vibration(file_obj, num_atoms)
        elif "derivative_coupling" in property:
            self._pull_derivative_coupling(file_obj, num_atoms)
        elif "wf_overlaps" in property:
            self._pull_overlaps(
                amplitudes_previous=kwargs.get("amplitudes_previous", None),
                amplitudes_current=kwargs.get("amplitudes_current", None),
            )
        else:
            raise ValueError("This type of calculation has not been implemented yet.")

    def _pull_vibration(self, file_obj, num_atoms):
        """
        Parse vibrational frequencies and normal modes from Q-Chem output.

        Stores 'frequency' and 'normal_mode' in ``self.results``.
        """
        ind_modes = []
        for i, line in enumerate(file_obj):
            if "Mode:" in line:
                ind_modes.append(i)
        freqs = []
        for i in ind_modes:
            freqs.append(file_obj[i + 1].split()[1:])
        freqs = np.array(freqs, dtype=float).flatten()
        modes = np.zeros((len(freqs), num_atoms, 3))
        i_count = 0
        # Number of lines between the mode: line and the displacement-matrix line in a Q-Chem output file.
        num_lines = 8
        for i in ind_modes:
            for j in range(num_atoms):
                temporal_data = file_obj[i + num_lines + j].split()[1:]
                num_modes = int(len(temporal_data) / 3)
                for k in range(num_modes):
                    modes[i_count + k, j, :] = temporal_data[k * 3 : k * 3 + 3]
            i_count += num_modes

        self.results["frequency"] = freqs
        self.results["normal_mode"] = modes

    def _pull_energy(self, file_obj, excited_amplitudes=False):
        """
        Parse ground and excited state energies from Q-Chem output.

        Optionally extracts excited state amplitudes from the FCHK file.
        Stores 'energy' (and 'excited_amplitudes' if requested) in ``self.results``.
        """
        energies = {}
        for j, line in enumerate(file_obj):
            excited = re.search(r"Total energy for state\s+(\d+)\s*[:=]?\s+([-+\d.EeDd]+)", line)
            if excited:
                energies[int(excited.group(1))] = float(excited.group(2).replace("D", "E").replace("d", "e"))
            elif "energy in the final basis set" in line.lower():
                energies[0] = float(line.split()[-1])
            elif 0 not in energies and ("SCF time:" in line or "Timing for Total SCF" in line):
                # Keep compatibility with output variants whose total energy is
                # printed two lines after the SCF timing header.
                try:
                    energies[0] = float(file_obj[j + 2].split()[-1])
                except (ValueError, IndexError):
                    pass
        missing = set(self.state_indices) - set(energies)
        if missing:
            raise ValueError("Q-Chem output is missing total energies for native states %s." % sorted(missing))
        self.results["energy"] = np.array([energies[ind] for ind in self.state_indices])
        # Extract excited state amplitudes if requested.
        if excited_amplitudes:
            self.results["excited_amplitudes"] = {}
            (
                x_alpha,
                y_alpha,
                x_beta,
                y_beta,
                num_basis_functions,
                num_alpha_electrons,
                num_beta_electrons,
                num_excited_states,
                mo_alpha,
                mo_beta,
            ) = self._pull_excited_amplitudes()
            self.results["excited_amplitudes"]["x_alpha"] = x_alpha
            self.results["excited_amplitudes"]["y_alpha"] = y_alpha
            self.results["excited_amplitudes"]["x_beta"] = x_beta
            self.results["excited_amplitudes"]["y_beta"] = y_beta
            self.results["excited_amplitudes"][
                "num_basis_functions"
            ] = num_basis_functions
            self.results["excited_amplitudes"][
                "num_alpha_electrons"
            ] = num_alpha_electrons
            self.results["excited_amplitudes"][
                "num_beta_electrons"
            ] = num_beta_electrons
            self.results["excited_amplitudes"][
                "num_excited_states"
            ] = num_excited_states
            self.results["excited_amplitudes"]["mo_alpha"] = mo_alpha
            self.results["excited_amplitudes"]["mo_beta"] = mo_beta
            self.results["excited_amplitudes"] = self._project_amplitudes(self.results["excited_amplitudes"])

    def _project_amplitudes(self, amplitudes):
        """Copy a snapshot into the selected response-root order exactly once."""
        amplitudes = copy.deepcopy(amplitudes)
        if "state_indices" in amplitudes:
            if tuple(amplitudes["state_indices"]) != self.state_indices:
                raise ValueError("Q-Chem amplitude snapshots use different selected states.")
            if amplitudes.get("excited_multiplicity", self.excited_multiplicity) != self.excited_multiplicity:
                raise ValueError("Q-Chem amplitude snapshots use different spin manifolds.")
            return amplitudes
        for key in ("x_alpha", "y_alpha", "x_beta", "y_beta"):
            values = np.asarray(amplitudes[key])
            if max(self.excited_state_indices, default=-1) >= len(values):
                raise ValueError("Q-Chem amplitude snapshot does not contain every selected response root.")
            amplitudes[key] = values[list(self.excited_state_indices)].copy()
        amplitudes["num_excited_states"] = len(self.excited_state_indices)
        amplitudes["state_indices"] = self.state_indices
        amplitudes["excited_multiplicity"] = self.excited_multiplicity
        return amplitudes

    def _pull_gradient(self, file_obj, num_atoms):
        """
        Parse gradients for each state from Q-Chem output.

        Stores 'gradient' in ``self.results``.
        """
        gradient = np.zeros((num_atoms, 3, len(self.state_inds_gradient)))
        flag_words_grad = [
            "Gradient of the state energy (including CIS Excitation Energy)",
            "Gradient of SCF Energy",
        ]
        flag_words_out_grad = ["Gradient time", "Max gradient component"]
        for state_ind in range(len(self.state_inds_gradient)):
            flag_in = next(
                flag for flag in flag_words_grad if flag in file_obj[state_ind]
            )
            flag_out = next(
                flag for flag in flag_words_out_grad if flag in file_obj[state_ind]
            )
            gradient_matrix = file_obj[state_ind].split(flag_in)[1]
            gradient_matrix = gradient_matrix.split(flag_out)[0]
            # Ensure that the gradient matrix is clean.
            for f in flag_words_out_grad:
                if gradient_matrix.count(f) != 0:
                    gradient_matrix = gradient_matrix.split(f)[0]
            stop_ind = 0
            i_count = 1
            gradient_matrix = gradient_matrix.splitlines()
            while stop_ind < num_atoms:
                jndx = np.array(gradient_matrix[i_count].split(), dtype=int) - 1
                gradient[jndx, 0, state_ind] = np.array(
                    gradient_matrix[i_count + 1].split()[1:], dtype=float
                )
                gradient[jndx, 1, state_ind] = np.array(
                    gradient_matrix[i_count + 2].split()[1:], dtype=float
                )
                gradient[jndx, 2, state_ind] = np.array(
                    gradient_matrix[i_count + 3].split()[1:], dtype=float
                )
                i_count += 4
                stop_ind = jndx.max() + 1
        self.results["gradient"] = gradient

    def _pull_derivative_coupling(self, file_obj, num_atoms, etf=None):
        """
        Parse derivative couplings between electronic states from Q-Chem output.

        Stores 'derivative_coupling' in ``self.results`` as a dictionary
        keyed by state-pair tuples.
        """
        etf = self.use_etfs if etf is None else etf
        l_states = []  # Line that identifies the states involved in derivative coupling
        l_etf = (
            []
        )  # Line that identifies the location of derivative coupling's values with ETF corrections.
        l_no_etf = (
            []
        )  # Line that identifies the location of derivative coupling's values without ETF corrections.
        for i, line in enumerate(file_obj):
            if "between states" in line:
                l_states.append(i)
            elif "with ETF" in line:
                l_etf.append(i)
            elif "without ETF" in line:
                l_no_etf.append(i)

        # Gather derivative coupling values.
        derivative_coupling_matrix = np.zeros((num_atoms, 3))
        derivative_coupling_dictionary = {}
        for i, states in enumerate(l_states):
            temp = file_obj[states].split()
            # States involved in derivative coupling.
            i_st = int(temp[-3])
            j_st = int(temp[-1])
            # Base line for derivative coupling values.
            locations = l_etf if etf else l_no_etf
            if i >= len(locations):
                raise ValueError("Q-Chem output is missing requested %s ETF derivative couplings." % ("with" if etf else "without"))
            base = locations[i]
            for j in range(num_atoms):
                derivative_coupling_matrix[j, :] = np.array(
                    file_obj[base + 3 + j].split()[1:], dtype=float
                )
            if i_st in self._native_to_state and j_st in self._native_to_state:
                pair = (self._native_to_state[i_st], self._native_to_state[j_st])
                # Native Q-Chem output is 1/Bohr; the calculator contract is 1/Angstrom.
                derivative_coupling_dictionary[pair] = derivative_coupling_matrix.copy() * numerical_constants.ANGSTROM_TO_BOHR
        requested = getattr(self, "_requested_coupling_pairs", list(itertools.combinations(range(self.num_quantum_states), 2)))
        selected = {}
        for pair in requested:
            if self.excited_multiplicity == 3 and 0 in pair:
                selected[pair] = np.zeros((num_atoms, 3))
            elif pair in derivative_coupling_dictionary:
                selected[pair] = derivative_coupling_dictionary[pair]
            elif pair[::-1] in derivative_coupling_dictionary:
                selected[pair] = -derivative_coupling_dictionary[pair[::-1]]
            else:
                raise ValueError("Q-Chem output is missing derivative coupling for selected states %s." % (pair,))
        self.results["derivative_coupling"] = selected

    def _pull_overlaps(self, amplitudes_previous=None, amplitudes_current=None):
        """
        Parse wavefunction overlaps between two geometries from Q-Chem output.

        Determine which method to use for computing overlaps based on the method
        used for excited state calculations (TDDFT or CIS) and call the corresponding method.
        The overlaps are stored in ``self.results["wf_overlaps"]`` as a matrix.
        """
        if amplitudes_previous is None:
            raise ValueError(
                "Previous excited state amplitudes must be provided \n"
                "to compute wavefunction overlaps."
            )
        if amplitudes_current is None:
            raise ValueError(
                "Current excited state amplitudes must be provided \n"
                "to compute wavefunction overlaps."
            )
        amplitudes_previous = self._project_amplitudes(amplitudes_previous)
        amplitudes_current = self._project_amplitudes(amplitudes_current)
        # Create instance attributes needed for computing overlaps.
        self.num_basis_functions = amplitudes_current["num_basis_functions"]
        self.num_alpha_electrons = amplitudes_current["num_alpha_electrons"]
        self.num_beta_electrons = amplitudes_current["num_beta_electrons"]
        self.num_excited_states = amplitudes_current["num_excited_states"]
        self.mo_alpha_curr = np.reshape(
            amplitudes_current["mo_alpha"],
            (self.num_basis_functions, self.num_basis_functions),
            order="F",
        )
        self.mo_beta_curr = np.reshape(
            amplitudes_current["mo_beta"],
            (self.num_basis_functions, self.num_basis_functions),
            order="F",
        )
        self.mo_alpha_prev = np.reshape(
            amplitudes_previous["mo_alpha"],
            (self.num_basis_functions, self.num_basis_functions),
            order="F",
        )
        self.mo_beta_prev = np.reshape(
            amplitudes_previous["mo_beta"],
            (self.num_basis_functions, self.num_basis_functions),
            order="F",
        )
        # Compute MO overlaps between two geometries from AO overlaps.
        mo_overlaps_alpha, mo_overlaps_beta = self._compute_mo_overlaps()

        # Compute overlaps matrices.
        if self.method_es.lower() == "tddft" and not self.tda:
            self._get_overlaps_tddft(
                amplitudes_previous["x_alpha"],
                amplitudes_previous["y_alpha"],
                amplitudes_previous["x_beta"],
                amplitudes_previous["y_beta"],
                amplitudes_current["x_alpha"],
                amplitudes_current["y_alpha"],
                amplitudes_current["x_beta"],
                amplitudes_current["y_beta"],
                mo_overlaps_alpha,
                mo_overlaps_beta,
            )
        elif self.method_es.lower() == "cis" or self.tda:
            self._get_overlaps_cis(
                amplitudes_previous["x_alpha"],
                amplitudes_current["x_alpha"],
                amplitudes_previous["x_beta"],
                amplitudes_current["x_beta"],
                mo_overlaps_alpha,
                mo_overlaps_beta,
            )
        else:
            raise ValueError(
                f"Method {self.method_es} not recognized. Supported methods are 'tddft' and 'cis'."
            )
        if self.excited_multiplicity == 3:
            self.results["wf_overlaps"][0, 1:] = 0.0
            self.results["wf_overlaps"][1:, 0] = 0.0

    def _get_overlaps_cis(
        self,
        alpha_prev,
        alpha_curr,
        beta_prev,
        beta_curr,
        mo_overlaps_alpha,
        mo_overlaps_beta,
    ):
        """
        Compute wavefunction overlaps using the CIS formalism.

        The results are stored in ``self.results["wf_overlaps"]`` as a matrix.
        """
        s_oo_alpha, s_ov_alpha, s_vo_alpha, s_vv_alpha = self._get_matrix_overlaps_s(
            self.num_alpha_electrons,
            mo_overlaps_alpha,
        )
        s_oo_beta, s_ov_beta, s_vo_beta, s_vv_beta = self._get_matrix_overlaps_s(
            self.num_beta_electrons,
            mo_overlaps_beta,
        )

        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo_alpha)
        gs_overlap_alpha = sing * np.exp(log_determinant)
        sing, log_determinant = np.linalg.slogdet(s_oo_beta)
        gs_overlap_beta = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_es_alpha = self._compute_overlaps_gs_es(
            alpha_curr, s_oo_alpha, s_ov_alpha
        )
        overlaps_gs_es_beta = self._compute_overlaps_gs_es(
            beta_curr, s_oo_beta, s_ov_beta
        )
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_es_gs_alpha = self._compute_overlaps_es_gs(
            alpha_prev, s_oo_alpha, s_vo_alpha
        )
        overlaps_es_gs_beta = self._compute_overlaps_es_gs(
            beta_prev, s_oo_beta, s_vo_beta
        )
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_es_es_alpha = self._compute_overlaps_es_es(
            alpha_prev,
            alpha_curr,
            s_oo_alpha,
            s_ov_alpha,
            s_vo_alpha,
            s_vv_alpha,
        )
        overlaps_es_es_beta = self._compute_overlaps_es_es(
            beta_prev,
            beta_curr,
            s_oo_beta,
            s_ov_beta,
            s_vo_beta,
            s_vv_beta,
        )
        # Opposite-spin excitations factor into two single-spin determinants.
        overlaps_es_es_alpha_beta = np.outer(overlaps_es_gs_alpha, overlaps_gs_es_beta)
        overlaps_es_es_beta_alpha = np.outer(overlaps_es_gs_beta, overlaps_gs_es_alpha)

        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1),
            dtype=np.result_type(overlaps_es_es_alpha, overlaps_es_es_beta),
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap_alpha * gs_overlap_beta
        self.results["wf_overlaps"][0, 1:] = (
            overlaps_gs_es_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_gs_es_beta
        )
        self.results["wf_overlaps"][1:, 0] = (
            overlaps_es_gs_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_es_gs_beta
        )
        self.results["wf_overlaps"][1:, 1:] = (
            overlaps_es_es_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_es_es_beta
            + overlaps_es_es_alpha_beta
            + overlaps_es_es_beta_alpha
        )

    def _get_overlaps_tddft(
        self,
        x_alpha_prev,
        y_alpha_prev,
        x_beta_prev,
        y_beta_prev,
        x_alpha_curr,
        y_alpha_curr,
        x_beta_curr,
        y_beta_curr,
        mo_overlaps_alpha,
        mo_overlaps_beta,
    ):
        """
        Compute wavefunction overlaps using the TDDFT formalism.

        The results are stored in ``self.results["wf_overlaps"]`` as a matrix.
        """
        s_oo_alpha, s_ov_alpha, s_vo_alpha, s_vv_alpha = self._get_matrix_overlaps_s(
            self.num_alpha_electrons,
            mo_overlaps_alpha,
        )
        s_oo_beta, s_ov_beta, s_vo_beta, s_vv_beta = self._get_matrix_overlaps_s(
            self.num_beta_electrons,
            mo_overlaps_beta,
        )
        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo_alpha)
        gs_overlap_alpha = sing * np.exp(log_determinant)
        sing, log_determinant = np.linalg.slogdet(s_oo_beta)
        gs_overlap_beta = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_es_x_alpha = self._compute_overlaps_gs_es(
            x_alpha_curr, s_oo_alpha, s_ov_alpha
        )
        overlaps_gs_es_y_alpha = self._compute_overlaps_gs_es(
            y_alpha_curr, s_oo_alpha, s_ov_alpha
        )
        overlaps_gs_es_alpha = overlaps_gs_es_x_alpha - overlaps_gs_es_y_alpha
        overlaps_gs_es_x_beta = self._compute_overlaps_gs_es(
            x_beta_curr, s_oo_beta, s_ov_beta
        )
        overlaps_gs_es_y_beta = self._compute_overlaps_gs_es(
            y_beta_curr, s_oo_beta, s_ov_beta
        )
        overlaps_gs_es_beta = overlaps_gs_es_x_beta - overlaps_gs_es_y_beta
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_es_gs_x_alpha = self._compute_overlaps_es_gs(
            x_alpha_prev, s_oo_alpha, s_vo_alpha
        )
        overlaps_es_gs_y_alpha = self._compute_overlaps_es_gs(
            y_alpha_prev, s_oo_alpha, s_vo_alpha
        )
        overlaps_es_gs_alpha = overlaps_es_gs_x_alpha - overlaps_es_gs_y_alpha
        overlaps_es_gs_x_beta = self._compute_overlaps_es_gs(
            x_beta_prev, s_oo_beta, s_vo_beta
        )
        overlaps_es_gs_y_beta = self._compute_overlaps_es_gs(
            y_beta_prev, s_oo_beta, s_vo_beta
        )
        overlaps_es_gs_beta = overlaps_es_gs_x_beta - overlaps_es_gs_y_beta
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_es_es_x_x_alpha = self._compute_overlaps_es_es(
            x_alpha_prev,
            x_alpha_curr,
            s_oo_alpha,
            s_ov_alpha,
            s_vo_alpha,
            s_vv_alpha,
        )
        overlaps_es_es_x_y_alpha = self._compute_overlaps_es_es(
            x_alpha_prev,
            y_alpha_curr,
            s_oo_alpha,
            s_ov_alpha,
            s_vo_alpha,
            s_vv_alpha,
        )
        overlaps_es_es_y_x_alpha = self._compute_overlaps_es_es(
            y_alpha_prev,
            x_alpha_curr,
            s_oo_alpha,
            s_ov_alpha,
            s_vo_alpha,
            s_vv_alpha,
        )
        overlaps_es_es_y_y_alpha = self._compute_overlaps_es_es(
            y_alpha_prev,
            y_alpha_curr,
            s_oo_alpha,
            s_ov_alpha,
            s_vo_alpha,
            s_vv_alpha,
        )
        overlaps_es_es_x_x_beta = self._compute_overlaps_es_es(
            x_beta_prev,
            x_beta_curr,
            s_oo_beta,
            s_ov_beta,
            s_vo_beta,
            s_vv_beta,
        )
        overlaps_es_es_x_y_beta = self._compute_overlaps_es_es(
            x_beta_prev,
            y_beta_curr,
            s_oo_beta,
            s_ov_beta,
            s_vo_beta,
            s_vv_beta,
        )
        overlaps_es_es_y_x_beta = self._compute_overlaps_es_es(
            y_beta_prev,
            x_beta_curr,
            s_oo_beta,
            s_ov_beta,
            s_vo_beta,
            s_vv_beta,
        )
        overlaps_es_es_y_y_beta = self._compute_overlaps_es_es(
            y_beta_prev,
            y_beta_curr,
            s_oo_beta,
            s_ov_beta,
            s_vo_beta,
            s_vv_beta,
        )
        overlaps_es_es_alpha = (
            overlaps_es_es_x_x_alpha
            - overlaps_es_es_x_y_alpha
            - overlaps_es_es_y_x_alpha
            + overlaps_es_es_y_y_alpha
        )
        overlaps_es_es_beta = (
            overlaps_es_es_x_x_beta
            - overlaps_es_es_x_y_beta
            - overlaps_es_es_y_x_beta
            + overlaps_es_es_y_y_beta
        )
        # The X-Y vectors above give all XX-XY-YX+YY opposite-spin terms.
        overlaps_es_es_alpha_beta = np.outer(overlaps_es_gs_alpha, overlaps_gs_es_beta)
        overlaps_es_es_beta_alpha = np.outer(overlaps_es_gs_beta, overlaps_gs_es_alpha)

        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1),
            dtype=np.result_type(overlaps_es_es_alpha, overlaps_es_es_beta),
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap_alpha * gs_overlap_beta
        self.results["wf_overlaps"][0, 1:] = (
            overlaps_gs_es_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_gs_es_beta
        )
        self.results["wf_overlaps"][1:, 0] = (
            overlaps_es_gs_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_es_gs_beta
        )
        self.results["wf_overlaps"][1:, 1:] = (
            overlaps_es_es_alpha * gs_overlap_beta
            + gs_overlap_alpha * overlaps_es_es_beta
            + overlaps_es_es_alpha_beta
            + overlaps_es_es_beta_alpha
        )

    def _get_matrix_overlaps_s(self, num_electrons, mo_overlaps):
        """
        Compute blocks of the MO overlap matrix S. Return the blocks s_oo, s_ov, s_vo, s_vv.
        """
        s_oo = mo_overlaps[0:num_electrons, 0:num_electrons]
        s_ov = mo_overlaps[0:num_electrons, num_electrons:]
        s_vo = mo_overlaps[num_electrons:, 0:num_electrons]
        s_vv = mo_overlaps[num_electrons:, num_electrons:]
        return s_oo, s_ov, s_vo, s_vv

    def _compute_mo_overlaps(self):
        """
        Compute molecular orbital overlaps between two geometries from atomic orbital overlaps.

        Returns the molecular orbital overlaps for alpha and beta spin orbitals.
        """
        ao_overlaps = self._pull_ao_overlaps()

        mo_overlaps_alpha = (
            self.mo_alpha_prev.conj().T @ ao_overlaps @ self.mo_alpha_curr
        )

        mo_overlaps_beta = self.mo_beta_prev.conj().T @ ao_overlaps @ self.mo_beta_curr

        return mo_overlaps_alpha, mo_overlaps_beta

    def _compute_overlaps_gs_es(self, excited_amplitudes, s_oo, s_ov):
        """
        Compute single-spin GS(1)-ES(2) overlaps, including det(s_oo).
        The unchanged opposite-spin determinant is supplied by the caller.
        """
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix, s_ov)
        x = np.asarray(excited_amplitudes)
        overlaps_gs_es = determinant_a_matrix * np.einsum("eia,ia->e", x, w_matrix)
        return overlaps_gs_es

    def _compute_overlaps_es_gs(self, excited_amplitudes, s_oo, s_vo):
        """
        Compute single-spin ES(1)-GS(2) overlaps, including det(s_oo).
        The unchanged opposite-spin determinant is supplied by the caller.
        """
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix.T, s_vo.T).T
        x = np.asarray(excited_amplitudes).conj()
        overlaps_es_gs = determinant_a_matrix * np.einsum("eia,ai->e", x, w_matrix)
        return overlaps_es_gs

    def _compute_overlaps_es_es(
        self, geometry_1_amplitudes, geometry_2_amplitudes, s_oo, s_ov, s_vo, s_vv
    ):
        """
        Compute overlap between excited states at geometry 1 and excited states at geometry 2.
        """
        a_matrix = s_oo
        num_occupied_orbitals = a_matrix.shape[0]
        x1 = np.asarray(geometry_1_amplitudes)
        x2 = np.asarray(geometry_2_amplitudes)
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        inverse_a_matrix = np.linalg.solve(a_matrix, np.eye(num_occupied_orbitals))
        g_matrix = inverse_a_matrix @ s_ov
        overlap = np.zeros(
            (x1.shape[0], x2.shape[0]),
            dtype=np.result_type(x1, x2, s_oo, s_ov, s_vo, s_vv, np.float64),
        )
        for j in range(num_occupied_orbitals):
            aj = a_matrix[j, :]
            column_j = inverse_a_matrix[:, j]
            delta = s_vo - aj[None, :]
            delta_dot_column_j = delta @ column_j
            delta_dot_g = delta @ g_matrix
            for i in range(num_occupied_orbitals):
                q = inverse_a_matrix[i, j]
                gi = g_matrix[i, :]
                aji = a_matrix[j, i]
                delta_i = delta[:, i]
                alpha = s_vv - s_vo[:, i][:, None] - s_ov[j, :][None, :] + aji
                b21 = (delta_dot_g - delta_i[:, None]) + alpha * (gi[None, :] - 1.0)
                b22 = 1.0 + delta_dot_column_j[:, None] + alpha * q
                determinant_m = determinant_a_matrix * (gi[None, :] * b22 - q * b21)
                conjx = np.conj(x1[:, j, :])
                x2i = x2[:, i, :]
                overlap += np.einsum(
                    "eb,fa,ba->ef", conjx, x2i, determinant_m, optimize=True
                )
        return overlap

    def _pull_ao_overlaps(self):
        """
        Read the atomic orbital overlaps between two geometries from the Q-Chem scratch folder.
        """
        qcscratch = os.environ["QCSCRATCH"]  # Q-Chem scratch folder.
        file_ao_overlap = os.path.join(
            qcscratch, self.folder_scratch, "MO-overlaps", "AO-overlap-TwoGeoms.txt"
        )
        with open(file_ao_overlap, "r") as file:
            data = file.readlines()
        overlaps_init = []
        for line in data:
            temp_values = np.array(line.split(), dtype=float)
            if len(temp_values) == 8:
                overlaps_init.append(temp_values[:-1])
            elif len(temp_values) == 0:
                continue
            else:
                overlaps_init.append(temp_values)
        length = len(np.concatenate(overlaps_init))
        if length != (self.num_basis_functions * self.num_basis_functions):
            raise ValueError("The number of overlap integrals read is not correct")
        ao_overlaps = np.zeros((self.num_basis_functions, self.num_basis_functions))
        num_blocks = int(len(overlaps_init) / self.num_basis_functions)
        for i in range(self.num_basis_functions):
            temp_values = np.concatenate(
                [
                    overlaps_init[self.num_basis_functions * j + i]
                    for j in range(num_blocks)
                ]
            )
            ao_overlaps[:, i] = temp_values
        return ao_overlaps

    def _pull_excited_amplitudes(self):
        """
        Read the excited state amplitudes from the FCHK file generated by Q-Chem.

        Extracts the information of the excited state amplitudes for TDDFT or CIS calculations.
        Supports restricted and unrestricted references. The information that is
        extracted includes the alpha and beta x and y amplitudes for TDDFT (or x
        amplitudes for CIS), the number of basis functions, the number of alpha and
        beta electrons, the number of excited states, and the alpha and beta MO
        coefficients read from the Q-Chem scratch file 53.0.

        The results are returned as a tuple of
        (x_alpha, y_alpha, x_beta, y_beta, num_basis_functions, num_alpha_electrons,
        num_beta_electrons, num_excited_states, mo_alpha, mo_beta).
        """
        file_fchk = self.label + ".fchk"
        with open(file_fchk, "r") as f:
            data = f.readlines()
        keywords_to_find = [
            "Number of alpha electrons",
            "Number of beta electrons",
            "Number of basis functions",
            "Number of Excited States",
        ]
        keywords_to_find_alpha = [
            "Alpha X Amplitudes",
            "Alpha Y Amplitudes",
            "Alpha Amplitudes",
        ]
        keywords_to_find_beta = [
            "Beta X Amplitudes",
            "Beta Y Amplitudes",
            "Beta Amplitudes",
        ]
        keywords_to_stop_alpha = [
            "Beta X Amplitudes",
            "Beta Y Amplitudes",
            "Beta Amplitudes",
        ]
        keywords_to_stop_beta = [
            "Alpha Y Amplitudes",
            "Dipole_Data",
            "Number of independent functions",  # For older Q-Chem versions.
        ]

        for i, line in enumerate(data):
            if keywords_to_find[0] in line:
                num_alpha_electrons = int(line.split()[-1])
            if keywords_to_find[1] in line:
                num_beta_electrons = int(line.split()[-1])
            if keywords_to_find[2] in line:
                num_basis_functions = int(line.split()[-1])
            if keywords_to_find[3] in line:
                num_excited_states = int(line.split()[-1])
                index_restart = i

        # Get molecular coefficients from binary file in Q-Chem scratch folder.
        qcscratch = os.environ["QCSCRATCH"]  # Q-Chem scratch folder.
        file_mo_coeff = os.path.join(qcscratch, self.folder_scratch, "53.0")
        data_mo_coeff = np.fromfile(file_mo_coeff, dtype=float)
        mo_alpha = data_mo_coeff[: num_basis_functions * num_basis_functions]
        mo_beta = data_mo_coeff[
            num_basis_functions
            * num_basis_functions : 2
            * num_basis_functions
            * num_basis_functions
        ]
        x_alpha = []
        y_alpha = []
        x_beta = []
        y_beta = []
        for i, line in enumerate(data[index_restart:]):
            if keywords_to_find_alpha[0] in line:
                dimension_amplitudes_alpha = int(line.split()[-1])
                j = i + 1 + index_restart
                while keywords_to_stop_alpha[0] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find_alpha[1] in line:
                j = i + 1 + index_restart
                while keywords_to_stop_alpha[1] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    y_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find_alpha[2] in line:
                dimension_amplitudes_alpha = int(line.split()[-1])
                j = i + 1 + index_restart
                while keywords_to_stop_alpha[2] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    y_alpha.append(np.zeros_like(temp_values))
                    j = j + 1
            if keywords_to_find_beta[0] in line:
                dimension_amplitudes_beta = int(line.split()[-1])
                j = i + 1 + index_restart
                while keywords_to_stop_beta[0] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_beta.append(temp_values)
                    j = j + 1
            if keywords_to_find_beta[1] in line:
                j = i + 1 + index_restart
                while keywords_to_stop_beta[1] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    y_beta.append(temp_values)
                    j = j + 1
            if keywords_to_find_beta[2] in line:
                dimension_amplitudes_beta = int(line.split()[-1])
                j = i + 1 + index_restart
                while not any(key in data[j] for key in keywords_to_stop_beta[1:]):
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_beta.append(temp_values)
                    y_beta.append(np.zeros_like(temp_values))
                    j = j + 1

        # Check that the number of basis functions used for Q-Chem corresponds
        # to the number of basis functions determined from the basis set information.
        # Q-Chem sometimes automatically projects out near-linear dependencies,
        # which reduces the number of MOs below the number of basis functions.

        num_basis_functions_check_alpha = (
            int((dimension_amplitudes_alpha / num_excited_states) / num_alpha_electrons)
            + num_alpha_electrons
        )
        num_basis_functions_check_beta = (
            int((dimension_amplitudes_beta / num_excited_states) / num_beta_electrons)
            + num_beta_electrons
        )

        if (
            num_basis_functions_check_alpha != num_basis_functions
            or num_basis_functions_check_beta != num_basis_functions
        ):
            logger.critical(
                "The number of basis functions used for Q-Chem does not correspond to "
                "the number of basis functions determined from the basis set information."
            )
            logger.info(
                "Q-Chem sometimes automatically projects out near-linear dependencies, "
                "which reduces the number of MOs below the number of basis functions."
            )
            raise ValueError(
                "The number of basis functions used for Q-Chem does not correspond to "
                "the number of basis functions determined from the basis set information."
            )

        x_alpha = np.concatenate(x_alpha)
        x_alpha = np.reshape(
            x_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )
        y_alpha = np.concatenate(y_alpha)
        y_alpha = np.reshape(
            y_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )

        x_beta = np.concatenate(x_beta)
        x_beta = np.reshape(
            x_beta,
            (
                num_excited_states,
                num_beta_electrons,
                (num_basis_functions - num_beta_electrons),
            ),
        )
        y_beta = np.concatenate(y_beta)
        y_beta = np.reshape(
            y_beta,
            (
                num_excited_states,
                num_beta_electrons,
                (num_basis_functions - num_beta_electrons),
            ),
        )

        return (
            x_alpha,
            y_alpha,
            x_beta,
            y_beta,
            num_basis_functions,
            num_alpha_electrons,
            num_beta_electrons,
            num_excited_states,
            mo_alpha,
            mo_beta,
        )

    def write_input(self, **kwargs):
        """
        Write the Q-Chem input file for the requested properties.
        """
        kwargs = {key: value for key, value in kwargs.items()
                  if value.get("calc_property", True)}
        properties = kwargs.keys()
        filename = self.label + ".inp"
        job_specs = self._build_job_specs(properties)
        with open(filename, "w") as file_obj:
            self._write_job(file_obj, job_specs, **kwargs)
