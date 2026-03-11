"""
This module contains the Q-Chem interface for ab initio calculations.
"""

import os
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
        self.method_es = kwargs["method_es"]

        # Set default parameters.
        if kwargs.get("exchange", None) is None:
            kwargs["exchange"] = "B3LYP"
            self.method_es = "tddft"
        if kwargs.get("basis", None) is None:
            kwargs["basis"] = "6-31G*"
        # Ensuring that all kwargs are lowercase.
        kwargs = {str(k).lower(): v for k, v in kwargs.items()}
        self.method_es = self.method_es.lower()
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
        # Job specifications for Q-Chem input file.
        self.job_templates = {
            "energy": {
                "name": "energy",
                "write_derivative_coupling": False,
                "excited_amplitudes": False,
                "qchem_parameters": {
                    "cis": {
                        "jobtype": "SP",
                        "cis_n_roots": self.kwargs.get("cis_n_roots"),
                        "cis_singlets": self.kwargs.get("cis_singlets"),
                        "cis_triplets": self.kwargs.get("cis_triplets"),
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                    "tddft": {
                        "jobtype": "SP",
                        "cis_n_roots": self.kwargs.get("cis_n_roots"),
                        "cis_singlets": self.kwargs.get("cis_singlets"),
                        "cis_triplets": self.kwargs.get("cis_triplets"),
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                },
            },
            "gradient": {
                "name": "gradient",
                "write_derivative_coupling": False,
                "qchem_parameters": {
                    "cis": {
                        "jobtype": "FORCE",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                    "tddft": {
                        "jobtype": "FORCE",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                },
            },
            "frequency": {
                "name": "frequency",
                "write_derivative_coupling": False,
                "qchem_parameters": {
                    "cis": {
                        "jobtype": "FREQ",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                    "tddft": {
                        "jobtype": "FREQ",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                },
            },
            "derivative_coupling": {
                "name": "derivative_coupling",
                "write_derivative_coupling": True,
                "qchem_parameters": {
                    "cis": {
                        "jobtype": "SP",
                        "basis": self.kwargs.get("basis"),
                        "method": self.kwargs.get("exchange"),
                        "cis_n_roots": self.kwargs.get("cis_n_roots"),
                        "cis_singlets": self.kwargs.get("cis_singlets"),
                        "cis_triplets": self.kwargs.get("cis_triplets"),
                        "calc_nac": "TRUE",
                        "cis_der_numstate": int(self.kwargs.get("cis_n_roots")) + 1,
                        "sym_ignore": "TRUE",
                    },
                    "tddft": {
                        "jobtype": "SP",
                        "basis": self.kwargs.get("basis"),
                        "method": self.kwargs.get("exchange"),
                        "cis_n_roots": self.kwargs.get("cis_n_roots"),
                        "cis_singlets": self.kwargs.get("cis_singlets"),
                        "cis_triplets": self.kwargs.get("cis_triplets"),
                        "calc_nac": "TRUE",
                        "cis_der_numstate": int(self.kwargs.get("cis_n_roots")) + 1,
                        "sym_ignore": "TRUE",
                    },
                },
            },
            "wf_overlaps": {
                "name": "wf_overlaps",
                "write_derivative_coupling": False,
                "qchem_parameters": {
                    "cis": {
                        "jobtype": "SP",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                    "tddft": {
                        "jobtype": "SP",
                        "basis": self.kwargs.get("basis"),
                        "exchange": self.kwargs.get("exchange"),
                        "sym_ignore": "TRUE",
                    },
                },
            },
        }

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
        name  : label (e.g. 'gradient', 'derivative_coupling', 'frequency', 'energy')
        write_derivative_coupling : bool, whether to emit $derivative_coupling
        qchem_parameters : dict of dicts, keyed by method_es, of Q-Chem parameters to be written in $rem section.
        """

        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        # Check if the asked properties are implemented.
        self._check_requested_properties(properties)
        properties = list(dict.fromkeys(properties))
        return [self.job_templates[p].copy() for p in properties]

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

    def _add_parameters_to_job_spec(self, job_spec):
        """
        Add parameters from kwargs to job_spec.
        """
        if self.kwargs.get(job_spec["name"], None) is not None:
            self.kwargs[job_spec["name"]] = {
                str(k).lower(): v for k, v in self.kwargs[job_spec["name"]].items()
            }
            job_spec["qchem_parameters"][self.method_es] = {
                str(k).lower(): v
                for k, v in job_spec["qchem_parameters"][self.method_es].items()
            }
            for key, value in self.kwargs[job_spec["name"]].items():
                if key in job_spec["qchem_parameters"][self.method_es]:
                    job_spec["qchem_parameters"][self.method_es][key] = value
                elif key not in job_spec["qchem_parameters"][self.method_es]:
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
            job = self._add_parameters_to_job_spec(job)
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
                if job["name"] == "energy" and kwargs["energy"]["excited_amplitudes"]:
                    job["qchem_parameters"][self.method_es]["GUI"] = "2"
                    job["qchem_parameters"][self.method_es]["IQMOL_FCHK"] = "TRUE"
                if job["name"] == "derivative_coupling":
                    self._get_state_inds_derivative_coupling(
                        job,
                        kwargs["derivative_coupling"].get(
                            "state_inds_derivative_coupling", None
                        ),
                    )
                    if len(self.state_inds_derivative_coupling) != int(
                        job["qchem_parameters"][self.method_es].get("cis_der_numstate")
                    ):
                        job["qchem_parameters"][self.method_es]["cis_der_numstate"] = (
                            len(self.state_inds_derivative_coupling)
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
        if state_inds_gradient is None:
            num_states = int(self.kwargs.get("cis_n_roots")) + 1
            return [i for i in range(num_states)]
        elif isinstance(state_inds_gradient, (int, np.integer)):
            return [state_inds_gradient]
        else:
            return state_inds_gradient

    def _write_gradient_jobs(
        self, file_obj, job_spec, flag=0, state_inds_gradient=None
    ):
        """
        Handles the writing for gradient jobs requested.
        """
        self.state_inds_gradient = self._get_state_inds_gradient(state_inds_gradient)
        for j, i in enumerate(self.state_inds_gradient):
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
        if state_inds_derivative_coupling is None:
            n_s = int(job["qchem_parameters"][self.method_es].get("cis_der_numstate"))
            state_inds_derivative_coupling = [i for i in range(n_s)]
        elif isinstance(state_inds_derivative_coupling, (int, np.integer)):
            state_inds_derivative_coupling = [
                i for i in range(state_inds_derivative_coupling + 1)
            ]
        self.state_inds_derivative_coupling = state_inds_derivative_coupling

    def _write_derivative_coupling(self, file_obj, job):
        """
        Write the $derivative_coupling section to the Q-Chem input file.
        """
        file_obj.write("$derivative_coupling\n")
        file_obj.write(
            f"{self.state_inds_derivative_coupling[0]} is the reference state\n"
        )
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
        properties = kwargs.keys()
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
        corresponding to the requested properties, if the number of jobs
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
                    amplitudes_current=kwargs["wf_overlaps"].get(
                        "amplitudes_current", None
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
        # Number of lines between the Mode: line and the displacement-matrix line in a Q-Chem output file.
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
        # Get the number of states to be extracted from the output file.
        if self.kwargs.get("cis_n_roots") is None:
            nt_states = 1
        else:
            nt_states = int(self.kwargs.get("cis_n_roots")) + 1

        keywords_for_gs_energy = [
            "SCF time:",
            "Timing for Total SCF",
            "Total energy in the final basis set",
        ]
        keywords_for_excited_energy = ["Total energy for state"]

        i = 0
        energy = np.zeros(nt_states)
        for j, line in enumerate(file_obj):
            # Extract ground state energy.
            if any(word in line for word in keywords_for_gs_energy):
                try:
                    energy[i] = float(file_obj[j + 2].split()[-1])
                    i += 1
                except:
                    pass
            # Extract excited state energies.
            elif any(word in line for word in keywords_for_excited_energy):
                if i >= nt_states:
                    logger.warning(
                        "More states found in output file than specified by cis_n_roots."
                        " Only the first %d states will be extracted." % nt_states
                    )
                    break
                energy[i] = float(line.split()[-2])
                i += 1
        self.results["energy"] = energy
        # Extract excited state amplitudes if requested.
        if excited_amplitudes:
            self.results["excited_amplitudes"] = {}
            (
                x_alpha,
                y_alpha,
                num_basis_functions,
                num_alpha_electrons,
                num_excited_states,
            ) = self._pull_excited_amplitudes()
            self.results["excited_amplitudes"]["x"] = x_alpha
            self.results["excited_amplitudes"]["y"] = y_alpha
            self.results["excited_amplitudes"][
                "num_basis_functions"
            ] = num_basis_functions
            self.results["excited_amplitudes"][
                "num_alpha_electrons"
            ] = num_alpha_electrons
            self.results["excited_amplitudes"][
                "num_excited_states"
            ] = num_excited_states

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

    def _pull_derivative_coupling(self, file_obj, num_atoms, etf=True):
        """
        Parse derivative couplings between electronic states from Q-Chem output.

        Stores 'derivative_coupling' in ``self.results`` as a dictionary
        keyed by state-pair tuples.
        """
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
            base = l_etf[i] if etf else l_no_etf[i]
            for j in range(num_atoms):
                derivative_coupling_matrix[j, :] = np.array(
                    file_obj[base + 3 + j].split()[1:], dtype=float
                )
            derivative_coupling_dictionary[(i_st, j_st)] = (
                derivative_coupling_matrix.copy()
            )
        self.results["derivative_coupling"] = derivative_coupling_dictionary

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
        # Create instance attributes needed for computing overlaps.
        self.num_basis_functions = amplitudes_current["num_basis_functions"]
        self.num_alpha_electrons = amplitudes_current["num_alpha_electrons"]
        self.num_excited_states = amplitudes_current["num_excited_states"]
        # Extract molecular orbitals overlaps between two geometries.
        mo_overlaps = self._pull_mo_overlaps()
        # Compute overlaps matrices.
        if self.method_es.lower() == "tddft":
            self._get_overlaps_tddft(
                amplitudes_previous["x"],
                amplitudes_previous["y"],
                amplitudes_current["x"],
                amplitudes_current["y"],
                mo_overlaps,
            )
        elif self.method_es.lower() == "cis":
            self._get_overlaps_cis(
                amplitudes_previous["x"],
                amplitudes_current["x"],
                mo_overlaps,
            )
        else:
            raise ValueError(
                f"Method {self.method_es} not recognized. Supported methods are 'tddft' and 'cis'."
            )

    def _get_overlaps_cis(self, alpha_prev, alpha_curr, mo_overlaps):
        """
        Compute wavefunction overlaps using the CIS formalism.

        The results are stored in ``self.results["wf_overlaps"]`` as a matrix.
        """
        s_oo = mo_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        s_ov = mo_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        s_vo = mo_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        s_vv = mo_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo)
        gs_overlap = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_ex = self._compute_overlaps_gs_ex(alpha_curr, s_oo, s_ov)
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_ex_gs = self._compute_overlaps_ex_gs(alpha_prev, s_oo, s_vo)
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_ex_ex = self._compute_overlaps_ex_ex(
            alpha_prev, alpha_curr, s_oo, s_ov, s_vo, s_vv
        )
        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1)
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap
        self.results["wf_overlaps"][0, 1:] = 2.0 * overlaps_gs_ex
        self.results["wf_overlaps"][1:, 0] = 2.0 * overlaps_ex_gs
        self.results["wf_overlaps"][1:, 1:] = 2.0 * overlaps_ex_ex

    def _get_overlaps_tddft(self, x_prev, y_prev, x_curr, y_curr, mo_overlaps):
        """
        Compute wavefunction overlaps using the TDDFT formalism.

        The results are stored in ``self.results["wf_overlaps"]`` as a matrix.
        """
        s_oo = mo_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        s_ov = mo_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        s_vo = mo_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        s_vv = mo_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo)
        gs_overlap = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_ex_x = self._compute_overlaps_gs_ex(x_curr, s_oo, s_ov)
        overlaps_gs_ex_y = self._compute_overlaps_gs_ex(y_curr, s_oo, s_ov)
        overlaps_gs_ex = overlaps_gs_ex_x - overlaps_gs_ex_y
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_ex_gs_x = self._compute_overlaps_ex_gs(x_prev, s_oo, s_vo)
        overlaps_ex_gs_y = self._compute_overlaps_ex_gs(y_prev, s_oo, s_vo)
        overlaps_ex_gs = overlaps_ex_gs_x - overlaps_ex_gs_y
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_ex_ex_x_x = self._compute_overlaps_ex_ex(
            x_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_x_y = self._compute_overlaps_ex_ex(
            x_prev, y_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_x = self._compute_overlaps_ex_ex(
            y_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_y = self._compute_overlaps_ex_ex(
            y_prev, y_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex = (
            overlaps_ex_ex_x_x
            - overlaps_ex_ex_x_y
            - overlaps_ex_ex_y_x
            + overlaps_ex_ex_y_y
        )
        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1)
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap
        self.results["wf_overlaps"][0, 1:] = 2.0 * overlaps_gs_ex
        self.results["wf_overlaps"][1:, 0] = 2.0 * overlaps_ex_gs
        self.results["wf_overlaps"][1:, 1:] = 2.0 * overlaps_ex_ex

    def _compute_overlaps_gs_ex(self, excited_amplitudes, s_oo, s_ov):
        """
        Compute overlap between ground state at geometry 1 and excited states at geometry 2.
        """
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix, s_ov)
        x = np.asarray(excited_amplitudes)
        overlaps_gs_ex = determinant_a_matrix * np.einsum("eia,ia->e", x, w_matrix)
        return overlaps_gs_ex

    def _compute_overlaps_ex_gs(self, excited_amplitudes, s_oo, s_vo):
        """
        Compute overlap between excited states at geometry 1 and ground state at geometry 2.
        """
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix.T, s_vo.T).T
        x = np.asarray(excited_amplitudes)
        overlaps_ex_gs = determinant_a_matrix * np.einsum("eia,ai->e", x, w_matrix)
        return overlaps_ex_gs

    def _compute_overlaps_ex_ex(
        self, geometry_1_amplitudes, geometry_2_amplitudes, s_oo, s_ov, s_vo, s_vv
    ):
        """
        Compute overlap between excited states at geometry 1 and excited states at geometry 2.
        """
        a_matrix = s_oo
        num_occupied_orbitals = a_matrix.shape[0]
        x = np.asarray(geometry_1_amplitudes)
        x2 = np.asarray(geometry_2_amplitudes)
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        inverse_a_matrix = np.linalg.solve(a_matrix, np.eye(num_occupied_orbitals))
        g_matrix = inverse_a_matrix @ s_ov
        overlap = np.zeros((x.shape[0], x2.shape[0]))
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
                conjx = np.conj(x[:, j, :])
                x2ia = x2[:, i, :]
                overlap += np.einsum(
                    "eb,fa,ba->ef", conjx, x2ia, determinant_m, optimize=True
                )
        return overlap

    def _pull_mo_overlaps(self):
        """
        Read the molecular orbital overlaps between two geometries from the Q-Chem scratch folder.
        """
        qcscratch = os.environ["QCSCRATCH"]  # Q-Chem scratch folder.
        file_mo_overlap = os.path.join(
            qcscratch, self.folder_scratch, "MO-overlaps", "MO-overlap-TwoGeoms.txt"
        )
        with open(file_mo_overlap, "r") as file:
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
        mo_overlaps = np.zeros((self.num_basis_functions, self.num_basis_functions))
        num_blocks = int(len(overlaps_init) / self.num_basis_functions)
        for i in range(self.num_basis_functions):
            temp_values = np.concatenate(
                [
                    overlaps_init[self.num_basis_functions * j + i]
                    for j in range(num_blocks)
                ]
            )
            mo_overlaps[i, :] = temp_values
        mo_overlaps = mo_overlaps.T
        return mo_overlaps

    def _pull_excited_amplitudes(self):
        """
        Read the excited state amplitudes from the FCHK file generated by Q-Chem.

        Extracts the information of the excited state amplitudes for TDDFT or CIS calculations.
        It only supports closed-shell calculations.
        The information that is extracted includes the x and y amplitudes for TDDFT or
        alpha amplitudes for CIS, as well as the number of basis functions,
        the number of alpha electrons, and the number of excited states.

        The results are returned as a tuple of
        (x_alpha, y_alpha, num_basis_functions, num_alpha_electrons, num_excited_states).
        """
        file_fchk = self.label + ".fchk"
        with open(file_fchk, "r") as f:
            data = f.readlines()
        keywords_to_find = [
            "Number of alpha electrons",
            "Number of beta electrons",
            "Number of basis functions",
            "Number of Excited States",
            "Alpha X Amplitudes",
            "Alpha Y Amplitudes",
            "Alpha Amplitudes",
        ]
        keyword_to_stop = ["Beta X Amplitudes", "Beta Y Amplitudes", "Beta Amplitudes"]

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

        if num_alpha_electrons != num_beta_electrons:
            raise ValueError("The interface only supports closed-shell calculations.")
        x_alpha = []
        y_alpha = []
        for i, line in enumerate(data[index_restart:]):
            if keywords_to_find[4] in line:
                dimension_amplitudes = int(line.split()[-1])
                j = i + 1 + index_restart
                while keyword_to_stop[0] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[5] in line:
                j = i + 1 + index_restart
                while keyword_to_stop[1] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    y_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[6] in line:
                dimension_amplitudes = int(line.split()[-1])
                j = i + 1 + index_restart
                while keyword_to_stop[2] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    y_alpha.append(np.zeros_like(temp_values))
                    j = j + 1

        # Check that the number of basis functions used for Q-Chem corresponds
        # to the number of basis functions determined from the basis set information.
        # Q-Chem sometimes automatically projects out near-linear dependencies,
        # which reduces the number of MOs below the number of basis functions.

        num_basis_functions_check = (
            int((dimension_amplitudes / num_excited_states) / num_alpha_electrons)
            + num_alpha_electrons
        )
        if num_basis_functions_check != num_basis_functions:
            logger.critical(
                "The number of basis functions used for Q-Chem does not correspond to "
                "the number of basis functions determined from the basis set information."
            )
            logger.info(
                "Q-Chem sometimes automatically projects out near-linear dependencies",
                "which reduces the number of MOs below the number of basis functions.",
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
        return (
            x_alpha,
            y_alpha,
            num_basis_functions,
            num_alpha_electrons,
            num_excited_states,
        )

    def write_input(self, **kwargs):
        """
        Write the Q-Chem input file for the requested properties.
        """
        properties = kwargs.keys()
        filename = self.label + ".inp"
        job_specs = self._build_job_specs(properties)
        with open(filename, "w") as file_obj:
            self._write_job(file_obj, job_specs, **kwargs)
