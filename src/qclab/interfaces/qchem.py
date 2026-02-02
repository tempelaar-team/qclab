"""
This module contains the Q-Chem interface for ab initio calculations.
"""

import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase import Atoms
from qclab import numerical_constants


class QCLabQChemInterface(FileIOCalculator):
    """
    Q-Chem ASE interface for QC Lab calculations.

    Based on the ASE Q-Chem calculator:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/qchem
    """

    _legacy_default_command = "qchem PREFIX.inp PREFIX.out"

    def __init__(
        self,
        atom_positions,  # In units of Bohr.
        atom_masses,  # In units of electron mass.
        atom_names,
        label="qchem",
        folder_scratch="qclab_job",
        nt=1,
        np=1,
        **kwargs,
    ):
        ignore_bad_restart_file = FileIOCalculator._deprecated
        restart = None
        # Ensuring that all kwargs are lowercase.
        kwargs = {str(k).lower(): v for k, v in kwargs.items()}
        atoms = Atoms(
            symbols=atom_names,
            positions=atom_positions / numerical_constants.ANGSTROM_TO_BOHR,
        )
        atoms.set_masses(atom_masses / numerical_constants.AMU_TO_EMASS)
        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        # Store objects globally.
        self.atom_names = atom_names
        self.atom_masses = atom_masses
        self.atom_positions = atom_positions
        self.atoms = atoms
        self.folder_scratch = folder_scratch
        # Build command.
        self.command = "qchem"
        if np != 1:
            self.command += f" -np {np}"
        if nt != 1:
            self.command += f" -nt {nt}"
        self.command += " PREFIX.inp PREFIX.out"
        self.command += "\t" + folder_scratch
        self.command += " > /dev/null 2>&1"
        # Properties implemented.
        self.implemented_properties = [
            "energy",
            "gradient",
            "derivative_coupling",
            "frequency",
            "wf_overlaps",
        ]

    def read(self, label):
        raise NotImplementedError

    def _build_job_specs(self, properties):
        """
        Map ASE properties for a list of Q-Chem jobs.

        Each job spec is a dict with:
            name  : label (e.g. 'gradient', 'derivative_coupling', 'frequency', 'energy')
            jobtype : Q-Chem JOBTYPE ('FORCE', 'FREQ', 'SP')
            write_derivative_coupling : bool, whether to emit $derivative_coupling
        """
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        props = set(properties)
        # Check if the asked properties are implemented.
        self._check_requested_properties(properties)
        job_specs = []
        for prop in props:
            if "energy" in prop:
                job_specs.append(
                    {
                        "name": "energy",
                        "jobtype": "SP",
                        "write_derivative_coupling": False,
                    }
                )
            if "gradient" in prop:
                job_specs.append(
                    {
                        "name": "gradient",
                        "jobtype": "FORCE",
                        "write_derivative_coupling": False,
                    }
                )
            if "frequency" in prop:
                job_specs.append(
                    {
                        "name": "frequency",
                        "jobtype": "FREQ",
                        "write_derivative_coupling": False,
                    }
                )
            if "derivative_coupling" in prop:
                job_specs.append(
                    {
                        "name": "derivative_coupling",
                        "jobtype": "SP",
                        "write_derivative_coupling": True,
                    }
                )
            if "wf_overlaps" in prop:
                job_specs.append(
                    {
                        "name": "wf_overlaps",
                        "jobtype": "SP",
                        "write_derivative_coupling": False,
                    }
                )
        return job_specs

    def _check_requested_properties(self, job_requested):
        """
        Checks if the requested properties are implemented.
        """
        for job in job_requested:
            if job not in self.implemented_properties:
                raise ValueError(f"Requested property '{job}' is not implemented.")

    def _write_job(self, file_obj, job_spec, **kwargs):
        """
        Write  Q-Chem jobs (single job or multiple jobs that share same geometry).
        """
        for i, job in enumerate(job_spec):
            if "gradient" in job["name"]:
                # This determines if the gradient job belongs to multiple jobs or a single one.
                if len(job_spec) == 1 or i == 0:
                    flag = 0
                else:
                    flag = i + 1
                self._write_gradient_jobs(
                    file_obj,
                    job,
                    flag,
                    kwargs["gradient"].get("state_inds_gradient", None),
                )
            elif "wf_overlaps" in job["name"]:
                if len(job_spec) == 1 or i == 0:
                    flag = 0
                else:
                    flag = i + 1
                if kwargs["wf_overlaps"]["atom_positions_previous"] is None:
                    raise ValueError(
                        "Previous geometry must be provided"
                        "when requesting wf_overlaps."
                    )
                atoms_previous = Atoms(
                    symbols=self.atom_names,
                    positions=kwargs["wf_overlaps"]["atom_positions_previous"]
                    / numerical_constants.ANGSTROM_TO_BOHR,
                )
                atoms_previous.set_masses(
                    self.atom_masses / numerical_constants.AMU_TO_EMASS
                )
                self._write_wf_overlaps_jobs(file_obj, job, atoms_previous, flag)
            else:
                self._write_comments_qchem(file_obj, job["name"], ind=i)
                self._write_geometry_qchem(file_obj, ind=i)
                if job["name"] == "energy":
                    excited_amplitudes_flag = kwargs["energy"]["excited_amplitudes"]
                else:
                    excited_amplitudes_flag = False
                self._write_job_definition_qchem(
                    file_obj,
                    job,
                    excited_amplitudes=excited_amplitudes_flag,
                    state_inds_derivative_coupling=kwargs.get(
                        "derivative_coupling", {}
                    ).get("state_inds_derivative_coupling", None),
                )
                if job["write_derivative_coupling"]:
                    self._write_derivative_coupling_qchem(
                        file_obj,
                        kwargs.get("derivative_coupling", {}).get(
                            "state_inds_derivative_coupling", None
                        ),
                    )

    def _write_gradient_jobs(
        self, file_obj, job_spec, flag=0, state_inds_gradient=None
    ):
        """
        Handles the writing for gradient jobs requested.
        """
        if state_inds_gradient == None:
            n_s = int(self.parameters.get("cis_n_roots", 1)) + 1
            state_inds_gradient = [i for i in range(n_s)]
        elif isinstance(state_inds_gradient, (int, np.integer)):
            state_inds_gradient = [state_inds_gradient]

        for j, i in enumerate(state_inds_gradient):
            if i > 0:
                self.parameters["cis_state_deriv"] = str(i)
            if flag == 0:
                ind = j
            else:
                ind = flag
            self._write_comments_qchem(file_obj, f"gradient state {i}", ind)
            self._write_geometry_qchem(file_obj, ind)
            self._write_job_definition_qchem(file_obj, job_spec, flag=i)

    def _write_wf_overlaps_jobs(self, file_obj, job_spec, atoms_previous, flag=0):
        """
        Handles the writing for wavefunction overlaps jobs requested.
        """
        ind = flag
        # Create the input for the previous geometry.
        self._write_comments_qchem(file_obj, job_spec["name"] + " previous step", ind)
        # Write geometry.
        file_obj.write("$molecule\n")
        if (
            "multiplicity" not in self.parameters
            or self.parameters["multiplicity"] is None
        ):
            tot_magmom = self.atoms.get_initial_magnetic_moments().sum()
            mult = int(tot_magmom) + 1
        else:
            mult = int(self.parameters["multiplicity"])
        charge = int(self.parameters.get("charge", 0))
        file_obj.write("   %d %d\n" % (charge, mult))

        for a in atoms_previous:
            file_obj.write("   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z))
        file_obj.write("$end\n\n")
        # Job definition.
        keys_for_wf_overlaps = ["mo_overlaps_two_geoms"]
        if all(key in self.parameters for key in keys_for_wf_overlaps):
            pass
        else:
            self.parameters["mo_overlaps_two_geoms"] = 1
        if self.parameters.get("mo_overlaps_two_geoms") != 1:
            self.parameters["mo_overlaps_two_geoms"] = 1
        self._write_job_definition_qchem(file_obj, job_spec)

        # Creating input for the current geometry.
        self._write_comments_qchem(
            file_obj, job_spec["name"] + " current step", ind + 1
        )
        self._write_geometry_qchem(file_obj, ind=0)
        self.parameters["mo_overlaps_two_geoms"] = 2
        self._write_job_definition_qchem(file_obj, job_spec)

    def _write_comments_qchem(self, file_obj, job, ind=0):
        """
        Write comments and job separator for Q-Chem input file.
        """
        if ind != 0:
            file_obj.write("@@@\n\n")
        file_obj.write("$comment\n")
        file_obj.write(f"QC Lab generated input file ({job} job)\n")
        file_obj.write("$end\n\n")

    def _write_geometry_qchem(self, file_obj, ind=0):
        if ind == 0:
            file_obj.write("$molecule\n")
            if (
                "multiplicity" not in self.parameters
                or self.parameters["multiplicity"] is None
            ):
                tot_magmom = self.atoms.get_initial_magnetic_moments().sum()
                mult = int(tot_magmom) + 1
            else:
                mult = int(self.parameters["multiplicity"])
            charge = int(self.parameters.get("charge", 0))
            file_obj.write("   %d %d\n" % (charge, mult))
            for a in self.atoms:
                file_obj.write(
                    "   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z)
                )
            file_obj.write("$end\n\n")
        else:
            file_obj.write("$molecule\n")
            file_obj.write("read\n")
            file_obj.write("$end\n\n")

    def _write_job_definition_qchem(
        self,
        file_obj,
        job_spec,
        flag=1,
        excited_amplitudes=False,
        state_inds_derivative_coupling=None,
    ):
        file_obj.write("$rem\n")
        file_obj.write(f"   JOBTYPE           {job_spec['jobtype']}\n")
        parameters = self._get_qchem_job_specific_parameters(
            job=job_spec["name"],
            flag=flag,
            excited_amplitudes=excited_amplitudes,
            state_inds_derivative_coupling=state_inds_derivative_coupling,
        )
        for parameter, v_str in parameters.items():
            file_obj.write("   %-25s   %s\n" % (parameter.upper(), v_str))
            # Always ignore symmetry; otherwise, the coordinates will change the previously established origin.
        file_obj.write("   %-25s   %s\n" % ("SYM_IGNORE", "TRUE"))
        file_obj.write("$end\n\n")

    def _get_qchem_job_specific_parameters(self, **kwargs):
        parameters = {}
        if kwargs["job"] == "energy":
            keywords_to_avoid = [
                "calc_nac",
                "cis_der_numstate",
                "cis_state_deriv",
                "mo_overlaps_two_geoms",
                "charge",
                "multiplicity",
                "jobtype",
            ]
            if kwargs["excited_amplitudes"]:
                self.parameters["GUI"] = "2"
                self.parameters["IQMOL_FCHK"] = "TRUE"
        elif kwargs["job"] == "gradient":
            if kwargs["flag"] == 0:
                keywords_to_avoid = [
                    "calc_nac",
                    "cis_der_numstate",
                    "mo_overlaps_two_geoms",
                    "charge",
                    "multiplicity",
                    "jobtype",
                    "cis_state_deriv",
                    "cis_n_roots",
                    "cis_singlets",
                    "cis_triplets",
                    "internal_stability",
                    "gui",
                    "iqmol_fchk",
                ]
            else:
                keywords_to_avoid = [
                    "calc_nac",
                    "cis_der_numstate",
                    "mo_overlaps_two_geoms",
                    "charge",
                    "multiplicity",
                    "jobtype",
                    "internal_stability",
                    "gui",
                    "iqmol_fchk",
                ]
        elif kwargs["job"] == "derivative_coupling":
            keywords_to_avoid = [
                "charge",
                "multiplicity",
                "jobtype",
                "internal_stability",
                "mo_overlaps_two_geoms",
                "gui",
                "iqmol_fchk",
            ]
            self.parameters["calc_nac"] = True
            if kwargs["state_inds_derivative_coupling"] is None:
                self.parameters["cis_der_numstate"] = (
                    int(self.parameters["cis_n_roots"]) + 1
                )
        elif kwargs["job"] == "wf_overlaps":
            keywords_to_avoid = [
                "calc_nac",
                "cis_der_numstate",
                "charge",
                "multiplicity",
                "jobtype",
                "cis_state_deriv",
                "cis_n_roots",
                "cis_singlets",
                "cis_triplets",
                "internal_stability",
                "gui",
                "iqmol_fchk",
            ]
        elif kwargs["job"] == "frequency":
            keywords_to_avoid = [
                "calc_nac",
                "cis_der_numstate",
                "mo_overlaps_two_geoms",
                "charge",
                "multiplicity",
                "jobtype",
                "gui",
                "iqmol_fchk",
            ]
        for param, value in self.parameters.items():
            if param.lower() in keywords_to_avoid:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                v_str = value.upper()
            else:
                v_str = str(value)
            parameters[param] = v_str
        return parameters

    def _write_derivative_coupling_qchem(
        self, file_obj, state_inds_derivative_coupling=None
    ):
        if state_inds_derivative_coupling is None:
            n_s = int(self.parameters.get("cis_der_numstate", 1))
            state_inds_derivative_coupling = [i for i in range(n_s)]
        elif isinstance(state_inds_derivative_coupling, (int, np.integer)):
            state_inds_derivative_coupling = [
                i for i in range(state_inds_derivative_coupling + 1)
            ]
        file_obj.write("$derivative_coupling\n")
        file_obj.write(f"{state_inds_derivative_coupling[0]} is the reference state\n")
        file_obj.write(" ".join(str(s) for s in state_inds_derivative_coupling) + "\n")
        file_obj.write("$end\n\n")

    def read_results(self, **kwargs):

        properties = kwargs.keys()
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        num_atoms = len(self.atoms)
        filename = self.label + ".out"
        with open(filename, "r") as file_obj:
            file_content = file_obj.read()
        file_obj.close()
        state_inds_gradient = kwargs.get("gradient", {}).get(
            "state_inds_gradient", None
        )
        self._check_qchem_normal_termination(
            file_content, properties, state_inds_gradient
        )
        self._organize_data_qchem_file(file_content, properties, num_atoms, **kwargs)

    def _check_qchem_normal_termination(
        self, file_content, properties, state_inds_gradient
    ):
        """
        Check Q-Chem normal termination.
        """
        normal_termination = "for using Q-Chem"
        if "gradient" in properties:
            if state_inds_gradient is None:
                num_qchem_jobs = int(self.parameters.get("cis_n_roots", 1)) + len(
                    properties
                )
            elif isinstance(state_inds_gradient, (int, np.integer)):
                num_qchem_jobs = 1 + len(properties) - 1
            else:
                num_qchem_jobs = len(state_inds_gradient) + len(properties) - 1
            if file_content.count(normal_termination) < num_qchem_jobs:
                raise ValueError("Q-Chem did not terminate normally.")
        elif "wf_overlaps" in properties:
            num_qchem_jobs = len(properties) + 1
            if file_content.count(normal_termination) < num_qchem_jobs:
                raise ValueError("Q-Chem did not terminate normally.")
        else:
            if file_content.count(normal_termination) < len(properties):
                raise ValueError("Q-Chem did not terminate normally.")

    def _organize_data_qchem_file(self, file_content, properties, num_atoms, **kwargs):
        """
        Organize Q-Chem data according to the requested properties.
        """
        normal_termination = "for using Q-Chem"
        for job in properties:
            if "gradient" in job:
                job_comment = f"QC Lab generated input file (gradient state"
                temporal_data = file_content.split(job_comment)
                state_inds_gradient = kwargs["gradient"].get(
                    "state_inds_gradient", None
                )
                if state_inds_gradient is None:
                    num_qchem_jobs = int(self.parameters.get("cis_n_roots", 1)) + 1
                elif isinstance(state_inds_gradient, (int, np.integer)):
                    num_qchem_jobs = 1
                else:
                    num_qchem_jobs = len(state_inds_gradient)
                gradient_files = temporal_data[1:num_qchem_jobs]
                gradient_files.append(
                    temporal_data[num_qchem_jobs].split(normal_termination)[0]
                )
                self._pull_data(
                    job,
                    gradient_files,
                    num_atoms,
                    state_inds_gradient=state_inds_gradient,
                )
            elif "wf_overlaps" in job:
                job_comment = f"QC Lab generated input file (wf_overlaps"
                temporal_data = file_content.split(job_comment)
                coupling_files = temporal_data[2].split(normal_termination)[0]
                self._pull_data(
                    job,
                    coupling_files.splitlines(),
                    num_atoms,
                    amplitudes_previous=kwargs["wf_overlaps"].get(
                        "amplitudes_previous", None
                    ),
                    amplitudes_current=kwargs["wf_overlaps"].get(
                        "amplitudes_current", None
                    ),
                )
            else:
                job_comment = f"QC Lab generated input file ({job}"
                temporal_data = file_content.split(job_comment)
                job_file = temporal_data[1].split(normal_termination)[0]
                if "energy" in job:
                    excited_amplitudes = kwargs["energy"].get(
                        "excited_amplitudes", False
                    )
                else:
                    excited_amplitudes = False
                self._pull_data(
                    job,
                    job_file.splitlines(),
                    num_atoms,
                    excited_amplitudes=excited_amplitudes,
                )

    def _pull_data(self, property, file_obj, num_atoms, **kwargs):
        if "energy" in property:
            self._pull_energy(
                file_obj, excited_amplitudes=kwargs.get("excited_amplitudes", False)
            )
        elif "gradient" in property:
            self._pull_gradient(
                file_obj, num_atoms, kwargs.get("state_inds_gradient", None)
            )
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
        num_lines = 8  # Number of lines between the Mode: line and the displacement-matrix line in a Q-Chem output file.
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
        if self.parameters.get("cis_n_roots") is None:
            nt_states = 1
        else:
            nt_states = int(self.parameters.get("cis_n_roots")) + 1

        energy = np.zeros(nt_states)
        i = 0
        gs_found = False
        for j, line in enumerate(file_obj):
            if "SCF time:" in line:
                energy[i] = float(file_obj[j + 2].split()[-1])
                i += 1
                gs_found = True
            elif "Timing for Total SCF" in line and not (gs_found):
                try:
                    energy[i] = float(file_obj[j + 2].split()[-1])
                    i += 1
                    gs_found = True
                except:
                    pass
            elif "Total energy in the final basis set" in line and not (gs_found):
                energy[i] = float(line.split()[-1])
                i += 1
                gs_found = True
            elif "Total energy for state" in line:
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
            ) = self._pull_excited_amplitudes_qchem()
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

    def _pull_gradient(self, file_obj, num_atoms, state_inds_gradient=None):
        if state_inds_gradient is None:
            num_gradient = int(self.parameters.get("cis_n_roots", 1)) + 1
            state_inds_gradient = [i for i in range(num_gradient)]
        elif isinstance(state_inds_gradient, (int, np.integer)):
            state_inds_gradient = [state_inds_gradient]

        gradient = np.zeros((num_atoms, 3, len(state_inds_gradient)))
        flag_words_grad = ["Gradient of the state energy", "Gradient of SCF Energy"]
        flag_words_out_grad = ["Gradient time", "Max gradient component"]
        for state_ind in range(len(state_inds_gradient)):
            output_data = file_obj[state_ind].splitlines()
            for ind, line in enumerate(output_data):
                if any(word in line for word in flag_words_grad):
                    i_count = ind + 1
                    line_2 = output_data[i_count]
                    while not any(word in line_2 for word in flag_words_out_grad):
                        jndx = np.array(line_2.split(), dtype=int) - 1
                        gradient[jndx, 0, state_ind] = np.array(
                            output_data[i_count + 1].split()[1:], dtype=float
                        )
                        gradient[jndx, 1, state_ind] = np.array(
                            output_data[i_count + 2].split()[1:], dtype=float
                        )
                        gradient[jndx, 2, state_ind] = np.array(
                            output_data[i_count + 3].split()[1:], dtype=float
                        )
                        i_count += 4
                        line_2 = output_data[i_count]
                    break
        self.results["gradient"] = gradient

    def _pull_derivative_coupling(self, file_obj, num_atoms, etf=True):
        l_states = []  # Line that identifies the states involved in derivative coupling.
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
        # Create global variables needed for computing overlaps.
        self.num_basis_functions = amplitudes_current["num_basis_functions"]
        self.num_alpha_electrons = amplitudes_current["num_alpha_electrons"]
        self.num_excited_states = amplitudes_current["num_excited_states"]
        # Extract molecular orbitals overlaps between two geometries.
        mo_overlaps = self._pull_mo_overlaps_qchem()
        # Compute overlaps matrices.
        self._get_overlaps_TDDFT(
            amplitudes_previous["x"],
            amplitudes_previous["y"],
            amplitudes_current["x"],
            amplitudes_current["y"],
            mo_overlaps,
        )

    def _get_overlaps_TDDFT(self, x_prev, y_prev, x_curr, y_curr, mo_overlaps):
        s_oo = mo_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        s_ov = mo_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        s_vo = mo_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        s_vv = mo_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        # Compute <GS^(1)|GS^(2)>.
        gs_overlap = np.linalg.det(s_oo)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_ex_x = self._compute_overlap_gs_ex(x_curr, s_oo, s_ov)
        overlaps_gs_ex_y = self._compute_overlap_gs_ex(y_curr, s_oo, s_ov)
        overlaps_gs_ex = overlaps_gs_ex_x - overlaps_gs_ex_y
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_ex_gs_x = self._compute_overlap_ex_gs(x_prev, s_oo, s_vo)
        overlaps_ex_gs_y = self._compute_overlap_ex_gs(y_prev, s_oo, s_vo)
        overlaps_ex_gs = overlaps_ex_gs_x - overlaps_ex_gs_y
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_ex_ex_x_x = self._compute_overlap_ex_ex(
            x_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_x_y = self._compute_overlap_ex_ex(
            x_prev, y_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_x = self._compute_overlap_ex_ex(
            y_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_y = self._compute_overlap_ex_ex(
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

    def _compute_overlap_gs_ex(self, excited_amplitudes, s_oo, s_ov):
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix, s_ov)
        x = np.asarray(excited_amplitudes)
        overlaps_gs_ex = determinant_a_matrix * np.einsum("eia,ia->e", x, w_matrix)
        return overlaps_gs_ex

    def _compute_overlap_ex_gs(self, excited_amplitudes, s_oo, s_vo):
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix.T, s_vo.T).T
        x = np.asarray(excited_amplitudes)
        overlaps_ex_gs = determinant_a_matrix * np.einsum("eia,ai->e", x, w_matrix)
        return overlaps_ex_gs

    def _compute_overlap_ex_ex(
        self, geometry_1_amplitudes, geometry_2_amplitudes, s_oo, s_ov, s_vo, s_vv
    ):
        a_matrix = s_oo
        num_occpied_orbitals = a_matrix.shape[0]
        x = np.asarray(geometry_1_amplitudes)
        x2 = np.asarray(geometry_2_amplitudes)
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinat_a_matrix = sign * np.exp(log_determinant)
        inverse_a_matrix = np.linalg.solve(a_matrix, np.eye(num_occpied_orbitals))
        g_matrix = inverse_a_matrix @ s_ov
        overlap = np.zeros((x.shape[0], x2.shape[0]))
        for j in range(num_occpied_orbitals):
            aj = a_matrix[j, :]
            column_j = inverse_a_matrix[:, j]
            delta = s_vo - aj[None, :]
            delta_dot_column_j = delta @ column_j
            delta_dot_g = delta @ g_matrix
            for i in range(num_occpied_orbitals):
                q = inverse_a_matrix[i, j]
                gi = g_matrix[i, :]
                aji = a_matrix[j, i]
                delta_i = delta[:, i]
                alpha = s_vv - s_vo[:, i][:, None] - s_ov[j, :][None, :] + aji
                b21 = (delta_dot_g - delta_i[:, None]) + alpha * (gi[None, :] - 1.0)
                b22 = 1.0 + delta_dot_column_j[:, None] + alpha * q
                determinant_m = determinat_a_matrix * (gi[None, :] * b22 - q * b21)
                conjx = np.conj(x[:, j, :])
                x2ia = x2[:, i, :]
                overlap += np.einsum(
                    "eb,fa,ba->ef", conjx, x2ia, determinant_m, optimize=True
                )
        return overlap

    def _pull_mo_overlaps_qchem(self):
        qcscratch = os.environ["QCSCRATCH"]  # Q-Chem scratch folder.
        file_mo_overlap = (
            qcscratch
            + "/"
            + self.folder_scratch
            + "/"
            + "MO-overlaps/MO-overlap-TwoGeoms.txt"
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

    def _pull_excited_amplitudes_qchem(self):
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
            raise ValueError("The software only supports closed-shell calculations.")
        x_alpha = []
        y_alpha = []
        for i, line in enumerate(data[index_restart:]):
            if keywords_to_find[4] in line:
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
                j = i + 1 + index_restart
                while keyword_to_stop[2] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    y_alpha.append(np.zeros_like(temp_values))
                    j = j + 1
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
        system_changes = None
        properties = kwargs.keys()
        FileIOCalculator.write_input(self, self.atoms, properties, system_changes)
        filename = self.label + ".inp"
        job_specs = self._build_job_specs(properties)
        with open(filename, "w") as file_obj:
            self._write_job(file_obj, job_specs, **kwargs)
