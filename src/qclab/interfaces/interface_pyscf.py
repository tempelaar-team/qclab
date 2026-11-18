"""
This module contains the PySCF interface for ab initio calculations.
"""

import os
import subprocess
import logging
import numpy as np
from qclab import numerical_constants
import pyscf as qm
from pyscf import lib,dft,scf,tddft,tdscf,hessian
from pyscf.hessian import thermo
from pyscf.scf import ucphf
from qclab.numerical_constants import ANGSTROM_TO_BOHR, AMU_TO_EMASS


logger = logging.getLogger(__name__)


class QCLabPySCFInterface:
    """
    PySCF interface for ab initio calculations. 
    """

    def __init__(
        self,
        atom_masses,  # In units of electron mass.
        atom_names,
        label="pyscf",
        folder_scratch="qclab_job",
        num_threads=1,
        num_procs=1,
        **kwargs,
    ):
    # Store instance attributes.
        self.atom_names = atom_names
        self.atom_masses = atom_masses
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
        # Setting default param
        self.kwargs["method_es"] = self.kwargs.get("method_es","tddft")
        self.kwargs["basis"] = self.kwargs.get("basis","6-31g")
        self.kwargs["functional"] = self.kwargs.get("functional","pbe")
        self.kwargs["unrestricted"] = self.kwargs.get("unrestricted",False)
        self.kwargs["singlet"] = self.kwargs.get("singlet",True)

    def calculator(self, atom_positions, properties):
        """
        Main function that handles the calculation and results retrieval
        for the requested properties. 
        """
        properties = {
            name.lower(): parameters
            for name, parameters in properties.items()
            }
        # Check if the requested properties are implemented
        self._check_properties(properties)
        # Reseting results and electronic states from the previous geometry.
        self.results = {}
        self.mf = None
        self.es = None
        # Set molecule
        self.mol = self._molecule(atom_positions)
        # Compute ground state
        self.mf = self._gs(self.mol)
        # Compute each requested property
        self.eflag = 0
        if "energy" in properties:
            self.energy(properties["energy"])
            self.eflag = 1
        for prop in properties.keys():
            if prop.lower() == "frequency":
                self.vibration(properties[prop],self.eflag)
            elif prop.lower() == "wf_overlaps":
                self.wf_overlaps(properties[prop],self.eflag)
            elif prop.lower() == "derivative_coupling":
                self.derivative_coupling(properties[prop],self.eflag)
            elif prop.lower() == "gradient":
                self.gradient(properties[prop],self.eflag)


    def _check_properties(self, properties):
        for prop in properties.keys():
            if prop not in self.implemented_properties:
                raise ValueError(f"Property {prop} is not implemented.")

    def energy(self, energy_parameters):
        """
        Compute The energy of the system.
        """
        self.es = self._excited_states(energy_parameters)
        self.results["energy"] = np.zeros(self.kwargs["nroots"]+1)
        self.results["energy"][0] = self.mf.e_tot
        self.results["energy"][1:] =  self.es.e_tot

        if energy_parameters["excited_amplitudes"]:
            self.results["excited_amplitudes"] = {"xy":self.es.xy,
                                                  "MO_coeff":self.mf.mo_coeff}

    def vibration(self, vibration_parameters,eflag):
        """
        Compute the vibrational frequencies and normal modes of the system.
        """
        if vibration_parameters["state_vibrations"]>0 and eflag ==0:
            self.es = self._excited_states()
            self.eflag = 1
        self._frequencies(vibration_parameters)
        self.results["frequency"] = self.vib["freq_wavenumber"]
        self.results["normal_mode"] = self.vib["norm_mode"]

    def gradient(self, graident_parameters, eflag):
        """
        Compute gradients of the system for the requested states.
        """
        # Check the states where the gradient is going to be compute
        graident_parameters["state_inds_gradient"] = self._get_state_inds_gradient(
                                            graident_parameters["state_inds_gradient"]
                                            )
        if (
            any(state > 0 for state in graident_parameters["state_inds_gradient"])
            and eflag == 0
            ):
            self.es = self._excited_states()
            self.eflag = 1
        self.results["gradient"] = self._compute_gradients(graident_parameters)


    def wf_overlaps(self, overlaps_parameters, eflag):
        """
        Compute the wavefunction overlaps between different geometries
        """
        # Check that previous parameters are provided
        if overlaps_parameters.get("atom_positions_previous", None) is None:
            raise ValueError("Previous geometry must be provided.")
        if overlaps_parameters.get("amplitudes_previous", None) is None:
            raise ValueError("Previous amplitudes must be provided.")
        # Check that current excited states are computed 
        if eflag ==0:
            self.es = self._excited_states()
            self.eflag = 1
        # Reshape excited amplitudes
        self.nelec_alpha, self.nelec_beta = self.mf.mol.nelec
        (x_alpha_curr, y_alpha_curr, x_beta_curr, y_beta_curr,
        x_alpha_prev, y_alpha_prev, x_beta_prev, y_beta_prev
        ) = self._reshape_excited_amplitudes(overlaps_parameters)
        # Compute GS for previous geometry
        mol_prev = self._molecule(overlaps_parameters["atom_positions_previous"])
        MO_coeff_prev =  overlaps_parameters["amplitudes_previous"]["MO_coeff"]
        # Compute overlaps between previous and current GS
        overlaps_mo_alpha, overlaps_mo_beta = self._compute_mo_overlaps(
                                                    mol_prev, MO_coeff_prev
                                                    )
        # Compute matrix overlaps S blocks
        s_oo_alpha, s_ov_alpha, s_vo_alpha, s_vv_alpha = self._get_matrix_overlaps_s(
                    self.nelec_alpha,
                    overlaps_mo_alpha,
                )
        s_oo_beta, s_ov_beta, s_vo_beta, s_vv_beta = self._get_matrix_overlaps_s(
                    self.nelec_beta,
                    overlaps_mo_beta,
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
        # Opposite-spin excitations factor into two single-spin determinants.
        overlaps_es_es_alpha_beta = np.outer(overlaps_es_gs_alpha, overlaps_gs_es_beta)
        overlaps_es_es_beta_alpha = np.outer(overlaps_es_gs_beta, overlaps_gs_es_alpha)
        self.results["wf_overlaps"] = np.zeros(
            (self.kwargs["nroots"] + 1, self.kwargs["nroots"] + 1),
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



    def derivative_coupling(self, coupling_parameters,eflag):
        """
        Compute the analytic derivative couplings <Psi_I|d/dR Psi_J> for CIS
        and TDDFT-TDA (restricted and unrestricted). Units are 1/Bohr.
        """
        if coupling_parameters is None:
            coupling_parameters = {}
        coupling_parameters["state_inds_derivative_coupling"] = self._get_state_inds_derivative_coupling(
                                                coupling_parameters.get("state_inds_derivative_coupling",None)  
                                                                )
        n_state = self.kwargs["nroots"] + 1
        if eflag==0:
            self.es = self._excited_states()
            self.eflag=1
        natm = self.mol.natm
        coupling = np.zeros((n_state, n_state, natm, 3))
        # Unrestricted view of the reference and amplitudes (restricted
        # amplitudes are mapped to spin orbitals as x_beta = +/- x_alpha).
        mf_u = scf.addons.convert_to_uhf(self.mf)
        x_alpha, x_beta = self._spin_amplitudes(mf_u)
        # Half-derivative overlap integrals, shared by all terms.
        s1 = -self.mol.intor("int1e_ipovlp", comp=3)
        # GS-ES couplings <Psi_0|d/dR Psi_J>
        coupling_gs_es = self._compute_coupling_gs_es(
            mf_u, x_alpha, x_beta, s1, coupling_parameters
        )
        coupling[0, 1:] = coupling_gs_es
        coupling[1:, 0] = -coupling_gs_es
        # ES-ES couplings <Psi_I|d/dR Psi_J>
        coupling_es_es = self._compute_coupling_es_es(
            mf_u, x_alpha, x_beta, s1, coupling_parameters
        )
        coupling[1:, 1:] = coupling_es_es
        coupling = coupling * ANGSTROM_TO_BOHR
        coupling_dict = {}

        for ii, i in enumerate(
            coupling_parameters["state_inds_derivative_coupling"]
            ):
            for j in coupling_parameters["state_inds_derivative_coupling"][ii+1:]:
                coupling_dict[(i, j)] = coupling[i, j]

        self.results["derivative_coupling"] = coupling_dict

    def _molecule(self,atom_positions):
        mol = qm.gto.Mole()
        mol.atom = self._geometry(atom_positions)
        mol.unit = "Bohr"
        mol.basis = self.kwargs["basis"]
        mol.charge = self.kwargs["charge"]
        mol.spin = self.kwargs["spin"]
        mol.symmetry = False
        mol.verbose = 7
        mol.output = self.kwargs["output_abinitio"]
        mol.build()
        return mol
    
    def _geometry(self, atom_positions):
        geo = []
        for i,ele in enumerate(self.atom_names):
            geo.append( f"""{ele} {atom_positions[i,0]} 
                       {atom_positions[i,1]}
                        {atom_positions[i,2]}"""
                        )
        return geo

    def _gs(self, molecule):
        if self.kwargs["method_es"].lower() == "cis":
            mf = self._cis(molecule)
        elif self.kwargs["method_es"].lower() == "tddft":
            mf = self._dft(molecule)
        self.gs_energy = mf.e_tot
        if not mf.converged:
            raise RuntimeError("PySCF SCF did not converge")
        return mf

    '''
    This block has the function relate to gradient.
    '''
    def _get_state_inds_gradient(self, state_inds_gradient):
        """
        Return a list of state indices for gradient calculations.

        If no indices are provided, defaults to all states (ground + excited).
        """
        if state_inds_gradient is None:
            num_states = int(self.kwargs.get("nroots")) + 1
            return [i for i in range(num_states)]
        elif isinstance(state_inds_gradient, (int, np.integer)):
            return [state_inds_gradient]
        else:
            return state_inds_gradient

    def _compute_gradients(self, gradient_parameters):
        with lib.with_omp_threads(self.num_procs):
            gradients = np.zeros((self.mol.natm, 3, len(gradient_parameters["state_inds_gradient"])))
            for ii,i in enumerate(gradient_parameters["state_inds_gradient"]):
                if i==0:
                    grad_method = self.mf.nuc_grad_method()
                    if isinstance(self.mf, (dft.rks.RKS, dft.uks.UKS)):
                        grad_method.grid_response = True
                    value = grad_method.kernel()
                else:
                    grad_method = self.es.nuc_grad_method()
                    grad_method.cphf_conv_tol = gradient_parameters.get("cphf_conv_tol", 1e-10)
                    grad_method.cphf_max_cycle = gradient_parameters.get("cphf_max_cycle", 100)
                    value = grad_method.kernel(state=i)
                value = np.asarray(value)
                if value.shape != (self.mol.natm, 3) or not np.all(np.isfinite(value)):
                    raise RuntimeError("PySCF returned an invalid gradient for state %d." % i)
                gradients[:, :, ii] = value
        return gradients


    '''
    This block has the functions relate to energy calculation for excited states.
    '''
    def _excited_states(self, energy_parameters=None):
        # Check that the GS is already computed otherwise it computes it.
        if getattr(self, "mf", None) is None:
            self.mf = self._gs(self.mol)
        # Chooses the method to get ES based on user options
        if self.kwargs["method_es"].lower() == "cis":
            es = self._cis_es(energy_parameters)
        elif self.kwargs["method_es"].lower() == "tddft":
            es = self._dft_es(energy_parameters)
        return es  

    def _cis(self, molecule):
        with lib.with_omp_threads(self.num_procs):
            if self.kwargs["unrestricted"]:
                mf = qm.scf.UHF(molecule)
            else:
                mf = qm.scf.RHF(molecule)
            mf.kernel()
        return mf

    def _cis_es(self, energy_parameters):
        with lib.with_omp_threads(self.num_procs):
            es = qm.tdscf.TDA(self.mf)
            es.nroots = self.kwargs["nroots"]
            es.nstates = self.kwargs["nroots"]
            es.singlet = self.kwargs["singlet"]
            es.kernel()
        return es


    def _dft(self, molecule):
        with lib.with_omp_threads(self.num_procs):
            if self.kwargs["unrestricted"]:
                mf = qm.dft.UKS(molecule)
            else:
                mf = qm.dft.RKS(molecule)
            mf.xc = self.kwargs["functional"]
            mf.kernel()
        return  mf

    def _dft_es(self, energy_parameters):
        with lib.with_omp_threads(self.num_procs):
            es = qm.tddft.TDA(self.mf)
            es.nroots = self.kwargs["nroots"]
            es.nstates = self.kwargs["nroots"]
            es.singlet = self.kwargs["singlet"]
            es.kernel()
        return es            

    ##Function for getting spin multiplicity of excited states
    '''
    The following function is designed only for the case where
    the calculation is unrestricted and does not have a filter
    for singlet or triplet states
    '''
    def _spin_square(self):
        ## Check that is unrestricted
        if not self.kwargs["unrestricted"]:
            raise ValueError(" This function is only for unrestricted calculations")
        # Get overlaps matrix
        S = self.mf.get_ovlp()
        # Moleclar coeffients
        ca, cb =  self.mf.mo_coeff
        # S^2 for GS
        Spin_square_gs = self.mf.spin_square()[0]
        # Electron occupation
        oa, ob = [occ > 0 for occ in self.mf.mo_occ]
        # matrix overlap between alpha and beta
        sab = ca.conj().T @ S @ cb
        soo = sab[np.ix_(oa, ob)]
        sov = sab[np.ix_(oa, ~ob)]
        svo = sab[np.ix_(~oa, ob)]
        svv = sab[np.ix_(~oa, ~ob)]
        # Getting the spin multiplicity
        s2 = []
        multiplicity = []
        for xy in self.es.xy:
            (xa,xb),y = xy
            #same spin contributions
            delta_a = self._norm(xa.T @ soo) - self._norm(xa.conj() @ svo)
            delta_b = self._norm(soo @ xb.conj()) - self._norm(sov @ xb.T)
            # Mixed-spin contribution
            coupling = np.einsum("ia,ab,jb,ij->",
                         xa.conj(), svv, xb, soo.conj(),
                         optimize=True,
                        ).real
            # Normalize the complete state, including BOTH spin channels.
            norm = self._norm(xa) + self._norm(xb)
            if norm <= 0:
                raise ValueError("The excitation vector has zero norm.")
            ss2 = Spin_square_gs + (
                    delta_a.real + delta_b.real - 2.0 * coupling
                ) / norm
            s2.append(ss2)
            multiplicity.append(np.sqrt(1.0 + 4.0 * ss2))
        s2 = np.asarray(s2)
        multiplicity = np.asarray(multiplicity)      
        return s2, multiplicity
    def _norm(self,a):
        return np.vdot(a, a).real


    '''
    This block compute the frequencies and normal modes
    '''
    def _frequencies(self, frequency_parameters):
        ind_state = frequency_parameters.get("state_vibrations",0)
        step = frequency_parameters.get("step_displacement",0.005)
        # Checks that the states are calculate
        if getattr(self, "mf", None) is None:
            raise ValueError("GS must be calculated.")
        if ind_state > 0 and getattr(self, "es", None) is None:
            raise ValueError("Excited state must be calculated.")
        # Getting Hessian
        if ind_state == 0:
            mol = self.mf.mol
            with lib.with_omp_threads(self.num_procs):
                hess = self.mf.Hessian().kernel()
        else:
            mol = self.es.mol.copy()
            scan = self.es.nuc_grad_method().as_scanner(state=ind_state)
            # Convergence parameters
            scan.base._scf.conv_tol = frequency_parameters.get("scf_conv_tol",1e-11)
            scan.base._scf.conv_tol_grad = frequency_parameters.get("scf_conv_tol_grad",1e-8)
            scan.base.conv_tol = frequency_parameters.get("base_conv_tol",1e-9)
            scan.cphf_conv_tol = frequency_parameters.get("cphf_conv_tol",1e-10)


            xyz = mol.atom_coords(unit="Bohr")
            h2 = np.empty((xyz.size, xyz.size))

            # Creates the hessian using small displacements 
            for j in range(xyz.size):
                plus, minus = xyz.copy(), xyz.copy()
                plus.flat[j] += step
                minus.flat[j] -= step

                h2[:, j] = (
                    self._gradient_at(mol, scan, ind_state, plus, frequency_parameters) 
                    - self._gradient_at(mol, scan, ind_state, minus, frequency_parameters)
                    ) / (2 * step)
            # Return the scanner and its shared caches to this geometry.
            scan(mol)

            # Convert (3N, 3N) to PySCF ordering: (atom, atom, xyz, xyz).
            hess = h2.reshape(
                mol.natm, 3, mol.natm, 3
            ).transpose(0, 2, 1, 3)

        # Remove small numerical asymmetry.
        hess = 0.5 * (hess + hess.transpose(1, 0, 3, 2))
        mass_amu = (
                np.asarray(self.atom_masses)
                /AMU_TO_EMASS
            )

        vib = thermo.harmonic_analysis(
            mol,
            hess,
            mass=mass_amu,
            imaginary_freq=False,
        )

        vib["hessian"] = hess
        self.vib = vib

    def _gradient_at(self, mol, scan, state, coords, frequency_parameters):
        with lib.with_omp_threads(self.num_procs):
            scan.base._scf.max_cycle = frequency_parameters.get("scf_max_cycle", 500)
            scan.base.max_cycle = frequency_parameters.get("es_max_cycle", 500)
            # Creates the new geometry displaced
            displaced = mol.set_geom_(
                coords, unit="Bohr", inplace=False
            )
            # Re-run excited state calculation
            ener, gradient = scan(displaced)

            if (not scan.base._scf.converged
                    or not scan.base.converged[state - 1]):
                raise RuntimeError(
                    "SCF or selected excited root did not converge."
                )

        return np.asarray(gradient).reshape(-1)

    '''
    This block computes the overlaps between two different geometries.
    '''
    def _reshape_excited_amplitudes(self, overlap_parameters):
        """
        Reshape the excited amplitudes matrices.
        """
        excited_amplitudes_prev = overlap_parameters["amplitudes_previous"]["xy"]
        # Check that the previous and current amplitudes have the same number of states
        if  len(excited_amplitudes_prev) != len(self.es.xy):
            raise ValueError(" The number of excited state from previous and\n",
                              "current geometries must be the same.")
        if self.kwargs["unrestricted"]:
            vir_orbi_alpha = self.mf.mo_coeff[0].shape[1] - self.nelec_alpha
            vir_orbi_beta = self.mf.mo_coeff[1].shape[1] - self.nelec_beta 
        else:
            vir_orbi_alpha = self.mf.mo_coeff.shape[1] - self.nelec_alpha
            vir_orbi_beta = vir_orbi_alpha
        
        x_alpha_curr = np.zeros((self.kwargs["nroots"],self.nelec_alpha, vir_orbi_alpha),dtype=complex)
        y_alpha_curr = np.zeros((self.kwargs["nroots"],self.nelec_alpha, vir_orbi_alpha),dtype=complex)
        x_beta_curr = np.zeros((self.kwargs["nroots"],self.nelec_beta, vir_orbi_beta),dtype=complex)
        y_beta_curr = np.zeros((self.kwargs["nroots"],self.nelec_beta, vir_orbi_beta),dtype=complex)
        x_alpha_prev = np.zeros((self.kwargs["nroots"],self.nelec_alpha, vir_orbi_alpha),dtype=complex)
        y_alpha_prev = np.zeros((self.kwargs["nroots"],self.nelec_alpha, vir_orbi_alpha),dtype=complex)
        x_beta_prev = np.zeros((self.kwargs["nroots"],self.nelec_beta, vir_orbi_beta),dtype=complex)
        y_beta_prev = np.zeros((self.kwargs["nroots"],self.nelec_beta, vir_orbi_beta),dtype=complex)
        for i, xy in enumerate(self.es.xy):
            if self.kwargs["unrestricted"]:
                ((x_alpha_curr[i,:,:], x_beta_curr[i,:,:]), 
                (y_alpha_curr[i,:,:], y_beta_curr[i,:,:])) = xy
                ((x_alpha_prev[i,:,:], x_beta_prev[i,:,:]), 
                (y_alpha_prev[i,:,:], y_beta_prev[i,:,:])) = excited_amplitudes_prev[i]
            else:
                (x_alpha_curr[i,:,:], 
                y_alpha_curr[i,:,:]) = xy
                (x_alpha_prev[i,:,:],
                y_alpha_prev[i,:,:]) = excited_amplitudes_prev[i]
                
                if self.kwargs["singlet"]:
                    x_beta_curr[i,:,:] = x_alpha_curr[i,:,:]
                    y_beta_curr[i,:,:] = y_alpha_curr[i,:,:]
                    x_beta_prev[i,:,:] = x_alpha_prev[i,:,:]
                    y_beta_prev[i,:,:] = y_alpha_prev[i,:,:]
                else:
                    x_beta_curr[i,:,:] = -x_alpha_curr[i,:,:]
                    y_beta_curr[i,:,:] = -y_alpha_curr[i,:,:]
                    x_beta_prev[i,:,:] = -x_alpha_prev[i,:,:]
                    y_beta_prev[i,:,:] = -y_alpha_prev[i,:,:]                  

        return (
                x_alpha_curr, y_alpha_curr, x_beta_curr, y_beta_curr,
                x_alpha_prev, y_alpha_prev, x_beta_prev, y_beta_prev
                )

    def _compute_mo_overlaps(self, mol_prev, MO_coeff_prev):
        # Compute the overlap matrix between AO basis
        overlaps_ao = qm.gto.intor_cross("int1e_ovlp", mol_prev, self.mol)
        # Obtain the MO coefficients for the previous and current geometries
        if self.kwargs["unrestricted"]:
            ca_prev, cb_prev = MO_coeff_prev
            ca_curr, cb_curr = self.mf.mo_coeff
        else:
            ca_prev = MO_coeff_prev
            ca_curr = self.mf.mo_coeff
            cb_prev = ca_prev
            cb_curr = ca_curr
        # Compute the overlap matrices between MO 
        overlaps_mo_alpha = ca_prev.conj().T @ overlaps_ao @ ca_curr
        overlaps_mo_beta = cb_prev.conj().T @ overlaps_ao @ cb_curr
        return overlaps_mo_alpha, overlaps_mo_beta
    
    def _get_matrix_overlaps_s(self, num_electrons, mo_overlaps):
        """
        Compute blocks of the MO overlap matrix S. Return the blocks s_oo, s_ov, s_vo, s_vv.
        """
        s_oo = mo_overlaps[0:num_electrons, 0:num_electrons]
        s_ov = mo_overlaps[0:num_electrons, num_electrons:]
        s_vo = mo_overlaps[num_electrons:, 0:num_electrons]
        s_vv = mo_overlaps[num_electrons:, num_electrons:]
        return s_oo, s_ov, s_vo, s_vv

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

    '''
    This block computes the analytic derivative couplings (CIS and TDDFT-TDA).
    The excited states are treated as CIS-like (pseudo-)wavefunctions,
    |Psi_J> = sum_ia X^J_ia |Phi_i^a>, so that

        d_0J = sum_ia X^J_ia <phi_i|d phi_a>                  (Z-vector)
        d_IJ = X_I^T (dA/dR) X_J / (w_J - w_I)
               + sum_pq <phi_p|d phi_q> gamma^IJ_pq            (I != J)

    The orbital response enters through a Z-vector (one CPHF solve per
    state) for GS-ES, and through PySCF's TDA gradient Lagrangian for ES-ES,
    where X_I^T (dA/dR) X_J is obtained with the polarization identity
    4 X_I^T A' X_J = G(X_I + X_J) - G(X_I - X_J).
    All spin bookkeeping is done in the unrestricted (spin-orbital) form.
    '''

    def _get_state_inds_derivative_coupling(
        self, state_inds_derivative_coupling=None
         ):
        """
        Return a list/tuple of state indices for derivative coupling calculations.
        If no indices are provided, defaults to all states (ground + excited).
        """
        if state_inds_derivative_coupling is None:
            n_s = self.kwargs["nroots"] + 1
            state_inds_derivative_coupling = [i for i in range(n_s)]
        elif isinstance(state_inds_derivative_coupling, (int, np.integer)):
            state_inds_derivative_coupling = [
                i for i in range(state_inds_derivative_coupling + 1)
            ]
        return state_inds_derivative_coupling


    def _spin_amplitudes(self, mf_u):
        """
        Return the TDA amplitudes as spin-orbital blocks of shape
        (nroots, nocc, nvir) for alpha and beta.
        """
        nocc_alpha = int(np.count_nonzero(mf_u.mo_occ[0] > 0))
        nocc_beta = int(np.count_nonzero(mf_u.mo_occ[1] > 0))
        nvir_alpha = mf_u.mo_occ[0].size - nocc_alpha
        nvir_beta = mf_u.mo_occ[1].size - nocc_beta
        x_alpha = np.zeros((self.kwargs["nroots"], nocc_alpha, nvir_alpha),dtype=complex)
        x_beta = np.zeros((self.kwargs["nroots"], nocc_beta, nvir_beta),dtype=complex)
        for i, xy in enumerate(self.es.xy):
            if self.kwargs["unrestricted"]:
                (x_alpha[i], x_beta[i]), _ = xy
            else:
                x_alpha[i] = xy[0]
                if self.kwargs["singlet"]:
                    x_beta[i] = x_alpha[i]
                else:
                    x_beta[i] = -x_alpha[i]
        return x_alpha, x_beta

    def _contract_half_overlap(self, s1, p_matrix):
        """
        Contract sum_{mu nu} <chi_mu|d/dR_A chi_nu> P_{mu nu} for every atom.
        Returns an array of shape (natm, 3).
        """
        aoslices = self.mol.aoslice_by_atom()
        out = np.zeros((self.mol.natm, 3))
        for ia in range(self.mol.natm):
            p0, p1 = aoslices[ia, 2:]
            out[ia] = np.einsum("xnm,mn->x", s1[:, p0:p1], p_matrix[:, p0:p1])
        return out

    def _compute_coupling_gs_es(self, mf_u, x_alpha, x_beta, s1, coupling_parameters):
        """
        Compute <Psi_0|d/dR Psi_J> for all excited states with a Z-vector.
        """
        nroots = self.kwargs["nroots"]
        etf = coupling_parameters.get("etf", False)
        mo_coeff = mf_u.mo_coeff
        mo_occ = mf_u.mo_occ
        mo_energy = mf_u.mo_energy
        occ = [mo_occ[s] > 0 for s in range(2)]
        orbo = [mo_coeff[s][:, occ[s]] for s in range(2)]
        orbv = [mo_coeff[s][:, ~occ[s]] for s in range(2)]
        nocc = [orbo[s].shape[1] for s in range(2)]
        nvir = [orbv[s].shape[1] for s in range(2)]
        # Response kernel of the reference (includes f_xc for DFT).
        with lib.with_omp_threads(self.num_procs):
            vresp = mf_u.gen_response(hermi=1)

        def fvind(z):
            z = z.reshape(-1, nvir[0] * nocc[0] + nvir[1] * nocc[1])
            nset = z.shape[0]
            za = z[:, :nvir[0] * nocc[0]].reshape(nset, nvir[0], nocc[0])
            zb = z[:, nvir[0] * nocc[0]:].reshape(nset, nvir[1], nocc[1])
            dm_a = np.einsum("pa,xai,qi->xpq", orbv[0], za, orbo[0])
            dm_b = np.einsum("pa,xai,qi->xpq", orbv[1], zb, orbo[1])
            dm = np.asarray((dm_a + dm_a.transpose(0, 2, 1),
                             dm_b + dm_b.transpose(0, 2, 1)))
            v1 = vresp(dm)
            v1_a = np.einsum("pa,xpq,qi->xai", orbv[0], v1[0], orbo[0])
            v1_b = np.einsum("pa,xpq,qi->xai", orbv[1], v1[1], orbo[1])
            return np.hstack((v1_a.reshape(nset, -1), v1_b.reshape(nset, -1)))

        # Z-vector equations (e_a - e_i) z + G[z] = X, one per state.
        x_vo = (x_alpha.transpose(0, 2, 1), x_beta.transpose(0, 2, 1))
        with lib.with_omp_threads(self.num_procs):
            z_vo, _ = ucphf.solve(
                fvind, mo_energy, mo_occ, (-x_vo[0], -x_vo[1]),
                max_cycle=coupling_parameters.get("cphf_max_cycle", 50),
                tol=coupling_parameters.get("cphf_conv_tol", 1e-8),
            )
        z_vo = [np.asarray(z_vo[s]).reshape(nroots, nvir[s], nocc[s]) for s in range(2)]
        # Skeleton derivative Fock matrices dF/dR at fixed density.
        with lib.with_omp_threads(self.num_procs):
            h1ao = mf_u.Hessian().make_h1(mo_coeff, mo_occ)
        # Response potential of the symmetric Z densities.
        z_ao = [np.einsum("pa,xai,qi->xpq", orbv[s], z_vo[s], orbo[s]) for s in range(2)]
        v_z = vresp(np.asarray([z_ao[s] + z_ao[s].transpose(0, 2, 1) for s in range(2)]))
        coupling = np.zeros((nroots, self.mol.natm, 3))
        for state in range(nroots):
            p_matrix = np.zeros_like(z_ao[0][0])
            for s in range(2):
                x_ao = orbv[s] @ x_vo[s][state] @ orbo[s].T
                z_ao_state = z_ao[s][state]
                # -1/2 X.S^R and (optionally) -X.<chi|d chi>_antisym
                m_matrix = -0.5 * x_ao
                # -z_ai e_i S^R_ai
                m_matrix -= orbv[s] @ (z_vo[s][state] * mo_energy[s][occ[s]]) @ orbo[s].T
                # -S^R_kj (C_o^T V[z] C_o)_kj
                v_oo = orbo[s].T @ (0.5 * v_z[s][state]) @ orbo[s]
                m_matrix -= orbo[s] @ v_oo @ orbo[s].T
                p_matrix += m_matrix + m_matrix.T
                if not etf:
                    p_matrix += -0.5 * (x_ao - x_ao.T)
                # z . F^[R]
                for ia in range(self.mol.natm):
                    coupling[state, ia] += np.einsum(
                        "xmn,mn->x", h1ao[s][ia], z_ao_state
                    )
            coupling[state] += self._contract_half_overlap(s1, p_matrix)
        return coupling

    def _compute_coupling_es_es(self, mf_u, x_alpha, x_beta, s1, coupling_parameters):
        """
        Compute <Psi_I|d/dR Psi_J> between excited states.
        """
        nroots = self.kwargs["nroots"]
        etf = coupling_parameters.get("etf", False)
        degeneracy_threshold = coupling_parameters.get("degeneracy_threshold", 1e-6)
        excitation_energies = np.asarray(self.es.e)
        mo_coeff = mf_u.mo_coeff
        occ = [mf_u.mo_occ[s] > 0 for s in range(2)]
        orbo = [mo_coeff[s][:, occ[s]] for s in range(2)]
        orbv = [mo_coeff[s][:, ~occ[s]] for s in range(2)]
        # TDA gradient object (handles RHF/UHF/RKS/UKS and singlet/triplet).
        td_grad = self.es.nuc_grad_method()
        td_grad.cphf_conv_tol = coupling_parameters.get("cphf_conv_tol", 1e-8)
        td_grad.cphf_max_cycle = coupling_parameters.get("cphf_max_cycle", 50)
        coupling = np.zeros((nroots, nroots, self.mol.natm, 3))
        for i in range(nroots):
            for j in range(i + 1, nroots):
                # Orbital-relaxed X_I^T (dA/dR) X_J via polarization identity.
                with lib.with_omp_threads(self.num_procs):
                    grad_plus = td_grad.grad_elec(
                        self._tda_amplitudes(i, j, 1.0), self.kwargs["singlet"]
                    )
                    grad_minus = td_grad.grad_elec(
                        self._tda_amplitudes(i, j, -1.0), self.kwargs["singlet"]
                    )
                hellmann_feynman = 0.25 * (grad_plus - grad_minus)
                energy_gap = excitation_energies[j] - excitation_energies[i]
                if abs(energy_gap) < degeneracy_threshold:
                    logger.warning(
                        "States %d and %d are (near-)degenerate; "
                        "the derivative coupling diverges.", i + 1, j + 1
                    )
                coupling[i, j] = hellmann_feynman / energy_gap
                # Orbital overlap-derivative term sum_pq <phi_p|d phi_q> gamma_pq.
                if not etf:
                    p_matrix = np.zeros((self.mol.nao, self.mol.nao))
                    for s, x in enumerate((x_alpha, x_beta)):
                        gamma_vv = x[i].T @ x[j]
                        gamma_oo = -x[j] @ x[i].T
                        gamma_ao = (orbv[s] @ gamma_vv @ orbv[s].T
                                    + orbo[s] @ gamma_oo @ orbo[s].T)
                        p_matrix += 0.5 * (gamma_ao - gamma_ao.T)
                    coupling[i, j] += self._contract_half_overlap(s1, p_matrix)
                coupling[j, i] = -coupling[i, j]
        return coupling

    def _tda_amplitudes(self, i, j, sign):
        """
        Build the amplitudes X_I + sign * X_J in PySCF's own xy format
        (Y = 0 for TDA) to be passed to the gradient Lagrangian.
        """
        if self.kwargs["unrestricted"]:
            (xa_i, xb_i), _ = self.es.xy[i]
            (xa_j, xb_j), _ = self.es.xy[j]
            xa = xa_i + sign * xa_j
            xb = xb_i + sign * xb_j
            return (xa, xb), (np.zeros_like(xa), np.zeros_like(xb))
        x = self.es.xy[i][0] + sign * self.es.xy[j][0]
        return x, np.zeros_like(x)
