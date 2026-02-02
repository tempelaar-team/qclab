"""
This module contains numerical constants.

Values for physical constants are taken from 2022 CODATA recommended values.
Reference publication:
Mohr et al. Rev. Mod. Phys. 2025, 97 (2), 025002. https://doi.org/10.1103/RevModPhys.97.025002.
"""

import numpy as np

# Numerical threshold for near zero values.
SMALL = 1e-10

# Numerical threshold for nonadiabatic coupling gauge fixing.
# The misalignment is allowed to be 1e-3 of the magnitude of the coupling.
GAUGE_FIX_THRESHOLD = 1e-3

# Finite difference step size.
FINITE_DIFFERENCE_DELTA = 1e-6

# Speed of light [m/s].
C_M_PER_S = 299792458

# Planck constant [J * s].
H_J_S = 6.62607015e-34

# Boltzmann constant [J/K].
K_B_J_PER_K = 1.380649e-23

# Hartree energy in Joules.
HA_J = 4.3597447222060e-18

# Reduced Planck constant [J·s].
HBAR_J_S = H_J_S / (2 * np.pi)

# Reference temperature [K].
T_REF_K = 300

# Thermal energy at the reference temperature [J].
KBT_REF_J = K_B_J_PER_K * T_REF_K

# h * c [J*m].
HC_J_M = H_J_S * C_M_PER_S

# Conversion between inverse centimeters to reference energy.
# kBT / invcm = (100 [cm / m]) * hc[J*m] / kBT [J]
# A[INVCM] * INVCM_TO_KBT_REF = A[KBT_REF]
INVCM_TO_KBT_REF = 100 * HC_J_M / KBT_REF_J

# Alias to old name for backwards-compatibility.
INVCM_TO_300K = INVCM_TO_KBT_REF

# Atomic mass units to Kilograms.
AMU_TO_KG = 1.66053906892e-27

# Electron mass to Kilograms.
EMASS_TO_KG = 9.1093837139e-31

# Atomic mass units to Electron mass.
AMU_TO_EMASS = AMU_TO_KG / EMASS_TO_KG

# Atomic units of time to Femtoseconds.
AU_TIME_TO_FS = (HBAR_J_S / HA_J) * 1e15