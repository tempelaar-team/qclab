.. _numerical-constants:

==============================
Numerical Constants
==============================

The :mod:`qclab.numerical_constants` module collects numerical
thresholds and unit-conversion factors used throughout QC Lab. Models
and ingredients are expected to read these values rather than hard-code
magic numbers in their bodies.

Values for the underlying physical constants are taken from the 2022
CODATA recommended values (Mohr et al., *Rev. Mod. Phys.* **97**,
025002 (2025), https://doi.org/10.1103/RevModPhys.97.025002).

----

Numerical thresholds
====================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Name
     - Default
     - Purpose
   * - ``SMALL``
     - ``1e-10``
     - Generic near-zero cutoff used by gauge-fixing,
       degeneracy detection, and other numerical guards.
   * - ``GAUGE_FIX_THRESHOLD``
     - ``1e-3``
     - Maximum allowed misalignment (relative to the magnitude of the
       coupling) when fixing the adiabatic-basis gauge.
   * - ``FINITE_DIFFERENCE_DELTA``
     - ``1e-6``
     - Step size used by the finite-difference fallbacks for
       ``dh_c_dzc`` and ``dh_qc_dzc`` when no analytical gradient
       ingredient is provided.

These thresholds are deliberately exposed as module-level constants so
that they can be tuned per-model; ingredients that need a different
finite-difference step size, for example, can read
``model.constants.dh_c_dzc_finite_difference_delta`` instead and
override the default.

----

Unit conversions
================

QC Lab does not enforce a single unit system; each model documents the
units of its own constants. The conversions listed below are the ones
used by the built-in models (``FMOComplex``, ``AbInitio``) and by the
Q-Chem interface, and are convenient when assembling new models.

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Name
     - Value
     - Conversion
   * - ``EV_TO_INVCM``
     - 8065.610420
     - electronvolts to wavenumbers (cm\ :sup:`-1`)
   * - ``HA_TO_EV``
     - 27.21138625
     - Hartrees to electronvolts
   * - ``EV_TO_HA``
     - ``1 / HA_TO_EV``
     - electronvolts to Hartrees
   * - ``INVCM_TO_HA``
     - ``EV_TO_HA / EV_TO_INVCM``
     - wavenumbers to Hartrees
   * - ``ANGSTROM_TO_BOHR``
     - 1.8897259886
     - Angstroms to Bohr
   * - ``AMU_TO_EMASS``
     - 1822.89
     - atomic mass units to electron mass
   * - ``AU_TIME_TO_FS``
     - 0.02419
     - atomic units of time to femtoseconds

----

Reference-temperature unit system
=================================

A handful of QC Lab models — most prominently
:class:`~qclab.models.FMOComplex` — express all energies as multiples of
the thermal energy at a reference temperature of 300 K, with the
corresponding time unit ``hbar / kBT(300 K) ≈ 25.46 fs``. The conversion
factor between wavenumbers and that reference unit is exposed under two
names:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Meaning
   * - ``INVCM_TO_KBT_REF``
     - Multiplier that converts a wavenumber (cm\ :sup:`-1`) into the
       300 K reference unit, i.e. ``A[INVCM] * INVCM_TO_KBT_REF =
       A[KBT_REF]``.
   * - ``INVCM_TO_300K``
     - Backwards-compatible alias for ``INVCM_TO_KBT_REF``.

The auxiliary constants used to build that conversion (``C_M_PER_S``,
``H_J_S``, ``K_B_J_PER_K``, ``HBAR_J_S``, ``T_REF_K``, ``KBT_REF_J``,
``HC_J_M``) are also available as module-level attributes for use by
analysis scripts.

----

Auto-generated reference
========================

.. automodule:: qclab.numerical_constants
   :members:
   :undoc-members:
   :member-order: bysource
   :no-value:
