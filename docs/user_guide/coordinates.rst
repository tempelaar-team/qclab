.. _coordinates:

===========
Coordinates
===========

QC Lab uses a complex-valued classical coordinate formalism to implement both Models and Algorithms. This enables simulations in QC Lab to be entirely
invariant to the representation in which they are carried out. As such, the same algorithm can simulate a
model in real or reciprocal space (or indeed, any space at all) without modification. The complex-valued classical coordinate formalism is introduced in detail in `Miyazaki et al. 2024 <https://doi.org/10.1021/acs.jctc.4c00555>`_.
We include a brief description of the formalism here and document some useful functions for converting between real-valued and complex-valued coordinates.


Complex-valued coordinates in QC Lab are denoted ``z`` by convention, whereas the real-valued position and momentum coordinates are ``q`` and ``p``, respectively.
The complex-valued coordinate can be constructed from the real-valued coordinates and the coordinate masses by introducing a set of "weights" ``h`` which is denoted ``classical_coordinate_weight`` in
the Model object.

.. math::

    z_{\xi} = \sqrt{\frac{m_{\xi} h_{\xi}}{2}}\left(q_{\xi} + i\frac{p_{\xi}}{m_{\xi} h_{\xi}}\right)


Users may notice that :math:`h_{\xi}` plays the same role as a frequency in relating the relative displacements of position and momentum.
For that reason, it is often a convenient choice when working with harmonic oscillators to set :math:`h_{\xi}` to the harmonic frequency. Algorithms in
QC Lab are implemented in a manner that is invariant to the choice of weights. Likewise, Ingredients in QC Lab are also invariant to the choice of weights.
However, Model objects may enforce some choice of weights as part of their initialization rather than allowing this to be changed by a user.

Conversions to and from complex-valued classical coordinates are conveniently implemented in the following functions.

.. automodule:: qclab.functions
   :members: z_to_q, z_to_p, qp_to_z
   :undoc-members:
   :member-order: bysource
   :no-value:


.. _physical_z_conj:

Physical Positions and Momenta in a General Basis
-------------------------------------------------

A basis in which the coordinates ``q`` and ``p`` of the conversion functions above are a physical position and a physical momentum is called a physical basis.
A Model can also be defined in another basis, for example with the phonons of a lattice in reciprocal space.
If :math:`z = U z'`, where :math:`z'` is in a physical basis and :math:`U` is unitary, the physical z-conjugation matrix is

.. math::

    S = U U^{T}.

It is the symmetric unitary matrix for which :math:`z \rightarrow S z^{*}` reverses all physical momenta while leaving all physical positions unchanged, and it equals the identity in a physical basis.
The physical position and momentum components of :math:`z` are :math:`(z + S z^{*})/2` and :math:`(z - S z^{*})/2`, as calculated by the functions ``calc_z_physical_position_component`` and ``calc_z_physical_momentum_component`` in ``qclab.functions``.

The following four Recipe entries determine :math:`S` from the Hamiltonians of the Model and write it to the State object as ``physical_z_conj_matrix``.
They are placed in an initialization Recipe after the classical coordinates are initialized:

.. code-block:: python

    from functools import partial
    from qclab import tasks

    physical_z_conj_recipe = [
        tasks.update_dh_qc_dzc,
        tasks.initialize_z_probe,
        partial(
            tasks.update_classical_force,
            z_name="z_probe",
            classical_force_name="classical_force_probe",
        ),
        tasks.initialize_physical_z_conj_matrix,
    ]

The first entry evaluates the gradient of the quantum–classical interaction Hamiltonian with respect to :math:`z^{*}`, the second sets up displaced coordinates, and the third evaluates the classical force at those coordinates; ``initialize_physical_z_conj_matrix`` uses the gradient and the forces to determine :math:`S`.

The MSMASH Algorithm includes these entries in its initialization Recipe, for use in its rescaling and frustrated-hop reversal of the physical momentum (described in :ref:`msmash_source`).
The frustrated-hop reversal of the physical momentum assumes that the classical Hamiltonian is the sum of a kinetic energy that is quadratic in the physical momenta and a potential energy that depends on the physical positions only.
A Model for which this does not hold can supply the ``physical_momentum_reversal`` Ingredient, for example ``physical_momentum_reversal_numerical`` (see :ref:`Ingredients <ingredient>`).

A Model can also supply :math:`S` through the ``physical_z_conj_matrix`` Ingredient (see :ref:`Ingredients <ingredient>`).
This is needed when the quantum–classical interaction Hamiltonian depends on the physical momenta, in which case :math:`S` cannot be determined from the Hamiltonians.
If :math:`S` cannot be determined and no Ingredient supplies it, a warning is logged and the identity is used, which is valid only in a physical basis.
Ingredients that act on the real and imaginary parts of :math:`z` directly, such as ``hop_harmonic`` and the ``init_classical`` Ingredients, are written for a physical basis.
A Model defined in another basis should use them only where they are unchanged by the change of basis, as for harmonic modes of equal frequency and weight, and otherwise supply its own.


