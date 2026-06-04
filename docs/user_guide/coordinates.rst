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


