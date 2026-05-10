.. _coordinates:

===========
Coordinates
===========

QC Lab uses a complex-valued classical coordinate formalism to implement both Models and Algorithms. This enables simulations in QC Lab to be entirely
invariant to the representation in which they are carried out. As such, the same algorithm can simulate a
model in real or reciprocal space (or indeed, any space at all) without modification. The complex-valued classical coordinate formalism is introduced in detail in `Miyazaki et al. 2024 <https://doi.org/10.1021/acs.jctc.4c00555>`_.
We include a brief description of the formalism here and document some useful functions for converting between real-valued and complex-valued coordinates.


Complex-valued coordinates in QC Lab are denoted as ``z`` by convention, whereas the real-valued position and momentum coordinates are ``q`` and ``p``, respectively.
The complex-valued coordinate can be constructed from the real-valued coordinates and the coordinate masses by introducing a set of "weights" ``h`` which is denoted ``classical_coordinate_weight`` in
the Model object.

.. math::

    z_{n} = \sqrt{\frac{m_{n} h_{n}}{2}}\left(q + i\frac{p}{m_{n} h_{n}}\right)


Readers with a keen eye may notice that :math:`h_{n}` plays the same role as a frequency in relating the relative displacements of position and momentum.
For that reason, it is often a convenient choice when working with harmonic oscillators to set :math:`h_{n}` to the harmonic frequency. Algorithms in
QC Lab are implemented in a manner that is invariant to the choice of weights. Likewise, ingredients in QC Lab are also invariant to the choice of weights.
However, Models may enforce some choice of weights as part of their initialization rather than enabling this to be changed by a user.

Conversions to and from complex-valued classical coordinates are conveniently implemented in the following functions.

.. automodule:: qclab.functions
   :members: z_to_q, z_to_p, qp_to_z
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:


Gradients in complex-valued coordinates
---------------------------------------

When implementing a new gradient ingredient (e.g. ``dh_qc_dzc`` or ``dh_c_dzc``)
it is often natural to differentiate the underlying physics with respect to
:math:`q` and :math:`p`, then convert the result to the gradient with respect
to :math:`z^{*}` that QC Lab consumes internally. The reverse direction is
also occasionally needed (for example when comparing to phase-space
gradients reported by an electronic-structure code). Two helpers are
provided for this:

.. automodule:: qclab.functions
   :members: dqdp_to_dzc, dzdzc_to_dqdp
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

The Tully models (``TullyProblemOne``, ``TullyProblemTwo``, ``TullyProblemThree``)
and the ``AbInitio`` model use ``dqdp_to_dzc`` to assemble their analytical
gradient ingredients. See :ref:`Functions <functions>` for the full list of
low-level helpers.


